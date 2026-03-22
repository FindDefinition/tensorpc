# agent manage local SSH connections and compute resources.
import abc
import asyncio
from collections.abc import Awaitable, Coroutine
import enum
import os
from pathlib import Path
import re
import shlex
import tempfile
import time
import traceback
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    Optional,
    TypeVar,
    TypedDict,
    Union,
)

from contextlib import nullcontext
import humanize
import psutil
import rich
from tensorpc.apps.cm.components.raft_mgr_panel import RaftManagerPanel
from tensorpc.apps.cm.constants import TENSORPC_ENV_CM_NODEMGR_GROUP_ID
from tensorpc.apps.cm.coretypes import (
    CM_LOGGER,
    GroupSSHStatus,
    RaftMgrActions,
    UserCmd,
    SSHWorkerConfig,
    UserCmdType,
    WorkerInfo,
    ResourceInfo,
    NodeFlags,
    GroupNodeSpec,
    WorkerUIType,
    WorkerSelectItem,
    WorkerSSHStatus,
    WorkerUISSHState,
)
from tensorpc.apps.cm.lrucache import LRUCache
from tensorpc.apps.cm.serv_names import master_serv_names
from tensorpc.apps.dbg.components.dbgpanel import MasterDebugPanel
from tensorpc.core.asynctools import cancel_task
from tensorpc.core.distributed.comm import AsyncGRPCComm
from tensorpc.core.distributed.comm.grpcimpl import AsyncGRPCCommConfig
from tensorpc.core.distributed.raft import (
    ProposeResult,
    RaftConfig,
    RaftEventType,
    RaftNode,
    AsyncRaftComm,
    RaftRole,
    PeerInfo,
)
from tensorpc.core.distributed.raft import (
    AppendEntriesRequest,
    AppendEntriesResponse,
    InstallSnapshotRequest,
    InstallSnapshotResponse,
    RequestVoteRequest,
    RequestVoteResponse,
    StateMachine,
)
from tensorpc.apps.cm.constants import (
    TENSORPC_ENV_CM_NODEMGR_RANK,
    TENSORPC_ENV_CM_NODEMGR_WORLD_SIZE,
    TENSORPC_ENV_CM_NODEMGR_URL_WITH_PORT,
    TENSORPC_ENV_CM_NODEMGR_BACKEND
)
from tensorpc.dock.components import mui
from tensorpc.autossh.core import (
    CommandEvent,
    CommandEventType,
    Event as SSHEvent,
    SSHEventType,
)
from tensorpc.core.serviceunit import ServiceEventType
from tensorpc.dock import terminal
import dataclasses as dataclasses_plain
from tensorpc.autossh.core import SSHConnDesc
from tensorpc.core import dataclass_dispatch as dataclasses, marker, prim
from tensorpc.core.datamodel.draftstore import DraftSimpleFileStoreBackend
from tensorpc.utils.pyspyutil import PyspyTraceMode

if TYPE_CHECKING:
    from _typeshed import DataclassInstance as StandardDataclass


class AsyncGrpcRaftComm(AsyncRaftComm, AsyncGRPCComm):
    def __init__(self, group_id: str, cfg: AsyncGRPCCommConfig):
        AsyncGRPCComm.__init__(self, cfg)
        self._group_id = group_id

    async def request_vote(
        self, peer_id: str, req: RequestVoteRequest
    ) -> RequestVoteResponse:
        return await self.remote_call(
            peer_id, master_serv_names.RAFT_REQUEST_VOTE, self._group_id, req
        )

    async def append_entries(
        self, peer_id: str, req: AppendEntriesRequest
    ) -> AppendEntriesResponse:
        return await self.remote_call(
            peer_id, master_serv_names.RAFT_APPEND_ENTRIES, self._group_id, req
        )

    async def install_snapshot(
        self, peer_id: str, req: InstallSnapshotRequest
    ) -> InstallSnapshotResponse:
        return await self.remote_call(
            peer_id, master_serv_names.RAFT_INSTALL_SNAPSHOT, self._group_id, req
        )


class A2AWorkerState(TypedDict):
    worker_info: WorkerInfo
    is_connected: bool


class A2AStateMachinBaseState(TypedDict):
    last_cmd: Optional[UserCmd]
    cmd_version: int
    # used for runtime cmds, e.g. when subprocesses
    # enter pause state controlled by tensorpc,
    # we use this to tell worker release breakpoints.
    runtime_cmd: Optional[Any]
    runtime_cmd_uid: Union[int, str]


class A2AWorkerQueryResult(A2AStateMachinBaseState):
    leader_info: Optional[PeerInfo]
    is_leader: bool


class A2AStateMachineState(A2AStateMachinBaseState):
    workers: dict[str, A2AWorkerState]


class A2AStateMachine(StateMachine):
    """State machine for all-to-all (training) tasks."""

    def __init__(self, world_size: int):
        self.raft_state: A2AStateMachineState = {
            "last_cmd": None,
            "cmd_version": -1,
            "workers": {},
            "runtime_cmd": None,
            "runtime_cmd_uid": -1,
        }
        self.worker_status_dict: dict[str, WorkerSSHStatus] = {}

    def apply(self, command: UserCmd, log_index: int):
        # worker events are ignored.
        if command.type == UserCmdType.REGISTER_WORKER:
            assert command.worker_info is not None
            worker_info = command.worker_info
            # if worker_info.uid in self.raft_state["workers"]:
            #     CM_LOGGER.warning(f"Try to register worker {worker_info.uid} which is already registered.")
            self.raft_state["workers"][worker_info.uid] = {
                "worker_info": worker_info,
                "is_connected": True,
            }
            if worker_info.uid not in self.worker_status_dict:
                self.worker_status_dict[worker_info.uid] = WorkerSSHStatus(
                    status="idle", last_ts=time.time_ns()
                )
            return
        if command.type == UserCmdType.UNREGISTER_WORKER:
            assert command.worker_info is not None
            worker_info = command.worker_info
            if worker_info.uid in self.raft_state["workers"]:
                self.raft_state["workers"].pop(worker_info.uid)
                self.worker_status_dict.pop(worker_info.uid, None)
            else:
                CM_LOGGER.warning(
                    f"Try to unregister worker {worker_info.uid} which is not registered."
                )
            return
        self.raft_state["last_cmd"] = command
        self.raft_state["cmd_version"] = log_index

    def serialize(self) -> Any:
        worker_copies = self.raft_state["workers"].copy()
        res = self.raft_state.copy()
        res["workers"] = worker_copies
        return res

    def deserialize(self, data: Any):
        self.raft_state = data.copy()
        self.raft_state["workers"] = data["workers"].copy()
        cur_ts = time.time_ns()
        for worker_id in self.raft_state["workers"]:
            self.worker_status_dict[worker_id] = WorkerSSHStatus(
                status="idle", last_ts=cur_ts
            )

    def has_worker(self, worker_id: str) -> bool:
        return worker_id in self.raft_state["workers"]


@dataclasses_plain.dataclass
class ExtendedRaftNode(RaftNode):
    pass


# TODO implement replica worker (for serving/inference).


@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class SSHA2AWorkerState:
    cur_leader_info: WorkerInfo = dataclasses.field(
        default_factory=lambda: WorkerInfo(peer_info=PeerInfo(uid="", url=""), rank=-1)
    )
    cur_cmd: Optional[UserCmd] = None
    cmd_version: int = -1
    finished_rt_cmd_uids: LRUCache[Union[int, str], Any] = dataclasses.field(
        default_factory=lambda: LRUCache(100)
    )


class AsyncDebouncer:
    def __init__(self, wait_time: float):
        self._wait_time = wait_time
        self._task: Optional[asyncio.Task] = None
        self._is_running_user_func: bool = False
        self._done_event = asyncio.Event()
        self._lock = asyncio.Lock()

    async def _debounce_loop(
        self,
        callback: Callable[..., Coroutine[None, None, Any]],
        *args: Any,
        **kwargs: Any,
    ):
        await asyncio.sleep(self._wait_time)
        self._done_event.clear()
        self._is_running_user_func = True
        try:
            await callback(*args, **kwargs)
        except:
            # we catch all exceptions to avoid task being cancelled without notice, which may cause user confusion.
            CM_LOGGER.exception("Exception occurred in debounced function:")
        finally:
            self._is_running_user_func = False
            self._done_event.set()

    async def call(
        self,
        callback: Callable[..., Coroutine[None, None, Any]],
        *args: Any,
        **kwargs: Any,
    ):
        async with self._lock:
            if (
                not self._is_running_user_func
                and self._task is not None
                and not self._task.done()
            ):
                self._task.cancel()
            if self._is_running_user_func:
                await self._done_event.wait()
            self._task = asyncio.create_task(
                self._debounce_loop(callback, *args, **kwargs)
            )

    async def cancel(self):
        if self._task is not None and not self._task.done():  
            await cancel_task(self._task)


@dataclasses.dataclass
class SSHWorkerUILeaderState:
    workers: list[WorkerInfo]

@dataclasses_plain.dataclass
class CmdTaskState:
    task: asyncio.Task
    event: asyncio.Event

class SSHA2AWorker:
    def __init__(
        self,
        group_id: str,
        rank: int,
        world_size: int,
        peer_info: PeerInfo,
        init_raft_infos: list[PeerInfo],
        cfg: SSHWorkerConfig,
        comm: AsyncGRPCComm,
        propose_fn: Callable[[UserCmd, bool, Optional[int]], Coroutine[None, None, ProposeResult]],
        debug_panel_fn: Callable[..., Awaitable[None]],
        fetch_pyspy_info_fn: Callable[[PyspyTraceMode, bool], Awaitable[Any]],
        resource_info: Optional[ResourceInfo] = None,
    ):
        # super().__init__(group_id, peer_info.uid, cfg.log_to_stdout)
        self._peer_info = peer_info
        self._group_id = group_id
        self._rank = rank
        self._world_size = world_size
        self._cur_raft_infos = init_raft_infos
        self._is_raft_worker = peer_info.uid in set(
            info.uid for info in init_raft_infos
        )
        # we use negative rank to indicate raft-only node.
        self._is_compute_worker = rank >= 0
        self._cfg = cfg
        self._comm = comm
        self._resource_info = resource_info

        self._raft_node: Optional[ExtendedRaftNode] = None
        self._workers_observe_task: Optional[asyncio.Task] = None
        self._leader_observe_task: Optional[asyncio.Task] = None

        self._worker_state = SSHA2AWorkerState()

        self._cmd_apply_lock = asyncio.Lock()
        self._awake_event = asyncio.Event()

        self._debouncer = AsyncDebouncer(wait_time=self._cfg.worker_update_debounce)
        self._terminal = terminal.AsyncSSHTerminal(log_to_stdout=cfg.log_to_stdout)
        # for single-node case, it's safe to allow stdin for debugging.
        self._terminal.prop(disableStdin=world_size > 1)
        self._propose_fn = propose_fn
        self._cmd_task: Optional[CmdTaskState] = None

        self._sync_lock = asyncio.Lock()
        self._debug_panel_fn = debug_panel_fn
        self._fetch_pyspy_info_fn = fetch_pyspy_info_fn
        self._local_shutdown_event: Optional[asyncio.Event] = None

        self.setup_ui()

    def is_started(self):
        return self._local_shutdown_event is not None

    def setup_ui(self):
        ui_state = WorkerUISSHState(
            id=self._peer_info.uid, 
            world_size=self._world_size,
            num_connected=0,
            ssh_status=WorkerSSHStatus(status="idle", last_ts=0),
            is_raft_node=self._is_raft_worker,
        )
        self.ssh_dm = mui.DataModel(ui_state, [])
        draft = self.ssh_dm.get_draft()
        panel = RaftManagerPanel(
            self._is_raft_worker,
            self._group_id,
            self._peer_info.uid,
            self.ssh_dm,
            self._raft_manager_act,
            self._debug_panel_fn if self._cfg.enable_debug_panel else None,
            self._fetch_pyspy_info_fn if self._is_raft_worker else None,
            draft,
            self._terminal,
        )
        if panel.debug_panel is not None:
            panel.debug_panel.event_breakpoint_process_change.on(self._on_has_bkpt_change)

        self._raft_mgr_panel = panel
        self.ssh_dm.init_add_layout([
            panel,
        ])
        layout_res = mui.VBox(
            [
                self.ssh_dm,
            ]
        ).prop(
            flexDirection="column", flex=1, overflow="hidden", border="1px solid gray"
        )

        self._ssh_ui = layout_res 

    async def _on_term_menu(self, item_id: str):
        if item_id == "Clear":
            await self._terminal.clear()
        else:
            raise ValueError(f"Unknown menu item {item_id}")
    
    async def _on_has_bkpt_change(self, num_bkpt_proc):
        prev_is_paused = self.ssh_dm.model.ssh_status.is_paused
        if num_bkpt_proc > 0:
            cur_is_paused = True 
        else:
            cur_is_paused = False
        if prev_is_paused != cur_is_paused:
            async with self.ssh_dm.draft_update(allow_unmounted=True) as draft:
                draft.ssh_status.is_paused = cur_is_paused
            # sync to leader immediately.
            self.awake_leader_observe_loop()

    def get_terminal(self) -> terminal.AsyncSSHTerminal:
        return self._terminal

    async def start(self):
        assert not self.is_started(), "Worker already started"
        self._local_shutdown_event = asyncio.Event()
        if self._is_compute_worker:
            # connect to ssh
            # raft-only node don't run user cmd, so no need to connect ssh.
            await self._connect_ssh()
        if self._is_raft_worker:
            raft_peers = [
                peer for peer in self._cur_raft_infos if peer.uid != self._peer_info.uid
            ]
            self._raft_node = ExtendedRaftNode(
                self_peer=self._peer_info,
                peers=raft_peers,
                comm=AsyncGrpcRaftComm(self._group_id, self._cfg.comm_cfg),
                config=self._cfg.raft_cfg,
                state_machine=A2AStateMachine(world_size=len(self._cur_raft_infos)),
            )
            self._raft_node.events.on(RaftEventType.COMMIT_APPLIED, self._sync_status_to_ui)
            await self._raft_node.start()
            self._workers_observe_task = asyncio.create_task(
                self._worker_observe_loop(self._local_shutdown_event)
            )
            if self._cfg.workdir.strip() != "":
                workdir = Path(self._cfg.workdir) 
                fs_backend = DraftSimpleFileStoreBackend(workdir, verbose_fs=False, with_bak=True)
                self.ssh_dm.connect_draft_store(f"_cm_raft_group_store_{self._peer_info.uid}_{self._group_id}", fs_backend)
        if self._is_compute_worker:
            self._leader_observe_task = asyncio.create_task(
                self._raft_leader_observe_loop(self._awake_event, self._local_shutdown_event)
            )

    def awake_leader_observe_loop(self):
        if self._leader_observe_task is not None:
            self._awake_event.set()

    async def stop(self):
        if self._local_shutdown_event is None:
            return 
        self._local_shutdown_event.set()
        await self._debouncer.cancel()
        if self._is_compute_worker:
            self._terminal.term_or_kill_all_ssh_child(is_term=False)
            await self._terminal.disconnect()
        if self._raft_node is not None:
            await self._raft_node.stop()
            self._raft_node = None
            if self._workers_observe_task is not None:
                await self._workers_observe_task
                self._workers_observe_task = None
        if self._leader_observe_task is not None:
            await self._leader_observe_task
            self._leader_observe_task = None
        if self._cmd_task is not None:
            self._cmd_task.event.set()
            await self._cmd_task.task
            self._cmd_task = None
        self._local_shutdown_event = None

    async def _raft_manager_act(self, act: RaftMgrActions):
        assert self._raft_node is not None
        cmd: Optional[UserCmd] = None
        if act == RaftMgrActions.START_OR_CANCEL:
            is_start = self.ssh_dm.model.can_workers_run_cmd
            if is_start:
                cmd_str = self.ssh_dm.model.user_cmd
                cmd = UserCmd(
                    type=UserCmdType.SHELL_CMD,
                    content=cmd_str,
                ) 
            else:
                cmd = UserCmd(
                    type=UserCmdType.TRY_CTRL_C,
                    content="",
                )
        elif act == RaftMgrActions.SHUTDOWN_ALL:
            cmd = UserCmd(
                type=UserCmdType.TRY_CTRL_C,
                content="",
            )
        elif act == RaftMgrActions.KILL_ALL:
            cmd = UserCmd(
                type=UserCmdType.KILL_TO_IDLE,
                content="",
            )
        elif act == RaftMgrActions.RECONNECT_ALL_CLIENT:
            cmd = UserCmd(
                type=UserCmdType.RECONNECT_SSH,
                content="",
            )
        if cmd is None:
            CM_LOGGER.error(f"Unimplemented raft manager action {act}")
            return 
        sync_all_workers = self._world_size <= 8 # TODO avoid hardcode
        run_iff_num_worker = None
        final_query_res: Optional[ProposeResult] = None
        leader_info: Optional[PeerInfo] = None

        for raft_info in self._cur_raft_infos:
            if raft_info.uid == self._peer_info.uid:
                query_res = await self._propose_fn(
                    cmd, sync_all_workers, run_iff_num_worker
                )
            else:
                try:
                    query_res = await self._comm.remote_call(
                        raft_info.url,
                        master_serv_names.GROUP_PROPOSE_CMD,
                        self._group_id,
                        cmd,
                        sync_all_workers=sync_all_workers, 
                    )
                except Exception as e:
                    CM_LOGGER.exception(
                        f"Failed to query worker heartbeat from {raft_info.uid}: {e}"
                    )
                    continue
            if query_res.success:
                final_query_res = query_res
                break
            if not query_res.success and query_res.leader_info is not None:
                try:
                    leader_info = query_res.leader_info
                    if query_res.leader_info.uid == self._peer_info.uid:
                        query_res = await self._propose_fn(
                            cmd, sync_all_workers, run_iff_num_worker
                        )
                    else:
                        query_res = await self._comm.remote_call(
                            query_res.leader_info.url,
                            master_serv_names.GROUP_PROPOSE_CMD,
                            self._group_id,
                            cmd,
                            sync_all_workers=sync_all_workers, 
                        )
                    break
                except Exception as e:
                    CM_LOGGER.exception(
                        f"Failed to query worker heartbeat from {raft_info.uid}: {e}"
                    )
                    continue
            if query_res.success:
                final_query_res = query_res
                break
        if leader_info is not None:
            # reorder leader info to first in self._cur_raft_infos
            self._cur_raft_infos = [
                info for info in self._cur_raft_infos if info.uid != leader_info.uid
            ]
            self._cur_raft_infos.insert(0, leader_info)
        if final_query_res is None:
            CM_LOGGER.error("Failed to propose cmd. usually due to raft group not ready.")
            return 

    @property 
    def comm(self):
        return self._comm

    def get_all_worker_states(self):
        assert (
            self._is_raft_worker
            and self._raft_node is not None
            and isinstance(self._raft_node.state_machine, A2AStateMachine)
        )
        state = self._raft_node.state_machine.raft_state
        worker_states: list[A2AWorkerState] = []
        for worker_id, worker_state in state["workers"].items():
            worker_states.append(worker_state)
        return worker_states

    def get_all_raft_infos(self) -> list[PeerInfo]:
        return self._cur_raft_infos

    def get_worker_info(self) -> WorkerInfo:
        return WorkerInfo(
            peer_info=self._peer_info, rank=self._rank, resource=self._resource_info
        )

    def is_raft_leader(self) -> bool:
        return self._is_raft_worker and self._raft_node is not None and self._raft_node.role == RaftRole.LEADER

    def get_leader_info(self) -> Optional[PeerInfo]:
        if self._raft_node is not None:
            return self._raft_node.get_leader_peer_info()
        else:
            # we always put leader to first of the raft list based on
            # leader rpc result.
            return self._cur_raft_infos[0]

    def get_node_spec(self) -> GroupNodeSpec:
        flags = NodeFlags(0)
        if self._is_raft_worker:
            flags |= NodeFlags.IS_RAFT_NODE
        if self._is_compute_worker:
            flags |= NodeFlags.IS_COMPUTE_NODE
        if self._raft_node is not None and self._raft_node.role == RaftRole.LEADER:
            flags |= NodeFlags.IS_RAFT_LEADER
        return GroupNodeSpec(
            peer_info=self._peer_info, flags=flags, resource=self._resource_info
        )

    def _get_init_cmds(self):
        init_cmds = [
            f" export {TENSORPC_ENV_CM_NODEMGR_URL_WITH_PORT}=localhost:{prim.get_server_grpc_port()}\n",
            f" export {TENSORPC_ENV_CM_NODEMGR_RANK}={self._rank}\n",
            f" export {TENSORPC_ENV_CM_NODEMGR_WORLD_SIZE}={self._world_size}\n",
            f" export {TENSORPC_ENV_CM_NODEMGR_BACKEND}=clustermgr\n",
            f" export {TENSORPC_ENV_CM_NODEMGR_GROUP_ID}={self._group_id}\n",
        ]
        if self._cfg.env_fwd_re != "":
            # use re to capture env thatt need to forward to ssh
            env_fwd_re = re.compile(self._cfg.env_fwd_re)
            envs = os.environ.copy()
            for k, v in envs.items():
                if env_fwd_re.match(k):
                    vv = shlex.quote(v)
                    if v != vv:
                        init_cmds.append(f" export {k}={vv}\n")
                    else:
                        init_cmds.append(f' export {k}="{v}"\n')

        return init_cmds

    async def _sync_status_to_ui(self):
        assert self._raft_node is not None and isinstance(
            self._raft_node.state_machine, A2AStateMachine
        )
        state_machine = self._raft_node.state_machine
        workers = state_machine.raft_state["workers"]
        worker_status_dict = state_machine.worker_status_dict
        cur_items: dict[str, list[WorkerSelectItem]] = {
            "disconnected": [],
            "error": [],
            "running": [],
            "idle": [],
            "paused": []
        }
        uid_to_item: dict[str, WorkerSelectItem] = {}
        last_ssh_ts_max = -1
        for worker_id, worker_info in workers.items():
            worker_status = worker_status_dict[worker_id]
            ssh_status = worker_status.status
            if worker_status.is_paused and worker_status.status == "running":
                ssh_status = "paused"
            if ssh_status not in cur_items:
                cur_items[ssh_status] = []
            if worker_status.last_ssh_out_ts > 0:
                last_ssh_ts_max = max(last_ssh_ts_max, worker_status.last_ssh_out_ts)
            select_item = WorkerSelectItem(
                id=worker_id,
                label=f"{worker_info['worker_info'].rank} ({worker_info['worker_info'].peer_info.url})",
                ssh_status=ssh_status,
                url=worker_info["worker_info"].peer_info.url,
                rank=worker_info["worker_info"].rank,
            )
            cur_items[ssh_status].append(select_item)
            uid_to_item[worker_id] = select_item
        for items in cur_items.values():
            # sort workers by (rank, url) in ascending order.
            items.sort(key=lambda x: (x.rank, x.url))
        ssh_status_order = ["paused", "disconnected", "error", "running", "idle"]
        final_list: list[WorkerSelectItem] = []
        for status in ssh_status_order:
            if status in cur_items:
                final_list.extend(cur_items[status])
        num_idle_or_err = len(cur_items["idle"]) + len(cur_items["error"])
        async with self._sync_lock:
            async with self.ssh_dm.draft_update() as draft:
                draft.workers = final_list
                cur_worker_item = self.ssh_dm.model.cur_worker
                if cur_worker_item is not None:
                    cur_worker_id = cur_worker_item.id
                    if cur_worker_id in uid_to_item:
                        draft.cur_worker = uid_to_item[cur_worker_id]
                    else:
                        draft.cur_worker = None
                # TODO we should use is_connected in raft state
                draft.num_connected = len(final_list)
                draft.num_paused = len(cur_items["paused"])
                draft.can_workers_run_cmd = num_idle_or_err == self._world_size
                if self._raft_node is not None:
                    leader_info = self._raft_node.get_leader_peer_info()
                    if leader_info is not None:
                        draft.cur_leader_id = leader_info.uid
                        draft.cur_leader_url = leader_info.url
                if len(cur_items["paused"]) > 0:
                    group_status = GroupSSHStatus.HAS_PAUSED_PROCESS
                elif len(final_list) != self._world_size:
                    group_status = GroupSSHStatus.HAS_DISCONNECTED
                elif len(cur_items["running"]) > 0:
                    if len(cur_items["running"]) != self._world_size:
                        group_status = GroupSSHStatus.HAS_PARTIAL_RUNNING
                    else:
                        group_status = GroupSSHStatus.HAS_RUNNING
                elif len(cur_items["error"]) > 0:
                    group_status = GroupSSHStatus.ALL_IDLE_WITH_LAST_ERROR
                else:
                    group_status = GroupSSHStatus.ALL_IDLE_WITHOUT_ERROR
                draft.group_ssh_status = int(group_status)
                if last_ssh_ts_max > 0:
                    cur_ts = time.time_ns()
                    duration_ns = cur_ts - last_ssh_ts_max
                    draft.worker_last_activity = f"{humanize.naturaldelta(duration_ns / 1e9)} ago"
                    
    async def _ssh_event_cb(self, event: SSHEvent):
        if event.ev_type == SSHEventType.Command:
            assert isinstance(event, CommandEvent)
            if event.type == CommandEventType.COMMAND_OUTPUT_START:
                async with self.ssh_dm.draft_update(allow_unmounted=True) as draft:
                    draft.ssh_status.status = "running"
                    draft.ssh_status.exit_code = None
                # sync to leader immediately.
                self.awake_leader_observe_loop()

            elif event.type == CommandEventType.COMMAND_COMPLETE:
                async with self.ssh_dm.draft_update(allow_unmounted=True) as draft:
                    if event.arg is not None:
                        return_code = int(event.arg)
                        if return_code != 0:
                            draft.ssh_status.status = "error"
                        else:
                            draft.ssh_status.status = "idle"
                        draft.ssh_status.exit_code = return_code
                    else:
                        draft.ssh_status.status = "idle"
                    draft.ssh_status.is_paused = False
                # sync to leader immediately.
                self.awake_leader_observe_loop()

            elif event.type == CommandEventType.PROMPT_END:
                async with self.ssh_dm.draft_update(allow_unmounted=True) as draft:
                    draft.ssh_status.status = "idle"
                    draft.ssh_status.exit_code = None
                    draft.ssh_status.is_paused = False

        elif (
            event.ev_type == SSHEventType.Eof or event.ev_type == SSHEventType.Exception
        ):
            async with self.ssh_dm.draft_update(allow_unmounted=True) as draft:
                draft.ssh_status.status = "disconnected"
                draft.ssh_status.exit_code = None
                draft.ssh_status.is_paused = False
            # sync to leader immediately.
            self.awake_leader_observe_loop()

    async def _connect_ssh(self):
        init_cmds = self._get_init_cmds()
        await self._terminal.connect_with_new_desc(
            self._cfg.local_ssh_desc,
            init_cmds=init_cmds,
            event_callback=self._ssh_event_cb,
        )
        term_state = self._terminal.get_current_state()
        assert term_state is not None 
        if self._raft_mgr_panel.debug_panel is not None:
            self._raft_mgr_panel.debug_panel.set_parent_pid(term_state.pid)

    def get_components_dict(self) -> dict[str, mui.FlexBox]:
        return {
            WorkerUIType.TERMINAL.value: self._ssh_ui,
        }

    async def _worker_observe_loop(self, shutdown_ev: asyncio.Event):
        # only raft group watch workers.
        if self._raft_node is None:
            return
        shutdown_ev_task = asyncio.create_task(
            shutdown_ev.wait(), name="raft_leader_observe_loop_local_shutdown_wait")

        global_shutdown_ev = prim.get_async_shutdown_event()
        global_shutdown_ev_task = asyncio.create_task(global_shutdown_ev.wait())
        while True:
            sleep_task = asyncio.create_task(
                asyncio.sleep(self._cfg.worker_check_interval)
            )
            done, pending = await asyncio.wait(
                [global_shutdown_ev_task, shutdown_ev_task, sleep_task], return_when=asyncio.FIRST_COMPLETED
            )
            if global_shutdown_ev_task in done or shutdown_ev_task in done:
                for task in pending:
                    task.cancel()
                CM_LOGGER.warning("Shutdown event received, stop worker observe loop.")
                break
            # check all workers last timestamp
            cur_ts = time.time_ns()
            try:
                assert isinstance(self._raft_node.state_machine, A2AStateMachine)
                worker_status_dict = self._raft_node.state_machine.worker_status_dict
                raft_workers = self._raft_node.state_machine.raft_state["workers"]
                for worker_id, worker_state in raft_workers.items():
                    worker_info = worker_state["worker_info"]
                    if not worker_state["is_connected"]:
                        continue
                    if worker_id not in worker_status_dict:
                        worker_status_dict[worker_id] = WorkerSSHStatus(
                            status="idle", last_ts=cur_ts
                        )
                        continue
                    status = worker_status_dict[worker_id]
                    if cur_ts - status.last_ts > self._cfg.worker_disconnect_timeout * 1e9:
                        # mark worker as disconnected
                        worker_state["is_connected"] = False
                        await self._raft_node.propose(
                            UserCmd(
                                type=UserCmdType.UNREGISTER_WORKER,
                                worker_info=worker_info,
                            )
                        )
                        worker_status_dict[worker_id].status = "disconnected"
                await self._sync_status_to_ui()
            except:
                traceback.print_exc()
                raise

    async def _run_runtime_cmd(self, cmd: Any):
        # TODO implement this.
        raise NotImplementedError

    async def _raft_leader_observe_loop(self, awake_event: asyncio.Event, shutdown_ev: asyncio.Event):
        global_shutdown_ev = prim.get_async_shutdown_event()
        global_shutdown_ev_task = asyncio.create_task(
            global_shutdown_ev.wait(), name="raft_leader_observe_loop_shutdown_wait"
        )
        shutdown_ev_task = asyncio.create_task(
            shutdown_ev.wait(), name="raft_leader_observe_loop_local_shutdown_wait")
        awake_immediately_task = asyncio.create_task(
            awake_event.wait(), name="raft_leader_observe_loop_awake_wait"
        )

        while True:
            sleep_task = asyncio.create_task(
                asyncio.sleep(self._cfg.leader_check_interval)
            )
            done, pending = await asyncio.wait(
                [global_shutdown_ev_task, shutdown_ev_task, sleep_task, awake_immediately_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if global_shutdown_ev_task in done or shutdown_ev_task in done:
                for task in pending:
                    task.cancel()
                CM_LOGGER.warning("Shutdown event received, stop worker observe loop.")
                break
            if awake_immediately_task in done:
                # cancel sleep task
                sleep_task.cancel()
                awake_event.clear()
                awake_immediately_task = asyncio.create_task(awake_event.wait())
            last_ssh_ts :int = -1
            cur_ssh_state = self._terminal.get_current_state()
            if cur_ssh_state is not None:
                last_ssh_ts = cur_ssh_state.last_ts
            my_ssh_status = dataclasses.replace(self.ssh_dm.model.ssh_status, last_ssh_out_ts=last_ssh_ts)
            query_args = {
                "group_id": self._group_id,
                "worker_peer_info": self._peer_info,
                "rank": self._rank,
                "ssh_status": my_ssh_status,
            }
            query_res: Optional[A2AWorkerQueryResult] = None
            leader_info: Optional[PeerInfo] = None
            for raft_info in self._cur_raft_infos:
                if raft_info.uid == self._peer_info.uid:
                    query_res = await self.handle_worker_heartbeat(
                        self._peer_info, self._rank, my_ssh_status
                    )
                else:
                    try:
                        query_res = await self._comm.remote_call(
                            raft_info.url,
                            master_serv_names.RAFT_WORKER_HEARTBEAT,
                            **query_args,
                        )
                    except Exception as e:
                        CM_LOGGER.exception(
                            f"Failed to query worker heartbeat from {raft_info.uid}: {e}"
                        )
                        continue
                if query_res is not None and query_res["leader_info"] is not None:
                    try:
                        leader_info = query_res["leader_info"]
                        if leader_info.uid == self._peer_info.uid:
                            query_res = await self.handle_worker_heartbeat(
                                self._peer_info, self._rank, my_ssh_status
                            )
                        else:
                            query_res = await self._comm.remote_call(
                                leader_info.url,
                                master_serv_names.RAFT_WORKER_HEARTBEAT,
                                **query_args,
                            )
                        break
                    except Exception as e:
                        CM_LOGGER.exception(
                            f"Failed to query worker heartbeat from {raft_info.uid}: {e}"
                        )
                        continue
            if leader_info is not None:
                # reorder leader info to first in self._cur_raft_infos
                self._cur_raft_infos = [
                    info for info in self._cur_raft_infos if info.uid != leader_info.uid
                ]
                self._cur_raft_infos.insert(0, leader_info)
            if query_res is None:
                CM_LOGGER.error(
                    "Failed to query worker heartbeat from all raft peers. exiting observe loop."
                )
                for task in pending:
                    task.cancel()
                break
            if not query_res["is_leader"]:
                # may be no leader. warn and retry in next loop.
                CM_LOGGER.warning(
                    "No leader or failed to get leader info. retrying in next loop."
                )
                continue
            last_cmd = query_res["last_cmd"]
            cmd_version = query_res["cmd_version"]
            if cmd_version == -1:
                # no cmd from leader, just continue.
                continue
            async with self._cmd_apply_lock:
                if (
                    query_res["runtime_cmd"] is not None
                    and query_res["runtime_cmd_uid"]
                    not in self._worker_state.finished_rt_cmd_uids
                ):
                    # we have a new runtime cmd to run. run it immediately.
                    CM_LOGGER.warning(
                        f"Received new runtime cmd {query_res['runtime_cmd']} with uid {query_res['runtime_cmd_uid']} from leader, running it."
                    )
                    await self._run_runtime_cmd(query_res["runtime_cmd"])
                    self._worker_state.finished_rt_cmd_uids.put(
                        query_res["runtime_cmd_uid"], query_res["runtime_cmd"]
                    )

                assert cmd_version >= 0 and last_cmd is not None
                if cmd_version == self._worker_state.cmd_version:
                    # already up to date. do nothing.
                    continue
                elif cmd_version < self._worker_state.cmd_version:
                    CM_LOGGER.warning(
                        f"Received stale cmd version {cmd_version} from leader, current version is {self._worker_state.cmd_version}. ignoring."
                    )
                    continue

                term_state = self._terminal.get_current_state()
                if term_state is None:
                    CM_LOGGER.warning(
                        "Failed to get terminal state, may be terminal is not ready. retrying in next loop."
                    )
                    continue
                if last_cmd.type == UserCmdType.RECONNECT_SSH:
                    if self._cmd_task is not None:
                        self._cmd_task.event.set()
                        await self._cmd_task.task
                        self._cmd_task = None
                    await self.get_terminal().disconnect()
                    await self._connect_ssh()
                elif last_cmd.type == UserCmdType.TRY_CTRL_C:
                    await self.get_terminal().send_ctrl_c()
                elif last_cmd.type == UserCmdType.KILL_TO_IDLE:
                    killed_procs = self.get_terminal().term_or_kill_all_ssh_child(
                        is_term=False
                    )
                    if self._cmd_task is not None:
                        self._cmd_task.event.set()
                        await self._cmd_task.task
                        self._cmd_task = None

                    CM_LOGGER.warning(
                        f"Kill to idle cmd received, killed procs: {killed_procs}"
                    )
                    # TODO should we wait here?
                elif last_cmd.type == UserCmdType.SHELL_CMD:
                    if (
                        term_state.current_cmd != ""
                        and term_state.current_cmd_rpc_future is not None
                    ):
                        # kill current task
                        killed_procs = self.get_terminal().term_or_kill_all_ssh_child(
                            is_term=False
                        )
                        if self._cmd_task is not None:
                            self._cmd_task.event.set()
                            await self._cmd_task.task
                            self._cmd_task = None

                        CM_LOGGER.warning(
                            f"Kill to idle cmd received, killed procs: {killed_procs}"
                        )
                    fut = term_state.current_cmd_rpc_future
                    if fut is not None:
                        await fut
                    shell_cmd = last_cmd.content
                    exit_ev = asyncio.Event()
                    self._cmd_task = CmdTaskState(asyncio.create_task(self._cmd_waiter(shell_cmd, exit_ev)), exit_ev)
                    # await self.get_terminal().ssh_command_rpc_future(shell_cmd + "\n")
                else:
                    CM_LOGGER.warning(
                        f"Received unknown cmd type {last_cmd.type} from leader, ignoring."
                    )
                    continue
                self._worker_state.cur_cmd = last_cmd
                self._worker_state.cmd_version = cmd_version

    async def _cmd_waiter(self, cmd: str, exit_ev: asyncio.Event):
        CM_LOGGER.warning("Launch command:")
        rich.print(cmd)
        ssh_state = self._terminal.get_current_state()
        assert ssh_state is not None 
        shell_info = ssh_state.shell_info
        assert shell_info is not None 
        if shell_info.type == "zsh":
            suffix = "zsh"
            shell_cmd_prefix = "zsh -i"
        elif shell_info.type == "bash":
            suffix = "sh"
            shell_cmd_prefix = "bash -i"
        else:
            raise RuntimeError(f"Unsupported shell type: {shell_info.type}")

        if self._cfg.workdir.strip() != "":
            tempfile_ctx = nullcontext()
        else:
            tempfile_ctx = tempfile.NamedTemporaryFile(mode="w", prefix=f"{self._group_id}-rank-cmd-{self._rank}-", suffix=f".{suffix}", delete=True)
        with tempfile_ctx as tmp_file:
            if tmp_file is None:
                shell_file_path = Path(self._cfg.workdir) / "sync" / f"{self._group_id}-rank-cmd-{self._rank}.{suffix}"
                with shell_file_path.open("w") as f:
                    if shell_info.type == "bash":
                        f.write("unset HISTFILE\n")
                    f.write(cmd)
            else:
                shell_file_path = Path(tmp_file.name)
                if shell_info.type == "bash":
                    tmp_file.write("unset HISTFILE\n")
                tmp_file.write(cmd)
                tmp_file.flush()
            shell_cmd = f" {shell_cmd_prefix} {shell_file_path.absolute()}"
            shutdown_ev = prim.get_async_shutdown_event()
            shutdown_ev_task = asyncio.create_task(shutdown_ev.wait(), name="ft-ssh-cmdwaiter-wait")
            try:
                run_cmd_task = asyncio.create_task(self._terminal.ssh_command_rpc(shell_cmd), name="cmd task")
                done, pending = await asyncio.wait(
                    [shutdown_ev_task, run_cmd_task],
                    return_when=asyncio.FIRST_COMPLETED)
                if shutdown_ev_task in done:
                    await cancel_task(run_cmd_task)
                    # TODO use ctrl-c->terminal->kill sequence
                    await self._terminal.disconnect()
                    return 
                assert run_cmd_task in done, "run_cmd_task should be done"
                res = run_cmd_task.result()
            except:
                CM_LOGGER.error("cmd waiter error.", exc_info=True)
                raise
            finally:
                exit_ev.set()
                self._cmd_task = None
                CM_LOGGER.warning("cmd waiter finished.")


    async def handle_worker_heartbeat(
        self,
        worker_peer_info: PeerInfo,
        rank: int,
        ssh_status: WorkerSSHStatus,
    ) -> A2AWorkerQueryResult:
        # this handler should only be called in raft nodes.
        assert self._raft_node is not None
        raft_node = self._raft_node
        worker_uid = worker_peer_info.uid
        worker_info = WorkerInfo(peer_info=worker_peer_info, rank=rank)
        # for log in group.worker._raft_node.log
        if raft_node.role != RaftRole.LEADER:
            # return leader info
            return {
                "last_cmd": None,
                "cmd_version": -1,
                "leader_info": raft_node.get_leader_peer_info(),
                "runtime_cmd": None,
                "runtime_cmd_uid": -1,
                "is_leader": False,
            }
        assert isinstance(self._raft_node.state_machine, A2AStateMachine)
        state = self._raft_node.state_machine.raft_state
        # update worker status

        if not self._raft_node.state_machine.has_worker(worker_uid):
            # add new cmd to raft log
            await self._raft_node.propose(
                UserCmd(
                    type=UserCmdType.REGISTER_WORKER,
                    worker_info=worker_info,
                )
            )
        else:
            is_connected = state["workers"][worker_uid]["is_connected"]
            if not is_connected:
                # update worker to connected status
                # TODO better way to update connected status, may be new cmd type.
                await self._raft_node.propose(
                    UserCmd(
                        type=UserCmdType.REGISTER_WORKER,
                        worker_info=worker_info,
                    )
                )
        # update last timestamp for this worker
        worker_status_dict = self._raft_node.state_machine.worker_status_dict
        if worker_uid not in worker_status_dict:
            worker_status_dict[worker_uid] = WorkerSSHStatus(
                status=ssh_status.status, last_ts=time.time_ns(), exit_code=ssh_status.exit_code,
                is_paused=ssh_status.is_paused, last_ssh_out_ts=ssh_status.last_ssh_out_ts
            )
        else:
            worker_status = self._raft_node.state_machine.worker_status_dict[
                worker_uid
            ]
            worker_status.last_ts = time.time_ns()
            worker_status.status = ssh_status.status
            worker_status.exit_code = ssh_status.exit_code
            worker_status.is_paused = ssh_status.is_paused
            worker_status.last_ssh_out_ts = ssh_status.last_ssh_out_ts
        await self._debouncer.call(self._sync_status_to_ui)
        # print("!!!", worker_uid, self._peer_info.uid, ssh_status.status)
        return {
            "last_cmd": state["last_cmd"],
            "cmd_version": state["cmd_version"],
            "leader_info": None,
            "runtime_cmd": state["runtime_cmd"],
            "runtime_cmd_uid": state["runtime_cmd_uid"],
            "is_leader": True,
        }

