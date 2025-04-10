import asyncio
import enum
import json
from pathlib import Path
import traceback
from typing import Any, Callable, Optional, Union

import async_timeout
from tensorpc.autossh.core import SSHConnDesc
from tensorpc.core.asyncclient import AsyncRemoteManager
from tensorpc.core.asynctools import cancel_task
from tensorpc.core.datamodel.events import DraftChangeEvent
from tensorpc.dock import terminal
import tensorpc.core.dataclass_dispatch as dataclasses
import uuid
from tensorpc.dock.components import mui
from tensorpc.utils import get_service_key_by_type, rich_logging
from tensorpc.utils.wait_tools import get_primary_ip 
from tensorpc.core import marker, prim
from tensorpc.core.moduleid import import_dynamic_func
import tensorpc.core.datamodel as D
import psutil 

class FTStatus(enum.IntEnum):
    OK = 0
    MASTER_DISCONNECTED = 1
    WORKER_DISCONNECTED = 2
    UNKNOWN = 3

class SSHStatus(enum.IntEnum):
    IDLE = 0
    DISCONNECTED = 1
    RUNNING = 2

LOGGER = rich_logging.get_logger("distssh")

@dataclasses.dataclass
class FTSSHServerArgs:
    rank: int
    world_size: int

    # used to save its ip to a folder to ensure
    # failed worker can discover the master
    # assume your cluster has a NAS.
    ip_folder: str
    cmd: str
    password: str
    username: str = "root"
    max_retries: int = 1
    log_path: Optional[str] = None
    # rate limit for sending message to slack or other notification service
    msg_throttle: int = 30
    # distributed arguments
    master_discovery_fn: Optional[str] = None
    heartbeat_interval: int = 5
    # message parser arguments
    # called every time a new line event is received
    # sig: def func(events: list[Event], current_cmd: str)
    msg_handler: Optional[str] = None
    # called once after init, can be used to log environment info
    # to file.
    init_info_getter: Optional[str] = None
    # called when error happens.
    # sig: def func(title: str, msg: str)
    error_handle_throttle: int = 15
    error_handler: Optional[str] = None
    # if set, log traceback of all child process to file (main-thread only)
    # only enabled if log_path is set and pyspy_period is set, 
    # won't be logged to stdout.
    pyspy_period: Optional[int] = None
    request_pty: bool = False

@dataclasses.dataclass
class FTState:
    label: str
    rank: int
    uuid: str
    ip: str
    is_master: bool 
    cur_cmd: Optional[str] = None
    status: FTStatus = FTStatus.OK
    ssh_status: SSHStatus = SSHStatus.IDLE

    def get_box_state(self):
        return FTStatusBoxState(
            rank=self.rank,
            ip=self.ip,
            status=self.status,
            ssh_status=self.ssh_status,
        )

@dataclasses.dataclass
class MasterUIState:
    client_states: list[FTState]
    selected_client_state: Optional[dict[str, Any]] = None
    cur_cmd: str = ""
    cmd_history: list[str] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class FTStatusBoxState:
    rank: int 
    ip: str
    status: FTStatus
    ssh_status: SSHStatus


class WorkersStatusBox(mui.DataFlexBox):
    def __init__(self, init_data_list: list[FTStatusBoxState], box_size: int = 8):
        self._box_template = mui.HBox([])
        self._box_template.prop(width=box_size, height=box_size)
        self._box_template.bind_fields(backgroundColor="color")

        super().__init__(self._box_template, init_data_list)
        self.prop(flexFlow="row wrap", flex=1)



class FaultToleranceUIMaster(mui.FlexBox):
    def __init__(self, state: FTState, ui_state: MasterUIState, states: list[FTState], term: terminal.AsyncSSHTerminal):
        if state.is_master:
            title = "Main Worker"
        else:
            title = f"Worker ({state.rank})"

        start_or_stop_btn = mui.IconButton(mui.IconType.Stop).prop(iconSize="small", size="small")
        header_str = mui.Typography(title).prop(variant="h6", color="primary")
        kill_btn = mui.IconButton(mui.IconType.Delete).prop(iconSize="small", size="small",
            confirmTitle="Dangerous Operation", confirmMessage="Are you sure to kill running process?")
        rank_select = mui.Autocomplete("Workers", []).prop(muiMargin="dense", size="small")
        self.worker_status_box = WorkersStatusBox([state.get_box_state() for state in states])
        header = mui.HBox([
            mui.HBox([
                header_str,
            ]).prop(flex=1),
            start_or_stop_btn,
            kill_btn,
        ])
        self._remote_box = mui.HBox([])
        self.dm = mui.DataModel(state, [
            header,
        ])
        draft = self.dm.get_draft()
        start_or_stop_btn.bind_fields(icon=D.where(draft.cur_cmd != None, mui.IconType.Stop, mui.IconType.PlayArrow))
        kill_btn.bind_fields(disabled=D.where(draft.cur_cmd == None, True, False))
        self.ui_dm = mui.DataModel(ui_state, [
            
        ])
        master_draft = self.ui_dm.get_draft()

        self.ui_dm.install_draft_change_handler(master_draft.client_states, self._handle_client_state_change, handle_child_change=True)

        rank_select.bind_draft_change(master_draft.selected_client_state)
        rank_select.bind_fields(options=master_draft.client_states)

        super().__init__([
            self.dm,
            self.ui_dm,
            term,
        ])
        self.prop(flexDirection="column", flex=1)

    async def _handle_client_state_change(self, ev: DraftChangeEvent):
        states: list[FTState] = ev.new_value
        ui_states = [state.get_box_state() for state in states]
        await self.send_and_wait(self.worker_status_box.update_event(dataList=ui_states))


class FaultToleranceUIClient(mui.FlexBox):
    def __init__(self, state: FTState, term: terminal.AsyncSSHTerminal):

        self._client_box = mui.HBox([])
        self.dm = mui.DataModel(state, [

        ])
        pass 


class FaultToleranceSSHServer:
    def __init__(self,
                 config_dict: dict,
                 default_url: str = "localhost:22") -> None:
        cfg = FTSSHServerArgs(**config_dict)
        self._cfg = cfg
        self._conn_desc = SSHConnDesc(default_url, cfg.username, cfg.password)
        self._terminal = terminal.AsyncSSHTerminal()
        self._master_rank = 0
        ip = get_primary_ip()
        state = FTState(
            label=f"{cfg.rank} ({ip})",
            rank=cfg.rank,
            uuid=uuid.uuid4().hex,
            ip=ip,
            is_master=cfg.rank == self._master_rank,
        )
        self._is_master = cfg.rank == self._master_rank
        self._master_ui_state = MasterUIState(
            client_states=[],
            cur_cmd="",
            cmd_history=[],
        )
        for j in range(cfg.world_size):
            if j == self._master_rank:
                self._master_ui_state.client_states.append(state)
            else:
                init_client_state = FTState(
                    label=f"{j} ({ip})",
                    rank=j,
                    uuid="",
                    ip="",
                    is_master=False,
                    status=FTStatus.UNKNOWN,
                )
                self._master_ui_state.client_states.append(init_client_state)
        self._master_ui = FaultToleranceUIMaster(state, self._master_ui_state, self._terminal)
        self._client_ui = FaultToleranceUIClient(state, self._terminal)

        self._master_discovery_fn: Optional[Callable[[], Optional[str]]] = None

        self._client_robjs: dict[int, AsyncRemoteManager] = {}
        self._master_robj: Optional[AsyncRemoteManager] = None

        self._loop_task: Optional[asyncio.Task] = None
        self._cmd_task: Optional[asyncio.Task] = None

    @property 
    def state(self):
        if self._is_master:
            return self._master_ui.dm.model 
        else:
            return self._client_ui.dm.model

    @marker.mark_server_event(event_type=marker.ServiceEventType.Init)
    async def _init(self):
        await self._terminal.connect_with_new_desc(self._conn_desc)
        file_name = Path(self._cfg.ip_folder) / f"{self.state.rank}-{self.state.ip}.json"
        if self._cfg.master_discovery_fn is not None:
            self._master_discovery_fn = import_dynamic_func(
                self._cfg.master_discovery_fn, is_func_id=True)
        else:
            if not Path(self._cfg.ip_folder).exists():
                Path(self._cfg.ip_folder).mkdir(parents=True, exist_ok=True)
            assert Path(self._cfg.ip_folder).exists(), f"{self._cfg.ip_folder} does not exist"
            with file_name.open("w") as f:
                json.dump({
                    "ip": self.state.ip,
                    "uuid": self.state.uuid,
                }, f, indent=4)
        self._loop_task = asyncio.create_task(self._heartbeat_loop())

    @marker.mark_server_event(event_type=marker.ServiceEventType.Exit)
    async def _close(self):
        if self.state.is_master:
            for rank, robj in self._client_robjs.items():
                try:
                    await robj.close()
                except:
                    traceback.print_exc()
        else:
            if self._master_robj is not None:
                try:
                    await self._master_robj.close()
                except:
                    traceback.print_exc()

    def _get_ssh_child_pids(self):
        state = self._terminal.get_current_state()
        if state is None:
            return []
        ssh_pid = state.pid
        ssh_proc = psutil.Process(ssh_pid)
        child_pids = []
        for child in ssh_proc.children(recursive=True):
            if child.pid != ssh_pid:
                child_pids.append(child.pid)
        return child_pids

    def _kill_all_ssh_child(self):
        state = self._terminal.get_current_state()
        if state is None:
            return []
        ssh_pid = state.pid
        ssh_proc = psutil.Process(ssh_pid)
        child_pids = []
        for child in ssh_proc.children(recursive=True):
            if child.pid != ssh_pid:
                child.kill()
        
    async def master_run_cmd(self, cmd: str):
        assert self._cmd_task is None, "master can only run one command at a time" 
        if self.state.status != FTStatus.OK:
            raise RuntimeError("master is not in OK state")
        self._cmd_task = asyncio.create_task(self._cmd_waiter(cmd))
        for robj in self._client_robjs.values():
            await robj.remote_call(get_service_key_by_type(FaultToleranceSSHServer, "client_run_cmd"), cmd)

    async def master_cancel_cmd(self):
        if self._cmd_task is not None:
            state = self._terminal.send_ctrl_c()
            async with async_timeout.timeout(30):
                await self._cmd_task

    async def client_run_cmd(self, cmd: str):
        assert self._cmd_task is None, "master can only run one command at a time" 
        if self.state.status != FTStatus.OK:
            raise RuntimeError("worker is not in OK state")
        self._cmd_task = asyncio.create_task(self._cmd_waiter(cmd))

    async def _cmd_waiter(self, cmd: str):
        shutdown_ev = prim.get_async_shutdown_event()
        shutdown_ev_task = asyncio.create_task(shutdown_ev.wait())
        run_cmd_task = asyncio.create_task(self._terminal.ssh_command_rpc(cmd))
        done, pending = await asyncio.wait(
            [shutdown_ev_task, run_cmd_task],
            return_when=asyncio.FIRST_COMPLETED)
        if shutdown_ev_task in done:
            await cancel_task(run_cmd_task)
            # TODO use ctrl-c->terminal->kill sequence
            await self._terminal.disconnect()
            return 

    async def set_worker_state(self, state: FTState):
        assert state.rank != self._master_rank, "master rank should not be set"
        if self.state.is_master:
            async with self._master_ui.ui_dm.draft_update() as draft_master:
                draft_master.client_states[state.rank] = state
            if state.rank in self._client_robjs:
                prev_robj = self._client_robjs.pop(state.rank)
                try:
                    await prev_robj.close()
                except:
                    traceback.print_exc()
            robj = AsyncRemoteManager(f"{state.ip}:{prim.get_server_grpc_port()}")
            self._client_robjs[state.rank] = robj

    async def _query_master_robj(self, num_retry: int = 12, retry_interval: int = 5):
        folder_p = Path(self._cfg.ip_folder)
        port = prim.get_server_grpc_port()
        for i in range(num_retry):
            if self._master_discovery_fn is not None:
                master_ip = self._master_discovery_fn()
                if master_ip is not None:
                    robj = AsyncRemoteManager(f"{master_ip}:{port}")
                    return robj
            else:
                master_files = list(folder_p.glob(f"{self._master_rank}-*.json"))
                if len(master_files) > 0:
                    with master_files[0].open("r") as f:
                        master_info = json.load(f)
                        master_ip = master_info["ip"]
                        robj = AsyncRemoteManager(f"{master_ip}:{port}")
                        return robj
            await asyncio.sleep(retry_interval)

    async def _master_wait_for_all_worker(self, num_retry: int = 30, retry_interval: int = 2):
        for i in range(num_retry):
            if len(self._client_robjs) == self._cfg.world_size - 1:
                return True
            await asyncio.sleep(retry_interval)
        return False

    async def _heartbeat_loop(self):
        shutdown_ev = prim.get_async_shutdown_event()
        shutdown_ev_task = asyncio.create_task(shutdown_ev.wait())

        if self.state.is_master:
            while True:
                sleep_task = asyncio.create_task(
                    asyncio.sleep(self._cfg.heartbeat_interval))
                done, _ = await asyncio.wait(
                    [shutdown_ev_task, sleep_task],
                    return_when=asyncio.FIRST_COMPLETED)
                if shutdown_ev_task in done:
                    await cancel_task(sleep_task)
                    break 
                if self.state.status == FTStatus.OK:
                    for client_state in self._master_ui_state.client_states:
                        if not client_state.is_master:
                            assert client_state.status != FTStatus.UNKNOWN
                    tasks = []
                    ranks: list[int] = []
                    for rank, robj in self._client_robjs.items():
                        tasks.append(robj.health_check(timeout=10))
                        ranks.append(rank)
                    res = asyncio.gather(*tasks, return_exceptions=True)
                    has_disconnect = False
                    async with self._master_ui.dm.draft_update() as draft:
                        async with self._master_ui.ui_dm.draft_update() as draft_master:
                            for rank, r in zip(ranks, res):
                                if isinstance(r, BaseException):
                                    draft_master.client_states[rank].status = FTStatus.UNKNOWN
                                    poped_robj = self._client_robjs.pop(rank)
                                    try:
                                        await poped_robj.close()
                                    except:
                                        traceback.print_exc()
                                    LOGGER.warning(f"worker {rank} disconnected")
                                    has_disconnect = True
                            if has_disconnect:
                                draft.status = FTStatus.WORKER_DISCONNECTED
                else:
                    res = await self._master_wait_for_all_worker()
                    if not res:
                        LOGGER.warning("master wait for all worker timeout")
                        shutdown_ev.set()
                        break
                    async with self._master_ui.dm.draft_update() as draft:
                        draft.status = FTStatus.OK
        else:
            while True:
                sleep_task = asyncio.create_task(
                    asyncio.sleep(self._cfg.heartbeat_interval))
                done, _ = await asyncio.wait(
                    [shutdown_ev_task, sleep_task],
                    return_when=asyncio.FIRST_COMPLETED)
                if shutdown_ev_task in done:
                    await cancel_task(sleep_task)
                    break 
                if self.state.status == FTStatus.OK:
                    if self._master_robj is None:
                        robj = await self._query_master_robj()
                        if robj is None:
                            prim.get_async_shutdown_event().set()
                            break 
                        self._master_robj = robj
                    else:
                        robj = self._master_robj
                    try:
                        await self._master_robj.remote_call(get_service_key_by_type(FaultToleranceSSHServer, "set_worker_state"), self.state)
                    except:
                        traceback.print_exc()
                        self._master_robj = None 
                        async with self._client_ui.dm.draft_update() as draft:
                            draft.status = FTStatus.MASTER_DISCONNECTED
                else:
                    robj = await self._query_master_robj()
                    if robj is None:
                        prim.get_async_shutdown_event().set()
                        break
                    self._master_robj = robj