import asyncio
import enum
from functools import partial
import json
from pathlib import Path
import traceback
from typing import Any, Awaitable, Callable, Optional, Union

import async_timeout
import grpc
from tensorpc.autossh.core import SSHConnDesc
from tensorpc.core.asyncclient import AsyncRemoteManager
from tensorpc.core.asynctools import cancel_task
from tensorpc.core.datamodel.draftstore import DraftSimpleFileStoreBackend
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
from tensorpc.dock.serv_names import serv_names as app_serv_names
from tensorpc.apps.distssh.constants import (TENSORPC_DISTSSH_UI_KEY)

class CmdStatus(enum.IntEnum):
    IDLE = 0
    RUNNING = 1


class FTStatus(enum.IntEnum):
    OK = 0
    MASTER_DISCONNECTED = 1
    WORKER_DISCONNECTED = 2
    UNKNOWN = 3

class SSHStatus(enum.IntEnum):
    IDLE = 0
    DISCONNECTED = 1
    RUNNING = 2
    ERROR = 3

LOGGER = rich_logging.get_logger("distssh")

@dataclasses.dataclass
class FTSSHServerArgs:
    rank: int
    world_size: int

    # used to save its ip to a folder to ensure
    # failed worker can discover the master
    # assume your cluster has a NAS.
    workdir: str
    cmd: str
    password: str
    username: str = "root"
    # max_retries: int = 1
    # log_path: Optional[str] = None
    # distributed arguments
    master_discovery_fn: Optional[str] = None
    heartbeat_interval: int = 5
    # 5 min
    # when some worker or master disconnected, we assume
    # your cluster manager will restart it. so we 
    # wait for 5 min to check if the worker is really.
    disconnect_total_retry: int = 60
    disconnect_rpc_check_timeout: int = 2
    # cmd shutdown configs
    cmd_shutdown_timeout: int = 10
    cmd_ctrl_c_retry: int = 2

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


@dataclasses.dataclass
class MasterUIState:
    cmd_status: CmdStatus
    client_states: list[FTState]
    selected_client_state: Optional[dict[str, Any]] = None
    cmd: str = "echo $HOME"
    cmd_history: list[str] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class FTStatusBoxState:
    id: str
    rank: int 
    ip: str
    status: FTStatus
    ssh_status: SSHStatus
    color: str
    selected: bool
    @staticmethod 
    def from_ft_state(ft_state: FTState, selected: bool):
        if ft_state.status == FTStatus.WORKER_DISCONNECTED or ft_state.status == FTStatus.UNKNOWN:
            color = "gray"
        else:
            if ft_state.ssh_status == SSHStatus.IDLE:
                color = "deepskyblue"
            elif ft_state.ssh_status == SSHStatus.DISCONNECTED:
                color = "red"
            elif ft_state.ssh_status == SSHStatus.RUNNING:
                color = "lime"
            elif ft_state.ssh_status == SSHStatus.ERROR:
                color = "red"
            else:
                color = "gray"
        return FTStatusBoxState(
            id=str(ft_state.rank),
            rank=ft_state.rank,
            ip=ft_state.ip,
            status=ft_state.status,
            ssh_status=ft_state.ssh_status,
            color=color,
            selected=selected,
        )


class WorkersStatusBox(mui.DataFlexBox):
    def __init__(self, init_data_list: list[FTStatusBoxState], on_click: Callable[[mui.Event], mui.CORO_NONE], box_size: int = 10):
        self._box_template = mui.HBox([])
        box_size_px = f"{box_size}px"
        self._box_template.prop(width=box_size_px, height=box_size_px, margin="2px")
        self._box_template.bind_fields(backgroundColor="color", border=f"where(selected, '2px solid lightpink', '2px solid transparent')",)
        self._box_template.event_click.on_standard(on_click)
        self._selected_idx = -1
        super().__init__(self._box_template, init_data_list)
        self.prop(flexFlow="row wrap", padding="10px")


class FaultToleranceUIMaster(mui.FlexBox):
    def __init__(self, master_rank: int, ui_state: MasterUIState, term: terminal.AsyncSSHTerminal, port: int,
            start_or_cancel_fn: Callable[[], mui.CORO_NONE],
            stop_fn: Callable[[], mui.CORO_NONE], kill_fn: Callable[[], mui.CORO_NONE]):
        master_state = ui_state.client_states[master_rank]
        states = ui_state.client_states
        self._master_rank = master_rank
        self._port = port
        if master_state.is_master:
            title = "Main Worker"
        else:
            title = f"Worker ({master_state.rank})"

        start_or_cancel_btn = mui.IconButton(mui.IconType.PlayArrow, start_or_cancel_fn).prop(iconSize="small", size="small")
        stop_btn = mui.IconButton(mui.IconType.Stop, stop_fn).prop(iconSize="small", size="small", muiColor="error",
            confirmTitle="Dangerous Operation", confirmMessage="Are you sure to kill running process?",
            tooltip="shutdown command")
        kill_btn = mui.IconButton(mui.IconType.Delete, kill_fn).prop(iconSize="small", size="small",
            confirmTitle="Dangerous Operation", confirmMessage="Are you sure to kill running process?",
            tooltip="kill all child process")

        header_str = mui.Typography(title).prop(variant="body2", color="primary")
        rank_select = mui.Autocomplete("Workers", []).prop(muiMargin="dense", size="small")
        self.worker_status_box = WorkersStatusBox([FTStatusBoxState.from_ft_state(state, False) for state in states], self._on_status_box_click)
        header = mui.HBox([
            mui.HBox([
                header_str,
            ]).prop(flex=1),
            start_or_cancel_btn,
            stop_btn,
            kill_btn,
        ])
        # self._remote_box = mui.HBox([])
        self._code_editor = mui.SimpleCodeEditor("echo $HOME", "bash").prop(debounce=300, height="300px", border="1px solid gray")
        self._terminal_box = mui.VBox([
            term,
        ]).prop(flex=1, overflow="auto")
        self._remote_terminal_box = mui.HBox([

        ]).prop(flex=1, overflow="auto")
        self._terminal_panel = mui.MatchCase.binary_selection(True, self._terminal_box, self._remote_terminal_box)
        self.dm = mui.DataModel(ui_state, [
            header,
            rank_select,
            self.worker_status_box,
            self._code_editor,
            self._terminal_panel,
        ])
        master_draft = self.dm.get_draft()
        self._terminal_panel.bind_fields(condition=D.logical_or(master_draft.selected_client_state == None, D.cast_any_draft(master_draft.selected_client_state["rank"], int) == master_rank))
        start_or_cancel_btn.bind_fields(icon=D.where(master_draft.cmd_status == CmdStatus.IDLE, mui.IconType.PlayArrow, mui.IconType.Stop))
        stop_btn.bind_fields(disabled=D.where(master_draft.cmd_status == CmdStatus.IDLE, True, False))
        kill_btn.bind_fields(disabled=D.where(master_draft.cmd_status == CmdStatus.IDLE, True, False))
        self._code_editor.bind_draft_change_uncontrolled(master_draft.cmd)
        # FIXME can't install to worker_status_box
        self.dm.install_draft_change_handler(
            master_draft.client_states, self._handle_client_state_change, 
            handle_child_change=True)
        self.dm.install_draft_change_handler(
            master_draft.selected_client_state, self._handle_selected_box_change)

        rank_select.bind_draft_change(master_draft.selected_client_state)
        rank_select.bind_fields(options=master_draft.client_states)

        super().__init__([
            self.dm,
        ])
        self.prop(flexDirection="column", flex=1)

    async def _on_status_box_click(self, ev: mui.Event):
        rank = ev.get_indexes_checked()[0]
        self.dm.get_draft().selected_client_state = dataclasses.asdict(self.dm.model.client_states[rank])

    async def _handle_client_state_change(self, ev: DraftChangeEvent):
        if ev.new_value is not None:
            states: list[FTState] = ev.new_value
            selected_idx = -1
            selected_state = self.dm.model.selected_client_state
            if selected_state is not None:
                selected_idx = selected_state["rank"]
            ui_states = [FTStatusBoxState.from_ft_state(state, i == selected_idx) for i, state in enumerate(states)]
            await self.put_app_event(self.worker_status_box.update_event(dataList=ui_states))
        else:
            await self.put_app_event(self.worker_status_box.update_event(dataList=[]))

    async def _handle_selected_box_change(self, ev: DraftChangeEvent):
        selected_state_dict = ev.new_value 
        if selected_state_dict is not None:
            rank = selected_state_dict["rank"]
            ip = selected_state_dict["ip"]
            if rank == self._master_rank:
                await self._remote_terminal_box.set_new_layout([])
            else:
                await self._remote_terminal_box.set_new_layout([
                    mui.RemoteBoxGrpc(ip, self._port, TENSORPC_DISTSSH_UI_KEY).prop(flex=1)
                ])
                # await self._remote_terminal_box.set_new_layout([
                #     mui.Markdown("## DEBUG")
                # ])
            async with self.worker_status_box.draft_update(FTStatusBoxState) as dctx:
                with dctx.group(rank):
                    dctx.draft.selected = True 
                with dctx.group(None):
                    dctx.draft.selected = False 
        else:
            await self._remote_terminal_box.set_new_layout([])


class FaultToleranceUIClient(mui.FlexBox):
    def __init__(self, state: FTState, term: terminal.AsyncSSHTerminal):
        title = f"Worker ({state.rank})"
        header_str = mui.Typography(title).prop(variant="body2", color="primary")
        self._terminal_box = mui.VBox([
            term,
        ]).prop(flex=1, overflow="auto")
        self.dm = mui.DataModel(state, [
            header_str,
            self._terminal_box
        ])
        super().__init__([
            self.dm,
        ])
        self.prop(flexDirection="column", flex=1, border="1px solid gray")


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
            cmd_status=CmdStatus.IDLE,
            client_states=[],
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
        self._master_ui = FaultToleranceUIMaster(self._master_rank, self._master_ui_state, self._terminal, prim.get_server_grpc_port(),
            self._master_start_or_cancel, partial(self._master_shutdown_or_kill_cmd, just_kill=False), 
            partial(self._master_shutdown_or_kill_cmd, just_kill=True))

        self._client_ui = FaultToleranceUIClient(state, self._terminal)

        self._master_discovery_fn: Optional[Callable[[], Optional[str]]] = None

        self._client_robjs: dict[int, AsyncRemoteManager] = {}
        self._master_robj: Optional[AsyncRemoteManager] = None

        self._loop_task: Optional[asyncio.Task] = None
        self._cmd_task: Optional[asyncio.Task] = None

        self._disconnect_retry_count = 0

    @property 
    def state(self):
        if self._is_master:
            return self._master_ui.dm.model.client_states[self._master_rank]
        else:
            return self._client_ui.dm.model

    @marker.mark_server_event(event_type=marker.ServiceEventType.Init)
    async def _init(self):
        await self._terminal.connect_with_new_desc(self._conn_desc)
        file_name = Path(self._cfg.workdir) / f"distssh-rank-{self.state.rank}.json"
        if self._cfg.master_discovery_fn is not None:
            self._master_discovery_fn = import_dynamic_func(
                self._cfg.master_discovery_fn, is_func_id=True)
        else:
            if not Path(self._cfg.workdir).exists():
                Path(self._cfg.workdir).mkdir(parents=True, exist_ok=True)
            assert Path(self._cfg.workdir).exists(), f"{self._cfg.workdir} does not exist"
            with file_name.open("w") as f:
                json.dump({
                    "ip": self.state.ip,
                    "uuid": self.state.uuid,
                }, f, indent=4)
        self._loop_task = asyncio.create_task(self._heartbeat_loop())
        if self._is_master:
            workdir = Path(self._cfg.workdir) 
            if not workdir.exists():
                workdir.mkdir(parents=True, exist_ok=True, mode=0o755)
            fs_backend = DraftSimpleFileStoreBackend(workdir)
            self._master_ui.dm.connect_draft_store("_distssh_store", fs_backend)
        set_layout_service = prim.get_service(
            app_serv_names.REMOTE_COMP_SET_LAYOUT_OBJECT)
        if self._is_master:
            await set_layout_service(TENSORPC_DISTSSH_UI_KEY, self._master_ui)
        else:
            await set_layout_service(TENSORPC_DISTSSH_UI_KEY, self._client_ui)

    @marker.mark_server_event(event_type=marker.ServiceEventType.Exit)
    async def _close(self):
        await self._terminal.disconnect()
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

    def _master_check_is_all_ssh_idle_or_err(self):
        for client_state in self._master_ui.dm.model.client_states:
            if client_state.ssh_status != SSHStatus.IDLE and client_state.ssh_status != SSHStatus.ERROR:
                return False 
        return True 

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

    def _term_or_kill_all_ssh_child(self, is_term: bool):
        state = self._terminal.get_current_state()
        if state is None:
            return []
        ssh_pid = state.pid
        ssh_proc = psutil.Process(ssh_pid)
        for child in ssh_proc.children(recursive=True):
            if child.pid != ssh_pid:
                if is_term:
                    child.terminate()
                else:
                    child.kill()
        
    def _num_client_robj_is_valid(self):
        if self._is_master:
            return len(self._client_robjs) == self._cfg.world_size - 1
        else:
            return self._master_robj is not None

    async def _master_start_or_cancel(self):
        if self._master_ui.dm.model.cmd_status == CmdStatus.IDLE:
            await self._master_run_cmd(self._master_ui.dm.model.cmd)
        else:
            await self._master_cancel_cmd()

    async def _master_run_cmd(self, cmd: str):
        assert self._cmd_task is None, "master can only run one command at a time, shutdown it first." 
        cmd = cmd.strip()
        if self.state.status != FTStatus.OK or not self._num_client_robj_is_valid():
            raise RuntimeError("master is not in OK state")
        for client_state in self._master_ui.dm.model.client_states:
            if client_state.ssh_status != SSHStatus.IDLE and client_state.ssh_status != SSHStatus.ERROR:
                raise RuntimeError(f"worker {client_state.rank} is not in IDLE state")  
        async with self._master_ui.dm.draft_update() as draft:
            draft.cmd_status = CmdStatus.RUNNING
        for robj in self._client_robjs.values():
            await robj.remote_call(get_service_key_by_type(FaultToleranceSSHServer, "client_run_cmd"), cmd)
        self._cmd_task = asyncio.create_task(self._cmd_waiter(cmd))

    async def cancel_cmd(self):
        if self._cmd_task is not None:
            await self._terminal.send_ctrl_c()

    async def _master_cancel_cmd(self):
        if self._master_ui.dm.model.cmd_status != CmdStatus.IDLE:
            await self.cancel_cmd()
            for robj in self._client_robjs.values():
                await robj.remote_call(get_service_key_by_type(FaultToleranceSSHServer, "cancel_cmd"))

    async def _master_shutdown_or_kill_cmd(self, just_kill: bool = False):
        if self._cmd_task is not None:
            await self.shutdown_or_kill_cmd(just_kill)
            for robj in self._client_robjs.values():
                await robj.remote_call(get_service_key_by_type(FaultToleranceSSHServer, "shutdown_or_kill_cmd"), just_kill)

    async def shutdown_or_kill_cmd(self, just_kill: bool = False):
        if self._cmd_task is not None:
            await self._cmd_shutdown_sequence(just_kill)

    async def client_run_cmd(self, cmd: str):
        assert self._cmd_task is None, "master can only run one command at a time" 
        if self.state.status != FTStatus.OK:
            raise RuntimeError("worker is not in OK state")
        self._cmd_task = asyncio.create_task(self._cmd_waiter(cmd))


    async def client_cancel_cmd(self):
        if self._cmd_task is not None:
            await self._terminal.send_ctrl_c()


    async def _client_set_worker_state(self):
        assert self._master_robj is not None 
        try:
            await self._master_robj.remote_call(get_service_key_by_type(FaultToleranceSSHServer, "set_worker_state"), self.state)
        except:
            traceback.print_exc()
            self._master_robj = None 
            LOGGER.warning(f"Master disconnected")
            async with self._client_ui.dm.draft_update() as draft:
                draft.status = FTStatus.MASTER_DISCONNECTED


    async def _cmd_shutdown_sequence(self, try_cancel_and_term: bool = True):
        assert self._cmd_task is not None 
        if try_cancel_and_term:
            # 1. send ctrl c
            for i in range(self._cfg.cmd_ctrl_c_retry):
                await self._terminal.send_ctrl_c()
                try:
                    with async_timeout.timeout(self._cfg.cmd_shutdown_timeout):
                        await self._cmd_task
                        return
                except asyncio.TimeoutError:
                    LOGGER.warning(f"ctrl-c timeout, retry {i}")
                    if i == self._cfg.cmd_ctrl_c_retry - 1:
                        raise RuntimeError("ctrl-c timeout")
            # 2. send sigterm to all subprocess of ssh process
            self._term_or_kill_all_ssh_child(True)
            try:
                with async_timeout.timeout(self._cfg.cmd_shutdown_timeout):
                    await self._cmd_task
                    return
            except asyncio.TimeoutError:
                LOGGER.warning(f"sigterm timeout, perform kill.")
        # 3. send sigkill to all subprocess of ssh process
        self._term_or_kill_all_ssh_child(False)
        await self._cmd_task


    async def _cmd_waiter(self, cmd: str):
        shutdown_ev = prim.get_async_shutdown_event()
        shutdown_ev_task = asyncio.create_task(shutdown_ev.wait())
        LOGGER.warning("Launch command: %s", cmd)
        run_cmd_task = asyncio.create_task(self._terminal.ssh_command_rpc(cmd))
        if self._is_master:
            async with self._master_ui.dm.draft_update() as draft:
                draft.client_states[self._master_rank].cur_cmd = cmd 
                draft.client_states[self._master_rank].ssh_status = SSHStatus.RUNNING
        else:
            async with self._client_ui.dm.draft_update() as draft:
                draft.cur_cmd = cmd 
                draft.ssh_status = SSHStatus.RUNNING
        if not self._is_master:
            await self._client_set_worker_state()
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
        ssh_status = SSHStatus.IDLE if res.return_code == 0 else SSHStatus.ERROR

        if self._is_master:
            async with self._master_ui.dm.draft_update() as draft:
                draft.client_states[self._master_rank].cur_cmd = None
                draft.client_states[self._master_rank].ssh_status = ssh_status
            await self._master_sync_cmd_status()
        else:
            async with self._client_ui.dm.draft_update() as draft:
                draft.cur_cmd = None
                draft.ssh_status = ssh_status
            if self._master_robj is not None:
                await self._client_set_worker_state()
        self._cmd_task = None


    async def _master_sync_cmd_status(self):
        is_all_ssh_finish = self._master_check_is_all_ssh_idle_or_err()
        if is_all_ssh_finish:
            async with self._master_ui.dm.draft_update() as draft_master:
                draft_master.cmd_status = CmdStatus.IDLE
        else:
            async with self._master_ui.dm.draft_update() as draft_master:
                draft_master.cmd_status = CmdStatus.RUNNING


    async def set_worker_state(self, state: FTState):
        assert state.rank != self._master_rank, "master rank should not be set"
        if self.state.is_master:
            async with self._master_ui.dm.draft_update() as draft_master:
                draft_master.client_states[state.rank] = state
            if state.rank in self._client_robjs:
                prev_robj = self._client_robjs.pop(state.rank)
                try:
                    await prev_robj.close()
                except:
                    traceback.print_exc()
            robj = AsyncRemoteManager(f"{state.ip}:{prim.get_server_grpc_port()}")
            self._client_robjs[state.rank] = robj
            await self._master_sync_cmd_status()

    async def _query_master_robj(self):
        folder_p = Path(self._cfg.workdir)
        port = prim.get_server_grpc_port()
        if self._master_discovery_fn is not None:
            master_ip = self._master_discovery_fn()
            if master_ip is not None:
                robj = AsyncRemoteManager(f"{master_ip}:{port}")
                try:
                    await robj.health_check(timeout=self._cfg.disconnect_rpc_check_timeout, wait_for_ready=True)
                except grpc.aio.AioRpcError as e:
                    if e.code() == grpc.StatusCode.UNAVAILABLE:
                        robj = None
                    elif e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                        robj = None
                    else:
                        traceback.print_exc()
                        robj = None
                return robj
        else:
            master_files = list(folder_p.glob(f"distssh-rank-{self._master_rank}.json"))
            if len(master_files) > 0:
                with master_files[0].open("r") as f:
                    master_info = json.load(f)
                    master_ip = master_info["ip"]
                    uuid = master_info["uuid"]
                    robj = AsyncRemoteManager(f"{master_ip}:{port}")
                    try:
                        await robj.health_check(timeout=self._cfg.disconnect_rpc_check_timeout, wait_for_ready=True)
                    except grpc.aio.AioRpcError as e:
                        if e.code() == grpc.StatusCode.UNAVAILABLE:
                            robj = None
                        elif e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                            robj = None
                        else:
                            traceback.print_exc()
                            robj = None
                    return robj

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
                        tasks.append(robj.health_check(timeout=self._cfg.disconnect_rpc_check_timeout))
                        ranks.append(rank)
                    res = asyncio.gather(*tasks, return_exceptions=True)
                    has_disconnect = False
                    async with self._master_ui.dm.draft_update() as draft:
                        for rank, r in zip(ranks, res):
                            if isinstance(r, BaseException):
                                draft.client_states[rank].status = FTStatus.WORKER_DISCONNECTED
                                poped_robj = self._client_robjs.pop(rank)
                                try:
                                    await poped_robj.close()
                                except:
                                    traceback.print_exc()
                                LOGGER.warning(f"worker {rank} disconnected")
                                has_disconnect = True
                        if has_disconnect:
                            draft.client_states[self._master_rank].status = FTStatus.WORKER_DISCONNECTED
                else:
                    if len(self._client_robjs) != self._cfg.world_size - 1:
                        # get current disconnected worker rank
                        disconnected_ranks = []
                        for client_st in self._master_ui.dm.model.client_states:
                            if not client_st.is_master and client_st.rank not in self._client_robjs:
                                disconnected_ranks.append(client_st.rank)
                        self._disconnect_retry_count += 1
                        LOGGER.warning("master wait for all worker retry: %d, disconnected ranks: %s", self._disconnect_retry_count, str(disconnected_ranks))
                        if self._disconnect_retry_count > self._cfg.disconnect_total_retry:
                            LOGGER.warning("master wait for all worker timeout, exit.")
                            shutdown_ev.set()
                            break
                    else:
                        async with self._master_ui.dm.draft_update() as draft:
                            draft.client_states[self._master_rank].status = FTStatus.OK
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
                LOGGER.info("Worker Heartbeat|status: %s.", self.state.status.name)
                if self.state.status == FTStatus.OK:
                    if self._master_robj is None:
                        robj = await self._query_master_robj()
                        if robj is None:
                            LOGGER.warning(f"Master disconnected")
                            async with self._client_ui.dm.draft_update() as draft:
                                draft.status = FTStatus.MASTER_DISCONNECTED
                            continue 
                        self._master_robj = robj
                    else:
                        robj = self._master_robj
                    await self._client_set_worker_state()
                else:
                    robj = await self._query_master_robj()
                    if robj is None:
                        self._disconnect_retry_count += 1
                        LOGGER.warning("worker wait for master retry: %d", self._disconnect_retry_count)
                        if self._disconnect_retry_count > self._cfg.disconnect_total_retry:
                            LOGGER.warning("worker wait for master timeout, exit.")
                            prim.get_async_shutdown_event().set()
                            break
                    else:
                        LOGGER.warning(f"Master {robj.url} connected")

                        self._master_robj = robj
                        async with self._client_ui.dm.draft_update() as draft:
                            self._disconnect_retry_count = 0
                            draft.status = FTStatus.OK

        print("heartbeat loop exit")