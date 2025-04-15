import asyncio
import enum
from functools import partial
import json
from pathlib import Path
import time
import traceback
from typing import Any, Awaitable, Callable, Optional, Union

import async_timeout
import grpc
import rich
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
from tensorpc.apps.distssh.constants import (TENSORPC_DISTSSH_UI_KEY, TENSORPC_ENV_DISTSSH_URL_WITH_PORT)
from rich.syntax import Syntax
from ..typedefs import FTState, FTSSHServerArgs, FTStatus, SSHStatus, CmdStatus, MasterUIState
from ..components.sshui import FaultToleranceUIMaster, FaultToleranceUIClient

LOGGER = rich_logging.get_logger("distssh")

class SimpleLogEventType(enum.Enum):
    LINE = "L"
    INIT_METADATA = "M"

class SSHEventLogger:

    def __init__(self, outdir: Path, open_mode: str = "w"):
        self.outdir = outdir
        self.jsonl_writer = None
        self._open_mode = open_mode

    def open(self):
        if self.jsonl_writer is None:
            self.jsonl_writer = open(self.outdir, self._open_mode)

    def log(self, metrics: dict[str, Any], compact: bool = False):
        if compact:
            print(json.dumps(metrics, separators=(',', ':')),
                  file=self.jsonl_writer,
                  flush=True)
        else:
            print(json.dumps(metrics), file=self.jsonl_writer, flush=True)

    def close(self):
        if self.jsonl_writer is not None:
            self.jsonl_writer.close()
            self.jsonl_writer = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()


class FaultToleranceSSHServer:
    def __init__(self,
                 config_dict: dict,
                 default_url: str = "localhost:22") -> None:
        cfg = FTSSHServerArgs(**config_dict)
        self._cfg = cfg
        self._conn_desc = SSHConnDesc(default_url, cfg.username, cfg.password)
        self._terminal = terminal.AsyncSSHTerminal().prop(disableStdin=True)
        self._master_rank = 0
        ip = get_primary_ip()
        state = FTState(
            label=f"{cfg.rank} ({ip})",
            rank=cfg.rank,
            uuid=uuid.uuid4().hex,
            ip=ip,
            port=0,
            is_master=cfg.rank == self._master_rank,
            master_uuid="",
        )
        self._is_master = cfg.rank == self._master_rank
        if self._is_master:
            state.master_uuid = state.uuid
        master_ui_state = MasterUIState(
            cmd_status=CmdStatus.IDLE,
            client_states=[],
            cmd_history=[],
        )
        for j in range(cfg.world_size):
            if j == self._master_rank:
                master_ui_state.client_states.append(state)
            else:
                init_client_state = FTState(
                    label=f"{j} ({ip})",
                    rank=j,
                    uuid="",
                    ip="",
                    port=0,
                    is_master=False,
                    status=FTStatus.WORKER_DISCONNECTED,
                )
                master_ui_state.client_states.append(init_client_state)
        self._master_ui = FaultToleranceUIMaster(self._master_rank, master_ui_state, self._terminal, prim.get_server_grpc_port(),
            self._master_start_or_cancel, partial(self._master_shutdown_or_kill_cmd, just_kill=False), 
            partial(self._master_shutdown_or_kill_cmd, just_kill=True))

        self._client_ui = FaultToleranceUIClient(state, self._terminal)

        self._master_discovery_fn: Optional[Callable[[], Optional[tuple[str, str]]]] = None

        self._client_robjs: dict[int, AsyncRemoteManager] = {}
        self._master_robj: Optional[AsyncRemoteManager] = None

        self._loop_task: Optional[asyncio.Task] = None
        self._cmd_task: Optional[asyncio.Task] = None

        self._disconnect_retry_count = 0

        self._cur_logger: Optional[SSHEventLogger] = None
        self._cmd_start_ts: int = 0

        self._master_lock = asyncio.Lock()
        self._fast_wakeup_event = asyncio.Event()

    @property 
    def state(self):
        if self._is_master:
            return self._master_ui.dm.model.client_states[self._master_rank]
        else:
            return self._client_ui.dm.model

    @marker.mark_server_event(event_type=marker.ServiceEventType.Init)
    async def _init(self):
        await self._terminal.connect_with_new_desc(self._conn_desc, init_cmds=[
            f"export {TENSORPC_ENV_DISTSSH_URL_WITH_PORT}={self.state.ip}:{prim.get_server_grpc_port()}\n",
        ])
        term_state = self._terminal.get_current_state()
        assert term_state is not None 
        self._master_ui._master_panel.set_parent_pid(term_state.pid)
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
                    "port": prim.get_server_grpc_port(),
                    "uuid": self.state.uuid,
                }, f, indent=4)
        if self._is_master:
            self._master_ui.dm.model.client_states[self._master_rank].port = prim.get_server_grpc_port()
        else:
            self._client_ui.dm.model.port = prim.get_server_grpc_port()
        self._loop_task = asyncio.create_task(self._heartbeat_loop())
        self._master_state_backup_path: Optional[Path] = None 
        if self._is_master:
            workdir = Path(self._cfg.workdir) 
            if not workdir.exists():
                workdir.mkdir(parents=True, exist_ok=True, mode=0o755)
            fs_backend = DraftSimpleFileStoreBackend(workdir)
            self._master_ui.dm.connect_draft_store(f"_distssh_store_{self._cfg.world_size}", fs_backend)
            self._master_state_backup_path = fs_backend._get_abs_path(f"_distssh_store_backup_{self._cfg.world_size}")
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

    def set_logdir(self, logdir: str):
        # real logger will be created when a command started, and reset
        # when the command finished.
        self._cfg.logdir = logdir

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

    async def _master_call_all_client(self, key: str, *args, rpc_timeout: Optional[int] = None, rpc_is_health_check: bool = False, **kwargs):
        async with self._master_lock:
            is_all_worker_conn = self._num_client_robj_is_valid()
            tasks = []
            ranks: list[int] = []
            for rank, robj in self._client_robjs.items():
                if rpc_is_health_check:
                    tasks.append(robj.health_check(timeout=rpc_timeout))
                else:
                    tasks.append(robj.remote_call(key, *args, rpc_timeout=rpc_timeout, **kwargs))
                ranks.append(rank)
            res = await asyncio.gather(*tasks, return_exceptions=True)
            has_disconnect = False
            async with self._master_ui.dm.draft_update() as draft:
                res_dict: dict[int, Any] = {}
                for rank, r in zip(ranks, res):
                    if isinstance(r, BaseException):
                        draft.client_states[rank].status = FTStatus.WORKER_DISCONNECTED
                        poped_robj = self._client_robjs.pop(rank)
                        try:
                            await poped_robj.close()
                        except:
                            traceback.print_exc()
                        LOGGER.warning(f"worker {rank}({poped_robj.url}) disconnected")
                        has_disconnect = True
                    else:
                        res_dict[rank] = r
                if has_disconnect or not is_all_worker_conn:
                    draft.client_states[self._master_rank].status = FTStatus.WORKER_DISCONNECTED
                    return None 
                else:
                    return res_dict

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
        res = await self._master_call_all_client(get_service_key_by_type(FaultToleranceSSHServer, "client_run_cmd"), cmd)
        if res is not None:
            self._cmd_task = asyncio.create_task(self._cmd_waiter(cmd))
        return res

    async def cancel_cmd(self):
        if self._cmd_task is not None:
            await self._terminal.send_ctrl_c()

    async def _master_cancel_cmd(self):
        if self._master_ui.dm.model.cmd_status != CmdStatus.IDLE:
            await self.cancel_cmd()
            await self._master_call_all_client(get_service_key_by_type(FaultToleranceSSHServer, "cancel_cmd"))

    async def _master_shutdown_or_kill_cmd(self, just_kill: bool = False):
        res = await self._master_call_all_client(get_service_key_by_type(FaultToleranceSSHServer, "shutdown_or_kill_cmd"), just_kill)
        await self.shutdown_or_kill_cmd(just_kill)
        return res 

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


    async def _client_set_worker_state(self, wait_task: Optional[asyncio.Task] = None, handle_restart: bool = True):
        assert self._master_robj is not None 
        try:
            master_ftstate = await self._master_robj.remote_call(get_service_key_by_type(FaultToleranceSSHServer, "set_worker_state"), self.state)
            if handle_restart:
                if self.state.master_uuid != "" and self.state.master_uuid != master_ftstate.master_uuid:
                    # master restart. if cmd is running, shutdown it.
                    if self._cmd_task is not None:
                        await self._cmd_shutdown_sequence(wait_task=wait_task)
            self.state.master_uuid = master_ftstate.master_uuid
        except:
            traceback.print_exc()
            self._master_robj = None 
            LOGGER.warning(f"Master disconnected")
            async with self._client_ui.dm.draft_update() as draft:
                draft.status = FTStatus.MASTER_DISCONNECTED


    async def _cmd_shutdown_sequence(self, try_cancel_and_term: bool = True, wait_task: Optional[asyncio.Task] = None):
        if wait_task is None:
            assert self._cmd_task is not None 
            wait_task = self._cmd_task
        if try_cancel_and_term:
            # 1. send ctrl c
            for i in range(self._cfg.cmd_ctrl_c_retry):
                await self._terminal.send_ctrl_c()
                try:
                    async with async_timeout.timeout(self._cfg.cmd_shutdown_timeout):
                        await wait_task
                        return
                except asyncio.TimeoutError:
                    LOGGER.warning(f"ctrl-c timeout, retry {i}")
                    if i == self._cfg.cmd_ctrl_c_retry - 1:
                        raise RuntimeError("ctrl-c timeout")
            # 2. send sigterm to all subprocess of ssh process
            self._term_or_kill_all_ssh_child(True)
            try:
                async with async_timeout.timeout(self._cfg.cmd_shutdown_timeout):
                    await wait_task
                    return
            except asyncio.TimeoutError:
                LOGGER.warning(f"sigterm timeout, perform kill.")
        # 3. send sigkill to all subprocess of ssh process
        self._term_or_kill_all_ssh_child(False)
        await wait_task

    def _line_event_cb(self, ev: terminal.TerminalLineEvent):
        if self._cur_logger is not None:
            relative_ts = (ev.ts - self._cmd_start_ts) // 1000
            self._cur_logger.log({
                "ts": relative_ts,
                "t": SimpleLogEventType.LINE.value,
                "d": ev.d.decode("utf-8")
            })

    async def _cmd_waiter(self, cmd: str):
        LOGGER.warning("Launch command:")
        rich.print(cmd)

        ssh_state = self._terminal.get_current_state()
        assert ssh_state is not None 
        shell_info = ssh_state.shell_info
        assert shell_info is not None 
        if shell_info.type == "zsh":
            shell_file_path = Path(self._cfg.workdir) / f"_distssh-rank-cmd-{self.state.rank}.zsh"
            with shell_file_path.open("w") as f:
                f.write(cmd)
            cmd = f"zsh -i {shell_file_path.absolute()}"
        elif shell_info.type == "bash":
            shell_file_path = Path(self._cfg.workdir) / f"_distssh-rank-cmd-{self.state.rank}.sh"
            with shell_file_path.open("w") as f:
                f.write(cmd)
            cmd = f"bash -i {shell_file_path.absolute()}"
        else:
            raise RuntimeError(f"Unsupported shell type: {shell_info.type}")
        shutdown_ev = prim.get_async_shutdown_event()
        shutdown_ev_task = asyncio.create_task(shutdown_ev.wait())

        try:
            if self._cfg.logdir != "":
                logpath = Path(self._cfg.logdir) / f"cmd-log-{self.state.rank}.jsonl"
                if not logpath.parent.exists():
                    logpath.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                logger = SSHEventLogger(logpath, open_mode="a")
                logger.open()
                self._cur_logger = logger
            self._cmd_start_ts = time.time_ns()
            if self._cur_logger is not None:
                self._cur_logger.log({
                    "ts": self._cmd_start_ts,
                    "t": SimpleLogEventType.INIT_METADATA.value,
                    "d": {
                        "start_ts": self._cmd_start_ts,
                        "cmd": cmd,
                        "rank": self.state.rank,
                    }
                })
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
                await self._client_set_worker_state(wait_task=run_cmd_task)
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
                    await self._client_set_worker_state(handle_restart=False)
            self._cmd_task = None
        except:
            LOGGER.error("cmd waiter error.", exc_info=True)
            raise
        finally:
            if self._cur_logger is not None:
                self._cur_logger.close()
                self._cur_logger = None

    async def _master_sync_cmd_status(self):
        if self._master_ui.dm.model.cmd_status != CmdStatus.DURING_RESTART:
            is_all_ssh_finish = self._master_check_is_all_ssh_idle_or_err()
            if is_all_ssh_finish:
                async with self._master_ui.dm.draft_update() as draft_master:
                    draft_master.cmd_status = CmdStatus.IDLE
            else:
                async with self._master_ui.dm.draft_update() as draft_master:
                    draft_master.cmd_status = CmdStatus.RUNNING

    async def _master_start_cmd_restart_sequence(self):
        LOGGER.warning("Master try to restart cmd.")
        async with self._master_ui.dm.draft_update() as draft_master:
            draft_master.cmd_status = CmdStatus.DURING_RESTART
        self._fast_wakeup_event.set()

    async def set_worker_state(self, state: FTState) -> FTState:

        assert state.rank != self._master_rank, "master rank should not be set"
        assert self.state.is_master, "only master can set worker state"
        prev_client_state = self._master_ui.dm.model.client_states[state.rank]
        client_is_restart = (prev_client_state.uuid != state.uuid and prev_client_state.uuid != "")
        master_is_restart = (self.state.uuid != state.master_uuid and state.master_uuid != "")
        if client_is_restart or master_is_restart:
            await self._master_start_cmd_restart_sequence()
        async with self._master_ui.dm.draft_update() as draft_master:
            draft_master.client_states[state.rank] = state
        if state.rank in self._client_robjs:
            prev_robj = self._client_robjs[state.rank]
            if client_is_restart:
                try:
                    await prev_robj.close()
                except:
                    traceback.print_exc()
                self._client_robjs.pop(state.rank)
                LOGGER.warning(f"Worker {state.rank}({state.ip}:{state.port}) reconnected.")
                robj = AsyncRemoteManager(f"{state.ip}:{state.port}")
                self._client_robjs[state.rank] = robj
        else:
            LOGGER.warning(f"Worker {state.rank}({state.ip}:{state.port}) connected.")
            robj = AsyncRemoteManager(f"{state.ip}:{state.port}")
            self._client_robjs[state.rank] = robj
        await self._master_sync_cmd_status()
        return self.state

    async def _query_master_robj(self):
        folder_p = Path(self._cfg.workdir)
        port = prim.get_server_grpc_port()
        if self._master_discovery_fn is not None:
            master_ip_port = self._master_discovery_fn()
            if master_ip_port is not None:
                master_ip, port = master_ip_port
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
                    if "port" in master_info:
                        port = master_info["port"]
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

    async def _master_correct_status_if_restart(self):
        master_state = self._master_ui.dm.model
        if self._cmd_task is None and master_state.cmd_status == CmdStatus.RUNNING:
            async with self._master_ui.dm.draft_update() as draft:
                draft.cmd_status = CmdStatus.IDLE

    async def _heartbeat_loop(self):
        shutdown_ev = prim.get_async_shutdown_event()
        shutdown_ev_task = asyncio.create_task(shutdown_ev.wait(), name="heartbeat_shutdown")
        await self._master_correct_status_if_restart()
        try:
            if self.state.is_master:
                fast_wakeup_task = asyncio.create_task(
                    self._fast_wakeup_event.wait(), name="heartbeat_fast_wakeup")
                while True:
                    sleep_task = asyncio.create_task(
                        asyncio.sleep(self._cfg.heartbeat_interval), name="heartbeat_sleep")
                    done, _ = await asyncio.wait(
                        [shutdown_ev_task, sleep_task, fast_wakeup_task],
                        return_when=asyncio.FIRST_COMPLETED)
                    if shutdown_ev_task in done:
                        await cancel_task(sleep_task)
                        await cancel_task(fast_wakeup_task)
                        break 
                    if fast_wakeup_task in done:
                        self._fast_wakeup_event.clear()
                        fast_wakeup_task = asyncio.create_task(
                            self._fast_wakeup_event.wait(), name="heartbeat_fast_wakeup")
                    # LOGGER.warning("Master Heartbeat|status: %s.", self.state.status.name)
                    if self.state.status == FTStatus.OK:
                        res = await self._master_call_all_client("", rpc_timeout=self._cfg.disconnect_rpc_check_timeout, rpc_is_health_check=True)
                        if res is not None:
                            if self._master_ui.dm.model.cmd_status == CmdStatus.DURING_RESTART:
                                res = await self._master_shutdown_or_kill_cmd()
                                if res is not None:
                                    await self._master_run_cmd(self._master_ui.dm.model.cmd)
                    else:
                        if len(self._client_robjs) != self._cfg.world_size - 1:
                            # get current disconnected worker rank
                            disconnected_ranks = []
                            for client_st in self._master_ui.dm.model.client_states:
                                if not client_st.is_master and client_st.rank not in self._client_robjs:
                                    disconnected_ranks.append(client_st.rank)
                            self._disconnect_retry_count += 1
                            LOGGER.warning("master wait for all worker retry: %d/%d, disconnected ranks: %s", 
                                self._disconnect_retry_count, self._cfg.disconnect_total_retry, str(disconnected_ranks))
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
                        asyncio.sleep(self._cfg.heartbeat_interval), name="heartbeat_sleep")
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
                            else:
                                LOGGER.warning(f"Master {robj.url} connected")
                                self._master_robj = robj
                        else:
                            robj = self._master_robj
                        await self._client_set_worker_state(handle_restart=True)
                    else:
                        robj = await self._query_master_robj()
                        if robj is None:
                            self._disconnect_retry_count += 1
                            LOGGER.warning("worker wait for master retry: %d/%d", self._disconnect_retry_count,
                                self._cfg.disconnect_total_retry)
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
        except:
            LOGGER.warning("heartbeat loop exception", exc_info=True)
            raise 
        finally:
            print("heartbeat loop exit")

    def is_user_control_enabled(self):
        """pth control point will access this value and 
        enter breakpoint when set.
        """
        return self.state.is_user_control_enabled