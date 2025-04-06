import traceback
from typing import Optional

import grpc
from tensorpc import prim
from tensorpc.apps.dbg.constants import TracerConfig
from tensorpc.core import BuiltinServiceProcType, marker
import psutil
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.core.asyncclient import AsyncRemoteManager
from tensorpc.core.bgserver import BackgroundProcMeta
from tensorpc.dock.vscode.coretypes import VscodeBreakpoint
from tensorpc.utils.gpuusage import get_nvidia_gpu_measures 
import asyncio
from tensorpc.apps.dbg.serv_names import serv_names as dbg_serv_names
from tensorpc.utils.rich_logging import get_logger
from tensorpc.utils.proctitle import list_all_tensorpc_server_in_machine, set_tensorpc_server_process_title

DBG_LOGGER = get_logger("tensorpc.dbg")

@dataclasses.dataclass
class RelayMonitorConfig:
    # when enabled, monitor will scan tensorpc servers
    # based on process name (title).
    server_scan_interval: float = 5

@dataclasses.dataclass
class ChildRemoteCompInfo:
    proc_meta: BackgroundProcMeta
    pid: int 
    port: int 
    # clients that comes from client request
    manual_proc: bool
    parent_pid: Optional[int] = None
    robj: Optional[AsyncRemoteManager] = None

    def get_info_no_robj(self):
        return dataclasses.replace(self, robj=None)


def is_nested_child(child_pid: int, parent_pid: int) -> bool:
    """
    Check if the process with child_pid is a descendant (nested child) 
    of the process with parent_pid.
    """
    try:
        child_proc = psutil.Process(child_pid)
    except psutil.NoSuchProcess:
        return False

    # Get all parent processes.
    for ancestor in child_proc.parents():
        if ancestor.pid == parent_pid:
            return True

    return False

class RelayMonitor:
    def __init__(self, observed_pid: int, config_dict: dict):
        self._observed_pid = observed_pid
        self._cfg = RelayMonitorConfig(**config_dict)

        self._pid_servid_to_info: dict[tuple[int, str], ChildRemoteCompInfo] = {}
        self._shutdown_ev: asyncio.Event = asyncio.Event()

        self._scan_task: Optional[asyncio.Task] = None

        self._vscode_bkpts: dict[str, tuple[list[VscodeBreakpoint], int]] = {}

    @marker.mark_server_event(event_type=marker.ServiceEventType.Init)
    def _server_init(self):
        port = prim.get_server_grpc_port()

        set_tensorpc_server_process_title(
            BuiltinServiceProcType.RELAY_MONITOR, str(port))

        self._shutdown_ev.clear()
        self._scan_task = asyncio.create_task(self._scan_loop())

    @marker.mark_server_event(event_type=marker.ServiceEventType.Exit)
    async def on_exit(self):
        self._shutdown_ev.set()
        if self._scan_task:
            await self._scan_task
            self._scan_task = None

    async def _scan_loop(self):
        shutdown_task = asyncio.create_task(self._shutdown_ev.wait())

        wait_task = asyncio.create_task(
            asyncio.sleep(self._cfg.server_scan_interval)
        )

        while True:
            done, pending = await asyncio.wait(
                [shutdown_task, wait_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if shutdown_task in done:
                break
            if wait_task in done:
                proc_metas = list_all_tensorpc_server_in_machine(BuiltinServiceProcType.REMOTE_COMP)
                new_metas: dict[tuple[int, str], BackgroundProcMeta] = {}
                for proc_meta in proc_metas:
                    bg_meta = BackgroundProcMeta.from_trpc_proc_meta(proc_meta)

                    # ignore the observed process (usually SSH process)
                    if proc_meta.pid == self._observed_pid:
                        continue
                    # we only check process that is child of the observed process
                    if not is_nested_child(proc_meta.pid, self._observed_pid):
                        continue

                    key = (proc_meta.pid, bg_meta.server_uuid)
                    new_metas[key] = bg_meta

                # update cache, note that the robj is created lazily, not here.
                # 1. remove process that is not in the list
                for key in list(self._pid_servid_to_info.keys()):
                    if key not in new_metas:
                        val = self._pid_servid_to_info[key]
                        if not val.manual_proc:
                            old_info = self._pid_servid_to_info.pop(key)
                            if old_info.robj is not None:
                                try:
                                    await old_info.robj.close()
                                except:
                                    traceback.print_exc()
                        else:
                            # check pid exists
                            pid_exists = psutil.pid_exists(key[0])
                            if not pid_exists:
                                # remove the process
                                info = self._pid_servid_to_info.pop(key)
                                if info.robj is not None:
                                    try:
                                        await info.robj.close()
                                    except:
                                        traceback.print_exc()
                # 2. add new process
                for key, bg_meta in new_metas.items():
                    if key not in self._pid_servid_to_info:
                        self._pid_servid_to_info[key] = ChildRemoteCompInfo(
                            proc_meta=bg_meta,
                            manual_proc=False,
                            pid=bg_meta.pid,
                            port=bg_meta.port,
                            parent_pid=self._observed_pid,
                        )

                wait_task = asyncio.create_task(
                    asyncio.sleep(self._cfg.server_scan_interval)
                )

    async def get_current_infos(self):
        if self._scan_task is None:
            self._scan_task = asyncio.create_task(self._scan_loop())
        return {k: v.get_info_no_robj() for k, v in self._pid_servid_to_info.items()}

    def _cached_get_info_robj(self, key: tuple[int, str]):
        assert key in self._pid_servid_to_info
        info = self._pid_servid_to_info[key]
        if info.robj is None:
            info.robj = AsyncRemoteManager(
                url=f"localhost:{info.port}",
            )
        return info.robj

    def get_vscode_breakpoints(self):
        # clients call this to get the breakpoints
        return self._vscode_bkpts

    def set_vscode_breakpoints(self, bkpts: dict[str, tuple[list[VscodeBreakpoint], int]]):
        self._vscode_bkpts = bkpts

    async def leave_breakpoint(self, key: tuple[int, str], trace_cfg: Optional[TracerConfig]):
        assert key in self._pid_servid_to_info
        robj = self._cached_get_info_robj(key)
        return await robj.remote_call(dbg_serv_names.DBG_LEAVE_BREAKPOINT,
                    trace_cfg,
                    rpc_timeout=1)

    async def run_rpc_on_processes(self,
                                process_keys: list[tuple[int, str]],
                                service_key: str,
                                *args,
                                rpc_timeout: int = 1,
                                rpc_is_chunk_call: bool = False):
        all_tasks = []
        for key in process_keys:
            if key not in self._pid_servid_to_info:
                DBG_LOGGER.warning(
                    f"Process {key} not found in the cache, skipping RPC call.")
                continue 
            all_tasks.append(
                self.run_rpc_on_process(key,
                                      service_key,
                                      *args,
                                      rpc_timeout=rpc_timeout,
                                      rpc_is_chunk_call=rpc_is_chunk_call))
        return await asyncio.gather(*all_tasks)

    async def run_rpc_on_process(self,
                               key: tuple[int, str],
                               service_key: str,
                               *args,
                               rpc_timeout: int = 1,
                               rpc_is_chunk_call: bool = False):
        robj = self._cached_get_info_robj(key)
        if rpc_is_chunk_call:
            rpc_func = robj.chunked_remote_call
        else:
            rpc_func = robj.remote_call
        try:
            return rpc_func(
                    service_key,
                    *args,
                    rpc_timeout=rpc_timeout)
        except TimeoutError:
            traceback.print_exc()
            return None
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                return None
            else:
                traceback.print_exc()
                return None
