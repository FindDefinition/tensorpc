import io
import traceback
from typing import Optional

import grpc
from tensorpc import prim
from tensorpc.apps.dbg.constants import DebugInfo, DebugMetric, RelayMonitorChildInfo, TracerConfig
from tensorpc.core import BuiltinServiceProcType, marker
import psutil
from tensorpc.core import dataclass_dispatch as dataclasses
import dataclasses as dataclasses_plain
from tensorpc.core.asyncclient import AsyncRemoteManager
from tensorpc.core.bgserver import BackgroundProcMeta
from tensorpc.dock.vscode.coretypes import VscodeBreakpoint
from tensorpc.utils.gpuusage import get_nvidia_gpu_measures 
import asyncio
from tensorpc.apps.dbg.serv_names import serv_names as dbg_serv_names
from tensorpc.utils.rich_logging import get_logger
from tensorpc.utils.proctitle import list_all_tensorpc_server_in_machine, set_tensorpc_server_process_title
from tensorpc.apps.dbg.constants import (DebugServerProcessInfo, TENSORPC_DBG_FRAME_INSPECTOR_KEY,
                                    TENSORPC_DBG_TRACE_VIEW_KEY)

DBG_LOGGER = get_logger("tensorpc.dbg")

@dataclasses.dataclass
class RelayMonitorConfig:
    # when enabled, monitor will scan tensorpc servers
    # based on process name (title).
    server_scan_interval: float = 5

@dataclasses_plain.dataclass
class ChildRemoteCompInfo:
    proc_type: BuiltinServiceProcType
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
                proc_metas = list_all_tensorpc_server_in_machine({BuiltinServiceProcType.REMOTE_COMP, BuiltinServiceProcType.SERVER_WITH_DEBUG})
                new_metas: dict[tuple[int, str], tuple[BackgroundProcMeta, BuiltinServiceProcType]] = {}
                for proc_meta in proc_metas:
                    if proc_meta.type == BuiltinServiceProcType.SERVER_WITH_DEBUG:
                        server_id = proc_meta.args[0]
                        bg_meta = BackgroundProcMeta(proc_meta.pid, proc_meta.name, server_id, int(proc_meta.args[1]), server_id)
                    else:
                        bg_meta = BackgroundProcMeta.from_trpc_proc_meta(proc_meta)
                        # ignore the observed process (usually SSH process)
                        if proc_meta.pid == self._observed_pid:
                            continue
                        # we only check process that is child of the observed process
                        if not is_nested_child(proc_meta.pid, self._observed_pid):
                            continue
                    key = (proc_meta.pid, bg_meta.server_uuid)
                    new_metas[key] = (bg_meta, proc_meta.type)

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
                for key, (bg_meta, proc_type) in new_metas.items():
                    if key not in self._pid_servid_to_info:
                        self._pid_servid_to_info[key] = ChildRemoteCompInfo(
                            proc_type=proc_type,
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

    async def set_vscode_breakpoints_and_get_infos_list(self, bkpts: dict[str, tuple[list[VscodeBreakpoint], int]]):
        infos_list = list(self._pid_servid_to_info.values())
        infos_list.sort(key=lambda x: x.proc_meta.server_id)
        res_list: list[RelayMonitorChildInfo] = []
        for info in infos_list:
            bg_meta = info.proc_meta
            dbg_meta = DebugServerProcessInfo(str(info.pid),
                                            bg_meta.name, info.pid,
                                            bg_meta.server_uuid, bg_meta.server_id, bg_meta.port,
                                            proc_type=info.proc_type)
            robj = self._cached_get_info_robj((info.pid, info.proc_meta.server_uuid))

            try:
                if info.proc_type == BuiltinServiceProcType.SERVER_WITH_DEBUG:
                    debug_info: DebugInfo = await robj.remote_call(
                        dbg_serv_names.DBG_GET_CURRENT_DEBUG_INFO,
                        rpc_timeout=1)
                else:
                    debug_info: DebugInfo = await robj.remote_call(
                        dbg_serv_names.DBG_SET_BKPTS_AND_GET_CURRENT_INFO,
                        bkpts,
                        rpc_timeout=1)
                res = RelayMonitorChildInfo(dbg_meta, debug_info)
            except grpc.aio.AioRpcError as e:
                res = RelayMonitorChildInfo(dbg_meta, None, e.code())
            except:
                print("Failed to connect to", info.port)
                ss = io.StringIO()
                traceback.print_exc(file=ss)
                res = RelayMonitorChildInfo(dbg_meta, None, None, ss.getvalue())
                    
            res_list.append(res)
        return res_list

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
