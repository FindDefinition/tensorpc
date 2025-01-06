import ast
import asyncio
import dataclasses
import os
import threading
import time
import traceback
from pathlib import Path
from types import FrameType
from typing import Any, Dict, List, Optional, Tuple

import grpc
import gzip
from tensorpc import prim
from tensorpc.core import inspecttools, marker
from tensorpc.core.asyncclient import simple_remote_call_async
from tensorpc.core.funcid import (find_toplevel_func_node_by_lineno,
                                  find_toplevel_func_node_container_by_lineno)
from tensorpc.core.serviceunit import ServiceEventType
from tensorpc.dbg.constants import (
    TENSORPC_DBG_FRAME_INSPECTOR_KEY,
    TENSORPC_DBG_TRACE_VIEW_KEY,
    TENSORPC_ENV_DBG_DEFAULT_BREAKPOINT_ENABLE, BackgroundDebugToolsConfig,
    BreakpointEvent, BreakpointType,
    DebugDistributedInfo, DebugFrameInfo, DebugInfo, DebugMetric,
    DebugServerStatus, ExternalTrace, RecordMode, TraceMetrics, TraceResult,
    TracerConfig,
    TracerType)
from tensorpc.dbg.core.sourcecache import LineCache, PythonSourceASTCache, SourceChangeDiffCache
from tensorpc.dbg.serv_names import serv_names
from tensorpc.flow.client import list_all_app_in_machine
from tensorpc.flow.components.plus.dbg.bkptpanel import BreakpointDebugPanel
from tensorpc.flow.components.plus.dbg.traceview import TraceView
from tensorpc.flow.components.plus.objinspect.tree import BasicObjectTree
from tensorpc.flow.core.appcore import enter_app_context, get_app_context
from tensorpc.flow.serv_names import serv_names as app_serv_names
from tensorpc.flow.vscode.coretypes import VscodeBreakpoint
from tensorpc.utils.rich_logging import (
    TENSORPC_LOGGING_OVERRIDED_PATH_LINENO_KEY, get_logger)

LOGGER = get_logger("tensorpc.dbg")


@dataclasses.dataclass
class BreakpointMeta:
    name: Optional[str]
    type: BreakpointType
    path: str
    lineno: int
    line_text: str
    event: BreakpointEvent
    user_dict: Optional[Dict[str, Any]] = None
    mapped_lineno: Optional[int] = None


@dataclasses.dataclass
class TracerState:
    tracer: Any
    cfg: TracerConfig
    metric: TraceMetrics
    frame_identifier: Tuple[str, int]  # path, lineno
    force_stop: bool = False


class BackgroundDebugTools:

    def __init__(self) -> None:
        self._frame = None
        self._event: Optional[threading.Event] = None

        self._cur_status = DebugServerStatus.Idle
        self._cur_breakpoint: Optional[BreakpointMeta] = None

        self._cfg = BackgroundDebugToolsConfig(
            skip_breakpoint=not TENSORPC_ENV_DBG_DEFAULT_BREAKPOINT_ENABLE)

        self._ast_cache = PythonSourceASTCache()
        self._line_cache = LineCache()
        self._scd_cache = SourceChangeDiffCache()

        self._vscode_breakpoints: Dict[str, List[VscodeBreakpoint]] = {}
        # workspaceUri -> (path, lineno) -> VscodeBreakpoint
        self._vscode_breakpoints_dict: Dict[str, Dict[Tuple[Path, int],
                                                      VscodeBreakpoint]] = {}
        self._vscode_breakpoints_ts_dict: Dict[Path, int] = {}

        self._bkpt_lock = asyncio.Lock()

        self._cur_tracer_state: Optional[TracerState] = None

        self._trace_gzip_data_dict: Dict[str, Tuple[int, TraceResult]] = {}

        self._debug_metric = DebugMetric(0)

        self._distributed_meta: Optional[DebugDistributedInfo] = None


    def set_distributed_meta(self, meta: DebugDistributedInfo):
        self._distributed_meta = meta

    @marker.mark_server_event(event_type=ServiceEventType.Exit)
    def _on_exit(self):
        pass
        # if self._cur_tracer_state is not None:
        #     self._cur_tracer_state.tracer.stop()

    # @marker.mark_server_event(event_type=ServiceEventType.BeforeServerStart)
    async def try_fetch_vscode_breakpoints(self):
        all_app_metas = list_all_app_in_machine()
        for meta in all_app_metas:
            url = f"localhost:{meta.app_grpc_port}"
            try:
                bkpts = await simple_remote_call_async(
                    url, app_serv_names.APP_GET_VSCODE_BREAKPOINTS)
                if bkpts is not None:
                    LOGGER.info(
                        f"Fetch vscode breakpoints from App {meta.name}", url)
                    self._set_vscode_breakpoints_and_dict(bkpts)
                    break
            except:
                traceback.print_exc()

    async def set_skip_breakpoint(self, skip: bool):
        obj, app = prim.get_service(
            app_serv_names.REMOTE_COMP_GET_LAYOUT_ROOT_BY_KEY)(
                TENSORPC_DBG_FRAME_INSPECTOR_KEY)
        assert isinstance(obj, BreakpointDebugPanel)
        with enter_app_context(app):
            await obj._skip_further_bkpt(skip)

    def init_bkpt_debug_panel(self, panel: BreakpointDebugPanel):
        # panel may change the cfg
        panel._bkgd_debug_tool_cfg = self._cfg

    async def enter_breakpoint(self,
                               frame: FrameType,
                               event: BreakpointEvent,
                               type: BreakpointType,
                               name: Optional[str] = None):
        """should only be called in main thread (server runs in background thread)"""
        # FIXME better vscode breakpoint handling
        if self._cfg.skip_breakpoint:
            event.set()
            return
        async with self._bkpt_lock:
            assert prim.is_loopback_call(
            ), "this function should only be called in main thread"
            # may_changed_frame_lineno is used in breakpoint change detection.
            # user may change source code in vscode after program launch, so we
            # store code of frame when first see it, and compare it with current code
            # by difflib. if the frame lineno is inside a `equal` block, we map
            # frame lineno to the lineno in the new code.
            may_changed_frame_lineno = self._scd_cache.query_mapped_linenos(
                frame.f_code.co_filename, frame.f_lineno)
            if may_changed_frame_lineno < 1:
                may_changed_frame_lineno = frame.f_lineno
            # try:
            #     lines = self._line_cache.getlines(frame.f_code.co_filename)
            #     linetext = lines[frame.f_lineno - 1]
            # except:
            linetext = ""
            pid = os.getpid()
            self._cur_breakpoint = BreakpointMeta(
                name,
                type,
                frame.f_code.co_filename,
                frame.f_lineno,
                linetext,
                event,
                mapped_lineno=may_changed_frame_lineno)
            if self._cur_breakpoint is not None and self._cur_breakpoint.type == BreakpointType.Vscode:
                is_cur_bkpt_is_vscode = self._determine_vscode_bkpt_status(
                    self._cur_breakpoint, self._vscode_breakpoints_dict)
                if not is_cur_bkpt_is_vscode:
                    event.set()
                    # LOGGER.warning(
                    #     f"Skip Vscode breakpoint",
                    #     extra={
                    #         TENSORPC_LOGGING_OVERRIDED_PATH_LINENO_KEY:
                    #         (frame.f_code.co_filename, frame.f_lineno)
                    #     })
                    self._cur_breakpoint = None
                    self._debug_metric.total_skipped_bkpt += 1
                    return
            res_tracer = None
            is_record_stop = False
            if self._cur_tracer_state is not None and self._cur_tracer_state.tracer is not None:
                # is tracing
                cfg = self._cur_tracer_state.cfg
                metric = self._cur_tracer_state.metric
                is_same_bkpt = False
                is_inf_record = cfg.mode == RecordMode.INFINITE
                if cfg.mode == RecordMode.SAME_BREAKPOINT:
                    is_same_bkpt = self._cur_tracer_state.frame_identifier == (
                        frame.f_code.co_filename, frame.f_lineno)
                if not is_inf_record:
                    metric.breakpoint_count -= 1
                if (metric.breakpoint_count == 0 and not is_inf_record
                    ) or is_same_bkpt or self._cur_tracer_state.force_stop:
                    res_tracer = (self._cur_tracer_state.tracer,
                                  self._cur_tracer_state.cfg)
                    self._cur_tracer_state = None
                    is_record_stop = True
                if not is_record_stop:
                    event.set()
                    # if cfg.mode != RecordMode.INFINITE:
                    #     msg_str = f"Skip Vscode breakpoint (Remaining trace count: {metric.breakpoint_count})"
                    #     LOGGER.warning(
                    #         msg_str,
                    #         extra={
                    #             TENSORPC_LOGGING_OVERRIDED_PATH_LINENO_KEY:
                    #             (frame.f_code.co_filename, frame.f_lineno)
                    #         })
                    self._cur_breakpoint = None
                    self._debug_metric.total_skipped_bkpt += 1
                    return
            assert self._frame is None, "already in breakpoint, shouldn't happen"
            self._frame = frame
            self._debug_metric.total_skipped_bkpt = 0
            LOGGER.warning(
                f"Breakpoint({type.name}), "
                f"port = {prim.get_server_meta().port}, "
                f"pid = {pid}",
                extra={
                    TENSORPC_LOGGING_OVERRIDED_PATH_LINENO_KEY:
                    (frame.f_code.co_filename, frame.f_lineno)
                })
            obj, app = prim.get_service(
                app_serv_names.REMOTE_COMP_GET_LAYOUT_ROOT_BY_KEY)(
                    TENSORPC_DBG_FRAME_INSPECTOR_KEY)
            assert isinstance(obj, BreakpointDebugPanel)
            with enter_app_context(app):
                await obj.set_breakpoint_frame_meta(frame,
                                                    self.leave_breakpoint,
                                                    is_record_stop)
            self._cur_status = DebugServerStatus.InsideBreakpoint
            return res_tracer

    async def leave_breakpoint(self, trace_cfg: Optional[TracerConfig] = None):
        """should only be called from remote"""
        assert not prim.is_loopback_call(
        ), "this function should only be called from remote"
        async with self._bkpt_lock:
            obj, app = prim.get_service(
                app_serv_names.REMOTE_COMP_GET_LAYOUT_ROOT_BY_KEY)(
                    TENSORPC_DBG_FRAME_INSPECTOR_KEY)
            assert isinstance(obj, BreakpointDebugPanel)
            is_record_start = False
            if self._cur_breakpoint is not None:
                if trace_cfg is not None and trace_cfg.enable and self._frame is not None:
                    if self._cur_tracer_state is None:
                        is_record_start = True
            if get_app_context() is None:
                with enter_app_context(app):
                    await obj.leave_breakpoint(is_record_start)
            else:
                await obj.leave_breakpoint(is_record_start)
            self._cur_status = DebugServerStatus.Idle
            if self._cur_breakpoint is not None:
                if trace_cfg is not None and trace_cfg.enable and self._frame is not None:
                    if self._cur_tracer_state is None:
                        metric = TraceMetrics(trace_cfg.breakpoint_count)
                        self._cur_tracer_state = TracerState(
                            None, trace_cfg, metric,
                            (self._frame.f_code.co_filename,
                             self._frame.f_lineno))
                        self._cur_breakpoint.event.enable_trace_in_main_thread = True
                        self._cur_breakpoint.event.trace_cfg = trace_cfg
                self._cur_breakpoint.event.set()
                self._cur_breakpoint = None
            self._frame = None

    async def set_traceview_variable_inspect(self, var_name: str, var_obj: Any):
        tv_obj, tv_app = prim.get_service(
            app_serv_names.REMOTE_COMP_GET_LAYOUT_ROOT_BY_KEY)(
                TENSORPC_DBG_TRACE_VIEW_KEY)
        assert isinstance(tv_obj, TraceView)
        with enter_app_context(tv_app):
            await tv_obj.set_variable_trace_result(var_name, var_obj)

    def set_tracer(self, tracer: Any):
        assert self._cur_tracer_state is not None
        self._cur_tracer_state.tracer = tracer

    async def set_trace_data(self, trace_res: TraceResult, cfg: TracerConfig):
        obj, app = prim.get_service(
            app_serv_names.REMOTE_COMP_GET_LAYOUT_ROOT_BY_KEY)(
                TENSORPC_DBG_FRAME_INSPECTOR_KEY)
        assert isinstance(obj, BreakpointDebugPanel)
        with enter_app_context(app):
            await obj.set_perfetto_data(trace_res.single_results[0])
        for single_trace_res in trace_res.single_results:
            if single_trace_res.tracer_type == TracerType.VIZTRACER:
                tv_obj, tv_app = prim.get_service(
                    app_serv_names.REMOTE_COMP_GET_LAYOUT_ROOT_BY_KEY)(
                        TENSORPC_DBG_TRACE_VIEW_KEY)
                assert isinstance(tv_obj, TraceView)
                with enter_app_context(tv_app):
                    await tv_obj.set_trace_events(single_trace_res)
        if cfg.trace_timestamp is not None:
            name = "default"
            if cfg.trace_name is not None:
                name = cfg.trace_name
            trace_res_compressed = [
                dataclasses.replace(x, data=gzip.compress(x.data))
                for x in trace_res.single_results
            ]
            LOGGER.warning(
                f"Compress trace data: {len(trace_res.single_results[0].data)} -> {len(trace_res_compressed[0].data)}"
            )
            self._trace_gzip_data_dict[name] = (cfg.trace_timestamp,
                                                dataclasses.replace(
                                                    trace_res,
                                                    single_results=trace_res_compressed))

    def get_trace_data(self, name: str):
        if name in self._trace_gzip_data_dict:
            res = self._trace_gzip_data_dict[name]
            res_remove_trace_events: TraceResult = TraceResult([])
            for single_res in res[1].single_results:
                # remove raw trace events, they should only be used in remote comp.
                res_remove_trace_events.single_results.append(
                    dataclasses.replace(single_res, trace_events=None))
            return (res[0], res_remove_trace_events)
        return None

    def get_trace_data_timestamp(self, name: str):
        if name in self._trace_gzip_data_dict:
            res = self._trace_gzip_data_dict[name]
            return res[0]
        return None

    def get_trace_data_keys(self):
        return list(self._trace_gzip_data_dict.keys())

    def bkgd_get_cur_frame(self):
        return self._frame

    def get_cur_debug_info(self):
        frame_info: Optional[DebugFrameInfo] = None

        if self._frame is not None:
            qname = inspecttools.get_co_qualname_from_frame(self._frame)
            frame_info = DebugFrameInfo(self._frame.f_code.co_name, qname,
                                        self._frame.f_code.co_filename,
                                        self._frame.f_lineno)
        trace_cfg: Optional[TracerConfig] = None
        if self._cur_tracer_state is not None:
            trace_cfg = self._cur_tracer_state.cfg
        return DebugInfo(self._debug_metric, frame_info, trace_cfg, self._distributed_meta)

    def _get_filtered_local_vars(self, frame: FrameType):
        local_vars = frame.f_locals.copy()
        local_vars = inspecttools.filter_local_vars(local_vars)
        return local_vars

    def list_current_frame_vars(self):
        assert self._frame is not None
        local_vars = self._get_filtered_local_vars(self._frame)
        return list(local_vars.keys())

    def eval_expr_in_current_frame(self, expr: str):
        assert self._frame is not None
        local_vars = self._get_filtered_local_vars(self._frame)
        return eval(expr, None, local_vars)

    def _determine_vscode_bkpt_status(
            self, bkpt_meta: BreakpointMeta, 
            vscode_bkpt_dict: Dict[str, Dict[Tuple[Path, int],
                                             VscodeBreakpoint]]):
        if bkpt_meta.type == BreakpointType.Vscode:
            key = (Path(bkpt_meta.path).resolve(), bkpt_meta.mapped_lineno)
            # rich.print("BKPT", key, vscode_bkpt_dict)
            for bkpts in vscode_bkpt_dict.values():
                if key in bkpts:
                    return bkpts[key].enabled
        return False

    def _set_vscode_breakpoints_and_dict(
            self, bkpt_dict: Dict[str, tuple[List[VscodeBreakpoint], int]]):
        for wuri, (bkpts, ts) in bkpt_dict.items():
            new_bkpts: List[VscodeBreakpoint] = []
            for x in bkpts:
                if x.enabled and x.lineText is not None and (
                        ".breakpoint" in x.lineText
                        or ".vscode_breakpoint" in x.lineText):
                    new_bkpts.append(x)
            if wuri not in self._vscode_breakpoints_dict:
                self._vscode_breakpoints_dict[wuri] = {}
            self._vscode_breakpoints_dict[wuri] = {
                (Path(x.path).resolve(), x.line + 1): x
                for x in new_bkpts
            }
            # save bkpt timestamp
            for x in new_bkpts:
                self._vscode_breakpoints_ts_dict[Path(x.path).resolve()] = ts
            self._vscode_breakpoints[wuri] = new_bkpts

    async def set_vscode_breakpoints(self,
                                     bkpts: Dict[str, tuple[list[VscodeBreakpoint], int]]):
        self._set_vscode_breakpoints_and_dict(bkpts)
        if self._cur_breakpoint is not None and self._cur_breakpoint.type == BreakpointType.Vscode:
            # update mapped lineno here because file may change during breakpoint
            mtime = None 
            may_changed_frame_lineno = self._scd_cache.query_mapped_linenos(
                self._cur_breakpoint.path, self._cur_breakpoint.lineno)
            cache_entry = self._scd_cache.cache[self._cur_breakpoint.path]
            mtime = cache_entry.mtime
            if mtime is None:
                return 
            self._cur_breakpoint.mapped_lineno = may_changed_frame_lineno
            mtime_ns = int(mtime * 1e9)
            if Path(self._cur_breakpoint.path).resolve() in self._vscode_breakpoints_ts_dict:
                vscode_bkpt_ts = self._vscode_breakpoints_ts_dict[Path(self._cur_breakpoint.path).resolve()]
                if mtime_ns > vscode_bkpt_ts:
                    # vscode bkpt state is outdated, skip current check.
                    return
            is_cur_bkpt_is_vscode = self._determine_vscode_bkpt_status(
                self._cur_breakpoint, self._vscode_breakpoints_dict)
            # if not found, release this breakpoint
            if not is_cur_bkpt_is_vscode:
                await self.leave_breakpoint()

    async def set_vscode_breakpoints_and_get_cur_info(
            self, bkpts: Dict[str, tuple[List[VscodeBreakpoint], int]]):
        info = self.get_cur_debug_info()
        await self.set_vscode_breakpoints(bkpts)
        return info

    async def force_trace_stop(self):
        if self._cur_tracer_state is not None:
            self._cur_tracer_state.force_stop = True
            # actual stop will be done in next enter breakpoint.

    async def handle_code_selection_msg(self, code_segment: str, path: str,
                                        code_range: Tuple[int, int, int, int]):
        # print("WTF", code_segment, path, code_range)
        if self._frame is None:
            return
        obj, app = prim.get_service(
            app_serv_names.REMOTE_COMP_GET_LAYOUT_ROOT_BY_KEY)(
                TENSORPC_DBG_FRAME_INSPECTOR_KEY)
        assert isinstance(obj, BreakpointDebugPanel)
        # print(2)
        # parse path ast to get function location
        tree = self._ast_cache.getast(path)
        assert isinstance(tree, ast.Module)
        # print(tree)
        nodes = find_toplevel_func_node_container_by_lineno(
            tree, code_range[0])
        # print(res)
        if nodes is not None:
            node_qname = ".".join([n.name for n in nodes])
            cur_frame: Optional[FrameType] = self._frame
            with enter_app_context(app):
                while cur_frame is not None:
                    if Path(cur_frame.f_code.co_filename).resolve() == Path(
                            path).resolve():
                        qname = inspecttools.get_co_qualname_from_frame(
                            cur_frame)
                        # print(qname, node_qname)
                        if node_qname == qname:
                            # found. eval expr in this frame
                            try:
                                local_vars = cur_frame.f_locals
                                global_vars = cur_frame.f_globals
                                res = eval(code_segment, global_vars,
                                           local_vars)
                                await obj.set_frame_object(
                                    res, code_segment, nodes[-1], cur_frame)
                            except grpc.aio.AioRpcError as e:
                                del cur_frame
                                return
                            except Exception as e:
                                LOGGER.info(
                                    f"Eval code segment failed. exception: {e}"
                                )
                                # print(e)
                                # traceback.print_exc()
                                # await obj.send_exception(e)
                                del cur_frame
                                return
                    cur_frame = cur_frame.f_back
            del cur_frame
