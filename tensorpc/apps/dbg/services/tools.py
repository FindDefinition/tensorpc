import ast
import asyncio
import dataclasses
import os
import queue
import threading
import time
import traceback
from pathlib import Path
from types import FrameType
from typing import Any, Dict, List, Optional, Tuple

import grpc
import gzip
from tensorpc import prim
from tensorpc.apps.dbg.core.bkpt_events import BkptLaunchTraceEvent, BkptLeaveEvent, BreakpointEvent
from tensorpc.apps.dbg.core.bkptmgr import BreakpointManager, FrameLocMeta
from tensorpc.apps.dbg.model import Breakpoint, TracerState, TracerRuntimeState
from tensorpc.core import BuiltinServiceProcType, inspecttools, marker
from tensorpc.core.asyncclient import simple_remote_call_async
from tensorpc.core.datamodel.draft import capture_draft_update
from tensorpc.core.funcid import (find_toplevel_func_node_by_lineno,
                                  find_toplevel_func_node_container_by_lineno)
from tensorpc.core.serviceunit import ServiceEventType
from tensorpc.apps.dbg.constants import (
    TENSORPC_DBG_FRAME_INSPECTOR_KEY,
    TENSORPC_DBG_TRACE_VIEW_KEY,
    TENSORPC_ENV_DBG_DEFAULT_BREAKPOINT_ENABLE, BackgroundDebugToolsConfig,
    BreakpointType,
    DebugDistributedInfo, DebugFrameInfo, DebugInfo, DebugMetric,
    DebugServerStatus, ExternalTrace, RecordMode, TraceMetrics, TraceResult,
    TracerConfig,
    TracerType)
from tensorpc.apps.dbg.core.sourcecache import LineCache, PythonSourceASTCache, SourceChangeDiffCache
from tensorpc.apps.dbg.serv_names import serv_names
from tensorpc.dock.client import list_all_app_in_machine
from tensorpc.apps.dbg.components.bkptpanel_v2 import BreakpointDebugPanel
from tensorpc.apps.dbg.components.traceview import TraceView
from tensorpc.dock.components.plus.objinspect.tree import BasicObjectTree
from tensorpc.dock.core.appcore import enter_app_context, get_app_context
from tensorpc.dock.serv_names import serv_names as app_serv_names
from tensorpc.dock.vscode.coretypes import VscodeBreakpoint
from tensorpc.utils.proctitle import list_all_tensorpc_server_in_machine
from tensorpc.utils.rich_logging import (
    TENSORPC_LOGGING_OVERRIDED_PATH_LINENO_KEY, get_logger)

LOGGER = get_logger("tensorpc.dbg")


@dataclasses.dataclass
class BreakpointMeta:
    name: Optional[str]
    type: BreakpointType
    frame_loc: FrameLocMeta
    line_text: str
    q: queue.Queue[BreakpointEvent]
    user_dict: Optional[Dict[str, Any]] = None


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
        self._bkpt_mgr = BreakpointManager()
        # self._vscode_breakpoints: Dict[str, List[VscodeBreakpoint]] = {}
        # # workspaceUri -> (path, lineno) -> VscodeBreakpoint
        # self._vscode_breakpoints_dict: Dict[str, Dict[Tuple[Path, int],
        #                                               VscodeBreakpoint]] = {}
        # self._vscode_breakpoints_ts_dict: Dict[Path, int] = {}

        self._bkpt_lock = asyncio.Lock()

        self._cur_tracer_state: Optional[TracerRuntimeState] = None

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
        relay_proc_metas = list_all_tensorpc_server_in_machine(BuiltinServiceProcType.RELAY_MONITOR)
        for meta in relay_proc_metas:
            url = f"localhost:{meta.args[0]}"
            try:
                bkpts = await simple_remote_call_async(
                    url, serv_names.RELAY_GET_VSCODE_BKPTS)
                if bkpts is not None:
                    LOGGER.info(
                        f"Fetch vscode breakpoints from Relay Monitor {meta.name}", url)
                    self._bkpt_mgr.set_vscode_breakpoints(bkpts)
                    return
            except:
                traceback.print_exc()

        all_app_metas = list_all_app_in_machine()
        for meta in all_app_metas:
            url = f"localhost:{meta.app_grpc_port}"
            try:
                bkpts = await simple_remote_call_async(
                    url, app_serv_names.APP_GET_VSCODE_BREAKPOINTS)
                if bkpts is not None:
                    LOGGER.info(
                        f"Fetch vscode breakpoints from App {meta.name}", url)
                    self._bkpt_mgr.set_vscode_breakpoints(bkpts)
                    return
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

    async def set_external_frame(self, frame: Optional[FrameType]):
        """If we only need to inspect frame stack instead of enter
        a breakpoint (e.g. exception), we can set the frame here to 
        avoid pause the program by breakpoint.
        """
        obj, app = prim.get_service(
            app_serv_names.REMOTE_COMP_GET_LAYOUT_ROOT_BY_KEY)(
                TENSORPC_DBG_FRAME_INSPECTOR_KEY)
        assert isinstance(obj, BreakpointDebugPanel)
        draft = obj.dm.get_draft_type_only()
        if frame is None:
            with capture_draft_update() as ctx:
                draft.bkpt = None
            with enter_app_context(app):
                await obj.dm._update_with_jmes_ops(ctx._ops)
            return

        # set external frames to debugger UI.
        frame_select_items = Breakpoint.generate_frame_select_items(frame)
        frame_info = Breakpoint.get_frame_info_from_frame(frame)
        frame_loc = self._bkpt_mgr.get_frame_loc_meta(frame)

        bkpt_model = Breakpoint(
            BreakpointType.Normal,
            frame_info,
            frame_loc,
            frame_select_items,
            frame_select_items[0],
            frame, 
            self.leave_breakpoint,
            self.leave_breakpoint,
            is_external=True
        )
        with capture_draft_update() as ctx:
            draft.bkpt = bkpt_model
        with enter_app_context(app):
            await obj.dm._update_with_jmes_ops(ctx._ops)

    async def enter_breakpoint(self,
                               frame: FrameType,
                               q: queue.Queue,
                               type: BreakpointType,
                               name: Optional[str] = None):
        """should only be called in main thread (server runs in background thread)"""
        # FIXME better vscode breakpoint handling
        if self._cfg.skip_breakpoint:
            q.put(BkptLeaveEvent())
            return
        obj, app = prim.get_service(
            app_serv_names.REMOTE_COMP_GET_LAYOUT_ROOT_BY_KEY)(
                TENSORPC_DBG_FRAME_INSPECTOR_KEY)
        assert isinstance(obj, BreakpointDebugPanel)
        draft = obj.dm.get_draft_type_only()
        async with self._bkpt_lock:
            assert prim.is_loopback_call(
            ), "this function should only be called in main thread"
            # may_changed_frame_lineno is used in breakpoint change detection.
            # user may change source code in vscode after program launch, so we
            # store code of frame when first see it, and compare it with current code
            # by difflib. if the frame lineno is inside a `equal` block, we map
            # frame lineno to the lineno in the new code.
            # may_changed_frame_lineno = self._scd_cache.query_mapped_linenos(
            #     frame.f_code.co_filename, frame.f_lineno)
            # if may_changed_frame_lineno < 1:
            #     may_changed_frame_lineno = frame.f_lineno
            frame_loc = self._bkpt_mgr.get_frame_loc_meta(frame)
            linetext = ""
            pid = os.getpid()
            self._cur_breakpoint = BreakpointMeta(
                name,
                type,
                frame_loc,
                linetext,
                q)
            frame_select_items = Breakpoint.generate_frame_select_items(frame)
            frame_info = Breakpoint.get_frame_info_from_frame(frame)
            bkpt_model = Breakpoint(
                type,
                frame_info,
                frame_loc,
                frame_select_items,
                frame_select_items[0],
                frame, 
                self.leave_breakpoint,
                self.leave_breakpoint,
            )
            with capture_draft_update() as ctx:

                draft.bkpt = bkpt_model
                if obj.dm.model.tracer_state.runtime is not None:
                    pass 
            with enter_app_context(app):
                await obj.dm._update_with_jmes_ops(ctx._ops)

            if self._cur_breakpoint is not None and self._cur_breakpoint.type == BreakpointType.Vscode:
                is_cur_bkpt_is_vscode = self._bkpt_mgr.check_vscode_bkpt_is_enabled(
                    self._cur_breakpoint.frame_loc)
                if not is_cur_bkpt_is_vscode:
                    q.put(BkptLeaveEvent())
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
            if self._cur_tracer_state is not None:
                new_bkpt_cnt, is_record_stop = self._cur_tracer_state.increment_trace_state(
                    frame_loc)
                self._cur_tracer_state.metric.breakpoint_count = new_bkpt_cnt
                if not is_record_stop:
                    q.put(BkptLeaveEvent())
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
                else:
                    self._cur_tracer_state = None
                
            assert self._frame is None, "already in breakpoint, shouldn't happen"
            self._frame = frame
            self._debug_metric.total_skipped_bkpt = 0
            # obj, app = prim.get_service(
            #     app_serv_names.REMOTE_COMP_GET_LAYOUT_ROOT_BY_KEY)(
            #         TENSORPC_DBG_FRAME_INSPECTOR_KEY)
            # assert isinstance(obj, BreakpointDebugPanel)
                # await obj.set_breakpoint_frame_meta(frame,
                #                                     self.leave_breakpoint,
                #                                     is_record_stop)
            self._cur_status = DebugServerStatus.InsideBreakpoint
            LOGGER.warning(
                f"Breakpoint({type.name}), "
                f"port = {prim.get_server_meta().port}, "
                f"pid = {pid}",
                extra={
                    TENSORPC_LOGGING_OVERRIDED_PATH_LINENO_KEY:
                    (frame.f_code.co_filename, frame.f_lineno)
                })
            return is_record_stop

    async def leave_breakpoint(self, trace_cfg: Optional[TracerConfig] = None):
        """should only be called from remote"""
        assert not prim.is_loopback_call(
        ), "this function should only be called from remote"
        obj, app = prim.get_service(
            app_serv_names.REMOTE_COMP_GET_LAYOUT_ROOT_BY_KEY)(
                TENSORPC_DBG_FRAME_INSPECTOR_KEY)
        assert isinstance(obj, BreakpointDebugPanel)
        draft = obj.dm.get_draft_type_only()
        with capture_draft_update() as ctx:
            prev_bkpt = obj.dm.model.bkpt
            if trace_cfg is not None and trace_cfg.enable and prev_bkpt is not None:
                draft.tracer_state.runtime = TracerState.create_new_runtime(trace_cfg, prev_bkpt.frame_loc)
            draft.bkpt = None 

        async with self._bkpt_lock:
            # obj, app = prim.get_service(
            #     app_serv_names.REMOTE_COMP_GET_LAYOUT_ROOT_BY_KEY)(
            #         TENSORPC_DBG_FRAME_INSPECTOR_KEY)
            # assert isinstance(obj, BreakpointDebugPanel)
            is_record_start = False
            if self._cur_breakpoint is not None:
                if trace_cfg is not None and trace_cfg.enable and self._frame is not None:
                    if self._cur_tracer_state is None:
                        is_record_start = True
            
            if get_app_context() is None:
                with enter_app_context(app):
                    await obj.dm._update_with_jmes_ops(ctx._ops)
                    # await obj.leave_breakpoint(is_record_start)
            else:
                await obj.dm._update_with_jmes_ops(ctx._ops)
                # await obj.leave_breakpoint(is_record_start)
            self._cur_status = DebugServerStatus.Idle
            if self._cur_breakpoint is not None:
                if trace_cfg is not None and trace_cfg.enable and self._frame is not None:
                    if self._cur_tracer_state is None:
                        metric = TraceMetrics(trace_cfg.breakpoint_count)
                        self._cur_tracer_state = TracerRuntimeState(
                            trace_cfg, metric,
                            FrameLocMeta(self._frame.f_code.co_filename,
                            self._frame.f_lineno, -1))
                        # self._cur_breakpoint.event.enable_trace_in_main_thread = True
                        # self._cur_breakpoint.event.trace_cfg = trace_cfg
                    self._cur_breakpoint.q.put(BkptLaunchTraceEvent(trace_cfg))

                # self._cur_breakpoint.event.set()

                self._cur_breakpoint.q.put(BkptLeaveEvent())

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
        # self._cur_tracer_state.tracer = tracer

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
        obj, app = prim.get_service(
            app_serv_names.REMOTE_COMP_GET_LAYOUT_ROOT_BY_KEY)(
                TENSORPC_DBG_FRAME_INSPECTOR_KEY)
        assert isinstance(obj, BreakpointDebugPanel)

        model = obj.dm.model
        frame_info: Optional[DebugFrameInfo] = None
        debug_metric = self._debug_metric
        if model.bkpt is not None and model.bkpt.frame is not None:
            frame = model.bkpt.frame
            qname = inspecttools.get_co_qualname_from_frame(frame)
            frame_info = DebugFrameInfo(frame.f_code.co_name, qname,
                                        frame.f_code.co_filename,
                                        frame.f_lineno)
            debug_metric = DebugMetric(-1)
        trace_cfg: Optional[TracerConfig] = None
        if self._cur_tracer_state is not None:
            trace_cfg = self._cur_tracer_state.cfg
        return DebugInfo(debug_metric, frame_info, trace_cfg, self._distributed_meta)

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

    async def set_vscode_breakpoints(self,
                                     bkpts: Dict[str, tuple[list[VscodeBreakpoint], int]]):
        self._bkpt_mgr.set_vscode_breakpoints(bkpts)
        if self._cur_breakpoint is not None and self._cur_breakpoint.type == BreakpointType.Vscode:
            is_cur_bkpt_is_vscode = self._bkpt_mgr.check_vscode_bkpt_is_enabled_after_set_vscode_bkpt(self._cur_breakpoint.frame_loc)
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
        if self._frame is None:
            return
        obj, app = prim.get_service(
            app_serv_names.REMOTE_COMP_GET_LAYOUT_ROOT_BY_KEY)(
                TENSORPC_DBG_FRAME_INSPECTOR_KEY)
        assert isinstance(obj, BreakpointDebugPanel)
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
