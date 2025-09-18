import ast
import asyncio
import dataclasses
import io
import os
import queue
import threading
import time
import traceback
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

import grpc
import gzip
from tensorpc import prim
from tensorpc.apps.dbg.core.bkpt_events import BreakpointEvent, BkptLeaveEvent, BkptLaunchTraceEvent, BkptRunScriptEvent
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
from tensorpc.core.astex.sourcecache import LineCache, PythonSourceASTCache, SourceChangeDiffCache
from tensorpc.apps.dbg.serv_names import serv_names
from tensorpc.dock.client import list_all_app_in_machine
from tensorpc.apps.dbg.components.bkptpanel import BreakpointDebugPanel
from tensorpc.apps.dbg.components.traceview import TraceView
from tensorpc.dock.components.plus.objinspect.tree import BasicObjectTree
from tensorpc.dock.core.appcore import enter_app_context, get_app_context
from tensorpc.dock.serv_names import serv_names as app_serv_names
from tensorpc.dock.vscode.coretypes import VscodeBreakpoint
from tensorpc.utils.proctitle import list_all_tensorpc_server_in_machine
from tensorpc.utils.rich_logging import (
    TENSORPC_LOGGING_OVERRIDED_PATH_LINENO_KEY, get_logger)
import tarfile 

def _compress_bytes_to_tar(data: bytes, filename: str):
    ss = io.BytesIO()
    with tarfile.open(fileobj=ss, mode="w:gz") as tar:
        info = tarfile.TarInfo(filename)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return ss.getvalue()


LOGGER = get_logger("tensorpc.dbg")


class BackgroundDebugTools:

    def __init__(self) -> None:
        self._event: Optional[threading.Event] = None

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

    async def run_frame_script(self, code: str):
        dm, _ = self._get_bkgd_panel_dm_and_app()
        prev_bkpt = dm.model.bkpt
        if prev_bkpt is not None:
            assert prev_bkpt.queue is not None 
            prev_bkpt.queue.put(BkptRunScriptEvent(code))

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
                               should_enter_fn: Optional[Callable[[Breakpoint], bool]] = None):
        """should only be called in main thread (server runs in background thread).
        
        Args:
            frame (FrameType): the frame to be inspected.
            q (queue.Queue): the queue to be used for communication between
                background server and user program.
            type (BreakpointType): the type of the breakpoint.
            should_enter_fn (Callable[[Breakpoint], bool], optional): a function to determine
                whether to enter the breakpoint. Defaults to None. this is usually used
                when you have a external server to determine whether to enter the breakpoint.
        """
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
            pid = os.getpid()
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
                queue=q
            )
            if bkpt_model.type == BreakpointType.Vscode:
                is_cur_bkpt_is_vscode = self._bkpt_mgr.check_vscode_bkpt_is_enabled(
                    bkpt_model.frame_loc)
                if not is_cur_bkpt_is_vscode:
                    q.put(BkptLeaveEvent())
                    self._debug_metric.total_skipped_bkpt += 1
                    return
            cur_tracer_state = obj.dm.model.tracer_state.runtime
            is_record_stop = False
            if cur_tracer_state is not None:
                new_bkpt_cnt, is_record_stop = cur_tracer_state.increment_trace_state(
                    frame_loc)
                with capture_draft_update() as ctx:
                    draft.tracer_state.runtime.metric.breakpoint_count = new_bkpt_cnt # type: ignore
                with enter_app_context(app):

                    await obj.dm._update_with_jmes_ops(ctx._ops)
                if not is_record_stop:
                    q.put(BkptLeaveEvent())
                    self._debug_metric.total_skipped_bkpt += 1
                    return
                else:
                    with capture_draft_update() as ctx:
                        draft.tracer_state.runtime = None
                    with enter_app_context(app):
                        await obj.dm._update_with_jmes_ops(ctx._ops)
            should_enter = True
            if should_enter_fn is not None:
                should_enter = should_enter_fn(bkpt_model)
            if not should_enter:
                return 
            with capture_draft_update() as ctx:
                draft.bkpt = bkpt_model
                # if obj.dm.model.tracer_state.runtime is not None:
                #     pass 
            with enter_app_context(app):
                await obj.dm._update_with_jmes_ops(ctx._ops)

            self._debug_metric.total_skipped_bkpt = 0
            LOGGER.warning(
                f"Breakpoint({type.name}), "
                f"port = {prim.get_server_meta().port}, "
                f"pid = {pid}",
                extra={
                    TENSORPC_LOGGING_OVERRIDED_PATH_LINENO_KEY:
                    (frame.f_code.co_filename, frame.f_lineno)
                })
            return is_record_stop

    async def leave_breakpoint(self, trace_cfg: Optional[TracerConfig] = None, userdata: Any = None, should_raise: bool = False):
        """should only be called from remote.
        
        Args:
            trace_cfg (Optional[TracerConfig], optional): the trace config. Defaults to None.
            userdata (Any, optional): the user data. Defaults to None.
                user can use this to send a control message to foreground
                program.
        """
        assert not prim.is_loopback_call(
        ), "this function should only be called from remote"
        dm, app = self._get_bkgd_panel_dm_and_app()
        draft = dm.get_draft_type_only()
        prev_bkpt: Optional[Breakpoint] = None
        if dm.model.bkpt is None:
            # ignore if no breakpoint
            LOGGER.warning(
                "No breakpoint found."
            )
            return 
        with capture_draft_update() as ctx:
            prev_bkpt = dm.model.bkpt
            if trace_cfg is not None and trace_cfg.enable and prev_bkpt is not None:
                draft.tracer_state.runtime = TracerState.create_new_runtime(trace_cfg, prev_bkpt.frame_loc)
            draft.bkpt = None 
        async with self._bkpt_lock:
            with enter_app_context(app):
                await dm._update_with_jmes_ops(ctx._ops)
            if prev_bkpt is not None:
                assert prev_bkpt.queue is not None 
                if trace_cfg is not None and trace_cfg.enable:
                    prev_bkpt.queue.put(BkptLaunchTraceEvent(trace_cfg))
                prev_bkpt.queue.put(BkptLeaveEvent(userdata, should_raise))

    async def set_traceview_variable_inspect(self, var_name: str, var_obj: Any):
        tv_obj, tv_app = prim.get_service(
            app_serv_names.REMOTE_COMP_GET_LAYOUT_ROOT_BY_KEY)(
                TENSORPC_DBG_TRACE_VIEW_KEY)
        assert isinstance(tv_obj, TraceView)
        with enter_app_context(tv_app):
            await tv_obj.set_variable_trace_result(var_name, var_obj)

    def set_tracer(self, tracer: Any):
        pass 
        # assert self._cur_tracer_state is not None
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
            name = cfg.trace_name
            if cfg.manual_scope != "":
                name = f"{name}_{cfg.manual_scope}"
            use_tar = False
            uid = uuid.uuid4().hex
            if use_tar:
                trace_res_compressed = [
                    dataclasses.replace(x, data=_compress_bytes_to_tar(x.data, f"{uid}.tar"), is_tar=True, fname=f"{uid}.tar")
                    for x in trace_res.single_results
                ]
            else:
                trace_res_compressed = [
                    dataclasses.replace(x, data=gzip.compress(x.data), is_tar=False, fname=f"{uid}.gz")
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
        dm, app = self._get_bkgd_panel_dm_and_app()
        assert dm.model.bkpt is not None 
        return dm.model.bkpt.frame

    def _get_bkgd_panel_dm_and_app(self):
        obj, app = prim.get_service(
            app_serv_names.REMOTE_COMP_GET_LAYOUT_ROOT_BY_KEY)(
                TENSORPC_DBG_FRAME_INSPECTOR_KEY)
        assert isinstance(obj, BreakpointDebugPanel)
        return obj.dm, app

    def get_cur_debug_info(self):
        dm, app = self._get_bkgd_panel_dm_and_app()
        model = dm.model
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

        if model.tracer_state.runtime is not None:
            trace_cfg = model.tracer_state.runtime.cfg
        return DebugInfo(debug_metric, frame_info, trace_cfg, self._distributed_meta)

    def _get_filtered_local_vars(self, frame: FrameType):
        local_vars = frame.f_locals.copy()
        local_vars = inspecttools.filter_local_vars(local_vars)
        return local_vars

    def list_current_frame_vars(self):
        dm, app = self._get_bkgd_panel_dm_and_app()
        assert dm.model.bkpt is not None and dm.model.bkpt.frame is not None 
        local_vars = self._get_filtered_local_vars(dm.model.bkpt.frame)
        return list(local_vars.keys())

    def eval_expr_in_current_frame(self, expr: str):
        dm, app = self._get_bkgd_panel_dm_and_app()
        assert dm.model.bkpt is not None and dm.model.bkpt.frame is not None 
        local_vars = self._get_filtered_local_vars(dm.model.bkpt.frame)
        return eval(expr, None, local_vars)

    async def set_vscode_breakpoints(self,
                                     bkpts: Dict[str, tuple[list[VscodeBreakpoint], int]]):
        self._bkpt_mgr.set_vscode_breakpoints(bkpts)
        dm, app = self._get_bkgd_panel_dm_and_app()
        model = dm.model
        if model.bkpt is not None and model.bkpt.type == BreakpointType.Vscode:
            is_cur_bkpt_is_vscode = self._bkpt_mgr.check_vscode_bkpt_is_enabled_after_set_vscode_bkpt(model.bkpt.frame_loc)
            # if not found, release this breakpoint
            if not is_cur_bkpt_is_vscode:
                await self.leave_breakpoint()

    async def set_vscode_breakpoints_and_get_cur_info(
            self, bkpts: Dict[str, tuple[List[VscodeBreakpoint], int]]):
        info = self.get_cur_debug_info()
        await self.set_vscode_breakpoints(bkpts)
        return info

    async def trace_stop_in_next_bkpt(self):
        dm, app = self._get_bkgd_panel_dm_and_app()
        if dm.model.tracer_state.runtime is not None:
            async with dm.draft_update() as draft:
                draft.tracer_state.runtime.force_stop = True # type: ignore
            # actual stop will be done in next enter breakpoint.

    async def force_trace_stop(self):
        dm, app = self._get_bkgd_panel_dm_and_app()
        if dm.model.tracer_state.runtime is not None:
            async with dm.draft_update() as draft:
                draft.tracer_state.runtime.force_stop = True # type: ignore

    async def handle_code_selection_msg(self, code_segment: str, path: str,
                                        code_range: Tuple[int, int, int, int]):
        dm, app = self._get_bkgd_panel_dm_and_app()
        frame: Optional[FrameType] = None 
        if dm.model.bkpt is not None:
            frame = dm.model.bkpt.frame
        if frame is None:
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
            cur_frame: Optional[FrameType] = frame
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
