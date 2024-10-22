import ast
import dataclasses
from pathlib import Path
import threading
import traceback
from types import FrameType
from typing import Any, Dict, Optional, Tuple
from tensorpc.core import inspecttools
from tensorpc import prim
from tensorpc.core.funcid import find_toplevel_func_node_by_lineno, find_toplevel_func_node_container_by_lineno
from tensorpc.dbg.constants import TENSORPC_DBG_FRAME_INSPECTOR_KEY, DebugFrameMeta, DebugServerStatus, BackgroundDebugToolsConfig
from tensorpc.dbg.core.sourcecache import PythonSourceASTCache
from tensorpc.dbg.serv_names import serv_names
from tensorpc.flow.components.plus.dbg.bkptpanel import BreakpointDebugPanel
from tensorpc.flow.components.plus.objinspect.tree import BasicObjectTree
from tensorpc.flow.core.appcore import enter_app_context
from tensorpc.flow.serv_names import serv_names as app_serv_names


class BackgroundDebugTools:

    def __init__(self) -> None:
        self._frame = None
        self._event: Optional[threading.Event] = None

        self._cur_status = DebugServerStatus.Idle

        self._cfg = BackgroundDebugToolsConfig()

        self._ast_cache = PythonSourceASTCache()

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

    async def enter_breakpoint(self, frame: FrameType, event: threading.Event):
        """should only be called in main thread (server runs in background thread)"""
        if self._cfg.skip_breakpoint:
            event.set()
            return
        assert self._frame is None, "already in breakpoint, shouldn't happen"
        assert prim.is_loopback_call(
        ), "this function should only be called in main thread"
        self._frame = frame
        self._event = event
        obj, app = prim.get_service(
            app_serv_names.REMOTE_COMP_GET_LAYOUT_ROOT_BY_KEY)(
                TENSORPC_DBG_FRAME_INSPECTOR_KEY)
        assert isinstance(obj, BreakpointDebugPanel)
        with enter_app_context(app):
            await obj.set_breakpoint_frame_meta(frame, self.leave_breakpoint)
        self._cur_status = DebugServerStatus.InsideBreakpoint

    async def leave_breakpoint(self):
        """should only be called from remote"""
        assert not prim.is_loopback_call(
        ), "this function should only be called from remote"
        if self._event is not None:
            self._event.set()
            self._event = None
        self._frame = None
        obj, app = prim.get_service(
            app_serv_names.REMOTE_COMP_GET_LAYOUT_ROOT_BY_KEY)(
                TENSORPC_DBG_FRAME_INSPECTOR_KEY)
        assert isinstance(obj, BreakpointDebugPanel)
        with enter_app_context(app):
            await obj.leave_breakpoint()
        self._cur_status = DebugServerStatus.Idle

    def bkgd_get_cur_frame(self):
        return self._frame

    def get_cur_frame_meta(self):
        if self._frame is not None:
            qname = inspecttools.get_co_qualname_from_frame(self._frame)
            return DebugFrameMeta(self._frame.f_code.co_name, qname,
                                  self._frame.f_code.co_filename,
                                  self._frame.f_lineno)
        return None

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

    async def handle_code_selection_msg(self, code_segment: str, path: str, code_range: Tuple[int, int, int, int]):
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
        res = find_toplevel_func_node_container_by_lineno(tree, code_range[0])
        # print(res)
        if res is not None:
            node_qname = ".".join([n.name for n in res])
            cur_frame: Optional[FrameType] = self._frame
            with enter_app_context(app):
                while cur_frame is not None:
                    if Path(cur_frame.f_code.co_filename).resolve() == Path(path).resolve():
                        qname = inspecttools.get_co_qualname_from_frame(cur_frame)
                        # print(qname, node_qname)
                        if node_qname == qname:
                            # found. eval expr in this frame
                            try:
                                local_vars = cur_frame.f_locals
                                global_vars = cur_frame.f_globals
                                res = eval(code_segment, global_vars, local_vars)
                                await obj.tree_viewer.set_external_preview_layout(res, header=code_segment)
                            except Exception as e:
                                traceback.print_exc()
                                await obj.send_exception(e)
                                return 
                    cur_frame = cur_frame.f_back
