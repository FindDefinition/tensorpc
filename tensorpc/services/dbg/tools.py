import ast
import asyncio
import dataclasses
import os
from pathlib import Path
import threading
import traceback
from types import FrameType
from typing import Any, Dict, List, Optional, Tuple

import grpc
import rich
from tensorpc.core import inspecttools, marker
from tensorpc import prim
from tensorpc.core.asyncclient import simple_remote_call_async
from tensorpc.core.funcid import find_toplevel_func_node_by_lineno, find_toplevel_func_node_container_by_lineno
from tensorpc.core.serviceunit import ServiceEventType
from tensorpc.dbg.constants import TENSORPC_DBG_FRAME_INSPECTOR_KEY, TENSORPC_ENV_DBG_DEFAULT_BREAKPOINT_ENABLE, BreakpointType, DebugFrameMeta, DebugServerStatus, BackgroundDebugToolsConfig
from tensorpc.dbg.core.sourcecache import LineCache, PythonSourceASTCache
from tensorpc.dbg.serv_names import serv_names
from tensorpc.flow.client import list_all_app_in_machine
from tensorpc.flow.components.plus.dbg.bkptpanel import BreakpointDebugPanel
from tensorpc.flow.components.plus.objinspect.tree import BasicObjectTree
from tensorpc.flow.core.appcore import enter_app_context
from tensorpc.flow.serv_names import serv_names as app_serv_names
from tensorpc.flow.vscode.coretypes import VscodeBreakpoint
from tensorpc.utils.rich_logging import TENSORPC_LOGGING_OVERRIDED_PATH_LINENO_KEY, get_logger

LOGGER = get_logger("tensorpc.dbg")


@dataclasses.dataclass
class BreakpointMeta:
    name: Optional[str]
    type: BreakpointType
    path: str
    lineno: int
    line_text: str
    event: threading.Event


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

        self._vscode_breakpoints: List[VscodeBreakpoint] = []
        # (path, lineno) -> VscodeBreakpoint
        self._vscode_breakpoints_dict: Dict[Tuple[Path, int],
                                            VscodeBreakpoint] = {}

        self._bkpt_lock = asyncio.Lock()

    # @marker.mark_server_event(event_type=ServiceEventType.BeforeServerStart)
    async def try_fetch_vscode_breakpoints(self):
        all_app_metas = list_all_app_in_machine()
        for meta in all_app_metas:
            url = f"localhost:{meta.app_grpc_port}"
            try:
                bkpts = await simple_remote_call_async(
                    url, app_serv_names.APP_GET_VSCODE_BREAKPOINTS)
                LOGGER.info(f"Fetch vscode breakpoints from App {meta.name}",
                            url)
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
                               event: threading.Event,
                               type: BreakpointType,
                               name: Optional[str] = None):
        """should only be called in main thread (server runs in background thread)"""
        if self._cfg.skip_breakpoint:
            event.set()
            return
        async with self._bkpt_lock:
            assert self._frame is None, "already in breakpoint, shouldn't happen"
            assert prim.is_loopback_call(
            ), "this function should only be called in main thread"
            try:
                lines = self._line_cache.getlines(frame.f_code.co_filename)
                linetext = lines[frame.f_lineno - 1]
            except:
                linetext = ""
            pid = os.getpid()
            self._cur_breakpoint = BreakpointMeta(name, type,
                                                  frame.f_code.co_filename,
                                                  frame.f_lineno, linetext,
                                                  event)
            if self._cur_breakpoint is not None and self._cur_breakpoint.type == BreakpointType.Vscode:
                is_cur_bkpt_is_vscode = self._determine_vscode_bkpt_status(
                    self._cur_breakpoint, self._vscode_breakpoints_dict)
                if not is_cur_bkpt_is_vscode:
                    event.set()
                    LOGGER.warning(
                        f"Skip Vscode breakpoint",
                        extra={
                            TENSORPC_LOGGING_OVERRIDED_PATH_LINENO_KEY:
                            (frame.f_code.co_filename, frame.f_lineno)
                        })
                    self._cur_breakpoint = None
                    return
            self._frame = frame
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
                                                    self.leave_breakpoint)
            self._cur_status = DebugServerStatus.InsideBreakpoint

    async def leave_breakpoint(self):
        """should only be called from remote"""
        assert not prim.is_loopback_call(
        ), "this function should only be called from remote"
        async with self._bkpt_lock:
            if self._cur_breakpoint is not None:
                self._cur_breakpoint.event.set()
                self._cur_breakpoint = None
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

    def _determine_vscode_bkpt_status(
            self, bkpt_meta: BreakpointMeta,
            vscode_bkpt_dict: Dict[Tuple[Path, int], VscodeBreakpoint]):
        if bkpt_meta.type == BreakpointType.Vscode:
            key = (Path(bkpt_meta.path).resolve(), bkpt_meta.lineno)
            # rich.print("BKPT", key, vscode_bkpt_dict)
            if key in vscode_bkpt_dict:
                return vscode_bkpt_dict[key].enabled
        return False

    def _set_vscode_breakpoints_and_dict(self, bkpts: List[VscodeBreakpoint]):
        new_bkpts: List[VscodeBreakpoint] = []
        for x in bkpts:
            if x.enabled and x.lineText is not None and (
                    ".breakpoint" in x.lineText
                    or ".vscode_breakpoint" in x.lineText):
                new_bkpts.append(x)
        self._vscode_breakpoints_dict = {
            (Path(x.path).resolve(), x.line + 1): x
            for x in new_bkpts
        }
        self._vscode_breakpoints = new_bkpts

    async def set_vscode_breakpoints(self, bkpts: List[VscodeBreakpoint]):
        self._set_vscode_breakpoints_and_dict(bkpts)
        if self._cur_breakpoint is not None and self._cur_breakpoint.type == BreakpointType.Vscode:
            is_cur_bkpt_is_vscode = self._determine_vscode_bkpt_status(
                self._cur_breakpoint, self._vscode_breakpoints_dict)
            # if not found, release this breakpoint
            if not is_cur_bkpt_is_vscode:
                await self.leave_breakpoint()

    async def set_vscode_breakpoints_and_get_cur_meta(
            self, bkpts: List[VscodeBreakpoint]):
        meta = self.get_cur_frame_meta()
        await self.set_vscode_breakpoints(bkpts)
        return meta

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
        res = find_toplevel_func_node_container_by_lineno(tree, code_range[0])
        # print(res)
        if res is not None:
            node_qname = ".".join([n.name for n in res])
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
                                await obj.tree_viewer.set_external_preview_layout(
                                    res, header=code_segment)
                            except grpc.aio.AioRpcError as e:
                                return
                            except Exception as e:
                                print(e)
                                # traceback.print_exc()
                                # await obj.send_exception(e)
                                return
                    cur_frame = cur_frame.f_back
