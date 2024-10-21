import threading
from types import FrameType
from typing import Any, Dict, Optional
from tensorpc.core import inspecttools
from tensorpc import prim 
from tensorpc.dbg.constants import TENSORPC_DBG_FRAME_INSPECTOR_KEY
from tensorpc.dbg.serv_names import serv_names
from tensorpc.flow.components.plus.dbg.bkptpanel import BreakpointDebugPanel
from tensorpc.flow.components.plus.objinspect.tree import BasicObjectTree
from tensorpc.flow.serv_names import serv_names as app_serv_names

class BackgroundDebugTools:
    def __init__(self) -> None:
        self._frame = None
        self._event: Optional[threading.Event] = None

    async def enter_breakpoint(self, frame: FrameType, event: threading.Event):
        """should only be called in main thread (server runs in background thread)"""
        assert self._frame is None, "already in breakpoint, shouldn't happen"
        assert prim.is_loopback_call(), "this function should only be called in main thread"
        self._frame = frame
        self._event = event
        obj = prim.get_service(app_serv_names.REMOTE_COMP_GET_LAYOUT_ROOT_BY_KEY)(TENSORPC_DBG_FRAME_INSPECTOR_KEY)
        assert isinstance(obj, BreakpointDebugPanel)
        await obj.set_breakpoint_frame_meta(frame, self.leave_breakpoint)

    async def leave_breakpoint(self):
        """should only be called from remote"""
        assert not prim.is_loopback_call(), "this function should only be called from remote"
        if self._event is not None:
            self._event.set()
            self._event = None
        self._frame = None 
        obj = prim.get_service(app_serv_names.REMOTE_COMP_GET_LAYOUT_ROOT_BY_KEY)(TENSORPC_DBG_FRAME_INSPECTOR_KEY)
        assert isinstance(obj, BreakpointDebugPanel)
        await obj.leave_breakpoint()

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
