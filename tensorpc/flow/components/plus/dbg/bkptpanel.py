import threading
from types import FrameType
from typing import Any, Callable, Coroutine, Dict, Optional
from tensorpc.core import inspecttools
from tensorpc.flow.components import mui
from tensorpc.flow.components.plus.objinspect.tree import BasicObjectTree
from tensorpc.flow import marker
from tensorpc.flow.components.plus.styles import CodeStyles

from collections.abc import Mapping

class BreakpointDebugPanel(mui.FlexBox):
    def __init__(self):
        self.header = mui.Typography("").prop(variant="body1", 
            fontFamily=CodeStyles.fontFamily)

        self.pause_run_btn = mui.IconButton(mui.IconType.Pause, self._continue_bkpt).prop(size="small", iconFontSize="18px")
        self.header_actions = mui.HBox([
            self.pause_run_btn,
        ])
        self.header_actions.prop(flex=1, justifyContent="flex-end", paddingRight="4px", alignItems="center")
        self.header_container = mui.HBox([
            self.header,
            self.header_actions,
        ]).prop(paddingLeft="4px", alignItems="center", )
        self.tree_viewer = BasicObjectTree()
        self.content_container = mui.VBox([
            self.tree_viewer.prop(flex=1),
        ]).prop(flex=1)
        self._drawer = mui.Collapse([
            mui.VBox([
                mui.IconButton(mui.IconType.ChevronLeft, self._close_drawer).prop(size="small", iconFontSize="18px"),
                mui.Divider(),
                mui.Markdown("Hello Drawer!"),
            ]).prop(width="200px", flexShrink=0, alignItems="flex-end")
        ]).prop(triggered=False, orientation="horizontal")
        super().__init__([
            self._drawer,
            mui.Divider(orientation="vertical"),
            mui.VBox([
                mui.IconButton(mui.IconType.Menu, self._open_drawer).prop(size="small", iconFontSize="18px"),
                mui.Divider(),
                mui.IconButton(mui.IconType.Menu).prop(size="small", iconFontSize="18px"),
            ]),
            mui.Divider(orientation="vertical"),
            mui.VBox([
                self.header_container,
                mui.Divider(),
                self.content_container,
            ]).prop(flex=1),
        ])
        self.prop(flexDirection="row")
        self._cur_leave_bkpt_cb: Optional[Callable[[], Coroutine[None, None, Any]]] = None

    async def _open_drawer(self):
        await self.send_and_wait(self._drawer.update_event(triggered=True))

    async def _close_drawer(self):
        await self.send_and_wait(self._drawer.update_event(triggered=False))


    async def _continue_bkpt(self):
        if self._cur_leave_bkpt_cb is not None:
            await self._cur_leave_bkpt_cb()
            self._cur_leave_bkpt_cb = None
            await self.leave_breakpoint()

    async def set_breakpoint_frame_meta(self, frame: FrameType, leave_bkpt_cb: Callable[[], Coroutine[None, None, Any]]):
        frame_func_name = inspecttools.get_co_qualname_from_frame(frame)
        self._cur_leave_bkpt_cb = leave_bkpt_cb
        local_vars_for_inspect = self._get_filtered_local_vars(frame)
        await self.tree_viewer.set_object_dict(local_vars_for_inspect)
        await self.header.write(frame_func_name)
        await self.pause_run_btn.send_and_wait(self.pause_run_btn.update_event(icon=mui.IconType.PlayArrow))

    async def leave_breakpoint(self):
        await self.pause_run_btn.send_and_wait(self.pause_run_btn.update_event(icon=mui.IconType.Pause))
        await self.header.write("")
        await self.tree_viewer.set_object_dict({})

    def _get_filtered_local_vars(self, frame: FrameType):
        local_vars = frame.f_locals.copy()
        local_vars = inspecttools.filter_local_vars(local_vars)
        return local_vars
