import asyncio
import dataclasses
from time import sleep
from types import FrameType
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union
from tensorpc.constants import TENSORPC_BG_PROCESS_NAME_PREFIX
from tensorpc.core import inspecttools
from tensorpc.flow import appctx
from tensorpc.flow.components import mui
from tensorpc.flow.components.plus.objinspect.tree import BasicObjectTree
from tensorpc.flow.components.plus.styles import CodeStyles
from tensorpc.flow.components.plus.objinspect.inspector import ObjectInspector
from tensorpc.dbg.constants import BackgroundDebugToolsConfig, DebugFrameMeta

class BreakpointDebugPanel(mui.FlexBox):
    def __init__(self):
        self.header = mui.Typography("").prop(variant="body1", 
            fontFamily=CodeStyles.fontFamily)

        self.continue_btn = mui.IconButton(mui.IconType.PlayArrow, self._continue_bkpt).prop(size="small", iconFontSize="18px")
        self.skip_bkpt_run_btn = mui.IconButton(mui.IconType.DoubleArrow, self._skip_further_bkpt).prop(size="small", iconFontSize="18px")

        self.copy_path_btn = mui.IconButton(mui.IconType.ContentCopy, self._copy_frame_path_lineno)
        self.copy_path_btn.prop(size="small", iconFontSize="18px", disabled=True, tooltip="Copy Frame Path:Lineno")

        self.header_actions = mui.HBox([
            self.continue_btn,
            self.skip_bkpt_run_btn,
            self.copy_path_btn,
        ])
        self.header_actions.prop(flex=1, justifyContent="flex-end", paddingRight="4px", alignItems="center")
        self.header_container = mui.HBox([
            self.header,
            self.header_actions,
        ]).prop(paddingLeft="4px", alignItems="center", )
        self.tree_viewer = ObjectInspector(show_terminal=False, default_sizes=[100, 100])
        self.content_container = mui.VBox([
            self.tree_viewer.prop(flex=1),
        ]).prop(flex=1)
        super().__init__([
            self.header_container,
            mui.Divider(),
            self.content_container,
        ])
        self.prop(flexDirection="column")
        self._cur_leave_bkpt_cb: Optional[Callable[[], Coroutine[None, None, Any]]] = None

        self._cur_frame_meta: Optional[DebugFrameMeta] = None 

        self._bkgd_debug_tool_cfg: Optional[BackgroundDebugToolsConfig] = None

    async def _copy_frame_path_lineno(self):
        if self._cur_frame_meta is not None:
            path_lineno = f"{self._cur_frame_meta.path}:{self._cur_frame_meta.lineno}"
            await appctx.copy_text_to_clipboard(path_lineno)

    async def _skip_further_bkpt(self, skip: Optional[bool] = None):
        await self._continue_bkpt()
        if self._bkgd_debug_tool_cfg is not None:
            prev_skip = self._bkgd_debug_tool_cfg.skip_breakpoint
            target_skip = not self._bkgd_debug_tool_cfg.skip_breakpoint
            if skip is not None:
                target_skip = skip
            if prev_skip != target_skip:
                self._bkgd_debug_tool_cfg.skip_breakpoint = target_skip
                if target_skip:
                    await self.send_and_wait(self.skip_bkpt_run_btn.update_event(icon=mui.IconType.Pause))
                else:
                    await self.send_and_wait(self.skip_bkpt_run_btn.update_event(icon=mui.IconType.DoubleArrow))

    async def _continue_bkpt(self):
        if self._cur_leave_bkpt_cb is not None:
            await self._cur_leave_bkpt_cb()
            self._cur_leave_bkpt_cb = None
            await self.leave_breakpoint()

    async def set_breakpoint_frame_meta(self, frame: FrameType, leave_bkpt_cb: Callable[[], Coroutine[None, None, Any]]):
        frame_func_name = inspecttools.get_co_qualname_from_frame(frame)
        self._cur_leave_bkpt_cb = leave_bkpt_cb
        local_vars_for_inspect = self._get_filtered_local_vars(frame)
        await self.tree_viewer.tree.set_object_dict(local_vars_for_inspect)
        await self.header.write(frame_func_name)
        # await self.continue_btn.send_and_wait(self.continue_btn.update_event(icon=mui.IconType.PlayArrow))
        await self.copy_path_btn.send_and_wait(self.copy_path_btn.update_event(disabled=False))
        qname = inspecttools.get_co_qualname_from_frame(frame)

        self._cur_frame_meta = DebugFrameMeta(frame.f_code.co_name, qname, frame.f_code.co_filename, frame.f_lineno)

    async def leave_breakpoint(self):
        # await self.continue_btn.send_and_wait(self.continue_btn.update_event(icon=mui.IconType.PlayArrow))
        await self.header.write("")
        await self.tree_viewer.tree.set_object_dict({})
        await self.copy_path_btn.send_and_wait(self.copy_path_btn.update_event(disabled=True))

        self._cur_frame_meta = None

    def _get_filtered_local_vars(self, frame: FrameType):
        local_vars = frame.f_locals.copy()
        local_vars = inspecttools.filter_local_vars(local_vars)
        return local_vars
