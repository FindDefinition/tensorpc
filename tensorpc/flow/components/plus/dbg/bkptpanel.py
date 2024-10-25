import asyncio
import dataclasses
from time import sleep
from types import FrameType
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union
from tensorpc.constants import TENSORPC_BG_PROCESS_NAME_PREFIX
from tensorpc.core import inspecttools
from tensorpc.dbg.core.frame_id import get_frame_uid
from tensorpc.flow import appctx
from tensorpc.flow.components import mui
from tensorpc.flow.components.plus.dbg.frameobj import FrameObjectPreview
from tensorpc.flow.components.plus.objinspect.tree import BasicObjectTree
from tensorpc.flow.components.plus.scriptmgr import ScriptManager
from tensorpc.flow.components.plus.styles import CodeStyles
from tensorpc.flow.components.plus.objinspect.inspector import ObjectInspector
from tensorpc.dbg.constants import BackgroundDebugToolsConfig, DebugFrameMeta, DebugFrameState
from tensorpc.utils.loader import FrameModuleMeta
from .framescript import FrameScript

class BreakpointDebugPanel(mui.FlexBox):
    def __init__(self):
        self.header = mui.Typography("").prop(variant="caption", 
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
        self._all_frame_select = mui.Autocomplete("stack", [], self._select_frame)
        self._all_frame_select.prop(size="small", textFieldProps=mui.TextFieldProps(muiMargin="dense", fontFamily=CodeStyles.fontFamily), padding="0 3px 0 3px")
        self.header_actions.prop(flex=1, justifyContent="flex-end", paddingRight="4px", alignItems="center")
        self.header_container = mui.HBox([
            self._all_frame_select.prop(flex=1),
            self.header.prop(flex=2),
            self.header_actions,
        ]).prop(paddingLeft="4px", alignItems="center", )
        self.frame_script = FrameScript()
        custom_tabs = [
            mui.TabDef("",
                "1",
                ScriptManager(),
                icon=mui.IconType.Code,
                tooltip="script manager"),
            mui.TabDef("",
                "2",
                self.frame_script,
                icon=mui.IconType.DataArray,
                tooltip="frame script manager"),
        ]
        self._frame_obj_preview = FrameObjectPreview()
        self._frame_obj_preview.prop(width="100%", height="100%", overflow="hidden")
        self.tree_viewer = ObjectInspector(show_terminal=False, default_sizes=[100, 100], with_builtins=False, custom_tabs=custom_tabs,
            custom_preview=self._frame_obj_preview)
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
        self._cur_frame_state: DebugFrameState = DebugFrameState(None) 

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

    async def _select_frame(self, option: Dict[str, Any]):
        if self._cur_frame_state.frame is None:
            return
        cur_frame = self._cur_frame_state.frame
        count = option["count"]
        while count > 0:
            assert cur_frame is not None 
            cur_frame = cur_frame.f_back
            count -= 1
        assert cur_frame is not None 
        await self._set_frame_meta(cur_frame)

    async def _set_frame_meta(self, frame: FrameType):
        frame_func_name = inspecttools.get_co_qualname_from_frame(frame)
        local_vars_for_inspect = self._get_filtered_local_vars(frame)
        await self.tree_viewer.tree.update_root_object_dict(local_vars_for_inspect, keep_old=False)
        await self.header.write(f"{frame_func_name}({frame.f_lineno})")
        await self.frame_script.mount_frame(dataclasses.replace(self._cur_frame_state, frame=frame))

    async def set_breakpoint_frame_meta(self, frame: FrameType, leave_bkpt_cb: Callable[[], Coroutine[None, None, Any]]):
        qname = inspecttools.get_co_qualname_from_frame(frame)
        self._cur_frame_meta = DebugFrameMeta(frame.f_code.co_name, qname, frame.f_code.co_filename, frame.f_lineno)
        self._cur_frame_state.frame = frame
        self._cur_leave_bkpt_cb = leave_bkpt_cb
        await self.copy_path_btn.send_and_wait(self.copy_path_btn.update_event(disabled=False))
        
        cur_frame = frame
        frame_select_opts = []
        count = 0
        while cur_frame is not None:
            qname = inspecttools.get_co_qualname_from_frame(cur_frame)
            frame_select_opts.append({
                "label": qname,
                "count": count
            })
            count += 1
            cur_frame = cur_frame.f_back
        await self._all_frame_select.update_options(frame_select_opts, 0)
        await self._set_frame_meta(frame)
        frame_uid, frame_meta = get_frame_uid(frame)
        await self._frame_obj_preview.set_frame_meta(frame_uid, frame_meta.qualname)

    async def leave_breakpoint(self):
        await self.header.write("")
        await self.tree_viewer.tree.update_root_object_dict({}, keep_old=False)
        await self.copy_path_btn.send_and_wait(self.copy_path_btn.update_event(disabled=True))

        self._cur_frame_meta = None
        self._cur_frame_state.frame = None
        await self.frame_script.unmount_frame()
        await self._frame_obj_preview.clear_frame_variable()

    def _get_filtered_local_vars(self, frame: FrameType):
        local_vars = frame.f_locals.copy()
        local_vars = inspecttools.filter_local_vars(local_vars)
        return local_vars

    async def set_frame_object(self, obj: Any, expr: str):
        if expr.isidentifier():
            await self._frame_obj_preview.set_frame_variable(expr, obj)
        await self.tree_viewer.set_obj_preview_layout(
            obj, header=expr)

