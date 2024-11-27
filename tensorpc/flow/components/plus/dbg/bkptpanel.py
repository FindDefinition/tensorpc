import ast
import asyncio
import dataclasses
import enum
from time import sleep
from types import FrameType
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union
from tensorpc.constants import TENSORPC_BG_PROCESS_NAME_PREFIX
from tensorpc.core import inspecttools
from tensorpc.dbg.core.frame_id import get_frame_uid
from tensorpc.flow import appctx
from tensorpc.flow.components import chart, mui
from tensorpc.flow.components.plus.config import ConfigDialogEvent, ConfigPanelDialog
from tensorpc.flow.components.plus.dbg.frameobj import FrameObjectPreview
from tensorpc.flow.components.plus.dbg.perfetto_utils import zip_trace_result
from tensorpc.flow.components.plus.objinspect.tree import BasicObjectTree
from tensorpc.flow.components.plus.scriptmgr import ScriptManager
from tensorpc.flow.components.plus.styles import CodeStyles
from tensorpc.flow.components.plus.objinspect.inspector import ObjectInspector
from tensorpc.dbg.constants import BackgroundDebugToolsConfig, DebugFrameInfo, DebugFrameState, RecordMode, TracerConfig, TracerSingleResult, TracerUIConfig
from tensorpc.utils.loader import FrameModuleMeta
from .framescript import FrameScript


class DebugActions(enum.Enum):
    RECORD_TO_NEXT_SAME_BKPT = "Record To Same Breakpoint"
    RECORD_CUSTOM = "Launch Custom Record"


_DEFAULT_BKPT_CNT_FOR_SAME_BKPT = 10


class BreakpointDebugPanel(mui.FlexBox):

    def __init__(self):
        self.header = mui.Typography("").prop(variant="caption",
                                              fontFamily=CodeStyles.fontFamily)

        self.continue_btn = mui.IconButton(mui.IconType.PlayArrow,
                                           self._continue_bkpt).prop(
                                               size="small",
                                               iconFontSize="18px")
        self.skip_bkpt_run_btn = mui.IconButton(mui.IconType.DoubleArrow,
                                                self._skip_further_bkpt).prop(
                                                    size="small",
                                                    iconFontSize="18px")

        self.copy_path_btn = mui.IconButton(mui.IconType.ContentCopy,
                                            self._copy_frame_path_lineno)
        self.copy_path_btn.prop(size="small",
                                iconFontSize="18px",
                                disabled=True,
                                tooltip="Copy Frame Path:Lineno")

        self.record_btn = mui.IconButton(
            mui.IconType.FiberManualRecord,
            self._continue_bkpt_and_start_record).prop(size="small",
                                                       iconFontSize="18px",
                                                       muiColor="primary")
        self._header_more_menu = mui.MenuList([
            mui.MenuItem(id=DebugActions.RECORD_TO_NEXT_SAME_BKPT.value,
                         label=DebugActions.RECORD_TO_NEXT_SAME_BKPT.value),
            mui.MenuItem(id=DebugActions.RECORD_CUSTOM.value,
                         label=DebugActions.RECORD_CUSTOM.value),
        ],
                                              mui.IconButton(
                                                  mui.IconType.MoreVert).prop(
                                                      size="small",
                                                      iconFontSize="18px"))
        self._header_more_menu.prop(anchorOrigin=mui.Anchor("top", "right"))
        self._header_more_menu.event_contextmenu_select.on(
            self._handle_debug_more_actions)

        self.header_actions = mui.HBox([
            self.continue_btn,
            self.record_btn,
            self.skip_bkpt_run_btn,
            self.copy_path_btn,
            self._header_more_menu,
        ])
        self._all_frame_select = mui.Autocomplete("stack", [],
                                                  self._select_frame)
        self._all_frame_select.prop(size="small",
                                    textFieldProps=mui.TextFieldProps(
                                        muiMargin="dense",
                                        fontFamily=CodeStyles.fontFamily),
                                    padding="0 3px 0 3px")
        self._trace_launch_dialog = ConfigPanelDialog(
            self._on_trace_launch).prop(okLabel="Launch Record")

        self.header_actions.prop(flex=1,
                                 justifyContent="flex-end",
                                 paddingRight="4px",
                                 alignItems="center")
        self.frame_script = FrameScript()
        self._perfetto = chart.Perfetto().prop(width="100%", height="100%")
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
            mui.TabDef("",
                       "3",
                       self._perfetto,
                       icon=mui.IconType.Timeline,
                       tooltip="perfetto"),
        ]
        self._frame_obj_preview = FrameObjectPreview()

        self._frame_obj_preview.prop(width="100%",
                                     height="100%",
                                     overflow="hidden")
        self.tree_viewer = ObjectInspector(
            show_terminal=False,
            default_sizes=[100, 100],
            with_builtins=False,
            custom_tabs=custom_tabs,
            custom_preview=self._frame_obj_preview)
        if isinstance(self.tree_viewer.tree.tree, mui.TanstackJsonLikeTree):
            self.tree_viewer.tree.tree.prop(maxLeafRowFilterDepth=0,
                                            filterNodeValue=True)

        filter_input = mui.TextField("filter").prop(
            valueChangeTarget=(self.tree_viewer.tree.tree, "globalFilter"))
        tree = self.tree_viewer.tree.tree
        if isinstance(tree, mui.TanstackJsonLikeTree):
            filter_input.event_change.on(
                lambda val: tree.prop(globalFilter=val))

        self.header_container = mui.HBox([
            filter_input.prop(flex=1),
            self._all_frame_select.prop(flex=2),
            self.header.prop(flex=4),
            self.header_actions,
        ]).prop(
            paddingLeft="4px",
            alignItems="center",
        )

        self.content_container = mui.VBox([
            self.tree_viewer.prop(flex=1),
        ]).prop(flex=1)
        super().__init__([
            self.header_container,
            mui.Divider(),
            self.content_container,
            self._trace_launch_dialog,
        ])
        self.prop(flexDirection="column")
        self._cur_leave_bkpt_cb: Optional[Callable[[Optional[TracerConfig]],
                                                   Coroutine[None, None,
                                                             Any]]] = None

        self._cur_frame_meta: Optional[DebugFrameInfo] = None
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
                    await self.send_and_wait(
                        self.skip_bkpt_run_btn.update_event(
                            icon=mui.IconType.Pause))
                else:
                    await self.send_and_wait(
                        self.skip_bkpt_run_btn.update_event(
                            icon=mui.IconType.DoubleArrow))

    async def _continue_bkpt(self):
        if self._cur_leave_bkpt_cb is not None:
            await self._cur_leave_bkpt_cb(TracerConfig(enable=False))
            self._cur_leave_bkpt_cb = None
            # await self.leave_breakpoint()

    async def _continue_bkpt_and_start_record(self):
        if self._cur_leave_bkpt_cb is not None:
            await self._cur_leave_bkpt_cb(TracerConfig(enable=True))
            self._cur_leave_bkpt_cb = None
            # await self.leave_breakpoint()

    async def _handle_debug_more_actions(self, value: str):
        if self._cur_leave_bkpt_cb is not None:
            if value == DebugActions.RECORD_TO_NEXT_SAME_BKPT.value:
                await self._cur_leave_bkpt_cb(
                    TracerConfig(
                        enable=True,
                        mode=RecordMode.SAME_BREAKPOINT,
                        breakpoint_count=_DEFAULT_BKPT_CNT_FOR_SAME_BKPT))
                self._cur_leave_bkpt_cb = None
            elif value == DebugActions.RECORD_CUSTOM.value:
                await self._trace_launch_dialog.open_config_dialog(
                    TracerUIConfig())

    async def _on_trace_launch(self, cfg_ev: ConfigDialogEvent[TracerUIConfig]):
        config = cfg_ev.cfg
        if self._cur_leave_bkpt_cb is not None:
            await self._cur_leave_bkpt_cb(
                TracerConfig(enable=True,
                             mode=config.mode,
                             breakpoint_count=config.breakpoint_count,
                             trace_name=config.trace_name,
                             max_stack_depth=config.max_stack_depth))
            self._cur_leave_bkpt_cb = None

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
        await self.tree_viewer.tree.set_root_object_dict(
            local_vars_for_inspect)
        await self.header.write(f"{frame_func_name}({frame.f_lineno})")
        await self.frame_script.mount_frame(
            dataclasses.replace(self._cur_frame_state, frame=frame))

    async def set_breakpoint_frame_meta(
            self,
            frame: FrameType,
            leave_bkpt_cb: Callable[[Optional[TracerConfig]],
                                    Coroutine[None, None, Any]],
            is_record_stop: bool = False):
        qname = inspecttools.get_co_qualname_from_frame(frame)
        self._cur_frame_meta = DebugFrameInfo(frame.f_code.co_name, qname,
                                              frame.f_code.co_filename,
                                              frame.f_lineno)
        self._cur_frame_state.frame = frame
        self._cur_leave_bkpt_cb = leave_bkpt_cb
        ev = self.copy_path_btn.update_event(disabled=False)
        if is_record_stop:
            ev += self.record_btn.update_event(disabled=False,
                                               muiColor="primary")
        await self.send_and_wait(ev)
        cur_frame = frame
        frame_select_opts = []
        count = 0
        while cur_frame is not None:
            qname = inspecttools.get_co_qualname_from_frame(cur_frame)
            frame_select_opts.append({"label": qname, "count": count})
            count += 1
            cur_frame = cur_frame.f_back
        await self._all_frame_select.update_options(frame_select_opts, 0)
        await self._set_frame_meta(frame)
        frame_uid, frame_meta = get_frame_uid(frame)
        await self._frame_obj_preview.set_frame_meta(frame_uid,
                                                     frame_meta.qualname)

    async def leave_breakpoint(self, is_record_start: bool = False):
        await self.header.write("")
        await self.tree_viewer.tree.set_root_object_dict({})
        ev = self.copy_path_btn.update_event(disabled=True)
        if is_record_start:
            ev += self.record_btn.update_event(disabled=True,
                                               muiColor="success")
        await self.send_and_wait(ev)

        self._cur_frame_meta = None
        self._cur_frame_state.frame = None
        await self.frame_script.unmount_frame()
        await self._frame_obj_preview.clear_frame_variable()
        await self._frame_obj_preview.clear_preview_layouts()

    def _get_filtered_local_vars(self, frame: FrameType):
        local_vars = frame.f_locals.copy()
        local_vars = inspecttools.filter_local_vars(local_vars)
        return local_vars

    async def set_frame_object(
            self,
            obj: Any,
            expr: str,
            func_node: Optional[Union[ast.FunctionDef,
                                      ast.AsyncFunctionDef]] = None,
            cur_frame: Optional[FrameType] = None):
        if expr.isidentifier() and func_node is not None:
            await self._frame_obj_preview.set_folding_code(
                expr, func_node, cur_frame)
        del cur_frame
        #     await self._frame_obj_preview.set_frame_variable(expr, obj)
        await self._frame_obj_preview.set_user_selection_frame_variable(
            expr, obj)
        # await self.tree_viewer.set_obj_preview_layout(obj, header=expr)

    async def set_perfetto_data(self, trace_res: TracerSingleResult):
        zip_data = zip_trace_result([trace_res.data], [trace_res.external_events])
        await self._perfetto.set_trace_data(zip_data, title="trace")
