import ast
import asyncio
import dataclasses
import enum
from functools import partial
import os
from pathlib import Path
from time import sleep
from types import FrameType
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union
from tensorpc.apps.distssh.constants import TENSORPC_ENV_DISTSSH_WORKDIR
from tensorpc.constants import TENSORPC_BG_PROCESS_NAME_PREFIX
from tensorpc.core import inspecttools
from tensorpc.apps.dbg.core.frame_id import get_frame_uid
from tensorpc.core.datamodel.draftstore import DraftSimpleFileStoreBackend
from tensorpc.core.datamodel.events import DraftChangeEvent
from tensorpc.dock import appctx
from tensorpc.dock.components import chart, mui
from tensorpc.dock.components.plus.config import ConfigDialogEvent, ConfigPanelDialog
from tensorpc.apps.dbg.components.frame_obj_v2 import FrameObjectPreview
from tensorpc.apps.dbg.components.perfetto_utils import zip_trace_result
from tensorpc.dock.components.plus.objinspect.tree import BasicObjectTree
from tensorpc.dock.components.plus.scriptmgr import ScriptManager, ScriptManagerV2
from tensorpc.dock.components.plus.styles import CodeStyles
from tensorpc.dock.components.plus.objinspect.inspector import ObjectInspector
from tensorpc.apps.dbg.constants import BackgroundDebugToolsConfig, DebugFrameInfo, DebugFrameState, RecordMode, TracerConfig, TracerSingleResult, TracerUIConfig
from tensorpc.dock.core.appcore import AppSpecialEventType
from tensorpc.utils.loader import FrameModuleMeta
from .framescript import FrameScript
from tensorpc.apps.dbg.model import PyDbgModel, TracerState
import tensorpc.core.datamodel as D
from tensorpc.dock import marker
from tensorpc.constants import TENSORPC_FILE_NAME_PREFIX
from tensorpc.apps.dbg.constants import RemoteDebugEventType

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
        self._all_frame_select = mui.Autocomplete("stack", [])
        self._all_frame_select.prop(size="small",
                                    textFieldProps=mui.TextFieldProps(
                                        muiMargin="dense",
                                        variant="outlined"),
                                    padding="0 3px 0 3px")
        self._trace_launch_dialog = ConfigPanelDialog(
            self._on_trace_launch).prop(okLabel="Launch Record")

        self.header_actions.prop(flex=1,
                                 justifyContent="flex-end",
                                 paddingRight="4px",
                                 alignItems="center")
        self.header_actions_may_disable = mui.MatchCase.binary_selection(True, self.header_actions)
        self.frame_script_container = mui.VBox([]).prop(width="100%", height="100%", overflow="hidden")
        self._perfetto = chart.Perfetto().prop(width="100%", height="100%")
        custom_tabs = [
            mui.TabDef("",
                       "2",
                       self.frame_script_container,
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
            default_sizes=[60, 100],
            with_builtins=False,
            custom_tabs=custom_tabs,
            custom_preview=self._frame_obj_preview,
            horizontal=True)
        if isinstance(self.tree_viewer.tree.tree, mui.TanstackJsonLikeTree):
            self.tree_viewer.tree.tree.prop(# maxLeafRowFilterDepth=0,
                                            globalFilterContiguousOnly=True,
                                            filterNameTypeValue=True,
                                            rowFilterMatchProps=mui.FlexBoxProps(backgroundColor="beige"),
                                            filterFromLeafRows=True,
                                            filterFQNAsName=True)

        filter_input = mui.TextField("filter").prop(
            valueChangeTarget=(self.tree_viewer.tree.tree, "globalFilter"))
        filter_input.prop(variant="outlined", debounce=200)
        tree = self.tree_viewer.tree.tree
        if isinstance(tree, mui.TanstackJsonLikeTree):
            filter_input.event_change.on(
                lambda val: tree.prop(globalFilter=val))

        self.header_container = mui.HBox([
            filter_input.prop(flex=1),
            self._all_frame_select.prop(flex=2),
            self.header.prop(flex=4),
            self.header_actions_may_disable,
        ]).prop(
            paddingLeft="4px",
            alignItems="center",
        )

        self.content_container = mui.VBox([
            self.tree_viewer.prop(flex=1),
        ]).prop(flex=1)
        textfield_theme = mui.Theme(
            components={
                "MuiInputLabel": {
                    "defaultProps": {
                        "sx": {
                            "fontSize": "14px",
                            "fontFamily": CodeStyles.fontFamily,
                        },
                    },
                },
                "MuiOutlinedInput": {
                    "defaultProps": {
                        "sx": {
                            "fontSize": "14px",
                            "fontFamily": CodeStyles.fontFamily,
                        }
                    }
                },
                "MuiInput": {
                    "defaultProps": {
                        "sx": {
                            "fontSize": "14px",
                            "fontFamily": CodeStyles.fontFamily,
                        }
                    }
                }

            }
        )
        self.dm = mui.DataModel(PyDbgModel(TracerState(None, {})), [
            mui.ThemeProvider([
                self.header_container
            ], textfield_theme),
            mui.Divider(),
            self.content_container,
            self._trace_launch_dialog,
        ])

        draft = self.dm.get_draft_type_only()
        self.header_actions_may_disable.bind_fields(condition=D.not_null(not draft.bkpt.is_external, False))

        self.header.bind_fields(value=D.not_null(D.literal_val("%s(%s)") % (draft.bkpt.selected_frame_item["qualname"], draft.bkpt.selected_frame_item["lineno"]), ""))
        self._all_frame_select.bind_fields(options=D.not_null(draft.bkpt.frame_select_items, []))
        self._all_frame_select.bind_draft_change(draft.bkpt.selected_frame_item)
        self.dm.install_draft_change_handler(draft.bkpt.selected_frame_item, 
            self._handle_selected_frame_change, installed_comp=self.header_actions, user_eval_vars={
                "frame": draft.bkpt.frame
            })
        self.record_btn.bind_fields(
            disabled=draft.bkpt == None,
            muiColor=D.where(draft.bkpt == None, "success", "primary"),
        )
        self.copy_path_btn.bind_fields(
            disabled=draft.bkpt == None)
        super().__init__([
            self.dm,
        ])
        self.prop(flexDirection="column")
        self._cur_leave_bkpt_cb: Optional[Callable[[Optional[TracerConfig]],
                                                   Coroutine[None, None,
                                                             Any]]] = None

        self._cur_frame_meta: Optional[DebugFrameInfo] = None
        self._cur_frame_state: DebugFrameState = DebugFrameState(None)

        self._bkgd_debug_tool_cfg: Optional[BackgroundDebugToolsConfig] = None
    
        self._cur_frame_script: Optional[ScriptManagerV2] = None

        self._is_remote_mounted = False

    @marker.mark_did_mount
    async def _on_mount(self):
        appctx.use_app_special_event_handler(self, AppSpecialEventType.RemoteCompMount, self._frame_script_remote_comp_mount)
        appctx.use_app_special_event_handler(self, AppSpecialEventType.RemoteCompUnmount, self._frame_script_remote_comp_unmount)

    
    async def _frame_script_remote_comp_unmount(self, ev):
        self._is_remote_mounted = False
        await self.frame_script_container.set_new_layout([])

    async def _on_run_cur_script_distributed(self, script_mgr: ScriptManagerV2):
        if script_mgr.dm.model.cur_script_idx != -1:
            cur_script = script_mgr.dm.model.scripts[script_mgr.dm.model.cur_script_idx]
            cur_lang = cur_script.language
            code = cur_script.states[cur_lang].code
            await self.put_app_event(
                self.create_remote_comp_event(
                    RemoteDebugEventType.DIST_RUN_SCRIPT.value,
                    code))

    def _get_frame_script_from_frame(self, frame: FrameType, offset: int):
        cur_frame: Optional[FrameType] = frame
        count = offset
        while count > 0:
            assert cur_frame is not None
            cur_frame = cur_frame.f_back
            count -= 1
        assert cur_frame is not None
        frame_uid, frame_meta = get_frame_uid(cur_frame)
        distssh_workdir = os.getenv(TENSORPC_ENV_DISTSSH_WORKDIR)
        btn = mui.IconButton(mui.IconType.PlayArrow).prop(
            progressColor="primary", muiColor="secondary",
            tooltip="Run Distributed")

        if distssh_workdir is not None:
            distssh_workdir_framescript = Path(distssh_workdir) / "framescript"
            fs_backend = DraftSimpleFileStoreBackend(distssh_workdir_framescript)
            script_mgr = ScriptManagerV2(
                init_store_backend=(fs_backend, frame_uid),
                frame=frame,
                editor_path_uid=f"{distssh_workdir}/framescript/{frame_uid}",
                ext_buttons=[
                    btn,
                ],
            )
        else:
            script_mgr = ScriptManagerV2(
                enable_app_backend=False,
                frame=frame,
                editor_path_uid=frame_uid,
                ext_buttons=[
                    btn,
                ],
            )
        btn.event_click.on(partial(self._on_run_cur_script_distributed, script_mgr))
        return script_mgr

    async def _frame_script_remote_comp_mount(self, ev):
        self._is_remote_mounted = True
        if self.dm.model.bkpt is not None:
            frame = self.dm.model.bkpt.frame
            assert frame is not None 
            selected = self.dm.model.bkpt.selected_frame_item
            offset = 0
            if selected is not None:
                offset = selected["offset"]
            await self.frame_script_container.set_new_layout([
                self._get_frame_script_from_frame(frame, offset),
            ])

    async def _handle_selected_frame_change(self, ev: DraftChangeEvent):
        assert ev.user_eval_vars is not None 
        if ev.new_value is not None and ev.user_eval_vars["frame"] is not None:
            frame = ev.user_eval_vars["frame"]
            option = ev.new_value 
            cur_frame: Optional[FrameType] = frame
            count = option["offset"]
            while count > 0:
                assert cur_frame is not None
                cur_frame = cur_frame.f_back
                count -= 1
            assert cur_frame is not None
            await self._set_frame_meta(cur_frame)
            frame_uid, frame_meta = get_frame_uid(frame)
            await self._frame_obj_preview.set_frame_meta(frame_uid,
                                                        frame_meta.qualname)
            if self._is_remote_mounted:
                await self.frame_script_container.set_new_layout([
                    self._get_frame_script_from_frame(cur_frame, 0),
                ])
        else:
            # await self._frame_obj_preview.clear() 
            await self._frame_obj_preview.clear_frame_variable()
            await self._frame_obj_preview.clear_preview_layouts()
            await self.tree_viewer.tree.set_root_object_dict({})
            await self.frame_script_container.set_new_layout([])

    async def _copy_frame_path_lineno(self):
        if self.dm.model.bkpt is not None:
            info = self.dm.model.bkpt.selected_frame_item
            if info is not None:
                path_lineno = f"{info['path']}:{info['lineno']}"
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
        if self.dm.model.bkpt is not None:
            await self.dm.model.bkpt.get_launch_trace_fn()(TracerConfig(enable=False))

    async def _continue_bkpt_and_start_record(self):
        if self.dm.model.bkpt is not None:
            await self.dm.model.bkpt.get_launch_trace_fn()(TracerConfig(enable=True))

    async def _handle_debug_more_actions(self, value: str):
        if self.dm.model.bkpt is not None:
            if value == DebugActions.RECORD_TO_NEXT_SAME_BKPT.value:
                await self.dm.model.bkpt.get_launch_trace_fn()(
                    TracerConfig(
                        enable=True,
                        mode=RecordMode.SAME_BREAKPOINT,
                        breakpoint_count=_DEFAULT_BKPT_CNT_FOR_SAME_BKPT))
            elif value == DebugActions.RECORD_CUSTOM.value:
                await self._trace_launch_dialog.open_config_dialog(
                    TracerUIConfig())

    async def _on_trace_launch(self, cfg_ev: ConfigDialogEvent[TracerUIConfig]):
        config = cfg_ev.cfg
        if self.dm.model.bkpt is not None:
            await self.dm.model.bkpt.get_launch_trace_fn()(
                TracerConfig(enable=True,
                             mode=config.mode,
                             breakpoint_count=config.breakpoint_count,
                             trace_name=config.trace_name,
                             max_stack_depth=config.max_stack_depth))


    async def _set_frame_meta(self, frame: FrameType):
        # frame_func_name = inspecttools.get_co_qualname_from_frame(frame)
        local_vars_for_inspect = self._get_filtered_local_vars(frame)
        await self.tree_viewer.tree.set_root_object_dict(
            local_vars_for_inspect)
        # await self.header.write(f"{frame_func_name}({frame.f_lineno})")
        # await self.frame_script.mount_frame(
        #     dataclasses.replace(self._cur_frame_state, frame=frame))

    # async def set_breakpoint_frame_meta(
    #         self,
    #         frame: FrameType,
    #         leave_bkpt_cb: Callable[[Optional[TracerConfig]],
    #                                 Coroutine[None, None, Any]],
    #         is_record_stop: bool = False):
    #     qname = inspecttools.get_co_qualname_from_frame(frame)
    #     self._cur_frame_meta = DebugFrameInfo(frame.f_code.co_name, qname,
    #                                           frame.f_code.co_filename,
    #                                           frame.f_lineno)
    #     self._cur_frame_state.frame = frame
    #     self._cur_leave_bkpt_cb = leave_bkpt_cb
    #     ev = self.copy_path_btn.update_event(disabled=False)
    #     if is_record_stop:
    #         ev += self.record_btn.update_event(disabled=False,
    #                                            muiColor="primary")
    #     await self.send_and_wait(ev)
    #     cur_frame = frame
    #     frame_select_opts = []
    #     count = 0
    #     while cur_frame is not None:
    #         qname = inspecttools.get_co_qualname_from_frame(cur_frame)
    #         frame_select_opts.append({"label": qname, "count": count})
    #         count += 1
    #         cur_frame = cur_frame.f_back
    #     await self._all_frame_select.update_options(frame_select_opts, 0)
    #     await self._set_frame_meta(frame)
    #     frame_uid, frame_meta = get_frame_uid(frame)
    #     await self._frame_obj_preview.set_frame_meta(frame_uid,
    #                                                  frame_meta.qualname)

    # async def leave_breakpoint(self, is_record_start: bool = False):
    #     await self.header.write("")
    #     await self.tree_viewer.tree.set_root_object_dict({})
    #     ev = self.copy_path_btn.update_event(disabled=True)
    #     if is_record_start:
    #         ev += self.record_btn.update_event(disabled=True,
    #                                            muiColor="success")
    #     await self.send_and_wait(ev)

    #     self._cur_frame_meta = None
    #     self._cur_frame_state.frame = None
    #     # await self.frame_script.unmount_frame()
    #     await self._frame_obj_preview.clear_frame_variable()
    #     await self._frame_obj_preview.clear_preview_layouts()

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

