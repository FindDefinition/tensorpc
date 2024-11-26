import ast
import dataclasses
import enum
from functools import partial
import sys
from types import FrameType
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Optional,
                    Sequence, Set, Tuple, Type, TypeVar, Union)

from tensorpc.constants import TENSORPC_FILE_NAME_PREFIX
from tensorpc.core import funcid
from tensorpc.core.tree_id import UniqueTreeIdForTree
from tensorpc.dbg.core.frame_id import VariableMetaType, get_storage_frame_path
from tensorpc.flow import appctx
from tensorpc.flow.components import mui, three
from tensorpc.flow.components.plus.objinspect.tree import BasicObjectTree
from tensorpc.flow.components.plus.objview.script import get_frame_obj_layout_from_code, get_init_obj_convert_code
from tensorpc.flow.components.plus.styles import get_tight_icon_tab_theme
from tensorpc.flow.core.objtree import UserObjTreeProtocol
from tensorpc.utils.rich_logging import get_logger
from ..objview.preview import ObjectPreview, ObjectPreviewBase

LOGGER = get_logger("tensorpc.dbg")

class FrameObjTabType(enum.Enum):
    Preview = "preview"
    OriginPreview = "origin_preview"
    Tree = "tree"
    Editor = "editor"

@dataclasses.dataclass
class FrameObjectPreviewState:
    frame_id: str
    frame_qualname: str
    name: Optional[str] = None
    value: Any = None

    @property 
    def frame_object_id(self):
        assert self.name is not None 
        return f"{self.frame_id}_{self.name}"

    @property 
    def frame_object_path(self):
        assert self.name is not None 
        return f"{self.frame_id}_{self.name}.py"

@dataclasses.dataclass
class FrameObjectMeta:
    convert_code: str

@dataclasses.dataclass
class _FrameHolder:
    frame: Optional[FrameType]

class FrameObjectPreview(ObjectPreviewBase):

    def __init__(self):
        self._obj_preview = ObjectPreview(enable_reload=False)
        self._obj_user_sel_preview = ObjectPreview(enable_reload=False)

        self._obj_original_preview = ObjectPreview(enable_reload=False)

        self._fold_editor_container = mui.HBox([]).prop(flex=1)

        self._obj_simple_tree = BasicObjectTree(use_fast_tree=True)
        self._obj_simple_tree.prop(minHeight=0,
                        minWidth=0,
                        width="100%",
                        height="100%",
                        overflow="hidden")
        tab_theme = get_tight_icon_tab_theme()
        self._editor = mui.MonacoEditor("", "python", "")
        self._editor.prop(minHeight=0,
                        minWidth=0,
                        width="100%",
                        height="100%",
                        overflow="hidden")
        self._editor.event_editor_save.on(
            self._on_editor_save)
        self._editor.event_component_ready.on(
            self._on_editor_ready)
        tabdefs = [
            mui.TabDef("",
                       FrameObjTabType.Preview.value,
                       mui.HBox([
                            self._obj_preview.prop(flex=1),
                            mui.Divider("vertical"),
                            self._obj_user_sel_preview.prop(flex=1),
                            mui.Divider("vertical"),
                            self._fold_editor_container.prop(flex=1),

                       ]).prop(width="100%", height="100%", overflow="hidden"),
                       icon=mui.IconType.Preview,
                       tooltip="preview"),
            mui.TabDef("",
                       FrameObjTabType.OriginPreview.value,
                       self._obj_original_preview,
                       icon=mui.IconType.Preview,
                       tooltip="origin preview"),
            mui.TabDef("",
                       FrameObjTabType.Tree.value,
                       self._obj_simple_tree,
                       icon=mui.IconType.AccountTree,
                       tooltip="simple tree"),
            mui.TabDef("",
                       FrameObjTabType.Editor.value,
                       self._editor,
                       icon=mui.IconType.Code,
                       tooltip="editor"),
        ]
        super().__init__([
            mui.ThemeProvider([
                mui.Tabs(tabdefs, init_value=FrameObjTabType.Preview.value).prop(
                    panelProps=mui.FlexBoxProps(width="100%", padding=0),
                    orientation="vertical",
                    borderRight=1,
                    flex=1,
                    borderColor='divider',
                    tooltipPlacement="right")
            ], tab_theme)
        ])
        self.prop(flex=1, flexFlow="row nowrap", alignItems="stretch", border="1px solid #e0e0e0", overflow="hidden")
        self._cur_state: Optional[FrameObjectPreviewState] = None

        self._cur_fold_frame_holder: Optional[_FrameHolder] = None

    async def clear_preview_layout(self):
        await self._obj_preview.clear_preview_layout()

    async def set_preview_layout(
            self,
            layout: mui.LayoutType,
            header: Optional[str] = None):
        return await self._obj_preview.set_preview_layout(layout, header)

    async def set_obj_preview_layout(
            self,
            obj: Any,
            uid: Optional[str] = None,
            root: Optional[UserObjTreeProtocol] = None,
            header: Optional[str] = None): 
        if uid is not None and self._cur_state is not None:
            # TODO here we assume the inspector call this method.
            uid_obj = UniqueTreeIdForTree(uid)
            parts = uid_obj.parts 
            if len(parts) == 2:
                # we only check frame local variables
                var_name = parts[1]
                return await self.set_frame_variable(var_name, obj)
        await self._obj_preview.set_obj_preview_layout(obj, uid, root, header)
        await self._obj_original_preview.set_obj_preview_layout(obj, uid, root, header)

    async def set_frame_meta(self, frame_id: str, frame_qualname: str):
        self._cur_state = FrameObjectPreviewState(frame_id, frame_qualname)

    async def set_frame_variable(self, var_name: str, value: Any):
        assert self._cur_state is not None 
        self._cur_state.name = var_name
        self._cur_state.value = value
        await self._obj_simple_tree.set_root_object_dict({
            var_name: value
        })
        if appctx.get_app_storage().is_available():
            await self._on_editor_ready() 

    async def set_user_selection_frame_variable(self, var_name: str, value: Any):
        await self._obj_user_sel_preview.set_obj_preview_layout(value, None, header=f"Vscode: {var_name}")

    async def _fold_editor_sel(self, ev: mui.MonacoEditorSelectionEvent, frame_holder: _FrameHolder):
        cur_frame = frame_holder.frame
        if cur_frame is None:
            return 
        try:
            local_vars = cur_frame.f_locals
            global_vars = cur_frame.f_globals
            res = eval(ev.selectedCode, global_vars,
                        local_vars)
            await self.set_user_selection_frame_variable(
                ev.selectedCode, res)
        except Exception as e:
            LOGGER.info(f"Eval code segment failed. exception: {e}")
            return

    async def set_folding_code(self, var_name: str, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], var_frame: Optional[FrameType]):
        fold_code = funcid.fold_func_node_with_target_identifier_to_code(node, var_name, with_func=False)
        editor = mui.MonacoEditor(fold_code, "python", "").prop(readOnly=True, width="100%", height="100%")
        if var_frame is not None:
            self._cur_fold_frame_holder = _FrameHolder(var_frame)
            editor.event_editor_cursor_selection.on(partial(self._fold_editor_sel, frame_holder=self._cur_fold_frame_holder))
            del var_frame
        await self._fold_editor_container.set_new_layout([
            editor
        ])

    async def clear_frame_variable(self):
        self._cur_state = None
        await self.send_and_wait(
            self._editor.update_event(
                value="",
                path=""))

    async def clear_preview_layouts(self):
        if self._cur_fold_frame_holder is not None:
            frame = self._cur_fold_frame_holder.frame
            self._cur_fold_frame_holder.frame = None
            del frame
            self._cur_fold_frame_holder = None
        await self._obj_preview.clear_preview_layout()
        await self._obj_original_preview.clear_preview_layout()
        await self._obj_user_sel_preview.clear_preview_layout()
        await self._obj_simple_tree.set_root_object_dict({})
        await self._fold_editor_container.set_new_layout([])

    def _determine_convert_code_is_trivial(self, tree: ast.Module):
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                if node.name == "convert":
                    # TODO handle doc string
                    # analysis function node body
                    if len(node.body) > 1:
                        return True 
                    # one-line return stmt. check is  "return x"
                    if len(node.body) == 1 and isinstance(node.body[0], ast.Return):
                        if isinstance(node.body[0], ast.Name):
                            return True
                    break
        return False

    async def _on_editor_ready(self):
        if self._cur_state is not None and self._cur_state.name is not None:
            assert appctx.get_app().app_storage.is_available()
            var_storage_path = self._get_storage_var_path(self._cur_state.frame_id, self._cur_state.name)
            prev_obj_meta_dict = await appctx.read_data_storage(var_storage_path, raise_if_not_found=False)
            if prev_obj_meta_dict is None:
                init_code = get_init_obj_convert_code()
                meta = FrameObjectMeta(init_code)
            else:
                meta = FrameObjectMeta(**prev_obj_meta_dict)
            await self.send_and_wait(
                self._editor.update_event(
                    value=meta.convert_code,
                    path=self._cur_state.frame_object_path))
            await self._set_frame_obj_layout_from_code(meta.convert_code)

    def _get_storage_var_path(self, frame_id: str, var_name: str):
        storage_path = get_storage_frame_path(frame_id)
        return f"{storage_path}/{var_name}/{VariableMetaType.Layout.value}"

    async def _set_frame_obj_layout_from_code(self, value: str):
        if self._cur_state is not None and self._cur_state.name is not None:
            header = f"{self._cur_state.frame_qualname}::{self._cur_state.name}"
            init_code_for_compare = get_init_obj_convert_code()
            await self._obj_original_preview.set_obj_preview_layout(self._cur_state.value, None, header=header)
            if value.strip() == init_code_for_compare.strip():
                return await self._obj_preview.set_obj_preview_layout(self._cur_state.value, None, header=header)
            var_storage_path = self._get_storage_var_path(self._cur_state.frame_id, self._cur_state.name)
            prev_obj_meta_dict = await appctx.read_data_storage(var_storage_path, raise_if_not_found=False)
            if prev_obj_meta_dict is not None:
                meta = FrameObjectMeta(**prev_obj_meta_dict)
                meta = dataclasses.replace(meta, convert_code=value)
            else:
                meta = FrameObjectMeta(value)
            await appctx.save_data_storage(var_storage_path, dataclasses.asdict(meta))
            fname = f"<{TENSORPC_FILE_NAME_PREFIX}-scripts-{self._cur_state.frame_object_path}>"
            obj_converted, layouts = get_frame_obj_layout_from_code(fname, value, self._cur_state.value, )
            if layouts is None:
                return await self._obj_preview.set_obj_preview_layout(obj_converted, None, header=header)
            await self._obj_preview.set_preview_layout([mui.VBox(layouts).prop(flex=1, overflow="auto")], header=header)


    async def _on_editor_save(self, ev: mui.MonacoEditorSaveEvent):
        if self._cur_state is not None:
            value = ev.value
            await self._set_frame_obj_layout_from_code(value)

