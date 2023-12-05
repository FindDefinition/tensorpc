import asyncio
import enum
import inspect
import traceback
import types
from typing_extensions import TypeAlias
from functools import partial
from pathlib import Path
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Optional,
                    Set, Tuple, Type, Union)

import watchdog

from tensorpc.core.serviceunit import AppFuncType, ServFunctionMeta
from tensorpc.flow.flowapp.appcore import (AppSpecialEventType,
                                           _run_zeroarg_func,
                                           create_reload_metas, get_app,
                                           get_reload_manager)
from tensorpc.flow.flowapp.components import mui, plus, three
from tensorpc.flow.flowapp.components.plus.core import (
    ALL_OBJECT_LAYOUT_HANDLERS, ObjectLayoutCreator)
from tensorpc.flow.flowapp.core import (AppEditorFrontendEvent,
                                        FlowSpecialMethods, FrontendEventType,
                                        _get_obj_def_path)
from tensorpc.flow.flowapp.coretypes import TreeDragTarget

FlexLayoutInitType: TypeAlias = Union[List[Union[mui.FlexLayout.Row,
                                                     mui.FlexLayout.TabSet]],
                                          mui.FlexLayout.Row,
                                          mui.FlexLayout.TabSet,
                                          mui.FlexLayout.Tab, mui.FlexLayout.HBox,
                                          mui.FlexLayout.VBox, mui.MUIComponentType]
class AnyFlexLayout(mui.FlexLayout):

    def __init__(self,
                 children: Optional[FlexLayoutInitType] = None,
                 use_app_editor: bool = True) -> None:
        if children is None:
            children = []
        super().__init__(children)
        self.register_event_handler(FrontendEventType.Drop.value,
                                    self._on_drop)
        self.register_event_handler(
            FrontendEventType.ComplexLayoutCloseTab.value, self._on_tab_close)
        self.register_event_handler(
            FrontendEventType.ComplexLayoutTabReload.value,
            self._on_tab_reload)
        self.register_event_handler(
            FrontendEventType.ComplexLayoutSelectTab.value,
            self._on_tab_select)
        self.register_event_handler(
            FrontendEventType.ComplexLayoutSelectTabSet.value,
            self._on_tab_set_select)
        self.use_app_editor = use_app_editor
        # self._layout_to_watchdog: Dict[str, Tuple[]]
        self._current_bind_code_id = None
        self.prop(font=mui.FlexLayoutFontProps(size="14px"))

    async def _handle_reload_layout(self, layout: mui.FlexBox,
                                    create_layout: ServFunctionMeta,
                                    name: str):
        if create_layout.user_app_meta is not None and create_layout.user_app_meta.type == AppFuncType.CreateLayout:
            if layout._wrapped_obj is not None:
                layout_flex = create_layout.get_binded_fn()()
                assert isinstance(
                    layout_flex, mui.FlexBox
                ), f"create_layout must return a flexbox when use anylayout"
                layout_flex._flow_comp_def_path = _get_obj_def_path(
                    layout._wrapped_obj)
                layout_flex._wrapped_obj = layout._wrapped_obj
                await self.update_childs({name: layout_flex})
            else:
                layout_flex = create_layout.get_binded_fn()()
                await layout.set_new_layout(layout_flex)
            return layout_flex

    async def _bind_code_editor(self, obj, layout, name: str):
        app = get_app()
        # app.code_editor.external_path = inspect.getfile(type(obj))
        # lines, lineno = inspect.findsource(type(obj))
        # obj_path = inspect.getfile(type(obj))
        # await app.set_editor_value(value="".join(lines), lineno=lineno)
        if app._is_editable_app():
            eapp = app._get_self_as_editable_app()
            eapp._flowapp_observe(
                layout, partial(self._handle_reload_layout, name=name))

    async def _on_drop(self, target: Optional[TreeDragTarget]):
        if target is not None:
            obj = target.obj
            uid = target.tab_id if target.tab_id else target.tree_id
            obj_is_anylayout = get_reload_manager().query_obj_is_anylayout(obj)
            if isinstance(obj, mui.FlexBox):
                wrapped_obj = obj
            elif isinstance(obj, ObjectLayoutCreator):
                obj_is_anylayout = False
                obj = obj.create()
                wrapped_obj = obj
            else:
                if not isinstance(obj, mui.Component):
                    if obj_is_anylayout:
                        wrapped_obj = mui.flex_wrapper(obj, reload_mgr=self.flow_app_comp_core.reload_mgr)
                    else:
                        if type(obj) in ALL_OBJECT_LAYOUT_HANDLERS:
                            handler_cls = ALL_OBJECT_LAYOUT_HANDLERS[type(obj)]
                            wrapped_obj = handler_cls.from_object(obj)
                            assert type(obj) == handler_cls and isinstance(
                                wrapped_obj, mui.FlexBox)
                        else:
                            raise ValueError("this shouldn't happen", obj, type(obj))
                else:
                    wrapped_obj = obj
            wrapped_obj.set_flow_event_context_creator(target.context_creator)
            if obj_is_anylayout:
                await self._bind_code_editor(obj, wrapped_obj, uid)
            await self.update_childs({uid: wrapped_obj})

    async def _on_tab_close(self, data):
        # print("TAB CLOSE", data)
        name = data["complexLayoutTabNodeId"]
        if name in self._child_comps:
            comp = self._child_comps[name]
            await self.remove_childs_by_keys([name])
            app = get_app()
            if app._is_editable_app() and isinstance(comp, mui.FlexBox):
                eapp = app._get_self_as_editable_app()
                eapp._flowapp_remove_observer(comp)
            if app.code_editor.external_path is not None:
                app.code_editor.external_path = None
                # await app._recover_code_editor()
                self._current_bind_code_id = None

    async def _on_tab_select(self, data):
        child_id = data["id"]
        if child_id == self._current_bind_code_id:
            return
        child_comp = self._child_comps[child_id]
        # if isinstance(child_comp, mui.FlexBox):
        #     if child_comp._wrapped_obj is not None:
        #         self._current_bind_code_id = child_id
        #         await self._bind_code_editor(child_comp._wrapped_obj,
        #                                      child_comp, child_id)
        #     else:
        #         obj_is_anylayout = get_reload_manager().query_obj_is_anylayout(
        #             child_comp)
        #         if obj_is_anylayout:
        #             self._current_bind_code_id = child_id
        #             await self._bind_code_editor(child_comp, child_comp,
        #                                          child_id)

        # print("TAB SELECT", child_id)

    async def _on_tab_set_select(self, data):
        child_id = data["id"]
        child_comp = self._child_comps[child_id]
        if child_id == self._current_bind_code_id:
            return
        # if isinstance(child_comp, mui.FlexBox):
        #     if child_comp._wrapped_obj is not None:
        #         self._current_bind_code_id = child_id
        #         await self._bind_code_editor(child_comp._wrapped_obj,
        #                                      child_comp, child_id)
        #     else:
        #         obj_is_anylayout = get_reload_manager().query_obj_is_anylayout(
        #             child_comp)
        #         if obj_is_anylayout:
        #             self._current_bind_code_id = child_id
        #             await self._bind_code_editor(child_comp, child_comp,
        #                                          child_id)

        # print("TAB SET SELECT", child_id)

    async def _on_tab_reload(self, name):
        print("TODO reload")
        # print("TAB CLOSE", data)
        layout = self._child_comps[name]
        # await self._reload_child(layout, name, layout._flow_comp_def_path)
