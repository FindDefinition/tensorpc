import enum
import inspect
import types
from functools import partial
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Optional,
                    Set, Tuple, Type, Union)
from tensorpc.flow.flowapp.app import AppSpecialEventType

from tensorpc.flow.flowapp.components import mui, three, plus
from tensorpc.flow.flowapp.core import AppEditorFrontendEvent, FlowSpecialMethods, FrontendEventType, _get_obj_def_path
from tensorpc.flow.flowapp.components.plus.objinspect.tree import TreeDragTarget
from tensorpc.flow.flowapp.reload import reload_object_methods
from tensorpc.flow.flowapp.appctx import get_app, get_reload_manager
from tensorpc.flow.flowapp.components.plus.objinspect.core import ALL_OBJECT_LAYOUT_HANDLERS


class AnyFlexLayout(mui.FlexLayout):
    def __init__(self, use_app_editor: bool = True) -> None:
        super().__init__([])
        self.register_event_handler(FrontendEventType.Drop.value, self._on_drop)
        self.register_event_handler(FrontendEventType.ComplexLayoutCloseTab.value, self._on_tab_close)
        self.register_event_handler(FrontendEventType.ComplexLayoutTabReload.value, self._on_tab_reload)
        self.register_event_handler(FrontendEventType.ComplexLayoutSelectTab.value, self._on_tab_select)
        self.register_event_handler(FrontendEventType.ComplexLayoutSelectTabSet.value, self._on_tab_set_select)
        self.use_app_editor = use_app_editor
        self._app_save_reload_cb = None

    async def _reload_child(self, layout, name: str):
        if isinstance(layout, mui.FlexBox):
            if layout._wrapped_obj is not None:
                # for anylayout, we support layout reload
                # and onmount/onunmount reload.
                metas = reload_object_methods(layout._wrapped_obj)
                if metas is not None:
                    special_methods = FlowSpecialMethods(metas)
                    special_methods.bind(layout._wrapped_obj)
                    if special_methods.create_layout is not None:
                        layout_flex = special_methods.create_layout.get_binded_fn()()
                        assert isinstance(layout_flex,
                                            mui.FlexBox), f"create_layout must return a flexbox when use anylayout"
                        layout_flex._flow_comp_def_path = _get_obj_def_path(layout._wrapped_obj)
                        layout_flex._wrapped_obj = layout._wrapped_obj
                        await self.update_childs({name: layout_flex})

                        # await layout.set_new_layout(special_methods.create_layout.get_binded_fn()())
            else:
                metas = reload_object_methods(layout)
                if metas is not None:
                    special_methods = FlowSpecialMethods(metas)
                    special_methods.bind(layout)
                    if special_methods.create_layout is not None:
                        await layout.set_new_layout(special_methods.create_layout.get_binded_fn()())

    async def _bind_code_editor(self, obj, layout, name: str):
        app = get_app()
        app.code_editor.is_external = True 
        lines, lineno = inspect.findsource(type(obj))
        obj_path = inspect.getfile(type(obj))
        await app.set_editor_value(value = "".join(lines), lineno=lineno)
        if self._app_save_reload_cb is not None:
            app.unregister_app_special_event_handler(AppSpecialEventType.CodeEditorSave, self._app_save_reload_cb)
            self._app_save_reload_cb = None 
        self._app_save_reload_cb = partial(self._on_codeeditor_save, name=name, layout=layout, path=obj_path)
        app.register_app_special_event_handler(AppSpecialEventType.CodeEditorSave, self._app_save_reload_cb)

    async def _on_codeeditor_save(self, event: AppEditorFrontendEvent, layout, name: str, path: str):
        with open(path, "w") as f:
            f.write(event.data)
        await self._reload_child(layout, name)

    async def _on_drop(self, target: Optional[TreeDragTarget]):
        if target is not None:
            # print("DROP", comp_name)
            obj = target.obj
            uid = target.tab_id if target.tab_id else target.tree_id
            obj_is_anylayout = get_reload_manager().query_obj_is_anylayout(obj)
            if isinstance(obj, mui.FlexBox):
                wrapped_obj = obj
            else:
                if not isinstance(obj, mui.Component):
                    if obj_is_anylayout:
                        wrapped_obj = mui.flex_wrapper(obj)
                    else:
                        if type(obj) in ALL_OBJECT_LAYOUT_HANDLERS:
                            handler_cls = ALL_OBJECT_LAYOUT_HANDLERS[type(obj)]
                            wrapped_obj = handler_cls.from_object(obj)
                            assert type(obj) == handler_cls and isinstance(wrapped_obj, mui.FlexBox)
                        else:
                            raise ValueError("this shouldn't happen")
                else:
                    wrapped_obj = obj
            if obj_is_anylayout:
                await self._bind_code_editor(obj, wrapped_obj, uid)
            await self.update_childs({uid: wrapped_obj})

    async def _on_tab_close(self, data):
        # print("TAB CLOSE", data)
        name = data["complexLayoutTabNodeId"]
        await self.remove_childs_by_keys([name])
        app = get_app()
        if app.code_editor.is_external:
            app.code_editor.is_external = False 
            if self._app_save_reload_cb is not None:
                app.unregister_app_special_event_handler(AppSpecialEventType.CodeEditorSave, self._app_save_reload_cb)
                self._app_save_reload_cb = None 
            await app._recover_code_editor()


    async def _on_tab_select(self, data):
        child_id = data["id"]
        child_comp = self._child_comps[child_id]
        if isinstance(child_comp, mui.FlexBox):
            if child_comp._wrapped_obj is not None:
                await self._bind_code_editor(child_comp._wrapped_obj, child_comp, child_id)

        # print("TAB SELECT", child_id)

    async def _on_tab_set_select(self, data):
        child_id = data["id"]
        child_comp = self._child_comps[child_id]
        if isinstance(child_comp, mui.FlexBox):
            if child_comp._wrapped_obj is not None:
                await self._bind_code_editor(child_comp._wrapped_obj, child_comp, child_id)
        # print("TAB SET SELECT", child_id)

    async def _on_tab_reload(self, name):
        # print("TAB CLOSE", data)
        layout = self._child_comps[name]
        await self._reload_child(layout, name)
