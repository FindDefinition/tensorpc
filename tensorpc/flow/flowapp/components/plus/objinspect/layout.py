import asyncio
import enum
import inspect
import traceback
import types
from functools import partial
from pathlib import Path
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Optional,
                    Set, Tuple, Type, Union)

import watchdog

from tensorpc.flow.flowapp.appcore import (AppSpecialEventType,
                                           _run_zeroarg_func,
                                           create_reload_metas, get_app,
                                           get_reload_manager)
from tensorpc.flow.flowapp.components import mui, plus, three
from tensorpc.flow.flowapp.components.plus.objinspect.core import \
    ALL_OBJECT_LAYOUT_HANDLERS
from tensorpc.flow.flowapp.core import (AppEditorFrontendEvent,
                                        FlowSpecialMethods, FrontendEventType,
                                        _get_obj_def_path)
from tensorpc.flow.flowapp.coretypes import TreeDragTarget
from tensorpc.flow.flowapp.reload import reload_object_methods
from tensorpc.utils.reload import reload_method


class AnyFlexLayout(mui.FlexLayout):

    def __init__(self, use_app_editor: bool = True) -> None:
        super().__init__([])
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
        self._app_save_reload_cb = None
        self._app_watchdog_reload_cb = None
        self._app_watchdog_watch = None
        self._current_bind_code_id = None

    async def _reload_child_and_set_code(self, code: str, layout, name: str,
                                         path: str):
        app = get_app()
        await app.set_editor_value(value=code)
        await self._reload_child(layout, name, path)

    def __reload_callback(self, layout: mui.FlexBox, change_file: str):
        # TODO find a way to record history callback code and
        # reload only if code change
        uid_to_comp = layout._get_uid_to_comp_dict()

        resolved_path = str(Path(change_file).resolve())
        reload_metas = create_reload_metas(uid_to_comp, resolved_path)
        if reload_metas:
            try:
                res = self.flow_app_comp_core.reload_mgr.reload_type(
                    reload_metas[0].handler.cb)
                module = res[0].module
                module_dict = res[0].module_dict
            except:
                traceback.print_exc()
                return
            for meta in reload_metas:
                handler = meta.handler
                cb = handler.cb
                new_method, new_code = reload_method(cb, module_dict)
                # if new_code:
                #     meta.code = new_code
                if new_method is not None:
                    # print(new_method, "new_method")
                    handler.cb = new_method

    async def _reload_child(self, layout, name: str, path: str):
        if isinstance(layout, mui.FlexBox):
            app = get_app()

            if layout._wrapped_obj is not None:
                # for anylayout, we support layout reload
                # and onmount/onunmount reload.
                metas = reload_object_methods(
                    layout._wrapped_obj,
                    reload_mgr=self.flow_app_comp_core.reload_mgr)
                if metas is not None:
                    special_methods = FlowSpecialMethods(metas)
                    special_methods.bind(layout._wrapped_obj)
                    if special_methods.create_layout is not None:
                        layout_flex = special_methods.create_layout.get_binded_fn(
                        )()
                        assert isinstance(
                            layout_flex, mui.FlexBox
                        ), f"create_layout must return a flexbox when use anylayout"
                        layout_flex._flow_comp_def_path = _get_obj_def_path(
                            layout._wrapped_obj)
                        layout_flex._wrapped_obj = layout._wrapped_obj
                        await self.update_childs({name: layout_flex})
                        if path:
                            self.__reload_callback(layout_flex, path)
                    if special_methods.auto_run is not None:
                        asyncio.create_task(
                            app._run_autorun(
                                special_methods.auto_run.get_binded_fn()))
                else:
                    if path:
                        self.__reload_callback(layout, path)
                        # await layout.set_new_layout(special_methods.create_layout.get_binded_fn()())
            else:
                metas = reload_object_methods(
                    layout, reload_mgr=self.flow_app_comp_core.reload_mgr)
                if metas is not None:
                    special_methods = FlowSpecialMethods(metas)
                    special_methods.bind(layout)
                    if special_methods.create_layout is not None:
                        await layout.set_new_layout(
                            special_methods.create_layout.get_binded_fn()())
                    if special_methods.auto_run is not None:
                        asyncio.create_task(
                            app._run_autorun(
                                special_methods.auto_run.get_binded_fn()))
                if path:
                    self.__reload_callback(layout, path)

    async def _bind_code_editor(self, obj, layout, name: str):
        app = get_app()
        app.code_editor.is_external = True
        lines, lineno = inspect.findsource(type(obj))
        obj_path = inspect.getfile(type(obj))
        await app.set_editor_value(value="".join(lines), lineno=lineno)
        # TODO add watchdog for single app file
        if self._app_save_reload_cb is not None:
            app.unregister_app_special_event_handler(
                AppSpecialEventType.CodeEditorSave, self._app_save_reload_cb)
            self._app_save_reload_cb = None
        if self._app_watchdog_reload_cb is not None:
            app.unregister_app_special_event_handler(
                AppSpecialEventType.WatchDogChange,
                self._app_watchdog_reload_cb)
            self._app_watchdog_reload_cb = None

        self._app_save_reload_cb = partial(self._on_codeeditor_save,
                                           name=name,
                                           layout=layout,
                                           path=obj_path)

        if self._app_watchdog_watch is not None:
            if app._is_editable_app():
                wo = app._get_self_as_editable_app()._watchdog_observer
                if wo is not None:
                    wo.unschedule(self._app_watchdog_watch)
                    self._app_watchdog_watch = None
        app.register_app_special_event_handler(
            AppSpecialEventType.CodeEditorSave, self._app_save_reload_cb)
        if app._is_editable_app():
            eapp = app._get_self_as_editable_app()
            wo = eapp._watchdog_observer
            if wo is not None:
                self._app_watchdog_watch = wo.schedule(eapp._watchdog_watcher,
                                                       obj_path)
                self._app_watchdog_reload_cb = partial(
                    self._on_watchdog_change_save,
                    name=name,
                    layout=layout,
                    path=obj_path,
                    loop=eapp._loop)
                app.register_app_special_event_handler(
                    AppSpecialEventType.WatchDogChange,
                    self._app_watchdog_reload_cb)

    async def _on_codeeditor_save(self, data: str, layout, name: str,
                                  path: str):
        with open(path, "w") as f:
            f.write(data)
        await self._reload_child(layout, name, path)

    def _on_watchdog_change_save(self, data: Tuple[str, str], layout,
                                 name: str, path: str, loop):
        # WARNING watchdog callback must be sync
        change_path = data[1]
        # print(path, change_path)
        if change_path == path:
            ft = asyncio.run_coroutine_threadsafe(
                self._reload_child_and_set_code(data[0], layout, name, path),
                loop)
            ft.result()

    async def _on_drop(self, target: Optional[TreeDragTarget]):
        if target is not None:
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
                            assert type(obj) == handler_cls and isinstance(
                                wrapped_obj, mui.FlexBox)
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
        if name in self._child_comps:
            await self.remove_childs_by_keys([name])
            app = get_app()
            if self._app_watchdog_watch is not None:
                app = get_app()
                if app._is_editable_app():
                    eapp = app._get_self_as_editable_app()
                    if eapp._watchdog_observer is not None:
                        eapp._watchdog_observer.unschedule(
                            self._app_watchdog_watch)
                        eapp._watchdog_observer = None
            if self._app_watchdog_reload_cb is not None:
                app.unregister_app_special_event_handler(
                    AppSpecialEventType.WatchDogChange,
                    self._app_watchdog_reload_cb)
                self._app_watchdog_reload_cb = None
            if app.code_editor.is_external:
                app.code_editor.is_external = False
                if self._app_save_reload_cb is not None:
                    app.unregister_app_special_event_handler(
                        AppSpecialEventType.CodeEditorSave,
                        self._app_save_reload_cb)
                    self._app_save_reload_cb = None
                await app._recover_code_editor()
                self._current_bind_code_id = None

    async def _on_tab_select(self, data):
        child_id = data["id"]
        if child_id == self._current_bind_code_id:
            return
        child_comp = self._child_comps[child_id]
        if isinstance(child_comp, mui.FlexBox):
            if child_comp._wrapped_obj is not None:
                self._current_bind_code_id = child_id
                await self._bind_code_editor(child_comp._wrapped_obj,
                                             child_comp, child_id)
            else:
                obj_is_anylayout = get_reload_manager().query_obj_is_anylayout(
                    child_comp)
                if obj_is_anylayout:
                    self._current_bind_code_id = child_id
                    await self._bind_code_editor(child_comp, child_comp,
                                                 child_id)

        # print("TAB SELECT", child_id)

    async def _on_tab_set_select(self, data):
        child_id = data["id"]
        child_comp = self._child_comps[child_id]
        if child_id == self._current_bind_code_id:
            return
        if isinstance(child_comp, mui.FlexBox):
            if child_comp._wrapped_obj is not None:
                self._current_bind_code_id = child_id
                await self._bind_code_editor(child_comp._wrapped_obj,
                                             child_comp, child_id)
            else:
                obj_is_anylayout = get_reload_manager().query_obj_is_anylayout(
                    child_comp)
                if obj_is_anylayout:
                    self._current_bind_code_id = child_id
                    await self._bind_code_editor(child_comp, child_comp,
                                                 child_id)

        # print("TAB SET SELECT", child_id)

    async def _on_tab_reload(self, name):
        # print("TAB CLOSE", data)
        layout = self._child_comps[name]
        await self._reload_child(layout, name, "")
