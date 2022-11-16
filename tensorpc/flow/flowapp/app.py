# Copyright 2022 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Flow APP: simple GUI application in devflow"""

import ast
import asyncio
import base64
import contextlib
import contextvars
import dataclasses
import enum
import inspect
import io
import json
from pathlib import Path
import pickle
import threading
import time
import traceback
from types import ModuleType
from typing import (Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
                    Iterable, List, Optional, Set, Tuple, TypeVar, Union)

import numpy as np
import importlib.machinery
import importlib
import watchdog
import watchdog.events
from PIL import Image
from watchdog.observers import Observer
from tensorpc import simple_chunk_call_async
from tensorpc.constants import PACKAGE_ROOT
from tensorpc.core.asynctools import cancel_task
from tensorpc.core.serviceunit import ReloadableDynamicClass, ServiceUnit
from tensorpc.flow.coretypes import ScheduleEvent, StorageDataItem, get_object_type_meta
from tensorpc.flow.serv_names import serv_names
from tensorpc.utils.registry import HashableRegistry
from tensorpc.utils.reload import reload_method
from tensorpc.utils.uniquename import UniqueNamePool
from tensorpc.flow.client import MasterMeta
from .components import mui, three
from .core import (AppEditorEvent, AppEditorEventType, AppEditorFrontendEvent,
                   AppEditorFrontendEventType, AppEvent, AppEventType,
                   BasicProps, Component, ContainerBase, CopyToClipboardEvent,
                   LayoutEvent, TaskLoopEvent, UIEvent, UIExceptionEvent,
                   UIRunStatus, UIType, UIUpdateEvent, Undefined, UserMessage, ValueType,
                   undefined, EventHandler)
from tensorpc.utils.moduleid import get_qualname_of_type
from tensorpc.utils.moduleid import is_lambda, is_valid_function
ALL_APP_EVENTS = HashableRegistry()


class AppContext:
    def __init__(self, app: "App") -> None:
        self.app = app


APP_CONTEXT_VAR: contextvars.ContextVar[
    Optional[AppContext]] = contextvars.ContextVar("flowapp_context",
                                                   default=None)


def get_app_context() -> Optional[AppContext]:
    return APP_CONTEXT_VAR.get()


def get_app_storage():
    ctx = get_app_context()
    assert ctx is not None
    return ctx.app.get_persist_storage()


@contextlib.contextmanager
def _enter_app_conetxt(app: "App"):
    ctx = AppContext(app)
    token = APP_CONTEXT_VAR.set(ctx)
    try:
        yield ctx
    finally:
        APP_CONTEXT_VAR.reset(token)
    
_ROOT = "root"


class AppEditor:
    def __init__(self, init_value: str, language: str,
                 queue: "asyncio.Queue[AppEvent]") -> None:
        self._language = language
        self._value: str = init_value
        self.__freeze_language = False
        self._init_line_number = 1
        self._monaco_state: Optional[Any] = None
        self._queue = queue

    def set_init_line_number(self, val: int):
        self._init_line_number = val

    def freeze(self):
        self.__freeze_language = True

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val: str):
        self._value = val

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, val: str):
        if not self.__freeze_language:
            self._language = val
        else:
            raise ValueError("language freezed, you can't change it.")

    def get_state(self):
        state = {}
        state["language"] = self._language
        state["value"] = self._value
        state["monacoEditorState"] = self._monaco_state
        state["initLineNumber"] = self._init_line_number
        return state

    async def _send_editor_event(self, event: AppEditorEvent):
        await self._queue.put(AppEvent("", {AppEventType.AppEditor: event}))

    async def set_editor_value(self, value: str, language: str = ""):
        """use this method to set editor value and language.
        """
        self.value = value
        if language:
            self.language = language
        app_ev = AppEditorEvent(AppEditorEventType.SetValue, {
            "value": self.value,
            "language": self.language,
        })
        await self._send_editor_event(app_ev)

class App:
    """
    App Init Callbacks:
    1. app init/app init async
    2. set_persist_props_async for all comps
    """
    def __init__(self,
                 flex_flow: Union[str, Undefined] = "column nowrap",
                 justify_content: Union[str, Undefined] = undefined,
                 align_items: Union[str, Undefined] = undefined,
                 maxqsize: int = 10,
                 enable_value_cache: bool = False) -> None:
        self._uid_to_comp: Dict[str, Component] = {}
        self._queue: "asyncio.Queue[AppEvent]" = asyncio.Queue(
            maxsize=maxqsize)

        self._send_callback: Optional[Callable[[AppEvent],
                                               Coroutine[None, None,
                                                         None]]] = None
        root = mui.FlexBox(self._uid_to_comp,
                           inited=True,
                           uid=_ROOT,
                           queue=self._queue)
        root.prop(flex_flow=flex_flow,
                  justify_content=justify_content,
                  align_items=align_items)
        self._uid_to_comp[_ROOT] = root
        self.root = root
        self._enable_editor = False

        self.code_editor = AppEditor("", "python", self._queue)
        self._app_dynamic_cls: Optional[ReloadableDynamicClass] = None
        # other app can call app methods via service_unit
        self._app_service_unit: Optional[ServiceUnit] = None

        # loaded if you connect app node with a full data storage
        self._data_storage: Dict[str, Any] = {}

        self._force_special_layout_method = False

        self.__persist_storage: Dict[str, Any] = {}

        self.__previous_error_sync_props = {}
        self.__previous_error_persist_state = {}
        self._enable_value_cache = enable_value_cache
        self._flow_app_is_headless = False

        self.__flowapp_master_meta = MasterMeta()
        self.__flowapp_storage_cache: Dict[str, StorageDataItem] = {}

    async def save_data_storage(self,
                                key: str,
                                data: Any,
                                in_memory_limit: int = 100):
        data_enc = pickle.dumps(data)
        parts = key.split(".")
        assert len(parts) in [
            2, 3
        ], "must be graph_id.node_id.key or node_id.key (for current node)"
        assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        if len(parts) == 2:
            parts.insert(0, self.__flowapp_master_meta.graph_id)
        meta = get_object_type_meta(data, parts[-1])
        in_memory_limit_bytes = in_memory_limit * 1024 * 1024
        item = StorageDataItem(data, time.time_ns(), meta)
        if len(data_enc) <= in_memory_limit_bytes:
            self.__flowapp_storage_cache[key] = item
        await simple_chunk_call_async(self.__flowapp_master_meta.grpc_url,
                                      serv_names.FLOW_DATA_SAVE, parts[0],
                                      parts[1], parts[2], data_enc, meta,
                                      item.timestamp)

    async def read_data_storage(self, key: str, in_memory_limit: int = 100):
        meta = self.__flowapp_master_meta
        parts = key.split(".")
        assert len(parts) in [
            2, 3
        ], "must be graph_id.node_id.key or node_id.key (for current node)"
        assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        if len(parts) == 2:
            parts.insert(0, self.__flowapp_master_meta.graph_id)
        if key in self.__flowapp_storage_cache:
            item_may_invalid = self.__flowapp_storage_cache[key]
            res = await simple_chunk_call_async(meta.grpc_url,
                                                serv_names.FLOW_DATA_READ,
                                                parts[0], parts[1], parts[2],
                                                item_may_invalid.timestamp)
            if res is None:
                return pickle.loads(item_may_invalid.data)
            else:
                return pickle.loads(res.data)
        else:
            res: StorageDataItem = await simple_chunk_call_async(meta.grpc_url,
                                           serv_names.FLOW_DATA_READ, parts[0],
                                           parts[1], parts[2])
            in_memory_limit_bytes = in_memory_limit * 1024 * 1024
            data = pickle.loads(res.data)
            if len(res.data) <= in_memory_limit_bytes:
                self.__flowapp_storage_cache[key] = res
            return data


    def get_persist_storage(self):
        return self.__persist_storage

    def _get_simple_app_state(self):
        """get state of Input/Switch/Radio/Slider/Select
        """
        state: Dict[str, Any] = {}
        user_state: Dict[str, Any] = {}

        for comp in self.root._get_all_nested_childs():
            # automatic simple state store
            if isinstance(comp, (
                    mui.Input,
                    mui.Switch,
                    mui.RadioGroup,
                    mui.Slider,
                    mui.Select,
                    mui.MultipleSelect,
            )):
                state[comp._flow_uid] = {
                    "type": comp._flow_comp_type.value,
                    "props": comp.get_sync_props(),
                }
            # user state
            st = comp.get_persist_props()
            if st is not None:
                user_state[comp._flow_uid] = {
                    "type": comp._flow_comp_type.value,
                    "state": st,
                }
        # print("persist_storage_SAVE", self.__persist_storage, id(self.__persist_storage))
        return {
            "persist_storage": self.__persist_storage,
            "uistate": state,
            "userstate": user_state,
        }

    async def _restore_simple_app_state(self, state: Dict[str, Any]):
        """try to restore state of Input/Switch/Radio/Slider/Select
        no exception if fail.
        """
        uistate = state["uistate"]
        userstate = state["userstate"]
        # print("persist_storage", state["persist_storage"])
        if state["persist_storage"]:
            self.__persist_storage.update(state["persist_storage"])
        if self._enable_value_cache:
            ev = AppEvent("", {})
            for k, s in uistate.items():
                if k in self.root._uid_to_comp:
                    comp_to_restore = self.root._uid_to_comp[k]
                    if comp_to_restore._flow_comp_type.value == s["type"]:
                        comp_to_restore.set_props(s["props"])
                        ev += comp_to_restore.get_sync_event(True)
            with _enter_app_conetxt(self):
                for k, s in userstate.items():
                    if k in self.root._uid_to_comp:
                        comp_to_restore = self.root._uid_to_comp[k]
                        if comp_to_restore._flow_comp_type.value == s["type"]:
                            try:
                                await comp_to_restore.set_persist_props_async(
                                    s["state"])
                            except:
                                traceback.print_exc()
                                continue
            await self._queue.put(ev)

    def _app_force_use_layout_function(self):
        self._force_special_layout_method = True
        self.root._prevent_add_layout = True

    async def _app_run_layout_function(self,
                                       send_layout_ev: bool = False,
                                       with_code_editor: bool = True,
                                       reload: bool = False):
        self.root._prevent_add_layout = False
        prev_comps = self.__previous_error_sync_props.copy()
        prev_user_states = self.__previous_error_persist_state.copy()
        if reload:
            for u, c in self._uid_to_comp.items():
                prev_comps[u] = c._to_dict_with_sync_props()
                user_state = c.get_persist_props()
                if user_state is not None:
                    prev_user_states[u] = {
                        "type": c._flow_comp_type.value,
                        "state": user_state,
                    }
        await self.root._clear()
        self._uid_to_comp.clear()
        self.root._flow_uid = _ROOT
        try:
            res = self.app_create_layout()
            if isinstance(res, list):
                res = {str(i): v for i, v in enumerate(res)}
            self.__previous_error_sync_props.clear()
            self.__previous_error_persist_state.clear()
        except Exception as e:
            # TODO store
            traceback.print_exc()
            ss = io.StringIO()
            traceback.print_exc(file=ss)
            user_exc = UserMessage.create_error("", str(e), ss.getvalue())
            ev = UIExceptionEvent([user_exc])
            fbm = (
                "app_create_layout failed!!! check your app_create_layout. if "
                "you are using reloadable app, just check and save your app code!"
            )
            await self._queue.put(
                AppEvent(
                    "", {
                        AppEventType.UIException:
                        ev,
                        AppEventType.UpdateLayout:
                        LayoutEvent(
                            self._get_fallback_layout(fbm, with_code_editor))
                    }))
            return
        res_anno: Dict[str, Component] = {**res}
        self.root.add_layout(res_anno)
        self._uid_to_comp[_ROOT] = self.root
        self.root._prevent_add_layout = True
        if reload:
            # comps = self.root._get_all_nested_childs()
            with _enter_app_conetxt(self):

                for comp in self._uid_to_comp.values():
                    if comp._flow_uid in prev_comps:
                        if comp._flow_comp_type.value == prev_comps[
                                comp._flow_uid]["type"]:
                            comp.set_props(prev_comps[comp._flow_uid]["props"])
                    if comp._flow_uid in prev_user_states:
                        if comp._flow_comp_type.value == prev_user_states[
                                comp._flow_uid]["type"]:
                            await comp.set_persist_props_async(
                                prev_user_states[comp._flow_uid]["state"])
            del prev_comps
            del prev_user_states

        if send_layout_ev:
            ev = AppEvent(
                "", {
                    AppEventType.UpdateLayout:
                    LayoutEvent(self._get_app_layout(with_code_editor))
                })
            await self._queue.put(ev)

    def app_initialize(self):
        """override this to init app before server start
        """
        pass

    async def app_initialize_async(self):
        """override this to init app before server start
        """
        pass

    def app_create_layout(self) -> mui.LayoutType:
        """override this in EditableApp to support reloadable layout
        """
        return {}

    def app_create_node_layout(self) -> Optional[mui.LayoutType]:
        """override this in EditableApp to support layout without fullscreen
        if not provided, will use fullscreen layout
        """
        return None

    def app_create_side_layout(self) -> Optional[mui.LayoutType]:
        """override this in EditableApp to support side layout when selected
        if not provided, will use fullscreen layout
        """
        return None

    def _get_app_dynamic_cls(self):
        assert self._app_dynamic_cls is not None
        return self._app_dynamic_cls

    def _get_app_service_unit(self):
        assert self._app_service_unit is not None
        return self._app_service_unit

    def _get_app_layout(self, with_code_editor: bool = True):
        res = {
            "layout": {u: c.to_dict()
                       for u, c in self._uid_to_comp.items()},
            "enableEditor": self._enable_editor,
            "fallback": "",
        }
        if with_code_editor:
            res.update({
                "codeEditor": self.code_editor.get_state(),
            })
        # node_layout = self.app_create_node_layout()
        # if node_layout is not None:
        #     res["nodeLayout"] = mui.layout_unify(node_layout)
        # side_layout = self.app_create_side_layout()
        # if side_layout is not None:
        #     res["sideLayout"] = mui.layout_unify(side_layout)
        return res

    def _get_fallback_layout(self,
                             fallback_msg: str,
                             with_code_editor: bool = True):
        res = {
            "layout": {},
            "enableEditor": self._enable_editor,
            "fallback": fallback_msg,
        }
        if with_code_editor:
            res.update({
                "codeEditor": self.code_editor.get_state(),
            })
        return res

    def init_enable_editor(self):
        self._enable_editor = True

    def set_init_window_size(self, size: List[Union[int, Undefined]]):
        self.root.props.width = size[0]
        self.root.props.height = size[1]

    async def headless_main(self):
        """override this method to support headless mode.
        you can use headless methods for control UIs such as 
        btn.headless_click and inp.headless_write to trigger
        callbacks.
        """
        raise NotImplementedError(
            "headless_main not exists. "
            "override headless_main to run in headless mode.")

    async def flow_run(self, event: ScheduleEvent):
        """override this method to support flow. output data will be 
        sent to all child nodes if not None.
        """
        return None

    async def _handle_code_editor_event_system(self,
                                               event: AppEditorFrontendEvent):
        if event.type == AppEditorFrontendEventType.SaveEditorState:
            self.code_editor._monaco_state = event.data
            return
        elif event.type == AppEditorFrontendEventType.Save:
            self.code_editor.value = event.data
        with _enter_app_conetxt(self):
            return await self.handle_code_editor_event(event)

    async def handle_code_editor_event(self, event: AppEditorFrontendEvent):
        """override this method to support vscode editor.
        """
        return

    async def _send_editor_event(self, event: AppEditorEvent):
        await self._queue.put(AppEvent("", {AppEventType.AppEditor: event}))

    async def set_editor_value(self, value: str, language: str = ""):
        """use this method to set editor value and language.
        """
        self.code_editor.value = value
        if language:
            self.code_editor.language = language
        app_ev = AppEditorEvent(
            AppEditorEventType.SetValue, {
                "value": self.code_editor.value,
                "language": self.code_editor.language,
            })
        await self._send_editor_event(app_ev)

    async def handle_event(self, ev: UIEvent):
        for uid, data in ev.uid_to_data.items():
            comp = self._uid_to_comp[uid]
            await comp.handle_event(data)

    async def _handle_event_with_ctx(self, ev: UIEvent):
        # TODO run control from other component
        with _enter_app_conetxt(self):
            for uid, data in ev.uid_to_data.items():
                comp = self._uid_to_comp[uid]
                await comp.handle_event(data)

    async def copy_text_to_clipboard(self, text: str):
        """copy to clipboard in frontend."""
        await self._queue.put(
            AppEvent(
                "",
                {AppEventType.CopyToClipboard: CopyToClipboardEvent(text)}))


_WATCHDOG_MODIFY_EVENT_TYPES = Union[watchdog.events.DirModifiedEvent,
                                     watchdog.events.FileModifiedEvent]


class _WatchDogForAppFile(watchdog.events.FileSystemEventHandler):
    def __init__(
            self, on_modified: Callable[[_WATCHDOG_MODIFY_EVENT_TYPES],
                                        None]) -> None:
        super().__init__()
        self._on_modified = on_modified

    def on_modified(self, event: _WATCHDOG_MODIFY_EVENT_TYPES):
        return self._on_modified(event)

@dataclasses.dataclass
class _CompReloadMeta:
    uid: str
    handler: EventHandler 


def _create_reload_metas(uid_to_comp: Dict[str, Component], path: str):
    path_resolve = str(Path(path).resolve())
    metas: List[_CompReloadMeta] = []
    for k, v in uid_to_comp.items():
        # try:
        #     # if comp is inside tensorpc official, ignore it.
        #     Path(v_file).relative_to(PACKAGE_ROOT)
        #     continue 
        # except:
        #     pass
        for handler_type, handler in v._flow_event_handlers.items():
            if not isinstance(handler, Undefined):
                cb = handler.cb 
                if not is_valid_function(cb) or is_lambda(cb):
                    continue 
                cb_file = str(Path(inspect.getfile(cb)).resolve())
                if cb_file != path_resolve:
                    continue
                # code, _ = inspect.getsourcelines(cb)
                metas.append(_CompReloadMeta(k, handler))
    return metas 

class _SimpleCodeManager:
    def __init__(self, paths: List[str]) -> None:
        self.file_to_code: Dict[str, str] = {}
        for path in paths:
            resolved_path = str(Path(path).resolve())
            with open(resolved_path, "r") as f:
                self.file_to_code[path] = f.read()

    def update_code_from_editor(self, path: str, new_data: str):
        resolved_path = str(Path(path).resolve())
        if new_data == self.file_to_code[resolved_path]:
            return False 
        self.file_to_code[resolved_path] = new_data
        return True

    def update_code(self, change_file: str):
        resolved_path = str(Path(change_file).resolve())
        if resolved_path not in self.file_to_code:
            return False 
        with open(change_file, "r") as f:
            new_data = f.read()
        if new_data == self.file_to_code[resolved_path]:
            return False 
        self.file_to_code[resolved_path] = new_data
        return True

    def get_code(self, change_file: str):
        resolved_path = str(Path(change_file).resolve())
        return self.file_to_code[resolved_path]


class EditableApp(App):
    def __init__(self,
                 reloadable_layout: bool = False,
                 use_app_editor: bool = True,
                 flex_flow: Union[str, Undefined] = "column nowrap",
                 justify_content: Union[str, Undefined] = undefined,
                 align_items: Union[str, Undefined] = undefined,
                 maxqsize: int = 10,
                 observed_files: Optional[List[str]] = None) -> None:
        super().__init__(flex_flow, justify_content, align_items, maxqsize)
        self._use_app_editor = use_app_editor
        if use_app_editor:
            lines, lineno = inspect.findsource(type(self))
            self.code_editor.value = "".join(lines)
            self.code_editor.language = "python"
            self.code_editor.set_init_line_number(lineno)
            self.code_editor.freeze()
        self._watchdog_prev_content = ""
        self._flow_reloadable_layout = reloadable_layout
        if reloadable_layout:
            self._app_force_use_layout_function()
        self._flow_observed_files = observed_files

    def app_initialize(self):
        dcls = self._get_app_dynamic_cls()
        path = dcls.file_path
        self._watchdog_watcher = None
        self._watchdog_observer = None
        if not self._flow_app_is_headless:
            observer = Observer()
            self._watchdog_watcher = _WatchDogForAppFile(
                self._watchdog_on_modified)
            if self._flow_observed_files is not None:
                for p in self._flow_observed_files:
                    assert Path(p).exists(), f"{p} must exist"
                paths = set(self._flow_observed_files)
                self._flow_comp_mgr = _SimpleCodeManager(self._flow_observed_files)
            else:
                self._flow_comp_mgr = _SimpleCodeManager(list(self.__get_default_observe_paths()))
                paths = set(self._flow_comp_mgr.file_to_code.keys())

            paths.add(str(Path(path).resolve()))
            # print(paths)
            for p in paths:
                observer.schedule(self._watchdog_watcher, p, recursive=False)
            observer.start()
            self._watchdog_observer = observer
        else:
            self._flow_comp_mgr = None 
        self._watchdog_ignore_next = False
        self._loop = asyncio.get_running_loop()
        self._watch_lock = threading.Lock()

    def __get_default_observe_paths(self):
        res: Set[str] = set()
        for k, v in self._uid_to_comp.items():
            v_file = v._flow_comp_def_path
            if not v_file:
                continue 
            try:
                # if comp is inside tensorpc official, ignore it.
                Path(v_file).relative_to(PACKAGE_ROOT)
                continue 
            except:
                pass
            res.add(v_file)
        res.add(self._get_app_dynamic_cls().file_path)
        return res

    def __reload_callback(self, change_file: str, mod_is_reloaded: bool):
        # TODO find a way to record history callback code and 
        # reload only if code change
        assert self._flow_comp_mgr is not None 
        resolved_path = str(Path(change_file).resolve())
        reload_metas = _create_reload_metas(self._uid_to_comp, resolved_path)
        if reload_metas:
            module = inspect.getmodule(reload_metas[0].handler.cb)
            if module is None:
                return
            # now module is valid, reload it.
            if not mod_is_reloaded:
                try:
                    importlib.reload(module)
                except:
                    traceback.print_exc()
                    return
            for meta in reload_metas:
                handler = meta.handler
                cb = handler.cb 
                new_method, new_code = reload_method(cb, module.__dict__)
                # if new_code:
                #     meta.code = new_code
                if new_method is not None:
                    # print(new_method, "new_method")
                    handler.cb = new_method


    def _watchdog_on_modified(self, ev: _WATCHDOG_MODIFY_EVENT_TYPES):
        if isinstance(ev, watchdog.events.FileModifiedEvent):
            dcls = self._get_app_dynamic_cls()
            resolved_path = str(Path(ev.src_path).resolve())
            resolved_app_path = str(Path(dcls.file_path).resolve())
            is_app_file_changed = resolved_app_path == resolved_path
            print("WATCHDOG", ev)
            with self._watch_lock:
                if self._flow_comp_mgr is None:
                    return
                if not self._flow_comp_mgr.update_code(ev.src_path):
                    return 
                new_data = self._flow_comp_mgr.get_code(ev.src_path)
                try:
                    ast.parse(new_data, filename=ev.src_path)
                except:
                    traceback.print_exc()
                    return
                try:
                    # watchdog callback can't fail
                    code_changed: List[str] = []
                    if is_app_file_changed:
                        if self._use_app_editor:
                            fut = asyncio.run_coroutine_threadsafe(
                                self.set_editor_value(new_data), self._loop)
                            fut.result()
                        code_changed = self._reload_app_file()
                        layout_func_changed = App.app_create_layout.__qualname__ in code_changed
                        if layout_func_changed:
                            fut = asyncio.run_coroutine_threadsafe(
                                self._app_run_layout_function(
                                    True, with_code_editor=False, reload=True),
                                self._loop)
                            fut.result()
                    already_reloaded = is_app_file_changed
                    self.__reload_callback(resolved_path, already_reloaded)
                except:
                    traceback.print_exc()
                    return


    def _reload_app_file(self):
        comps = self._uid_to_comp
        # callback_dict = {}
        # for k, v in comps.items():
        #     cb = v.get_callback()
        #     if cb is not None:
        #         callback_dict[k] = cb
        new_cb, code_changed = self._get_app_dynamic_cls().reload_obj_methods(
            self, {})
        self._get_app_service_unit().reload_metas()
        # for k, v in comps.items():
        #     if k in new_cb:
        #         v.set_callback(new_cb[k])
        return code_changed

    async def handle_code_editor_event(self, event: AppEditorFrontendEvent):
        """override this method to support vscode editor.
        """
        if self._use_app_editor:
            app_path = self._get_app_dynamic_cls().file_path
            if event.type == AppEditorFrontendEventType.Save:
                with self._watch_lock:
                    # self._watchdog_ignore_next = True
                    with open(app_path, "w") as f:
                        f.write(event.data)
                    # self._watchdog_prev_content = event.data
                    if self._flow_comp_mgr is not None:
                        self._flow_comp_mgr.update_code_from_editor(app_path, event.data)
                    code_changed = self._reload_app_file()
                    layout_func_changed = App.app_create_layout.__qualname__ in code_changed
                    if layout_func_changed:
                        await self._app_run_layout_function(
                            True, with_code_editor=False, reload=True)
        return


class EditableLayoutApp(EditableApp):
    def __init__(self,
                 use_app_editor: bool = True,
                 flex_flow: Union[str, Undefined] = "column nowrap",
                 justify_content: Union[str, Undefined] = undefined,
                 align_items: Union[str, Undefined] = undefined,
                 maxqsize: int = 10,
                 observed_files: Optional[List[str]] = None) -> None:
        super().__init__(True, use_app_editor, flex_flow, justify_content,
                         align_items, maxqsize, observed_files)
