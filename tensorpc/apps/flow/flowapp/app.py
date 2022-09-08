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

import asyncio
import base64
import contextlib
import enum
import inspect
import io
import threading
import time
import traceback
from typing import (Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
                    Iterable, List, Optional, Tuple, TypeVar, Union)

import numpy as np
import watchdog
import watchdog.events
from PIL import Image
from tensorpc.apps.flow.coretypes import ScheduleEvent
from tensorpc.core.asynctools import cancel_task
from tensorpc.core.serviceunit import ReloadableDynamicClass, ServiceUnit
from tensorpc.utils.registry import HashableRegistry
from tensorpc.utils.uniquename import UniqueNamePool
from watchdog.observers import Observer
import contextvars

from .components import mui, three
from .core import (AppEditorEvent, AppEditorEventType, AppEditorFrontendEvent,
                   AppEditorFrontendEventType, AppEvent, AppEventType,
                   BasicProps, Component, ContainerBase, CopyToClipboardEvent,
                   LayoutEvent, TaskLoopEvent, UIEvent, UIExceptionEvent,
                   UIRunStatus, UIType, UIUpdateEvent, Undefined,
                   UserException, undefined)

ALL_APP_EVENTS = HashableRegistry()


class AppContext:

    def __init__(self, app: "App") -> None:
        self.app = app


APP_CONTEXT_VAR: contextvars.ContextVar[
    Optional[AppContext]] = contextvars.ContextVar("flowapp_context",
                                                   default=None)


def get_app_context() -> Optional[AppContext]:
    return APP_CONTEXT_VAR.get()


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

    def __init__(
        self,
        init_value: str,
        language: str,
    ) -> None:
        self._language = language
        self._value: str = init_value
        self.__freeze_language = False
        self._init_line_number = 1
        self._monaco_state: Optional[Any] = None

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


class App:
    # TODO find a way to sync state to frontend uncontrolled elements.
    def __init__(self,
                 flex_flow: Union[str, Undefined] = "column nowrap",
                 justify_content: Union[str, Undefined] = undefined,
                 align_items: Union[str, Undefined] = undefined,
                 maxqsize: int = 10) -> None:
        self._uid_to_comp: Dict[str, Component] = {}
        self._queue: "asyncio.Queue[AppEvent]" = asyncio.Queue(
            maxsize=maxqsize)

        self._send_callback: Optional[Callable[[AppEvent],
                                               Coroutine[None, None,
                                                         None]]] = None
        root = mui.FlexBox(_ROOT, self._queue, self._uid_to_comp, inited=True)
        root.prop(flex_flow=flex_flow,
                  justify_content=justify_content,
                  align_items=align_items)
        self._uid_to_comp[_ROOT] = root
        self.root = root
        self._enable_editor = False

        self.code_editor = AppEditor("", "python")
        self._app_dynamic_cls: Optional[ReloadableDynamicClass] = None
        # other app can call app methods via service_unit
        self._app_service_unit: Optional[ServiceUnit] = None

        # loaded if you connect app node with a full data storage
        self._data_storage: Dict[str, Any] = {}

        self._force_special_layout_method = False

    def _get_simple_app_state(self):
        """get state of Input/Switch/Radio/Slider/Select
        """
        state: Dict[str, Any] = {}
        for comp in self.root._get_all_nested_childs():
            if isinstance(comp, (
                    mui.Input,
                    mui.Switch,
                    mui.RadioGroup,
                    mui.Slider,
                    mui.Select,
                    mui.MultipleSelect,
            )):
                state[comp.uid] = {
                    "type": comp.type.value,
                    "props": comp.get_sync_props(),
                }
        return state

    async def _restore_simple_app_state(self, state: Dict[str, Any]):
        """try to restore state of Input/Switch/Radio/Slider/Select
        no exception if fail.
        """
        ev = AppEvent("", {})
        for k, s in state.items():
            if k in self.root._uid_to_comp:
                comp_to_restore = self.root._uid_to_comp[k]
                if comp_to_restore.type.value == s["type"]:
                    comp_to_restore.set_props(s["props"])
                    pylance_wtf = ev.merge_new(
                        comp_to_restore.get_sync_event(True))
                    ev = pylance_wtf
        await self._queue.put(ev)

    def _app_force_use_layout_function(self):
        self._force_special_layout_method = True
        self.root._prevent_add_layout = True

    async def _app_run_layout_function(self,
                                       send_layout_ev: bool = False,
                                       with_code_editor: bool = True,
                                       reload: bool = False):
        self.root._prevent_add_layout = False
        prev_comps = {}
        if reload:
            prev_comps = {
                u: c._to_dict_with_sync_props()
                for u, c in self._uid_to_comp.items()
            }
        await self.root._clear()
        self._uid_to_comp.clear()
        self.root.uid = _ROOT
        try:
            res = self.app_create_layout()
        except Exception as e:
            traceback.print_exc()
            ss = io.StringIO()
            traceback.print_exc(file=ss)
            user_exc = UserException("", str(e), ss.getvalue())
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
            for comp in self._uid_to_comp.values():
                if comp.uid in prev_comps:
                    if comp.type.value == prev_comps[comp.uid]["type"]:
                        comp.set_props(prev_comps[comp.uid]["props"])
            del prev_comps
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

    def app_create_layout(self) -> Dict[str, mui.MUIComponentType]:
        """override this in EditableApp to support reloadable layout
        """
        return {}

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
        # with _enter_app_conetxt(self):
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


class EditableApp(App):

    def __init__(self,
                 reloadable_layout: bool = False,
                 use_app_editor: bool = True,
                 flex_flow: Union[str, Undefined] = "column nowrap",
                 justify_content: Union[str, Undefined] = undefined,
                 align_items: Union[str, Undefined] = undefined,
                 maxqsize: int = 10) -> None:
        super().__init__(flex_flow, justify_content, align_items, maxqsize)
        self._use_app_editor = use_app_editor
        if use_app_editor:
            lines, lineno = inspect.findsource(type(self))
            self.code_editor.value = "".join(lines)
            self.code_editor.language = "python"
            self.code_editor.set_init_line_number(lineno)
            self.code_editor.freeze()
        self._watchdog_prev_content = ""
        if reloadable_layout:
            self._app_force_use_layout_function()

    def app_initialize(self):
        dcls = self._get_app_dynamic_cls()
        path = dcls.file_path
        observer = Observer()
        self._watch = _WatchDogForAppFile(self._watchdog_on_modified)
        observer.schedule(self._watch, path, recursive=False)
        observer.start()
        self.observer = observer
        self._watchdog_ignore_next = False
        self._loop = asyncio.get_running_loop()
        self._watch_lock = threading.Lock()

    def _watchdog_on_modified(self, ev: _WATCHDOG_MODIFY_EVENT_TYPES):
        if isinstance(ev, watchdog.events.FileModifiedEvent):
            print("WATCHDOG", ev)
            with self._watch_lock:
                if self._watchdog_ignore_next:
                    self._watchdog_ignore_next = False
                    return
                with open(ev.src_path, "r") as f:
                    new_data = f.read()
                # we have no way to distringuish save event and external save.
                # so we compare data with previous saved result.
                if new_data != self._watchdog_prev_content:
                    if self._use_app_editor:
                        fut = asyncio.run_coroutine_threadsafe(
                            self.set_editor_value(new_data), self._loop)
                        fut.result()
                    layout_func_changed = self._reload_app_file()
                    if layout_func_changed:
                        fut = asyncio.run_coroutine_threadsafe(
                            self._app_run_layout_function(
                                True, with_code_editor=False, reload=True),
                            self._loop)
                        fut.result()

    def _reload_app_file(self):
        comps = self._uid_to_comp
        callback_dict = {}
        for k, v in comps.items():
            cb = v.get_callback()
            if cb is not None:
                callback_dict[k] = cb
        new_cb, code_changed = self._get_app_dynamic_cls().reload_obj_methods(
            self, callback_dict)
        self._get_app_service_unit().reload_metas()

        for k, v in comps.items():
            if k in new_cb:
                v.set_callback(new_cb[k])
        return App.app_create_layout.__name__ in code_changed

    async def handle_code_editor_event(self, event: AppEditorFrontendEvent):
        """override this method to support vscode editor.
        """
        if self._use_app_editor:
            if event.type == AppEditorFrontendEventType.Save:
                with self._watch_lock:
                    self._watchdog_ignore_next = True
                    with open(inspect.getfile(type(self)), "w") as f:
                        f.write(event.data)
                    self.code_editor.value = event.data
                    self._watchdog_prev_content = event.data
                    layout_func_changed = self._reload_app_file()
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
                 maxqsize: int = 10) -> None:
        super().__init__(True, use_app_editor, flex_flow, justify_content,
                         align_items, maxqsize)
