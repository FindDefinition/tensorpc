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
"""Flow APP: simple GUI application in devflow
Reload System

Layout Instance: App itself and layout objects created on AnyFlexLayout.



"""
import ast
import asyncio
import base64
import contextlib
import contextvars
import dataclasses
import enum
import importlib
import importlib.machinery
import inspect
import io
import json
import pickle
import runpy
import sys
import threading
import time
import tokenize
import traceback
import types
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import (Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
                    Iterable, List, Optional, Set, Tuple, Type, TypeVar, Union)

import numpy as np
import watchdog
import watchdog.events
from typing_extensions import ParamSpec
from watchdog.observers import Observer
from watchdog.observers.api import ObservedWatch

from tensorpc import simple_chunk_call_async
from tensorpc.autossh.coretypes import SSHTarget
from tensorpc.constants import PACKAGE_ROOT, TENSORPC_FILE_NAME_PREFIX, TENSORPC_FLOW_FUNC_META_KEY, TENSORPC_OBSERVED_FUNCTION_ATTR
from tensorpc.core.asynctools import cancel_task
from tensorpc.core.defs import FileResource
from tensorpc.core.funcid import remove_common_indent_from_code
from tensorpc.core.inspecttools import get_all_members_by_type
from tensorpc.core.moduleid import (get_qualname_of_type, is_lambda,
                                    is_tensorpc_dynamic_path,
                                    is_valid_function, loose_isinstance)
from tensorpc.core.rprint_dispatch import rprint
from tensorpc.core.serviceunit import (ObjectReloadManager,
                                       ObservedFunctionRegistryProtocol,
                                       ReloadableDynamicClass,
                                       ServFunctionMeta, ServiceUnit,
                                       SimpleCodeManager, get_qualname_to_code)
from tensorpc.flow.client import MasterMeta
from tensorpc.flow.constants import TENSORPC_FLOW_COMP_UID_TEMPLATE_SPLIT, TENSORPC_FLOW_EFFECTS_OBSERVE
from tensorpc.core.tree_id import UniqueTreeId, UniqueTreeIdForTree
from tensorpc.flow.coretypes import ScheduleEvent, StorageDataItem

from tensorpc.flow.flowapp.components.plus.objinspect.inspector import get_exception_frame_stack
from tensorpc.flow.flowapp.components.plus.objinspect.treeitems import TraceTreeItem
from tensorpc.flow.flowapp.reload import (AppReloadManager,
                                          bind_and_reset_object_methods,
                                          reload_object_methods)
from tensorpc.flow.jsonlike import JsonLikeNode, parse_obj_to_jsonlike
from tensorpc.flow.langserv.pyrightcfg import LanguageServerConfig
from tensorpc.flow.marker import AppFunctionMeta, AppFuncType
from tensorpc.flow.serv_names import serv_names
from tensorpc.utils.registry import HashableRegistry
from tensorpc.utils.reload import reload_method
from tensorpc.utils.uniquename import UniqueNamePool

from .appcore import (ALL_OBSERVED_FUNCTIONS, AppContext, AppSpecialEventType,
                      _CompReloadMeta, Event, create_reload_metas)
from .appcore import enter_app_conetxt
from .appcore import enter_app_conetxt as _enter_app_conetxt
from .appcore import get_app, get_app_context
from .components import mui, plus, three
from tensorpc.core.tracer import FrameResult, Tracer, TraceType
from .core import (AppComponentCore, AppEditorEvent, AppEditorEventType,
                   AppEditorFrontendEvent, AppEditorFrontendEventType,
                   AppEvent, AppEventType, BasicProps, Component,
                   ContainerBase, CopyToClipboardEvent, EventHandler,
                   FlowSpecialMethods, ForEachResult, FrontendEventType,
                   LayoutEvent, TaskLoopEvent, UIEvent, UIExceptionEvent,
                   UIRunStatus, UIType, UIUpdateEvent, Undefined, UserMessage,
                   ValueType, undefined)
from tensorpc.core.event_emitter.aio import AsyncIOEventEmitter



ALL_APP_EVENTS = HashableRegistry()
P = ParamSpec('P')

T = TypeVar('T')

T_comp = TypeVar("T_comp")


def _run_func_with_app(app, func: Callable[P, T], *args: P.args,
                       **kwargs: P.kwargs) -> T:
    with _enter_app_conetxt(app):
        return func(*args, **kwargs)


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

        # for object inspector only
        # TODO better way to implement
        self.external_path: Optional[str] = None

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


T = TypeVar("T")


@dataclasses.dataclass
class _LayoutObserveMeta:
    # one type (base class) may related to multiple layouts
    layouts: Dict[Union[mui.FlexBox, "App"], Optional[Callable[[mui.FlexBox, ServFunctionMeta],
                                Coroutine[None, None, Optional[mui.FlexBox]]]]]
    qualname_prefix: str
    # if type is None, it means they are defined in global scope.
    type: Type
    is_leaf: bool
    metas: List[ServFunctionMeta]
    # callback: Optional[Callable[[mui.FlexBox, ServFunctionMeta],
    #                             Coroutine[None, None, Optional[mui.FlexBox]]]]


@dataclasses.dataclass
class _WatchDogWatchEntry:
    obmetas: Dict[ObjectReloadManager.TypeUID, _LayoutObserveMeta]
    watch: Optional[ObservedWatch]

class _FlowAppObserveContext:
    def __init__(self) -> None:
        self._removed_layouts: List[mui.FlexBox] = []
        self._added_layouts_and_cbs: Dict[mui.FlexBox, Optional[Callable[[mui.FlexBox, ServFunctionMeta],
                                    Coroutine]]] = {}
        self._reloaded_layout_pairs: List[Tuple[mui.FlexBox, mui.FlexBox]] = []

_FLOWAPP_OBSERVE_CONTEXT: contextvars.ContextVar[
    Optional[_FlowAppObserveContext]] = contextvars.ContextVar("_FLOWAPP_OBSERVE_CONTEXT", default=None)

def _get_flowapp_observe_context():
    return _FLOWAPP_OBSERVE_CONTEXT.get()

@contextlib.contextmanager
def _enter_flowapp_observe_context(ctx: _FlowAppObserveContext):
    token = _FLOWAPP_OBSERVE_CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _FLOWAPP_OBSERVE_CONTEXT.reset(token)


class App:
    """
    App Init Callbacks:
    1. app init/app init async
    2. set_persist_props_async for all comps
    """

    def __init__(self,
                 flex_flow: Union[str, Undefined] = "column nowrap",
                 maxqsize: int = 10,
                 enable_value_cache: bool = False,
                 external_root: Optional[mui.FlexBox] = None,
                 external_wrapped_obj: Optional[Any] = None,
                 reload_manager: Optional[AppReloadManager] = None) -> None:
        # self._uid_to_comp: Dict[str, Component] = {}
        self._queue: "asyncio.Queue[AppEvent]" = asyncio.Queue(
            maxsize=maxqsize)
        if reload_manager is None:
            reload_manager = AppReloadManager(ALL_OBSERVED_FUNCTIONS)
        # self._flow_reload_manager = reload_manager

        self._flow_app_comp_core = AppComponentCore(self._queue,
                                                    reload_manager)
        self._send_callback: Optional[Callable[[AppEvent],
                                               Coroutine[None, None,
                                                         None]]] = None
        self._is_external_root = False
        self._use_app_editor = False
        # self.__flowapp_external_wrapped_obj = external_wrapped_obj
        root_uid = UniqueTreeId.from_parts([_ROOT])

        if external_root is not None:
            # TODO better mount
            root = external_root
            external_root._flow_uid = root_uid
            # if root._children is not None:
            #     # consume this _children
            #     root.add_layout(root._children)
            #     root._children = None
            # layout saved in external_root
            # self._uid_to_comp = root._uid_to_comp
            root._attach(root_uid, self._flow_app_comp_core)

            self._is_external_root = True
        else:
            root = mui.FlexBox(uid=root_uid,
                               app_comp_core=self._flow_app_comp_core)
            root.prop(flexFlow=flex_flow)
            if external_wrapped_obj is not None:
                root._wrapped_obj = external_wrapped_obj
                self._is_external_root = True

        # self._uid_to_comp[_ROOT] = root
        self.root = root.prop(minHeight=0, minWidth=0)
        self._enable_editor = False
        self._dialog_z_index: Optional[int] = None
        self._flowapp_special_eemitter: AsyncIOEventEmitter[
            AppSpecialEventType, Any] = AsyncIOEventEmitter()
        self._flowapp_thread_id = threading.get_ident()
        self._flowapp_enable_exception_inspect: bool = False

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
        # for app and dynamic layout in AnyFlexLayout
        self._flowapp_change_observers: Dict[str, _WatchDogWatchEntry] = {}

        self._flowapp_is_inited: bool = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._flowapp_enable_lsp: bool = False
        self._flowapp_internal_lsp_config: LanguageServerConfig = LanguageServerConfig(
        )
        self._flowapp_internal_lsp_config.python.analysis.pythonPath = sys.executable
        self._flowapp_observed_func_registry: Optional[
            ObservedFunctionRegistryProtocol] = None
        self._flowapp_file_resource_handlers: Dict[str, Callable[[], Union[
            bytes, FileResource, Coroutine[None, None,
                                           Union[bytes, FileResource]]]]] = {}

    @property
    def _flow_reload_manager(self):
        return self._flow_app_comp_core.reload_mgr

    def add_file_resource(
        self, key: str,
        handler: Callable[..., Union[bytes, FileResource,
                                    Coroutine[None, None,
                                              Union[bytes, FileResource]]]]):
        self._flowapp_file_resource_handlers[key] = handler

    def remove_file_resource(
        self,
        key: str,
    ):
        if key in self._flowapp_file_resource_handlers:
            self._flowapp_file_resource_handlers.pop(key)

    def set_enable_language_server(self, enable: bool):
        """must be setted before app init (in layout function), only valid
        in app init. layout reload won't change this setting
        """
        self._flowapp_enable_lsp = enable

    def get_language_server_settings(self):
        """must be setted before app init (in layout function), only valid
        in app init. layout reload won't change this setting
        """
        return self._flowapp_internal_lsp_config

    def set_observed_func_registry(self,
                                   registry: ObservedFunctionRegistryProtocol):
        self._flowapp_observed_func_registry = registry
        self._flow_reload_manager.update_observed_registry(registry)

    def register_app_special_event_handler(self, type: AppSpecialEventType,
                                           handler: Callable[[Any],
                                                             mui._CORO_NONE]):
        assert isinstance(type, AppSpecialEventType)
        self._flowapp_special_eemitter.on(type, handler)

    def unregister_app_special_event_handler(
            self, type: AppSpecialEventType,
            handler: Callable[[Any], mui._CORO_NONE]):
        assert isinstance(type, AppSpecialEventType)
        self._flowapp_special_eemitter.remove_listener(type, handler)

    def unregister_app_special_event_handlers(self, type: AppSpecialEventType):
        assert isinstance(type, AppSpecialEventType)
        self._flowapp_special_eemitter.remove_all_listeners(type)

    def _get_user_app_object(self):
        if self._is_external_root:
            if self.root._wrapped_obj is not None:
                return self.root._wrapped_obj
            return self.root
        else:
            return self

    def _is_wrapped_obj(self):
        return self._is_external_root and self.root._wrapped_obj is not None

    async def save_data_storage(self,
                                key: str,
                                data: Any,
                                node_id: Optional[str] = None,
                                graph_id: Optional[str] = None,
                                in_memory_limit: int = 1000):
        data_enc = pickle.dumps(data)
        assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        if graph_id is None:
            graph_id = self.__flowapp_master_meta.graph_id
        if node_id is None:
            node_id = self.__flowapp_master_meta.node_id
        meta = parse_obj_to_jsonlike(data, key, UniqueTreeIdForTree.from_parts([key]))
        in_memory_limit_bytes = in_memory_limit * 1024 * 1024
        meta.userdata = {
            "timestamp": time.time_ns(),
        }
        item = StorageDataItem(data_enc, meta)
        if len(data_enc) <= in_memory_limit_bytes:
            self.__flowapp_storage_cache[key] = item
        if len(data_enc) > in_memory_limit_bytes:
            raise ValueError("you can't store object more than 1GB size",
                             len(data_enc))
        await simple_chunk_call_async(self.__flowapp_master_meta.grpc_url,
                                      serv_names.FLOW_DATA_SAVE, graph_id,
                                      node_id, key, data_enc, meta,
                                      item.timestamp)

    async def read_data_storage(self,
                                key: str,
                                node_id: Optional[str] = None,
                                graph_id: Optional[str] = None,
                                in_memory_limit: int = 100):
        meta = self.__flowapp_master_meta
        assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        if graph_id is None:
            graph_id = self.__flowapp_master_meta.graph_id
        if node_id is None:
            node_id = self.__flowapp_master_meta.node_id
        if key in self.__flowapp_storage_cache:
            item_may_invalid = self.__flowapp_storage_cache[key]
            res = await simple_chunk_call_async(meta.grpc_url,
                                                serv_names.FLOW_DATA_READ,
                                                graph_id, node_id, key,
                                                item_may_invalid.timestamp)
            if res is None:
                return pickle.loads(item_may_invalid.data)
            else:
                return pickle.loads(res.data)
        else:
            res: StorageDataItem = await simple_chunk_call_async(
                meta.grpc_url, serv_names.FLOW_DATA_READ, graph_id, node_id,
                key)
            in_memory_limit_bytes = in_memory_limit * 1024 * 1024
            data = pickle.loads(res.data)
            if len(res.data) <= in_memory_limit_bytes:
                self.__flowapp_storage_cache[key] = res
            return data

    async def remove_data_storage_item(self,
                                       key: Optional[str],
                                       node_id: Optional[str] = None,
                                       graph_id: Optional[str] = None):
        meta = self.__flowapp_master_meta
        assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        if graph_id is None:
            graph_id = self.__flowapp_master_meta.graph_id
        if node_id is None:
            node_id = self.__flowapp_master_meta.node_id
        await simple_chunk_call_async(meta.grpc_url,
                                      serv_names.FLOW_DATA_DELETE_ITEM,
                                      graph_id, node_id, key)
        if key is None:
            self.__flowapp_storage_cache.clear()
        else:
            if key in self.__flowapp_storage_cache:
                self.__flowapp_storage_cache.pop(key)

    async def rename_data_storage_item(self,
                                       key: str,
                                       newname: str,
                                       node_id: Optional[str] = None,
                                       graph_id: Optional[str] = None):
        meta = self.__flowapp_master_meta
        assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        if graph_id is None:
            graph_id = self.__flowapp_master_meta.graph_id
        if node_id is None:
            node_id = self.__flowapp_master_meta.node_id
        await simple_chunk_call_async(meta.grpc_url,
                                      serv_names.FLOW_DATA_RENAME_ITEM,
                                      graph_id, node_id, key, newname)
        if key in self.__flowapp_storage_cache:
            if newname not in self.__flowapp_storage_cache:
                item = self.__flowapp_storage_cache.pop(key)
                self.__flowapp_storage_cache[newname] = item

    async def list_data_storage(self,
                                node_id: Optional[str] = None,
                                graph_id: Optional[str] = None):
        meta = self.__flowapp_master_meta
        assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        if graph_id is None:
            graph_id = self.__flowapp_master_meta.graph_id
        if node_id is None:
            node_id = self.__flowapp_master_meta.node_id
        res: List[dict] = await simple_chunk_call_async(
            meta.grpc_url, serv_names.FLOW_DATA_LIST_ITEM_METAS, graph_id,
            node_id)
        for x in res:
            # TODO remove this (old style uid compat)
            if "|" not in x["id"]:
                x["id"] = UniqueTreeId.from_parts([x["id"]]).uid_encoded
        return [JsonLikeNode(**x) for x in res]

    async def list_all_data_storage_nodes(self,
                                          graph_id: Optional[str] = None):
        meta = self.__flowapp_master_meta
        assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        if graph_id is None:
            graph_id = self.__flowapp_master_meta.graph_id
        res: List[str] = await simple_chunk_call_async(
            meta.grpc_url, serv_names.FLOW_DATA_QUERY_DATA_NODE_IDS, graph_id)
        return res

    async def get_ssh_node_data(self, node_id: str):
        meta = self.__flowapp_master_meta
        assert self.__flowapp_master_meta.is_inside_devflow, "you must call this in devflow apps."
        res: SSHTarget = await simple_chunk_call_async(
            meta.grpc_url, serv_names.FLOW_GET_SSH_NODE_DATA, meta.graph_id,
            node_id)
        return res

    def get_persist_storage(self):
        return self.__persist_storage

    def get_observed_func_registry(self):
        registry = self._flowapp_observed_func_registry
        if registry is None:
            registry = ALL_OBSERVED_FUNCTIONS
        return registry

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
                state[comp._flow_uid_encoded] = {
                    "type": comp._flow_comp_type.value,
                    "props": comp.get_sync_props(),
                }
            # user state
            st = comp.get_persist_props()
            if st is not None:
                user_state[comp._flow_uid_encoded] = {
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
        uid_to_comp = self.root._get_uid_encoded_to_comp_dict()
        if self._enable_value_cache:
            ev = AppEvent("", {})
            for k, s in uistate.items():
                if k in uid_to_comp:
                    comp_to_restore = uid_to_comp[k]
                    if comp_to_restore._flow_comp_type.value == s["type"]:
                        comp_to_restore.set_props(s["props"])
                        ev += comp_to_restore.get_sync_event(True)
            with _enter_app_conetxt(self):
                for k, s in userstate.items():
                    if k in uid_to_comp:
                        comp_to_restore = uid_to_comp[k]
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

    async def _app_run_layout_function(
        self,
        send_layout_ev: bool = False,
        with_code_editor: bool = True,
        reload: bool = False,
        decorator_fn: Optional[Callable[[], Union[mui.LayoutType,
                                                  mui.FlexBox]]] = None):
        self.root._prevent_add_layout = False
        prev_comps = self.__previous_error_sync_props.copy()
        prev_user_states = self.__previous_error_persist_state.copy()
        uid_to_comp = self.root._get_uid_encoded_to_comp_dict()
        if reload:
            for u, c in uid_to_comp.items():
                prev_comps[u] = c._to_dict_with_sync_props()
                user_state = c.get_persist_props()
                if user_state is not None:
                    prev_user_states[u] = {
                        "type": c._flow_comp_type.value,
                        "state": user_state,
                    }
        if reload:
            detached = self.root._detach()
            # make sure did_mount is called from root to leaf (breadth first order)
            detached_items = list(detached.items())
            detached_items.sort(key=lambda x: len(x[0].split(".")),
                                reverse=False)

            await self.root._run_special_methods(
                [], [x[1] for x in detached_items], self._flow_reload_manager)
            del detached
        root_uid = UniqueTreeId.from_parts([_ROOT])

        await self.root._clear()
        # self._uid_to_comp.clear()
        self.root._flow_uid = root_uid
        new_is_flex = False
        res: mui.LayoutType = {}
        wrapped_obj = self.root._wrapped_obj
        attached: Dict[str, Component] = {}
        try:
            with _enter_app_conetxt(self):
                if decorator_fn is not None:
                    temp_res = decorator_fn()
                    if isinstance(temp_res, mui.FlexBox):
                        # if temp_res._children is not None:
                        #     # consume this _children
                        #     temp_res.add_layout(temp_res._children)
                        #     temp_res._children = None
                        # temp_res._flow_uid = _ROOT
                        attached = temp_res._attach(root_uid,
                                                    self._flow_app_comp_core)
                        # self._uid_to_comp = temp_res._uid_to_comp
                        new_is_flex = True
                        self.root = temp_res
                        self.root._wrapped_obj = wrapped_obj
                    else:
                        res = temp_res
                else:
                    res = self.app_create_layout()
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
        if not new_is_flex:
            if isinstance(res, list):
                res = {str(i): v for i, v in enumerate(res)}
            res_anno: Dict[str, Component] = {**res}
            self.root.add_layout(res_anno)
            attached = self.root._attach(root_uid, self._flow_app_comp_core)
        uid_to_comp = self.root._get_uid_encoded_to_comp_dict()
        # self._uid_to_comp[_ROOT] = self.root
        self.root._prevent_add_layout = True
        if reload:
            # comps = self.root._get_all_nested_childs()
            with _enter_app_conetxt(self):
                for comp in uid_to_comp.values():
                    if comp._flow_uid_encoded in prev_comps:
                        if comp._flow_comp_type.value == prev_comps[
                                comp._flow_uid_encoded]["type"]:
                            comp.set_props(prev_comps[comp._flow_uid_encoded]["props"])
                    if comp._flow_uid_encoded in prev_user_states:
                        if comp._flow_comp_type.value == prev_user_states[
                                comp._flow_uid_encoded]["type"]:
                            await comp.set_persist_props_async(
                                prev_user_states[comp._flow_uid_encoded]["state"])
            del prev_comps
            del prev_user_states

        if send_layout_ev:
            ev = AppEvent(
                "", {
                    AppEventType.UpdateLayout:
                    LayoutEvent(self._get_app_layout(with_code_editor))
                })
            await self._queue.put(ev)
            if reload:
                # make sure did_mount is called from leaf to root (reversed breadth first order)
                attached_items = list(attached.items())
                attached_items.sort(key=lambda x: len(x[0].split(".")),
                                    reverse=True)

                await self.root._run_special_methods(
                    [x[1] for x in attached_items], [],
                    self._flow_reload_manager)

    def app_initialize(self):
        """override this to init app before server start
        """
        pass

    async def app_initialize_async(self):
        """override this to init app before server start
        """
        self._loop = asyncio.get_running_loop()
        uid_to_comp = self.root._get_uid_to_comp_dict()
        # make sure did_mount is called from leaf to root (reversed breadth first order)
        uid_to_comp_items = list(uid_to_comp.items())
        uid_to_comp_items.sort(key=lambda x: len(x[0].parts),
                               reverse=True)
        with enter_app_conetxt(self):
            for _, v in uid_to_comp_items:
                special_methods = v.get_special_methods(
                    self._flow_reload_manager)
                if special_methods.did_mount is not None:
                    await v.run_callback(
                        special_methods.did_mount.get_binded_fn(),
                        sync_status_first=False,
                        change_status=False)
                for k, effects in v.effects._flow_effects.items():
                    for effect in effects:
                        res = await v.run_callback(
                            effect,
                            sync_status_first=False,
                            change_status=False)
                        if res is not None:
                            # res is effect
                            v.effects._flow_unmounted_effects[k].append(res)

    def app_terminate(self):
        """override this to init app after server stop
        """
        pass

    async def app_terminate_async(self):
        """override this to init app after server stop
        """
        uid_to_comp = self.root._get_uid_encoded_to_comp_dict()
        with enter_app_conetxt(self):
            for v in uid_to_comp.values():
                special_methods = v.get_special_methods(
                    self._flow_reload_manager)
                if special_methods.will_unmount is not None:
                    await v.run_callback(
                        special_methods.will_unmount.get_binded_fn(),
                        sync_status_first=False,
                        change_status=False)

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

    def __repr__(self):
        return f"App[{self._get_app_dynamic_cls().module_path}]"

    def _get_app_service_unit(self):
        assert self._app_service_unit is not None
        return self._app_service_unit

    def _get_app_layout(self, with_code_editor: bool = True):
        uid_to_comp = self.root._get_uid_encoded_to_comp_dict()
        # print({k: v._flow_uid for k, v in uid_to_comp.items()})
        res = {
            "layout": {
                u: c.to_dict()
                for u, c in uid_to_comp.items()
            },
            "enableEditor": self._enable_editor,
            "fallback": "",
        }
        if self._dialog_z_index is not None:
            res["zIndex"] = self._dialog_z_index
        # if with_code_editor:
        #     res.update({
        #         "codeEditor": self.code_editor.get_state(),
        #     })
        # node_layout = self.app_create_node_layout()
        # if node_layout is not None:
        #     res["nodeLayout"] = mui.layout_unify(node_layout)
        # side_layout = self.app_create_side_layout()
        # if side_layout is not None:
        #     res["sideLayout"] = mui.layout_unify(side_layout)
        return res

    def _get_app_editor_state(self):
        res = {
            "enableEditor": self._enable_editor,
            "codeEditor": self.code_editor.get_state(),
        }
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

    def set_editor_value_event(self,
                               value: str,
                               language: str = "",
                               lineno: Optional[int] = None):
        self.code_editor.value = value
        if language:
            self.code_editor.language = language
        res: Dict[str, Any] = {
            "value": self.code_editor.value,
            "language": self.code_editor.language,
        }
        if lineno is not None:
            res["lineno"] = lineno
        app_ev = AppEditorEvent(AppEditorEventType.SetValue, res)
        return app_ev

    async def set_editor_value(self,
                               value: str,
                               language: str = "",
                               lineno: Optional[int] = None):
        """use this method to set editor value and language.
        """
        await self._send_editor_event(
            self.set_editor_value_event(value, language, lineno))

    @staticmethod
    async def __handle_dnd_event(handler: EventHandler,
                                 src_handler: EventHandler, src_event: Event):
        res = await src_handler.run_event_async(src_event)
        ev_res = Event(FrontendEventType.Drop.value, res, src_event.keys)
        await handler.run_event_async(ev_res)

    def _is_editable_app(self):
        return isinstance(self, EditableApp)

    def _get_self_as_editable_app(self):
        assert isinstance(self, EditableApp)
        return self

    async def handle_event(self, ev: UIEvent, is_sync: bool = False):
        res: Dict[str, Any] = {}
        for uid, data in ev.uid_to_data.items():
            keys: Union[Undefined, List[str]] = undefined
            uid_original = uid

            if TENSORPC_FLOW_COMP_UID_TEMPLATE_SPLIT in uid:
                parts = uid.split(TENSORPC_FLOW_COMP_UID_TEMPLATE_SPLIT)
                uid = parts[0]
                keys = parts[1:]
            indexes = undefined
            if len(data) == 3 and data[2] is not None:
                indexes = list(map(int, data[2].split(".")))
            event = Event(data[0], data[1], keys, indexes)
            if event.type == FrontendEventType.Drop.value:
                # TODO add event context stack here.
                src_data = data[1]
                src_uid = src_data["uid"]
                src_comp = self.root._get_comp_by_uid(src_uid)
                collect_handlers = src_comp.get_event_handlers(
                    FrontendEventType.DragCollect.value)
                comp = self.root._get_comp_by_uid(uid)
                handlers = comp.get_event_handlers(data[0])
                # print(src_uid, comp, src_comp, handler, collect_handler)
                if handlers is not None and collect_handlers is not None:
                    src_event = Event(FrontendEventType.DragCollect.value,
                                      src_data["data"], keys, indexes)
                    cbs = []
                    for handler in handlers.handlers:
                        cb = partial(self.__handle_dnd_event,
                                     handler=handler,
                                     src_handler=collect_handlers.handlers[0],
                                     src_event=src_event)
                        cbs.append(cb)
                    comp._task = asyncio.create_task(
                        comp.run_callbacks(cbs, sync_status_first=False))
            elif event.type == FrontendEventType.FileDrop.value:
                # for file drop, we can't use regular drop above, so
                # just convert it to drop event, no drag collect needed.
                comps = self.root._get_comps_by_uid(uid)
                ctxes = [
                    c._flow_event_context_creator() for c in comps
                    if c._flow_event_context_creator is not None
                ]
                with contextlib.ExitStack() as stack:
                    for ctx in ctxes:
                        stack.enter_context(ctx)
                    res[uid_original] = await comps[-1].handle_event(
                        Event(FrontendEventType.Drop.value, data[1], keys,
                              indexes),
                        is_sync=is_sync)
            else:
                comps = self.root._get_comps_by_uid(uid)
                ctxes = [
                    c._flow_event_context_creator() for c in comps
                    if c._flow_event_context_creator is not None
                ]
                # comps[-1].flow_event_emitter.emit(event.type, event.data)
                with contextlib.ExitStack() as stack:
                    for ctx in ctxes:
                        stack.enter_context(ctx)
                    res[uid_original] = await comps[-1].handle_event(
                        event, is_sync=is_sync)
        if is_sync:
            return res

    async def _handle_event_with_ctx(self, ev: UIEvent, is_sync: bool = False):
        # TODO run control from other component
        with _enter_app_conetxt(self):
            return await self.handle_event(ev, is_sync)

    async def _run_autorun(self, cb: Callable):
        try:
            coro = cb()
            if inspect.iscoroutine(coro):
                await coro
            self._flowapp_special_eemitter.emit(AppSpecialEventType.AutoRunEnd,
                                                None)
        except:
            traceback.print_exc()
            if self._flowapp_enable_exception_inspect:
                await self._inspect_exception()

    async def _inspect_exception(self):
        try:
            comp = self.find_component(plus.ObjectInspector)
            if comp is not None and comp.enable_exception_inspect:
                await comp.set_object(get_exception_frame_stack(), "exception")
        except:
            traceback.print_exc()

    def _inspect_exception_sync(self):
        try:
            comp = self.find_component(plus.ObjectInspector)
            if comp is not None and comp.enable_exception_inspect:
                comp.set_object_sync(get_exception_frame_stack(), "exception")
        except:
            traceback.print_exc()

    async def copy_text_to_clipboard(self, text: str):
        """copy to clipboard in frontend."""
        await self._queue.put(
            AppEvent(
                "",
                {AppEventType.CopyToClipboard: CopyToClipboardEvent(text)}))

    def find_component(
            self,
            type: Type[T],
            validator: Optional[Callable[[T], bool]] = None) -> Optional[T]:
        """find component in comp tree. breath-first.
        """
        res: List[Optional[T]] = [None]

        def handler(name, comp):
            if loose_isinstance(comp, (type, )):
                if (validator is None) or (validator is not None
                                           and validator(comp)):
                    res[0] = comp
                    return ForEachResult.Return
            elif isinstance(
                    comp, mui.FlexBox
            ) and comp._wrapped_obj is not None and loose_isinstance(
                    comp._wrapped_obj, (type, )):
                if (validator is None) or (validator is not None
                                           and validator(comp._wrapped_obj)):
                    res[0] = comp._wrapped_obj
                    return ForEachResult.Return

        self.root._foreach_comp(handler)
        return res[0]

    def find_all_components(
            self,
            type: Type[T],
            check_nested: bool = False,
            validator: Optional[Callable[[T], bool]] = None) -> List[T]:
        res: List[T] = []

        def handler(name, comp):
            if isinstance(comp, type):
                if (validator is None) or (validator is not None
                                           and validator(comp)):
                    res.append(comp)
                    # tell foreach to continue instead of search children
                    if not check_nested:
                        return ForEachResult.Continue

        self.root._foreach_comp(handler)
        return res

    async def _recover_code_editor(self):
        if self._use_app_editor:
            obj = type(self._get_user_app_object())
            lines, lineno = inspect.findsource(obj)
            await self.set_editor_value(value="".join(lines), lineno=lineno)


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
                 maxqsize: int = 10,
                 observed_files: Optional[List[str]] = None,
                 external_root: Optional[mui.FlexBox] = None,
                 external_wrapped_obj: Optional[Any] = None,
                 reload_manager: Optional[AppReloadManager] = None) -> None:
        super().__init__(flex_flow,
                         maxqsize,
                         external_root=external_root,
                         external_wrapped_obj=external_wrapped_obj,
                         reload_manager=reload_manager)
        self._use_app_editor = use_app_editor
        if use_app_editor:
            obj = type(self._get_user_app_object())
            lines, lineno = inspect.findsource(obj)
            self.code_editor.value = "".join(lines)
            self.code_editor.language = "python"
            self.code_editor.set_init_line_number(lineno)
            self.code_editor.freeze()
        self._watchdog_prev_content = ""
        self._flow_reloadable_layout = reloadable_layout
        if reloadable_layout:
            self._app_force_use_layout_function()
        self._flow_observed_files = observed_files
        self._init_observe_paths: Set[str] = set()
        self._protect_app_observe_call: bool = False

    @contextlib.contextmanager
    def _flowapp_protect_app_observe_call(self, ctx: _FlowAppObserveContext):
        try:
            self._protect_app_observe_call = True
            with _enter_flowapp_observe_context(ctx) as ctx:
                yield ctx
        finally:
            self._protect_app_observe_call = False

    def app_initialize(self):
        super().app_initialize()
        dcls = self._get_app_dynamic_cls()
        path = dcls.file_path
        user_obj = self._get_user_app_object()
        metas_dict = self._flow_reload_manager.query_type_method_meta_dict(
            type(user_obj))

        # for m in metas:
        #     m.bind(user_obj)
        # qualname_prefix = type(user_obj).__qualname__
        obentry = _WatchDogWatchEntry({}, None)
        for meta_type_uid, meta_item in metas_dict.items():
            if meta_item.type is not None:
                # TODO should we ignore global functions?
                qualname_prefix = meta_type_uid[1]
                obmeta = _LayoutObserveMeta({self: None}, qualname_prefix,
                                            meta_item.type, meta_item.is_leaf,
                                            meta_item.metas)
                obentry.obmetas[meta_type_uid] = obmeta
        # obentry = _WatchDogWatchEntry(
        #     [_LayoutObserveMeta(self, qualname_prefix, metas, None)], None)
        self._flowapp_change_observers[path] = obentry
        self._watchdog_watcher = None
        self._watchdog_observer = None
        registry = self.get_observed_func_registry()
        if not self._flow_app_is_headless:
            observer = Observer()
            self._watchdog_watcher = _WatchDogForAppFile(
                self._watchdog_on_modified)
            if self._flow_observed_files is not None:
                for p in self._flow_observed_files:
                    assert Path(p).exists(), f"{p} must exist"
                paths = set(self._flow_observed_files)
            else:
                paths = set(self.__get_default_observe_paths())
            paths.add(str(Path(path).resolve()))

            for p in registry.get_path_to_qname().keys():
                paths.add(str(Path(p).resolve()))
            self._init_observe_paths.update(paths)
            self._flowapp_code_mgr = SimpleCodeManager(list(paths))
            paths = set(self._flowapp_code_mgr.file_to_entry.keys())
            # print(paths)
            # add all observed function paths
            for p in paths:
                observer.schedule(self._watchdog_watcher, p, recursive=False)
            observer.start()
            self._watchdog_observer = observer
        else:
            self._flowapp_code_mgr = None
        self._watchdog_ignore_next = False
        self._loop = asyncio.get_running_loop()
        self._watch_lock = threading.Lock()

    def __observe_layout_effect(self, obj: mui.FlexBox, callback: Optional[Callable[[mui.FlexBox, ServFunctionMeta],
                                    Coroutine]] = None):
        self._flowapp_observe(obj, callback)
        return partial(self._flowapp_remove_observer, obj)

    def observe_layout(
        self,
        obj: mui.FlexBox,
        callback: Optional[Callable[[mui.FlexBox, ServFunctionMeta],
                                    Coroutine]] = None):
        if not obj.effects.has_effect_key(TENSORPC_FLOW_EFFECTS_OBSERVE):
            # already observed
            obj.effects.use_effect(partial(self.__observe_layout_effect, obj, callback), key=TENSORPC_FLOW_EFFECTS_OBSERVE)
            if obj.is_mounted():
                self._flowapp_observe(obj, callback)
                # TODO better code
                obj.effects._flow_unmounted_effects[TENSORPC_FLOW_EFFECTS_OBSERVE].append(partial(self._flowapp_remove_observer, obj))

    def _flowapp_observe(
        self,
        obj: mui.FlexBox,
        callback: Optional[Callable[[mui.FlexBox, ServFunctionMeta],
                                    Coroutine]] = None):
        ctx = _get_flowapp_observe_context()
        if ctx is not None:
            ctx._added_layouts_and_cbs[obj] = callback
            return
        # TODO better error msg if app editable not enabled
        path = obj._flow_comp_def_path
        assert self._protect_app_observe_call is False, "you can't call observe inside observe reload callback"
        assert path != "" and self._watchdog_observer is not None
        path_resolved = self._flow_reload_manager._resolve_path_may_in_memory(
            path)
        if path_resolved not in self._flowapp_change_observers:
            self._flowapp_change_observers[
                path_resolved] = _WatchDogWatchEntry({}, None)
        obentry = self._flowapp_change_observers[path_resolved]
        if len(obentry.obmetas) == 0 and not is_tensorpc_dynamic_path(path):
            # no need to schedule watchdog.
            if path_resolved not in self._init_observe_paths:
                watch = self._watchdog_observer.schedule(
                    self._watchdog_watcher, path, False)
                obentry.watch = watch
        assert self._flowapp_code_mgr is not None
        if not self._flowapp_code_mgr._check_path_exists(path):
            self._flowapp_code_mgr._add_new_code(
                path, self._flow_reload_manager.in_memory_fs)
        metas_dict = self._flow_reload_manager.query_type_method_meta_dict(
            type(obj._get_user_object()))

        for meta_type_uid, meta_item in metas_dict.items():
            if meta_item.type is not None:
                if meta_type_uid in obentry.obmetas:
                    obentry.obmetas[meta_type_uid].layouts[obj] = callback
                else:
                    qualname_prefix = meta_type_uid[1]
                    obmeta = _LayoutObserveMeta({obj: callback}, qualname_prefix,
                                                meta_item.type,
                                                meta_item.is_leaf,
                                                meta_item.metas)
                    obentry.obmetas[meta_type_uid] = obmeta

    def _flowapp_remove_observer(self, obj: mui.FlexBox):
        ctx = _get_flowapp_observe_context()
        if ctx is not None:
            ctx._removed_layouts.append(obj)
            return
        path = obj._flow_comp_def_path
        assert path != "" and self._watchdog_observer is not None
        path_resolved = self._flow_reload_manager._resolve_path_may_in_memory(
            path)
        assert self._protect_app_observe_call is False, "you can't call remove observer inside observe reload callback"
        assert self._flowapp_code_mgr is not None
        # self._flowapp_code_mgr._remove_path(path)
        if path_resolved in self._flowapp_change_observers:
            obentry = self._flowapp_change_observers[path_resolved]
            types_to_remove: List[ObjectReloadManager.TypeUID] = []
            for k, v in obentry.obmetas.items():
                if obj in v.layouts:
                    v.layouts.pop(obj)
                if len(v.layouts) == 0:
                    types_to_remove.append(k)
            for k in types_to_remove:
                del obentry.obmetas[k]
            # new_obmetas: List[_LayoutObserveMeta] = []
            # for obmeta in obentry.obmetas:
            #     if obj is not obmeta.layout:
            #         new_obmetas.append(obmeta)
            # obentry.obmetas = new_obmetas
            if len(obentry.obmetas) == 0 and obentry.watch is not None:
                if path_resolved not in self._init_observe_paths:
                    self._watchdog_observer.unschedule(obentry.watch)

    def __get_default_observe_paths(self):
        uid_to_comp = self.root._get_uid_encoded_to_comp_dict()
        res: Set[str] = set()
        for k, v in uid_to_comp.items():
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

    def __get_callback_metas_in_file(self, change_file: str,
                                     layout: mui.FlexBox):
        uid_to_comp = layout._get_uid_encoded_to_comp_dict()
        resolved_path = self._flow_reload_manager._resolve_path_may_in_memory(
            change_file)
        return create_reload_metas(uid_to_comp, resolved_path)

    async def _reload_object_with_new_code(self,
                                           path: str,
                                           new_code: Optional[str] = None):
        """reload order:
        for leaf type, we will support new method. if code change, we will replace all method with reloaded methods.
        for base types, we don't support new method. if code change, only reload method with same name. if the method
            is already reloaded in child type, it will be ignored.
        """
        assert self._flowapp_code_mgr is not None
        resolved_path = self._flowapp_code_mgr._resolve_path(path)
        if self._use_app_editor:
            dcls = self._get_app_dynamic_cls()
            resolved_app_path = self._flowapp_code_mgr._resolve_path(
                dcls.file_path)

            if resolved_path == resolved_app_path and new_code is not None:
                await self.set_editor_value(new_code)
        try:
            changes = self._flowapp_code_mgr.update_code(
                resolved_path, new_code)
        except:
            # ast parse error
            traceback.print_exc()
            return
        if changes is None:
            return
        rprint(f"<watchdog> {path}")
        new, change, _ = changes
        new_data = self._flowapp_code_mgr.get_code(resolved_path)
        is_reload = False
        is_callback_change = False
        callbacks_of_this_file: Optional[List[_CompReloadMeta]] = None
        observe_ctx = _FlowAppObserveContext()
        try:
            with self._flowapp_protect_app_observe_call(observe_ctx):
                if resolved_path in self._flowapp_change_observers:
                    obmetas = self._flowapp_change_observers[
                        resolved_path].obmetas.copy()
                    obmetas_items = list(obmetas.items())
                    # sort obmetas_items by mro
                    obmetas_items.sort(key=lambda x: len(x[1].type.mro()),
                                    reverse=True)
                    # store accessed metas in inheritance tree
                    resolved_metas: Dict[ObjectReloadManager.TypeUID,
                                        Set[str]] = {}
                    # print("len(obmetas)", resolved_path, len(obmetas))
                    for type_uid, obmeta in obmetas_items:
                        # get changed metas for special methods
                        # print(new, change)
                        if type_uid not in resolved_metas:
                            resolved_metas[type_uid] = set()
                        changed_metas: List[ServFunctionMeta] = []
                        for m in obmeta.metas:
                            if m.qualname in change:
                                changed_metas.append(m)
                        new_method_names: List[str] = [
                            x for x in new if x.startswith(obmeta.qualname_prefix)
                            and x != obmeta.qualname_prefix
                        ]
                        # layout = obmeta.layout
                        for layout in obmeta.layouts:
                            if not is_callback_change:
                                if isinstance(layout, App):
                                    callbacks_of_this_file = self.__get_callback_metas_in_file(
                                        resolved_path, self.root)
                                else:
                                    # TODO should we check all callbacks instead of changed layout?
                                    callbacks_of_this_file = self.__get_callback_metas_in_file(
                                        resolved_path, self.root)
                                # print(len(callbacks_of_this_file), "callbacks_of_this_file 0", change.keys())
                                for cb_meta in callbacks_of_this_file:
                                    # print(cb_meta.cb_qualname)
                                    if cb_meta.cb_qualname in change:
                                        is_callback_change = True
                                        break
                            # print("is_callback_change", is_callback_change)
                        # for m in changed_metas:
                        #     print(m.qualname, "CHANGED")
                        # do reload, run special methods
                        flow_special_for_check = FlowSpecialMethods(changed_metas)
                        do_reload = flow_special_for_check.contains_special_method(
                        ) or bool(new_method_names)
                        # print("do_reload", do_reload)
                        if not do_reload:
                            continue
                        # rprint(obmeta.layouts)
                        for i, (layout, layout_reload_cb) in enumerate(obmeta.layouts.items()):
                            changed_user_obj = None

                            if layout is self:
                                # reload app
                                if changed_metas or bool(new_method_names):
                                    # reload app servunit and method
                                    changed_user_obj = self._get_user_app_object()
                                    # self._get_app_dynamic_cls(
                                    # ).reload_obj_methods(user_obj, {}, self._flow_reload_manager)
                                    self._get_app_service_unit().reload_metas(
                                        self._flow_reload_manager)
                            else:
                                assert isinstance(layout,
                                                mui.FlexBox), f"{type(layout)}"
                                # if self.code_editor.external_path is not None and new_code is None:
                                #     if str(
                                #             Path(self.code_editor.external_path).
                                #             resolve()) == resolved_path:
                                #         await self.set_editor_value(new_data, lineno=0)
                                # reload dynamic layout
                                if changed_metas or bool(new_method_names):
                                    changed_user_obj = layout._get_user_object()
                            # print("RTX", changed_user_obj, new_method_names)
                            if changed_user_obj is not None:
                                # reload_res = self._flow_reload_manager.reload_type(
                                #     type(changed_user_obj))
                                reload_res = self._flow_reload_manager.reload_type(
                                    obmeta.type)

                                if not is_reload:
                                    is_reload = reload_res.is_reload
                                updated_type = reload_res.type_meta.get_local_type_from_module_dict(
                                    reload_res.module_entry.module_dict)
                                # recreate metas with new type and new qualname_to_code
                                # TODO handle special methods in mro
                                updated_metas = self._flow_reload_manager.query_type_method_meta(
                                    updated_type, include_base=False)
                                # if is leaf type, reload all meta, otherwise only reload meta saved initally.
                                if is_reload and obmeta.is_leaf:
                                    obmeta.metas = updated_metas
                                # print("CHANGED USER OBJ", len(obmetas))
                                changed_metas = [
                                    m for m in updated_metas
                                    if m.qualname in change
                                ]
                                if obmeta.is_leaf:
                                    changed_metas += [
                                        m for m in updated_metas
                                        if m.qualname in new
                                    ]
                                else:
                                    prev_meta_names = [
                                        x.qualname for x in obmeta.metas
                                    ]
                                    changed_metas = list(
                                        filter(
                                            lambda x: x.qualname in
                                            prev_meta_names, changed_metas))
                                changed_metas_candidate = changed_metas
                                new_changed_metas: List[ServFunctionMeta] = []
                                # if meta already reloaded in child type, ignore it.
                                for c in changed_metas_candidate:
                                    if c.name not in resolved_metas[type_uid]:
                                        resolved_metas[type_uid].add(c.name)
                                        new_changed_metas.append(c)
                                change_metas = new_changed_metas
                                for c in change_metas:
                                    c.bind(changed_user_obj)
                                    print(f"{c.name}, ------------")
                                    # print(c.code)
                                    # print("-----------")
                                    # print(change[c.qualname])
                                # we need to update metas of layout with new type.
                                # meta is binded in bind_and_reset_object_methods
                                if changed_metas:
                                    bind_and_reset_object_methods(
                                        changed_user_obj, changed_metas)
                                if layout is self:
                                    self._get_app_dynamic_cls(
                                    ).module_dict = reload_res.module_entry.module_dict
                                    self._get_app_service_unit().reload_metas(
                                        self._flow_reload_manager)
                            # use updated metas to run special methods such as create_layout and auto_run
                            if changed_metas:
                                flow_special = FlowSpecialMethods(changed_metas)
                                with _enter_app_conetxt(self):
                                    if flow_special.create_layout:
                                        fn = flow_special.create_layout.get_binded_fn(
                                        )
                                        if isinstance(layout, App):
                                            await self._app_run_layout_function(
                                                True,
                                                with_code_editor=False,
                                                reload=True,
                                                decorator_fn=fn)
                                        else:
                                            if layout_reload_cb is not None:
                                                # handle layout in callback
                                                new_layout = await layout_reload_cb(
                                                    layout,
                                                    flow_special.create_layout)
                                                # if new_layout is not None:
                                                    # obmeta.layouts[i] = new_layout
                                                    # observe_ctx._reloaded_layout_pairs.append((layout, new_layout))
                                            # dynamic layout
                                    if flow_special.create_preview_layout:
                                        if not isinstance(layout, App):
                                            if layout_reload_cb is not None:
                                                # handle layout in callback
                                                new_layout = await layout_reload_cb(
                                                    layout, flow_special.
                                                    create_preview_layout)
                                                # if new_layout is not None:
                                                    # obmeta.layouts[i] = new_layout
                                                    # observe_ctx._reloaded_layout_pairs.append((layout, new_layout))
                                    for auto_run in flow_special.auto_runs:
                                        if auto_run is not None:
                                            await self._run_autorun(
                                                auto_run.get_binded_fn())
                            # handle special methods
                ob_registry = self.get_observed_func_registry()
                observed_func_changed = ob_registry.observed_func_changed(
                    resolved_path, change)
                if observed_func_changed:
                    # this handler only handle one file, so we only need to reload once.
                    first_func_qname_pair = ob_registry.get_path_to_qname(
                    )[resolved_path][0]
                    entry = ob_registry[first_func_qname_pair[0]]
                    entry_func_unwrap = inspect.unwrap(entry.current_func)
                    reload_res = self._flow_reload_manager.reload_type(
                        entry_func_unwrap)
                    if not is_reload:
                        is_reload = reload_res.is_reload
                    # for qname in observed_func_changed:
                    with _enter_app_conetxt(self):
                        for qname in observed_func_changed:
                            entry = ob_registry[qname]
                            if len(entry.autorun_block_symbol) == 0:
                                if entry.autorun_when_changed:
                                    await self._run_autorun(entry.current_func)
                            else:
                                if entry.first_changed_block_idx < len(entry.body_code_blocks):
                                    first_changed_block_idx = entry.first_changed_block_idx
                                    if not entry.autorun_locals:
                                        first_changed_block_idx = 0
                                    code_to_run = "\n".join(entry.body_code_blocks[first_changed_block_idx:])
                                    code_to_run = remove_common_indent_from_code(code_to_run)
                                    code_locals = entry.autorun_locals
                                    func_globals = entry.current_func.__globals__
                                    code_comp = compile(code_to_run, f"<{TENSORPC_FILE_NAME_PREFIX}-scripts-{entry.qualname}>", "exec")
                                    exec(code_comp, func_globals, code_locals)
                        self._flowapp_special_eemitter.emit(
                            AppSpecialEventType.ObservedFunctionChange,
                            observed_func_changed)
                # print(is_callback_change, is_reload)
                if is_callback_change or is_reload:
                    # reset all callbacks in this file
                    if callbacks_of_this_file is None:
                        callbacks_of_this_file = self.__get_callback_metas_in_file(
                            resolved_path, self.root)
                    # print(len(callbacks_of_this_file), "callbacks_of_this_file")
                    if callbacks_of_this_file:
                        cb_real = callbacks_of_this_file[0].cb_real
                        reload_res = self._flow_reload_manager.reload_type(
                            inspect.unwrap(cb_real))
                        for meta in callbacks_of_this_file:
                            print("RELOAD CB", meta.cb_qualname)
                            handler = meta.handler
                            cb = inspect.unwrap(handler.cb)
                            new_method, _ = reload_method(
                                cb, reload_res.module_entry.module_dict)
                            if new_method is not None:
                                handler.cb = new_method

        except:
            # watchdog thread can't fail
            traceback.print_exc()
            return
        finally:
            # print("START MODIFY OMBETA")
            # for pair in observe_ctx._reloaded_layout_pairs:
            #     observe_ctx._removed_layouts.remove(pair[0])
            #     observe_ctx._added_layouts_and_cbs.pop(pair[1])
            # print(len(observe_ctx._removed_layouts), len(observe_ctx._added_layouts_and_cbs))
            for layout in observe_ctx._removed_layouts:
                self._flowapp_remove_observer(layout)
            for layout, cb in observe_ctx._added_layouts_and_cbs.items():
                self._flowapp_observe(layout, cb)
            return 

    def _watchdog_on_modified(self, ev: _WATCHDOG_MODIFY_EVENT_TYPES):
        # which event trigger reload?
        # 1. special method code change
        # 2. callback code change (handled outsite)
        # 3. new methods detected in layout

        # WARNING: other events WON'T trigger reload.

        # what happened when reload?
        # 1. all method of object (app or dynamic layout) will be reset
        # 2. all callback defined in changed file will be reset
        # 3. if layout function changed, load new layout
        # 4. if mount/unmount function changed, reset them
        # 5. if autorun changed, run them

        if isinstance(ev, watchdog.events.FileModifiedEvent):
            dcls = self._get_app_dynamic_cls()
            rprint("<watchdog>", ev)
            with self._watch_lock:
                if self._flowapp_code_mgr is None or self._loop is None:
                    return
                asyncio.run_coroutine_threadsafe(
                    self._reload_object_with_new_code(ev.src_path), self._loop)

    def _reload_app_file(self):
        # comps = self._uid_to_comp
        # callback_dict = {}
        # for k, v in comps.items():
        #     cb = v.get_callback()
        #     if cb is not None:
        #         callback_dict[k] = cb
        if self._is_external_root:
            obj = self.root
            if self.root._wrapped_obj is not None:
                obj = self.root._wrapped_obj
            new_cb, code_changed = self._get_app_dynamic_cls(
            ).reload_obj_methods(obj, {}, self._flow_reload_manager)
        else:
            new_cb, code_changed = self._get_app_dynamic_cls(
            ).reload_obj_methods(self, {}, self._flow_reload_manager)
        self._get_app_service_unit().reload_metas(self._flow_reload_manager)
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
                    if self.code_editor.external_path is not None:
                        path = self.code_editor.external_path
                    else:
                        path = app_path
                    # if self.code_editor.external_path is None:
                    with open(path, "w") as f:
                        f.write(event.data)
                    await self._reload_object_with_new_code(path, event.data)
        return


class EditableLayoutApp(EditableApp):

    def __init__(self,
                 use_app_editor: bool = True,
                 flex_flow: Union[str, Undefined] = "column nowrap",
                 maxqsize: int = 10,
                 observed_files: Optional[List[str]] = None,
                 external_root: Optional[mui.FlexBox] = None,
                 reload_manager: Optional[AppReloadManager] = None) -> None:
        super().__init__(True,
                         use_app_editor,
                         flex_flow,
                         maxqsize,
                         observed_files,
                         external_root=external_root,
                         reload_manager=reload_manager)


async def _run_zeroarg_func(cb: Callable):
    try:
        coro = cb()
        if inspect.iscoroutine(coro):
            await coro
    except:
        traceback.print_exc()
