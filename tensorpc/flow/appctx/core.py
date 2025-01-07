# Copyright 2024 Yan Yan
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

import asyncio
import contextlib
from functools import partial
import threading
from typing import (Any, AsyncGenerator, Awaitable, Callable, ContextManager, Coroutine, Dict,
                    Iterable, List, Optional, Set, Tuple, Type, TypeVar, Union)

from typing_extensions import ParamSpec

from tensorpc.core.serviceunit import ObservedFunctionRegistryProtocol
from tensorpc.flow.core.appcore import (AppSpecialEventType, RemoteCompEvent, enter_app_context, find_component,
                                        find_component_by_uid, get_app, get_app_storage,
                                        find_all_components, get_app_context,
                                        get_editable_app, get_reload_manager,
                                        is_inside_app, observe_function,
                                        enqueue_delayed_callback, run_coro_sync,
                                        app_is_remote_comp)
from tensorpc.flow.components import plus
from tensorpc.flow.components.plus.objinspect.controllers import ThreadLocker
from tensorpc.flow.core.context import ALL_APP_CONTEXT_GETTERS

P = ParamSpec('P')

T = TypeVar('T')


def thread_locker_wait_sync(*, _frame_cnt: int = 2):
    comp = find_component(ThreadLocker)
    if comp is None:
        return
    assert comp is not None, "you must add ThreadLocker to your UI, you can find it in inspector builtins."
    return comp.wait_sync(loop=get_app()._loop, _frame_cnt=_frame_cnt)


async def save_data_storage(key: str,
                            data: Any,
                            node_id: Optional[str] = None,
                            graph_id: Optional[str] = None,
                            in_memory_limit: int = 100,
                            raise_if_exist: bool = False):
    app = get_app()
    await app.app_storage.save_data_storage(key, data, node_id, graph_id, in_memory_limit,
                                raise_if_exist)


async def read_data_storage(key: str,
                            node_id: Optional[str] = None,
                            graph_id: Optional[str] = None,
                            in_memory_limit: int = 100,
                            raise_if_not_found: bool = True) -> Any:
    app = get_app()
    return await app.app_storage.read_data_storage(key, node_id, graph_id, in_memory_limit,
                                       raise_if_not_found)


async def glob_read_data_storage(glob_prefix: str,
                                           node_id: Optional[str] = None,
                                           graph_id: Optional[str] = None):
    app = get_app()
    return await app.app_storage.glob_read_data_storage(glob_prefix, node_id,
                                                      graph_id)


async def remove_data_storage(key: Optional[str],
                              node_id: Optional[str] = None,
                              graph_id: Optional[str] = None) -> Any:
    app = get_app()
    return await app.app_storage.remove_data_storage_item(key, node_id, graph_id)


async def rename_data_storage_item(key: str,
                                   newname: str,
                                   node_id: Optional[str] = None,
                                   graph_id: Optional[str] = None) -> Any:
    app = get_app()
    return await app.app_storage.rename_data_storage_item(key, newname, node_id, graph_id)


async def list_data_storage(node_id: Optional[str] = None,
                            graph_id: Optional[str] = None,
                            glob_prefix: Optional[str] = None):
    app = get_app()
    return await app.app_storage.list_data_storage(node_id, graph_id, glob_prefix)


async def list_all_data_storage_nodes(
        graph_id: Optional[str] = None) -> List[str]:
    app = get_app()
    return await app.app_storage.list_all_data_storage_nodes(graph_id)


async def data_storage_has_item(key: str,
                                node_id: Optional[str] = None,
                                graph_id: Optional[str] = None):
    app = get_app()
    return await app.app_storage.data_storage_has_item(key, node_id, graph_id)

async def copy_text_to_clipboard(data: str):
    app = get_app()
    return await app.copy_text_to_clipboard(data)

async def get_vscode_storage():
    app = get_app()
    return await app.get_vscode_storage_lazy()

def get_vscode_state():
    app = get_app()
    return app.get_vscode_state()

def set_app_z_index(z_index: int):
    app = get_app()
    app._dialog_z_index = z_index


def set_observed_func_registry(registry: ObservedFunctionRegistryProtocol):
    app = get_app()
    return app.set_observed_func_registry(registry)


def run_with_exception_inspect(func: Callable[P, T], *args: P.args,
                               **kwargs: P.kwargs) -> T:
    """WARNING: we shouldn't run this function in run_in_executor.
    """
    comp = find_component(plus.ObjectInspector)
    assert comp is not None, "you must add inspector to your UI to use exception inspect"
    return comp.run_with_exception_inspect(func, *args, **kwargs)


async def run_with_exception_inspect_async(func: Callable[P, T], *args: P.args,
                                           **kwargs: P.kwargs) -> T:
    comp = find_component(plus.ObjectInspector)
    assert comp is not None, "you must add inspector to your UI to use exception inspect"
    return await comp.run_with_exception_inspect_async(func, *args, **kwargs)


def _run_func_with_app(app, func: Callable[P, T], *args: P.args,
                       **kwargs: P.kwargs) -> T:
    with enter_app_context(app):
        return func(*args, **kwargs)


def _run_func_with_context_creators(ctx_creators: List[Callable[[], Optional[ContextManager]]], func: Callable[P, T], *args: P.args,
                       **kwargs: P.kwargs) -> T:
    ctxes = [
        c() for c in ctx_creators
    ]
    with contextlib.ExitStack() as stack:
        for ctx in ctxes:
            if ctx is not None:
                stack.enter_context(ctx)
        return func(*args, **kwargs)


async def run_in_executor_with_exception_inspect(func: Callable[P, T],
                                                 *args: P.args,
                                                 **kwargs: P.kwargs) -> T:
    """run a sync function in executor with exception inspect.
    """
    comp = find_component(plus.ObjectInspector)
    if comp is None:
        return await asyncio.get_running_loop().run_in_executor(
            None, _run_func_with_app, get_app(), func, *args,
            **kwargs)  # type: ignore
    assert comp is not None, "you must add inspector to your UI to use exception inspect"
    return await comp.run_in_executor_with_exception_inspect(
        _run_func_with_app, get_app(), func, *args, **kwargs)


def _ctx_creator(ctx: Any, ctx_enterer: Callable[[Any], ContextManager]):
    return ctx_enterer(ctx)

async def run_in_executor(func: Callable[P, T],
                            *args: P.args,
                            **kwargs: P.kwargs) -> T:
    """run a sync function in executor with all app context entered.
    default run_in_executor don't enter contexts such app context 
    and compute flow context.
    """
    creators: List[Callable[[], Optional[ContextManager]]] = []
    for v in ALL_APP_CONTEXT_GETTERS:
        ctx = v[0]()
        if ctx is not None:
            creators.append(partial(_ctx_creator, ctx, v[1]))

    return await asyncio.get_running_loop().run_in_executor(
        None, _run_func_with_context_creators, creators, func, *args, # type: ignore
        **kwargs)  

async def run_in_executor_with_contexts(func: Callable[P, T],
                        ctx_creators: List[Callable[[], Optional[ContextManager]]],
                            *args: P.args,
                            **kwargs: P.kwargs) -> T:
    """run a sync function in executor.
    """
    app = get_app()
    return await asyncio.get_running_loop().run_in_executor(
        None, _run_func_with_context_creators, [lambda: enter_app_context(app)] + ctx_creators, partial(func, **kwargs), *args) # type: ignore

def register_app_special_event_handler(event: AppSpecialEventType,
                                             handler: Callable):
    app = get_app()
    return app.register_app_special_event_handler(event, handler)

def unregister_app_special_event_handler(event: AppSpecialEventType,
                                           handler: Callable):
    app = get_app()
    return app.unregister_app_special_event_handler(event, handler)

def register_remote_comp_event_handler(key: str,
                                        handler: Callable[[RemoteCompEvent], Any]):
    app = get_app()
    return app.register_remote_comp_event_handler(key, handler)

def unregister_remote_comp_event_handler(key: str,
                                        handler: Callable[[RemoteCompEvent], Any]):
    app = get_app()
    return app.unregister_remote_comp_event_handler(key, handler)

def register_simple_rpc_handler(key: str, handler: Callable):
    app = get_app()
    return app.register_app_simple_rpc_handler(key, handler)

def unregister_simple_rpc_handler(key: str):
    app = get_app()
    return app.unregister_app_simple_rpc_handler(key)