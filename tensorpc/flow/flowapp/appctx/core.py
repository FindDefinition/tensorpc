# Copyright 2023 Yan Yan
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
from typing import (Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
                    Iterable, List, Optional, Set, Tuple, Type, TypeVar, Union)

from typing_extensions import ParamSpec

from tensorpc.core.serviceunit import ObservedFunctionRegistryProtocol
from tensorpc.flow.flowapp.appcore import (enter_app_conetxt, find_component,
                                           find_component_by_uid, get_app,
                                           get_app_context, get_editable_app,
                                           get_reload_manager, is_inside_app,
                                           observe_function)
from tensorpc.flow.flowapp.components import plus
from tensorpc.flow.flowapp.components.plus.objinspect.controllers import ThreadLocker


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
                            in_memory_limit: int = 100):
    app = get_app()
    await app.save_data_storage(key, data, node_id, graph_id, in_memory_limit)


async def read_data_storage(key: str,
                            node_id: Optional[str] = None,
                            graph_id: Optional[str] = None,
                            in_memory_limit: int = 100) -> Any:
    app = get_app()
    return await app.read_data_storage(key, node_id, graph_id, in_memory_limit)


async def remove_data_storage(key: Optional[str],
                              node_id: Optional[str] = None,
                              graph_id: Optional[str] = None) -> Any:
    app = get_app()
    return await app.remove_data_storage_item(key, node_id, graph_id)


async def rename_data_storage_item(key: str,
                                   newname: str,
                                   node_id: Optional[str] = None,
                                   graph_id: Optional[str] = None) -> Any:
    app = get_app()
    return await app.rename_data_storage_item(key, newname, node_id, graph_id)


async def list_data_storage(node_id: str, graph_id: Optional[str] = None):
    app = get_app()
    return await app.list_data_storage(node_id, graph_id)


async def list_all_data_storage_nodes(
        graph_id: Optional[str] = None) -> List[str]:
    app = get_app()
    return await app.list_all_data_storage_nodes(graph_id)

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
    with enter_app_conetxt(app):
        return func(*args, **kwargs)


async def run_in_executor_with_exception_inspect(func: Callable[P, T],
                                                 *args: P.args,
                                                 **kwargs: P.kwargs) -> T:
    """run a sync function in executor with exception inspect.
    """
    comp = find_component(plus.ObjectInspector)
    if comp is None:
        return await asyncio.get_running_loop().run_in_executor(
            None, _run_func_with_app, get_app(), func, *args, **kwargs)  # type: ignore
    assert comp is not None, "you must add inspector to your UI to use exception inspect"
    return await comp.run_in_executor_with_exception_inspect(
        _run_func_with_app, get_app(), func, *args, **kwargs)

def run_coro_sync(coro: Coroutine) -> Any:
    loop = get_app()._loop
    assert loop is not None
    if get_app()._flowapp_thread_id == threading.get_ident():
        # we can't wait fut here
        task = asyncio.create_task(coro)
        # we can't wait fut here
        return task
        # return fut
    else:
        # we can wait fut here.
        fut = asyncio.run_coroutine_threadsafe(
            coro, loop)
        return fut.result()
