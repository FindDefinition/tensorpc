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
import inspect
from tensorpc.flow.flowapp.appcore import get_app_context, get_app, get_editable_app, find_component, get_reload_manager, enter_app_conetxt, find_component_by_uid, observe_function
from tensorpc.flow.flowapp.components import plus
from typing import (Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
                    Iterable, List, Optional, Set, Tuple, Type, TypeVar, Union)
from typing_extensions import ParamSpec
P = ParamSpec('P')

T = TypeVar('T')


async def obj_inspector_update_locals():
    comp = find_component(plus.ObjectInspector)
    if comp is None:
        return 
    assert comp is not None, "you must add inspector to your UI"
    await comp.update_locals(_frame_cnt=2)


def obj_inspector_update_locals_sync():
    comp = find_component(plus.ObjectInspector)
    if comp is None:
        return 
    assert comp is not None, "you must add inspector to your UI"
    return comp.update_locals_sync(_frame_cnt=2, loop=get_app()._loop)


async def obj_inspector_set_object(obj, key: str):
    comp = find_component(plus.ObjectInspector)
    if comp is None:
        return 
    assert comp is not None, "you must add inspector to your UI"
    await comp.set_object(obj, key)


def obj_inspector_set_object_sync(obj, key: str):
    comp = find_component(plus.ObjectInspector)
    if comp is None:
        return 
    
    assert comp is not None, "you must add inspector to your UI"
    return comp.set_object_sync(obj, key, get_app()._loop)


def get_simple_canvas():
    comp = find_component(plus.SimpleCanvas)
    assert comp is not None, "you must add simple canvas to your UI"
    return comp


def get_simple_canvas_may_exist():
    """for conditional visualization
    """
    comp = find_component(plus.SimpleCanvas)
    return comp


def register_obj_reload(*objs, autorun_func):
    app = get_editable_app()
    name = ""
    if isinstance(autorun_func, str):
        name = autorun_func
    else:
        name = autorun_func.__name__
    for obj in objs:
        app._flowapp_object_observe(obj, name)


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
        return await asyncio.get_running_loop().run_in_executor(func, *args, **kwargs)
    assert comp is not None, "you must add inspector to your UI to use exception inspect"
    return await comp.run_in_executor_with_exception_inspect(
        _run_func_with_app, get_app(), func, *args, **kwargs)


async def save_data_storage(key: str,
                            node_id: str,
                            data: Any,
                            graph_id: Optional[str] = None,
                            in_memory_limit: int = 100):
    app = get_app()
    await app.save_data_storage(key, node_id, data, graph_id, in_memory_limit)


async def read_data_storage(key: str,
                            node_id: str,
                            graph_id: Optional[str] = None,
                            in_memory_limit: int = 100) -> Any:
    app = get_app()
    return await app.read_data_storage(key, node_id, graph_id, in_memory_limit)

async def remove_data_storage(key: Optional[str],
                            node_id: str,
                            graph_id: Optional[str] = None) -> Any:
    app = get_app()
    return await app.remove_data_storage_item(key, node_id, graph_id)

async def rename_data_storage_item(key: str,
                            newname: str, 
                            node_id: str,
                            graph_id: Optional[str] = None) -> Any:
    app = get_app()
    return await app.rename_data_storage_item(key, newname, node_id, graph_id)

async def list_data_storage(node_id: str):
    app = get_app()
    return await app.list_data_storage(node_id)


async def list_all_data_storage_nodes() -> List[str]:
    app = get_app()
    return await app.list_all_data_storage_nodes()


