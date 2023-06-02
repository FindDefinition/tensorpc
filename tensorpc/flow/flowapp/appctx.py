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
import inspect
from typing import (TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
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
if TYPE_CHECKING:
    from tensorpc.flow import App


P = ParamSpec('P')

T = TypeVar('T')

def func_forward(func: Callable[P, T], *args: P.args,
                       **kwargs: P.kwargs) -> T:
    return func(*args, **kwargs)

def func_forward2(func: Callable[P, T],
                            *args: P.args,
                            **kwargs: P.kwargs) -> T:
    """run a sync function in executor with exception inspect.
    """
    res = func_forward(
        func_forward, func, *args, **kwargs)
    return res
def add(a: int, b: int) -> int:
    return a + b
def main():
    res = func_forward(add, 1, 2)
    res2 = func_forward2(add, 1, 2)

async def obj_inspector_update_locals(*,
                                      exclude_self: bool = False,
                                      key: Optional[str] = None,
                                      _frame_cnt: int = 2):
    comp = find_component(plus.ObjectInspector)
    if comp is None:
        return
    assert comp is not None, "you must add inspector to your UI"
    if key is None:
        await comp.update_locals(_frame_cnt=_frame_cnt, exclude_self=exclude_self)
    else:
        await comp.update_locals(_frame_cnt=_frame_cnt,
                                 exclude_self=exclude_self,
                                 key=key)


def obj_inspector_update_locals_sync(*,
                                     exclude_self: bool = False,
                                     key: Optional[str] = None):
    comp = find_component(plus.ObjectInspector)
    if comp is None:
        return
    assert comp is not None, "you must add inspector to your UI"
    if key is None:
        return comp.update_locals_sync(_frame_cnt=2,
                                       loop=get_app()._loop,
                                       exclude_self=exclude_self)
    else:
        return comp.update_locals_sync(_frame_cnt=2,
                                       loop=get_app()._loop,
                                       exclude_self=exclude_self,
                                       key=key)
    
def thread_locker_wait_sync():
    comp = find_component(ThreadLocker)
    if comp is None:
        return
    assert comp is not None, "you must add ThreadLocker to your UI, you can find it in inspector builtins."
    return comp.wait_sync(loop=get_app()._loop, _frame_cnt=2)

@contextlib.contextmanager
def trace_sync(key: str = "trace",
               traced_types: Optional[Tuple[Type]] = None,
               traced_names: Optional[Set[str]] = None,
               traced_folders: Optional[Set[str]] = None,
               trace_return: bool = True,
               depth: int = 5,
               use_return_locals: bool = False,
               *,
               _frame_cnt=5):
    """trace, store call vars, then write result to ObjectInspector.
    """
    comp = find_component(plus.ObjectInspector)
    if comp is None:
        yield 
        return 
    assert comp is not None, "you must add inspector to your UI"
    with comp.trace_sync(key,
                           traced_types,
                           traced_names,
                           traced_folders,
                           trace_return,
                           depth,
                           use_return_locals,
                           _frame_cnt=_frame_cnt,
                           loop=get_app()._loop):
        yield

@contextlib.contextmanager
def trace_sync_return(key: str = "trace",
               traced_types: Optional[Tuple[Type]] = None,
               traced_names: Optional[Set[str]] = None,
               traced_folders: Optional[Set[str]] = None,
               trace_return: bool = True,
               depth: int = 5,
               *,
               _frame_cnt=5):
    """trace, store local vars in return stmt, then write result to ObjectInspector.
    """
    comp = find_component(plus.ObjectInspector)
    if comp is None:
        yield 
        return 
    assert comp is not None, "you must add inspector to your UI"
    with  comp.trace_sync(key,
                           traced_types,
                           traced_names,
                           traced_folders,
                           trace_return,
                           depth,
                           True,
                           _frame_cnt=_frame_cnt,
                           loop=get_app()._loop):
        yield

@contextlib.asynccontextmanager
async def trace(key: str = "trace",
        traced_types: Optional[Tuple[Type]] = None,
        traced_names: Optional[Set[str]] = None,
        traced_folders: Optional[Set[str]] = None,
        trace_return: bool = True,
        depth: int = 5,
        use_return_locals: bool = False,
        *,
        _frame_cnt=5):
    """async trace, store local vars / args in return stmt, then write result to ObjectInspector.
    """
    comp = find_component(plus.ObjectInspector)
    if comp is None:
        yield 
        return 
    assert comp is not None, "you must add inspector to your UI"
    async with comp.trace(key,
                           traced_types,
                           traced_names,
                           traced_folders,
                           trace_return,
                           depth,
                           use_return_locals,
                           _frame_cnt=_frame_cnt):
        yield

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


def _run_func_with_app(app: "App", func: Callable[P, T], *args: P.args,
                       **kwargs: P.kwargs) -> T:
    with enter_app_conetxt(app):
        return func(*args, **kwargs)

async def run_in_executor(func: Callable[P, T],
                            *args: P.args,
                            **kwargs: P.kwargs) -> T:
    """run a sync function in executor with exception inspect.
    """
    ft = asyncio.get_running_loop().run_in_executor(
        None, _run_func_with_app, get_app(), func, *args, **kwargs)
    return await ft

async def run_in_executor_with_exception_inspect(func: Callable[P, T],
                                                 *args: P.args,
                                                 **kwargs: P.kwargs) -> T:
    """run a sync function in executor with exception inspect.
    """
    comp = find_component(plus.ObjectInspector)
    if comp is None:
        ft = asyncio.get_running_loop().run_in_executor(
            func, *args, **kwargs)
        return await ft
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


async def list_data_storage(node_id: str, graph_id: Optional[str] = None):
    app = get_app()
    return await app.list_data_storage(node_id, graph_id)


async def list_all_data_storage_nodes(
        graph_id: Optional[str] = None) -> List[str]:
    app = get_app()
    return await app.list_all_data_storage_nodes(graph_id)


def set_observed_func_registry(registry: ObservedFunctionRegistryProtocol):
    app = get_app()
    return app.set_observed_func_registry(registry)


async def read_inspector_item(uid: str):
    app = get_app()
    comp = app.find_component(plus.ObjectInspector)
    assert comp is not None, "you must add inspector to your UI to use exception inspect"
    return await comp.get_object_by_uid(uid)
