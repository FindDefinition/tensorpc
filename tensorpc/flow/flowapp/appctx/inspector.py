import asyncio
import contextlib
from functools import partial
import inspect
from pathlib import Path
import types
from typing import (Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
                    Iterable, List, Optional, Set, Tuple, Type, TypeVar, Union)

from typing_extensions import ParamSpec

from tensorpc.flow.flowapp.appcore import (enter_app_conetxt, find_component,
                                           get_app)
from tensorpc.flow.flowapp.components import plus, mui

P = ParamSpec('P')

T = TypeVar('T')


async def update_locals(*,
                        exclude_self: bool = False,
                        key: Optional[str] = None):
    comp = find_component(plus.ObjectInspector)
    if comp is None:
        return
    assert comp is not None, "you must add inspector to your UI"
    if key is None:
        await comp.update_locals(_frame_cnt=2, exclude_self=exclude_self)
    else:
        await comp.update_locals(_frame_cnt=2,
                                 exclude_self=exclude_self,
                                 key=key)


def update_locals_sync(*,
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


@contextlib.contextmanager
def trace_sync(traced_locs: List[Union[str, Path, types.ModuleType]],
               key: str = "trace",
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
    with comp.trace_sync(traced_locs,
                         key,
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
def trace_sync_return(traced_locs: List[Union[str, Path, types.ModuleType]],
                      key: str = "trace",
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
    with comp.trace_sync(traced_locs,
                         key,
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
async def trace(traced_locs: List[Union[str, Path, types.ModuleType]],
                key: str = "trace",
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
    async with comp.trace(traced_locs,
                          key,
                          traced_types,
                          traced_names,
                          traced_folders,
                          trace_return,
                          depth,
                          use_return_locals,
                          _frame_cnt=_frame_cnt):
        yield


async def set_object(obj, key: str, expand_level: int = 0):
    comp = find_component(plus.ObjectInspector)
    if comp is None:
        return
    assert comp is not None, "you must add inspector to your UI"
    await comp.set_object(obj, key, expand_level=expand_level)


def set_object_sync(obj, key: str, expand_level: int = 0):
    comp = find_component(plus.ObjectInspector)
    if comp is None:
        return

    assert comp is not None, "you must add inspector to your UI"
    return comp.set_object_sync(obj, key, get_app()._loop, expand_level)


async def read_item(uid: str):
    app = get_app()
    comp = app.find_component(plus.ObjectInspector)
    assert comp is not None, "you must add inspector to your UI to use exception inspect"
    return await comp.get_object_by_uid(uid)

def has_object(key: str):
    comp = find_component(plus.ObjectInspector)
    if comp is None:
        return False
    assert comp is not None, "you must add inspector to your UI"
    return comp.tree.has_object(key)

def set_custom_layout_sync(layout: mui.FlexBox):
    comp = find_component(plus.ObjectInspector)
    if comp is None:
        return
    assert comp is not None, "you must add inspector to your UI"
    return comp.set_custom_layout_sync(loop=get_app()._loop,
                                    layout=layout)

async def set_custom_layout(layout: mui.FlexBox):
    comp = find_component(plus.ObjectInspector)
    if comp is None:
        return
    assert comp is not None, "you must add inspector to your UI"
    return await comp.set_custom_layout(layout=layout)
