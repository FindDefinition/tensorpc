

import asyncio
import contextlib
from functools import partial
import inspect
from typing import (Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
                    Iterable, List, Optional, Set, Tuple, Type, TypeVar, Union)

from typing_extensions import ParamSpec

from tensorpc.flow.flowapp.appcore import (enter_app_conetxt, find_component,
                                           get_app)
from tensorpc.flow.flowapp.components import plus
from tensorpc.utils.uniquename import UniqueNamePool
def get_simple_canvas(key: Optional[str] = None) -> plus.SimpleCanvas:
    if key is not None:
        comp = find_component(plus.SimpleCanvas, lambda x: x.key == key)
    else:
        comp = find_component(plus.SimpleCanvas)
    assert comp is not None, "you must add simple canvas to your UI"
    return comp


def get_simple_canvas_may_exist(key: Optional[str] = None):
    """for conditional visualization
    """
    if key is not None:
        comp = find_component(plus.SimpleCanvas, lambda x: x.key == key)
    else:
        comp = find_component(plus.SimpleCanvas)
    return comp

async def unknown_visualization(obj: Any, tree_id: str, key: Optional[str] = None):
    return await get_simple_canvas(key)._unknown_visualization(tree_id, obj, ignore_registry=False)

async def unknown_visualization_no_registry(obj: Any, tree_id: str, key: Optional[str] = None):
    return await get_simple_canvas(key)._unknown_visualization(tree_id, obj, ignore_registry=True)

async def unknown_visualization_temp_objs(*objs, vis_root_id: str = "", canvas_key: Optional[str] = None, **kwobjs):
    pool = UniqueNamePool()
    canvas = get_simple_canvas(canvas_key)
    for i, o in enumerate(objs):
        uid = pool(str(i))
        if vis_root_id != "":
            uid = f"{vis_root_id}.{uid}"
        await canvas._unknown_visualization(uid, o)
    for k, o in kwobjs.items():
        uid = pool(k)
        if vis_root_id != "":
            uid = f"{vis_root_id}.{uid}"
        await canvas._unknown_visualization(uid, o)

