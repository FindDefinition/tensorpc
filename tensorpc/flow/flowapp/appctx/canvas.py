

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

def get_simple_canvas():
    comp = find_component(plus.SimpleCanvas)
    assert comp is not None, "you must add simple canvas to your UI"
    return comp


def get_simple_canvas_may_exist():
    """for conditional visualization
    """
    comp = find_component(plus.SimpleCanvas)
    return comp

async def unknown_visualization(obj: Any, tree_id: str):
    return await get_simple_canvas()._unknown_visualization(tree_id, obj, ignore_registry=True)