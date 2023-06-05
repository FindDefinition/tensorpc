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
from . import inspector, canvas


P = ParamSpec('P')

T = TypeVar('T')
    
def thread_locker_wait_sync():
    comp = find_component(ThreadLocker)
    if comp is None:
        return
    assert comp is not None, "you must add ThreadLocker to your UI, you can find it in inspector builtins."
    return comp.wait_sync()


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

