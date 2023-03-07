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

from tensorpc.flow.flowapp.app import get_app_context, get_app, find_component, get_reload_manager
from tensorpc.flow.flowapp.components import plus 
from typing import (Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
                    Iterable, List, Optional, Set, Tuple, Type, TypeVar, Union)


async def obj_inspector_update_locals():
    comp = find_component(plus.ObjectInspector)
    assert comp is not None, "you must add inspector to your UI"
    await comp.update_locals(_frame_cnt=2)

def obj_inspector_update_locals_sync():
    comp = find_component(plus.ObjectInspector)
    assert comp is not None, "you must add inspector to your UI"
    return comp.update_locals_sync(_frame_cnt=2)

async def obj_inspector_set_object(obj, key: str):
    comp = find_component(plus.ObjectInspector)
    assert comp is not None, "you must add inspector to your UI"
    await comp.set_object(obj, key)

def obj_inspector_set_object_sync(obj, key: str):
    comp = find_component(plus.ObjectInspector)
    assert comp is not None, "you must add inspector to your UI"
    return comp.set_object_sync(obj, key)

def get_simple_canvas():
    comp = find_component(plus.SimpleCanvas)
    assert comp is not None, "you must add simple canvas to your UI"
    return comp 

def get_simple_canvas_may_exist():
    """for conditional visualization
    """
    comp = find_component(plus.SimpleCanvas)
    return comp 

