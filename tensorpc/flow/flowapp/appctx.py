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

from tensorpc.flow.flowapp.app import get_app_context, get_app
from typing import (Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
                    Iterable, List, Optional, Set, Tuple, Type, TypeVar, Union)

T = TypeVar("T")

def find_component(type: Type[T]) -> Optional[T]:
    appctx = get_app_context()
    assert appctx is not None, "you must use this function in app"
    return appctx.app.find_component(type)

def get_reload_manager():
    appctx = get_app_context()
    assert appctx is not None, "you must use this function in app"
    return appctx.app._flow_reload_manager
