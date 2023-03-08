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

from typing import Callable, Optional, List, Dict, TypeVar, Generic
from tensorpc.flow.flowapp.components import mui, three, plus
import inspect

T = TypeVar("T")


class ListSlider(mui.Slider, Generic[T]):
    """a slider that used for list.
    """

    def __init__(self,
                 label: str,
                 callback: Callable[[T], mui._CORO_NONE],
                 init: Optional[List[T]] = None) -> None:
        if init is None:
            init = []
        super().__init__(label, 0, max(1, len(init)), 1, self._callback)
        # save callback to standard flow event handlers to enable reload for user callback
        self.__callback_key = "list_slider_ev_handler"
        self.register_event_handler(self.__callback_key,
                                    callback,
                                    backend_only=True)
        self.obj_list: List[T] = init

    async def update_list(self, objs: List[T]):
        self.obj_list = objs
        await self.update_ranges(0, len(objs) - 1, 1)

    async def _callback(self, value: mui.NumberType):
        handler = self.get_event_handler(self.__callback_key)
        if handler is not None:
            index = int(value)
            if index >= len(self.obj_list):
                return
            obj = self.obj_list[index]
            coro = handler.cb(obj)
            if inspect.iscoroutine(coro):
                await coro
