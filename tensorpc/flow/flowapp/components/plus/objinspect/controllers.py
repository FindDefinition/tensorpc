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

from typing import Any, Callable, Optional, List, Dict, TypeVar, Generic, Union
from tensorpc.flow.flowapp.components import mui

from tensorpc.flow.marker import mark_create_preview_layout 
import inspect

T = TypeVar("T")

class CallbackSlider(mui.FlexBox):
    """a slider that used for list.
    """

    def __init__(self) -> None:
        self.slider = mui.Slider(mui.undefined, 0, 1, 1).prop(flex=1)
        super().__init__([self.slider])
        self.slider.register_event_handler(mui.FrontendEventType.Change.value,
                                    self._default_callback)
        self.prop(width="100%", flex_flow="row nowrap", padding_left="5px", padding_right="5px")

    async def _default_callback(self, index):
        pass
    
    @mark_create_preview_layout
    def tensorpc_flow_preview_layout(self):
        return self

    async def update_callback(self, length: int, cb: Callable[[Any], mui._CORO_NONE]):
        self.slider.register_event_handler(mui.FrontendEventType.Change.value,
                                    cb)
        print(length - 1)
        await self.slider.update_ranges(0, length - 1, 1)

