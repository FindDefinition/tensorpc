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

import threading
from types import FrameType
from typing import Any, Callable, Optional, List, Dict, TypeVar, Generic, Union
from tensorpc.core import inspecttools
from tensorpc.flow.flowapp.components import mui

from tensorpc.flow.marker import mark_autorun, mark_create_preview_layout, mark_did_mount, mark_will_unmount 
import inspect
import asyncio
from tensorpc.flow.flowapp.appcore import get_app


T = TypeVar("T")

class CallbackSlider(mui.FlexBox):
    """a slider that used for list.
    """

    def __init__(self) -> None:
        self.slider = mui.Slider(0, 1, 1).prop(flex=1)
        super().__init__([self.slider])
        self.slider.register_event_handler(mui.FrontendEventType.Change.value,
                                    self._default_callback)
        self.prop(width="100%", flexFlow="row nowrap", paddingLeft="5px", paddingRight="5px")

    async def _default_callback(self, index):
        pass
    
    @mark_create_preview_layout
    def tensorpc_flow_preview_layout(self):
        return self

    async def update_callback(self, length: int, cb: Callable[[Any], mui._CORO_NONE]):
        self.slider.register_event_handler(mui.FrontendEventType.Change.value,
                                    cb)
        await self.slider.update_ranges(0, length - 1, 1)


class ThreadLocker(mui.FlexBox):
    """a locker designed for long-time running sync function, you can
    use locker to wait for GUI release. e.g., you run a deeplearning
    train function, you can visualize intermediate results in GUI (via capture), and
    wait for GUI release to continue training.
    """
    def __init__(self) -> None:
        self.text = mui.Markdown("### Thread Locker\n:green[Unlocked]")
        self.event = threading.Event()
        super().__init__([
            self.text,
            mui.HBox([
                mui.Button("Release", self._on_release),
                mui.Button("Raise", self._on_raise),
                mui.Button("Capture", self._on_capture),
            ], wrap=True),
        ])
        self.prop(width="100%", flexFlow="column", paddingLeft="5px", paddingRight="5px")

        self._prev_frame: Optional[FrameType] = None

        self._need_raise: bool = False

    async def _on_release(self):
        self.event.set()

    async def _on_raise(self):
        self._need_raise = True
        self.event.set()
    
    async def _on_capture(self):
        """capture current locals to inspector.
        """
        from tensorpc.flow import appctx , plus
        if self._prev_frame is None:
            return 
        frame_name = self._prev_frame.f_code.co_name
        inspector = appctx.find_component(plus.ObjectInspector)
        assert inspector is not None 
        local_vars = self._prev_frame.f_locals.copy()
        await inspector.tree.set_object(inspecttools.filter_local_vars(local_vars),
                                   "locals" + f"-{frame_name}")

    def wait_sync(self, loop: Optional[asyncio.AbstractEventLoop] = None, msg: str = "", *, _frame_cnt: int = 1):
        assert get_app()._flowapp_thread_id != threading.get_ident(), "you must use this function in a thread."
        self._need_raise = False 
        cur_frame = inspect.currentframe()
        assert cur_frame is not None
        frame = cur_frame
        while _frame_cnt > 0:
            frame = cur_frame.f_back
            assert frame is not None
            cur_frame = frame
            _frame_cnt -= 1
        if loop is None:
            loop = asyncio.get_running_loop()
        fut = asyncio.run_coroutine_threadsafe(
            self.text.write("### Thread Locker\n:red[Locked]"), loop)
        fut.result()
        self._prev_frame = cur_frame
        print(f"ThreadLocker wait {msg}")
        self.event.clear()
        self.event.wait()
        fut = asyncio.run_coroutine_threadsafe(
            self.text.write("### Thread Locker\n:green[Unlocked]"), loop)

        self._prev_frame = None 
        if self._need_raise == True:
            self._need_raise = False 
            raise ValueError("You click raise exception button in ThreadLocker.")

    @mark_will_unmount
    async def _unmount(self):
        self.event.set()

    @mark_did_mount
    async def _mount(self):
        self.event.clear()

class MarkdownViewer(mui.FlexBox):
    """this class is existed for standard InspectPanel-based usage.
    When you add it to your layout, you can access it by 
    appctx.find_component(plus.MarkdownViewer).
    this class has been added to Inspector builtins.
    """
    def __init__(self) -> None:
        self.info = mui.Markdown()
        super().__init__([
            self.info,
        ])
        self.prop(width="100%")

    async def write(self, content: str):
        return await self.info.write(content)

