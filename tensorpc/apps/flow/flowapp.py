# Copyright 2022 Yan Yan
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
"""Flow APP: simple GUI application in devflow
"""

import asyncio
import enum
import numpy as np
from typing import Any, Awaitable, Callable, Coroutine, Dict, List, Optional, Tuple, Union
from PIL import Image

import base64
import io
from tensorpc.utils.registry import HashableRegistry

ALL_APP_EVENTS = HashableRegistry()


def _encode_image(img: np.ndarray):
    pil_img = Image.fromarray(img)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    b64_bytes = base64.b64encode(buffered.getvalue())
    img_str = b64_bytes.decode("utf-8")
    img_str = f"data:image/png;base64,{img_str}"
    return img_str

def _encode_image_bytes(img: np.ndarray):
    pil_img = Image.fromarray(img)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    b64_bytes = base64.b64encode(buffered.getvalue())
    return b"data:image/png;base64," + b64_bytes

async def async_range(start, stop=None, step=1):
    """same as range but schedule other tasks to run in every iteration
    """
    if stop:
        range_ = range(start, stop, step)
    else:
        range_ = range(start)
    for i in range_:
        yield i
        # allow other task run, important for
        # long-time loop based sync task
        await asyncio.sleep(0)


class UIType(enum.Enum):
    # controls
    Buttons = 0
    Input = 1
    Switch = 2

    # outputs
    Image = 10
    Text = 11


class AppEventType(enum.Enum):
    # controls
    UIEvent = 0
    UpdateLayout = 1


@ALL_APP_EVENTS.register(key=AppEventType.UIEvent.value)
class UIEvent:
    def __init__(self, uid: str, data: Any) -> None:
        self.uid = uid
        self.data = data

    def to_dict(self):
        return {
            "uid": self.uid,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data["uid"], data["data"])


@ALL_APP_EVENTS.register(key=AppEventType.UpdateLayout.value)
class LayoutEvent:
    def __init__(self, data) -> None:
        self.data = data

    def to_dict(self):
        return self.data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data)


def app_event_from_data(data: Dict[str, Any]) -> "AppEvent":
    for k, v in ALL_APP_EVENTS.items():
        if k == data["type"]:
            return AppEvent(data["uid"], AppEventType(data["type"]),
                            v.from_dict(data["event"]))
    raise ValueError("not found", data["type"])


# class ControlEvent:
#     def __init__(self, uid: str, data: Any) -> None:
#         self.uid = uid
#         self.data = data

#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]):
#         return cls(data["uid"], data["data"])


class AppEvent:
    def __init__(self, uid: str, type: AppEventType,
                 event: Union[UIEvent, LayoutEvent]) -> None:
        self.uid = uid
        self.event = event
        self.type = type

    def to_dict(self):
        return {
            "uid": self.uid,
            "event": self.event.to_dict(),
            "type": self.type.value,
        }


class Component:
    def __init__(self, uid: str, type: UIType, queue: asyncio.Queue) -> None:
        self.type = type
        self.uid = uid
        self.queue = queue

    def to_dict(self):
        return {
            "type": self.type.value,
            "uid": self.uid,
            "state": {},
        }

    def create_update_event(self, data: Any):
        ev = UIEvent(self.uid, data)
        # uid is set in flowapp service later.
        return AppEvent("", AppEventType.UIEvent, ev)


class Images(Component):
    # TODO keep last event state?
    def __init__(self, uid: str, count: int, queue: asyncio.Queue) -> None:
        super().__init__(uid, UIType.Image, queue)
        self.count = count
        self.image_str: bytes = b""

    async def show(self, index: int, image: np.ndarray):
        encoded = _encode_image_bytes(image)
        self.image_str = encoded
        await self.queue.put(
            self.create_update_event({
                "index": index,
                "image": encoded,
            }))

    async def show_raw(self, index: int, image_b64_bytes: bytes):
        self.image_str = image_b64_bytes
        await self.queue.put(
            self.create_update_event({
                "index": index,
                "image": image_b64_bytes,
            }))

    def show_raw_event(self, index: int, image_b64_bytes: bytes):
        self.image_str = image_b64_bytes
        
        return self.create_update_event({
                "index": index,
                "image": image_b64_bytes,
            })

    def to_dict(self):
        res = super().to_dict()
        res["count"] = self.count
        res["state"] = {
            "image": self.image_str,
        }
        return res 


class Text(Component):
    def __init__(self, init: str, uid: str, queue: asyncio.Queue) -> None:
        super().__init__(uid, UIType.Text, queue)
        self.value = init

    async def write(self, content: str):
        await self.queue.put(self.create_update_event({
            "value": self.value
        }))
        self.value = content

    def to_dict(self):
        res = super().to_dict()
        res["state"] = {
            "value": self.value,
        }
        return res 


class Buttons(Component):
    def __init__(self, uid: str, names: List[str],
                 callback: Callable[[str], Awaitable[None]],
                 queue: asyncio.Queue) -> None:
        super().__init__(uid, UIType.Buttons, queue)
        self.names = names
        self.callback = callback

    def to_dict(self):
        res = super().to_dict()
        res["names"] = self.names 
        return res 

class Input(Component):
    def __init__(self, uid: str, label: str,
                 callback: Callable[[str], Awaitable[None]],
                 queue: asyncio.Queue) -> None:
        super().__init__(uid, UIType.Input, queue)
        self.label = label
        self.callback = callback

        self.value: str = ""

    def to_dict(self):
        res = super().to_dict()
        res["label"] = self.label 
        res["state"] = {
            "value": self.value,
        } 
        return res 

class Switch(Component):
    def __init__(self, uid: str, label: str,
                 callback: Callable[[bool], Awaitable[None]],
                 queue: asyncio.Queue) -> None:
        super().__init__(uid, UIType.Switch, queue)
        self.label = label
        self.callback = callback
        self.checked = False

    def to_dict(self):
        res = super().to_dict()
        res["label"] = self.label 
        res["state"] = {
            "checked": self.checked,
        } 
        return res 

class App:
    def __init__(self) -> None:
        self._components: List[Component] = []
        self._uid_to_comp: Dict[str, Component] = {}
        self._queue = asyncio.Queue(maxsize=10)

        self._init_size = [480, 640]
        self._send_callback: Optional[Callable[[AppEvent], Coroutine[None, None, None]]] = None

    def _get_app_layout(self):
        return {
            "windowSize": self._init_size,
            "layout": [c.to_dict() for c in self._components],
        }

    def set_init_window_size(self, size: List[int]):
        self._init_size = size

    async def _handle_control_event(self, ev: UIEvent):
        # TODO schedule callback as a task to avoid RPC hang
        comp = self._uid_to_comp[ev.uid]
        if isinstance(comp, Buttons):
            await comp.callback(ev.data)
        elif isinstance(comp, Input):
            await comp.callback(ev.data)
            comp.value = ev.data
        elif isinstance(comp, Switch):
            await comp.callback(ev.data)
            comp.checked = ev.data
        else:
            raise NotImplementedError

    def add_buttons(self, names: List[str],
                    callback: Callable[[str], Awaitable[None]]):
        uid = str(len(self._components))
        ui = Buttons(uid, names, callback, self._queue)
        self._components.append(ui)
        self._uid_to_comp[uid] = ui
        return ui

    def add_input(self, label: str, callback: Callable[[str],
                                                       Awaitable[None]]):
        uid = str(len(self._components))
        ui = Input(uid, label, callback, self._queue)
        self._components.append(ui)
        self._uid_to_comp[uid] = ui

        return ui

    def add_switch(self, label: str, callback: Callable[[bool],
                                                        Awaitable[None]]):
        uid = str(len(self._components))
        ui = Switch(uid, label, callback, self._queue)
        self._components.append(ui)
        self._uid_to_comp[uid] = ui
        return ui

    def add_images(self, count: int):
        uid = str(len(self._components))
        ui = Images(uid, count, self._queue)
        self._components.append(ui)
        self._uid_to_comp[uid] = ui
        return ui

    def add_text(self, init: str):
        uid = str(len(self._components))
        ui = Text(init, uid, self._queue)
        self._components.append(ui)
        self._uid_to_comp[uid] = ui
        return ui
