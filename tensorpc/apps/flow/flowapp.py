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
import base64
import enum
import io
import time
import traceback
from typing import (Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
                    Iterable, List, Optional, Tuple, TypeVar, Union)
from tensorpc.utils.uniquename import UniqueNamePool
import numpy as np
from PIL import Image
from tensorpc.core.asynctools import cancel_task
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


class UIType(enum.Enum):
    # controls
    Buttons = 0
    Input = 1
    Switch = 2
    Select = 3
    Slider = 4
    RadioGroup = 5
    CodeEditor = 6
    Button = 7

    # outputs
    Image = 10
    Text = 11

    # special
    TaskLoop = 100
    FlexBox = 101


class AppEventType(enum.Enum):
    # layout events
    UpdateLayout = 0

    # ui event
    UIEvent = 10
    # clipboard
    CopyToClipboard = 20


class UIRunStatus(enum.Enum):
    Stop = 0
    Running = 1
    Pause = 2


class TaskLoopEvent(enum.Enum):
    Start = 0
    Stop = 1
    Pause = 2


@ALL_APP_EVENTS.register(key=AppEventType.UIEvent.value)
class UIEvent:

    def __init__(self, uid_to_data: Dict[str, Any]) -> None:
        self.uid_to_data = uid_to_data

    def to_dict(self):
        return self.uid_to_data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data)

    def merge_new(self, new):
        assert isinstance(new, UIEvent)
        res_uid_to_data: Dict[str, Any] = self.uid_to_data.copy()
        for k, v in new.uid_to_data.items():
            if k in self.uid_to_data:
                res_uid_to_data[k] = {**v, **self.uid_to_data[k]}
            else:
                res_uid_to_data[k] = v
        return UIEvent(res_uid_to_data)


@ALL_APP_EVENTS.register(key=AppEventType.UpdateLayout.value)
class LayoutEvent:

    def __init__(self, data) -> None:
        self.data = data

    def to_dict(self):
        return self.data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data)

    def merge_new(self, new):
        assert isinstance(new, LayoutEvent)

        return new


@ALL_APP_EVENTS.register(key=AppEventType.CopyToClipboard.value)
class CopyToClipboardEvent:

    def __init__(self, text: str) -> None:
        self.text = text

    def to_dict(self):
        return {
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data["text"])

    def merge_new(self, new):
        assert isinstance(new, CopyToClipboardEvent)
        return new


APP_EVENT_TYPES = Union[UIEvent, LayoutEvent, CopyToClipboardEvent]


def app_event_from_data(data: Dict[str, Any]) -> "AppEvent":
    type_to_event: Dict[AppEventType, APP_EVENT_TYPES] = {}
    for evtypeval, evdata in data["typeToEvents"]:
        found = False
        for k, v in ALL_APP_EVENTS.items():
            if k == evtypeval:
                type_to_event[AppEventType(k)] = v.from_dict(evdata)
                found = True
                break
        if not found:
            raise ValueError("not found", evtypeval)
    return AppEvent(data["uid"], type_to_event)


# class ControlEvent:
#     def __init__(self, uid: str, data: Any) -> None:
#         self.uid = uid
#         self.data = data

#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]):
#         return cls(data["uid"], data["data"])


class AppEvent:

    def __init__(self, uid: str, type_to_event: Dict[AppEventType,
                                                     APP_EVENT_TYPES]) -> None:
        self.uid = uid
        self.type_to_event = type_to_event

    def to_dict(self):
        # here we don't use dict for typeToEvents due to frontend key type limit.
        t2e = [(k.value, v.to_dict()) for k, v in self.type_to_event.items()]
        # make sure layout is proceed firstly.
        t2e.sort(key=lambda x: x[0])
        return {"uid": self.uid, "typeToEvents": t2e}

    def merge_new(self, new: "AppEvent"):
        new_type_to_event: Dict[AppEventType,
                                APP_EVENT_TYPES] = new.type_to_event.copy()
        for k, v in self.type_to_event.items():
            if k in new.type_to_event:
                new_type_to_event[k] = v.merge_new(new.type_to_event[k])
            else:
                new_type_to_event[k] = v
        return AppEvent(self.uid, new_type_to_event)


class Component:

    def __init__(self,
                 uid: str,
                 type: UIType,
                 queue: asyncio.Queue,
                 flex: Optional[str] = None,
                 align_self: Optional[str] = None) -> None:
        self.type = type
        self.uid = uid
        self.queue = queue

        self._status = UIRunStatus.Stop
        # task for callback of controls
        # if previous control callback hasn't finished yet,
        # the new control event will be IGNORED
        self._task: Optional[asyncio.Task] = None
        self._flex = flex
        self._align_self = align_self
        self.parent = ""

    def to_dict(self):
        res = {
            "type": self.type.value,
            "uid": self.uid,
            # "parent": self.parent,
            "state": {
                "status": self._status.value,
            },
        }
        if self._flex is not None:
            res["flex"] = self._flex
        if self._align_self is not None:
            res["alignSelf"] = self._align_self
        return res

    def create_update_event(self, data: Any):
        ev = UIEvent({self.uid: data})
        # uid is set in flowapp service later.
        return AppEvent("", {AppEventType.UIEvent: ev})

    async def run_callback(self,
                           cb: Callable[[], Coroutine[None, None, None]],
                           sync_state: bool = False):
        self._status = UIRunStatus.Running
        await self.sync_status(sync_state)
        try:
            await cb()
        except:
            traceback.print_exc()
            raise
        finally:
            self._status = UIRunStatus.Stop
            await self.sync_status(sync_state)

    async def sync_status(self, sync_state: bool = False):
        await self.queue.put(
            self.create_update_event({"status": self._status.value}))


class Images(Component):
    # TODO keep last event state?
    def __init__(self,
                 uid: str,
                 queue: asyncio.Queue,
                 flex: Optional[str] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.Image, queue, flex, align_self)
        self.image_str: bytes = b""

    async def show(self, image: np.ndarray):
        encoded = _encode_image_bytes(image)
        self.image_str = encoded
        await self.queue.put(self.create_update_event({
            "image": encoded,
        }))

    async def show_raw(self, image_b64_bytes: bytes):
        self.image_str = image_b64_bytes
        await self.queue.put(
            self.create_update_event({
                "image": image_b64_bytes,
            }))

    def show_raw_event(self, image_b64_bytes: bytes):
        self.image_str = image_b64_bytes

        return self.create_update_event({
            "image": image_b64_bytes,
        })

    def to_dict(self):
        res = super().to_dict()
        res["state"] = {
            "image": self.image_str,
        }
        return res


class Text(Component):

    def __init__(self,
                 init: str,
                 uid: str,
                 queue: asyncio.Queue,
                 flex: Optional[str] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.Text, queue, flex, align_self)
        self.value = init

    async def write(self, content: str):
        await self.queue.put(self.create_update_event({"value": self.value}))
        self.value = content

    def to_dict(self):
        res = super().to_dict()
        res["state"] = {
            "value": self.value,
        }
        return res


class Button(Component):

    def __init__(self,
                 uid: str,
                 name: str,
                 callback: Callable[[str], Coroutine[None, None, None]],
                 queue: asyncio.Queue,
                 flex: Optional[str] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.Button, queue, flex, align_self)
        self.name = name
        self.callback = callback

    def to_dict(self):
        res = super().to_dict()
        res["name"] = self.name
        return res


class Buttons(Component):

    def __init__(self,
                 uid: str,
                 names: List[str],
                 callback: Callable[[str], Coroutine[None, None, None]],
                 queue: asyncio.Queue,
                 flex: Optional[str] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.Buttons, queue, flex, align_self)
        self.names = names
        self.callback = callback

    def to_dict(self):
        res = super().to_dict()
        res["names"] = self.names
        return res


class FlexBox(Component):

    def __init__(self,
                 uid: str,
                 pool: UniqueNamePool,
                 queue: asyncio.Queue,
                 uid_to_comp: Dict[str, Component],
                 flex_flow: Optional[str] = None,
                 justify_content: Optional[str] = None,
                 align_items: Optional[str] = None,
                 flex: Optional[str] = None,
                 align_self: Optional[str] = None,
                 width: Optional[Union[str, int]] = None,
                 height: Optional[Union[str, int]] = None) -> None:
        super().__init__(uid, UIType.FlexBox, queue, flex, align_self)
        self._pool = pool
        self._uid_to_comp = uid_to_comp
        self.flex_flow = flex_flow
        self.justify_content = justify_content
        self.align_items = align_items
        self.width = width
        self.height = height
        self._childs: List[str] = []

    def to_dict(self):
        res = super().to_dict()
        res["childs"] = self._childs
        if self.flex_flow is not None:
            res["flexFlow"] = self.flex_flow
        if self.justify_content is not None:
            res["justifyContent"] = self.justify_content
        if self.align_items is not None:
            res["alignItems"] = self.align_items
        if self.width is not None:
            res["width"] = self.width
        if self.height is not None:
            res["height"] = self.height
        return res

    def add_flex_box(self,
                     flex_flow: Optional[str] = None,
                     justify_content: Optional[str] = None,
                     align_items: Optional[str] = None,
                     flex: Optional[str] = None,
                     align_self: Optional[str] = None):
        uid = self._pool("box")
        ui = FlexBox(uid, self._pool, self.queue, self._uid_to_comp, flex_flow,
                     justify_content, align_items, flex, align_self)
        self._uid_to_comp[uid] = ui
        ui.parent = self.uid
        self._childs.append(uid)
        return ui

    def add_buttons(self,
                    names: List[str],
                    callback: Callable[[str], Coroutine[None, None, None]],
                    flex: Optional[str] = None,
                    align_self: Optional[str] = None):
        uid = self._pool("btns")
        ui = Buttons(uid, names, callback, self.queue, flex, align_self)
        self._uid_to_comp[uid] = ui
        ui.parent = self.uid
        self._childs.append(uid)
        return ui

    def add_button(self,
                   name: str,
                   callback: Callable[[str], Coroutine[None, None, None]],
                   flex: Optional[str] = None,
                   align_self: Optional[str] = None):
        uid = self._pool("btn")
        ui = Button(uid, name, callback, self.queue, flex, align_self)
        self._uid_to_comp[uid] = ui
        ui.parent = self.uid
        self._childs.append(uid)
        return ui

    def add_input(self,
                  label: str,
                  callback: Optional[Callable[[str], Coroutine[None, None,
                                                               None]]] = None,
                  flex: Optional[str] = None,
                  align_self: Optional[str] = None):
        uid = self._pool("inp")
        ui = Input(uid, label, callback, self.queue, flex, align_self)
        self._uid_to_comp[uid] = ui
        ui.parent = self.uid
        self._childs.append(uid)
        return ui

    def add_code_editor(self,
                        language: str,
                        callback: Optional[Callable[[str],
                                                    Coroutine[None, None,
                                                              None]]] = None,
                        flex: Optional[str] = None,
                        align_self: Optional[str] = None):
        uid = self._pool("code")
        ui = CodeEditor(uid, language, callback, self.queue, flex, align_self)
        self._uid_to_comp[uid] = ui
        ui.parent = self.uid
        self._childs.append(uid)
        return ui

    def add_switch(self,
                   label: str,
                   callback: Optional[Callable[[bool],
                                               Coroutine[None, None,
                                                         None]]] = None,
                   flex: Optional[str] = None,
                   align_self: Optional[str] = None):
        uid = self._pool("swi")
        ui = Switch(uid, label, callback, self.queue, flex, align_self)
        self._uid_to_comp[uid] = ui
        ui.parent = self.uid
        self._childs.append(uid)
        return ui

    def add_image(self,
                  flex: Optional[str] = None,
                  align_self: Optional[str] = None):
        uid = self._pool("img")
        ui = Images(uid, self.queue, flex, align_self)
        self._uid_to_comp[uid] = ui
        ui.parent = self.uid
        self._childs.append(uid)
        return ui

    def add_text(self,
                 init: str,
                 flex: Optional[str] = None,
                 align_self: Optional[str] = None):
        uid = self._pool("text")
        ui = Text(init, uid, self.queue, flex, align_self)
        self._uid_to_comp[uid] = ui
        ui.parent = self.uid
        self._childs.append(uid)
        return ui

    def add_radio_group(self,
                        names: List[str],
                        row: bool,
                        callback: Optional[Callable[[str],
                                                    Coroutine[None, None,
                                                              None]]] = None,
                        flex: Optional[str] = None,
                        align_self: Optional[str] = None):
        uid = self._pool("radio")
        ui = RadioGroup(uid, names, row, callback, self.queue, flex,
                        align_self)
        self._uid_to_comp[uid] = ui
        ui.parent = self.uid
        self._childs.append(uid)
        return ui

    def add_select(self,
                   label: str,
                   items: List[Tuple[str, Any]],
                   callback: Optional[Callable[[Any], Coroutine[None, None,
                                                                None]]] = None,
                   flex: Optional[str] = None,
                   align_self: Optional[str] = None):
        uid = self._pool("select")
        ui = Select(uid, label, items, callback, self.queue, flex, align_self)
        self._uid_to_comp[uid] = ui
        ui.parent = self.uid
        self._childs.append(uid)
        return ui

    def add_slider(self,
                   label: str,
                   begin: Union[int, float],
                   end: Union[int, float],
                   step: Union[int, float],
                   callback: Optional[Callable[[Union[int, float]],
                                               Coroutine[None, None,
                                                         None]]] = None,
                   flex: Optional[str] = None,
                   align_self: Optional[str] = None):
        uid = self._pool("slider")
        ui = Slider(uid, label, begin, end, step, callback, self.queue, flex,
                    align_self)
        self._uid_to_comp[uid] = ui
        ui.parent = self.uid
        self._childs.append(uid)
        return ui

    def add_task_loop(self,
                      label: str,
                      callback: Callable[[], Coroutine[None, None, None]],
                      update_period: float = 0.2,
                      flex: Optional[str] = None,
                      align_self: Optional[str] = None):
        """use ASYNC LOOP in this ui!!!!!!!!!!!!!
        DON'T USE OTHER LOOP!!!
        """
        uid = self._pool("task")
        ui = TaskLoop(uid, label, callback, self.queue, update_period, flex,
                      align_self)
        self._uid_to_comp[uid] = ui
        ui.parent = self.uid
        self._childs.append(uid)
        return ui


class RadioGroup(Component):

    def __init__(self,
                 uid: str,
                 names: List[str],
                 row: bool,
                 callback: Optional[Callable[[str], Coroutine[None, None,
                                                              None]]],
                 queue: asyncio.Queue,
                 flex: Optional[str] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.RadioGroup, queue, flex, align_self)
        self.names = names
        if callback is None:
            callback = self._default_callback
        self.callback = callback
        self.row = row
        self.value = names[0]

    def to_dict(self):
        res = super().to_dict()
        res["names"] = self.names
        res["row"] = self.row
        res["state"] = {
            "value": self.value,
        }
        return res

    async def _default_callback(self, value: str):
        self.value = value

    async def update_value(self, value: Any):
        assert value in self.names
        await self.queue.put(self.create_update_event({"value": value}))
        self.value = value


class Input(Component):

    def __init__(self,
                 uid: str,
                 label: str,
                 callback: Optional[Callable[[str], Coroutine[None, None,
                                                              None]]],
                 queue: asyncio.Queue,
                 flex: Optional[str] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.Input, queue, flex, align_self)
        self.label = label
        if callback is None:
            callback = self._default_callback
        self.callback = callback
        self.value: str = ""

    def to_dict(self):
        res = super().to_dict()
        res["label"] = self.label
        res["state"] = {
            "value": self.value,
        }
        return res

    async def _default_callback(self, value: str):
        self.value = value


class CodeEditor(Component):

    def __init__(self,
                 uid: str,
                 language: str,
                 callback: Optional[Callable[[str], Coroutine[None, None,
                                                              None]]],
                 queue: asyncio.Queue,
                 flex: Optional[str] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.CodeEditor, queue, flex, align_self)
        self.language = language
        if callback is None:
            callback = self._default_callback
        self.callback = callback

        self.value: str = ""

    def to_dict(self):
        res = super().to_dict()
        res["state"] = {
            "language": self.language,
            "value": self.value,
        }
        return res

    async def _default_callback(self, value: str):
        self.value = value


class Switch(Component):

    def __init__(self,
                 uid: str,
                 label: str,
                 callback: Optional[Callable[[bool], Coroutine[None, None,
                                                               None]]],
                 queue: asyncio.Queue,
                 flex: Optional[str] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.Switch, queue, flex, align_self)
        self.label = label
        if callback is None:
            callback = self._default_callback
        self.callback = callback
        self.checked = False

    def to_dict(self):
        res = super().to_dict()
        res["label"] = self.label
        res["state"] = {
            "checked": self.checked,
        }
        return res

    async def _default_callback(self, value: bool):
        self.checked = value


class Select(Component):

    def __init__(self,
                 uid: str,
                 label: str,
                 items: List[Tuple[str, Any]],
                 callback: Optional[Callable[[Any], Coroutine[None, None,
                                                              None]]],
                 queue: asyncio.Queue,
                 flex: Optional[str] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.Select, queue, flex, align_self)
        self.label = label
        if callback is None:
            callback = self._default_callback
        self.callback = callback
        assert len(items) > 0
        self.items = items
        self.value = items[0][1]

    def to_dict(self):
        res = super().to_dict()
        res["label"] = self.label
        res["state"] = {
            "items": self.items,
            "value": self.value,
        }
        return res

    async def update_items(self, items: List[Tuple[str, Any]], selected: int):
        await self.queue.put(
            self.create_update_event({
                "items": items,
                "value": items[selected][1]
            }))
        self.items = items
        self.value = items[selected][1]

    async def update_value(self, value: Any):
        assert value in [x[1] for x in self.items]
        await self.queue.put(self.create_update_event({"value": value}))
        self.value = value

    async def _default_callback(self, value: Any):
        self.value = value


class Slider(Component):

    def __init__(self,
                 uid: str,
                 label: str,
                 begin: Union[int, float],
                 end: Union[int, float],
                 step: Union[int, float],
                 callback: Optional[Callable[[Union[int, float]],
                                             Coroutine[None, None, None]]],
                 queue: asyncio.Queue,
                 flex: Optional[str] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.Slider, queue, flex, align_self)
        self.label = label
        if callback is None:
            callback = self._default_callback
        self.callback = callback
        assert end > begin and step < end - begin
        self.ranges = (begin, end, step)
        self.value = begin

    def to_dict(self):
        res = super().to_dict()
        res["label"] = self.label
        res["state"] = {
            "value": self.value,
            "ranges": self.ranges,
        }
        return res

    async def update_ranges(self, begin: Union[int, float],
                            end: Union[int, float], step: Union[int, float]):
        await self.queue.put(
            self.create_update_event({"ranges": (begin, end, step)}))
        self.ranges = (begin, end, step)
        assert end > begin and step < end - begin
        self.value = begin

    async def update_value(self, value: Union[int, float]):
        assert value >= self.ranges[0] and value <= self.ranges[1]
        await self.queue.put(self.create_update_event({"value": value}))
        self.value = value

    async def _default_callback(self, value: Union[int, float]):
        self.value = value


_T = TypeVar("_T")


class TaskLoop(Component):

    def __init__(self,
                 uid: str,
                 label: str,
                 loop_callbcak: Callable[[], Coroutine[None, None, None]],
                 queue: asyncio.Queue,
                 update_period: float = 0.2,
                 flex: Optional[str] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.TaskLoop, queue, flex, align_self)
        self.label = label
        self.loop_callbcak = loop_callbcak

        self.progresses: List[float] = [0.0]
        self.stack_count = 0
        self.pause_event = asyncio.Event()
        self.pause_event.set()
        self.update_period = update_period

    def to_dict(self):
        res = super().to_dict()
        res["state"] = {
            "label": self.label,
            "progresses": self.progresses,
        }
        return res

    async def task_loop(self,
                        it: Iterable[_T],
                        total: int = -1) -> AsyncGenerator[_T, None]:
        if isinstance(it, list):
            total = len(it)
        try:
            cnt = 0
            t = time.time()
            dura = 0.0
            if self.stack_count > 0:
                # keep root progress
                self.progresses.append(0.0)
            self.stack_count += 1
            for item in it:
                yield item
                # await asyncio.sleep(0)
                await self.pause_event.wait()
                cnt += 1
                dura += time.time() - t

                if total > 0 and dura > self.update_period:
                    dura = 0
                    prog = cnt / total
                    await self.update_progress(prog, self.stack_count - 1)
                t = time.time()
            await self.update_progress(1.0, self.stack_count - 1)
        finally:
            self.stack_count -= 1
            self.pause_event.set()
            if len(self.progresses) > 1:
                self.progresses.pop()

    async def update_progress(self, progress: float, index: int):
        progress = max(0, min(progress, 1))
        self.progresses[index] = progress
        await self.queue.put(
            self.create_update_event({"progresses": self.progresses}))

    async def update_label(self, label: str):
        await self.queue.put(self.create_update_event({"label": label}))
        self.label = label


_ROOT = "root"


class App:
    # TODO find a way to sync state to frontend uncontrolled elements.
    def __init__(self,
                 flex_flow: Optional[str] = "column nowrap",
                 justify_content: Optional[str] = None,
                 align_items: Optional[str] = None,
                 maxqsize: int = 10) -> None:
        self._uid_to_comp: Dict[str, Component] = {}
        # TODO when app run offline, make sure queue put won't block program.
        self._queue = asyncio.Queue(maxsize=maxqsize)

        self._send_callback: Optional[Callable[[AppEvent],
                                               Coroutine[None, None,
                                                         None]]] = None
        self._pool = UniqueNamePool()
        root = FlexBox(_ROOT, self._pool, self._queue, self._uid_to_comp,
                       flex_flow, justify_content, align_items)
        self._uid_to_comp[_ROOT] = root
        self._pool(_ROOT)

        self.root = root

    def _get_app_layout(self):
        return {
            "layout": {u: c.to_dict()
                       for u, c in self._uid_to_comp.items()},
        }

    def set_init_window_size(self, size: List[Union[int, None]]):
        self.root.width = size[0]
        self.root.height = size[1]

    async def _handle_control_event(self, ev: UIEvent):
        # TODO run control fron other component
        for uid, data in ev.uid_to_data.items():
            comp = self._uid_to_comp[uid]
            if isinstance(comp, (Buttons, Button, Input, Switch, Select,
                                 Slider, RadioGroup, CodeEditor)):
                if comp._status == UIRunStatus.Running:
                    # TODO send exception if ignored click
                    print("IGNORE EVENT", comp._status)
                    return
                elif comp._status == UIRunStatus.Stop:
                    cb = comp.callback
                    comp._task = asyncio.create_task(
                        comp.run_callback(lambda: cb(data)))
            elif isinstance(comp, TaskLoop):
                if data == TaskLoopEvent.Start.value:
                    if comp._status == UIRunStatus.Stop:
                        comp._task = asyncio.create_task(
                            comp.run_callback(comp.loop_callbcak))
                    else:
                        print("IGNORE TaskLoop EVENT", comp._status)
                elif data == TaskLoopEvent.Pause.value:
                    if comp._status == UIRunStatus.Running:
                        # pause
                        comp.pause_event.clear()
                        comp._status = UIRunStatus.Pause
                    elif comp._status == UIRunStatus.Pause:
                        comp.pause_event.set()
                        comp._status = UIRunStatus.Running
                    else:
                        print("IGNORE TaskLoop EVENT", comp._status)
                elif data == TaskLoopEvent.Stop.value:
                    if comp._status == UIRunStatus.Running:
                        await cancel_task(comp._task)
                    else:
                        print("IGNORE TaskLoop EVENT", comp._status)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

    async def copy_text_to_clipboard(self, text: str):
        """copy to clipboard in frontend."""
        await self._queue.put(
            AppEvent(
                "",
                {AppEventType.CopyToClipboard: CopyToClipboardEvent(text)}))
