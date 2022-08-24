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

import asyncio
import base64
import dataclasses
import io
import time
from typing import (Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
                    Iterable, List, Optional, Tuple, Type, TypeVar, Union)
from typing_extensions import Literal, TypeAlias
import numpy as np
from PIL import Image as PILImage
import json
from ..core import (AppEvent, BasicProps, Component, ComponentBaseProps,
                    ContainerBase, TaskLoopEvent, UIEvent, UIType, Undefined,
                    undefined, TBaseComp, ValueType)

_CORO_NONE = Union[Coroutine[None, None, None], None]


def _encode_image_bytes(img: np.ndarray):
    pil_img = PILImage.fromarray(img)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    b64_bytes = base64.b64encode(buffered.getvalue())
    return b"data:image/png;base64," + b64_bytes


@dataclasses.dataclass
class MUIBasicProps(BasicProps):
    pass


@dataclasses.dataclass
class MUIComponentBaseProps(ComponentBaseProps):
    pass


class MUIComponentBase(Component[TBaseComp, "MUIComponentType"]):
    pass


class MUIContainerBase(ContainerBase[TBaseComp, "MUIComponentType"]):
    pass


@dataclasses.dataclass
class FlexBoxProps(ComponentBaseProps):
    # TODO add literal here.
    align_content: Union[str, Undefined] = undefined
    align_items: Union[str, Undefined] = undefined
    justify_content: Union[str, Undefined] = undefined
    flex_direction: Union[str, Undefined] = undefined
    flex_wrap: Union[str, Undefined] = undefined
    flex_flow: Union[str, Undefined] = undefined


# we can't let mui use three component.
@dataclasses.dataclass
class MUIFlexBoxProps(FlexBoxProps):
    pass


MUIComponentType: TypeAlias = Union[MUIBasicProps, MUIComponentBase[TBaseComp], MUIContainerBase[TBaseComp],
                         MUIFlexBoxProps, MUIComponentBaseProps]


class Images(MUIComponentBase[MUIComponentBaseProps]):
    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.Image, MUIComponentBaseProps, queue)
        self.image_str: bytes = b""

    async def show(self, image: np.ndarray):
        encoded = _encode_image_bytes(image)
        self.image_str = encoded
        await self.queue.put(self.create_update_event({
            "image": encoded,
        }))

    async def show_raw(self, image_bytes: bytes, suffix: str):
        await self.queue.put(self.show_raw_event(image_bytes, suffix))

    def show_raw_event(self, image_bytes: bytes, suffix: str):
        raw = b'data:image/' + suffix.encode(
            "utf-8") + b';base64,' + base64.b64encode(image_bytes)
        self.image_str = raw
        return self.create_update_event({
            "image": raw,
        })

    def get_state(self):
        state = super().get_state()
        state["image"] = self.image_str
        return state

    def set_state(self, state: Dict[str, Any]):
        super().set_state(state)
        if "image" in state:
            self.image_str = state["image"]


class Plotly(MUIComponentBase[MUIComponentBaseProps]):
    def __init__(self,
                 data: Optional[list] = None,
                 layout: Optional[dict] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.Plotly, MUIComponentBaseProps, queue)
        if data is None:
            data = []
        if layout is None:
            layout = {}
        self.data = data
        self.layout = layout

    async def show_raw(self, data: list, layout: Any):
        await self.queue.put(
            self.create_update_event({
                "data": data,
                "layout": layout,
            }))

    def get_state(self):
        state = super().get_state()
        state["data"] = self.data
        state["layout"] = self.layout
        return state

    def set_state(self, state: Dict[str, Any]):
        super().set_state(state)
        if "data" in state:
            self.data = state["data"]
        if "layout" in state:
            self.layout = state["layout"]


class ChartJSLine(MUIComponentBase[MUIComponentBaseProps]):
    def __init__(self,
                 data: Optional[Any] = None,
                 options: Optional[Any] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ChartJSLine, MUIComponentBaseProps, queue)
        if data is None:
            data = {}
        if options is None:
            options = {}
        self.data = data
        self.options = options

    async def show_raw(self, data: list, options: Any):
        await self.queue.put(
            self.create_update_event({
                "data": data,
                "options": options,
            }))

    def get_state(self):
        state = super().get_state()
        state["data"] = self.data
        state["options"] = self.options
        return state

    def set_state(self, state: Dict[str, Any]):
        super().set_state(state)
        if "data" in state:
            self.data = state["data"]
        if "options" in state:
            self.options = state["options"]


class Text(MUIComponentBase[MUIComponentBaseProps]):
    def __init__(self,
                 init: str,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.Text, MUIComponentBaseProps, queue)
        self.value = init

    async def write(self, content: str):
        self.value = content
        await self.queue.put(self.create_update_event({"value": self.value}))

    def get_state(self):
        state = super().get_state()
        state["value"] = self.value
        return state

    def set_state(self, state: Dict[str, Any]):
        super().set_state(state)
        if "value" in state:
            self.value = state["value"]


class ListItemText(MUIComponentBase[MUIComponentBaseProps]):
    def __init__(self,
                 init: str,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ListItemText, MUIComponentBaseProps,
                         queue)
        self.value = init

    async def write(self, content: str):
        self.value = content
        await self.queue.put(self.create_update_event({"value": self.value}))

    def get_state(self):
        state = super().get_state()
        state["value"] = self.value
        return state

    def set_state(self, state: Dict[str, Any]):
        super().set_state(state)
        if "value" in state:
            self.value = state["value"]


class Divider(MUIComponentBase[MUIComponentBaseProps]):
    def __init__(self,
                 orientation: Union[Literal["horizontal"], Literal["vertical"]] = "horizontal",
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.Divider, MUIComponentBaseProps, queue)
        self.orientation = orientation
        assert orientation == "horizontal" or orientation == "vertical"

    def to_dict(self):
        res = super().to_dict()
        res["orientation"] = self.orientation
        return res 

class Button(MUIComponentBase[MUIComponentBaseProps]):
    def __init__(self,
                 name: str,
                 callback: Callable[[], _CORO_NONE],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.Button, MUIComponentBaseProps, queue)
        self.name = name
        self.callback = callback

    def to_dict(self):
        res = super().to_dict()
        res["name"] = self.name
        return res

    async def headless_click(self):
        return await self.queue.put(UIEvent({self.uid: self.name}))

    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val


class ListItemButton(MUIComponentBase[MUIComponentBaseProps]):
    def __init__(self,
                 name: str,
                 callback: Callable[[], _CORO_NONE],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ListItemButton, MUIComponentBaseProps,
                         queue)
        self.name = name
        self.callback = callback

    def to_dict(self):
        res = super().to_dict()
        res["name"] = self.name
        return res

    async def headless_click(self):
        return await self.queue.put(UIEvent({self.uid: self.name}))

    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val


class Buttons(MUIComponentBase[MUIComponentBaseProps]):
    def __init__(self,
                 names: List[str],
                 callback: Callable[[str], _CORO_NONE],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.Buttons, MUIComponentBaseProps, queue)
        self.names = names
        self.callback = callback

    def to_dict(self):
        res = super().to_dict()
        res["names"] = self.names
        return res

    async def headless_click(self, index: int):
        return await self.queue.put(UIEvent({self.uid: self.names[index]}))

    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val


class FlexBox(MUIContainerBase[MUIFlexBoxProps]):
    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 _init_dict: Optional[Dict[str, MUIComponentType]] = None,
                 base_type: UIType = UIType.FlexBox,
                 inited: bool = False) -> None:
        super().__init__(base_type, MUIFlexBoxProps, uid, queue, uid_to_comp,
                         _init_dict, inited)


class MUIList(MUIContainerBase[MUIFlexBoxProps]):
    def __init__(self,
                 uid: str,
                 queue: asyncio.Queue,
                 uid_to_comp: Dict[str, Component],
                 _init_dict: Optional[Dict[str, MUIComponentType]] = None,
                 subheader: str = "",
                 inited: bool = False) -> None:
        super().__init__(UIType.MUIList,
                         MUIFlexBoxProps,
                         uid,
                         queue=queue,
                         uid_to_comp=uid_to_comp,
                         _init_dict=_init_dict,
                         inited=inited)
        self.subheader = subheader

    def get_state(self):
        state = super().get_state()
        state["subheader"] = self.subheader
        return state

    def set_state(self, state: Dict[str, Any]):
        super().set_state(state)
        if "subheader" in state:
            self.subheader = state["subheader"]


def VBox(layout: Dict[str, MUIComponentType]):
    res = FlexBox("", asyncio.Queue(), {}, _init_dict=layout)
    res.prop(flex_flow="column nowrap")
    return res


def HBox(layout: Dict[str, MUIComponentType], ):
    res = FlexBox("", asyncio.Queue(), {}, _init_dict=layout)
    res.prop(flex_flow="row nowrap")
    return res


def Box(layout: Dict[str, MUIComponentType]):
    return FlexBox("", asyncio.Queue(), {}, _init_dict=layout)


def VList(layout: Dict[str, MUIComponentType],
          subheader: str = ""):
    return MUIList("",
                   asyncio.Queue(), {},
                   subheader=subheader,
                   _init_dict=layout)


class RadioGroup(MUIComponentBase[MUIComponentBaseProps]):
    def __init__(self,
                 names: List[str],
                 row: bool,
                 callback: Optional[Callable[[str], Coroutine[None, None,
                                                              None]]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.RadioGroup, MUIComponentBaseProps, queue)
        self.names = names
        self.callback = callback
        self.row = row
        self.value = names[0]

    def to_dict(self):
        res = super().to_dict()
        res["names"] = self.names
        res["row"] = self.row
        return res

    def get_state(self):
        state = super().get_state()
        state["value"] = self.value
        return state

    def set_state(self, state: Dict[str, Any]):
        super().set_state(state)
        # noexcept here.
        if state["value"] in self.names:
            self.value = state["value"]

    def state_change_callback(self, data: str):
        self.value = data

    async def update_value(self, value: Any):
        assert value in self.names
        await self.queue.put(self.create_update_event({"value": value}))
        self.value = value

    async def headless_click(self, index: int):
        return await self.queue.put(UIEvent({self.uid: self.names[index]}))

    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val


class Input(MUIComponentBase[MUIComponentBaseProps]):
    def __init__(self,
                 label: str,
                 multiline: bool = False,
                 password: bool = False,
                 callback: Optional[Callable[[str], Coroutine[None, None,
                                                              None]]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 init: str = "") -> None:
        super().__init__(uid, UIType.Input, MUIComponentBaseProps, queue)
        self.label = label
        self.callback = callback
        self.value: str = init
        self.multiline = multiline
        self.password = password

    def to_dict(self):
        res = super().to_dict()
        res["label"] = self.label
        res["multiline"] = self.multiline
        res["password"] = self.password
        return res

    def get_state(self):
        state = super().get_state()
        state["value"] = self.value
        return state

    def set_state(self, state: Dict[str, Any]):
        super().set_state(state)
        self.value = state["value"]

    def state_change_callback(self, data: str):
        self.value = data

    async def headless_write(self, content: str):
        return await self.queue.put(UIEvent({self.uid: content}))

    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val

    def json(self):
        return json.loads(self.value)

    def float(self):
        return float(self.value)

    def int(self):
        return int(self.value)


# class CodeEditor(MUIComponentBase[MUIComponentBaseProps]):

#     def __init__(self,
#                  language: str,
#                  callback: Optional[Callable[[str], Coroutine[None, None,
#                                                               None]]] = None,
#                  uid: str = "",
#                  queue: Optional[asyncio.Queue] = None,
#                  flex: Union[int, str, Undefined] = undefined,
#                  align_self: Union[str, Undefined] = undefined) -> None:
#         super().__init__(uid, UIType.CodeEditor, queue, flex, align_self)
#         self.language = language
#         self.callback = callback
#         self.value: str = ""

#     def get_state(self):
#         state = super().get_state()
#         state["language"] = self.language
#         state["value"] = self.value
#         return state

#     def state_change_callback(self, data: str):
#         self.value = data


class Switch(MUIComponentBase[MUIComponentBaseProps]):
    def __init__(self,
                 label: str,
                 callback: Optional[Callable[[bool], Coroutine[None, None,
                                                               None]]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.Switch, MUIComponentBaseProps, queue)
        self.label = label
        self.callback = callback
        self.checked = False

    def to_dict(self):
        res = super().to_dict()
        res["label"] = self.label
        return res

    def get_state(self):
        state = super().get_state()
        state["checked"] = self.checked
        return state

    def set_state(self, state: Dict[str, Any]):
        super().set_state(state)
        if "checked" in state:
            self.checked = state["checked"]

    def state_change_callback(self, data: bool):
        self.checked = data

    async def headless_write(self, checked: bool):
        return await self.queue.put(UIEvent({self.uid: checked}))

    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val

    def __bool__(self):
        return self.checked


class Select(MUIComponentBase[MUIComponentBaseProps]):
    def __init__(self,
                 label: str,
                 items: List[Tuple[str, ValueType]],
                 callback: Optional[Callable[[ValueType],
                                             Coroutine[None, None,
                                                       None]]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.Select, MUIComponentBaseProps, queue)
        self.label = label
        self.callback = callback
        assert len(items) > 0
        self.items = items
        # item value must implement eq/ne
        self.value = ""

    def to_dict(self):
        res = super().to_dict()
        res["label"] = self.label
        return res

    def get_state(self):
        state = super().get_state()
        state["items"] = self.items
        state["value"] = self.value
        return state

    def set_state(self, state: Dict[str, Any]):
        super().set_state(state)
        if "value" in state and "items" in state:
            value = state["value"]
            items = state["items"]
            if items == self.items:
                self.value = value
                self.items = items

    async def update_items(self, items: List[Tuple[str, ValueType]],
                           selected: int):
        await self.queue.put(
            self.create_update_event({
                "items": items,
                "value": items[selected][1]
            }))
        self.items = items
        self.value = items[selected][1]

    async def update_value(self, value: ValueType):
        assert value in [x[1] for x in self.items]
        await self.queue.put(self.create_update_event({"value": value}))
        self.value = value

    def update_value_no_sync(self, value: ValueType):
        assert value in [x[1] for x in self.items]
        self.value = value

    def state_change_callback(self, value: ValueType):
        self.value = value

    async def headless_select(self, value: ValueType):
        return await self.queue.put(UIEvent({self.uid: value}))

    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val


class MultipleSelect(MUIComponentBase[MUIComponentBaseProps]):
    def __init__(self,
                 label: str,
                 items: List[Tuple[str, ValueType]],
                 callback: Optional[Callable[[List[ValueType]],
                                             Coroutine[None, None,
                                                       None]]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.MultipleSelect, MUIComponentBaseProps,
                         queue)
        self.label = label
        self.callback = callback
        assert len(items) > 0
        self.items = items
        # item value must implement eq/ne
        self.values: List[ValueType] = []

    def to_dict(self):
        res = super().to_dict()
        res["label"] = self.label
        return res

    def get_state(self):
        state = super().get_state()
        state["items"] = self.items
        state["values"] = self.values
        return state

    def set_state(self, state: Dict[str, Any]):
        super().set_state(state)
        if "values" in state and "items" in state:
            values = state["values"]
            items = state["items"]
            if [list(x) for x in items] == [list(x) for x in self.items]:
                self.values = values
                self.items = items

    async def update_items(self,
                           items: List[Tuple[str, Any]],
                           selected: Optional[List[int]] = None):
        if selected is None:
            selected = []
        await self.queue.put(
            self.create_update_event({
                "items": items,
                "values": [items[s][1] for s in selected]
            }))
        self.items = items
        self.values = [items[s][1] for s in selected]

    async def update_value(self, values: List[ValueType]):
        for v in values:
            assert v in [x[1] for x in self.items]
        await self.queue.put(self.create_update_event({"values": values}))
        self.values = values

    def update_value_no_sync(self, values: List[ValueType]):
        for v in values:
            assert v in [x[1] for x in self.items]
        self.values = values

    def state_change_callback(self, values: List[ValueType]):
        self.values = values

    async def headless_select(self, values: List[ValueType]):
        return await self.queue.put(UIEvent({self.uid: values}))

    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val


class Slider(MUIComponentBase[MUIComponentBaseProps]):
    def __init__(self,
                 label: str,
                 begin: Union[int, float],
                 end: Union[int, float],
                 step: Union[int, float],
                 callback: Optional[Callable[[Union[int, float]],
                                             _CORO_NONE]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.Slider, MUIComponentBaseProps, queue)
        self.label = label
        self.callback = callback
        assert end > begin and step < end - begin
        self.ranges = (begin, end, step)
        self.value = begin

    def to_dict(self):
        res = super().to_dict()
        res["label"] = self.label
        return res

    def get_state(self):
        state = super().get_state()
        state["ranges"] = self.ranges
        state["value"] = self.value
        return state

    def set_state(self, state: Dict[str, Any]):
        super().set_state(state)
        if "value" in state and "ranges" in state:
            self.value = state["value"]
            self.ranges = state["ranges"]

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

    def state_change_callback(self, value: Union[int, float]):
        self.value = value

    async def headless_change(self, value: Union[int, float]):
        return await self.queue.put(UIEvent({self.uid: value}))

    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val


_T = TypeVar("_T")


class TaskLoop(MUIComponentBase[MUIComponentBaseProps]):
    def __init__(self,
                 label: str,
                 loop_callbcak: Callable[[], _CORO_NONE],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 update_period: float = 0.2) -> None:
        super().__init__(uid, UIType.TaskLoop, MUIComponentBaseProps, queue)
        self.label = label
        self.loop_callbcak = loop_callbcak

        self.progresses: List[float] = [0.0]
        self.stack_count = 0
        self.pause_event = asyncio.Event()
        self.pause_event.set()
        self.update_period = update_period

    def get_state(self):
        state = super().get_state()
        state["label"] = self.label
        state["progresses"] = self.progresses
        return state

    def get_callback(self):
        return self.loop_callbcak

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

    async def headless_run(self):
        return await self.queue.put(
            UIEvent({self.uid: TaskLoopEvent.Start.value}))

    def set_callback(self, val: Any):
        self.loop_callbcak = val
