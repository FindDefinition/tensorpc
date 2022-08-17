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
import io
import time
from typing import (Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
                    Iterable, List, Optional, Tuple, TypeVar, Union)

import numpy as np
from PIL import Image

from ..core import AppEvent, Component, TaskLoopEvent, UIEvent, UIType, ContainerBase

_CORO_NONE = Union[Coroutine[None, None, None], None]


def _encode_image_bytes(img: np.ndarray):
    pil_img = Image.fromarray(img)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    b64_bytes = base64.b64encode(buffered.getvalue())
    return b"data:image/png;base64," + b64_bytes


class Images(Component):
    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
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

    def get_state(self):
        state = super().get_state()
        state["image"] = self.image_str
        return state


class Plotly(Component):
    def __init__(self,
                 data: Optional[list] = None,
                 layout: Optional[dict] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.Plotly, queue, flex, align_self)
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


class ChartJSLine(Component):
    def __init__(self,
                 data: Optional[Any] = None,
                 options: Optional[Any] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.ChartJSLine, queue, flex, align_self)
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


class Text(Component):
    def __init__(self,
                 init: str,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.Text, queue, flex, align_self)
        self.value = init

    async def write(self, content: str):
        self.value = content
        await self.queue.put(self.create_update_event({"value": self.value}))

    def get_state(self):
        state = super().get_state()
        state["value"] = self.value
        return state


class ListItemText(Component):
    def __init__(self,
                 init: str,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.ListItemText, queue, flex, align_self)
        self.value = init

    async def write(self, content: str):
        self.value = content
        await self.queue.put(self.create_update_event({"value": self.value}))

    def get_state(self):
        state = super().get_state()
        state["value"] = self.value
        return state


class Divider(Component):
    def __init__(self,
                 orientation: str = "horizontal",
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.Divider, queue, flex, align_self)
        self.orientation = orientation
        assert orientation == "horizontal" or orientation == "vertical"

    def to_dict(self):
        res = super().to_dict()
        res["orientation"] = self.orientation


class Button(Component):
    def __init__(self,
                 name: str,
                 callback: Callable[[], _CORO_NONE],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.Button, queue, flex, align_self)
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


class ListItemButton(Component):
    def __init__(self,
                 name: str,
                 callback: Callable[[], _CORO_NONE],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.ListItemButton, queue, flex, align_self)
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


class Buttons(Component):
    def __init__(self,
                 names: List[str],
                 callback: Callable[[str], _CORO_NONE],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.Buttons, queue, flex, align_self)
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


class Points(Component):
    def __init__(self,
                 names: List[str],
                 callback: Callable[[str], _CORO_NONE],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.Buttons, queue, flex, align_self)
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


class FlexBox(ContainerBase):
    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 flex_flow: Optional[str] = None,
                 justify_content: Optional[str] = None,
                 align_items: Optional[str] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None,
                 width: Optional[Union[str, int]] = None,
                 height: Optional[Union[str, int]] = None,
                 overflow: Optional[str] = None,
                 min_width: Optional[Union[str, int]] = None,
                 min_height: Optional[Union[str, int]] = None,
                 _init_dict: Optional[Dict[str, Component]] = None,
                 base_type: UIType = UIType.FlexBox,
                 inited: bool = False) -> None:
        super().__init__(UIType.FlexBox, uid, queue, flex, align_self,
                         uid_to_comp, _init_dict, inited)
        self.flex_flow = flex_flow
        self.justify_content = justify_content
        self.align_items = align_items
        self.width = width
        self.height = height
        self.overflow = overflow
        self.min_width = min_width
        self.min_height = min_height

    def to_dict(self):
        res = super().to_dict()
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
        if self.min_height is not None:
            res["minHeight"] = self.min_height
        if self.overflow is not None:
            res["overflow"] = self.overflow
        if self.min_width is not None:
            res["minWidth"] = self.min_width
        return res

    # async def update_child(self, comps: List[Union[Component, str]]):
    #     newchilds: List[str] = []
    #     for c in comps:
    #         uid = ""
    #         if isinstance(c, "str"):
    #             uid = c
    #         else:
    #             uid = c.uid
    #         assert uid in self._uid_to_comp
    #         newchilds.append(uid)
    #     self._childs = newchilds
    #     await self.queue.put(self.create_update_comp_event(self.get_state()))

    def add_flex_box(self,
                     flex_flow: Optional[str] = None,
                     justify_content: Optional[str] = None,
                     align_items: Optional[str] = None,
                     flex: Optional[Union[int, str]] = None,
                     align_self: Optional[str] = None,
                     width: Optional[Union[str, int]] = None,
                     height: Optional[Union[str, int]] = None,
                     overflow: Optional[str] = None,
                     min_width: Optional[Union[str, int]] = None,
                     min_height: Optional[Union[str, int]] = None):
        ui = FlexBox("",
                     self.queue,
                     self._uid_to_comp,
                     flex_flow,
                     justify_content,
                     align_items,
                     flex,
                     align_self,
                     width,
                     height,
                     overflow,
                     min_width,
                     min_height,
                     inited=self.inited)
        self.add_component("box", ui)
        return ui

    def add_buttons(self,
                    names: List[str],
                    callback: Callable[[str], _CORO_NONE],
                    flex: Optional[Union[int, str]] = None,
                    align_self: Optional[str] = None):
        ui = Buttons(names, callback, "", self.queue, flex, align_self)
        self.add_component("btns", ui)
        return ui

    def add_button(self,
                   name: str,
                   callback: Callable[[], _CORO_NONE],
                   flex: Optional[Union[int, str]] = None,
                   align_self: Optional[str] = None):
        ui = Button(name, callback, "", self.queue, flex, align_self)
        self.add_component("btn", ui)
        return ui

    def add_list_item_button(self,
                             name: str,
                             callback: Callable[[], _CORO_NONE],
                             flex: Optional[Union[int, str]] = None,
                             align_self: Optional[str] = None):
        # TODO check parent must be list or collapse
        ui = ListItemButton(name, callback, "", self.queue, flex, align_self)
        self.add_component("lbtn", ui)
        return ui

    def add_input(self,
                  label: str,
                  callback: Optional[Callable[[str], Coroutine[None, None,
                                                               None]]] = None,
                  flex: Optional[Union[int, str]] = None,
                  align_self: Optional[str] = None):
        ui = Input(label, callback, "", self.queue, flex, align_self)
        self.add_component("inp", ui)
        return ui

    # def add_code_editor(self,
    #                     language: str,
    #                     callback: Optional[Callable[[str],
    #                                                 Coroutine[None, None,
    #                                                           None]]] = None,
    #                     flex: Optional[Union[int, str]] = None,
    #                     align_self: Optional[str] = None):
    #     ui = CodeEditor(language, callback, "", self.queue, flex, align_self)
    #     self.add_component("code", ui)
    #     return ui

    def add_switch(self,
                   label: str,
                   callback: Optional[Callable[[bool],
                                               Coroutine[None, None,
                                                         None]]] = None,
                   flex: Optional[Union[int, str]] = None,
                   align_self: Optional[str] = None):
        ui = Switch(label, callback, "", self.queue, flex, align_self)
        self.add_component("switch", ui)
        return ui

    def add_image(self,
                  flex: Optional[Union[int, str]] = None,
                  align_self: Optional[str] = None):
        ui = Images("", self.queue, flex, align_self)
        self.add_component("img", ui)
        return ui

    def add_text(self,
                 init: str,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None):
        ui = Text(init, "", self.queue, flex, align_self)
        self.add_component("text", ui)
        return ui

    def add_list_item_text(self,
                           init: str,
                           flex: Optional[Union[int, str]] = None,
                           align_self: Optional[str] = None):
        ui = ListItemText(init, "", self.queue, flex, align_self)
        self.add_component("ltext", ui)
        return ui

    def add_radio_group(self,
                        names: List[str],
                        row: bool,
                        callback: Optional[Callable[[str],
                                                    Coroutine[None, None,
                                                              None]]] = None,
                        flex: Optional[Union[int, str]] = None,
                        align_self: Optional[str] = None):
        ui = RadioGroup(names, row, callback, "", self.queue, flex, align_self)
        self.add_component("radio", ui)
        return ui

    def add_select(self,
                   label: str,
                   items: List[Tuple[str, Any]],
                   callback: Optional[Callable[[Any], Coroutine[None, None,
                                                                None]]] = None,
                   flex: Optional[Union[int, str]] = None,
                   align_self: Optional[str] = None):
        ui = Select(label, items, callback, "", self.queue, flex, align_self)
        self.add_component("select", ui)
        return ui

    def add_slider(self,
                   label: str,
                   begin: Union[int, float],
                   end: Union[int, float],
                   step: Union[int, float],
                   callback: Optional[Callable[[Union[int, float]],
                                               Coroutine[None, None,
                                                         None]]] = None,
                   flex: Optional[Union[int, str]] = None,
                   align_self: Optional[str] = None):
        ui = Slider(label, begin, end, step, callback, "", self.queue, flex,
                    align_self)
        self.add_component("slider", ui)
        return ui

    def add_divider(self,
                    orientation: str = "horizontal",
                    flex: Optional[Union[int, str]] = None,
                    align_self: Optional[str] = None):
        ui = Divider(orientation, "", self.queue, flex, align_self)
        self.add_component("divider", ui)
        return ui

    def add_task_loop(self,
                      label: str,
                      callback: Callable[[], _CORO_NONE],
                      update_period: float = 0.2,
                      flex: Optional[Union[int, str]] = None,
                      align_self: Optional[str] = None):
        """use ASYNC LOOP in this ui!!!!!!!!!!!!!
        DON'T USE OTHER LOOP!!!
        """
        ui = TaskLoop(label, callback, "", self.queue, update_period, flex,
                      align_self)
        self.add_component("task", ui)
        return ui


class MUIList(FlexBox):
    def __init__(self,
                 uid: str,
                 queue: asyncio.Queue,
                 uid_to_comp: Dict[str, Component],
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None,
                 width: Optional[Union[str, int]] = None,
                 height: Optional[Union[str, int]] = None,
                 overflow: Optional[str] = None,
                 min_width: Optional[Union[str, int]] = None,
                 min_height: Optional[Union[str, int]] = None,
                 _init_dict: Optional[Dict[str, Component]] = None,
                 subheader: str = "",
                 inited: bool = False) -> None:
        super().__init__(uid,
                         queue=queue,
                         uid_to_comp=uid_to_comp,
                         flex=flex,
                         align_self=align_self,
                         width=width,
                         height=height,
                         overflow=overflow,
                         min_width=min_width,
                         min_height=min_height,
                         _init_dict=_init_dict,
                         base_type=UIType.MUIList,
                         inited=inited)
        self.subheader = subheader

    def get_state(self):
        state = super().get_state()
        state["subheader"] = self.subheader
        return state


def VBox(layout: Dict[str, Component],
         justify_content: Optional[str] = None,
         align_items: Optional[str] = None,
         flex: Optional[Union[int, str]] = None,
         align_self: Optional[str] = None,
         width: Optional[Union[str, int]] = None,
         height: Optional[Union[str, int]] = None,
         overflow: Optional[str] = None,
         min_width: Optional[Union[str, int]] = None,
         min_height: Optional[Union[str, int]] = None):
    return FlexBox("",
                   asyncio.Queue(), {},
                   flex_flow="column nowrap",
                   justify_content=justify_content,
                   align_items=align_items,
                   flex=flex,
                   align_self=align_self,
                   width=width,
                   height=height,
                   overflow=overflow,
                   min_width=min_width,
                   min_height=min_height,
                   _init_dict=layout)


def HBox(layout: Dict[str, Component],
         justify_content: Optional[str] = None,
         align_items: Optional[str] = None,
         flex: Optional[Union[int, str]] = None,
         align_self: Optional[str] = None,
         width: Optional[Union[str, int]] = None,
         height: Optional[Union[str, int]] = None,
         overflow: Optional[str] = None,
         min_width: Optional[Union[str, int]] = None,
         min_height: Optional[Union[str, int]] = None):
    return FlexBox("",
                   asyncio.Queue(), {},
                   flex_flow="row nowrap",
                   justify_content=justify_content,
                   align_items=align_items,
                   flex=flex,
                   align_self=align_self,
                   width=width,
                   height=height,
                   overflow=overflow,
                   min_width=min_width,
                   min_height=min_height,
                   _init_dict=layout)


def Box(layout: Dict[str, Component],
        flex_flow: Optional[str] = None,
        justify_content: Optional[str] = None,
        align_items: Optional[str] = None,
        flex: Optional[Union[int, str]] = None,
        align_self: Optional[str] = None,
        width: Optional[Union[str, int]] = None,
        height: Optional[Union[str, int]] = None,
        overflow: Optional[str] = None,
        min_width: Optional[Union[str, int]] = None,
        min_height: Optional[Union[str, int]] = None):
    return FlexBox("",
                   asyncio.Queue(), {},
                   flex_flow=flex_flow,
                   justify_content=justify_content,
                   align_items=align_items,
                   flex=flex,
                   align_self=align_self,
                   width=width,
                   height=height,
                   overflow=overflow,
                   min_width=min_width,
                   min_height=min_height,
                   _init_dict=layout)


def VList(layout: Dict[str, Component],
          subheader: str = "",
          flex: Optional[Union[int, str]] = None,
          align_self: Optional[str] = None,
          width: Optional[Union[str, int]] = None,
          height: Optional[Union[str, int]] = None,
          min_width: Optional[Union[str, int]] = None,
          min_height: Optional[Union[str, int]] = None,
          overflow: Optional[str] = None):
    return MUIList("",
                   asyncio.Queue(), {},
                   flex=flex,
                   align_self=align_self,
                   overflow=overflow,
                   subheader=subheader,
                   width=width,
                   height=height,
                   min_width=min_width,
                   min_height=min_height,
                   _init_dict=layout)


class RadioGroup(Component):
    def __init__(self,
                 names: List[str],
                 row: bool,
                 callback: Optional[Callable[[str], Coroutine[None, None,
                                                              None]]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.RadioGroup, queue, flex, align_self)
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


class Input(Component):
    def __init__(self,
                 label: str,
                 callback: Optional[Callable[[str], Coroutine[None, None,
                                                              None]]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None,
                 init: str = "") -> None:
        super().__init__(uid, UIType.Input, queue, flex, align_self)
        self.label = label
        self.callback = callback
        self.value: str = init

    def to_dict(self):
        res = super().to_dict()
        res["label"] = self.label
        return res

    def get_state(self):
        state = super().get_state()
        state["value"] = self.value
        return state

    def state_change_callback(self, data: str):
        self.value = data

    async def headless_write(self, content: str):
        return await self.queue.put(UIEvent({self.uid: content}))

    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val


# class CodeEditor(Component):

#     def __init__(self,
#                  language: str,
#                  callback: Optional[Callable[[str], Coroutine[None, None,
#                                                               None]]] = None,
#                  uid: str = "",
#                  queue: Optional[asyncio.Queue] = None,
#                  flex: Optional[Union[int, str]] = None,
#                  align_self: Optional[str] = None) -> None:
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


class Switch(Component):
    def __init__(self,
                 label: str,
                 callback: Optional[Callable[[bool], Coroutine[None, None,
                                                               None]]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.Switch, queue, flex, align_self)
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

    def state_change_callback(self, data: bool):
        self.checked = data

    async def headless_write(self, checked: bool):
        return await self.queue.put(UIEvent({self.uid: checked}))

    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val


class Select(Component):
    def __init__(self,
                 label: str,
                 items: List[Tuple[str, Any]],
                 callback: Optional[Callable[[Any], Coroutine[None, None,
                                                              None]]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.Select, queue, flex, align_self)
        self.label = label
        self.callback = callback
        assert len(items) > 0
        self.items = items
        self.value = items[0][1]

    def to_dict(self):
        res = super().to_dict()
        res["label"] = self.label
        return res

    def get_state(self):
        state = super().get_state()
        state["items"] = self.items
        state["value"] = self.value
        return state

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

    def update_value_no_sync(self, value: Any):
        assert value in [x[1] for x in self.items]
        self.value = value

    def state_change_callback(self, value: Any):
        self.value = value

    async def headless_select(self, value: Any):
        return await self.queue.put(UIEvent({self.uid: value}))

    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val


class Slider(Component):
    def __init__(self,
                 label: str,
                 begin: Union[int, float],
                 end: Union[int, float],
                 step: Union[int, float],
                 callback: Optional[Callable[[Union[int, float]],
                                             _CORO_NONE]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.Slider, queue, flex, align_self)
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


class TaskLoop(Component):
    def __init__(self,
                 label: str,
                 loop_callbcak: Callable[[], _CORO_NONE],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 update_period: float = 0.2,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.TaskLoop, queue, flex, align_self)
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

    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val
