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
"""Flow APP: simple GUI application in devflow"""

import asyncio
import base64
import enum
import inspect
import io
import time
import traceback
from typing import (Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
                    Iterable, List, Optional, Tuple, TypeVar, Union)
from tensorpc.apps.flow.coretypes import ScheduleEvent
from tensorpc.core.serviceunit import ReloadableDynamicClass
from tensorpc.utils.uniquename import UniqueNamePool
import numpy as np
from PIL import Image
from tensorpc.core.asynctools import cancel_task
from tensorpc.utils.registry import HashableRegistry

ALL_APP_EVENTS = HashableRegistry()

_CORO_NONE = Union[Coroutine[None, None, None], None]


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
    ListItemButton = 8
    ListItemText = 9

    # outputs
    Image = 10
    Text = 11
    Plotly = 12
    ChartJSLine = 13

    # special
    TaskLoop = 100
    FlexBox = 101
    MUIList = 102
    Divider = 103


class AppEventType(enum.Enum):
    # layout events
    UpdateLayout = 0
    UpdateComponents = 1
    DeleteComponents = 2

    # ui event
    UIEvent = 10
    # clipboard
    CopyToClipboard = 20
    # schedule event, won't be sent to frontend.
    ScheduleNext = 100
    # special UI event
    AppEditor = 200

class UIRunStatus(enum.Enum):
    Stop = 0
    Running = 1
    Pause = 2


class TaskLoopEvent(enum.Enum):
    Start = 0
    Stop = 1
    Pause = 2

class AppEditorEventType(enum.Enum):
    SetValue = 0
    RevealLine = 1

class AppEditorFrontendEventType(enum.Enum):
    Save = 0
    Change = 1
    SaveEditorState = 2

class AppEditorFrontendEvent:

    def __init__(self, type: AppEditorFrontendEventType, data: Any) -> None:
        self.type = type
        self.data = data

    def to_dict(self):
        return {
            "type": self.type.value,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(AppEditorFrontendEventType(data["type"]), data["data"])

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


@ALL_APP_EVENTS.register(key=AppEventType.AppEditor.value)
class AppEditorEvent:

    def __init__(self, type: AppEditorEventType, data) -> None:
        self.data = data
        self.type = type

    def to_dict(self):
        return {
            "type": self.type.value,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(AppEditorEventType(data["type"]), data["data"])

    def merge_new(self, new):
        assert isinstance(new, AppEditorEvent)
        return new

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


@ALL_APP_EVENTS.register(key=AppEventType.UpdateComponents.value)
class UpdateComponentsEvent:

    def __init__(self, data: Dict[str, Any]) -> None:
        self.data = data

    def to_dict(self):
        return self.data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data)

    def merge_new(self, new):
        assert isinstance(new, UpdateComponentsEvent)
        return UpdateComponentsEvent({
            **new.data,
            **self.data,
        })


@ALL_APP_EVENTS.register(key=AppEventType.DeleteComponents.value)
class DeleteComponentsEvent:

    def __init__(self, data: List[str]) -> None:
        self.data = data

    def to_dict(self):
        return self.data

    @classmethod
    def from_dict(cls, data: List[str]):
        return cls(data)

    def merge_new(self, new):
        assert isinstance(new, DeleteComponentsEvent)
        return DeleteComponentsEvent(list(set(self.data + new.data)))


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

@ALL_APP_EVENTS.register(key=AppEventType.ScheduleNext.value)
class ScheduleNextForApp:

    def __init__(self, data) -> None:
        self.data = data

    def to_dict(self):
        return self.data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data)

    def merge_new(self, new):
        assert isinstance(new, ScheduleNextForApp)
        return new

APP_EVENT_TYPES = Union[UIEvent, LayoutEvent, CopyToClipboardEvent,
                        UpdateComponentsEvent, DeleteComponentsEvent,
                        ScheduleNextForApp, AppEditorEvent]


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
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        self.type = type
        self.uid = uid
        self._queue = queue

        self._status = UIRunStatus.Stop
        # task for callback of controls
        # if previous control callback hasn't finished yet,
        # the new control event will be IGNORED
        self._task: Optional[asyncio.Task] = None
        self._flex = flex
        self._align_self = align_self
        self.parent = ""

    def get_callback(self) -> Optional[Callable]:
        return None 

    def set_callback(self, val: Any):
        return  

    def to_dict(self):
        res = {
            "type": self.type.value,
            "uid": self.uid,
            # "parent": self.parent,
            "state": self.get_state(),
        }
        if self._flex is not None:
            res["flex"] = self._flex
        if self._align_self is not None:
            res["alignSelf"] = self._align_self
        return res

    def get_state(self):
        return {
            "status": self._status.value,
        }

    @property
    def queue(self):
        assert self._queue is not None, "you must add ui by flexbox.add_xxx"
        return self._queue

    def state_change_callback(self, data: Any):
        pass

    def create_update_event(self, data: Any):
        ev = UIEvent({self.uid: data})
        # uid is set in flowapp service later.
        return AppEvent("", {AppEventType.UIEvent: ev})

    def create_update_comp_event(self, updates: Dict[str, Any]):
        ev = UpdateComponentsEvent(updates)
        # uid is set in flowapp service later.
        return AppEvent("", {AppEventType.UpdateComponents: ev})

    def create_delete_comp_event(self, deletes: List[str]):
        ev = DeleteComponentsEvent(deletes)
        # uid is set in flowapp service later.
        return AppEvent("", {AppEventType.DeleteComponents: ev})

    async def run_callback(self,
                           cb: Callable[[], _CORO_NONE],
                           sync_state: bool = False):
        self._status = UIRunStatus.Running
        await self.sync_status(sync_state)
        try:
            coro = cb()
            if inspect.iscoroutine(coro):
                await coro
        except:
            traceback.print_exc()
            raise
        finally:
            self._status = UIRunStatus.Stop
            await self.sync_status(sync_state)

    async def sync_status(self, sync_state: bool = False):
        if sync_state:
            await self.queue.put(self.create_update_event(self.get_state()))
        else:
            await self.queue.put(
                self.create_update_event({"status": self._status.value}))
    
    async def sync_state(self):
        return await self.sync_status(True)

    def get_sync_event(self, sync_state: bool = False):
        if sync_state:
            return self.create_update_event(self.get_state())
        else:
            return self.create_update_event({"status": self._status.value})


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
        return await self.queue.put(UIEvent({
            self.uid: self.name
        }))

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
        return await self.queue.put(UIEvent({
            self.uid: self.name
        }))

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
        return await self.queue.put(UIEvent({
            self.uid: self.names[index]
        }))

    def get_callback(self):
        return self.callback 

    def set_callback(self, val: Any):
        self.callback = val  

class FlexBox(Component):

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
                 _init_dict: Optional[Dict[str, Component]] = None,
                 base_type: UIType = UIType.FlexBox,
                 inited: bool = False) -> None:
        super().__init__(uid, base_type, queue, flex, align_self)
        if inited:
            assert queue is not None and uid_to_comp is not None
        if uid_to_comp is None:
            uid_to_comp = {}
        self._pool = UniqueNamePool()
        self._uid_to_comp = uid_to_comp
        self.flex_flow = flex_flow
        self.justify_content = justify_content
        self.align_items = align_items
        self.width = width
        self.height = height
        self.overflow = overflow

        self._init_dict = _init_dict

        self._childs: List[str] = []

        self.inited = inited

    def _attach_to_app(self, queue: asyncio.Queue):
        # update queue of this and childs
        if not self.inited:
            self._queue = queue
            for k, v in self._uid_to_comp.items():
                v._queue = queue
                if isinstance(v, FlexBox):
                    v.inited = True
            self.inited = True

    def _update_child_ns_and_comp_dict(self, ns: str, uid_to_comp: Dict[str, Component]):
        prev_uid_to_comp = self._uid_to_comp
        self._uid_to_comp = uid_to_comp
        for k, v in prev_uid_to_comp.items():
            new_name = f"{ns}.{k}"
            v.uid = new_name
            uid_to_comp[v.uid] = v 
            if isinstance(v, FlexBox):
                v._uid_to_comp = uid_to_comp
        

    def __getitem__(self, key: str):
        assert key in self._childs
        return self._uid_to_comp[self._get_uid_with_ns(key)]

    def _get_all_nested_child_recursive(self, name: str, res: List[Component]):
        comp = self[name]
        res.append(comp)
        if isinstance(comp, FlexBox):
            for child in comp._childs:
                self._get_all_nested_child_recursive(child, res)

    def _get_all_nested_child(self, name: str):
        res: List[Component] = []
        self._get_all_nested_child_recursive(name, res)
        return res

    def _get_all_nested_childs(self):
        comps: List[Component] = []
        for c in self._childs:
            comps.extend(self._get_all_nested_child(c))
        return comps

    def _get_all_child_comp_uids(self):
        return [self[n].uid for n in self._childs]

    def _get_uid_with_ns(self, name: str):
        if not self.inited:
            return (f"{name}")
        return (f"{self.uid}.{name}")

    def _add_prop_to_ui(self, ui: Component):
        if self.inited:
            ui._queue = self.queue
        ui.parent = self.uid
        if isinstance(ui, FlexBox):
            ui._pool = self._pool
            ui._uid_to_comp = self._uid_to_comp

    def add_component(self,
                      name: str,
                      comp: Component,
                      add_to_state: bool = True,
                      anonymous: bool = False):
        uid = self._get_uid_with_ns(name)
        if anonymous:
            uid = self._pool(uid)
        comp.uid = uid
        if add_to_state:
            assert uid not in self._uid_to_comp
        assert comp.parent == "", "this component must not be added before."
        self._add_prop_to_ui(comp)
        if add_to_state:
            self._uid_to_comp[comp.uid] = comp
            self._childs.append(name)
        return comp

    def _extract_layout(self, layout: Dict[str, Component]):
        """ get all components without add to app state.
        """
        # we assume layout is a ordered dict (python >= 3.7)
        comps: List[Component] = []
        for k, v in layout.items():
            comps.append(self.add_component(k, v, False, anonymous=False))
            if isinstance(v, FlexBox):
                msg = "you must use VBox/HBox/Box instead in dict layout"
                assert v._init_dict is not None, msg
                comps.extend(v._extract_layout(v._init_dict))
        return comps

    def add_layout(self, layout: Dict[str, Component]):
        """ {
            btn0: Button(...),
            box0: VBox({
                btn1: Button(...),
                ...
            }, flex...),
        }
        """
        # we assume layout is a ordered dict (python >= 3.7)
        for k, v in layout.items():
            if not isinstance(v, FlexBox):
                self.add_component(k, v, anonymous=False)
            else:
                if v._init_dict is not None:
                    # consume this _init_dict
                    v.add_layout(v._init_dict)
                    v._init_dict = None
                # update uids of childs of v, 
                ns = f"{k}"
                if self.uid != "":
                    ns = f"{self.uid}.{k}"
                v._update_child_ns_and_comp_dict(ns, self._uid_to_comp)
                if not v.inited and self.inited:
                    v._attach_to_app(self._queue)
                self.add_component(k, v, anonymous=False)

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
        if self.overflow is not None:
            res["overflow"] = self.overflow
        return res

    def get_state(self):
        state = super().get_state()
        state["childs"] = self._get_all_child_comp_uids()
        return state

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

    async def set_new_layout(self, layout: Dict[str, Component]):
        # remove all first
        comps_to_remove: List[Component] = []
        for c in self._childs:
            comps_to_remove.extend(self._get_all_nested_child(c))
        for c in comps_to_remove:
            self._uid_to_comp.pop(c.uid)
        self._childs.clear()
        # TODO should we merge two events to one?
        await self.queue.put(
            self.create_delete_comp_event([c.uid for c in comps_to_remove]))
        self.add_layout(layout)
        comps_frontend = {
            c.uid: c.to_dict()
            for c in self._get_all_nested_childs()
        }
        # make sure all child of this box is rerendered.
        comps_frontend[self.uid] = self.to_dict()
        await self.queue.put(self.create_update_comp_event(comps_frontend))

    async def remove_childs_by_keys(self, keys: List[str]):
        comps_to_remove: List[Component] = []
        for c in keys:
            comps_to_remove.extend(self._get_all_nested_child(c))
        for c in comps_to_remove:
            self._uid_to_comp.pop(c.uid)
        for k in keys:
            self._childs.remove(k)
        # make sure all child of this box is rerendered.
        await self.queue.put(
            self.create_delete_comp_event([c.uid for c in comps_to_remove]))
        # TODO combine two event to one
        await self.queue.put(
            self.create_update_comp_event({self.uid: self.to_dict()}))

    async def update_childs(self, layout: Dict[str, Component]):
        new_comps = self._extract_layout(layout)
        update_comps_frontend = {c.uid: c.to_dict() for c in new_comps}
        for c in new_comps:
            if c.uid in self._uid_to_comp:
                item = self._uid_to_comp.pop(c.uid)
                item.parent = ""  # mark invalid
            self._uid_to_comp[c.uid] = c
        for n in layout.keys():
            if n not in self._childs:
                self._childs.append(n)
        # make sure all child of this box is rerendered.
        update_comps_frontend[self.uid] = self.to_dict()
        await self.queue.put(
            self.create_update_comp_event(update_comps_frontend))

    def add_flex_box(self,
                     flex_flow: Optional[str] = None,
                     justify_content: Optional[str] = None,
                     align_items: Optional[str] = None,
                     flex: Optional[Union[int, str]] = None,
                     align_self: Optional[str] = None,
                     width: Optional[Union[str, int]] = None,
                     height: Optional[Union[str, int]] = None,
                     overflow: Optional[str] = None):
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
         overflow: Optional[str] = None):
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
                   _init_dict=layout)


def HBox(layout: Dict[str, Component],
         justify_content: Optional[str] = None,
         align_items: Optional[str] = None,
         flex: Optional[Union[int, str]] = None,
         align_self: Optional[str] = None,
         width: Optional[Union[str, int]] = None,
         height: Optional[Union[str, int]] = None,
         overflow: Optional[str] = None):
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
                   _init_dict=layout)


def Box(layout: Dict[str, Component],
        flex_flow: Optional[str] = None,
        justify_content: Optional[str] = None,
        align_items: Optional[str] = None,
        flex: Optional[Union[int, str]] = None,
        align_self: Optional[str] = None,
        width: Optional[Union[str, int]] = None,
        height: Optional[Union[str, int]] = None,
        overflow: Optional[str] = None):
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
                   _init_dict=layout)


def VList(layout: Dict[str, Component],
          subheader: str = "",
          flex: Optional[Union[int, str]] = None,
          align_self: Optional[str] = None,
          width: Optional[Union[str, int]] = None,
          height: Optional[Union[str, int]] = None,
          overflow: Optional[str] = None):
    return MUIList("",
                   asyncio.Queue(), {},
                   flex=flex,
                   align_self=align_self,
                   overflow=overflow,
                   subheader=subheader,
                   width=width,
                   height=height,
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
        return await self.queue.put(UIEvent({
            self.uid: self.names[index]
        }))

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
        return await self.queue.put(UIEvent({
            self.uid: content
        }))

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
        return await self.queue.put(UIEvent({
            self.uid: checked
        }))

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
        return await self.queue.put(UIEvent({
            self.uid: value
        }))

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
        return await self.queue.put(UIEvent({
            self.uid: value
        }))

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
        return await self.queue.put(UIEvent({
            self.uid: TaskLoopEvent.Start.value
        }))

    def get_callback(self):
        return self.callback 

    def set_callback(self, val: Any):
        self.callback = val  

_ROOT = "root"

class AppEditor:

    def __init__(self,
                init_value: str,
                 language: str,
                 ) -> None:
        self._language = language
        self._value: str = init_value
        self.__freeze_language = False
        self._init_line_number = 1
        self._monaco_state: Optional[Any] = None

    def set_init_line_number(self, val: int):
        self._init_line_number = val

    def freeze(self):
        self.__freeze_language = True

    @property 
    def value(self):
        return self._value

    @value.setter 
    def value(self, val: str):
        self._value = val 

    @property 
    def language(self):
        return self._language

    @language.setter 
    def language(self, val: str):
        if not self.__freeze_language:
            self._language = val 
        else:
            raise ValueError("language freezed, you can't change it.")

    def get_state(self):
        state = {}
        state["language"] = self._language
        state["value"] = self._value
        state["monacoEditorState"] = self._monaco_state
        state["initLineNumber"] = self._init_line_number
        return state


class App:
    # TODO find a way to sync state to frontend uncontrolled elements.
    def __init__(self,
                 flex_flow: Optional[str] = "column nowrap",
                 justify_content: Optional[str] = None,
                 align_items: Optional[str] = None,
                 maxqsize: int = 10) -> None:
        self._uid_to_comp: Dict[str, Component] = {}
        self._queue = asyncio.Queue(maxsize=maxqsize)

        self._send_callback: Optional[Callable[[AppEvent],
                                               Coroutine[None, None,
                                                         None]]] = None
        root = FlexBox(_ROOT,
                       self._queue,
                       self._uid_to_comp,
                       flex_flow,
                       justify_content,
                       align_items,
                       inited=True)
        self._uid_to_comp[_ROOT] = root

        self.root = root
        self._enable_editor = False

        self.code_editor = AppEditor("", "python")
        self._app_dynamic_cls: Optional[ReloadableDynamicClass] = None
        # loaded if you connect app node with a full data storage
        self._data_storage: Dict[str, Any] = {}

    def _get_app_dynamic_cls(self):
        assert self._app_dynamic_cls is not None 
        return self._app_dynamic_cls

    def _get_app_layout(self):
        return {
            "layout": {u: c.to_dict()
                       for u, c in self._uid_to_comp.items()},
            "enableEditor": self._enable_editor,
            "codeEditor": self.code_editor.get_state(),
        }

    def init_enable_editor(self):
        self._enable_editor = True

    def set_init_window_size(self, size: List[Union[int, None]]):
        self.root.width = size[0]
        self.root.height = size[1]

    async def headless_main(self):
        """override this method to support headless mode.
        you can use headless methods for control UIs such as 
        btn.headless_click and inp.headless_write to trigger
        callbacks.
        """
        raise NotImplementedError("headless_main not exists. "
            "override headless_main to run in headless mode.")

    async def flow_run(self, event: ScheduleEvent):
        """override this method to support flow. output data will be 
        sent to all child nodes if not None.
        """
        return None

    async def _handle_code_editor_event_system(self, event: AppEditorFrontendEvent):
        if event.type == AppEditorFrontendEventType.SaveEditorState:
            self.code_editor._monaco_state = event.data
            return 
        return await self.handle_code_editor_event(event)

    async def handle_code_editor_event(self, event: AppEditorFrontendEvent):
        """override this method to support vscode editor.
        """
        return

    async def _send_editor_event(self, event: AppEditorEvent):
        await self._queue.put(AppEvent("", {AppEventType.AppEditor: event}))

    async def set_editor_value(self, value: str, language: str):
        """use this method to set editor value and language.
        """
        self.code_editor.value = value 
        self.code_editor.language = language 
        app_ev = AppEditorEvent(AppEditorEventType.SetValue, {
            "value": value,
            "language": language,
        })
        await self._send_editor_event(app_ev)

    async def _handle_control_event(self, ev: UIEvent):
        # TODO run control fron other component
        for uid, data in ev.uid_to_data.items():
            comp = self._uid_to_comp[uid]
            # sync state after every callback
            if isinstance(comp, (Switch, Select, Slider, RadioGroup)):
                if comp._status == UIRunStatus.Running:
                    # TODO send exception if ignored click
                    print("IGNORE EVENT", comp._status)
                    return
                elif comp._status == UIRunStatus.Stop:
                    cb1 = comp.callback
                    comp.state_change_callback(data)
                    if cb1 is not None:

                        def ccb(cb):
                            return lambda: cb(data)

                        comp._task = asyncio.create_task(
                            comp.run_callback(ccb(cb1), True))
                    else:
                        await comp.sync_status(True)
            # no sync state
            elif isinstance(comp, (Input, )):
                if comp._status == UIRunStatus.Running:
                    # TODO send exception if ignored click
                    print("IGNORE EVENT", comp._status)
                    return
                elif comp._status == UIRunStatus.Stop:
                    cb = comp.callback
                    comp.state_change_callback(data)
                    # we can't update input state
                    # because input is an uncontrolled
                    # component.
                    if cb is not None:

                        def ccb(cb):
                            return lambda: cb(data)

                        comp._task = asyncio.create_task(
                            comp.run_callback(ccb(cb)))
                    # else:
                    #     await comp.sync_status(True)

            elif isinstance(comp, (Button, ListItemButton)):
                if comp._status == UIRunStatus.Running:
                    # TODO send exception if ignored click
                    print("IGNORE EVENT", comp._status)
                    return
                elif comp._status == UIRunStatus.Stop:
                    cb2 = comp.callback
                    comp._task = asyncio.create_task(
                        comp.run_callback(lambda: cb2()))
            elif isinstance(comp, (Buttons)):
                if comp._status == UIRunStatus.Running:
                    # TODO send exception if ignored click
                    print("IGNORE EVENT", comp._status)
                    return
                elif comp._status == UIRunStatus.Stop:
                    cb3 = comp.callback
                    comp._task = asyncio.create_task(
                        comp.run_callback(lambda: cb3(data)))
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

class EditableApp(App):
    def __init__(self, flex_flow: Optional[str] = "column nowrap", justify_content: Optional[str] = None, align_items: Optional[str] = None, maxqsize: int = 10) -> None:
        super().__init__(flex_flow, justify_content, align_items, maxqsize)
        lines, lineno = inspect.findsource(type(self))
        self.code_editor.value = "".join(lines) 
        self.code_editor.language = "python"
        self.code_editor.set_init_line_number(lineno)
        self.code_editor.freeze()

    async def handle_code_editor_event(self, event: AppEditorFrontendEvent):
        """override this method to support vscode editor.
        """
        # TODO if other editor change app code?
        if event.type == AppEditorFrontendEventType.Save:
            with open(inspect.getfile(type(self)), "w") as f:
                f.write(event.data)
            self.code_editor.value = event.data
            comps = self._uid_to_comp
            callback_dict = {}
            for k, v in comps.items():
                cb = v.get_callback()
                if cb is not None:
                    callback_dict[k] = cb
            new_cb = self._get_app_dynamic_cls().reload_obj_methods(self, callback_dict)
            for k, v in comps.items():
                if k in new_cb:
                    v.set_callback(new_cb[k])
        return
