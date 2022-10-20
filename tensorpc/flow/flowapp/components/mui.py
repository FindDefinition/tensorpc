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
import enum
from typing import (Any, AsyncGenerator, AsyncIterable, Awaitable, Callable, Coroutine, Dict,
                    Iterable, List, Optional, Tuple, Type, TypeVar, Union,
                    TYPE_CHECKING)
from typing_extensions import Literal, TypeAlias
import numpy as np
from PIL import Image as PILImage
import json
import inspect
from tensorpc.flow.flowapp.components.common import handle_change_event, handle_change_event_no_arg
from tensorpc.core.asynctools import cancel_task
from ..core import (AppEvent, AppEventType, BasicProps, Component,
                    ContainerBase, NumberType, T_child, TaskLoopEvent, UIEvent,
                    UIRunStatus, UIType, Undefined, undefined, T_base_props,
                    T_container_props, ContainerBaseProps,
                    ValueType, Fragment, EventHandler, EventType)

from .. import colors

if TYPE_CHECKING:
    from .three import ThreeCanvas

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

_OverflowType = Union[Literal["visible"], Literal["hidden"], Literal["scroll"], Literal["auto"]]
PointerEventsProperties: TypeAlias = Union[Literal["auto"], Literal["none"],
                                Literal["visiblePainted"],
                                Literal["visibleFill"],
                                Literal["visibleStroke"], Literal["visible"],
                                Literal["painted"], Literal["fill"],
                                Literal["stroke"], Literal["all"],
                                Literal["inherit"]]

@dataclasses.dataclass
class FlexComponentBaseProps(BasicProps):
    """all props must have a default value, 
    manage state by your self.
    """
    position: Union[Literal["absolute", "relative"], Undefined] = undefined
    top: Union[ValueType, Undefined] = undefined
    bottom: Union[ValueType, Undefined] = undefined
    left: Union[ValueType, Undefined] = undefined
    right: Union[ValueType, Undefined] = undefined
    z_index: Union[ValueType, Undefined] = undefined

    flex: Union[ValueType, Undefined] = undefined
    align_self: Union[str, Undefined] = undefined
    flex_grow: Union[str, Undefined] = undefined
    flex_shrink: Union[str, Undefined] = undefined
    flex_basis: Union[str, Undefined] = undefined

    height: Union[ValueType, Undefined] = undefined
    width: Union[ValueType, Undefined] = undefined
    max_height: Union[ValueType, Undefined] = undefined
    max_width: Union[ValueType, Undefined] = undefined
    min_height: Union[ValueType, Undefined] = undefined
    min_width: Union[ValueType, Undefined] = undefined
    padding: Union[ValueType, Undefined] = undefined
    padding_top: Union[ValueType, Undefined] = undefined
    padding_bottom: Union[ValueType, Undefined] = undefined
    padding_left: Union[ValueType, Undefined] = undefined
    padding_right: Union[ValueType, Undefined] = undefined
    margin: Union[ValueType, Undefined] = undefined
    margin_top: Union[ValueType, Undefined] = undefined
    margin_left: Union[ValueType, Undefined] = undefined
    margin_right: Union[ValueType, Undefined] = undefined
    margin_bottom: Union[ValueType, Undefined] = undefined

    overflow: Union[_OverflowType, Undefined] = undefined
    overflow_y: Union[_OverflowType, Undefined] = undefined
    overflow_x: Union[_OverflowType, Undefined] = undefined 

    color: Union[ValueType, Undefined] = undefined
    background_color: Union[ValueType, Undefined] = undefined
    font_size: Union[ValueType, Undefined] = undefined
    font_family: Union[str, Undefined] = undefined
    border: Union[str, Undefined] = undefined
    border_top: Union[ValueType, Undefined] = undefined
    border_left: Union[ValueType, Undefined] = undefined
    border_right: Union[ValueType, Undefined] = undefined
    border_bottom: Union[ValueType, Undefined] = undefined
    border_color: Union[str, Undefined] = undefined
    white_space: Union[Literal["normal", "pre", "nowrap", "pre-wrap", "pre-line", "break-spaces"], Undefined] = undefined
    pointer_events: Union[PointerEventsProperties, Undefined] = undefined

class IconType(enum.Enum):
    """all icons used in devflow.
    """
    Start = 0
    Stop = 1
    Pause = 2 
    Close = 3 
    Code = 4
    FullScreen = 5
    Trash = 6 
    More = 7 
    Setting = 8


@dataclasses.dataclass
class MUIComponentBaseProps(FlexComponentBaseProps):
    pass


class MUIComponentBase(Component[T_base_props, "MUIComponentType"]):
    pass


class MUIContainerBase(ContainerBase[T_container_props, T_child]):
    pass


@dataclasses.dataclass
class FlexBoxProps(FlexComponentBaseProps):
    # TODO add literal here.
    align_content: Union[str, Undefined] = undefined
    align_items: Union[str, Undefined] = undefined
    justify_content: Union[str, Undefined] = undefined
    flex_direction: Union[str, Undefined] = undefined
    flex_wrap: Union[str, Undefined] = undefined
    flex_flow: Union[str, Undefined] = undefined

# we can't let mui use three component.
@dataclasses.dataclass
class MUIFlexBoxProps(FlexBoxProps, ContainerBaseProps):
    pass

_TypographyVarient: TypeAlias = Literal['body1', 'body2', 'button', 'caption',
                                        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                                        'inherit', 'overline', 'subtitle1',
                                        'subtitle2']

_StdColor: TypeAlias = Literal['default', 'primary', 'secondary', 'error',
                               'info', 'success', 'warning']

_StdColorNoDefault: TypeAlias = Literal['primary', 'secondary', 'error',
                               'info', 'success', 'warning']

async def _handle_standard_event(comp: Component, data: Any):
    if comp.props.status == UIRunStatus.Running.value:
        # TODO send exception if ignored click
        print("IGNORE EVENT", comp.props.status)
        return
    elif comp.props.status == UIRunStatus.Stop.value:
        cb1 = comp.get_callback()
        comp.state_change_callback(data)
        if cb1 is not None:
            def ccb(cb):
                return lambda: cb(data)

            comp._task = asyncio.create_task(comp.run_callback(ccb(cb1), True))
        else:
            await comp.sync_status(True)


async def _handle_button_event(comp: Union["Button", "ListItemButton"],
                               data: Any):
    if comp.props.status == UIRunStatus.Running.value:
        # TODO send exception if ignored click
        print("IGNORE EVENT", comp.props.status)
        return
    elif comp.props.status == UIRunStatus.Stop.value:
        cb2 = comp.callback
        comp._task = asyncio.create_task(comp.run_callback(lambda: cb2()))


MUIComponentType: TypeAlias = Union[MUIComponentBase, MUIContainerBase, Fragment]

@dataclasses.dataclass
class ImageProps(MUIComponentBaseProps):
    image: bytes = dataclasses.field(default_factory=bytes)

class Images(MUIComponentBase[ImageProps]):

    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.Image, ImageProps, queue)
        # self.image_str: bytes = b""

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["image"] = self.props.image
        return res

    async def show(self, image: np.ndarray):
        encoded = _encode_image_bytes(image)
        self.props.image = encoded
        # self.image_str = encoded
        await self.put_app_event(self.update_event(image=encoded))

    async def show_raw(self, image_bytes: bytes, suffix: str):
        await self.put_app_event(self.show_raw_event(image_bytes, suffix))

    def show_raw_event(self, image_bytes: bytes, suffix: str):
        raw = b'data:image/' + suffix.encode(
            "utf-8") + b';base64,' + base64.b64encode(image_bytes)
        # self.image_str = raw
        self.props.image = raw
        return self.update_event(image=raw)

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

@dataclasses.dataclass
class PlotlyProps(BasicProps):
    data: list = dataclasses.field(default_factory=list) 
    layout: dict = dataclasses.field(default_factory=dict) 

# @dataclasses.dataclass
# class PlotlyLayoutTitle:



class Plotly(MUIComponentBase[PlotlyProps]):
    """see https://plotly.com/javascript/ for documentation"""
    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.Plotly, PlotlyProps, queue)

    async def show_raw(self, data: list, layout: Any):
        self.props.data = data
        self.props.layout = layout
        await self.put_app_event(self.update_event(data=data, layout=layout))

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["data"] = self.props.data
        res["layout"] = self.props.layout
        return res

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    @staticmethod
    def layout_no_margin():
        return {
            # "height": 240,
            "autosize": 'true',
            "margin": {
                "l": 0,
                "r": 0,
                "b": 0,
                "t": 0,
                #   "pad": 0
            },
            "yaxis": {
                "automargin": True,
            },
            "xaxis": {
                "automargin": True,
            },
        }



@dataclasses.dataclass
class TextProps(MUIComponentBaseProps):
    value: str = "" 


class Text(MUIComponentBase[TextProps]):

    def __init__(self,
                 init: str,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.Text, TextProps, queue)
        self.props.value = init

    async def write(self, content: str):
        self.props.value = content
        await self.put_app_event(self.update_event(value=content))

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["value"] = self.props.value
        return res

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    @property 
    def value(self):
        return self.props.value


class ListItemText(MUIComponentBase[TextProps]):

    def __init__(self,
                 init: str,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ListItemText, TextProps,
                         queue)
        self.props.value = init

    async def write(self, content: str):
        self.props.value = content
        await self.put_app_event(self.update_event(value=content))

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["value"] = self.props.value
        return res

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    @property 
    def value(self):
        return self.props.value

_SEVERITY_TYPES: TypeAlias = Literal["error", "warning", "success", "info"]

@dataclasses.dataclass
class AlertProps(MUIComponentBaseProps):
    value: str = "" 
    severity: _SEVERITY_TYPES = "info"
    title: Union[str, Undefined] = undefined
    mui_color: Union[_SEVERITY_TYPES, Undefined] = undefined 
    variant: Union[Literal["filled", "outlined", "standard"], Undefined] = undefined

class Alert(MUIComponentBase[AlertProps]):

    def __init__(self,
                 value: str,
                 severity: _SEVERITY_TYPES, 
                 title: str = "",
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.Alert, AlertProps, queue)
        self.props.value = value
        self.props.severity = severity
        self.props.title = title

    async def write(self, content: str):
        self.props.value = content
        await self.put_app_event(self.update_event(value=content))

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["value"] = self.props.value
        return res

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    @property 
    def value(self):
        return self.props.value


@dataclasses.dataclass
class DividerProps(MUIComponentBaseProps):
    orientation: Union[Literal["horizontal", "vertical"], Undefined] = undefined

class Divider(MUIComponentBase[DividerProps]):

    def __init__(self,
                 orientation: Union[Literal["horizontal"],
                                    Literal["vertical"]] = "horizontal",
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.Divider, DividerProps, queue)
        self.props.orientation = orientation
        assert orientation == "horizontal" or orientation == "vertical"

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

class HDivider(Divider):
    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__("horizontal", uid, queue)

class VDivider(Divider):
    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__("vertical", uid, queue)

_BtnGroupColor: TypeAlias = Literal['inherit', 'primary', 'secondary', 'error',
                                    'info', 'success', 'warning']

@dataclasses.dataclass
class ButtonProps(MUIComponentBaseProps):
    name: str = ""
    mui_color: Union[_BtnGroupColor, Undefined] = undefined
    disabled: Union[bool, Undefined] = undefined
    full_width: Union[bool, Undefined] = undefined
    size: Union[Literal["small", "medium", "large"], Undefined] = undefined
    variant: Union[Literal["contained", "outlined", "text"],
                   Undefined] = undefined
    loading: Union[Undefined, bool] = undefined 
    loading_indicator: Union[Undefined, str] = undefined

class Button(MUIComponentBase[ButtonProps]):

    def __init__(self,
                 name: str,
                 callback: Callable[[], _CORO_NONE],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.Button, ButtonProps, queue)
        self.props.name = name
        self.callback = callback

    async def headless_click(self):
        return await self.put_app_event(AppEvent("", {AppEventType.UIEvent: UIEvent({self._flow_uid: self.props.name})}))

    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val

    async def handle_event(self, ev: EventType):
        await handle_change_event_no_arg(self, sync_first=True)

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class ButtonGroupProps(MUIFlexBoxProps):
    orientation: Union[Literal["horizontal", "vertical"],
                       Undefined] = undefined
    mui_color: Union[_BtnGroupColor, Undefined] = undefined
    disabled: Union[bool, Undefined] = undefined
    full_width: Union[bool, Undefined] = undefined
    size: Union[Literal["small", "medium", "large"], Undefined] = undefined
    variant: Union[Literal["contained", "outlined", "text"],
                   Undefined] = undefined

class ButtonGroup(MUIContainerBase[ButtonGroupProps, Button]):

    def __init__(self,
                 children: Union[List[Button], Dict[str, Button]],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.ButtonGroup, ButtonGroupProps, uid, queue,
                         uid_to_comp, children, inited)
        for v in children.values():
            assert isinstance(v, Button), "all childs must be button"

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)


@dataclasses.dataclass
class AccordionDetailsProps(MUIFlexBoxProps):
    pass

@dataclasses.dataclass
class AccordionSummaryProps(MUIFlexBoxProps):
    pass

class AccordionDetails(MUIContainerBase[AccordionDetailsProps, MUIComponentType]):

    def __init__(self,
                 children: Dict[str, MUIComponentType],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.AccordionDetail, AccordionDetailsProps, uid, queue,
                         uid_to_comp, children, inited)

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

class AccordionSummary(MUIContainerBase[AccordionSummaryProps, MUIComponentType]):

    def __init__(self,
                 children: Dict[str, MUIComponentType],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.AccordionSummary, AccordionSummaryProps, uid, queue,
                         uid_to_comp, children, inited)

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

@dataclasses.dataclass
class AccordionProps(MUIFlexBoxProps):
    disabled: Union[Undefined, bool] = undefined
    expanded: bool = False 
    square: Union[Undefined, bool] = undefined
    disable_gutters: Union[Undefined, bool] = undefined

class Accordion(MUIContainerBase[AccordionProps, Union[AccordionDetails, AccordionSummary]]):

    def __init__(self,
                 summary: AccordionSummary,
                 details: Optional[AccordionDetails] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        children: Dict[str, Union[AccordionDetails, AccordionSummary]] = {
            "summary": summary
        }
        if details is not None:
            children["details"] = details
        for v in children.values():
            assert isinstance(v, (AccordionSummary, AccordionDetails)), "all childs must be summary or detail"
        super().__init__(UIType.Accordion, AccordionProps, uid, queue,
                         uid_to_comp, children, inited)

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["expanded"] = self.props.expanded
        return res

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)
        
    def state_change_callback(self, data: bool):
        self.props.expanded = data
        
    async def handle_event(self, ev: EventType):
        await handle_change_event(self, ev)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

class ListItemButton(MUIComponentBase[ButtonProps]):

    def __init__(self,
                 name: str,
                 callback: Callable[[], _CORO_NONE],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ListItemButton, ButtonProps,
                         queue)
        self.props.name = name
        self.callback = callback

    async def headless_click(self):
        uiev = UIEvent({self._flow_uid: self.props.name})
        return await self.put_app_event(AppEvent("", {AppEventType.UIEvent: uiev}))

    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val

    async def handle_event(self, ev: EventType):
        await handle_change_event_no_arg(self, sync_first=False)

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

class FlexBox(MUIContainerBase[MUIFlexBoxProps, MUIComponentType]):

    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 _children: Optional[Union[List[MUIComponentType], Dict[str, MUIComponentType]]] = None,
                 base_type: UIType = UIType.FlexBox,
                 inited: bool = False) -> None:
        if _children is not None and isinstance(_children, list):
            _children = {str(i): v for i, v in enumerate(_children)}
        super().__init__(base_type, MUIFlexBoxProps, uid, queue, uid_to_comp,
                         _children, inited)

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

@dataclasses.dataclass
class MUIListProps(MUIFlexBoxProps):
    subheader: str = ""

class MUIList(MUIContainerBase[MUIListProps, MUIComponentType]):

    def __init__(self,
                 uid: str,
                 queue: asyncio.Queue,
                 uid_to_comp: Dict[str, Component],
                 _children: Optional[Dict[str, MUIComponentType]] = None,
                 subheader: str = "",
                 inited: bool = False) -> None:
        super().__init__(UIType.MUIList,
                         MUIListProps,
                         uid,
                         queue=queue,
                         uid_to_comp=uid_to_comp,
                         _children=_children,
                         inited=inited)
        self.props.subheader = subheader

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

def VBox(layout: Union[List[MUIComponentType], Dict[str, MUIComponentType]]):
    res = FlexBox("", asyncio.Queue(), {}, _children=layout)
    res.prop(flex_flow="column")
    return res


def HBox(layout: Union[List[MUIComponentType], Dict[str, MUIComponentType]]):
    res = FlexBox("", asyncio.Queue(), {}, _children=layout)
    res.prop(flex_flow="row")
    return res


def Box(layout: Union[List[MUIComponentType], Dict[str, MUIComponentType]]):
    return FlexBox("", asyncio.Queue(), {}, _children=layout)


def VList(layout: Dict[str, MUIComponentType], subheader: str = ""):
    return MUIList("",
                   asyncio.Queue(), {},
                   subheader=subheader,
                   _children=layout)

@dataclasses.dataclass
class RadioGroupProps(MUIComponentBaseProps):
    names: List[str] = dataclasses.field(default_factory=list)
    row: Union[Undefined, bool] = undefined
    value: str = ""

class RadioGroup(MUIComponentBase[RadioGroupProps]):

    def __init__(self,
                 names: List[str],
                 row: bool,
                 callback: Optional[Callable[[str], Coroutine[None, None,
                                                              None]]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.RadioGroup, RadioGroupProps, queue)
        self.props.names = names
        self.callback = callback
        self.props.row = row
        self.props.value = names[0]

    def state_change_callback(self, data: str):
        self.props.value = data

    def validate_props(self, props: Dict[str, Any]):
        if "names" in props:
            return props["names"] == self.props.names
        return False

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["value"] = self.props.value
        return res

    async def update_value(self, value: Any):
        assert value in self.props.names
        await self.put_app_event(self.create_update_event({"value": value}))
        self.props.value = value

    async def headless_click(self, index: int):
        uiev = UIEvent({self._flow_uid: self.props.names[index]})
        return await self.put_app_event(AppEvent("", {AppEventType.UIEvent: uiev}))

    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val

    async def handle_event(self, ev: EventType):
        await handle_change_event(self, ev)

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    @property 
    def value(self):
        return self.props.value

_HTMLInputType: TypeAlias = Literal["button", "checkbox", "color", "date", 
    "datetime-local", "email", "file", "hidden", "image", "month", "number",
    "password", "radio", "range", 'reset', "search", "submit", "tel", "text", 
    "time", "url", "week"]

@dataclasses.dataclass
class InputProps(MUIComponentBaseProps):
    label: str = ""
    multiline: bool = False 
    value: str = ""
    mui_color: Union[_StdColorNoDefault, Undefined] = undefined
    disabled: Union[bool, Undefined] = undefined 
    error: Union[bool, Undefined] = undefined 
    full_width: Union[bool, Undefined] = undefined 
    rows: Union[NumberType, str, Undefined] = undefined 
    size: Union[Undefined, Literal["small", "medium"]] = undefined
    mui_margin: Union[Undefined, Literal["dense", "none", "normal"]] = undefined
    variant: Union[Undefined, Literal["filled", "outlined", "standard"]] = undefined
    type: Union[Undefined, _HTMLInputType] = undefined

class Input(MUIComponentBase[InputProps]):

    def __init__(self,
                 label: str,
                 multiline: bool = False,
                 callback: Optional[Callable[[str], _CORO_NONE]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 init: str = "") -> None:
        super().__init__(uid, UIType.Input, InputProps, queue)
        self.props.label = label
        self.callback = callback
        self.props.value = init
        self.props.multiline = multiline

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["value"] = self.props.value
        return res

    @property 
    def value(self):
        return self.props.value

    def state_change_callback(self, data: str):
        self.props.value = data

    async def headless_write(self, content: str):
        uiev = UIEvent({self._flow_uid: content})
        return await self.put_app_event(AppEvent("", {AppEventType.UIEvent: uiev}))

    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val

    def json(self):
        return json.loads(self.props.value)

    def float(self):
        return float(self.props.value)

    def int(self):
        return int(self.props.value)

    async def handle_event(self, ev: EventType):

        if self.props.status == UIRunStatus.Running.value:
            # TODO send exception if ignored click
            print("IGNORE EVENT", self.props.status)
            return
        elif self.props.status == UIRunStatus.Stop.value:
            cb = self.callback
            self.state_change_callback(ev[1])
            # we can't update input state
            # because input is an uncontrolled
            # component.
            if cb is not None:
                def ccb(cb):
                    return lambda: cb(ev[1])
                self._task = asyncio.create_task(self.run_callback(ccb(cb)))

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

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

@dataclasses.dataclass
class SwitchProps(MUIComponentBaseProps):
    label: str = ""
    checked: bool = False 
    size: Union[Literal["small", "medium"], Undefined] = undefined
    mui_color: Union[_BtnGroupColor, Undefined] = undefined
    label_placement: Union[Literal["top", "start", "bottom", "end"], Undefined] = undefined

class SwitchBase(MUIComponentBase[SwitchProps]):

    def __init__(self,
                 label: str,
                 base_type: UIType,
                 callback: Optional[Callable[[bool], _CORO_NONE]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, base_type, SwitchProps, queue)
        self.props.label = label
        self.callback = callback
        self.props.checked = False

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["checked"] = self.props.checked
        return res

    @property 
    def checked(self):
        return self.props.checked

    def state_change_callback(self, data: bool):
        self.props.checked = data

    async def headless_write(self, checked: bool):
        uiev = UIEvent({self._flow_uid: checked})
        return await self.put_app_event(AppEvent("", {AppEventType.UIEvent: uiev}))

    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val

    def __bool__(self):
        return self.props.checked

    async def handle_event(self, ev: EventType):
        await handle_change_event(self, ev)

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

class Switch(SwitchBase):

    def __init__(self,
                 label: str,
                 callback: Optional[Callable[[bool], _CORO_NONE]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(label, UIType.Switch, callback, uid, queue)

class Checkbox(SwitchBase):

    def __init__(self,
                 label: str,
                 callback: Optional[Callable[[bool], _CORO_NONE]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(label, UIType.Checkbox, callback, uid, queue)

@dataclasses.dataclass
class SelectProps(MUIComponentBaseProps):
    label: str = ""
    items: List[Tuple[str, ValueType]] = dataclasses.field(default_factory=list) 
    value: ValueType = ""
    size: Union[Undefined, Literal["small", "medium"]] = undefined
    mui_margin: Union[Undefined, Literal["dense", "none", "normal"]] = undefined
    variant: Union[Undefined, Literal["filled", "outlined", "standard"]] = undefined

class Select(MUIComponentBase[SelectProps]):

    def __init__(self,
                 label: str,
                 items: List[Tuple[str, ValueType]],
                 callback: Optional[Callable[[ValueType],
                                             _CORO_NONE]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.Select, SelectProps, queue)
        self.props.label = label
        self.callback = callback
        # assert len(items) > 0
        self.props.items = items
        # item value must implement eq/ne
        self.props.value = ""
        self.props.size = "small"

    @property 
    def value(self):
        return self.props.value

    def validate_props(self, props: Dict[str, Any]):
        if "items" in props:
            if len(self.props.items) == 0:
                # if user init a empty select, use previous state
                return True
        if "value" in props:
            value = props["value"]
            return value in [x[1] for x in self.props.items]
        return False

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["value"] = self.props.value
        res["items"] = self.props.items
        return res

    async def update_items(self, items: List[Tuple[str, ValueType]],
                           selected: int):
        await self.put_app_event(
            self.create_update_event({
                "items": items,
                "value": items[selected][1]
            }))
        self.props.items = items
        self.props.value = items[selected][1]

    async def update_value(self, value: ValueType):
        assert value in [x[1] for x in self.props.items]
        await self.put_app_event(self.create_update_event({"value": value}))
        self.props.value = value

    def update_value_no_sync(self, value: ValueType):
        assert value in [x[1] for x in self.props.items]
        self.props.value = value

    def state_change_callback(self, value: ValueType):
        self.props.value = value

    async def headless_select(self, value: ValueType):
        uiev = UIEvent({self._flow_uid: value})
        return await self.put_app_event(AppEvent("", {AppEventType.UIEvent: uiev}))

    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val

    async def handle_event(self, ev: EventType):
        await handle_change_event(self, ev)

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

@dataclasses.dataclass
class MultipleSelectProps(MUIComponentBaseProps):
    label: str = ""
    items: List[Tuple[str, ValueType]] = dataclasses.field(default_factory=list) 
    values: List[ValueType] = dataclasses.field(default_factory=list) 

class MultipleSelect(MUIComponentBase[MultipleSelectProps]):

    def __init__(self,
                 label: str,
                 items: List[Tuple[str, ValueType]],
                 callback: Optional[Callable[[List[ValueType]], _CORO_NONE]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.MultipleSelect, MultipleSelectProps,
                         queue)
        self.props.label = label
        self.callback = callback
        assert len(items) > 0
        self.props.items = items
        # item value must implement eq/ne
        self.props.values = []

    @property 
    def values(self):
        return self.props.values

    def validate_props(self, props: Dict[str, Any]):
        if "value" in props:
            value = props["value"]
            return value in [x[1] for x in self.props.items]
        return False

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["values"] = self.props.values
        return res

    async def update_items(self,
                           items: List[Tuple[str, Any]],
                           selected: Optional[List[int]] = None):
        if selected is None:
            selected = []
        await self.put_app_event(
            self.create_update_event({
                "items": items,
                "values": [items[s][1] for s in selected]
            }))
        self.props.items = items
        self.props.values = [items[s][1] for s in selected]

    async def update_value(self, values: List[ValueType]):
        for v in values:
            assert v in [x[1] for x in self.props.items]
        await self.put_app_event(self.create_update_event({"values": values}))
        self.props.values = values

    def update_value_no_sync(self, values: List[ValueType]):
        for v in values:
            assert v in [x[1] for x in self.props.items]
        self.props.values = values

    def state_change_callback(self, values: List[ValueType]):
        self.props.values = values

    async def headless_select(self, values: List[ValueType]):
        uiev = UIEvent({self._flow_uid: values})
        return await self.put_app_event(AppEvent("", {AppEventType.UIEvent: uiev}))

    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val

    async def handle_event(self, ev: EventType):
        await handle_change_event(self, ev)

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

@dataclasses.dataclass
class SliderProps(MUIComponentBaseProps):
    label: str = ""
    ranges: Tuple[NumberType, NumberType, NumberType] = (0, 1, 0) 
    value: NumberType = 0


class Slider(MUIComponentBase[SliderProps]):

    def __init__(self,
                 label: str,
                 begin: NumberType,
                 end: NumberType,
                 step: NumberType,
                 callback: Optional[Callable[[NumberType],
                                             _CORO_NONE]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.Slider, SliderProps, queue)
        self.props.label = label
        self.callback = callback
        assert end > begin and step <= end - begin
        self.props.ranges = (begin, end, step)
        self.props.value = begin

    @property 
    def value(self):
        return self.props.value

    def validate_props(self, props: Dict[str, Any]):
        if "value" in props:
            value = props["value"]
            return (value >= self.props.ranges[0] and value < self.props.ranges[1])
        return False

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["value"] = self.props.value
        return res

    async def update_ranges(self, begin: NumberType,
                            end: NumberType, step: NumberType):
        self.props.ranges = (begin, end, step)
        assert end > begin and step < end - begin
        self.props.value = begin
        await self.put_app_event(
            self.create_update_event({
                "ranges": (begin, end, step),
                "value": self.props.value
            }))

    async def update_value(self, value: NumberType):
        assert value >= self.props.ranges[0] and value <= self.props.ranges[1]
        await self.put_app_event(self.create_update_event({"value": value}))
        self.props.value = value

    def state_change_callback(self, value: NumberType):
        self.props.value = value

    async def headless_change(self, value: NumberType):
        uiev = UIEvent({self._flow_uid: value})
        return await self.put_app_event(AppEvent("", {AppEventType.UIEvent: uiev}))

    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val

    async def handle_event(self, ev: EventType):
        await handle_change_event(self, ev)

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

_T = TypeVar("_T")

@dataclasses.dataclass
class TaskLoopProps(MUIComponentBaseProps):
    label: str = ""
    progresses: List[float] = dataclasses.field(default_factory=list)
    linear: Union[Undefined, bool] = undefined
    task_status: Union[Undefined, int] = undefined

class TaskLoop(MUIComponentBase[TaskLoopProps]):
    """task loop that user use task_loop to start task.
    """

    def __init__(self,
                 label: str,
                 loop_callbcak: Callable[[], _CORO_NONE],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 update_period: float = 0.2) -> None:
        super().__init__(uid, UIType.TaskLoop, TaskLoopProps, queue)
        self.props.label = label
        self.loop_callbcak = loop_callbcak

        self.props.progresses = [0.0]
        self.stack_count = 0
        self.pause_event = asyncio.Event()
        self.pause_event.set()
        self.update_period = update_period


    def get_callback(self):
        return self.loop_callbcak

    async def task_loop(self,
                        it: Union[Iterable[_T], AsyncIterable[_T]],
                        total: int = -1) -> AsyncGenerator[_T, None]:
        if isinstance(it, list):
            total = len(it)
        try:
            cnt = 0
            t = time.time()
            dura = 0.0
            if self.stack_count > 0:
                # keep root progress
                self.props.progresses.append(0.0)
            else:
                # reset states
                await self.update_progress(0.0, 0)
            self.stack_count += 1
            if inspect.isasyncgen(it):
                async for item in it:
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
            else:
                for item in it: # type: ignore
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
            if len(self.props.progresses) > 1:
                self.props.progresses.pop()

    async def update_progress(self, progress: float, index: int):
        progress = max(0, min(progress, 1))
        self.props.progresses[index] = progress
        await self.send_app_event_and_wait(
            self.update_event(progresses=self.props.progresses))

    async def update_label(self, label: str):
        await self.send_app_event_and_wait(self.update_event(label=label))
        self.props.label = label

    async def headless_run(self):
        uiev = UIEvent({self._flow_uid: TaskLoopEvent.Start.value})
        return await self.put_app_event(AppEvent("", {AppEventType.UIEvent: uiev}))

    def set_callback(self, val: Any):
        self.loop_callbcak = val

    async def handle_event(self, ev: EventType):
        data = ev[1]
        if data == TaskLoopEvent.Start.value:
            if self.props.status == UIRunStatus.Stop.value:
                self._task = asyncio.create_task(
                    self.run_callback(self.loop_callbcak))
            else:
                print("IGNORE TaskLoop EVENT", self.props.status)
        elif data == TaskLoopEvent.Pause.value:
            if self.props.status == UIRunStatus.Running.value:
                # pause
                self.pause_event.clear()
                self.props.status = UIRunStatus.Pause.value
            elif self.props.status == UIRunStatus.Pause.value:
                self.pause_event.set()
                self.props.status = UIRunStatus.Running.value
            else:
                print("IGNORE TaskLoop EVENT", self.props.status)
        elif data == TaskLoopEvent.Stop.value:
            if self.props.status == UIRunStatus.Running.value:
                await cancel_task(self._task)
            else:
                print("IGNORE TaskLoop EVENT", self.props.status)
        else:
            raise NotImplementedError

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

class RawTaskLoop(MUIComponentBase[TaskLoopProps]):
    """task loop that user control all events.
    """
    def __init__(self,
                 label: str,
                 callbcak: Callable[[int], _CORO_NONE],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 update_period: float = 0.2) -> None:
        super().__init__(uid, UIType.TaskLoop, TaskLoopProps, queue)
        self.props.label = label
        self.callbcak = callbcak

        self.props.progresses = [0.0]
        self.stack_count = 0
        self.pause_event = asyncio.Event()
        self.pause_event.set()
        self.update_period = update_period

    def get_callback(self):
        return self.callbcak

    async def update_progress(self, progress: float, index: int):
        progress = max(0, min(progress, 1))
        self.props.progresses[index] = progress
        await self.send_app_event_and_wait(
            self.update_event(progresses=self.props.progresses))

    async def update_label(self, label: str):
        await self.send_app_event_and_wait(self.update_event(label=label))
        self.props.label = label

    async def headless_event(self, ev: TaskLoopEvent):
        uiev = UIEvent({self._flow_uid: ev.value})
        return await self.put_app_event(AppEvent("", {AppEventType.UIEvent: uiev}))

    def set_callback(self, val: Any):
        self.callbcak = val

    async def handle_event(self, data: EventType):
        await handle_change_event(self, data)

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

@dataclasses.dataclass
class TypographyProps(MUIComponentBaseProps):
    align: Union[Literal["center", "inherit", "justify", "left", "right"],
                 Undefined] = undefined
    gutter_bottom: Union[bool, Undefined] = undefined
    no_wrap: Union[bool, Undefined] = undefined
    variant: Union[_TypographyVarient, Undefined] = undefined
    paragraph: Union[bool, Undefined] = undefined
    value: str = ""



class Typography(MUIComponentBase[TypographyProps]):

    def __init__(self,
                 init: str,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.Typography, TypographyProps, queue)
        self.props.value = init

    async def write(self, content: str):
        self.props.value = content
        await self.put_app_event(self.create_update_event({"value": self.props.value}))

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["value"] = self.props.value
        return res

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

@dataclasses.dataclass
class PaperProps(MUIFlexBoxProps):
    elevation: Union[int, Undefined] = undefined
    square: Union[bool, Undefined] = undefined
    variant: Union[Literal["elevation", "outlined"], Undefined] = undefined


class Paper(MUIContainerBase[PaperProps, MUIComponentType]):

    def __init__(self,
                 children: Dict[str, MUIComponentType],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.Paper, PaperProps, uid, queue, uid_to_comp,
                         children, inited)

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

@dataclasses.dataclass
class FormControlProps(MUIFlexBoxProps):
    size: Union[Undefined, Literal["small", "medium"]] = undefined
    mui_margin: Union[Undefined, Literal["dense", "none", "normal"]] = undefined

class FormControl(MUIContainerBase[FormControlProps, MUIComponentType]):

    def __init__(self,
                 children: Dict[str, MUIComponentType],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.Paper, FormControlProps, uid, queue, uid_to_comp,
                         children, inited)

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

@dataclasses.dataclass
class CollapseProps(MUIFlexBoxProps):
    orientation: Union[Literal["horizontal", "vertical"],
                       Undefined] = undefined
    timeout: Union[NumberType, Undefined] = undefined


class Collapse(MUIContainerBase[CollapseProps, MUIComponentType]):

    def __init__(self,
                 children: Dict[str, MUIComponentType],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.Collapse, CollapseProps, uid, queue,
                         uid_to_comp, children, inited)

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

# @dataclasses.dataclass
# class AccordionProps(MUIFlexBoxProps):
#     orientation: Union[Literal["horizontal", "vertical"],
#                        Undefined] = undefined
#     timeout: Union[NumberType, Undefined] = undefined


# class Accordion(MUIContainerBase[AccordionProps, MUIComponentType]):

#     def __init__(self,
#                  children: Dict[str, MUIComponentType],
#                  uid: str = "",
#                  queue: Optional[asyncio.Queue] = None,
#                  uid_to_comp: Optional[Dict[str, Component]] = None,
#                  inited: bool = False) -> None:
#         super().__init__(UIType.Accordion, AccordionProps, uid, queue,
#                          uid_to_comp, children, inited)

#     @property 
#     def prop(self):
#         propcls = self.propcls
#         return self._prop_base(propcls, self)


@dataclasses.dataclass
class ChipProps(MUIComponentBaseProps):
    color: Union[_StdColor, str, Undefined] = undefined
    clickable: Union[bool, Undefined] = undefined
    size: Union[Literal["small", "medium"], Undefined] = undefined
    variant: Union[Literal["filled", "outlined"], Undefined] = undefined
    label: str = ""

class Chip(MUIComponentBase[ChipProps]):

    def __init__(self,
                 label: str,
                 callback: Optional[Callable[[], _CORO_NONE]] = None,
                 delete_callback: Optional[Callable[[], _CORO_NONE]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.Chip, ChipProps, queue)
        self.props.label = label
        self.callback = callback
        self.delete_callback = delete_callback

    def to_dict(self):
        res = super().to_dict()
        res["label"] = self.props.label
        return res

    async def headless_click(self):
        uiev = UIEvent({self._flow_uid: self.props.label})
        return await self.put_app_event(AppEvent("", {AppEventType.UIEvent: uiev}))


    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val

    async def handle_event(self, ev: EventType):
        # TODO add delete support
        if self.props.status == UIRunStatus.Running.value:
            # TODO send exception if ignored click
            print("IGNORE EVENT", self.props.status)
            return
        elif self.props.status == UIRunStatus.Stop.value:
            cb2 = self.get_callback()
            if cb2 is not None:
                self._task = asyncio.create_task(
                    self.run_callback(lambda: cb2()))

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

def get_control_value(comp: Union[Input, Switch, RadioGroup, Select, MultipleSelect, Slider]):
    if isinstance(comp, Input):
        return comp.value 
    elif isinstance(comp, Switch):
        return comp.checked 
    elif isinstance(comp, RadioGroup):
        return comp.value 
    elif isinstance(comp, Select):
        return comp.value 
    elif isinstance(comp, MultipleSelect):
        return comp.values
    elif isinstance(comp, Slider):
        return comp.value 
    else:
        raise NotImplementedError("not a control ui")



@dataclasses.dataclass
class AppTerminalProps(BasicProps):
    flex: Union[ValueType, Undefined] = undefined


class AppTerminal(MUIComponentBase[AppTerminalProps]):

    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.AppTerminal, AppTerminalProps, queue)

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

# @dataclasses.dataclass
# class TabProps(MUIBasicProps):
#     label: str = "" 
#     value: str = ""
#     wrapped: Union[Undefined, bool] = undefined
#     disabled: Union[Undefined, bool] = undefined
#     icon: Union[Undefined, str] = undefined
#     icon_position: Union[Literal["start", "end", "bottom", "top"],
#                        Undefined] = undefined

# class Tab(MUIComponentBase[TabProps]):

#     def __init__(self,
#                  label: str,
#                  value: str,
#                  uid: str = "",
#                  queue: Optional[asyncio.Queue] = None) -> None:
#         super().__init__(uid, UIType.Tab, TabProps, queue)
#         self.props.label = label
#         self.props.value = value

#     @property 
#     def prop(self):
#         propcls = self.propcls
#         return self._prop_base(propcls, self)

#     @property 
#     def update_event(self):
#         propcls = self.propcls
#         return self._update_props_base(propcls)

@dataclasses.dataclass
class TabListProps(MUIFlexBoxProps):
    tabs: List[Tuple[str, str]] = dataclasses.field(default_factory=list)
    orientation: Union[Literal["horizontal", "vertical"],
                       Undefined] = undefined
    variant: Union[Literal["scrollable", "vertical", "fullWidth"],
                       Undefined] = undefined
    text_color: Union[Literal["inherit", "primary", "secondary"],
                       Undefined] = undefined
    centered: Union[Undefined, bool] = undefined 
    indicator_color: Union[Literal["primary", "secondary"],
                       Undefined] = undefined
    visible_scrollbar: Union[Undefined, bool] = undefined 


class TabList(MUIComponentBase[TabListProps]):

    def __init__(self,
                tabs: List[Tuple[str, str]],
                 callback: Callable[[str], _CORO_NONE],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.TabList, TabListProps, queue)
        self.callback = callback
        self.props.tabs = tabs

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val

    async def handle_event(self, ev: EventType):
        await handle_change_event(self, ev, sync_first=True)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class TabContextProps(ContainerBaseProps):
    value: str = ""

class TabContext(MUIContainerBase[TabContextProps, MUIComponentType]):

    def __init__(self,
                 children: Union[List[MUIComponentType], Dict[str, MUIComponentType]],
                 value: str,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.TabContext, TabContextProps, uid, queue,
                         uid_to_comp, children, inited)
        self.props.value = value

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)


    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def change_tab(self, value: str):
        return await self.send_app_event_and_wait(self.update_event(value=value))
    
    async def headless_change_tab(self, value: str):
        return await self.put_app_event(AppEvent("", {AppEventType.UIEvent: UIEvent({self._flow_uid: value})}))


@dataclasses.dataclass
class TabPanelProps(MUIFlexBoxProps):
    value: str = ""

class TabPanel(MUIContainerBase[TabPanelProps, MUIComponentType]):

    def __init__(self,
                 children: Union[List[MUIComponentType], Dict[str, MUIComponentType]],
                 value: str,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.TabPanel, TabPanelProps, uid, queue,
                         uid_to_comp, children, inited)
        self.props.value = value

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)


    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class CircularProgressProps(MUIFlexBoxProps):
    value: NumberType = 0
    label_color: Union[Undefined, str] = undefined 
    mui_color: Union[_BtnGroupColor, Undefined] = undefined 
    label_variant: Union[_TypographyVarient, Undefined] = undefined
    size: Union[Undefined, str, NumberType] = undefined 
    variant: Union[Undefined, Literal["determinate", "indeterminate"]] = undefined
    thickness: Union[Undefined, NumberType] = undefined

class CircularProgress(MUIComponentBase[CircularProgressProps]):

    def __init__(self,
                 init_value: NumberType = 0,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.CircularProgress, CircularProgressProps, queue)
        self.props.value = init_value

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def update_value(self, value: NumberType):
        value = min(max(value, 0), 100)
        await self.send_app_event_and_wait(self.update_event(value=value))


@dataclasses.dataclass
class LinearProgressProps(MUIFlexBoxProps):
    value: NumberType = 0
    label_color: Union[Undefined, str] = undefined 
    mui_color: Union[_BtnGroupColor, Undefined] = undefined 
    label_variant: Union[_TypographyVarient, Undefined] = undefined
    variant: Union[Undefined, Literal["determinate", "indeterminate", "buffer", "query"]] = undefined


class LinearProgress(MUIComponentBase[LinearProgressProps]):

    def __init__(self,
                 init_value: NumberType = 0,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.LinearProgress, LinearProgressProps, queue)
        self.props.value = init_value

    @property 
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property 
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def update_value(self, value: NumberType):
        value = min(max(value, 0), 100)
        await self.send_app_event_and_wait(self.update_event(value=value))
