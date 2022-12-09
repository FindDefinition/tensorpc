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
import copy
import dataclasses
import enum
import inspect
import io
import json
import time

from typing import (TYPE_CHECKING, Any, AsyncGenerator, AsyncIterable,
                    Awaitable, Callable, Coroutine, Dict, Iterable, List,
                    Optional, Tuple, Type, TypeVar, Union)

import numpy as np
from PIL import Image as PILImage
from typing_extensions import Literal, TypeAlias

from tensorpc.core.asynctools import cancel_task
from tensorpc.flow.flowapp.components.common import (handle_standard_event)

from .. import colors
from ..core import (AppEvent, AppEventType, BasicProps, Component,
                    ContainerBase, ContainerBaseProps, EventHandler, EventType,
                    Fragment, FrontendEventType, NumberType, T_base_props,
                    T_child, T_container_props, TaskLoopEvent, UIEvent,
                    UIRunStatus, UIType, Undefined, ValueType, undefined,
                    create_ignore_usr_msg, ALL_POINTER_EVENTS,
                    _get_obj_def_path)
from tensorpc.flow.constants import TENSORPC_ANYLAYOUT_FUNC_NAME
if TYPE_CHECKING:
    from .three import ThreeCanvas

_CORO_NONE = Union[Coroutine[None, None, None], None]


_PIL_FORMAT_TO_SUFFIX = {
    "JPEG": "jpg",
    "PNG": "png"
}

def _encode_image_bytes(img: np.ndarray, format: str = "JPEG"):
    pil_img = PILImage.fromarray(img)
    buffered = io.BytesIO()
    pil_img.save(buffered, format=format)
    b64_bytes = base64.b64encode(buffered.getvalue())
    suffix = _PIL_FORMAT_TO_SUFFIX[format]
    return b"data:image/" + suffix.encode("utf-8") + b";base64," + b64_bytes


@dataclasses.dataclass
class MUIBasicProps(BasicProps):
    pass


_OverflowType = Union[Literal["visible"], Literal["hidden"], Literal["scroll"],
                      Literal["auto"]]
PointerEventsProperties: TypeAlias = Union[Literal["auto"], Literal["none"],
                                           Literal["visiblePainted"],
                                           Literal["visibleFill"],
                                           Literal["visibleStroke"],
                                           Literal["visible"],
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
    white_space: Union[Literal["normal", "pre", "nowrap", "pre-wrap",
                               "pre-line", "break-spaces"],
                       Undefined] = undefined
    pointer_events: Union[PointerEventsProperties, Undefined] = undefined


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

MUIComponentType: TypeAlias = Union[MUIComponentBase, MUIContainerBase,
                                    Fragment]

LayoutType: TypeAlias = Union[List[MUIComponentType], Dict[str,
                                                           MUIComponentType]]


def layout_unify(layout: LayoutType):
    if isinstance(layout, list):
        layout = {str(i): v for i, v in enumerate(layout)}
    return layout


@dataclasses.dataclass
class ImageProps(MUIComponentBaseProps):
    image: Union[Undefined, bytes] = undefined


class Image(MUIComponentBase[ImageProps]):
    def __init__(self) -> None:
        super().__init__(UIType.Image, ImageProps, allowed_events=ALL_POINTER_EVENTS)
        # self.image_str: bytes = b""

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["image"] = self.props.image
        return res

    async def show(self, image: np.ndarray, format: str = "JPEG", set_size: bool = False):
        encoded = _encode_image_bytes(image, format)
        self.props.image = encoded
        # self.image_str = encoded
        if set_size:
            ev = self.update_event(image=encoded, width=image.shape[1], height=image.shape[0])
        else:
            ev = self.update_event(image=encoded)
        await self.put_app_event(ev)

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

    async def handle_event(self, ev: EventType):
        await handle_standard_event(self, ev, sync_first=True)

# TODO remove this
Images = Image

@dataclasses.dataclass
class TextProps(MUIComponentBaseProps):
    value: str = ""


class ListItemText(MUIComponentBase[TextProps]):
    def __init__(self, init: str) -> None:
        super().__init__(UIType.ListItemText, TextProps)
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
    variant: Union[Literal["filled", "outlined", "standard"],
                   Undefined] = undefined


class Alert(MUIComponentBase[AlertProps]):
    def __init__(self,
                 value: str,
                 severity: _SEVERITY_TYPES,
                 title: str = "") -> None:
        super().__init__(UIType.Alert, AlertProps)
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
    orientation: Union[Literal["horizontal", "vertical"],
                       Undefined] = undefined


class Divider(MUIComponentBase[DividerProps]):
    def __init__(
        self,
        orientation: Union[Literal["horizontal"],
                           Literal["vertical"]] = "horizontal"
    ) -> None:
        super().__init__(UIType.Divider, DividerProps)
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
    def __init__(self) -> None:
        super().__init__("horizontal")


class VDivider(Divider):
    def __init__(self) -> None:
        super().__init__("vertical")


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
    def __init__(self, name: str, callback: Callable[[], _CORO_NONE]) -> None:
        super().__init__(UIType.Button, ButtonProps,
                         [FrontendEventType.Click.value])
        self.props.name = name
        self.callback = callback
        self.register_event_handler(FrontendEventType.Click.value, callback)

    async def headless_click(self):
        return await self.put_app_event(
            AppEvent(
                "", {
                    AppEventType.UIEvent:
                    UIEvent({
                        self._flow_uid: [FrontendEventType.Click.value, None]
                    })
                }))

    async def handle_event(self, ev: EventType):
        await handle_standard_event(self, ev, sync_first=True)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class IconType(enum.IntEnum):
    RestartAlt = 0
    Menu = 1
    Settings = 2
    Save = 3
    Close = 4
    ExpandMore = 5
    ExpandLess = 6
    Add = 7
    ChevronLeft = 8
    ChevronRight = 9
    Delete = 10
    AddCard = 11
    Clear = 12
    Fullscreen = 13
    PlayArrow = 14
    Pause = 15
    Stop = 16
    MoreVert = 17
    FullscreenExit = 18
    Code = 19
    Terminal = 20
    Videocam = 21
    CameraAlt = 22
    DragHandle = 23
    Dataset = 24
    DataObject = 25
    DataArray = 26
    Cached = 27
    SwapVert = 28


@dataclasses.dataclass
class IconButtonProps(MUIComponentBaseProps):
    mui_color: Union[_BtnGroupColor, Undefined] = undefined
    disabled: Union[bool, Undefined] = undefined
    size: Union[Literal["small", "medium", "large"], Undefined] = undefined
    icon: int = 0
    icon_size: Union[Literal["small", "medium", "large", "inherit"],
                     Undefined] = undefined
    icon_font_size: Union[NumberType, Undefined] = undefined


class IconButton(MUIComponentBase[IconButtonProps]):
    def __init__(self, icon: IconType, callback: Callable[[],
                                                          _CORO_NONE]) -> None:
        super().__init__(UIType.IconButton, IconButtonProps,
                         [FrontendEventType.Click.value])
        self.props.icon = icon.value
        self.register_event_handler(FrontendEventType.Click.value, callback)

    async def headless_click(self):
        return await self.put_app_event(
            AppEvent(
                "", {
                    AppEventType.UIEvent:
                    UIEvent({
                        self._flow_uid: [FrontendEventType.Click.value, None]
                    })
                }))

    async def handle_event(self, ev: EventType):
        await handle_standard_event(self, ev, sync_first=True)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class DialogProps(ContainerBaseProps):
    open: bool = False
    title: Union[str, Undefined] = undefined
    full_screen: Union[str, Undefined] = undefined
    full_width: Union[str, Undefined] = undefined
    max_width: Union[Literal['xs', 'sm', "md", "lg", "xl"],
                     Undefined] = undefined
    scroll: Union[Literal["body", "paper"], Undefined] = undefined


class Dialog(MUIContainerBase[DialogProps, MUIComponentType]):
    def __init__(self, children: LayoutType,
                 callback: Optional[Callable[[bool], _CORO_NONE]] = None) -> None:
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}

        super().__init__(UIType.Dialog,
                         DialogProps,
                         _children=children,
                         allowed_events=[FrontendEventType.DialogClose.value])
        if callback is not None:
            self.register_event_handler(FrontendEventType.DialogClose.value,
                                        callback)

    async def set_open(self, open: bool):
        await self.send_and_wait(self.update_event(open=open))

    async def handle_event(self, ev: EventType):
        await handle_standard_event(self, ev, sync_first=True)

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["open"] = self.props.open
        return res

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    def state_change_callback(
            self,
            value: bool,
            type: ValueType = FrontendEventType.DialogClose.value):
        # this only triggered when dialog closed, so we always set
        # open to false.
        self.props.open = False 

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
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.ButtonGroup, ButtonGroupProps, uid_to_comp,
                         children, inited)
        for v in children.values():
            assert isinstance(v, Button), "all childs must be button"

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class ToggleButtonProps(MUIComponentBaseProps):
    value: ValueType = ""
    name: str = ""
    selected: Union[Undefined, bool] = undefined
    tooltip: Union[str, Undefined] = undefined
    mui_color: Union[_BtnGroupColor, Undefined] = undefined
    disabled: Union[bool, Undefined] = undefined
    full_width: Union[bool, Undefined] = undefined
    size: Union[Literal["small", "medium", "large"], Undefined] = undefined
    icon: Union[int, Undefined] = undefined
    icon_size: Union[Literal["small", "medium", "large"],
                     Undefined] = undefined
    icon_font_size: Union[NumberType, Undefined] = undefined


class ToggleButton(MUIComponentBase[ToggleButtonProps]):
    def __init__(self,
                 value: ValueType,
                 icon: Union[IconType, Undefined] = undefined,
                 name: str = "",
                 callback: Optional[Callable[[bool], _CORO_NONE]] = None) -> None:
        super().__init__(UIType.ToggleButton, ToggleButtonProps)
        if isinstance(icon, Undefined):
            assert name != "", "if icon not provided, you must provide a valid name"
        else:
            self.props.icon = icon.value
        self.props.name = name
        self.props.value = value
        if callback is not None:
            self.register_event_handler(FrontendEventType.Change.value,
                                        callback)

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["selected"] = self.props.selected
        return res

    def state_change_callback(
            self,
            data: bool,
            type: ValueType = FrontendEventType.Change.value):
        self.props.selected = data

    async def handle_event(self, ev: EventType):
        await handle_standard_event(self, ev)

    @property
    def checked(self):
        return self.props.selected

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class ToggleButtonGroupProps(MUIFlexBoxProps):
    value: Optional[Union[ValueType, List[ValueType]]] = None
    orientation: Union[Literal["horizontal", "vertical"],
                       Undefined] = undefined
    mui_color: Union[_BtnGroupColor, Undefined] = undefined
    disabled: Union[bool, Undefined] = undefined
    full_width: Union[bool, Undefined] = undefined
    exclusive: Union[bool, Undefined] = undefined
    size: Union[Literal["small", "medium", "large"], Undefined] = undefined
    name_or_icons: List[ValueType] = dataclasses.field(default_factory=list)
    values: List[ValueType] = dataclasses.field(default_factory=list)
    icon_size: Union[Literal["small", "medium", "large"],
                     Undefined] = undefined
    icon_font_size: Union[NumberType, Undefined] = undefined


class ToggleButtonGroup(MUIContainerBase[ToggleButtonGroupProps,
                                         ToggleButton]):
    def __init__(self,
                 children: Union[List[ToggleButton], Dict[str, ToggleButton]],
                 
                 exclusive: bool,
                 callback: Optional[Callable[
                     [Optional[Union[ValueType, List[ValueType]]]], _CORO_NONE]] = None,
                 value: Optional[Union[ValueType, List[ValueType]]] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.ToggleButtonGroup, ToggleButtonGroupProps,
                         uid_to_comp, children, inited,
                         [FrontendEventType.Change.value])
        for v in children.values():
            assert isinstance(v, ToggleButton), "all childs must be button"
            if not isinstance(v.props.icon, Undefined):
                self.props.name_or_icons.append(v.props.icon)
            else:
                self.props.name_or_icons.append(v.props.name)
            self.props.values.append(v.props.value)
        self.props.value = value
        self.props.exclusive = exclusive
        self.callback = callback
        if not exclusive:
            assert isinstance(value,
                              list), "if not exclusive, value must be a list"
        else:
            assert not isinstance(
                value, list), "if exclusive, value must not be a list"

        if callback is not None:
            self.register_event_handler(FrontendEventType.Change.value,
                                        callback)
    
    async def update_items(self, btns: List[ToggleButton], value: Optional[Union[ValueType, List[ValueType]]] = None):
        name_or_icons = []
        values = []
        for v in btns:
            assert isinstance(v, ToggleButton), "all childs must be button"
            if not isinstance(v.props.icon, Undefined):
                name_or_icons.append(v.props.icon)
            else:
                name_or_icons.append(v.props.name)
            values.append(v.props.value)
        if value is None:
            assert self.props.value in values
            value = self.props.value
        else:
            assert value in values 
        await self.send_and_wait(self.update_event(value=value, name_or_icons=name_or_icons, values=values))

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["value"] = self.props.value
        return res

    async def set_value(self, value: Optional[Union[ValueType, List[ValueType]]]):
        await self.send_and_wait(self.update_event(value=value))

    def state_change_callback(
            self,
            value: Union[ValueType, List[ValueType]],
            type: ValueType = FrontendEventType.Change.value):
        self.props.value = value

    async def handle_event(self, ev: EventType):
        await handle_standard_event(self, ev)


@dataclasses.dataclass
class AccordionDetailsProps(MUIFlexBoxProps):
    pass


@dataclasses.dataclass
class AccordionSummaryProps(MUIFlexBoxProps):
    pass


class AccordionDetails(MUIContainerBase[AccordionDetailsProps,
                                        MUIComponentType]):
    def __init__(self,
                 children: LayoutType,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.AccordionDetail, AccordionDetailsProps,
                         uid_to_comp, children, inited)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)


class AccordionSummary(MUIContainerBase[AccordionSummaryProps,
                                        MUIComponentType]):
    def __init__(self,
                 children: LayoutType,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.AccordionSummary, AccordionSummaryProps,
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


class Accordion(MUIContainerBase[AccordionProps, Union[AccordionDetails,
                                                       AccordionSummary]]):
    def __init__(self,
                 summary: AccordionSummary,
                 details: Optional[AccordionDetails] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        children: Dict[str, Union[AccordionDetails, AccordionSummary]] = {
            "summary": summary
        }
        if details is not None:
            children["details"] = details
        for v in children.values():
            assert isinstance(
                v, (AccordionSummary,
                    AccordionDetails)), "all childs must be summary or detail"
        super().__init__(UIType.Accordion, AccordionProps, uid_to_comp,
                         children, inited)

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["expanded"] = self.props.expanded
        return res

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    def state_change_callback(
            self,
            data: bool,
            type: ValueType = FrontendEventType.Change.value):
        self.props.expanded = data

    async def handle_event(self, ev: EventType):
        await handle_standard_event(self, ev)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class ListItemButton(MUIComponentBase[ButtonProps]):
    def __init__(self, name: str, callback: Callable[[], _CORO_NONE]) -> None:
        super().__init__(UIType.ListItemButton, ButtonProps,
                         [FrontendEventType.Click.value])
        self.props.name = name
        self.callback = callback
        self.register_event_handler(FrontendEventType.Click.value, callback)

    async def headless_click(self):
        uiev = UIEvent(
            {self._flow_uid: [FrontendEventType.Click.value, self.props.name]})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    async def handle_event(self, ev: EventType):
        await handle_standard_event(self, ev, sync_first=False)

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
                 children: Optional[LayoutType] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 base_type: UIType = UIType.FlexBox,
                 inited: bool = False,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 wrapped_obj: Optional[Any] = None) -> None:
        if children is not None and isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(base_type,
                         MUIFlexBoxProps,
                         uid_to_comp,
                         children,
                         inited,
                         uid=uid,
                         queue=queue)
        self._wrapped_obj = wrapped_obj

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
                 children: Optional[LayoutType] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 subheader: str = "",
                 inited: bool = False) -> None:
        if children is not None and isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.MUIList,
                         MUIListProps,
                         uid_to_comp=uid_to_comp,
                         _children=children,
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


def VBox(layout: LayoutType):
    res = FlexBox(children=layout)
    res.prop(flex_flow="column")
    return res


def HBox(layout: LayoutType):
    res = FlexBox(children=layout)
    res.prop(flex_flow="row")
    return res


def Box(layout: LayoutType):
    return FlexBox(children=layout)


def VList(layout: LayoutType, subheader: str = ""):
    return MUIList(subheader=subheader, children=layout)


@dataclasses.dataclass
class RadioGroupProps(MUIComponentBaseProps):
    names: List[str] = dataclasses.field(default_factory=list)
    row: Union[Undefined, bool] = undefined
    value: str = ""


class RadioGroup(MUIComponentBase[RadioGroupProps]):
    def __init__(
        self,
        names: List[str],
        row: bool,
        callback: Optional[Callable[[str], Coroutine[None, None, None]]] = None
    ) -> None:
        super().__init__(UIType.RadioGroup, RadioGroupProps,
                         [FrontendEventType.Change.value])
        self.props.names = names
        self.callback = callback
        self.props.row = row
        self.props.value = names[0]
        if callback is not None:
            self.register_event_handler(FrontendEventType.Change.value,
                                        callback)

    def state_change_callback(self,
                              data: str,
                              type: ValueType = FrontendEventType.Change.value
                              ):
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
        uiev = UIEvent({
            self._flow_uid:
            [FrontendEventType.Change.value, self.props.names[index]]
        })
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    async def handle_event(self, ev: EventType):
        await handle_standard_event(self, ev)

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
                                    "datetime-local", "email", "file",
                                    "hidden", "image", "month", "number",
                                    "password", "radio", "range", 'reset',
                                    "search", "submit", "tel", "text", "time",
                                    "url", "week"]


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
    mui_margin: Union[Undefined, Literal["dense", "none",
                                         "normal"]] = "dense"
    variant: Union[Undefined, Literal["filled", "outlined",
                                      "standard"]] = undefined
    type: Union[Undefined, _HTMLInputType] = undefined


class Input(MUIComponentBase[InputProps]):
    def __init__(self,
                 label: str,
                 multiline: bool = False,
                 callback: Optional[Callable[[str], _CORO_NONE]] = None,
                 init: str = "") -> None:
        super().__init__(UIType.Input, InputProps,
                         [FrontendEventType.Change.value])
        self.props.label = label
        self.callback = callback
        self.props.value = init
        self.props.multiline = multiline
        if callback is not None:
            self.register_event_handler(FrontendEventType.Change.value,
                                        callback)

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["value"] = self.props.value
        return res

    @property
    def value(self):
        return self.props.value

    def state_change_callback(self,
                              data: str,
                              type: ValueType = FrontendEventType.Change.value
                              ):
        self.props.value = data

    async def headless_write(self, content: str):
        uiev = UIEvent(
            {self._flow_uid: (FrontendEventType.Change.value, content)})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

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
    label_placement: Union[Literal["top", "start", "bottom", "end"],
                           Undefined] = undefined


class SwitchBase(MUIComponentBase[SwitchProps]):
    def __init__(
            self,
            label: str,
            base_type: UIType,
            callback: Optional[Callable[[bool], _CORO_NONE]] = None) -> None:
        super().__init__(base_type, SwitchProps,
                         [FrontendEventType.Change.value])
        self.props.label = label
        self.props.checked = False
        if callback is not None:
            self.register_event_handler(FrontendEventType.Change.value,
                                        callback)

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["checked"] = self.props.checked
        return res

    @property
    def checked(self):
        return self.props.checked

    def state_change_callback(
            self,
            data: bool,
            type: ValueType = FrontendEventType.Change.value):
        self.props.checked = data

    async def headless_write(self, checked: bool):
        uiev = UIEvent(
            {self._flow_uid: (FrontendEventType.Change.value, checked)})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    def __bool__(self):
        return self.props.checked

    async def handle_event(self, ev: EventType):
        await handle_standard_event(self, ev)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class Switch(SwitchBase):
    def __init__(
            self,
            label: str,
            callback: Optional[Callable[[bool], _CORO_NONE]] = None) -> None:
        super().__init__(label, UIType.Switch, callback)


class Checkbox(SwitchBase):
    def __init__(
            self,
            label: str,
            callback: Optional[Callable[[bool], _CORO_NONE]] = None) -> None:
        super().__init__(label, UIType.Checkbox, callback)


@dataclasses.dataclass
class SelectProps(MUIComponentBaseProps):
    label: str = ""
    items: List[Tuple[str,
                      ValueType]] = dataclasses.field(default_factory=list)
    value: ValueType = ""
    size: Union[Undefined, Literal["small", "medium"]] = undefined
    mui_margin: Union[Undefined, Literal["dense", "none",
                                         "normal"]] = undefined
    variant: Union[Undefined, Literal["filled", "outlined",
                                      "standard"]] = undefined


class Select(MUIComponentBase[SelectProps]):
    def __init__(
            self,
            label: str,
            items: List[Tuple[str, ValueType]],
            callback: Optional[Callable[[ValueType],
                                        _CORO_NONE]] = None) -> None:
        super().__init__(UIType.Select, SelectProps,
                         [FrontendEventType.Change.value])
        self.props.label = label
        self.callback = callback
        # assert len(items) > 0
        self.props.items = items
        # item value must implement eq/ne
        self.props.value = ""
        self.props.size = "small"
        if callback is not None:
            self.register_event_handler(FrontendEventType.Change.value,
                                        callback)

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

    def state_change_callback(
            self,
            value: ValueType,
            type: ValueType = FrontendEventType.Change.value):
        self.props.value = value

    async def headless_select(self, value: ValueType):
        uiev = UIEvent(
            {self._flow_uid: (FrontendEventType.Change.value, value)})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    async def handle_event(self, ev: EventType):
        await handle_standard_event(self, ev)

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
    items: List[Tuple[str,
                      ValueType]] = dataclasses.field(default_factory=list)
    values: List[ValueType] = dataclasses.field(default_factory=list)


class MultipleSelect(MUIComponentBase[MultipleSelectProps]):
    def __init__(
        self,
        label: str,
        items: List[Tuple[str, ValueType]],
        callback: Optional[Callable[[List[ValueType]], _CORO_NONE]] = None
    ) -> None:
        super().__init__(UIType.MultipleSelect, MultipleSelectProps,
                         [FrontendEventType.Change.value])
        self.props.label = label
        self.callback = callback
        assert len(items) > 0
        self.props.items = items
        # item value must implement eq/ne
        self.props.values = []
        if callback is not None:
            self.register_event_handler(FrontendEventType.Change.value,
                                        callback)

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

    def state_change_callback(
            self,
            values: List[ValueType],
            type: ValueType = FrontendEventType.Change.value):
        self.props.values = values

    async def headless_select(self, values: List[ValueType]):
        uiev = UIEvent(
            {self._flow_uid: (FrontendEventType.Change.value, values)})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    async def handle_event(self, ev: EventType):
        await handle_standard_event(self, ev)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class AutocompletePropsBase(MUIComponentBaseProps):
    label: str = ""
    # input_value: str = ""
    options: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    size: Union[Undefined, Literal["small", "medium"]] = undefined
    mui_margin: Union[Undefined, Literal["dense", "none",
                                         "normal"]] = undefined
    input_variant: Union[Undefined, Literal["filled", "outlined",
                                            "standard"]] = undefined
    variant: Union[Undefined, Literal["checkbox", "standard"]] = undefined

    disable_clearable: Union[Undefined, bool] = undefined
    disable_close_on_select: Union[Undefined, bool] = undefined
    clear_on_escape: Union[Undefined, bool] = undefined
    include_input_in_list: Union[Undefined, bool] = undefined
    disable_list_wrap: Union[Undefined, bool] = undefined
    open_on_focus: Union[Undefined, bool] = undefined
    auto_highlight: Union[Undefined, bool] = undefined
    auto_select: Union[Undefined, bool] = undefined
    disabled: Union[Undefined, bool] = undefined
    disable_portal: Union[Undefined, bool] = undefined
    blur_on_select: Union[Undefined, bool] = undefined
    clear_on_blur: Union[Undefined, bool] = undefined
    select_on_focus: Union[Undefined, bool] = undefined
    read_only: Union[Undefined, bool] = undefined
    free_solo: Union[Undefined, bool] = undefined

    group_by_key: Union[Undefined, str] = undefined
    limit_tags: Union[Undefined, int] = undefined


@dataclasses.dataclass
class AutocompleteProps(AutocompletePropsBase):
    value: Optional[Dict[str, Any]] = None


class Autocomplete(MUIComponentBase[AutocompleteProps]):
    def __init__(
        self,
        label: str,
        options: List[Dict[str, Any]],
        callback: Optional[Callable[[Dict[str, Any]],
                                    _CORO_NONE]] = None) -> None:
        super().__init__(UIType.AutoComplete, AutocompleteProps,
                         [FrontendEventType.Change.value])
        self.props.label = label
        self.callback = callback
        # assert len(items) > 0
        self.props.options = options
        # item value must implement eq/ne
        self.props.value = None
        self.props.size = "small"
        if callback is not None:
            self.register_event_handler(FrontendEventType.Change.value,
                                        callback)

    @property
    def value(self):
        return self.props.value

    def validate_props(self, props: Dict[str, Any]):
        return False

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["value"] = self.props.value
        res["options"] = self.props.options
        return res

    async def update_options(self, options: List[Dict[str, Any]],
                             selected: int):
        await self.put_app_event(
            self.create_update_event({
                "options": options,
                "value": options[selected]
            }))
        self.props.options = options
        self.props.value = options[selected]

    async def update_value(self, value: Optional[Dict[str, Any]]):
        await self.put_app_event(self.create_update_event({"value": value}))
        self.props.value = value

    def update_value_no_sync(self, value: Optional[Dict[str, Any]]):
        self.props.value = value

    def state_change_callback(
            self,
            value: Union[str, Optional[Dict[str, Any]]],
            type: ValueType = FrontendEventType.Change.value):
        if type == FrontendEventType.Change.value:
            if value is not None:
                assert isinstance(value, dict)
            self.props.value = value
        # else:
        #     assert isinstance(value, str)
        #     print("self.props.input_value", value, type)
        #     self.props.input_value = value

    async def headless_select(self, value: ValueType):
        uiev = UIEvent(
            {self._flow_uid: (FrontendEventType.Change.value, value)})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    async def handle_event(self, data: EventType):
        await handle_standard_event(self, data)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class MultipleAutocompleteProps(AutocompletePropsBase):
    value: List[Dict[str, Any]] = dataclasses.field(default_factory=list)


class MultipleAutocomplete(MUIComponentBase[MultipleAutocompleteProps]):
    def __init__(
        self,
        label: str,
        options: List[Dict[str, Any]],
        callback: Optional[Callable[[Dict[str, Any]],
                                    _CORO_NONE]] = None) -> None:
        super().__init__(UIType.MultipleAutoComplete,
                         MultipleAutocompleteProps,
                         [FrontendEventType.Change.value])
        for op in options:
            assert "label" in op, "must contains label in options"
        self.props.label = label
        self.callback = callback
        # assert len(items) > 0
        self.props.options = options
        # item value must implement eq/ne
        self.props.value = []
        self.props.size = "small"
        if callback is not None:
            self.register_event_handler(FrontendEventType.Change.value,
                                        callback)

    @property
    def value(self):
        return self.props.value

    def validate_props(self, props: Dict[str, Any]):
        return False

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["value"] = self.props.value
        res["options"] = self.props.options
        return res

    async def update_options(self,
                             options: List[Dict[str, Any]],
                             selected: Optional[List[int]] = None):
        if selected is None:
            selected = []
        await self.put_app_event(
            self.create_update_event({
                "options": options,
                "value": [options[s] for s in selected]
            }))
        self.props.options = options
        self.props.value = [options[s] for s in selected]

    async def update_value(self, value: List[Dict[str, Any]]):
        await self.put_app_event(self.create_update_event({"value": value}))
        self.props.value = value

    def update_value_no_sync(self, value: List[Dict[str, Any]]):
        self.props.value = value

    def state_change_callback(
            self,
            value: Union[str, List[Dict[str, Any]]],
            type: ValueType = FrontendEventType.Change.value):
        if type == FrontendEventType.Change.value:
            assert isinstance(value, list)
            self.props.value = value
        # else:
        #     assert isinstance(value, str)
        #     self.props.input_value = value

    async def headless_select(self, value: List[Dict[str, Any]]):
        uiev = UIEvent(
            {self._flow_uid: (FrontendEventType.Change.value, value)})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    async def handle_event(self, data: EventType):
        await handle_standard_event(self, data)

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
    def __init__(
            self,
            label: str,
            begin: NumberType,
            end: NumberType,
            step: NumberType,
            callback: Optional[Callable[[NumberType],
                                        _CORO_NONE]] = None) -> None:
        super().__init__(UIType.Slider, SliderProps,
                         [FrontendEventType.Change.value])
        self.props.label = label
        self.callback = callback
        assert end > begin and step <= end - begin
        self.props.ranges = (begin, end, step)
        self.props.value = begin
        if callback is not None:
            self.register_event_handler(FrontendEventType.Change.value,
                                        callback)

    @property
    def value(self):
        return self.props.value

    def validate_props(self, props: Dict[str, Any]):
        if "value" in props:
            value = props["value"]
            return (value >= self.props.ranges[0]
                    and value < self.props.ranges[1])
        return False

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["value"] = self.props.value
        return res

    async def update_ranges(self, begin: NumberType, end: NumberType,
                            step: NumberType):
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

    def state_change_callback(
            self,
            value: NumberType,
            type: ValueType = FrontendEventType.Change.value):
        self.props.value = value

    async def headless_change(self, value: NumberType):
        uiev = UIEvent(
            {self._flow_uid: (FrontendEventType.Change.value, value)})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    async def handle_event(self, ev: EventType):
        await handle_standard_event(self, ev)

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
                 loop_callbcak: Optional[Callable[[], _CORO_NONE]] = None,
                 update_period: float = 0.2,
                 raw_update: bool = False) -> None:
        super().__init__(UIType.TaskLoop, TaskLoopProps)
        self.props.label = label
        self.loop_callbcak = loop_callbcak

        self.props.progresses = [0.0]
        self.stack_count = 0
        self.pause_event = asyncio.Event()
        self.pause_event.set()
        self.update_period = update_period
        self._raw_update = raw_update

    async def task_loop(self,
                        it: Union[Iterable[_T], AsyncIterable[_T]],
                        total: int = -1) -> AsyncGenerator[_T, None]:
        if self._raw_update:
            raise ValueError("when raw update enabled, you can't use this function")
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
                for item in it:  # type: ignore
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
        await self.send_and_wait(
            self.update_event(progresses=self.props.progresses))

    async def update_label(self, label: str):
        await self.send_and_wait(self.update_event(label=label))
        self.props.label = label
    
    async def set_raw_update(self, enable: bool):
        if self.props.status != UIRunStatus.Stop.value:
            raise ValueError("you must set raw_update in stop status")
        if enable != self._raw_update:
            await self.clear()
        self._raw_update = enable

    async def clear(self):
        await cancel_task(self._task)
        await self.send_and_wait(
            self.update_event(
                task_status=UIRunStatus.Stop.value, progresses=[0]))

    async def headless_run(self):
        uiev = UIEvent({
            self._flow_uid:
            [FrontendEventType.Change.value, TaskLoopEvent.Start.value]
        })
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    async def handle_event(self, ev: EventType):
        if self._raw_update:
            return await handle_standard_event(self, ev)
        data = ev[1]
        if data == TaskLoopEvent.Start.value:
            if self.props.status == UIRunStatus.Stop.value:
                if self.loop_callbcak is not None:
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
                 callback: Callable[[int], _CORO_NONE],
                 update_period: float = 0.2) -> None:
        super().__init__(UIType.TaskLoop, TaskLoopProps,
                         [FrontendEventType.Change.value])
        self.props.label = label
        self.callback = callback

        self.props.progresses = [0.0]
        self.stack_count = 0
        self.pause_event = asyncio.Event()
        self.pause_event.set()
        self.update_period = update_period
        self.register_event_handler(FrontendEventType.Change.value, callback)

    async def update_progress(self, progress: float, index: int):
        progress = max(0, min(progress, 1))
        self.props.progresses[index] = progress
        await self.send_and_wait(
            self.update_event(progresses=self.props.progresses))


    async def update_label(self, label: str):
        await self.send_and_wait(self.update_event(label=label))
        self.props.label = label

    async def headless_event(self, ev: TaskLoopEvent):
        uiev = UIEvent(
            {self._flow_uid: [FrontendEventType.Change.value, ev.value]})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    async def handle_event(self, data: EventType):
        await handle_standard_event(self, data)

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
    def __init__(self, init: str) -> None:
        super().__init__(UIType.Typography, TypographyProps)
        self.props.value = init

    async def write(self, content: str):
        self.props.value = content
        await self.put_app_event(
            self.create_update_event({"value": self.props.value}))

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
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.Paper, PaperProps, uid_to_comp, children,
                         inited)

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
    mui_margin: Union[Undefined, Literal["dense", "none",
                                         "normal"]] = undefined


class FormControl(MUIContainerBase[FormControlProps, MUIComponentType]):
    def __init__(self,
                 children: Dict[str, MUIComponentType],
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.Paper, FormControlProps, uid_to_comp, children,
                         inited)

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
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.Collapse, CollapseProps, uid_to_comp, children,
                         inited)

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
#         super().__init__(UIType.Accordion, AccordionProps,
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
    def __init__(
        self,
        label: str,
        callback: Optional[Callable[[], _CORO_NONE]] = None,
        delete_callback: Optional[Callable[[], _CORO_NONE]] = None,
    ) -> None:
        super().__init__(
            UIType.Chip, ChipProps,
            [FrontendEventType.Change.value, FrontendEventType.Delete.value])
        self.props.label = label
        self.callback = callback
        self.delete_callback = delete_callback
        if callback is not None:
            self.register_event_handler(FrontendEventType.Click.value,
                                        callback)
        if delete_callback is not None:
            self.register_event_handler(FrontendEventType.Delete.value,
                                        delete_callback)

    def to_dict(self):
        res = super().to_dict()
        res["label"] = self.props.label
        return res

    async def headless_click(self):
        uiev = UIEvent({self._flow_uid: [FrontendEventType.Click.value, None]})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    async def handle_event(self, ev: EventType):
        # TODO add delete support
        if self.props.status == UIRunStatus.Running.value:
            # TODO send exception if ignored click
            print("IGNORE EVENT", self.props.status)
            return
        elif self.props.status == UIRunStatus.Stop.value:
            handler = self.get_event_handler(ev[0])
            if handler is not None:
                self._task = asyncio.create_task(
                    self.run_callback(lambda: handler.cb()))

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


def get_control_value(comp: Union[Input, Switch, RadioGroup, Select,
                                  MultipleSelect, Slider]):
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
    def __init__(self) -> None:
        super().__init__(UIType.AppTerminal, AppTerminalProps)

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
    def __init__(
        self,
        tabs: List[Tuple[str, str]],
        callback: Callable[[str], _CORO_NONE],
    ) -> None:
        super().__init__(UIType.TabList, TabListProps,
                         [FrontendEventType.Change.value])
        self.callback = callback
        self.props.tabs = tabs
        if callback is not None:
            self.register_event_handler(FrontendEventType.Change.value,
                                        callback)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    async def handle_event(self, ev: EventType):
        await handle_standard_event(self, ev, sync_first=True)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class TabContextProps(ContainerBaseProps):
    value: str = ""


class TabContext(MUIContainerBase[TabContextProps, MUIComponentType]):
    def __init__(self,
                 children: LayoutType,
                 value: str,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.TabContext, TabContextProps, uid_to_comp,
                         children, inited)
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
        return await self.send_and_wait(self.update_event(value=value))

    async def headless_change_tab(self, value: str):
        return await self.put_app_event(
            AppEvent(
                "", {
                    AppEventType.UIEvent:
                    UIEvent({
                        self._flow_uid:
                        [FrontendEventType.Change.value, value]
                    })
                }))


@dataclasses.dataclass
class TabPanelProps(MUIFlexBoxProps):
    value: str = ""


class TabPanel(MUIContainerBase[TabPanelProps, MUIComponentType]):
    def __init__(self,
                 children: LayoutType,
                 value: str,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.TabPanel, TabPanelProps, uid_to_comp, children,
                         inited)
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
    variant: Union[Undefined, Literal["determinate",
                                      "indeterminate"]] = undefined
    thickness: Union[Undefined, NumberType] = undefined


class CircularProgress(MUIComponentBase[CircularProgressProps]):
    def __init__(self, init_value: NumberType = 0) -> None:
        super().__init__(UIType.CircularProgress, CircularProgressProps)
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
        await self.send_and_wait(self.update_event(value=value))


@dataclasses.dataclass
class LinearProgressProps(MUIFlexBoxProps):
    value: NumberType = 0
    label_color: Union[Undefined, str] = undefined
    mui_color: Union[_BtnGroupColor, Undefined] = undefined
    label_variant: Union[_TypographyVarient, Undefined] = undefined
    variant: Union[Undefined, Literal["determinate", "indeterminate", "buffer",
                                      "query"]] = undefined


class LinearProgress(MUIComponentBase[LinearProgressProps]):
    def __init__(
        self,
        init_value: NumberType = 0,
    ) -> None:
        super().__init__(UIType.LinearProgress, LinearProgressProps)
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
        await self.send_and_wait(self.update_event(value=value))

def flex_wrapper(obj: Any):
    """wrap a object which define a layout function "tensorpc_flow_layout"
    enable simple layout creation for arbitrary object without inherit
    """
    func_name = TENSORPC_ANYLAYOUT_FUNC_NAME
    assert hasattr(obj, func_name), f"wrapped object must define a zero-arg function {func_name} that return a flexbox"
    layout_flex = getattr(obj, func_name)()
    assert isinstance(layout_flex, FlexBox), f"{func_name} must return a flexbox"
    # set _flow_comp_def_path to this object
    layout_flex._flow_comp_def_path = _get_obj_def_path(obj)
    layout_flex._wrapped_obj = obj
    return layout_flex

