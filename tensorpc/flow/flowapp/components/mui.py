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

import abc
import asyncio
import base64
import copy
from functools import partial

import tensorpc.core.dataclass_dispatch as dataclasses
import enum
import inspect
import io
import json
import time
import uuid

from typing import (TYPE_CHECKING, Any, AsyncGenerator, AsyncIterable,
                    Awaitable, Callable, Coroutine, Dict, Iterable, List,
                    Optional, Set, Tuple, Type, TypeVar, Union)

import numpy as np
from PIL import Image as PILImage
from typing_extensions import Literal, TypeAlias, TypedDict, Self
from pydantic import field_validator, model_validator

from tensorpc.core.tree_id import UniqueTreeId, UniqueTreeIdForTree
from .typemetas import Vector3Type
from tensorpc.core.asynctools import cancel_task
from tensorpc.core.defs import FileResource
from tensorpc.core.event_emitter.aio import AsyncIOEventEmitter
from tensorpc.core.serviceunit import AppFuncType, ObjectReloadManager, ReloadableDynamicClass, ServFunctionMeta
from tensorpc.flow.client import MasterMeta
from tensorpc.flow.flowapp.appcore import Event, EventDataType
from tensorpc.flow.flowapp.components.common import (handle_standard_event)
from tensorpc.flow.flowapp.reload import AppReloadManager
from ...jsonlike import JsonLikeType, BackendOnlyProp, ContextMenuData, JsonLikeNode, as_dict_no_undefined
from .. import colors
from ..core import (AppComponentCore, AppEvent, AppEventType, BasicProps,
                    Component, ContainerBase, ContainerBaseProps, EventHandler,
                    SimpleEventType, FlowSpecialMethods, Fragment,
                    FrontendEventType, NumberType, T_base_props, T_child,
                    T_container_props, TaskLoopEvent, UIEvent, UIRunStatus,
                    UIType, Undefined, ValueType, undefined,
                    create_ignore_usr_msg, ALL_POINTER_EVENTS,
                    _get_obj_def_path, MatchCase)
from tensorpc.flow.constants import TENSORPC_ANYLAYOUT_FUNC_NAME
if TYPE_CHECKING:
    from .three import Canvas

_CORO_NONE = Union[Coroutine[None, None, None], None]
CORO_NONE = Union[Coroutine[None, None, None], None]

_PIL_FORMAT_TO_SUFFIX = {"JPEG": "jpg", "PNG": "png"}


class Position(enum.IntEnum):
    TopLeft = 0
    TopCenter = 1
    TopRight = 2
    LeftCenter = 3
    Center = 4
    RightCenter = 5
    BottomLeft = 6
    BottomCenter = 7
    BottomRight = 8


@dataclasses.dataclass
class MUIBasicProps(BasicProps):
    pass


_OverflowType: TypeAlias = Literal["visible", "hidden", "scroll", "auto"]
PointerEventsProperties: TypeAlias = Literal["auto", "none", "visiblePainted",
                                             "visibleFill", "visibleStroke",
                                             "visible", "painted", "fill",
                                             "stroke", "all", "inherit"]


@dataclasses.dataclass
class FlexComponentBaseProps(BasicProps):
    """all props must have a default value, 
    manage state by your self.
    """
    display: Union[Literal["flex", "none", "block", "inline", "grid", "table",
                           "inline-block", "inline-flex"],
                   Undefined] = undefined
    cursor: Union[str, Undefined] = undefined
    position: Union[Literal["absolute", "relative", "fixed"],
                    Undefined] = undefined
    top: Union[ValueType, Undefined] = undefined
    bottom: Union[ValueType, Undefined] = undefined
    left: Union[ValueType, Undefined] = undefined
    right: Union[ValueType, Undefined] = undefined
    zIndex: Union[ValueType, Undefined] = undefined
    textAlign: Union[Literal["start", "end", "inherit", "left", "right", "center", "justify"], Undefined] = undefined
    flex: Union[ValueType, Undefined] = undefined
    alignSelf: Union[Literal["auto", "flex-start", "flex-end", "center",
                             "baseline", "stretch"], Undefined] = undefined
    flexGrow: Union[ValueType, Undefined] = undefined
    flexShrink: Union[ValueType, Undefined] = undefined
    flexBasis: Union[ValueType, Undefined] = undefined

    height: Union[ValueType, Undefined] = undefined
    width: Union[ValueType, Undefined] = undefined
    maxHeight: Union[ValueType, Undefined] = undefined
    maxWidth: Union[ValueType, Undefined] = undefined
    minHeight: Union[ValueType, Undefined] = undefined
    minWidth: Union[ValueType, Undefined] = undefined
    padding: Union[ValueType, Undefined] = undefined
    paddingTop: Union[ValueType, Undefined] = undefined
    paddingBottom: Union[ValueType, Undefined] = undefined
    paddingLeft: Union[ValueType, Undefined] = undefined
    paddingRight: Union[ValueType, Undefined] = undefined
    margin: Union[ValueType, Undefined] = undefined
    marginTop: Union[ValueType, Undefined] = undefined
    marginLeft: Union[ValueType, Undefined] = undefined
    marginRight: Union[ValueType, Undefined] = undefined
    marginBottom: Union[ValueType, Undefined] = undefined

    overflow: Union[_OverflowType, Undefined] = undefined
    overflowY: Union[_OverflowType, Undefined] = undefined
    overflowX: Union[_OverflowType, Undefined] = undefined

    color: Union[ValueType, Undefined] = undefined
    backgroundColor: Union[ValueType, Undefined] = undefined
    fontSize: Union[ValueType, Undefined] = undefined
    fontFamily: Union[str, Undefined] = undefined
    border: Union[str, Undefined] = undefined
    borderTop: Union[ValueType, Undefined] = undefined
    borderLeft: Union[ValueType, Undefined] = undefined
    borderRight: Union[ValueType, Undefined] = undefined
    borderBottom: Union[ValueType, Undefined] = undefined
    borderColor: Union[str, Undefined] = undefined
    borderRadius: Union[ValueType, Undefined] = undefined
    borderImage: Union[str, Undefined] = undefined

    whiteSpace: Union[Literal["normal", "pre", "nowrap", "pre-wrap",
                              "pre-line", "break-spaces"],
                      Undefined] = undefined
    wordBreak: Union[Literal["normal", "break-all", "keep-all", "break-word"],
                     Undefined] = undefined
    pointerEvents: Union[PointerEventsProperties, Undefined] = undefined


@dataclasses.dataclass
class MUIComponentBaseProps(FlexComponentBaseProps):
    pass


class MUIComponentBase(Component[T_base_props, "MUIComponentType"]):
    pass


class MUIContainerBase(ContainerBase[T_container_props, T_child]):
    pass


@dataclasses.dataclass
class FlexBoxProps(FlexComponentBaseProps):
    # element id only available in container
    elementId: Union[str, Undefined] = undefined
    alignContent: Union[Literal["flex-start", "flex-end", "center",
                                "space-between", "space-around", "stretch"],
                        Undefined] = undefined
    alignItems: Union[Literal["flex-start", "flex-end", "center", "baseline",
                              "stretch"], Undefined] = undefined
    justifyContent: Union[Literal["flex-start", "flex-end", "center",
                                  "space-between", "space-around",
                                  "space-evenly"], Undefined] = undefined
    flexDirection: Union[Literal["row", "row-reverse", "column",
                                 "column-reverse"], Undefined] = undefined
    flexWrap: Union[Literal["nowrap", "wrap", "wrap-reverse"],
                    Undefined] = undefined
    flexFlow: Union[str, Undefined] = undefined


# we can't let mui use three component.
@dataclasses.dataclass
class MUIFlexBoxProps(FlexBoxProps, ContainerBaseProps):
    pass


@dataclasses.dataclass
class MUIFlexBoxWithDndProps(MUIFlexBoxProps):
    draggable: Union[bool, Undefined] = undefined
    droppable: Union[bool, Undefined] = undefined
    allowedDndTypes: Union[List[str], Undefined] = undefined
    sxOverDrop: Union[Dict[str, Any], Undefined] = undefined
    allowFile: Union[bool, Undefined] = undefined
    dragType: Union[str, Undefined] = undefined
    dragData: Union[Dict[str, Any], Undefined] = undefined
    dragInChild: Union[bool, Undefined] = undefined
    takeDragRef: Union[bool, Undefined] = undefined
    className: Union[str, Undefined] = undefined

    @field_validator('sxOverDrop')
    def sx_over_drop_validator(cls, v: Union[Dict[str, Any], Undefined]):
        if isinstance(v, Undefined):
            return v
        # avoid nested check
        if "sxOverDrop" in v:
            v.pop("sxOverDrop")
        # validate sx over drop
        MUIFlexBoxWithDndProps(**v)
        return v


_TypographyVarient: TypeAlias = Literal['body1', 'body2', 'button', 'caption',
                                        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                                        'inherit', 'overline', 'subtitle1',
                                        'subtitle2']

_StdColor: TypeAlias = Literal['default', 'primary', 'secondary', 'error',
                               'info', 'success', 'warning']

_StdColorNoDefault: TypeAlias = Literal['primary', 'secondary', 'error',
                                        'info', 'success', 'warning']

MUIComponentType: TypeAlias = Union[MUIComponentBase, MUIContainerBase,
                                    Fragment, MatchCase]

LayoutType: TypeAlias = Union[List[MUIComponentType], Dict[str,
                                                           MUIComponentType]]


def layout_unify(layout: LayoutType):
    if isinstance(layout, list):
        layout = {str(i): v for i, v in enumerate(layout)}
    return layout


@dataclasses.dataclass
class ImageProps(MUIComponentBaseProps):
    image: Union[Undefined, str, bytes] = undefined
    alt: str = ""


class Image(MUIComponentBase[ImageProps]):

    def __init__(self) -> None:
        super().__init__(UIType.Image,
                         ImageProps,
                         allowed_events=ALL_POINTER_EVENTS)
        # self.image_str: bytes = b""
        self.event_click = self._create_event_slot(FrontendEventType.Click)
        self.event_double_click = self._create_event_slot(
            FrontendEventType.DoubleClick)
        self.event_pointer_enter = self._create_event_slot(
            FrontendEventType.Enter)
        self.event_pointer_leave = self._create_event_slot(
            FrontendEventType.Leave)
        self.event_pointer_down = self._create_event_slot(
            FrontendEventType.Down)
        self.event_pointer_up = self._create_event_slot(FrontendEventType.Up)
        self.event_pointer_move = self._create_event_slot(
            FrontendEventType.Move)
        self.event_pointer_over = self._create_event_slot(
            FrontendEventType.Over)
        self.event_pointer_out = self._create_event_slot(FrontendEventType.Out)
        self.event_pointer_context_menu = self._create_event_slot(
            FrontendEventType.ContextMenu)

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["image"] = self.props.image
        return res

    @staticmethod
    def encode_image_bytes(img: np.ndarray, format: str = "JPEG"):
        pil_img = PILImage.fromarray(img)
        buffered = io.BytesIO()
        pil_img.save(buffered, format=format)
        b64_bytes = base64.b64encode(buffered.getvalue())
        suffix = _PIL_FORMAT_TO_SUFFIX[format]
        return b"data:image/" + suffix.encode(
            "utf-8") + b";base64," + b64_bytes

    @staticmethod
    def encode_image_string(img: np.ndarray, format: str = "JPEG"):
        pil_img = PILImage.fromarray(img)
        buffered = io.BytesIO()
        pil_img.save(buffered, format=format)
        b64_bytes = base64.b64encode(buffered.getvalue()).decode("utf-8")
        suffix = _PIL_FORMAT_TO_SUFFIX[format]
        return "data:image/" + suffix + ";base64," + b64_bytes

    async def show(self,
                   image: np.ndarray,
                   format: str = "JPEG",
                   set_size: bool = False):
        encoded = self.encode_image_bytes(image, format)
        self.props.image = encoded
        # self.image_str = encoded
        if set_size:
            ev = self.update_event(image=encoded,
                                   width=image.shape[1],
                                   height=image.shape[0])
        else:
            ev = self.update_event(image=encoded)
        await self.put_app_event(ev)

    async def show_raw(self, image_bytes: bytes, suffix: str):
        await self.put_app_event(self.show_raw_event(image_bytes, suffix))

    async def clear(self):
        await self.put_app_event(self.update_event(image=undefined))

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

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)


@dataclasses.dataclass
class ListItemTextProps(MUIComponentBaseProps):
    value: str = ""
    disableTypography: Union[bool, Undefined] = undefined
    inset: Union[bool, Undefined] = undefined


class ListItemText(MUIComponentBase[ListItemTextProps]):

    def __init__(self, init: str = "") -> None:
        super().__init__(UIType.ListItemText, ListItemTextProps)
        self.props.value = init

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
    muiColor: Union[_SEVERITY_TYPES, Undefined] = undefined
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
_TooltipPlacement: TypeAlias = Literal['top', 'right', 'left', 'bottom',
                                       'bottom-end', 'bottom-start',
                                       'left-end', 'left-start', 'right-end',
                                       'right-start', 'top-end', 'top-start']


@dataclasses.dataclass
class ButtonProps(MUIComponentBaseProps):
    name: str = ""
    muiColor: Union[_BtnGroupColor, Undefined] = undefined
    disabled: Union[bool, Undefined] = undefined
    fullWidth: Union[bool, Undefined] = undefined
    size: Union[Literal["small", "medium", "large"], Undefined] = undefined
    variant: Union[Literal["contained", "outlined", "text"],
                   Undefined] = undefined
    loading: Union[Undefined, bool] = undefined
    loadingIndicator: Union[Undefined, str] = undefined


class Button(MUIComponentBase[ButtonProps]):

    def __init__(self,
                 name: str,
                 callback: Optional[Callable[[], _CORO_NONE]] = None) -> None:
        super().__init__(UIType.Button, ButtonProps,
                         [FrontendEventType.Click.value])
        self.props.name = name
        if callback is not None:
            self.register_event_handler(FrontendEventType.Click.value,
                                        callback,
                                        simple_event=True)
        self.event_click = self._create_event_slot(FrontendEventType.Click)

    async def headless_click(self):
        return await self.put_loopback_ui_event(
            (FrontendEventType.Click.value, None))

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self,
                                           ev,
                                           sync_status_first=True,
                                           is_sync=is_sync)

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
    Refresh = 29
    Grid3x3 = 30
    Help = 31
    Visibility = 32
    Mic = 33
    PlayCircleOutline = 34
    DragIndicator = 35
    Cancel = 36
    Done = 37
    Preview = 38
    Build = 39
    VisibilityOff = 40
    ManageAccounts = 41
    AccountCircle = 42
    BugReport = 43
    Dashboard = 44
    DashboardCustomize = 45
    Check = 46



@dataclasses.dataclass
class IconBaseProps:
    icon: Union[int, str] = 0
    iconSize: Union[Literal["small", "medium", "large", "inherit"],
                    Undefined] = undefined
    iconFontSize: Union[ValueType, Undefined] = undefined

    @field_validator('icon')
    def svg_validator(cls, v):
        if isinstance(v, Undefined):
            return v
        if isinstance(v, int):
            return v
        # if not v.startswith('data:image/svg+xml;base64'):
        #     raise ValueError(
        #         'you must use mui.IconButton.encode_svg to encode svg string')
        return v


@dataclasses.dataclass
class IconProps(BasicProps, IconBaseProps):
    takeDragRef: Union[Undefined, bool] = undefined


class Icon(MUIComponentBase[IconProps]):

    def __init__(self, icon: Union[IconType, str]) -> None:
        super().__init__(UIType.Icon, IconProps)
        if isinstance(icon, IconType):
            self.props.icon = icon.value
        else:
            self.prop(icon=self.encode_svg(icon))

    @staticmethod
    def encode_svg(svg: str) -> str:
        # we don't use img to show svg for now
        return svg
        base64_bytes = base64.b64encode(svg.strip().encode('utf-8'))
        base64_string = base64_bytes.decode('utf-8')
        return f"data:image/svg+xml;base64,{base64_string}"

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class IconButtonProps(MUIComponentBaseProps, IconBaseProps):

    muiColor: Union[_BtnGroupColor, Undefined] = undefined
    disabled: Union[bool, Undefined] = undefined
    size: Union[Literal["small", "medium", "large"], Undefined] = undefined
    edge: Union[Literal["start", "end"], Undefined] = undefined

    tooltip: Union[str, Undefined] = undefined
    tooltipPlacement: Union[_TooltipPlacement, Undefined] = undefined
    tooltipMultiline: Union[bool, Undefined] = undefined

    progressColor: Union[_BtnGroupColor, Undefined] = undefined
    progressSize: Union[NumberType, Undefined] = undefined
    # if defined, will show a confirm dialog before executing the callback
    confirmMessage: Union[str, Undefined] = undefined
    confirmTitle: Union[str, Undefined] = undefined


class IconButton(MUIComponentBase[IconButtonProps]):

    def __init__(self,
                 icon: Union[str, IconType],
                 callback: Optional[Callable[[], _CORO_NONE]] = None) -> None:
        super().__init__(UIType.IconButton, IconButtonProps,
                         [FrontendEventType.Click.value])
        if isinstance(icon, IconType):
            self.props.icon = icon.value
        else:
            self.prop(icon=self.encode_svg(icon))
        if callback is not None:
            self.register_event_handler(FrontendEventType.Click.value,
                                        callback,
                                        simple_event=True)
        self.event_click = self._create_event_slot(FrontendEventType.Click)

    @staticmethod
    def encode_svg(svg: str) -> str:
        return Icon.encode_svg(svg)

    async def headless_click(self):
        return await self.put_loopback_ui_event(
            (FrontendEventType.Click.value, None))

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self,
                                           ev,
                                           sync_status_first=True,
                                           is_sync=is_sync)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class ListItemIconProps(MUIComponentBaseProps):
    icon: Union[int, str] = 0
    iconSize: Union[Literal["small", "medium", "large", "inherit"],
                    Undefined] = undefined
    iconFontSize: Union[ValueType, Undefined] = undefined


class ListItemIcon(MUIComponentBase[ListItemIconProps]):

    def __init__(self, icon: Union[IconType, str]) -> None:
        super().__init__(UIType.ListItemIcon, ListItemIconProps)
        if isinstance(icon, IconType):
            self.props.icon = icon.value
        else:
            self.prop(icon=Icon.encode_svg(icon))

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class DialogProps(MUIFlexBoxProps):
    open: bool = False
    title: Union[str, Undefined] = undefined
    fullScreen: Union[bool, Undefined] = undefined
    fullWidth: Union[bool, Undefined] = undefined
    maxWidth: Union[Literal['xs', 'sm', "md", "lg", "xl"],
                    Undefined] = undefined
    scroll: Union[Literal["body", "paper"], Undefined] = undefined
    includeFormControl: Union[bool, Undefined] = undefined
    cancelLabel: Union[str, Undefined] = undefined
    okLabel: Union[str, Undefined] = undefined


class Dialog(MUIContainerBase[DialogProps, MUIComponentType]):

    def __init__(
            self,
            children: LayoutType,
            callback: Optional[Callable[[bool], _CORO_NONE]] = None) -> None:
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}

        super().__init__(UIType.Dialog,
                         DialogProps,
                         _children=children,
                         allowed_events=[FrontendEventType.ModalClose.value])
        if callback is not None:
            self.register_event_handler(FrontendEventType.ModalClose.value,
                                        callback)

        self.event_modal_close = self._create_event_slot(
            FrontendEventType.ModalClose)

    async def set_open(self, open: bool):
        await self.send_and_wait(self.update_event(open=open))

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self,
                                           ev,
                                           sync_status_first=True,
                                           is_sync=is_sync)

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
            type: ValueType = FrontendEventType.ModalClose.value):
        # this only triggered when dialog closed, so we always set
        # open to false.
        self.props.open = False


@dataclasses.dataclass
class DrawerProps(MUIFlexBoxProps):
    open: bool = False
    anchor: Union[Literal["left", "top", "right", "bottom"],
                  Undefined] = undefined
    variant: Union[Literal["permanent", "persistent", "temporary"],
                   Undefined] = undefined
    keepMounted: Union[bool, Undefined] = undefined
    containerId: Union[str, Undefined] = undefined


class Drawer(MUIContainerBase[DrawerProps, MUIComponentType]):

    def __init__(
            self,
            children: LayoutType,
            callback: Optional[Callable[[bool], _CORO_NONE]] = None) -> None:
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}

        super().__init__(UIType.Drawer,
                         DrawerProps,
                         _children=children,
                         allowed_events=[FrontendEventType.ModalClose.value])
        if callback is not None:
            self.register_event_handler(FrontendEventType.ModalClose.value,
                                        callback)

    async def set_open(self, open: bool):
        await self.send_and_wait(self.update_event(open=open))

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self,
                                           ev,
                                           sync_status_first=True,
                                           is_sync=is_sync)

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
            type: ValueType = FrontendEventType.ModalClose.value):
        # this only triggered when dialog closed, so we always set
        # open to false.
        self.props.open = False


@dataclasses.dataclass
class ButtonGroupProps(MUIFlexBoxProps):
    orientation: Union[Literal["horizontal", "vertical"],
                       Undefined] = undefined
    muiColor: Union[_BtnGroupColor, Undefined] = undefined
    disabled: Union[bool, Undefined] = undefined
    fullWidth: Union[bool, Undefined] = undefined
    size: Union[Literal["small", "medium", "large"], Undefined] = undefined
    variant: Union[Literal["contained", "outlined", "text"],
                   Undefined] = undefined


class ButtonGroup(MUIContainerBase[ButtonGroupProps, Button]):

    def __init__(self, children: Union[List[Button], Dict[str,
                                                          Button]]) -> None:
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.ButtonGroup, ButtonGroupProps, children)
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
class ToggleButtonProps(MUIComponentBaseProps, IconBaseProps):
    icon: Union[int, str, Undefined] = undefined

    value: ValueType = ""
    name: str = ""
    selected: Union[Undefined, bool] = undefined
    tooltip: Union[str, Undefined] = undefined
    tooltipPlacement: Union[_TooltipPlacement, Undefined] = undefined
    muiColor: Union[_BtnGroupColor, Undefined] = undefined
    disabled: Union[bool, Undefined] = undefined
    fullWidth: Union[bool, Undefined] = undefined
    size: Union[Literal["small", "medium", "large"], Undefined] = undefined


class ToggleButton(MUIComponentBase[ToggleButtonProps]):
    """value is used in toggle group. for standalone toggle button, it isn't used,
    you can use it as name.
    """

    def __init__(
            self,
            value: ValueType = "",
            icon: Union[IconType, str, Undefined] = undefined,
            name: str = "",
            callback: Optional[Callable[[bool], _CORO_NONE]] = None) -> None:
        super().__init__(UIType.ToggleButton,
                         ToggleButtonProps,
                         allowed_events=[FrontendEventType.Change.value])
        if name == "" and isinstance(value, str) and value != "":
            name = value
        if isinstance(icon, Undefined):
            assert name != "", "if icon not provided, you must provide a valid name"
        elif isinstance(icon, IconType):
            self.props.icon = icon.value
        else:
            self.props.icon = Icon.encode_svg(icon)
        self.props.name = name
        self.props.value = value
        if callback is not None:
            self.register_event_handler(FrontendEventType.Change.value,
                                        callback)
        self.event_change = self._create_event_slot(FrontendEventType.Change)

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["selected"] = self.props.selected
        return res

    def state_change_callback(
            self,
            data: bool,
            type: ValueType = FrontendEventType.Change.value):
        self.props.selected = data

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)

    @property
    def checked(self):
        return self.props.selected is True

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
    muiColor: Union[_BtnGroupColor, Undefined] = undefined
    disabled: Union[bool, Undefined] = undefined
    fullWidth: Union[bool, Undefined] = undefined
    exclusive: Union[bool, Undefined] = undefined
    size: Union[Literal["small", "medium", "large"], Undefined] = undefined
    nameOrIcons: List[Tuple[bool, ValueType]] = dataclasses.field(
        default_factory=list)
    values: List[ValueType] = dataclasses.field(default_factory=list)
    iconSize: Union[Literal["small", "medium", "large"], Undefined] = undefined
    iconFontSize: Union[ValueType, Undefined] = undefined
    enforceValueSet: Union[bool, Undefined] = undefined


class ToggleButtonGroup(MUIContainerBase[ToggleButtonGroupProps,
                                         ToggleButton]):

    def __init__(
            self,
            children: Union[List[ToggleButton], Dict[str, ToggleButton]],
            exclusive: bool = True,
            callback: Optional[
                Callable[[Optional[Union[ValueType, List[ValueType]]]],
                         _CORO_NONE]] = None,
            value: Optional[Union[ValueType, List[ValueType]]] = None) -> None:
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.ToggleButtonGroup,
                         ToggleButtonGroupProps,
                         children,
                         allowed_events=[FrontendEventType.Change.value])
        values_set: Set[ValueType] = set()
        for v in children.values():
            assert isinstance(v,
                              ToggleButton), "all childs must be toggle button"
            if not isinstance(v.props.icon, Undefined):

                self.props.nameOrIcons.append((True, v.props.icon))
            else:
                self.props.nameOrIcons.append((False, v.props.name))
            values_set.add(v.props.value)
            self.props.values.append(v.props.value)
        assert len(values_set) == len(
            self.props.values), "values must be unique"
        self.props.value = value
        self.props.exclusive = exclusive
        self.callback = callback
        if not exclusive:
            if value is not None:
                assert isinstance(
                    value, list), "if not exclusive, value must be a list"
        else:
            if value is not None:
                assert not isinstance(
                    value, list), "if exclusive, value must not be a list"

        if callback is not None:
            self.register_event_handler(FrontendEventType.Change.value,
                                        callback)
        self.event_change = self._create_event_slot(FrontendEventType.Change)

    async def update_items(self,
                           btns: List[ToggleButton],
                           value: Optional[Union[ValueType,
                                                 List[ValueType]]] = None):
        name_or_icons = []
        values = []
        for v in btns:
            assert isinstance(v, ToggleButton), "all childs must be button"
            if not isinstance(v.props.icon, Undefined):
                self.props.nameOrIcons.append((True, v.props.icon))
            else:
                self.props.nameOrIcons.append((False, v.props.name))
            values.append(v.props.value)
        if value is None:
            assert self.props.value in values
            value = self.props.value
        else:
            assert value in values
        await self.send_and_wait(
            self.update_event(value=value,
                              nameOrIcons=name_or_icons,
                              values=values))

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

    async def set_value(self, value: Optional[Union[ValueType,
                                                    List[ValueType]]]):
        await self.send_and_wait(self.update_event(value=value))

    def state_change_callback(
            self,
            value: Union[ValueType, List[ValueType]],
            type: ValueType = FrontendEventType.Change.value):
        self.props.value = value

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)


@dataclasses.dataclass
class AccordionDetailsProps(MUIFlexBoxProps):
    pass


@dataclasses.dataclass
class AccordionSummaryProps(MUIFlexBoxProps):
    pass


class AccordionDetails(MUIContainerBase[AccordionDetailsProps,
                                        MUIComponentType]):

    def __init__(self, children: LayoutType) -> None:
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.AccordionDetail, AccordionDetailsProps,
                         children)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)


class AccordionSummary(MUIContainerBase[AccordionSummaryProps,
                                        MUIComponentType]):

    def __init__(self, children: LayoutType) -> None:
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.AccordionSummary, AccordionSummaryProps,
                         children)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)


@dataclasses.dataclass
class AccordionProps(MUIFlexBoxProps):
    disabled: Union[Undefined, bool] = undefined
    expanded: bool = False
    square: Union[Undefined, bool] = undefined
    disableGutters: Union[Undefined, bool] = undefined


class Accordion(MUIContainerBase[AccordionProps, Union[AccordionDetails,
                                                       AccordionSummary]]):

    def __init__(self,
                 summary: AccordionSummary,
                 details: Optional[AccordionDetails] = None) -> None:
        children: Dict[str, Union[AccordionDetails, AccordionSummary]] = {
            "summary": summary
        }
        if details is not None:
            children["details"] = details
        for v in children.values():
            assert isinstance(
                v, (AccordionSummary,
                    AccordionDetails)), "all childs must be summary or detail"
        super().__init__(UIType.Accordion, AccordionProps, children)

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

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class ListItemButtonProps(MUIFlexBoxProps):
    alignItems: Union[Undefined, Literal["center", "flex-start"]] = undefined
    dense: Union[Undefined, bool] = undefined
    disabled: Union[Undefined, bool] = undefined
    disableGutters: Union[Undefined, bool] = undefined
    divider: Union[Undefined, bool] = undefined
    autoFocus: Union[Undefined, bool] = undefined
    selected: Union[Undefined, bool] = undefined


class ListItemButton(MUIContainerBase[ListItemButtonProps, MUIComponentType]):

    def __init__(self,
                 children: LayoutType,
                 callback: Optional[Callable[[], _CORO_NONE]] = None) -> None:
        if children is not None and isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}

        super().__init__(UIType.ListItemButton,
                         ListItemButtonProps,
                         children,
                         allowed_events=[FrontendEventType.Click.value])
        if callback is not None:
            self.register_event_handler(FrontendEventType.Click.value,
                                        callback)
        self.event_click = self._create_event_slot(FrontendEventType.Click)

    async def headless_click(self):
        return await self.put_loopback_ui_event(
            (FrontendEventType.Click.value, None))

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self,
                                           ev,
                                           sync_status_first=False,
                                           is_sync=is_sync)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class FlexBox(MUIContainerBase[MUIFlexBoxWithDndProps, MUIComponentType]):

    def __init__(self,
                 children: Optional[LayoutType] = None,
                 base_type: UIType = UIType.FlexBox,
                 uid: Optional[UniqueTreeId] = None,
                 app_comp_core: Optional[AppComponentCore] = None,
                 wrapped_obj: Optional[Any] = None) -> None:
        if children is not None and isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(base_type,
                         MUIFlexBoxWithDndProps,
                         children,
                         uid=uid,
                         app_comp_core=app_comp_core,
                         allowed_events=[
                             FrontendEventType.Drop.value,
                             FrontendEventType.DragCollect.value
                         ] + list(ALL_POINTER_EVENTS))
        self._wrapped_obj = wrapped_obj
        self.event_drop = self._create_event_slot(FrontendEventType.Drop)
        self.event_click = self._create_event_slot(FrontendEventType.Click)
        self.event_double_click = self._create_event_slot(
            FrontendEventType.DoubleClick)
        self.event_pointer_enter = self._create_event_slot(
            FrontendEventType.Enter)
        self.event_pointer_leave = self._create_event_slot(
            FrontendEventType.Leave)
        self.event_pointer_down = self._create_event_slot(
            FrontendEventType.Down)
        self.event_pointer_up = self._create_event_slot(FrontendEventType.Up)
        self.event_pointer_move = self._create_event_slot(
            FrontendEventType.Move)
        self.event_pointer_over = self._create_event_slot(
            FrontendEventType.Over)
        self.event_pointer_out = self._create_event_slot(FrontendEventType.Out)
        self.event_pointer_context_menu = self._create_event_slot(
            FrontendEventType.ContextMenu)
        self.event_drag_collect = self._create_event_slot(
            FrontendEventType.DragCollect)

    def as_drag_handle(self):
        self.props.takeDragRef = True
        return self

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    def get_wrapped_obj(self):
        return self._wrapped_obj

    def set_wrapped_obj(self, wrapped_obj: Any):
        self._wrapped_obj = wrapped_obj
        self._flow_comp_def_path = _get_obj_def_path(wrapped_obj)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    def get_special_methods(self, reload_mgr: AppReloadManager):
        user_obj = self._get_user_object()
        metas = reload_mgr.query_type_method_meta(type(user_obj),
                                                  no_code=True,
                                                  include_base=True)
        res = FlowSpecialMethods(metas)
        res.bind(user_obj)
        return res

    def _get_user_object(self):
        if self._wrapped_obj is not None:
            return self._wrapped_obj
        return self

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self,
                                           ev,
                                           sync_status_first=False,
                                           is_sync=is_sync)


class DragHandleFlexBox(FlexBox):

    def __init__(
        self,
        children: Optional[LayoutType] = None,
    ) -> None:
        if children is not None and isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(children)


@dataclasses.dataclass
class MUIListProps(MUIFlexBoxProps):
    subheader: str = ""
    disablePadding: Union[Undefined, bool] = undefined
    dense: Union[Undefined, bool] = undefined


class MUIList(MUIContainerBase[MUIListProps, MUIComponentType]):

    def __init__(self,
                 children: Optional[LayoutType] = None,
                 subheader: str = "") -> None:
        if children is not None and isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.MUIList, MUIListProps, _children=children)
        self.props.subheader = subheader

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


def VBox(layout: LayoutType, wrap: bool = False):
    res = FlexBox(children=layout)
    res.prop(flexFlow="column wrap" if wrap else "column nowrap")
    return res


def HBox(layout: LayoutType, wrap: bool = False):
    res = FlexBox(children=layout)
    res.prop(flexFlow="row wrap" if wrap else "row nowrap")
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
        callback: Optional[Callable[[str], Coroutine[None, None,
                                                     None]]] = None,
        row: bool = True,
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
        self.event_change = self._create_event_slot(FrontendEventType.Change)

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
        return await self.put_loopback_ui_event(
            (FrontendEventType.Change.value, self.props.names[index]))

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)

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
class InputBaseProps(MUIComponentBaseProps):
    multiline: Union[bool, Undefined] = undefined
    value: Union[Undefined, str] = undefined
    defaultValue: Union[Undefined, str] = undefined
    disabled: Union[bool, Undefined] = undefined
    error: Union[bool, Undefined] = undefined
    fullWidth: Union[bool, Undefined] = undefined
    rows: Union[NumberType, str, Undefined] = undefined
    type: Union[Undefined, _HTMLInputType] = undefined
    debounce: Union[Undefined, NumberType] = undefined
    required: Union[Undefined, bool] = undefined


T_input_base_props = TypeVar("T_input_base_props", bound=InputBaseProps)


class _InputBaseComponent(MUIComponentBase[T_input_base_props]):

    def __init__(
        self,
        callback: Optional[Callable[[str], _CORO_NONE]],
        type: UIType,
        prop_cls: Type[T_input_base_props],
        allowed_events: Optional[Iterable[EventDataType]] = None,
    ) -> None:
        super().__init__(type, prop_cls, allowed_events)
        self.callback = callback
        if callback is not None:
            self.register_event_handler(FrontendEventType.Change.value,
                                        callback)
        self.event_change = self._create_event_slot(FrontendEventType.Change)

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
            {self._flow_uid_encoded: (FrontendEventType.Change.value, content)})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    def json(self):
        assert not isinstance(self.props.value, Undefined)
        return json.loads(self.props.value)

    def float(self):
        assert not isinstance(self.props.value, Undefined)
        return float(self.props.value)

    def int(self):
        assert not isinstance(self.props.value, Undefined)
        return int(self.props.value)

    def str(self):
        assert not isinstance(self.props.value, Undefined)
        return str(self.props.value)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        # for fully controlled components, we need to sync the state after the
        # backend state chagne.
        sync_state_after_change = isinstance(self.props.debounce, Undefined)
        return await handle_standard_event(
            self,
            ev,
            is_sync=is_sync,
            sync_status_first=False,
            sync_state_after_change=sync_state_after_change)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class TextFieldProps(InputBaseProps):
    label: str = ""
    muiColor: Union[_StdColorNoDefault, Undefined] = undefined
    size: Union[Undefined, Literal["small", "medium"]] = undefined
    muiMargin: Union[Undefined, Literal["dense", "none", "normal"]] = "dense"
    variant: Union[Undefined, Literal["filled", "outlined",
                                      "standard"]] = undefined


class TextField(_InputBaseComponent[TextFieldProps]):

    def __init__(self,
                 label: str,
                 callback: Optional[Callable[[str], _CORO_NONE]] = None,
                 init: Union[Undefined, str] = ""):
        super().__init__(callback, UIType.TextField, TextFieldProps,
                         [FrontendEventType.Change.value])
        self.props.label = label
        self.props.value = init

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class InputProps(InputBaseProps):
    placeholder: str = ""
    muiColor: Union[Literal["primary", "secondary"], Undefined] = undefined
    muiMargin: Union[Undefined, Literal["dense", "none"]] = "dense"
    disableUnderline: Union[bool, Undefined] = undefined


class Input(_InputBaseComponent[InputProps]):

    def __init__(self,
                 placeholder: str,
                 callback: Optional[Callable[[str], _CORO_NONE]] = None,
                 init: Union[Undefined, str] = "") -> None:
        super().__init__(callback, UIType.Input, InputProps,
                         [FrontendEventType.Change.value])
        self.props.placeholder = placeholder
        self.props.value = init

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class MonacoEditorProps(MUIComponentBaseProps):
    value: Union[str, Undefined] = undefined
    language: Union[str, Undefined] = undefined
    path: Union[str, Undefined] = undefined
    debounce: Union[NumberType, Undefined] = undefined
    lspPort: Union[int, Undefined] = undefined


class _MonacoEditorControlType(enum.IntEnum):
    SetLineNumber = 0
    Save = 1


class MonacoEditor(MUIComponentBase[MonacoEditorProps]):

    def __init__(self, value: str, language: str, path: str) -> None:
        all_evs = [
            FrontendEventType.Change.value,
            FrontendEventType.EditorQueryState.value,
            FrontendEventType.EditorSave.value,
            FrontendEventType.EditorSaveState.value,
            FrontendEventType.EditorReady.value,
        ]
        super().__init__(UIType.MonacoEditor, MonacoEditorProps, all_evs)
        self.props.language = language
        self.props.path = path
        self.props.value = value
        self.view_state = None

        self.register_event_handler(FrontendEventType.EditorSaveState.value,
                                    self._default_on_save_state)
        self.register_event_handler(FrontendEventType.EditorQueryState.value,
                                    self._default_on_query_state)

        self.event_change = self._create_event_slot(FrontendEventType.Change)
        self.event_editor_save = self._create_event_slot(
            FrontendEventType.EditorSave)
        self.event_editor_ready = self._create_event_slot(
            FrontendEventType.EditorReady)

    def state_change_callback(
            self,
            data: Tuple[str, int, Any],
            type: ValueType = FrontendEventType.Change.value):
        self.value = data[0]

    def _default_on_save_state(self, state):
        self.view_state = state

    def _default_on_query_state(self):
        return self.view_state

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def set_line_number(self, lineno: int):
        ev = self.create_comp_event({
            "type":
            int(_MonacoEditorControlType.SetLineNumber),
            "value":
            lineno,
        })
        await self.send_and_wait(ev)

    async def save(self):
        ev = self.create_comp_event({
            "type": int(_MonacoEditorControlType.Save),
        })
        await self.send_and_wait(ev)


@dataclasses.dataclass
class SwitchProps(MUIComponentBaseProps):
    label: Union[str, Undefined] = undefined
    checked: bool = False
    disabled: Union[bool, Undefined] = undefined
    size: Union[Literal["small", "medium"], Undefined] = undefined
    muiColor: Union[_BtnGroupColor, Undefined] = undefined
    labelPlacement: Union[Literal["top", "start", "bottom", "end"],
                          Undefined] = undefined


class SwitchBase(MUIComponentBase[SwitchProps]):

    def __init__(self,
                 label: Union[str, Undefined],
                 base_type: UIType,
                 callback: Optional[Callable[[bool], _CORO_NONE]] = None,
                 init_value: bool = False) -> None:
        super().__init__(base_type, SwitchProps,
                         [FrontendEventType.Change.value])
        if not isinstance(label, Undefined):
            self.props.label = label
        self.props.checked = init_value
        if callback is not None:
            self.register_event_handler(FrontendEventType.Change.value,
                                        callback)
        self.event_change = self._create_event_slot(FrontendEventType.Change)

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["checked"] = self.props.checked
        return res

    def bind_obj_prop(self, obj: Any, prop: str):
        self.prop(checked=getattr(obj, prop))
        self.event_change.on(lambda checked: setattr(obj, prop, checked))
        return self

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
            {self._flow_uid_encoded: (FrontendEventType.Change.value, checked)})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    def __bool__(self):
        return self.props.checked

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)

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
                 label: Union[str, Undefined] = undefined,
                 callback: Optional[Callable[[bool], _CORO_NONE]] = None,
                 init_value: bool = False) -> None:
        super().__init__(label, UIType.Switch, callback, init_value)


class Checkbox(SwitchBase):

    def __init__(self,
                 label: Union[str, Undefined] = undefined,
                 callback: Optional[Callable[[bool], _CORO_NONE]] = None,
                 init_value: bool = False) -> None:
        super().__init__(label, UIType.Checkbox, callback, init_value)


# @dataclasses.dataclass
# class SelectPropsBase(MUIComponentBaseProps):
#     size: Union[Undefined, Literal["small", "medium"]] = undefined
#     muiMargin: Union[Undefined, Literal["dense", "none", "normal"]] = undefined
#     inputVariant: Union[Undefined, Literal["filled", "outlined",
#                                       "standard"]] = undefined
#     label: str = ""


# TODO refine this
@dataclasses.dataclass
class SelectBaseProps:
    size: Union[Undefined, Literal["small", "medium"]] = undefined
    muiMargin: Union[Undefined, Literal["dense", "none", "normal"]] = undefined
    variant: Union[Undefined, Literal["filled", "outlined",
                                      "standard"]] = undefined
    itemVariant: Union[Undefined, Literal["checkbox", "none"]] = undefined
    label: str = ""


@dataclasses.dataclass
class SelectProps(MUIComponentBaseProps, SelectBaseProps):
    items: List[Tuple[str,
                      ValueType]] = dataclasses.field(default_factory=list)
    value: ValueType = ""
    autoWidth: Union[Undefined, bool] = undefined


class Select(MUIComponentBase[SelectProps]):

    def __init__(self,
                 label: str,
                 items: List[Tuple[str, ValueType]],
                 callback: Optional[Callable[[ValueType], _CORO_NONE]] = None,
                 init_value: Optional[ValueType] = None) -> None:
        super().__init__(UIType.Select, SelectProps,
                         [FrontendEventType.Change.value])
        if init_value is not None:
            assert init_value in [x[1] for x in items]

        self.props.label = label
        self.callback = callback
        # assert len(items) > 0
        self.props.items = items
        # item value must implement eq/ne
        self.props.value = ""
        if init_value is not None:
            self.props.value = init_value
        self.props.size = "small"
        if callback is not None:
            self.register_event_handler(FrontendEventType.Change.value,
                                        callback)
        self.event_change = self._create_event_slot(FrontendEventType.Change)

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
            {self._flow_uid_encoded: (FrontendEventType.Change.value, value)})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class MultipleSelectProps(MUIComponentBaseProps, SelectBaseProps):
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
        self.event_change = self._create_event_slot(FrontendEventType.Change)

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
            {self._flow_uid_encoded: (FrontendEventType.Change.value, values)})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class AutocompletePropsBase(MUIComponentBaseProps, SelectBaseProps):
    # input_value: str = ""
    options: List[Dict[str, Any]] = dataclasses.field(default_factory=list)

    disableClearable: Union[Undefined, bool] = undefined
    disableCloseOnSelect: Union[Undefined, bool] = undefined
    clearOnEscape: Union[Undefined, bool] = undefined
    includeInputInList: Union[Undefined, bool] = undefined
    disableListWrap: Union[Undefined, bool] = undefined
    openOnFocus: Union[Undefined, bool] = undefined
    autoHighlight: Union[Undefined, bool] = undefined
    autoSelect: Union[Undefined, bool] = undefined
    disabled: Union[Undefined, bool] = undefined
    disablePortal: Union[Undefined, bool] = undefined
    blurOnSelect: Union[Undefined, bool] = undefined
    clearOnBlur: Union[Undefined, bool] = undefined
    selectOnFocus: Union[Undefined, bool] = undefined
    readOnly: Union[Undefined, bool] = undefined
    freeSolo: Union[Undefined, bool] = undefined
    handleHomeEndKeys: Union[Undefined, bool] = undefined
    groupByKey: Union[Undefined, str] = undefined
    limitTags: Union[Undefined, int] = undefined
    addOption: Union[Undefined, bool] = undefined


@dataclasses.dataclass
class AutocompleteProps(AutocompletePropsBase):
    value: Optional[Dict[str, Any]] = None


class Autocomplete(MUIComponentBase[AutocompleteProps]):

    class CreatableAutocompleteType(TypedDict):
        selectOnFocus: bool
        clearOnBlur: bool
        handleHomeEndKeys: bool
        freeSolo: bool
        addOption: bool

    # TODO should we force autocomplete use dataclass?
    def __init__(
        self,
        label: str,
        options: List[Dict[str, Any]],
        callback: Optional[Callable[[Dict[str, Any]],
                                    _CORO_NONE]] = None) -> None:
        super().__init__(UIType.AutoComplete, AutocompleteProps, [
            FrontendEventType.Change.value,
            FrontendEventType.SelectNewItem.value
        ])
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
        self.event_change = self._create_event_slot(FrontendEventType.Change)
        self.event_select_new_item = self._create_event_slot(
            FrontendEventType.SelectNewItem)

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
                             selected: Optional[int]):
        # await self.send_and_wait(
        #     self.create_update_event({
        #         "options": options,
        #         "value": options[selected]
        #     }))
        value: Optional[Dict[str, Any]] = None
        if selected is not None:
            value = options[selected]
        await self.send_and_wait(
            self.update_event(options=options, value=value))

        self.props.options = options
        self.props.value = value

    async def update_value(self, value: Optional[Dict[str, Any]]):
        await self.put_app_event(self.create_update_event({"value": value}))
        self.props.value = value

    def update_value_no_sync(self, value: Optional[Dict[str, Any]]):
        self.props.value = value

    def state_change_callback(
            self,
            value: Union[str, Optional[Dict[str, Any]]],
            type: ValueType = FrontendEventType.Change.value):
        # TODO handle str
        if type == FrontendEventType.Change.value:
            if value is not None:
                assert isinstance(value, dict)
            self.props.value = value
            # add new option
        # else:
        #     assert isinstance(value, str)
        #     print("self.props.input_value", value, type)
        #     self.props.input_value = value

    async def headless_select(self, value: ValueType):
        uiev = UIEvent(
            {self._flow_uid_encoded: (FrontendEventType.Change.value, value)})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    async def handle_event(self, data: Event, is_sync: bool = False):
        return await handle_standard_event(self, data, is_sync=is_sync)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    @staticmethod
    def get_creatable_option() -> "Autocomplete.CreatableAutocompleteType":
        return {
            "selectOnFocus": True,
            "clearOnBlur": True,
            "handleHomeEndKeys": True,
            "freeSolo": True,
            "addOption": True,
        }


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
        self.event_change = self._create_event_slot(FrontendEventType.Change)

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
            {self._flow_uid_encoded: (FrontendEventType.Change.value, value)})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    async def handle_event(self, data: Event, is_sync: bool = False):
        return await handle_standard_event(self, data, is_sync=is_sync)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class SliderBaseProps(MUIComponentBaseProps):
    label: Union[Undefined, str] = undefined
    ranges: Tuple[NumberType, NumberType, NumberType] = (0, 1, 0)
    vertical: Union[Undefined, bool] = undefined
    valueInput: Union[Undefined, bool] = undefined
    size: Union[Undefined, Literal["small", "medium"]] = undefined
    muiColor: Union[Undefined, Literal["primary", "secondary"]] = undefined


@dataclasses.dataclass
class SliderProps(SliderBaseProps):
    value: Union[Undefined, NumberType] = undefined
    defaultValue: Union[Undefined, NumberType] = undefined
    marks: Union[Undefined, bool] = undefined

class Slider(MUIComponentBase[SliderProps]):

    def __init__(self,
                 begin: NumberType,
                 end: NumberType,
                 step: Optional[NumberType] = None,
                 callback: Optional[Callable[[NumberType], _CORO_NONE]] = None,
                 label: Union[Undefined, str] = undefined,
                 init_value: Optional[NumberType] = None) -> None:
        super().__init__(UIType.Slider, SliderProps,
                         [FrontendEventType.Change.value])
        if isinstance(begin, int) and isinstance(end, int):
            if step is None:
                step = 1
        assert step is not None, "step must be specified for float type"
        self.props.label = label
        self.callback = callback
        assert end >= begin  #  and step <= end - begin
        self.props.ranges = (begin, end, step)
        if init_value is None:
            init_value = begin
        self.props.value = init_value
        if callback is not None:
            self.register_event_handler(FrontendEventType.Change.value,
                                        callback)
        self.event_change = self._create_event_slot(FrontendEventType.Change)

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

    async def update_ranges(self,
                            begin: NumberType,
                            end: NumberType,
                            step: Optional[NumberType] = None):
        if step is None:
            step = self.props.ranges[2]
        self.props.ranges = (begin, end, step)
        assert end >= begin and step <= end - begin
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
            {self._flow_uid_encoded: (FrontendEventType.Change.value, value)})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class RangeSliderProps(SliderBaseProps):
    value: Union[Undefined, Tuple[NumberType, NumberType]] = undefined
    defaultValue: Union[Undefined, Tuple[NumberType, NumberType]] = undefined


class RangeSlider(MUIComponentBase[RangeSliderProps]):

    def __init__(
            self,
            begin: NumberType,
            end: NumberType,
            step: Optional[NumberType] = None,
            callback: Optional[Callable[[NumberType], _CORO_NONE]] = None,
            label: Union[Undefined, str] = undefined,
            init_value: Optional[Tuple[NumberType,
                                       NumberType]] = None) -> None:
        super().__init__(UIType.Slider, RangeSliderProps,
                         [FrontendEventType.Change.value])
        if isinstance(begin, int) and isinstance(end, int):
            if step is None:
                step = 1
        assert step is not None, "step must be specified for float type"
        self.props.label = label
        self.callback = callback
        assert end >= begin  #  and step <= end - begin
        self.props.ranges = (begin, end, step)
        if init_value is not None:
            self.props.value = init_value
            assert init_value[0] <= init_value[1] and init_value[
                0] >= begin and init_value[1] <= end
        else:
            self.props.value = (begin, begin)

        if callback is not None:
            self.register_event_handler(FrontendEventType.Change.value,
                                        callback)
        self.event_change = self._create_event_slot(FrontendEventType.Change)

    @property
    def value(self):
        return self.props.value

    def _validate_range_value(self, value: Tuple[NumberType, NumberType]):
        return (value[0] >= self.props.ranges[0]
                and value[0] <= self.props.ranges[1]
                and value[1] >= self.props.ranges[0]
                and value[1] <= self.props.ranges[1] and value[0] <= value[1])

    def validate_props(self, props: Dict[str, Any]):
        if "value" in props:
            value = props["value"]

            return self._validate_range_value(value)
        return False

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["value"] = self.props.value
        return res

    async def update_ranges(self,
                            begin: NumberType,
                            end: NumberType,
                            step: Optional[NumberType] = None):
        if step is None:
            step = self.props.ranges[2]
        self.props.ranges = (begin, end, step)
        assert end >= begin and step <= end - begin
        self.props.value = (begin, begin)
        await self.put_app_event(
            self.create_update_event({
                "ranges": (begin, end, step),
                "value": self.props.value
            }))

    async def update_value(self, value: Tuple[NumberType, NumberType]):
        assert self._validate_range_value(value)
        await self.put_app_event(self.create_update_event({"value": value}))
        self.props.value = value

    def state_change_callback(
            self,
            value: Tuple[NumberType, NumberType],
            type: ValueType = FrontendEventType.Change.value):
        self.props.value = value

    async def headless_change(self, value: NumberType):
        uiev = UIEvent(
            {self._flow_uid_encoded: (FrontendEventType.Change.value, value)})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class BlenderSliderProps(MUIComponentBaseProps):
    ranges: Tuple[NumberType, NumberType, NumberType] = (0, 1, 0)
    value: Union[Undefined, NumberType] = undefined
    defaultValue: Union[Undefined, NumberType] = undefined
    dragSpeed: Union[Undefined, NumberType] = undefined
    debounce: Union[Undefined, NumberType] = undefined
    infSlider: Union[Undefined, bool] = undefined
    showControlButton: Union[Undefined, bool] = undefined
    color: Union[Undefined, str] = undefined
    hoverColor: Union[Undefined, str] = undefined
    clickColor: Union[Undefined, str] = undefined
    indicatorColor: Union[Undefined, str] = undefined
    iconColor: Union[Undefined, str] = undefined
    fractionDigits: Union[Undefined, int] = undefined
    isInteger: Union[Undefined, bool] = undefined


class BlenderSlider(MUIComponentBase[BlenderSliderProps]):

    def __init__(self,
                 begin: NumberType,
                 end: NumberType,
                 step: Optional[NumberType] = None,
                 callback: Optional[Callable[[NumberType], _CORO_NONE]] = None,
                 init_value: Optional[NumberType] = None) -> None:
        super().__init__(UIType.BlenderSlider, BlenderSliderProps,
                         [FrontendEventType.Change.value])
        if isinstance(begin, int) and isinstance(end, int):
            if step is None:
                step = 1
        assert step is not None, "step must be specified for float type"
        self.callback = callback
        assert end >= begin  #  and step <= end - begin
        self.props.ranges = (begin, end, step)
        if init_value is None:
            init_value = begin
        self.props.value = init_value
        if callback is not None:
            self.register_event_handler(FrontendEventType.Change.value,
                                        callback)
        self.event_change = self._create_event_slot(FrontendEventType.Change)

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

    async def update_ranges(self,
                            begin: NumberType,
                            end: NumberType,
                            step: Optional[NumberType] = None):
        if step is None:
            step = self.props.ranges[2]
        self.props.ranges = (begin, end, step)
        assert end >= begin and step <= end - begin
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
            {self._flow_uid_encoded: (FrontendEventType.Change.value, value)})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)

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
    taskStatus: Union[Undefined, int] = undefined


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
        # self.loop_callbcak = loop_callbcak
        self.__callback_key = "list_slider_ev_handler"
        if loop_callbcak is not None:
            self.register_event_handler(self.__callback_key,
                                        loop_callbcak,
                                        backend_only=True)

        self.props.progresses = [0.0]
        self.stack_count = 0
        self.pause_event = asyncio.Event()
        self.pause_event.set()
        self.update_period = update_period
        self._raw_update = raw_update

    async def task_loop(self,
                        it: Union[Iterable[_T], AsyncIterable[_T]],
                        total: int = -1,
                        start: int = -1) -> AsyncGenerator[_T, None]:
        if self._raw_update:
            raise ValueError(
                "when raw update enabled, you can't use this function")
        if isinstance(it, list):
            total = len(it)
        try:
            cnt = 0 if start < 0 else start
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
            self.update_event(taskStatus=UIRunStatus.Stop.value,
                              progresses=[0]))

    async def headless_run(self):
        uiev = UIEvent({
            self._flow_uid_encoded:
            (FrontendEventType.Change.value, TaskLoopEvent.Start.value)
        })
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    async def headless_stop(self):
        uiev = UIEvent({
            self._flow_uid_encoded:
            (FrontendEventType.Change.value, TaskLoopEvent.Stop.value)
        })
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    async def handle_event(self, ev: Event, is_sync: bool = False):
        if self._raw_update:
            return await handle_standard_event(self, ev, is_sync=is_sync)
        data = ev.data
        if data == TaskLoopEvent.Start.value:
            if self.props.status == UIRunStatus.Stop.value:
                handlers = self.get_event_handlers(self.__callback_key)
                if handlers is not None:
                    self._task = asyncio.create_task(
                        self.run_callbacks(
                            handlers.get_bind_event_handlers_noarg(ev),
                            True,
                            sync_status_first=True,
                            change_status=True))
                    self.props.status = UIRunStatus.Running.value
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
                self.props.status = UIRunStatus.Stop.value
            elif self.props.status == UIRunStatus.Pause.value:
                self.pause_event.set()
                await cancel_task(self._task)
                self.props.status = UIRunStatus.Stop.value
            else:
                print("IGNORE TaskLoop EVENT", self.props.status)
        else:
            raise NotImplementedError
        await self.sync_status(True)

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
            {self._flow_uid_encoded: (FrontendEventType.Change.value, ev.value)})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    async def handle_event(self, data: Event, is_sync: bool = False):
        return await handle_standard_event(self, data, is_sync=is_sync)

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
    gutterBottom: Union[bool, Undefined] = undefined
    noWrap: Union[bool, Undefined] = undefined
    variant: Union[_TypographyVarient, Undefined] = undefined
    paragraph: Union[bool, Undefined] = undefined
    muiColor: Union[_StdColorNoDefault, Undefined] = undefined
    value: Union[str, NumberType] = ""
    # if value is number, will apply this to number
    # we check fixed first, then precision
    fixedDigits: Union[Undefined, int] = undefined
    precisionDigits: Union[Undefined, int] = undefined
    # enableTooltipWhenOverflow?: boolean
    # tooltipPlacement?: TooltipProps['placement']
    # tooltipEnterDelay?: TooltipProps['enterDelay']
    # tooltipEnterNextDelay?: TooltipProps['enterNextDelay']
    # tooltipLeaveDelay?: TooltipProps['leaveDelay']
    enableTooltipWhenOverflow: Union[Undefined, bool] = undefined
    placement: Union[_TooltipPlacement, Undefined] = undefined
    tooltipEnterDelay: Union[Undefined, NumberType] = undefined
    tooltipEnterNextDelay: Union[Undefined, NumberType] = undefined
    tooltipLeaveDelay: Union[Undefined, NumberType] = undefined


@dataclasses.dataclass
class LinkProps(MUIComponentBaseProps):
    value: str = ""
    href: Union[Undefined, str] = undefined
    underline: Union[Undefined, Literal["always", "hover", "none"]] = undefined
    variant: Union[Undefined, _TypographyVarient] = undefined
    muiColor: Union[Undefined, _StdColorNoDefault] = undefined
    rel: Union[Undefined, str] = undefined
    target: Union[Undefined, str] = undefined
    download: Union[Undefined, str] = undefined
    isTensoRPCUri: Union[Undefined, bool] = undefined


class Link(MUIComponentBase[LinkProps]):

    def __init__(self, value: str, href: str = "#") -> None:
        super().__init__(UIType.Link, LinkProps)
        self.props.value = value
        self.props.href = href

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    @classmethod
    def safe_download_link(cls, value: str, href: str):
        link = cls(value, href)
        link.props.rel = "noopener noreferrer"
        link.props.target = "_blank"
        return link

    @classmethod
    def app_download_link(cls, value: str, key: str):
        link = cls(value, cls.encode_app_link(key))
        link.props.rel = "noopener noreferrer"
        link.props.target = "_blank"
        link.props.isTensoRPCUri = True
        return link

    @staticmethod
    def encode_app_link(key: str):
        import urllib.parse
        master_meta = MasterMeta()
        params = {
            "nodeUid": f"{master_meta.graph_id}@{master_meta.node_id}",
            "key": key
        }
        return urllib.parse.urlencode(params, doseq=True)


class Typography(MUIComponentBase[TypographyProps]):

    def __init__(self, init: Union[str, NumberType] = "") -> None:
        super().__init__(UIType.Typography, TypographyProps)
        self.props.value = init

    async def write(self, content: Union[str, NumberType]):
        assert isinstance(content, (str, int, float))
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
class MarkdownProps(MUIComponentBaseProps):
    katex: Union[bool, Undefined] = undefined
    codeHighlight: Union[bool, Undefined] = undefined
    emoji: Union[bool, Undefined] = undefined
    value: str = ""


class Markdown(MUIComponentBase[MarkdownProps]):
    """markdown with color support, gfm, latex math,
    code highlight and :emoji: support. note that only colored
    text and gfm are enabled by default, other features need to be
    enabled explicitly.

    Colored text: using the syntax :color[text to be colored], where color needs to be replaced with any of the color string in tensorpc.flow.flowapp.colors (e.g. :green[green text]).

    LaTeX expressions: by wrapping them in "$" or "$$" (the "$$" must be on their own lines). Supported LaTeX functions are listed at https://katex.org/docs/supported.html.

    Emoji: :EMOJICODE:. see https://github.com/ikatyang/emoji-cheat-sheet

    Examples:
        ":green[$\\sqrt{x^2+y^2}=1$] is a Pythagorean identity. :+1:"
        contains a colored text, a latex expression and a emoji.
    """

    def __init__(self, init: str = "") -> None:
        super().__init__(UIType.Markdown, MarkdownProps)
        self.props.value = init

    async def write(self, content: str):
        assert isinstance(content, str)
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

    def __init__(self, children: Optional[LayoutType] = None) -> None:
        if children is not None and isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.Paper, PaperProps, children)

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
    muiMargin: Union[Undefined, Literal["dense", "none", "normal"]] = undefined


class FormControl(MUIContainerBase[FormControlProps, MUIComponentType]):

    def __init__(self, children: Dict[str, MUIComponentType]) -> None:
        super().__init__(UIType.Paper, FormControlProps, children)

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
    triggered: Union[bool, Undefined] = undefined

    orientation: Union[Literal["horizontal", "vertical"],
                       Undefined] = undefined
    timeout: Union[NumberType, Undefined, Literal["auto"]] = undefined
    collapsedSize: Union[NumberType, Undefined] = undefined
    unmountOnExit: Union[bool, Undefined] = undefined


class Collapse(MUIContainerBase[CollapseProps, MUIComponentType]):

    def __init__(self, children: Optional[LayoutType] = None) -> None:
        if children is not None and isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.Collapse, CollapseProps, children)

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
#                  inited: bool = False) -> None:
#         super().__init__(UIType.Accordion, AccordionProps,
#                          children, inited)

#     @property
#     def prop(self):
#         propcls = self.propcls
#         return self._prop_base(propcls, self)


@dataclasses.dataclass
class ChipProps(MUIComponentBaseProps, IconBaseProps):
    muiColor: Union[_StdColor, str, Undefined] = undefined
    clickable: Union[bool, Undefined] = undefined
    deletable: Union[bool, Undefined] = undefined
    size: Union[Literal["small", "medium"], Undefined] = undefined
    variant: Union[Literal["filled", "outlined"], Undefined] = undefined
    label: str = ""
    icon: Union[IconType, Undefined] = undefined
    deleteIcon: Union[IconType, Undefined] = undefined


class Chip(MUIComponentBase[ChipProps]):

    def __init__(
        self,
        label: Optional[str] = None,
        callback: Optional[Callable[[], _CORO_NONE]] = None,
        delete_callback: Optional[Callable[[], _CORO_NONE]] = None,
    ) -> None:
        super().__init__(
            UIType.Chip, ChipProps,
            [FrontendEventType.Click.value, FrontendEventType.Delete.value])
        if label is not None:
            self.props.label = label
        self.callback = callback
        self.delete_callback = delete_callback
        if callback is not None:
            self.register_event_handler(FrontendEventType.Click.value,
                                        callback)
        if delete_callback is not None:
            self.register_event_handler(FrontendEventType.Delete.value,
                                        delete_callback)
        self.event_click = self._create_event_slot(FrontendEventType.Click)
        self.event_delete = self._create_event_slot(FrontendEventType.Delete)

    def to_dict(self):
        res = super().to_dict()
        res["label"] = self.props.label
        return res

    async def headless_click(self):
        return await self.put_loopback_ui_event(
            (FrontendEventType.Click.value, None))

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


def get_control_value(comp: Union[Input, Switch, RadioGroup, Select,
                                  MultipleSelect, Slider, BlenderSlider]):
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
    elif isinstance(comp, (Slider, BlenderSlider)):
        return comp.value
    else:
        raise NotImplementedError("not a control ui")


@dataclasses.dataclass
class AppTerminalProps(MUIFlexBoxProps):
    pass


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


@dataclasses.dataclass
class Theme:
    # TODO add detailed annotations
    components: Union[Dict[str, Any], Undefined] = undefined
    palette: Union[Dict[str, Any], Undefined] = undefined
    typography: Union[Dict[str, Any], Undefined] = undefined
    spacing: Union[Dict[str, Any], Undefined] = undefined
    breakpoints: Union[Dict[str, Any], Undefined] = undefined
    shadows: Union[Dict[str, Any], Undefined] = undefined
    transitions: Union[Dict[str, Any], Undefined] = undefined
    zIndex: Union[Dict[str, Any], Undefined] = undefined
    mixins: Union[Dict[str, Any], Undefined] = undefined
    shape: Union[Dict[str, Any], Undefined] = undefined


@dataclasses.dataclass
class ThemeProviderProps(MUIBasicProps, ContainerBaseProps):
    theme: Theme = dataclasses.field(default_factory=Theme)


class ThemeProvider(MUIContainerBase[ThemeProviderProps, MUIComponentType]):
    """see https://material-ui.com/customization/theming/ for more details.
    we only support static theme in this component.
    """

    def __init__(self, children: LayoutType, theme: Theme) -> None:
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.ThemeProvider, ThemeProviderProps, children)
        self.props.theme = theme

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class TabsProps(MUIFlexBoxProps):
    value: str = ""
    textColor: Union[Literal["inherit", "primary", "secondary"],
                     Undefined] = undefined
    indicatorColor: Union[Literal["primary", "secondary"],
                          Undefined] = undefined
    orientation: Union[Literal["horizontal", "vertical"],
                       Undefined] = undefined
    variant: Union[Literal["scrollable", "vertical", "fullWidth"],
                   Undefined] = undefined
    visibleScrollbar: Union[Undefined, bool] = undefined
    centered: Union[Undefined, bool] = undefined
    scrollButtons: Union[Literal["auto"], bool, Undefined] = undefined
    selectionFollowsFocus: Union[Undefined, bool] = undefined
    panelProps: Union[FlexBoxProps, Undefined] = undefined
    tooltipPlacement: Union[_TooltipPlacement, Undefined] = undefined
    tooltipMultiline: Union[bool, Undefined] = undefined
    # 300 by default
    tooltipEnterDelay: Union[NumberType, Undefined] = undefined
    tooltipLeaveDelay: Union[NumberType, Undefined] = undefined
    tooltipEnterNextDelay: Union[NumberType, Undefined] = undefined

@dataclasses.dataclass
class TabDef:
    label: str
    value: str
    component: Component
    wrapped: Union[Undefined, bool] = undefined
    disabled: Union[Undefined, bool] = undefined
    icon: Union[IconType, str, Undefined] = undefined
    iconPosition: Union[Literal["start", "end", "bottom", "top"],
                        Undefined] = undefined
    disableFocusRipple: Union[Undefined, bool] = undefined
    disableRipple: Union[Undefined, bool] = undefined
    iconSize: Union[Literal["small", "medium", "large", "inherit"],
                    Undefined] = undefined
    iconFontSize: Union[ValueType, Undefined] = undefined
    tooltip: Union[str, Undefined] = undefined
    labelComponent: Union[Component, Undefined] = undefined


class Tabs(MUIContainerBase[TabsProps, MUIComponentType]):

    @dataclasses.dataclass
    class ChildDef:
        tabDefs: List["TabDef"]

    def __init__(self,
                 tab_defs: List["TabDef"],
                 init_value: Optional[str] = None) -> None:
        all_values = [x.value for x in tab_defs]
        assert len(all_values) == len(set(all_values)), "values must be unique"
        super().__init__(UIType.Tabs,
                         TabsProps,
                         Tabs.ChildDef(tab_defs),
                         allowed_events=[
                             FrontendEventType.Change,
                         ])
        if init_value is not None:
            assert init_value in all_values
            self.props.value = init_value
        else:
            self.props.value = all_values[0]

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["value"] = self.props.value
        return res

    def state_change_callback(self, value: str, type: ValueType):
        self.props.value = value

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self,
                                           ev,
                                           is_sync=is_sync,
                                           sync_status_first=False,
                                           change_status=False)


@dataclasses.dataclass
class AllotmentProps(MUIFlexBoxProps):
    defaultSizes: Union[List[NumberType], Undefined] = undefined
    maxSize: Union[NumberType, Undefined] = undefined
    minSize: Union[NumberType, Undefined] = undefined
    proportionalLayout: Union[bool, Undefined] = undefined
    separator: Union[bool, Undefined] = undefined
    snap: Union[bool, Undefined] = undefined
    vertical: Union[bool, Undefined] = undefined


class Allotment(MUIContainerBase[AllotmentProps, MUIComponentType]):

    @dataclasses.dataclass
    class Pane:
        component: Component
        maxSize: Union[NumberType, Undefined] = undefined
        minSize: Union[NumberType, Undefined] = undefined
        priority: Union[NumberType, Undefined] = undefined
        preferredSize: Union[ValueType, Undefined] = undefined
        snap: Union[bool, Undefined] = undefined
        visible: Union[bool, Undefined] = undefined

    @dataclasses.dataclass
    class ChildDef:
        paneDefs: List["Allotment.Pane"]

    def __init__(self, children: Union[LayoutType,
                                       "Allotment.ChildDef"]) -> None:
        if not isinstance(children, Allotment.ChildDef):
            if isinstance(children, list):
                children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.Allotment, AllotmentProps, children, False)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


# class AllotmentPane(MUIContainerBase[AllotmentPaneProps, MUIComponentType]):
#     def __init__(self, children: LayoutType) -> None:
#         if isinstance(children, list):
#             children = {str(i): v for i, v in enumerate(children)}
#         super().__init__(UIType.AllotmentPane, AllotmentPaneProps, children,
#                          False)

#     @property
#     def prop(self):
#         propcls = self.propcls
#         return self._prop_base(propcls, self)

#     @property
#     def update_event(self):
#         propcls = self.propcls
#         return self._update_props_base(propcls)


@dataclasses.dataclass
class FlexLayoutFontProps:
    size: Union[str, Undefined] = undefined
    family: Union[str, Undefined] = undefined
    style: Union[str, Undefined] = undefined
    weight: Union[str, Undefined] = undefined


@dataclasses.dataclass
class FlexLayoutProps(ContainerBaseProps):
    modelJson: Union[Any, Undefined] = undefined
    # model change save debounce.
    debounce: Union[NumberType, Undefined] = undefined
    font: Union[FlexLayoutFontProps, Undefined] = undefined


class FlexLayout(MUIContainerBase[FlexLayoutProps, MUIComponentType]):
    """TODO currently we can't programatically configure FlexLayout
    after it's been initialized. After init, we only support dnd to add new component
    from other components.

    TODO support add new tab to a tabset.
    FL.HBox([
        component/tab/tabset,
        FL.HBox([

        ])
    ])
    """

    class Row:

        def __init__(self,
                     children: List[Union["FlexLayout.Row",
                                          "FlexLayout.TabSet",
                                          "FlexLayout.Tab"]],
                     weight: NumberType = 100) -> None:
            new_children: List[Union["FlexLayout.Row",
                                     "FlexLayout.TabSet"]] = []
            for c in children:
                if isinstance(c, FlexLayout.Tab):
                    new_children.append(FlexLayout.TabSet([c]))
                elif isinstance(c, (FlexLayout.TabSet, FlexLayout.Row)):
                    new_children.append(c)
                else:
                    assert not isinstance(c,
                                          (FlexLayout.HBox, FlexLayout.VBox))
                    new_children.append(FlexLayout.TabSet([c]))
            self.children = new_children
            self.weight = weight

        def get_components(self):
            res: List[MUIComponentType] = []
            for c in self.children:
                res.extend(c.get_components())
            return res

        def get_model_dict(self):
            return {
                "type": "row",
                "weight": self.weight,
                "children": [c.get_model_dict() for c in self.children]
            }

    class TabSet:

        def __init__(self,
                     children: List[Union[MUIComponentType, "FlexLayout.Tab"]],
                     weight: NumberType = 100) -> None:
            new_children: List[FlexLayout.Tab] = []
            for c in children:
                if isinstance(c, FlexLayout.Tab):
                    new_children.append(c)
                else:
                    new_children.append(FlexLayout.Tab(c))
            self.children = new_children
            self.weight = weight

        def get_model_dict(self):
            return {
                "type": "tabset",
                "weight": self.weight,
                "children": [c.get_model_dict() for c in self.children]
            }

        def get_components(self):
            res: List[MUIComponentType] = []
            for c in self.children:
                res.append(c.comp)
            return res

    class Tab:

        def __init__(self,
                     comp: MUIComponentType,
                     name: Optional[str] = None) -> None:
            self.comp = comp
            if name is None:
                name = type(comp).__name__
            self.name = name

        def get_model_dict(self):
            assert self.comp._flow_uid is not None 
            comp_last_uid = self.comp._flow_uid.parts[-1]
            return {
                "type": "tab",
                "id": comp_last_uid,
                "name": self.name,
                "component": "app",
                "config": {
                    "uid": self.comp._flow_uid_encoded
                }
            }

    class HBox:
        """will be parsed to row/tab/tabset
        """

        def __init__(self,
                     children: List[Union["FlexLayout.Row",
                                          "FlexLayout.TabSet",
                                          "FlexLayout.Tab", "FlexLayout.HBox",
                                          "FlexLayout.VBox",
                                          "MUIComponentType"]],
                     weight: NumberType = 100) -> None:
            self.children = children
            self.weight = weight

    class VBox:
        """will be parsed to row/tab/tabset
        """

        def __init__(self,
                     children: List[Union["FlexLayout.Row",
                                          "FlexLayout.TabSet",
                                          "FlexLayout.Tab", "FlexLayout.HBox",
                                          "FlexLayout.VBox",
                                          "MUIComponentType"]],
                     weight: NumberType = 100) -> None:
            self.children = children
            self.weight = weight

    @staticmethod
    def _parse_init_children_recursive(
            children: Union[MUIComponentType, "FlexLayout.HBox",
                            "FlexLayout.VBox", "FlexLayout.Row",
                            "FlexLayout.TabSet", "FlexLayout.Tab"],
            level: int = 0):
        if not isinstance(children, (FlexLayout.HBox, FlexLayout.VBox)):
            return children
        if level % 2 == 0:
            # row
            if isinstance(children, FlexLayout.HBox):
                new_children = []
                for c in children.children:
                    new_children.append(
                        FlexLayout._parse_init_children_recursive(
                            c, level + 1))
                return FlexLayout.Row(new_children, children.weight)
            else:
                new_children = []
                for c in children.children:
                    new_children.append(
                        FlexLayout._parse_init_children_recursive(
                            c, level + 2))
                return FlexLayout.Row([FlexLayout.Row(new_children)],
                                      children.weight)
        else:
            # tabset
            if isinstance(children, FlexLayout.VBox):
                new_children = []
                for c in children.children:
                    new_children.append(
                        FlexLayout._parse_init_children_recursive(
                            c, level + 1))
                return FlexLayout.Row(new_children, children.weight)
            else:
                new_children = []
                for c in children.children:
                    new_children.append(
                        FlexLayout._parse_init_children_recursive(
                            c, level + 2))
                return FlexLayout.Row([FlexLayout.Row(new_children)],
                                      children.weight)

    @staticmethod
    def _parse_init_children(children: Union["FlexLayout.HBox",
                                             "FlexLayout.VBox"],
                             level: int = 0):
        if level % 2 == 0:
            # row
            if isinstance(children, FlexLayout.HBox):
                new_children = []
                for c in children.children:
                    new_children.append(
                        FlexLayout._parse_init_children_recursive(
                            c, level + 1))
                return FlexLayout.Row(new_children, children.weight)
            else:
                new_children = []
                for c in children.children:
                    new_children.append(
                        FlexLayout._parse_init_children_recursive(
                            c, level + 2))
                return FlexLayout.Row([FlexLayout.Row(new_children)],
                                      children.weight)
        else:
            # tabset
            if isinstance(children, FlexLayout.VBox):
                new_children = []
                for c in children.children:
                    new_children.append(
                        FlexLayout._parse_init_children_recursive(
                            c, level + 1))
                return FlexLayout.Row(new_children, children.weight)
            else:
                new_children = []
                for c in children.children:
                    new_children.append(
                        FlexLayout._parse_init_children_recursive(
                            c, level + 2))
                return FlexLayout.Row([FlexLayout.Row(new_children)],
                                      children.weight)

    def __init__(
        self,
        children: Union[List[Union["FlexLayout.Row",
                                   "FlexLayout.TabSet"]], "FlexLayout.Row",
                        "FlexLayout.TabSet", "FlexLayout.Tab",
                        "FlexLayout.HBox", "FlexLayout.VBox", MUIComponentType]
    ) -> None:
        events = [
            FrontendEventType.ComplexLayoutCloseTab,
            FrontendEventType.ComplexLayoutSelectTab,
            FrontendEventType.ComplexLayoutSelectTabSet,
            FrontendEventType.ComplexLayoutTabReload,
            FrontendEventType.ComplexLayoutStoreModel,
            FrontendEventType.Drop,
        ]
        if isinstance(children, FlexLayout.Row):
            self._init_children_row = children
        elif isinstance(children, FlexLayout.TabSet):
            self._init_children_row = FlexLayout.Row([children])
        elif isinstance(children, FlexLayout.Tab):
            self._init_children_row = FlexLayout.Row(
                [FlexLayout.TabSet([children])])
        elif isinstance(children, (FlexLayout.HBox, FlexLayout.VBox)):
            self._init_children_row = FlexLayout._parse_init_children(children)
        elif isinstance(children, list):
            self._init_children_row = FlexLayout._parse_init_children(
                FlexLayout.HBox([*children]))
        else:
            self._init_children_row = FlexLayout.Row(
                [FlexLayout.TabSet([children])])
        comp_children = self._init_children_row.get_components()
        # we must generate uuid here because tab in FlexLayout need to have same id with component uid
        comp_children_dict = {str(uuid.uuid4()): v for v in comp_children}

        super().__init__(UIType.FlexLayout,
                         FlexLayoutProps,
                         comp_children_dict,
                         False,
                         allowed_events=[x.value for x in events])

        self.register_event_handler(
            FrontendEventType.ComplexLayoutStoreModel.value,
            self._on_save_model)

        self.event_close_tab = self._create_event_slot(
            FrontendEventType.ComplexLayoutCloseTab)
        self.event_select_tab = self._create_event_slot(
            FrontendEventType.ComplexLayoutSelectTab)
        self.event_select_tabset = self._create_event_slot(
            FrontendEventType.ComplexLayoutSelectTabSet)
        self.event_drop = self._create_event_slot(FrontendEventType.Drop)
        self.event_reload = self._create_event_slot(
            FrontendEventType.ComplexLayoutTabReload)

    def _on_save_model(self, model):
        self.props.modelJson = model

    def get_props(self):
        res = super().get_props()
        # we delay init model here because we need
        # to wait for all components to be initialized
        # to get uid of child components.
        if isinstance(self.props.modelJson, Undefined):
            res["modelJson"] = {
                "global": {
                    "tabEnableClose": True
                },
                "borders": [],
                "layout": self._init_children_row.get_model_dict()
            }
        return res

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self,
                                           ev,
                                           sync_status_first=False,
                                           is_sync=is_sync,
                                           sync_state_after_change=False,
                                           change_status=False)


@dataclasses.dataclass
class CircularProgressProps(MUIFlexBoxProps):
    value: Union[NumberType, Undefined] = undefined
    withLabel: Union[Undefined, bool] = undefined
    labelColor: Union[Undefined, str] = undefined
    muiColor: Union[_BtnGroupColor, Undefined] = undefined
    labelVariant: Union[_TypographyVarient, Undefined] = undefined
    size: Union[Undefined, str, NumberType] = undefined
    variant: Union[Undefined, Literal["determinate",
                                      "indeterminate"]] = undefined
    thickness: Union[Undefined, NumberType] = undefined


class CircularProgress(MUIComponentBase[CircularProgressProps]):

    def __init__(self,
                 init_value: Union[NumberType, Undefined] = undefined) -> None:
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
    value: Union[NumberType, Undefined] = undefined
    valueBuffer: Union[NumberType, Undefined] = undefined
    withLabel: Union[Undefined, bool] = undefined
    labelColor: Union[Undefined, str] = undefined
    muiColor: Union[_BtnGroupColor, Undefined] = undefined
    labelVariant: Union[_TypographyVarient, Undefined] = undefined
    variant: Union[Undefined, Literal["determinate", "indeterminate", "buffer",
                                      "query"]] = undefined


class LinearProgress(MUIComponentBase[LinearProgressProps]):

    def __init__(
        self,
        init_value: Union[NumberType, Undefined] = undefined,
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


@dataclasses.dataclass
class JsonViewerProps(MUIFlexBoxProps):
    data: Any = None


class JsonViewer(MUIComponentBase[JsonViewerProps]):

    def __init__(
        self,
        init_data: Any = None,
    ) -> None:
        super().__init__(UIType.JsonViewer, JsonViewerProps)
        self.props.data = init_data

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


def _default_json_node():
    return JsonLikeNode(UniqueTreeIdForTree.from_parts(["root"]), "root", JsonLikeType.Object.value, "Object",
                        undefined, 0, [])


class _TreeControlType(enum.IntEnum):
    UpdateSubTree = 0


@dataclasses.dataclass
class JsonLikeTreePropsBase(MUIFlexBoxProps):
    tree: JsonLikeNode = dataclasses.field(default_factory=_default_json_node)
    multiSelect: Union[Undefined, bool] = undefined
    disableSelection: Union[Undefined, bool] = undefined
    ignoreRoot: Union[Undefined, bool] = undefined
    # useFastTree: Union[Undefined, bool] = undefined
    contextMenus: Union[Undefined, List[ContextMenuData]] = undefined
    fixedSize: Union[Undefined, bool] = undefined


@dataclasses.dataclass
class JsonLikeTreeProps(JsonLikeTreePropsBase):
    disabledItemsFocusable: Union[Undefined, bool] = undefined
    rowSelection: List[str] = dataclasses.field(default_factory=list)
    expanded: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class TanstackJsonLikeTreeProps(JsonLikeTreePropsBase):
    rowSelection: Dict[str, bool] = dataclasses.field(default_factory=dict)
    expanded: Union[bool, Dict[str,
                               bool]] = dataclasses.field(default_factory=dict)


T_tview_base_props = TypeVar("T_tview_base_props", bound=JsonLikeTreePropsBase)


class JsonLikeTreeBase(MUIComponentBase[T_tview_base_props]):

    def __init__(self,
                 base_type: UIType,
                 prop_cls: Type[T_tview_base_props],
                 tree: Optional[JsonLikeNode] = None) -> None:
        if tree is None:
            tree = _default_json_node()
        tview_events = [
            FrontendEventType.Change.value,
            FrontendEventType.TreeItemSelectChange.value,
            FrontendEventType.TreeItemToggle.value,
            FrontendEventType.TreeLazyExpand.value,
            FrontendEventType.TreeItemFocus.value,
            FrontendEventType.TreeItemButton.value,
            FrontendEventType.ContextMenuSelect.value,
            FrontendEventType.TreeItemRename.value,
        ]
        super().__init__(base_type,
                         prop_cls,
                         allowed_events=tview_events,
                         json_only=True)
        self.props.tree = tree

        self.event_select = self._create_event_slot(
            FrontendEventType.TreeItemSelectChange)
        # selection/expand change
        self.event_change = self._create_event_slot(FrontendEventType.Change)

        self.event_toggle = self._create_event_slot(
            FrontendEventType.TreeItemToggle)
        self.event_lazy_expand = self._create_event_slot(
            FrontendEventType.TreeLazyExpand)
        self.event_focus = self._create_event_slot(
            FrontendEventType.TreeItemFocus)
        self.event_icon_button = self._create_event_slot(
            FrontendEventType.TreeItemButton)
        self.event_context_menu = self._create_event_slot(
            FrontendEventType.ContextMenuSelect)
        self.event_rename = self._create_event_slot(
            FrontendEventType.TreeItemRename)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    def _update_subtree_backend_recursive(self, root: JsonLikeNode,
                                          node: JsonLikeNode,
                                          parts: List[str]):
        if len(parts) == 1:
            root.children = list(
                map(lambda x: node if x.id == node.id else x, root.children))
            return root
        root.children = list(
            map(
                lambda x: self._update_subtree_backend_recursive(
                    x, node, parts[1:])
                if x.name == parts[0] else x, root.children))
        return root

    def _update_subtree_backend(self, node: JsonLikeNode):
        parts = node.id.parts
        if len(parts) == 1:
            if node.id == self.props.tree.id:
                self.props.tree = node
            return
        if parts[0] != self.props.tree.name:
            return

        return self._update_subtree_backend_recursive(self.props.tree, node,
                                                      parts[1:])

    async def update_subtree(self, node: JsonLikeNode):
        self._update_subtree_backend(node)
        return await self.send_and_wait(
            self.create_comp_event({
                "type": _TreeControlType.UpdateSubTree,
                "tree": node,
            }))


class JsonLikeTree(JsonLikeTreeBase[JsonLikeTreeProps]):

    def __init__(self, tree: Optional[JsonLikeNode] = None) -> None:
        super().__init__(UIType.JsonLikeTreeView, JsonLikeTreeProps, tree)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls, json_only=True)

    async def update_tree(self, tree: JsonLikeNode):
        await self.send_and_wait(self.update_event(tree=tree))

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self,
                                           ev,
                                           sync_status_first=False,
                                           change_status=False,
                                           is_sync=is_sync)

    def state_change_callback(
            self,
            value,
            type: ValueType = FrontendEventType.TreeItemSelectChange.value):
        # this only triggered when dialog closed, so we always set
        # open to false.
        if type == FrontendEventType.TreeItemSelectChange:
            self.prop(rowSelection=value)
        elif type == FrontendEventType.TreeItemExpandChange:
            self.prop(expanded=value)

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["rowSelection"] = self.props.rowSelection
        res["expanded"] = self.props.expanded
        return res

    async def select(self, ids: List[str]):
        await self.send_and_wait(self.update_event(rowSelection=ids))


class TanstackJsonLikeTree(JsonLikeTreeBase[TanstackJsonLikeTreeProps]):

    def __init__(self, tree: Optional[JsonLikeNode] = None) -> None:
        super().__init__(UIType.TanstackJsonLikeTreeView,
                         TanstackJsonLikeTreeProps, tree)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls, json_only=True)

    async def update_tree(self, tree: JsonLikeNode):
        await self.send_and_wait(self.update_event(tree=tree))

    async def handle_event(self, ev: Event, is_sync: bool = False):
        # select and expand event may received at the same time,
        # so we can't change status here.
        return await handle_standard_event(self,
                                           ev,
                                           sync_status_first=False,
                                           change_status=False,
                                           is_sync=is_sync)

    def state_change_callback(
            self,
            value,
            type: ValueType = FrontendEventType.TreeItemSelectChange.value):
        # this only triggered when dialog closed, so we always set
        # open to false.
        if type == FrontendEventType.TreeItemSelectChange:
            self.prop(rowSelection=value)
        elif type == FrontendEventType.TreeItemExpandChange:
            self.prop(expanded=value)

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["rowSelection"] = self.props.rowSelection
        res["expanded"] = self.props.expanded
        return res

    async def select(self, ids: List[str]):
        await self.send_and_wait(
            self.update_event(rowSelection={k: True
                                            for k in ids}))


class ControlNodeType(enum.IntEnum):
    Number = 0
    RangeNumber = 1
    Bool = 2
    Select = 3
    String = 4
    Folder = 5
    Vector2 = 6
    VectorN = 7
    ColorRGB = 8
    ColorRGBA = 9


@dataclasses.dataclass
class ControlColorRGB:
    r: NumberType
    g: NumberType
    b: NumberType


@dataclasses.dataclass
class ControlColorRGBA(ControlColorRGB):
    a: float


@dataclasses.dataclass
class ControlNode:
    id: str
    name: str
    type: int
    initValue: Union[Undefined, NumberType, bool, str, ControlColorRGBA,
                     Vector3Type, List[NumberType]] = undefined
    children: "List[ControlNode]" = dataclasses.field(default_factory=list)
    # for range
    min: Union[Undefined, NumberType] = undefined
    max: Union[Undefined, NumberType] = undefined
    step: Union[Undefined, NumberType] = undefined

    # for select
    selects: Union[Undefined, List[Tuple[str, ValueType]]] = undefined
    # for string
    rows: Union[Undefined, bool, int] = undefined

    alias: Union[Undefined, str] = undefined
    # for vectorN
    count: Union[Undefined, int] = undefined
    isInteger: Union[Undefined, bool] = undefined


@dataclasses.dataclass
class ControlDesp:
    type: int
    initValue: Union[Undefined, NumberType, bool, str, ControlColorRGBA,
                     Vector3Type, List[NumberType]] = undefined
    # for range
    min: Union[Undefined, NumberType] = undefined
    max: Union[Undefined, NumberType] = undefined
    step: Union[Undefined, NumberType] = undefined
    # for select
    selects: Union[Undefined, List[Tuple[str, ValueType]]] = undefined
    # for string
    rows: Union[Undefined, bool, int] = undefined
    # for vectorN
    count: Union[Undefined, int] = undefined
    isInteger: Union[Undefined, bool] = undefined


@dataclasses.dataclass
class DynamicControlsProps(MUIFlexBoxProps):
    nodes: List[ControlNode] = dataclasses.field(default_factory=list)
    # use_leva_style: bool = True
    collapsed: Union[Undefined, bool] = undefined
    debounce: Union[Undefined, NumberType] = undefined
    throttle: Union[Undefined, NumberType] = undefined
    title: Union[Undefined, str] = undefined
    # leva is uncontrolled component. if we change nodes,
    # the control won't be updated, so we must provide
    # a different react key to force component
    # remount.
    reactKey: Union[Undefined, str] = undefined


class DynamicControls(MUIComponentBase[DynamicControlsProps]):

    def __init__(self,
                 callback: Optional[Callable[[Tuple[str, Any]],
                                             _CORO_NONE]] = None,
                 init: Optional[List[ControlNode]] = None) -> None:
        super().__init__(UIType.DynamicControls,
                         DynamicControlsProps,
                         allowed_events=[FrontendEventType.Change.value])
        if init is not None:
            self.props.nodes = init
        if callback is not None:
            self.register_event_handler(FrontendEventType.Change.value,
                                        callback)
        self.event_change = self._create_event_slot(FrontendEventType.Change)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self,
                                           ev,
                                           sync_status_first=False,
                                           is_sync=is_sync)


@dataclasses.dataclass
class SimpleControlsProps(MUIFlexBoxProps):
    tree: List[JsonLikeNode] = dataclasses.field(default_factory=list)
    contextMenus: Union[Undefined, List[ContextMenuData]] = undefined
    reactKey: Union[Undefined, str] = undefined


class SimpleControls(MUIComponentBase[SimpleControlsProps]):

    def __init__(self,
                 callback: Optional[Callable[[Tuple[str, Any]],
                                             _CORO_NONE]] = None,
                 init: Optional[List[JsonLikeNode]] = None) -> None:
        super().__init__(UIType.SimpleControls,
                         SimpleControlsProps,
                         allowed_events=[FrontendEventType.Change.value])
        if init is not None:
            self.props.tree = init
        if callback is not None:
            self.register_event_handler(FrontendEventType.Change.value,
                                        callback)
        self.event_change = self._create_event_slot(FrontendEventType.Change)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self,
                                           ev,
                                           sync_status_first=False,
                                           is_sync=is_sync)


@dataclasses.dataclass
class MUIVirtualizedBoxProps(MUIFlexBoxWithDndProps):
    pass


class VirtualizedBox(MUIContainerBase[MUIVirtualizedBoxProps,
                                      MUIComponentType]):
    """ flex box that use data list and template component to render
    list of data with same UI components.
    """

    def __init__(self, children: Optional[LayoutType] = None) -> None:
        if children is not None and isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.VirtualizedBox,
                         MUIVirtualizedBoxProps,
                         children,
                         allowed_events=ALL_POINTER_EVENTS)
        self.event_click = self._create_event_slot(FrontendEventType.Click)
        self.event_double_click = self._create_event_slot(
            FrontendEventType.DoubleClick)
        self.event_pointer_enter = self._create_event_slot(
            FrontendEventType.Enter)
        self.event_pointer_leave = self._create_event_slot(
            FrontendEventType.Leave)
        self.event_pointer_down = self._create_event_slot(
            FrontendEventType.Down)
        self.event_pointer_up = self._create_event_slot(FrontendEventType.Up)
        self.event_pointer_move = self._create_event_slot(
            FrontendEventType.Move)
        self.event_pointer_over = self._create_event_slot(
            FrontendEventType.Over)
        self.event_pointer_out = self._create_event_slot(FrontendEventType.Out)
        self.event_pointer_context_menu = self._create_event_slot(
            FrontendEventType.ContextMenu)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class DataListControlType(enum.IntEnum):
    SetData = 0
    ScrollToIndex = 1


@dataclasses.dataclass
class MUIDataFlexBoxWithDndProps(MUIFlexBoxWithDndProps):
    dataList: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    idKey: str = "id"
    virtualized: Union[Undefined, bool] = undefined


@dataclasses.dataclass
class DataUpdate:
    index: int
    update: Any


class DataFlexBox(MUIContainerBase[MUIDataFlexBoxWithDndProps,
                                   MUIComponentType]):
    """ flex box that use data list and template component to render
    list of data with same UI components.
    """

    @dataclasses.dataclass
    class ChildDef:
        component: Component

    def __init__(self, children: Component) -> None:
        super().__init__(UIType.DataFlexBox,
                         MUIDataFlexBoxWithDndProps,
                         DataFlexBox.ChildDef(children),
                         allowed_events=[
                             FrontendEventType.Drop.value,
                             FrontendEventType.DragCollect.value
                         ] + list(ALL_POINTER_EVENTS))
        # backend events
        self.event_item_changed = self._create_emitter_event_slot(
            FrontendEventType.DataItemChange)
        self.event_click = self._create_event_slot(FrontendEventType.Click)
        self.event_double_click = self._create_event_slot(
            FrontendEventType.DoubleClick)
        self.event_pointer_enter = self._create_event_slot(
            FrontendEventType.Enter)
        self.event_pointer_leave = self._create_event_slot(
            FrontendEventType.Leave)
        self.event_pointer_down = self._create_event_slot(
            FrontendEventType.Down)
        self.event_pointer_up = self._create_event_slot(FrontendEventType.Up)
        self.event_pointer_move = self._create_event_slot(
            FrontendEventType.Move)
        self.event_pointer_over = self._create_event_slot(
            FrontendEventType.Over)
        self.event_pointer_out = self._create_event_slot(FrontendEventType.Out)
        self.event_pointer_context_menu = self._create_event_slot(
            FrontendEventType.ContextMenu)
        self.event_drag_collect = self._create_event_slot(
            FrontendEventType.DragCollect)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self,
                                           ev,
                                           sync_status_first=False,
                                           is_sync=is_sync)

    async def update_data_in_index(self, index: int, updates: Dict[str, Any]):
        return await self.update_datas_in_index([DataUpdate(index, updates)])

    async def update_datas_in_index(self, updates: List[DataUpdate]):
        for du in updates:
            self.props.dataList[du.index].update(du.update)
        return await self.send_and_wait(
            self.create_comp_event({
                "type":
                DataListControlType.SetData.value,
                "updates": [{
                    "index": x.index,
                    "update": x.update
                } for x in updates],
            }))

    async def _comp_bind_update_data(self, event: Event, prop_name: str):
        key = event.keys
        indexes = event.indexes
        assert not isinstance(key, Undefined) and not isinstance(
            indexes, Undefined)
        assert len(indexes) == 1, "update data list only supports single index"
        data = event.data
        data_item = self.props.dataList[indexes[0]]
        assert prop_name in data_item
        data_item[prop_name] = data
        await self.update_data_in_index(indexes[0], {prop_name: data})
        self.flow_event_emitter.emit(
            FrontendEventType.DataItemChange.value,
            Event(FrontendEventType.DataItemChange.value, (key, indexes[0]),
                  key, indexes))

    def bind_prop(self, comp: Component, prop_name: str):
        """bind a data prop with control component. no type check.
        **WARNING**: don't bind prop in nested template component, you 
        need to handle change event in nested template container
        by yourself.
        """
        if isinstance(comp, (Slider, BlenderSlider, _InputBaseComponent)):
            comp.props.value = undefined
            # assert isinstance(comp.value, Undefined), "slider and input must be uncontrolled."
        # TODO only support subset of all components
        if FrontendEventType.Change.value in comp._flow_allowed_events:
            # TODO change all control components to use value as its data prop name
            if "defaultValue" in comp._prop_field_names:
                comp.set_override_props(defaultValue=prop_name)
            elif "value" in comp._prop_field_names:
                comp.set_override_props(value=prop_name)
            elif "checked" in comp._prop_field_names:
                comp.set_override_props(checked=prop_name)
            comp.register_event_handler(FrontendEventType.Change.value,
                                        partial(self._comp_bind_update_data,
                                                prop_name=prop_name),
                                        simple_event=False)
        else:
            raise ValueError("only support components with change event")


class DataGridColumnSpecialType(enum.IntEnum):
    # master detail can't be used with expand.
    MasterDetail = 0
    Expand = 1
    Checkbox = 2
    Number = 3


@dataclasses.dataclass
class DataGridNumberCell:
    precision: Union[Undefined, int] = undefined
    fixed: Union[Undefined, int] = undefined
    fontSize: Union[Undefined, ValueType] = undefined
    color: Union[Undefined, str] = undefined
    fontFamily: Union[Undefined, str] = undefined


@dataclasses.dataclass
class DataGridColumnSpecialProps:
    NumberCell: Union[Undefined, DataGridNumberCell] = undefined


@dataclasses.dataclass
class DataGridColumnDef:
    """id resolution order: id -> accessorKey -> header
    accessorKey resolution order: accessorKey -> header
    """
    header: Union[Undefined, str] = undefined
    accessorKey: Union[Undefined, str] = undefined
    cell: Union[Undefined, Component] = undefined
    footer: Union[Undefined, str] = undefined
    id: Union[Undefined, str] = undefined
    columns: "List[DataGridColumnDef]" = dataclasses.field(
        default_factory=list)
    align: Union[Undefined, Literal["center", "inherit", "justify", "left",
                                    "right"]] = undefined
    editable: Union[Undefined, bool] = undefined
    specialType: Union[Undefined, int] = undefined
    width: Union[Undefined, int] = undefined
    editCell: Union[Undefined, Component] = undefined
    specialProps: Union[Undefined, DataGridColumnSpecialProps] = undefined

    def _id_resolution(self):
        id_resolu = self.id
        if isinstance(self.id, Undefined):
            if isinstance(self.accessorKey, Undefined):
                assert not isinstance(
                    self.header, Undefined
                ) and self.header != "", "you must provide a id or accessorKey if header is undefined or empty"
                id_resolu = self.header
            else:
                id_resolu = self.accessorKey
        else:
            id_resolu = self.id
        return id_resolu

    @model_validator(mode="after")
    def _check_id_header_accesskey_valid(self) -> Self:
        id_resolu = self._id_resolution()
        assert id_resolu != "", "id can't be empty"
        return self


@dataclasses.dataclass
class DataGridProxy(abc.ABC):
    numRows: int
    numColumns: int
    defaultData: Dict[str, Any]
    currentRange: Tuple[int, int] = (0, 0)
    currentDataList: List[Dict[str,
                               Any]] = dataclasses.field(default_factory=list)

    @abc.abstractmethod
    async def fetch_data(self, start: int, end: int) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    def fetch_data_sync(self, start: int, end: int) -> List[Dict[str, Any]]:
        raise NotImplementedError


@dataclasses.dataclass
class DataGridPropsBase:
    # proxy + lazy load for large dataset. only available with virtualization.
    # we can't put DataGridColumnDef here because
    # it may contain component.
    # WARNING when you use data proxy, id is set by us, not user,
    # it will be str(index) of your data list proxy.
    idKey: Union[Undefined, str] = undefined
    rowHover: Union[Undefined, bool] = undefined
    virtualized: Union[Undefined, bool] = undefined
    virtualizedInfScrolling: Union[Undefined, bool] = undefined
    enableRowSelection: Union[Undefined, bool] = undefined
    enableMultiRowSelection: Union[Undefined, bool] = False
    debugTable: Union[Undefined, bool] = undefined
    masterDetailUseFetch: Union[Undefined, bool] = undefined
    stickyHeader: Union[Undefined, bool] = undefined
    size: Union[Undefined, Literal["small", "medium"]] = undefined
    cellEdit: Union[Undefined, bool] = undefined
    rowSelection: Union[Undefined, bool] = undefined
    enableFilter: Union[Undefined, bool] = undefined
    fullWidth: Union[Undefined, bool] = undefined
    tableLayout: Union[Undefined, Literal["auto", "fixed"]] = undefined
    tableSxProps: Union[Undefined, Dict[str, Any]] = undefined
    # one component for each header.
    # use mui.MatchCase to select real component by id.
    customHeaderDatas: Union[Undefined, List[Dict[str, Any]]] = undefined
    customFooterDatas: Union[Undefined, List[Dict[str, Any]]] = undefined


@dataclasses.dataclass
class DataGridProps(MUIFlexBoxProps, DataGridPropsBase):
    # proxy + lazy load for large dataset. only available with virtualization.
    # we can't put DataGridColumnDef here because
    # it may contain component.
    # WARNING when you use data proxy, id is set by us, not user,
    # it will be str(index) of your data list proxy.
    dataList: Union[List[Dict[str, Any]],
                    DataGridProxy] = dataclasses.field(default_factory=list)

    @model_validator(mode='after')
    def _validator_post_root(self) -> Self:
        if isinstance(self.dataList, DataGridProxy):
            assert self.virtualized, "proxy mode only works with virtualized mode"
            assert self.dataList.numRows > 0 and self.dataList.numColumns > 0, "proxy mode must provide valid numRows and numColumns"
        return self


class DataGrid(MUIContainerBase[DataGridProps, MUIComponentType]):
    """data grid, it takes list of data (dict) and render them
    as table. note that this component don't use DataGrid in mui-X,
    it use Tanstack-Table + mui-Table based solution
    we support following pro features in mui-x DataGrid
    without commercial license: row virtualization, 
    lazy loading, tree data, header filters and master
    detail.
    """

    @dataclasses.dataclass
    class ChildDef:
        columnDefs: List[DataGridColumnDef]
        masterDetail: Union[Undefined, Component] = undefined
        customHeaders: Union[Undefined, List[Component]] = undefined
        customFooters: Union[Undefined, List[Component]] = undefined

        @field_validator('columnDefs')
        def column_def_validator(cls, v: List[DataGridColumnDef]):
            id_set: Set[str] = set()
            for cdef in v:
                id_resolu = cdef._id_resolution()
                assert id_resolu not in id_set, f"duplicate id {id_resolu}"
                id_set.add(id_resolu)
            return v

    ColumnDef: TypeAlias = DataGridColumnDef

    def __init__(
        self,
        column_def: List[DataGridColumnDef],
        init_data_list: Optional[Union[List[Dict[str, Any]],
                                       DataGridProxy]] = None,
        master_detail: Union[Undefined, Component] = undefined,
        customHeaders: Union[Undefined, List[Component]] = undefined,
        customFooters: Union[Undefined, List[Component]] = undefined,
        customHeaderDatas: Union[Undefined, List[Dict[str, Any]]] = undefined,
        customFooterDatas: Union[Undefined, List[Dict[str, Any]]] = undefined,
    ) -> None:
        super().__init__(
            UIType.DataGrid,
            DataGridProps,
            DataGrid.ChildDef(column_def, master_detail, customHeaders,
                              customFooters),
            False,
            allowed_events=[
                FrontendEventType.DataGridFetchDetail.value,
                FrontendEventType.DataGridRowSelection.value,
                FrontendEventType.DataGridRowRangeChanged.value,
                FrontendEventType.DataGridProxyLazyLoadRange.value,
            ])
        # TODO check set_new_layout argument, it must be DataGrid.ChildDef
        if init_data_list is not None:
            self.props.dataList = init_data_list
        self.event_fetch_detail = self._create_event_slot(
            FrontendEventType.DataGridFetchDetail)
        self.event_row_selection = self._create_event_slot(
            FrontendEventType.DataGridRowSelection)
        # backend events
        self.event_item_changed = self._create_emitter_event_slot(
            FrontendEventType.DataItemChange)
        self.event_proxy_lazy_load = self._create_event_slot(
            FrontendEventType.DataGridProxyLazyLoadRange)

        self.event_before_mount.on(self._proxy_init)
        self.event_proxy_lazy_load.on(self._data_lazy_load)
        self.prop(customHeaderDatas=customHeaderDatas,
                  customFooterDatas=customFooterDatas)

    def _proxy_init(self, event: Event):
        datalist = self.props.dataList
        idKey = "id"
        if not isinstance(self.props.idKey, Undefined):
            idKey = self.props.idKey
        if isinstance(datalist, DataGridProxy):
            length = datalist.numRows
            # fetch at least 10 item first
            init_data = datalist.fetch_data_sync(0, min(length, 10))
            for i, item in enumerate(init_data):
                item[idKey] = str(i)
            datalist.currentDataList = init_data
            datalist.currentRange = (0, len(init_data))
            # print(datalist)

    async def _data_lazy_load(self, range: Tuple[int, int]):
        datalist = self.props.dataList
        idKey = "id"
        if not isinstance(self.props.idKey, Undefined):
            idKey = self.props.idKey
        if isinstance(datalist, DataGridProxy):
            data = await datalist.fetch_data(range[0], range[1])
            for i, item in enumerate(data):
                item[idKey] = str(i + range[0])
            datalist.currentDataList = data
            datalist.currentRange = range
            await self.send_and_wait(self.update_event(dataList=data))
            return data

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)
    
    async def scroll_to_index(self, index: int):
        return await self.send_and_wait(
            self.create_comp_event({
                "type":
                DataListControlType.ScrollToIndex.value,
                "index": index,
            }))

    async def update_data_in_index(self, index: int, updates: Dict[str, Any]):
        return await self.update_datas_in_index([DataUpdate(index, updates)])

    async def update_datas_in_index(self, updates: List[DataUpdate]):
        assert not isinstance(self.props.dataList,
                              DataGridProxy), "can't update data in proxy mode"
        for du in updates:
            self.props.dataList[du.index].update(du.update)
        return await self.send_and_wait(
            self.create_comp_event({
                "type":
                DataListControlType.SetData.value,
                "updates": [{
                    "index": x.index,
                    "update": x.update
                } for x in updates],
            }))

    async def _comp_bind_update_data(self, event: Event, prop_name: str):
        assert not isinstance(self.props.dataList,
                              DataGridProxy), "can't update data in proxy mode"
        key = event.keys
        indexes = event.indexes
        # print(event, prop_name)
        assert not isinstance(key, Undefined) and not isinstance(
            indexes, Undefined)
        assert len(indexes) == 1, "update data list only supports single index"
        data = event.data
        data_item = self.props.dataList[indexes[0]]
        assert prop_name in data_item
        data_item[prop_name] = data
        await self.update_data_in_index(indexes[0], {prop_name: data})
        self.flow_event_emitter.emit(
            FrontendEventType.DataItemChange.value,
            Event(FrontendEventType.DataItemChange.value, (key, indexes[0]),
                  key, indexes))

    def bind_prop(self, comp: Component, prop_name: str):
        """bind a data prop with control component. no type check.
        """
        if isinstance(comp, (Slider, BlenderSlider, _InputBaseComponent)):
            comp.props.value = undefined
            # assert isinstance(comp.props.value, Undefined), "slider and input must be uncontrolled."
        if FrontendEventType.Change.value in comp._flow_allowed_events:
            # TODO change all control components to use value as its data prop name
            if "defaultValue" in comp._prop_field_names:
                comp.set_override_props(defaultValue=prop_name)
            elif "value" in comp._prop_field_names:
                comp.set_override_props(value=prop_name)
            elif "checked" in comp._prop_field_names:
                comp.set_override_props(checked=prop_name)
            comp.register_event_handler(FrontendEventType.Change.value,
                                        partial(self._comp_bind_update_data,
                                                prop_name=prop_name),
                                        simple_event=False)
        else:
            raise ValueError("only support components with change event")

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class MatrixDataGridItem:
    array: np.ndarray 
    columnOffset: int = 0


@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class MatrixDataGridProps(MUIFlexBoxProps, DataGridPropsBase):
    # dict of matrix with same number of rows.
    dataList: Dict[str, MatrixDataGridItem] = dataclasses.field(default_factory=dict)
    rowOffset: Union[Undefined, int] = undefined


class MatrixDataGrid(MUIContainerBase[MatrixDataGridProps, MUIComponentType]):
    """matrix data grid, it takes dict of np.ndarray.
    don't support data edit. it should only be used for matrix visualization.
    """
    ChildDef: TypeAlias = DataGrid.ChildDef
    ColumnDef: TypeAlias = DataGridColumnDef

    def __init__(
        self,
        column_def: DataGridColumnDef,
        init_data_list: Dict[str, Union[MatrixDataGridItem, np.ndarray]],
        master_detail: Union[Undefined, Component] = undefined,
        customHeaders: Union[Undefined, List[Component]] = undefined,
        customFooters: Union[Undefined, List[Component]] = undefined,
        customHeaderDatas: Union[Undefined, List[Dict[str, Any]]] = undefined,
        customFooterDatas: Union[Undefined, List[Dict[str, Any]]] = undefined,
    ) -> None:
        super().__init__(
            UIType.MatrixDataGrid,
            MatrixDataGridProps,
            DataGrid.ChildDef([column_def], master_detail, customHeaders,
                              customFooters),
            False,
            allowed_events=[
                FrontendEventType.DataGridFetchDetail.value,
                FrontendEventType.DataGridRowSelection.value,
                FrontendEventType.DataGridRowRangeChanged.value,
                FrontendEventType.DataGridProxyLazyLoadRange.value,
            ])
        # TODO check set_new_layout argument, it must be DataGrid.ChildDef
        data_list: Dict[str, MatrixDataGridItem] = {}
        for k, v in init_data_list.items():
            if isinstance(v, np.ndarray):
                data_list[k] = MatrixDataGridItem(v)
            else:
                data_list[k] = v
        self._check_data(data_list)
        self.props.dataList = data_list
        self.event_row_selection = self._create_event_slot(
            FrontendEventType.DataGridRowSelection)
        # backend events
        self.prop(customHeaderDatas=customHeaderDatas,
                  customFooterDatas=customFooterDatas)

    @staticmethod
    def _check_data(data_list: Dict[str, MatrixDataGridItem]):
        assert len(data_list) > 0, "empty data list not allowed"
        init_shape: List[int] = []
        for k, v in data_list.items():
            assert isinstance(v, MatrixDataGridItem), "data must be MatrixDataGridItem"
            assert isinstance(v.array, np.ndarray), "data must be MatrixDataGridItem"
            assert v.array.dtype != np.bool_ and v.array.dtype != np.float16, "bool, float16 and object dtype not supported"
            assert v.array.size > 0, "empty array not allowed"
            if not init_shape:
                init_shape = list(v.array.shape)
            else:
                if len(data_list) > 1:
                    assert len(v.array.shape) == len(init_shape), "all matrix must have same number of dimensions"
                    assert list(v.array.shape[:-1]) == list(init_shape[:-1]), f"all matrix must have same number of rows, {k} has {v.array.shape[:-1]} while others has {init_shape[:-1]}"
    
    @staticmethod
    def _check_data_np_dict(data_list: Dict[str, np.ndarray]):
        data_list_items: Dict[str, MatrixDataGridItem] = {}
        for k, v in data_list.items():
            if isinstance(v, np.ndarray):
                data_list_items[k] = MatrixDataGridItem(v)
            else:
                data_list_items[k] = v

        return MatrixDataGrid._check_data(data_list_items)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)

    @staticmethod
    def get_column_id(arr_key: str, column: int):
        return f"{arr_key}-{column}"
    
    async def scroll_to_index(self, index: int):
        return await self.send_and_wait(
            self.create_comp_event({
                "type":
                DataListControlType.ScrollToIndex.value,
                "index": index,
            }))


def flex_wrapper(obj: Any,
                 metas: Optional[List[ServFunctionMeta]] = None,
                 reload_mgr: Optional[ObjectReloadManager] = None):
    """wrap a object which define a layout function "tensorpc_flow_layout"
    enable simple layout creation for arbitrary object without inherit
    """
    # TODO watch added object in watchdog
    if metas is None:
        if reload_mgr is not None:
            metas = reload_mgr.query_type_method_meta(type(obj),
                                                      no_code=True,
                                                      include_base=True)
        else:
            metas = ReloadableDynamicClass.get_metas_of_regular_methods(
                type(obj), True, no_code=True)
    methods = FlowSpecialMethods(metas)
    if methods.create_layout is not None:
        fn = methods.create_layout.bind(obj)
        layout_flex = fn()
        assert isinstance(
            layout_flex, FlexBox
        ), f"create_layout must return a flexbox when use anylayout, {type(layout_flex)}"
        # set _flow_comp_def_path to this object
        layout_flex._flow_comp_def_path = _get_obj_def_path(obj)
        layout_flex._wrapped_obj = obj
        return layout_flex
    raise ValueError(
        f"wrapped object must define a zero-arg function with @marker.mark_create_layout and return a flexbox"
    )


def flex_preview_wrapper(obj: Any,
                         metas: Optional[List[ServFunctionMeta]] = None,
                         reload_mgr: Optional[ObjectReloadManager] = None):
    """wrap a object which define a layout function "tensorpc_flow_preview_layout"
    enable simple layout creation for arbitrary object without inherit
    """
    if metas is None:
        if reload_mgr is not None:
            metas = reload_mgr.query_type_method_meta(type(obj),
                                                      no_code=True,
                                                      include_base=True)
        else:
            metas = ReloadableDynamicClass.get_metas_of_regular_methods(
                type(obj), True, no_code=True)
    methods = FlowSpecialMethods(metas)
    if methods.create_preview_layout is not None:
        fn = methods.create_preview_layout.bind(obj)
        layout_flex = fn()
        assert isinstance(
            layout_flex, FlexBox
        ), f"create_preview_layout must return a flexbox when use anylayout, {type(layout_flex)}"
        # set _flow_comp_def_path to this object
        layout_flex._flow_comp_def_path = _get_obj_def_path(obj)
        layout_flex._wrapped_obj = obj
        return layout_flex
    raise ValueError(
        f"wrapped object must define a zero-arg function with @marker.mark_create_preview_layout and return a flexbox"
    )


@dataclasses.dataclass
class GridItemProps:
    i: str
    x: int
    y: int
    w: int
    h: int
    minW: Union[Undefined, int] = undefined
    maxW: Union[Undefined, int] = undefined
    minH: Union[Undefined, int] = undefined
    maxH: Union[Undefined, int] = undefined
    static: Union[Undefined, bool] = undefined
    isDraggable: Union[Undefined, bool] = undefined
    isResizable: Union[Undefined, bool] = undefined
    resizeHandles: Union[Undefined,
                         List[Literal["s", "w", "e", "n", "sw", "nw", "se",
                                      "ne"]]] = undefined
    isBounded: Union[Undefined, bool] = undefined


@dataclasses.dataclass
class GridLayoutProps(MUIFlexBoxProps):
    width: Union[Undefined, int] = undefined
    autoSize: Union[bool, Undefined] = undefined
    cols: Union[int, Undefined] = undefined
    draggableHandle: Union[Undefined, str] = undefined
    rowHeight: Union[Undefined, int] = undefined


@dataclasses.dataclass
class GridItem:
    component: Component
    name: str
    props: GridItemProps
    flexProps: Union[Undefined, MUIFlexBoxProps] = undefined


class GridLayout(MUIContainerBase[GridLayoutProps, MUIComponentType]):
    # we need to take ref of child, so we must use complex layout here.
    @dataclasses.dataclass
    class ChildDef:
        childs: List[GridItem]

    GridItem: TypeAlias = GridItem

    def __init__(self, children: List[GridItem]) -> None:
        super().__init__(UIType.GridLayout, GridLayoutProps,
                         GridLayout.ChildDef(children))

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class Anchor:
    vertical: Literal["top", "center", "bottom"]
    horizontal: Literal["left", "center", "right"]

@dataclasses.dataclass
class MenuItem:
    id: str
    label: Union[Undefined, str] = undefined
    icon: Union[IconType, Undefined, str] = undefined
    inset: Union[Undefined, bool] = undefined
    iconSize: Union[Undefined, Literal["inherit", "large", "medium",
                                       "small"]] = undefined
    iconFontSize: Union[Undefined, NumberType, str] = undefined
    divider: Union[Undefined, bool] = undefined
    autoFocus: Union[Undefined, bool] = undefined
    disableAutoFocusItem: Union[Undefined, bool] = undefined


@dataclasses.dataclass
class MenuListProps(MUIFlexBoxProps):
    dense: Union[Undefined, bool] = undefined
    disablePadding: Union[Undefined, bool] = undefined
    paperProps: Union[Undefined, PaperProps] = undefined
    boxProps: Union[Undefined, MUIFlexBoxProps] = undefined
    triggerMethod: Union[Undefined, Literal["click",
                                            "contextmenu"]] = undefined
    anchorOrigin: Union[Undefined, Anchor] = undefined
    transformOrigin: Union[Undefined, Anchor] = undefined
    menuItems: Union[List[MenuItem], Undefined] = undefined

    # @model_validator(mode='after')
    # def _validator_post_root(self) -> 'MenuListProps':
    #     assert not isinstance(self.menuItems, Undefined), "menuItems must be provided"
    #     return self


class MenuList(MUIContainerBase[MenuListProps, MUIComponentType]):

    @dataclasses.dataclass
    class ChildDef:
        component: Component

    def __init__(
            self,
            items: List[MenuItem],
            child: Component,
            callback: Optional[Callable[[str], _CORO_NONE]] = None) -> None:
        super().__init__(
            UIType.MenuList,
            MenuListProps,
            MenuList.ChildDef(child),
            allowed_events=[FrontendEventType.ContextMenuSelect.value])
        if callback is not None:
            self.register_event_handler(
                FrontendEventType.ContextMenuSelect.value,
                callback,
                simple_event=True)
        self.event_contextmenu_select = self._create_event_slot(
            FrontendEventType.ContextMenuSelect)
        self.prop(menuItems=items)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)
