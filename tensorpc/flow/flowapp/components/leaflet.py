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
import enum
import io
import time
from typing import (Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
                    Iterable, List, Optional, Tuple, Type, TypeVar, Union)

from tensorpc import compat
from tensorpc.flow.flowapp.appcore import Event

if compat.Python3_8AndLater:
    from typing import Literal
else:
    from typing_extensions import Literal

import tensorpc.core.dataclass_dispatch as dataclasses

import numpy as np
from tensorpc.utils.uniquename import UniqueNamePool
from typing_extensions import ParamSpec, TypeAlias

from ..core import (BasicProps, Component, SimpleEventType, ContainerBase,
                    FrontendEventType, NumberType, T_base_props, T_child,
                    UIRunStatus, UIType, Undefined, undefined,
                    ContainerBaseProps, T_container_props, Fragment,
                    EventHandler, create_ignore_usr_msg)
from .mui import (FlexBoxProps, MUIComponentType, MUIContainerBase)
from .common import handle_raw_event, handle_standard_event


class MapComponentBase(Component[T_base_props, "MapComponentType"]):
    pass


class MapContainerBase(ContainerBase[T_container_props, T_child]):
    pass


_CORO_NONE = Union[Coroutine[None, None, None], None]

MapComponentType: TypeAlias = Union[MapComponentBase, MapContainerBase,
                                    Fragment]


class MapEventType(enum.Enum):
    FlyTo = 0
    SetZoom = 1


class MapEventBase:

    def __init__(self, type: MapEventType) -> None:
        self.type = type

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type.value}


class MapEventFlyTo(MapEventBase):

    def __init__(self,
                 center: Tuple[NumberType, NumberType],
                 zoom: Optional[NumberType] = None) -> None:
        super().__init__(MapEventType.FlyTo)
        self.center = center
        self.zoom = zoom

    def to_dict(self):
        res = super().to_dict()
        res["center"] = self.center
        if self.zoom is not None:
            res["zoom"] = self.zoom

        return res


class MapEventSetZoom(MapEventBase):

    def __init__(self, zoom: NumberType) -> None:
        super().__init__(MapEventType.SetZoom)
        self.zoom = zoom

    def to_dict(self):
        res = super().to_dict()
        res["zoom"] = self.zoom
        return res


@dataclasses.dataclass
class MapContainerProps(ContainerBaseProps, FlexBoxProps):
    pass

MapLayoutType = Union[List[MapComponentType], Dict[str,
                                    MapComponentType]]
class MapContainer(MUIContainerBase[MapContainerProps, MapComponentType]):

    def __init__(self,
                 center: Tuple[NumberType, NumberType],
                 zoom: NumberType,
                 children: MapLayoutType,
                 inited: bool = False) -> None:
        if children is not None and isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        allow_evs = [
            FrontendEventType.MapZoom.value, FrontendEventType.MapMove.value
        ]
        super().__init__(UIType.LeafletMapContainer, MapContainerProps,
                         children, inited, allow_evs)
        self.center = center
        self.zoom = zoom
        self.event_move = self._create_event_slot(FrontendEventType.MapMove)
        self.event_zoom = self._create_event_slot(FrontendEventType.MapZoom)

    def to_dict(self):
        res = super().to_dict()
        res["center"] = self.center
        res["zoom"] = self.zoom
        return res

    async def handle_event(self, ev: Event, is_sync: bool = False):
        await handle_raw_event(ev, self, just_run=True)

    async def fly_to(self,
                     center: Tuple[NumberType, NumberType],
                     zoom: Optional[NumberType] = None):
        ev = MapEventFlyTo(center, zoom)
        return await self.send_and_wait(self.create_comp_event(ev.to_dict()))

    async def set_zoom(self, zoom: NumberType):
        ev = MapEventSetZoom(zoom)
        return await self.send_and_wait(self.create_comp_event(ev.to_dict()))

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class TileLayerProps(BasicProps):
    attribution: Union[Undefined, str] = undefined
    url: str = ""


class TileLayer(MapComponentBase[TileLayerProps]):
    """see https://leaflet-extras.github.io/leaflet-providers/preview/
    for all leaflet providers.
    """
    def __init__(
        self,
        url: str = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attribution:
        str = r"&copy; <a href=\"https://www.openstreetmap.org/copyright\">OpenStreetMap</a> contributors"
    ) -> None:
        super().__init__(UIType.LeafletTileLayer, TileLayerProps)
        self.props.url = url
        self.props.attribution = attribution

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class PathOptions:
    stroke: Union[Undefined, bool] = undefined
    color: Union[Undefined, str] = undefined
    weight: Union[Undefined, NumberType] = undefined
    opacity: Union[Undefined, NumberType] = undefined
    lineCap: Union[Undefined, Literal['butt', 'round', 'square',
                                       'inherit']] = undefined
    lineJoin: Union[Undefined, Literal['miter', 'round', 'bevel',
                                        'inherit']] = undefined
    dashArray: Union[Undefined, str, List[NumberType]] = undefined
    dashOffset: Union[Undefined, str] = undefined
    fill: Union[Undefined, bool] = undefined
    fillColor: Union[Undefined, str] = undefined
    fillOpacity: Union[Undefined, NumberType] = undefined
    fillRule: Union[Undefined, Literal['nonzero', 'evenodd',
                                        'inherit']] = undefined


@dataclasses.dataclass
class TooltipProps(ContainerBaseProps):
    sticky: Union[bool, Undefined] = undefined
    opacity: Union[NumberType, Undefined] = undefined
    direction: Union[Literal['right', 'left', 'top', 'bottom', 'center',
                             'auto'], Undefined] = undefined


class Tooltip(MapContainerBase[TooltipProps, MUIComponentType]):

    def __init__(
        self,
        children: Dict[str, MUIComponentType],
    ) -> None:
        super().__init__(UIType.LeafletTooltip,
                         TooltipProps,
                         _children=children)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class PopupProps(ContainerBaseProps):
    pass


class Popup(MapContainerBase[PopupProps, MUIComponentType]):

    def __init__(
        self,
        children: Dict[str, MUIComponentType],
    ) -> None:
        super().__init__(UIType.LeafletPopup, PopupProps, _children=children)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


MapElementChildType: TypeAlias = Union[Tooltip, Popup]


@dataclasses.dataclass
class PolylineProps(ContainerBaseProps, PathOptions):
    positions: Union[List[Tuple[NumberType, NumberType]],
                     Undefined] = undefined


class Polyline(MapContainerBase[PolylineProps, MapElementChildType]):

    def __init__(
        self,
        color: str = "black",
        positions: Union[List[Tuple[NumberType, NumberType]],
                         Undefined] = undefined,
        children: Optional[Dict[str, MapElementChildType]] = None,
    ) -> None:
        super().__init__(UIType.LeafletPolyline,
                         PolylineProps,
                         _children=children,
                         allowed_events=[
                                FrontendEventType.Click.value,
                         ])
        self.props.color = color
        self.props.positions = positions
        self.event_click = self._create_event_slot(FrontendEventType.Click)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        await handle_standard_event(self, ev, is_sync=is_sync)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def update_positions(self, positions: List[Tuple[NumberType,
                                                           NumberType]]):
        await self.send_and_wait(self.update_event(positions=positions))


@dataclasses.dataclass
class CircleProps(ContainerBaseProps, PathOptions):
    center: Union[Tuple[NumberType, NumberType], Undefined] = undefined
    radius: Union[Undefined, NumberType] = undefined


class Circle(MapContainerBase[CircleProps, MapElementChildType]):

    def __init__(
            self,
            center: Tuple[NumberType, NumberType],
            children: Optional[Dict[str, MapElementChildType]] = None) -> None:
        super().__init__(UIType.LeafletCircle, CircleProps, _children=children)
        self.props.center = center

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class CircleMarkerProps(ContainerBaseProps, PathOptions):
    center: Union[Tuple[NumberType, NumberType], Undefined] = undefined
    radius: Union[Undefined, NumberType] = undefined


class CircleMarker(MapContainerBase[CircleMarkerProps, MapElementChildType]):

    def __init__(self,
                 center: Tuple[NumberType, NumberType],
                 children: Dict[str, MapElementChildType],
                 callback: Optional[Callable[[], _CORO_NONE]] = None) -> None:
        super().__init__(UIType.LeafletCircleMarker,
                         CircleMarkerProps,
                         _children=children,
                         allowed_events=[FrontendEventType.Click.value])
        self.props.center = center
        if callback is not None:
            self.register_event_handler(FrontendEventType.Click.value,
                                        callback)
        self.event_click = self._create_event_slot(FrontendEventType.Click)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        await handle_standard_event(self, ev)


@dataclasses.dataclass
class MarkerProps(ContainerBaseProps):
    position: Union[Tuple[NumberType, NumberType], Undefined] = undefined
    opacity: Union[NumberType, Undefined] = undefined
    title: Union[str, Undefined] = undefined


class Marker(MapContainerBase[MarkerProps, MapElementChildType]):

    def __init__(self,
                 position: Tuple[NumberType, NumberType],
                 children: Dict[str, MapElementChildType],
                 callback: Optional[Callable[[], _CORO_NONE]] = None) -> None:
        super().__init__(UIType.LeafletMarker,
                         MarkerProps,
                         _children=children,
                         allowed_events=[FrontendEventType.Click.value])
        self.props.position = position
        if callback is not None:
            self.register_event_handler(FrontendEventType.Click.value,
                                        callback)
        self.event_click = self._create_event_slot(FrontendEventType.Click)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        await handle_standard_event(self, ev)
