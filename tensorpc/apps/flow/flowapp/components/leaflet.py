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

if compat.Python3_8AndLater:
    from typing import Literal
else:
    from typing_extensions import Literal

import dataclasses

import numpy as np
from tensorpc.utils.uniquename import UniqueNamePool
from typing_extensions import ParamSpec, TypeAlias

from ..core import (BasicProps, Component,
                    ContainerBase, NumberType, T_base_props, T_child,
                    UIRunStatus, UIType, Undefined,
                    undefined, ContainerBaseProps,
                    T_container_props, Fragment, EventHandler, create_ignore_usr_msg)
from .mui import (FlexBoxProps, MUIComponentType, MUIContainerBase)
from .common import handle_raw_event, handle_standard_event

class MapComponentBase(Component[T_base_props, "MapComponentType"]):
    pass
class MapContainerBase(ContainerBase[T_container_props, T_child]):
    pass

_CORO_NONE = Union[Coroutine[None, None, None], None]

MapComponentType: TypeAlias = Union[MapComponentBase, MapContainerBase, Fragment]

class MapEventType(enum.Enum):
    FlyTo = 0
    SetZoom = 1

class MapEventBase:
    def __init__(self, type: MapEventType) -> None:
        self.type = type

    def to_dict(self):
        return {
            "type": self.type.value 
        }

class MapEventFlyTo(MapEventBase):
    def __init__(self, center: Tuple[NumberType, NumberType], zoom: Optional[NumberType] = None) -> None:
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

class MapFrontendEventType(enum.Enum):
    Zoom = "zoom"
    Move = "move"

class MapContainer(MUIContainerBase[MapContainerProps, MapComponentType]):
    def __init__(self,
                center: Tuple[NumberType, NumberType],
                zoom: NumberType,
                 children: Dict[str, MapComponentType],

                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.LeafletMapContainer, MapContainerProps, uid, queue,
                         uid_to_comp, children, inited)
        self.center = center
        self.zoom = zoom 

    def to_dict(self):
        res = super().to_dict()
        res["center"] = self.center
        res["zoom"] = self.zoom
        return res

    def set_map_callback(
            self,
            on_move: Optional[Union[EventHandler, Undefined]] = None,
            on_zoom: Optional[Union[EventHandler,
                                            Undefined]] = None):
        pointer_event_map = {
            MapFrontendEventType.Move: on_move,
            MapFrontendEventType.Zoom: on_zoom,
        }
        for k, v in pointer_event_map.items():
            if v is not None:
                self._flow_event_handlers[k.value] = v

    async def handle_event(self, ev: Any):
        await handle_raw_event(ev, self)

    async def fly_to(self, center: Tuple[NumberType, NumberType], zoom: Optional[NumberType] = None):
        ev = MapEventFlyTo(center, zoom)
        return await self.send_app_event_and_wait(self.create_uncontrolled_comp_event(ev.to_dict()))

    async def set_zoom(self, zoom: NumberType):
        ev = MapEventSetZoom(zoom)
        return await self.send_app_event_and_wait(self.create_uncontrolled_comp_event(ev.to_dict()))

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
    def __init__(self,
                url: str = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                attribution: str = r"&copy; <a href=\"https://www.openstreetmap.org/copyright\">OpenStreetMap</a> contributors",
                uid: str = "",
                queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.LeafletTileLayer,
                         TileLayerProps, queue)
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
    line_cap: Union[Undefined, Literal['butt', 'round', 'square', 'inherit']] = undefined 
    line_join: Union[Undefined, Literal['miter', 'round', 'bevel', 'inherit']] = undefined 
    dash_array: Union[Undefined, str, List[NumberType]] = undefined 
    dash_offset: Union[Undefined, str] = undefined 
    fill: Union[Undefined, bool] = undefined 
    fill_color: Union[Undefined, str] = undefined 
    fill_opacity: Union[Undefined, NumberType] = undefined 
    fill_rule: Union[Undefined, Literal['nonzero', 'evenodd', 'inherit']] = undefined 


@dataclasses.dataclass
class TooltipProps(ContainerBaseProps):
    sticky: Union[bool, Undefined] = undefined
    opacity: Union[NumberType, Undefined] = undefined
    direction: Union[Literal['right', 'left', 'top', 'bottom', 'center', 'auto'], Undefined] = undefined

class Tooltip(MapContainerBase[TooltipProps, MUIComponentType]):
    def __init__(self,
                 children: Dict[str, MUIComponentType],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(UIType.LeafletTooltip,
                         TooltipProps, uid, queue,
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
    def __init__(self,
                 children: Dict[str, MUIComponentType],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(UIType.LeafletPopup,
                         PopupProps, uid, queue,
                         _children=children)

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
    positions: Union[List[Tuple[NumberType, NumberType]], Undefined] = undefined

class Polyline(MapContainerBase[PolylineProps, MapElementChildType]):
    def __init__(self,
                color: str = "black",
                positions: Union[List[Tuple[NumberType, NumberType]], Undefined] = undefined,
                children: Optional[Dict[str, MapElementChildType]] = None,
                on_click: Optional[Callable[[Tuple[NumberType, NumberType, NumberType]], _CORO_NONE]] = None,
                uid: str = "",
                queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(UIType.LeafletPolyline,
                         PolylineProps, uid, queue, _children=children)
        self.props.color = color 
        self.props.positions = positions
        self.on_click = on_click

    async def handle_event(self, ev: Any):
        await handle_standard_event(self, ev)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def update_positions(self, positions: List[Tuple[NumberType, NumberType]]):
        await self.send_app_event_and_wait(self.update_event(positions=positions))

    def get_callback(self) -> Optional[Callable]:
        return self.on_click

    def set_callback(self, cb: Callable):
        self.on_click = cb

@dataclasses.dataclass
class CircleProps(ContainerBaseProps, PathOptions):
    center: Union[Tuple[NumberType, NumberType], Undefined] = undefined
    radius: Union[Undefined, NumberType] = undefined

class Circle(MapContainerBase[CircleProps, MapElementChildType]):
    def __init__(self,
                 center: Tuple[NumberType, NumberType],
                children: Optional[Dict[str, MapElementChildType]] = None,
                uid: str = "",
                queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(UIType.LeafletCircle,
                         CircleProps, uid, queue, _children=children)
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
                 callback: Optional[Callable[[], _CORO_NONE]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(UIType.LeafletCircleMarker,
                         CircleMarkerProps, uid, queue,
                         _children=children)
        self.props.center = center
        self.callback = callback
    
    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Any):
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
                 callback: Optional[Callable[[], _CORO_NONE]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(UIType.LeafletMarker,
                         MarkerProps, uid, queue,
                         _children=children)
        self.props.position = position
        self.callback = callback
    
    def get_callback(self):
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Any):
        await handle_standard_event(self, ev)
