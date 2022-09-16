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

from ..core import (AppEvent, AppEventType, BasicProps, Component,
                    ContainerBase, NumberType, T_base_props, T_child,
                    TaskLoopEvent, UIEvent, UIRunStatus, UIType, Undefined,
                    ValueType, undefined, ContainerBaseProps,
                    T_container_props)
from .mui import (FlexBoxProps, MUIComponentType, MUIContainerBase,
                  _encode_image_bytes, _handle_button_event, PointerEventsProperties)


class MapComponentBase(Component[T_base_props, "MapComponentType"]):
    pass
class MapContainerBase(ContainerBase[T_container_props, T_child]):
    pass

_CORO_NONE = Union[Coroutine[None, None, None], None]

MapComponentType: TypeAlias = Union[MapComponentBase, MapContainerBase]

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

class MapContainer(MUIContainerBase[MapContainerProps, MapComponentType]):
    def __init__(self,
                center: Tuple[NumberType, NumberType],
                zoom: NumberType,
                 init_dict: Dict[str, MapComponentType],

                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.LeafletMapContainer, MapContainerProps, uid, queue,
                         uid_to_comp, init_dict, inited)
        self.center = center
        self.zoom = zoom 

    def to_dict(self):
        res = super().to_dict()
        res["center"] = self.center
        res["zoom"] = self.zoom
        return res

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
class MarkerProps(ContainerBaseProps):
    pass

class Marker(ContainerBase[MarkerProps, MUIComponentType]):
    def __init__(self,
                 init_dict: Dict[str, MUIComponentType],
                 callback: Optional[Callable[[], _CORO_NONE]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(UIType.LeafletMarker,
                         MarkerProps, uid, queue,
                         _init_dict=init_dict)
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
        if self.props.status == UIRunStatus.Running.value:
            # TODO send exception if ignored click
            print("IGNORE EVENT", self.props.status)
            return
        elif self.props.status == UIRunStatus.Stop.value:
            cb1 = self.get_callback()
            if cb1 is not None:
                self._task = asyncio.create_task(self.run_callback(cb1, True, sync_first=False))
            else:
                await self.sync_status(True)

@dataclasses.dataclass
class PolylineProps(BasicProps):
    color: str = "black"
    positions: Union[List[Tuple[NumberType, NumberType]], Undefined] = undefined

class Polyline(MapComponentBase[PolylineProps]):
    def __init__(self,
                color: str = "black",
                positions: Union[List[Tuple[NumberType, NumberType]], Undefined] = undefined,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.LeafletPolyline,
                         PolylineProps, queue)
        self.props.color = color 
        self.props.positions = positions

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
