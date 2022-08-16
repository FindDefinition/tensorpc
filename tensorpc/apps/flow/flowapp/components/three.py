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
from tensorpc.utils.uniquename import UniqueNamePool

from ..core import AppEvent, Component, TaskLoopEvent, UIEvent, UIType, ContainerBase


class ThreeComponentBase(Component):
    pass


class Points(ThreeComponentBase):

    def __init__(self,
                limit: int,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.ThreePoints, queue, flex, align_self)
        self.points = np.zeros((0, 3), np.float32)

        self.size = 3.0
        self.limit = limit
        self.intensity: Optional[np.ndarray] = None
        self.color: Optional[np.ndarray] = None
        self.attrs: Optional[np.ndarray] = None
        self.sizeAttenuation = False
        self.attrs: Optional[np.ndarray] = None
        self.attr_fields: Optional[List[str]] = None

    def to_dict(self):
        res = super().to_dict()
        res["limit"] = self.limit
        return res

    async def update_points(self,
                            points: np.ndarray,
                            intensity: Optional[np.ndarray] = None,
                            color: Optional[np.ndarray] = None,
                            attrs: Optional[np.ndarray] = None,
                            attr_fields: Optional[List[str]] = None):
        assert points.shape[0] <= self.limit, f"your points size must smaller than limit {self.limit}"
        upd: Dict[str, Any] = {
            "points": points,
        }
        if intensity is not None:
            upd["intensity"] = intensity
        if color is not None:
            upd["color"] = color
        if attrs is not None:
            if attrs.ndim == 1:
                attrs = attrs.reshape(-1, 1)
            if attr_fields is None:
                attr_fields = [f"{i}" for i in range(attrs.shape[1])]
            upd["attrs"] = attrs
            upd["attrFields"] = attr_fields

        self.points = points
        self.intensity = intensity
        self.color = color
        self.attrs = attrs
        self.attr_fields = attr_fields
        await self.queue.put(self.create_update_event(upd))

    def get_state(self):
        state = super().get_state()
        state["points"] = self.points
        state["size"] = self.size
        state["sizeAttenuation"] = self.sizeAttenuation
        if self.intensity is not None:
            state["intensity"] = self.intensity
        if self.color is not None:
            state["color"] = self.color
        if self.attrs is not None:
            assert self.attr_fields is not None, "you must provide attr fields"
            state["attrs"] = self.attrs
            state["attrFields"] = self.attr_fields

        return state

class BoundingBox(ThreeComponentBase):

    def __init__(self,
                dim: list,
                pos: list,
                rot: list,
                edgeWidth: float = 4,
                edgeColor: str = "green",
                emissive: str = "red",
                color: str = "red",
                opacity: float =  0.5,
                edgeOpacity: float = 0.5,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.ThreeBoundingBox, queue, flex, align_self)
        self.dim = dim
        self.pos = pos
        self.rot = rot
        self.edgeWidth = edgeWidth
        self.edgeColor = edgeColor
        self.emissive = emissive
        self.color = color
        self.opacity = opacity
        self.edgeOpacity = edgeOpacity

    def get_state(self):
        state = super().get_state()
        state = super().get_state()
        state.update({
            "dim": self.dim,
            "pos": self.pos,
            "rot": self.rot,
            "edgeWidth": self.edgeWidth,
            "edgeColor": self.edgeColor,
            "emissive": self.emissive,
            "color": self.color,
            "opacity": self.opacity,
            "edgeOpacity": self.edgeOpacity,
        })
        return state

class PerspectiveCamera(ThreeComponentBase):

    def __init__(self,
                makeDefault: bool,
                 position: List[float],
                 up: List[float],
                 fov: float,
                 aspect: float,
                 near: float,
                 far: float,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.ThreePerspectiveCamera, queue, flex, align_self)
        self.position = position
        self.up = up
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far
        self.makeDefault = makeDefault

    # TODO from camera matrix and intrinsics
    def to_dict(self):
        res = super().to_dict()
        res["makeDefault"] = self.makeDefault
        return res


    async def update_parameters(self, position: List[float], up: List[float],
                                fov: float, aspect: float, near: float,
                                far: float):
        upd: Dict[str, Any] = {
            "position": position,
            "up": up,
            "fov": fov,
            "aspect": aspect,
            "near": near,
            "far": far,
        }
        self.position = position
        self.up = up
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far
        await self.queue.put(self.create_update_event(upd))

    def get_state(self):
        state = super().get_state()
        state.update({
            "position": self.position,
            "up": self.up,
            "fov": self.fov,
            "aspect": self.aspect,
            "near": self.near,
            "far": self.far,

        })
        return state


class MapControl(ThreeComponentBase):

    def __init__(self,
                 enableDamping: bool,
                 dampingFactor: float,
                 minDistance: float,
                 maxDistance: float,
                 maxPolarAngle: float = np.pi,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.ThreeMapControl, queue, flex, align_self)
        self.enableDamping = enableDamping
        self.dampingFactor = dampingFactor
        self.minDistance = minDistance
        self.maxDistance = maxDistance
        self.maxPolarAngle = maxPolarAngle

    # TODO from camera matrix and intrinsics

    async def update_parameters(self,
                                enableDamping: bool,
                                dampingFactor: float,
                                minDistance: float,
                                maxDistance: float,
                                maxPolarAngle: float = np.pi):
        upd: Dict[str, Any] = {
            "enableDamping": enableDamping,
            "dampingFactor": dampingFactor,
            "minDistance": minDistance,
            "maxDistance": maxDistance,
            "maxPolarAngle": maxPolarAngle,
        }
        self.enableDamping = enableDamping
        self.dampingFactor = dampingFactor
        self.minDistance = minDistance
        self.maxDistance = maxDistance
        self.maxPolarAngle = maxPolarAngle
        await self.queue.put(self.create_update_event(upd))

    def get_state(self):
        state = super().get_state()
        state.update({
            "enableDamping": self.enableDamping,
            "dampingFactor": self.dampingFactor,
            "minDistance": self.minDistance,
            "maxDistance": self.maxDistance,
            "maxPolarAngle": self.maxPolarAngle,
        })
        return state


class ThreeCanvas(ContainerBase):

    def __init__(self,
                 init_dict: Dict[str, ThreeComponentBase],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        _init_dict_anno: Dict[str, Component] = {**init_dict}
        super().__init__(UIType.ThreeCanvas, uid, queue, flex, align_self,
                         uid_to_comp, _init_dict_anno, inited)
