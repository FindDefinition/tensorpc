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
from tensorpc.utils.uniquename import UniqueNamePool

from ..core import AppEvent, Component, TaskLoopEvent, UIEvent, UIType, ContainerBase
from .mui import _encode_image_bytes

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
        assert points.shape[
            0] <= self.limit, f"your points size must smaller than limit {self.limit}"
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

class Segments(ThreeComponentBase):
    def __init__(self,
                 limit: int,
                 line_width: float = 1.0,
                 color: Optional[str] = "black",
                 transparent: bool = True ,
                 opacity: float = 0.5,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.ThreeSegments, queue, flex, align_self)
        self.lines = np.zeros((0, 2, 3), np.float32)
        self.line_width = line_width
        self.limit = limit
        self.colors: Optional[np.ndarray] = None
        self.color = color
        self.transparent = transparent
        self.opacity = opacity


    def to_dict(self):
        res = super().to_dict()
        res["limit"] = self.limit
        return res

    async def update_lines(self,
                         lines: np.ndarray,
                         colors: Optional[np.ndarray] = None,
                         line_width: Optional[float] = None,
                         color: Optional[str] = None):
        assert lines.ndim == 3 and lines.shape[1] == 2 and lines.shape[
            2] == 3, f"{lines.shape}"
        
        assert lines.shape[
            0] <= self.limit, f"your points size must smaller than limit {self.limit}"
        upd: Dict[str, Any] = {
            "lines": lines,
        }
        if colors is not None:
            upd["colors"] = colors
        if color is not None:
            upd["color"] = color
            self.color = color
        if line_width is not None:
            upd["lineWidth"] = line_width
            self.line_width = line_width

        self.lines = lines.astype(np.float32)
        self.colors = colors
        await self.queue.put(self.create_update_event(upd))

    def get_state(self):
        state = super().get_state()
        state["lines"] = self.lines
        state["lineWidth"] = self.line_width
        if self.colors is not None:
            state["colors"] = self.colors
        state["opacity"] = self.opacity
        state["transparent"] = self.transparent
        state["color"] = self.color
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
                 opacity: float = 0.5,
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


class AxesHelper(ThreeComponentBase):
    def __init__(self,
                 length: float,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.ThreeAxesHelper, queue, flex, align_self)
        self.length = length

    def to_dict(self):
        res = super().to_dict()
        res["length"] = self.length
        return res


class InfiniteGridHelper(ThreeComponentBase):
    def __init__(self,
                 size1: float,
                 size2: float,
                 color: str,
                 distance: float = 8000,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.ThreeInfiniteGridHelper, queue, flex,
                         align_self)
        self.size1 = size1
        self.size2 = size2
        self.color = color
        self.distance = distance

    def to_dict(self):
        res = super().to_dict()
        res["size1"] = self.size1
        res["size2"] = self.size2
        res["color"] = self.color
        res["distance"] = self.distance
        return res


class Group(ContainerBase):
    def __init__(self,
                 position: Optional[Tuple[float, float, float]] = None,
                 rotation: Optional[Tuple[float, float, float]] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 _init_dict: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.ThreeGroup, uid, queue, None, None,
                         uid_to_comp, _init_dict, inited)
        self.visible = True
        self.position = position
        self.rotation = rotation

    def get_state(self):
        state = super().get_state()
        state.update({
            "visible": self.visible,
        })
        if self.position is not None:
            state["position"] = self.position
        if self.rotation is not None:
            state["rotation"] = self.rotation
        return state

    async def change_visible(self, visible: bool):
        self.visible = visible
        await self.queue.put(
            self.create_update_event({"visible": self.visible}))

    async def change_pose(self, position: Optional[Tuple[float, float, float]],
                          rotation: Optional[Tuple[float, float, float]]):

        self.position = position
        self.rotation = rotation

        await self.queue.put(
            self.create_update_event({
                "position": self.position,
                "rotation": self.rotation,
            }))


def group(init_dict: Dict[str, Union[ThreeComponentBase, Group]]):
    init_dict_anno: Dict[str, Component] = {**init_dict}
    return Group(_init_dict=init_dict_anno)

class Image(ThreeComponentBase):
    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.ThreeImage, queue, flex,
                         align_self)
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
        super().__init__(uid, UIType.ThreePerspectiveCamera, queue, flex,
                         align_self)
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

class OrthographicCamera(ThreeComponentBase):
    def __init__(self,
                 makeDefault: bool,
                 position: List[float],
                 up: Optional[List[float]] = None,
                 near: Optional[float] = None,
                 far: Optional[float] = None,
                 zoom: Optional[float] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.ThreeOrthographicCamera, queue, flex,
                         align_self)
        self.position = position
        self.zoom = zoom
        self.position = position
        self.up = up
        self.near = near
        self.far = far
        self.makeDefault = makeDefault

    # TODO from camera matrix and intrinsics
    def to_dict(self):
        res = super().to_dict()
        res["makeDefault"] = self.makeDefault
        return res

    async def update_parameters(self, position: List[float], up: List[float],
                                zoom: float, near: float,
                                far: float):
        upd: Dict[str, Any] = {
            "position": position,
            "up": up,
            "zoom": zoom,
            "near": near,
            "far": far,
        }
        self.position = position
        self.up = up
        self.zoom = zoom
        self.near = near
        self.far = far
        await self.queue.put(self.create_update_event(upd))

    def get_state(self):
        state = super().get_state()
        state.update({
            "position": self.position,
            "up": self.up,
            "zoom": self.zoom,
            "near": self.near,
            "far": self.far,
        })
        if self.up is not None:
            state["up"] = self.up
        if self.zoom is not None:
            state["zoom"] = self.zoom
        if self.near is not None:
            state["near"] = self.near
        if self.far is not None:
            state["far"] = self.far

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

class OrbitControl(ThreeComponentBase):
    def __init__(self,
                 enableDamping: bool,
                 dampingFactor: float,
                 minDistance: float,
                 maxDistance: float,
                 maxPolarAngle: float = np.pi,
                 screenSpacePanning: bool = False,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, UIType.ThreeOrbitControl, queue, flex, align_self)
        self.enableDamping = enableDamping
        self.dampingFactor = dampingFactor
        self.minDistance = minDistance
        self.maxDistance = maxDistance
        self.maxPolarAngle = maxPolarAngle
        self.screenSpacePanning = screenSpacePanning

    # TODO from camera matrix and intrinsics

    async def update_parameters(self,
                                enableDamping: bool,
                                dampingFactor: float,
                                minDistance: float,
                                maxDistance: float,
                                maxPolarAngle: float = np.pi,
                                screenSpacePanning: bool = False):
        upd: Dict[str, Any] = {
            "enableDamping": enableDamping,
            "dampingFactor": dampingFactor,
            "minDistance": minDistance,
            "maxDistance": maxDistance,
            "maxPolarAngle": maxPolarAngle,
            "screenSpacePanning": screenSpacePanning,
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
            "screenSpacePanning": self.screenSpacePanning,
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
