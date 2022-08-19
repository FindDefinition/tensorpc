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

from ..core import AppEvent, Component, TaskLoopEvent, UIEvent, UIType, ContainerBase, Undefined, undefined
from .mui import _encode_image_bytes

Vector3Type = Tuple[float, float, float]


class ThreeComponentBase(Component):
    pass


class Object3dBase(ThreeComponentBase):

    def __init__(self,
                 base_type: UIType,
                 position: Union[Vector3Type, Undefined] = undefined,
                 rotation: Union[Vector3Type, Undefined] = undefined,
                 up: Union[Vector3Type, Undefined] = undefined,
                 scale: Union[Vector3Type, Undefined] = undefined,
                 visible: Union[Vector3Type, Undefined] = undefined,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(uid, base_type, queue, flex, align_self)
        self.visible = visible
        self.position = position
        self.rotation = rotation
        self.scale = scale
        self.up = up

    def get_state(self):
        state = super().get_state()
        if self.position is not undefined:
            state["position"] = self.position
        if self.rotation is not undefined:
            state["rotation"] = self.rotation
        if self.scale is not undefined:
            state["scale"] = self.scale
        if self.up is not undefined:
            state["up"] = self.up
        if self.visible is not undefined:
            state["visible"] = self.visible
        return state

    def update_object3d_event(self,
                              position: Optional[Union[Vector3Type,
                                                       Undefined]] = None,
                              rotation: Optional[Union[Vector3Type,
                                                       Undefined]] = None,
                              up: Optional[Union[Vector3Type,
                                                 Undefined]] = None,
                              scale: Optional[Union[Vector3Type,
                                                    Undefined]] = None,
                              visible: Optional[Union[Undefined,
                                                      bool]] = None):
        """if not none, updated
        """
        upd: Dict[str, Any] = {}
        if position is not None:
            self.position = position
            upd["position"] = position
        if rotation is not None:
            self.rotation = rotation
            upd["rotation"] = rotation
        if up is not None:
            self.up = up
            upd["up"] = up
        if scale is not None:
            self.scale = scale
            upd["scale"] = scale
        if visible is not None:
            self.visible = visible
            upd["visible"] = visible
        return self.create_update_event(upd)

    async def update_object3d(self,
                              position: Optional[Union[Vector3Type,
                                                       Undefined]] = None,
                              rotation: Optional[Union[Vector3Type,
                                                       Undefined]] = None,
                              up: Optional[Union[Vector3Type,
                                                 Undefined]] = None,
                              scale: Optional[Union[Vector3Type,
                                                    Undefined]] = None,
                              visible: Optional[Union[Undefined,
                                                      bool]] = None):
        """if not none, updated
        """
        await self.queue.put(
            self.update_object3d_event(position, rotation, up, scale, visible))


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
                 transparent: bool = True,
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


class BoundingBox(Object3dBase):

    def __init__(self,
                 dimersion: Vector3Type,
                 position: Union[Vector3Type, Undefined] = undefined,
                 rotation: Union[Vector3Type, Undefined] = undefined,
                 up: Union[Vector3Type, Undefined] = undefined,
                 scale: Union[Vector3Type, Undefined] = undefined,
                 visible: Union[Vector3Type, Undefined] = undefined,
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
        super().__init__(UIType.ThreeBoundingBox, position, rotation, up,
                         scale, visible, uid, queue, flex, align_self)
        self.dimersion = dimersion
        self.edgeWidth = edgeWidth
        self.edgeColor = edgeColor
        self.emissive = emissive
        self.color = color
        self.opacity = opacity
        self.edgeOpacity = edgeOpacity

    def get_state(self):
        state = super().get_state()
        state.update({
            "dimersion": self.dimersion,
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
                 position: Union[Vector3Type, Undefined] = undefined,
                 rotation: Union[Vector3Type, Undefined] = undefined,
                 up: Union[Vector3Type, Undefined] = undefined,
                 scale: Union[Vector3Type, Undefined] = undefined,
                 visible: Union[Vector3Type, Undefined] = undefined,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 _init_dict: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.ThreeGroup, uid, queue, None, None,
                         uid_to_comp, _init_dict, inited)
        self.visible = visible
        self.position = position
        self.rotation = rotation
        self.scale = scale
        self.up = up

    def get_state(self):
        state = super().get_state()
        state.update({
            "position": self.position,
            "rotation": self.rotation,
            "scale": self.scale,
            "up": self.up,
            "visible": self.visible,
        })
        return state

    async def update_object3d(self,
                              position: Optional[Union[Vector3Type,
                                                       Undefined]] = None,
                              rotation: Optional[Union[Vector3Type,
                                                       Undefined]] = None,
                              up: Optional[Union[Vector3Type,
                                                 Undefined]] = None,
                              scale: Optional[Union[Vector3Type,
                                                    Undefined]] = None,
                              visible: Optional[Union[Undefined,
                                                      bool]] = None):
        """if not none, updated
        """
        upd: Dict[str, Any] = {}
        if position is not None:
            self.position = position
            upd["position"] = position
        if rotation is not None:
            self.rotation = rotation
            upd["rotation"] = rotation
        if up is not None:
            self.up = up
            upd["up"] = up
        if scale is not None:
            self.scale = scale
            upd["scale"] = scale
        if visible is not None:
            self.visible = visible
            upd["visible"] = visible
        await self.queue.put(self.create_update_event(upd))


def group(init_dict: Dict[str, Union[ThreeComponentBase, Group]]):
    init_dict_anno: Dict[str, Component] = {**init_dict}
    return Group(_init_dict=init_dict_anno)


class Image(Object3dBase):

    def __init__(self,
                 position: Union[Vector3Type, Undefined] = undefined,
                 rotation: Union[Vector3Type, Undefined] = undefined,
                 up: Union[Vector3Type, Undefined] = undefined,
                 scale: Union[Vector3Type, Undefined] = undefined,
                 visible: Union[Vector3Type, Undefined] = undefined,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(UIType.ThreeImage, position, rotation, up, scale,
                         visible, uid, queue, flex, align_self)
        self.image_str: bytes = b""

    async def show(self, image: np.ndarray):
        encoded = _encode_image_bytes(image)
        self.image_str = encoded
        await self.queue.put(self.create_update_event({
            "image": encoded,
        }))

    async def show_raw(self, image_bytes: bytes, suffix: str):
        await self.queue.put(self.show_raw_event(image_bytes, suffix))

    def show_raw_event(self, image_bytes: bytes, suffix: str):
        raw = b'data:image/' + suffix.encode(
            "utf-8") + b';base64,' + base64.b64encode(image_bytes)
        self.image_str = raw
        return self.create_update_event({
            "image": raw,
        })

    def get_state(self):
        state = super().get_state()
        state["image"] = self.image_str
        return state


class PerspectiveCamera(Object3dBase):

    def __init__(self,
                 makeDefault: bool,
                 position: Union[Vector3Type, Undefined] = undefined,
                 rotation: Union[Vector3Type, Undefined] = undefined,
                 up: Union[Vector3Type, Undefined] = undefined,
                 scale: Union[Vector3Type, Undefined] = undefined,
                 visible: Union[Vector3Type, Undefined] = undefined,
                 fov: Union[float, Undefined] = undefined,
                 aspect: Union[float, Undefined] = undefined,
                 near: Union[float, Undefined] = undefined,
                 far: Union[float, Undefined] = undefined,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(UIType.ThreePerspectiveCamera, position, rotation, up,
                         scale, visible, uid, queue, flex, align_self)
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

    async def update_parameters(self,
                                fov: Optional[Union[float, Undefined]] = None,
                                aspect: Optional[Union[float,
                                                       Undefined]] = None,
                                near: Optional[Union[float, Undefined]] = None,
                                far: Optional[Union[float, Undefined]] = None):
        upd: Dict[str, Any] = {}
        if fov is not None:
            self.fov = fov
            upd["fov"] = fov
        if aspect is not None:
            self.aspect = aspect
            upd["aspect"] = aspect
        if near is not None:
            self.near = near
            upd["near"] = near
        if far is not None:
            self.far = far
            upd["far"] = far
        await self.queue.put(self.create_update_event(upd))

    def get_state(self):
        state = super().get_state()
        state.update({
            "fov": self.fov,
            "aspect": self.aspect,
            "near": self.near,
            "far": self.far,
        })
        return state


class OrthographicCamera(Object3dBase):

    def __init__(self,
                 makeDefault: bool,
                 position: Union[Vector3Type, Undefined] = undefined,
                 rotation: Union[Vector3Type, Undefined] = undefined,
                 up: Union[Vector3Type, Undefined] = undefined,
                 scale: Union[Vector3Type, Undefined] = undefined,
                 visible: Union[Vector3Type, Undefined] = undefined,
                 near: Optional[float] = None,
                 far: Optional[float] = None,
                 zoom: Optional[float] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 flex: Optional[Union[int, str]] = None,
                 align_self: Optional[str] = None) -> None:
        super().__init__(UIType.ThreeOrthographicCamera, position, rotation,
                         up, scale, visible, uid, queue, flex, align_self)
        self.zoom = zoom
        self.position = position
        self.near = near
        self.far = far
        self.makeDefault = makeDefault

    # TODO from camera matrix and intrinsics
    def to_dict(self):
        res = super().to_dict()
        res["makeDefault"] = self.makeDefault
        return res

    async def update_parameters(self,
                                zoom: Optional[Union[float, Undefined]] = None,
                                near: Optional[Union[float, Undefined]] = None,
                                far: Optional[Union[float, Undefined]] = None):
        upd: Dict[str, Any] = {}
        if zoom is not None:
            self.zoom = zoom
            upd["zoom"] = zoom
        if near is not None:
            self.near = near
            upd["near"] = near
        if far is not None:
            self.far = far
            upd["far"] = far
        await self.queue.put(self.create_update_event(upd))

    def get_state(self):
        state = super().get_state()
        state.update({
            "zoom": self.zoom,
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
        super().__init__(uid, UIType.ThreeOrbitControl, queue, flex,
                         align_self)
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
