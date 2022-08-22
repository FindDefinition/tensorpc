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
from tensorpc import compat
from typing import (Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
                    Iterable, List, Optional, Tuple, TypeVar, Union)

if compat.Python3_8AndLater:
    from typing import Literal
else:
    from typing_extensions import Literal

import numpy as np
from tensorpc.utils.uniquename import UniqueNamePool
import dataclasses
from ..core import AppEvent, BasicProps, Component, TaskLoopEvent, UIEvent, UIType, ContainerBase, Undefined, undefined, ComponentBaseProps, T
from .mui import FlexBoxProps, _encode_image_bytes, MUIComponentType

Vector3Type = Tuple[float, float, float]


@dataclasses.dataclass
class ThreeFlexItemBoxProps(ComponentBaseProps):
    center_anchor: Union[bool, Undefined] = undefined  # false
    enable_flex: Union[bool, Undefined] = undefined  # false


@dataclasses.dataclass
class ThreeBasicProps(BasicProps):
    pass


@dataclasses.dataclass
class ThreeFlexPropsBase(FlexBoxProps):
    pass


class ThreeComponentBase(Component[T]):
    pass


class ThreeContainerBase(ContainerBase[T]):
    pass


ThreeComponentType = Union[ThreeComponentBase, ThreeContainerBase,
                           ThreeBasicProps, ThreeFlexPropsBase,
                           ThreeFlexItemBoxProps]


@dataclasses.dataclass
class Object3dBaseProps(ThreeFlexItemBoxProps):
    position: Union[Vector3Type, Undefined] = undefined
    rotation: Union[Vector3Type, Undefined] = undefined
    up: Union[Vector3Type, Undefined] = undefined
    scale: Union[Vector3Type, Undefined] = undefined
    visible: Union[bool, Undefined] = undefined


class Object3dBase(ThreeComponentBase[Object3dBaseProps]):

    def __init__(self,
                 base_type: UIType,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, base_type, Object3dBaseProps, queue)

    def get_state(self):
        state = super().get_state()
        if self.props.position is not undefined:
            state["position"] = self.props.position
        if self.props.rotation is not undefined:
            state["rotation"] = self.props.rotation
        if self.props.scale is not undefined:
            state["scale"] = self.props.scale
        if self.props.up is not undefined:
            state["up"] = self.props.up
        if self.props.visible is not undefined:
            state["visible"] = self.props.visible
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
            self.props.position = position
            upd["position"] = position
        if rotation is not None:
            self.props.rotation = rotation
            upd["rotation"] = rotation
        if up is not None:
            self.props.up = up
            upd["up"] = up
        if scale is not None:
            self.props.scale = scale
            upd["scale"] = scale
        if visible is not None:
            self.props.visible = visible
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


class Points(ThreeComponentBase[ThreeBasicProps]):

    def __init__(self,
                 limit: int,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreePoints, ThreeBasicProps, queue)
        self.points = np.zeros((0, 3), np.float32)

        self.size = 3.0
        self.limit = limit
        self.intensity: Optional[np.ndarray] = None
        self.colors: Optional[np.ndarray] = None
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
                            colors: Optional[np.ndarray] = None,
                            attrs: Optional[np.ndarray] = None,
                            attr_fields: Optional[List[str]] = None):
        assert points.shape[
            0] <= self.limit, f"your points size must smaller than limit {self.limit}"
        upd: Dict[str, Any] = {
            "points": points,
        }
        if intensity is not None:
            upd["intensity"] = intensity
        if colors is not None:
            upd["colors"] = colors
        if attrs is not None:
            if attrs.ndim == 1:
                attrs = attrs.reshape(-1, 1)
            if attr_fields is None:
                attr_fields = [f"{i}" for i in range(attrs.shape[1])]
            upd["attrs"] = attrs
            upd["attrFields"] = attr_fields

        self.points = points
        self.intensity = intensity
        self.colors = colors
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
        if self.colors is not None:
            state["color"] = self.colors
        if self.attrs is not None:
            assert self.attr_fields is not None, "you must provide attr fields"
            state["attrs"] = self.attrs
            state["attrFields"] = self.attr_fields

        return state


class Segments(ThreeComponentBase[ThreeBasicProps]):

    def __init__(self,
                 limit: int,
                 line_width: float = 1.0,
                 color: Optional[str] = "black",
                 transparent: bool = True,
                 opacity: float = 0.5,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeSegments, ThreeBasicProps, queue)
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


class Boxes2D(Object3dBase):

    def __init__(self,
                 limit: int,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(UIType.ThreeBoxes2D, uid, queue)
        self.centers = np.zeros((0, 2), np.float32)
        self.dimersions = np.zeros((0, 2), np.float32)
        self.colors: Union[np.ndarray, Undefined] = undefined
        self.limit = limit
        self.color: Union[str, Undefined] = undefined
        self.attrs: Union[List[str], Undefined] = undefined
        self.alpha: Union[float, Undefined] = undefined

    def to_dict(self):
        res = super().to_dict()
        res["limit"] = self.limit
        return res

    async def update_boxes(self,
                           centers: Optional[np.ndarray] = None,
                           dimersions: Optional[np.ndarray] = None,
                           colors: Optional[Union[np.ndarray,
                                                  Undefined]] = None,
                           color: Optional[Union[str, Undefined]] = None,
                           attrs: Optional[Union[List[str], Undefined]] = None,
                           alpha: Optional[Union[float, Undefined]] = None):
        upd: Dict[str, Any] = {}
        if centers is not None:
            assert centers.shape[
                0] <= self.limit, f"your centers size must smaller than limit {self.limit}"
            self.centers = centers
            upd["centers"] = centers

        if dimersions is not None:
            assert dimersions.shape[
                0] <= self.limit, f"your dimersions size must smaller than limit {self.limit}"
            self.dimersions = dimersions
            upd["dimersions"] = dimersions

        if colors is not None:
            if not isinstance(colors, Undefined):
                assert colors.shape[
                    0] <= self.limit, f"your colors size must smaller than limit {self.limit}"
            self.colors = colors
            upd["colors"] = colors
        if color is not None:
            self.color = color
            upd["color"] = color
        if attrs is not None:
            self.attrs = attrs
            upd["attrs"] = attrs
        if alpha is not None:
            self.alpha = alpha
            upd["alpha"] = alpha

        await self.queue.put(self.create_update_event(upd))

    def get_state(self):
        state = super().get_state()
        state.update({
            "colors": self.colors,
            "color": self.color,
            "centers": self.centers,
            "dimersions": self.dimersions,
            "attrs": self.attrs,
            "alpha": self.alpha,
        })
        return state


class BoundingBox(Object3dBase):

    def __init__(self,
                 dimersion: Vector3Type,
                 edgeWidth: float = 4,
                 edgeColor: str = "green",
                 emissive: str = "red",
                 color: str = "red",
                 opacity: float = 0.5,
                 edgeOpacity: float = 0.5,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(UIType.ThreeBoundingBox, uid, queue)
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


class AxesHelper(ThreeComponentBase[ThreeBasicProps]):

    def __init__(self,
                 length: float,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeAxesHelper, ThreeBasicProps, queue)
        self.length = length

    def to_dict(self):
        res = super().to_dict()
        res["length"] = self.length
        return res


class InfiniteGridHelper(ThreeComponentBase[ThreeBasicProps]):

    def __init__(self,
                 size1: float,
                 size2: float,
                 color: str,
                 distance: float = 8000,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeInfiniteGridHelper, ThreeBasicProps,
                         queue)
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


class Group(ContainerBase[Object3dBaseProps]):

    def __init__(self,
                 init_dict: Dict[str, ThreeComponentType],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        _init_dict_anno: Optional[Dict[str, Union[Component,
                                                  BasicProps]]] = None
        if init_dict is not None:
            _init_dict_anno = {**init_dict}
        super().__init__(UIType.ThreeGroup, Object3dBaseProps, uid, queue,
                         uid_to_comp, _init_dict_anno, inited)

    def get_state(self):
        state = super().get_state()
        state.update({
            "position": self.props.position,
            "rotation": self.props.rotation,
            "scale": self.props.scale,
            "up": self.props.up,
            "visible": self.props.visible,
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
            self.props.position = position
            upd["position"] = position
        if rotation is not None:
            self.props.rotation = rotation
            upd["rotation"] = rotation
        if up is not None:
            self.props.up = up
            upd["up"] = up
        if scale is not None:
            self.props.scale = scale
            upd["scale"] = scale
        if visible is not None:
            self.props.visible = visible
            upd["visible"] = visible
        await self.queue.put(self.create_update_event(upd))


class Image(Object3dBase):

    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(UIType.ThreeImage, uid, queue)
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
                 fov: Union[float, Undefined] = undefined,
                 aspect: Union[float, Undefined] = undefined,
                 near: Union[float, Undefined] = undefined,
                 far: Union[float, Undefined] = undefined,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(UIType.ThreePerspectiveCamera, uid, queue)
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
                 near: Optional[float] = None,
                 far: Optional[float] = None,
                 zoom: Optional[float] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(UIType.ThreeOrthographicCamera, uid, queue)
        self.zoom = zoom
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


class MapControl(ThreeComponentBase[ThreeBasicProps]):

    def __init__(self,
                 enableDamping: bool,
                 dampingFactor: float,
                 minDistance: float,
                 maxDistance: float,
                 maxPolarAngle: float = np.pi,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeMapControl, ThreeBasicProps, queue)
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


class OrbitControl(ThreeComponentBase[ThreeBasicProps]):

    def __init__(self,
                 enableDamping: bool,
                 dampingFactor: float,
                 minDistance: float,
                 maxDistance: float,
                 maxPolarAngle: float = np.pi,
                 screenSpacePanning: bool = False,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeOrbitControl, ThreeBasicProps, queue)
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


class PointerLockControl(ThreeComponentBase[ThreeBasicProps]):

    def __init__(self,
                 enabled: Union[bool, Undefined] = undefined,
                 minPolarAngle: Union[float, Undefined] = undefined,
                 maxPolarAngle: Union[float, Undefined] = undefined,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreePointerLockControl, ThreeBasicProps,
                         queue)
        self.enabled = enabled
        self.minPolarAngle = minPolarAngle
        self.maxPolarAngle = maxPolarAngle

    def to_dict(self):
        res = super().to_dict()
        if self.enabled is not undefined:
            res["enabled"] = self.enabled
        if self.minPolarAngle is not undefined:
            res["minPolarAngle"] = self.minPolarAngle
        if self.maxPolarAngle is not undefined:
            res["maxPolarAngle"] = self.maxPolarAngle
        return res


@dataclasses.dataclass
class FirstPersonControlProps(ThreeBasicProps):
    enabled: Union[bool, Undefined] = undefined
    movement_speed: Union[float, Undefined] = undefined
    auto_forward: Union[bool, Undefined] = undefined
    look_speed: Union[float, Undefined] = undefined
    look_vertical: Union[bool, Undefined] = undefined
    active_look: Union[bool, Undefined] = undefined
    height_speed: Union[bool, Undefined] = undefined
    height_coef: Union[float, Undefined] = undefined
    height_min: Union[float, Undefined] = undefined
    height_max: Union[float, Undefined] = undefined
    constrain_vertical: Union[bool, Undefined] = undefined
    vertical_min: Union[float, Undefined] = undefined
    vertical_max: Union[float, Undefined] = undefined
    mouse_drag_on: Union[bool, Undefined] = undefined


class FirstPersonControl(ThreeComponentBase[FirstPersonControlProps]):

    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreePointerLockControl,
                         FirstPersonControlProps, queue)


class ThreeCanvas(ContainerBase[ThreeBasicProps]):

    def __init__(self,
                 init_dict: Dict[str, Union[ThreeComponentBase,
                                            ThreeBasicProps]],
                 background: Union[str, Undefined] = undefined,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        _init_dict_anno: Optional[Dict[str, Union[Component,
                                                  BasicProps]]] = None
        if init_dict is not None:
            _init_dict_anno = {**init_dict}
        super().__init__(UIType.ThreeCanvas, ThreeBasicProps, uid, queue,
                         uid_to_comp, _init_dict_anno, inited)
        self.background = background

    def to_dict(self):
        res = super().to_dict()
        if self.background is not undefined:
            res["backgroundColor"] = self.background
        return res


@dataclasses.dataclass
class ThreeFlexProps(ThreeFlexPropsBase):
    size: Union[Vector3Type, Undefined] = undefined
    position: Union[Vector3Type, Undefined] = undefined
    direction: Union[str, Undefined] = undefined
    plane: Union[str, Undefined] = undefined
    scale_factor: Union[int, Undefined] = undefined


class Flex(ContainerBase[ThreeFlexProps]):

    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 _init_dict: Optional[Dict[str, ThreeComponentType]] = None,
                 inited: bool = False) -> None:
        _init_dict_anno: Optional[Dict[str, Union[Component,
                                                  BasicProps]]] = None
        if _init_dict is not None:
            _init_dict_anno = {**_init_dict}
        super().__init__(UIType.ThreeFlex, ThreeFlexProps, uid, queue,
                         uid_to_comp, _init_dict_anno, inited)


class ItemBox(ContainerBase[ThreeFlexItemBoxProps]):
    """if a three item have flex item prop enabled, it will
    be wrapped with a ItemBox automatically.
    """

    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 _init_dict: Optional[Dict[str, ThreeComponentType]] = None,
                 inited: bool = False) -> None:
        _init_dict_anno: Optional[Dict[str, Union[Component,
                                                  BasicProps]]] = None
        if _init_dict is not None:
            _init_dict_anno = {**_init_dict}
        super().__init__(UIType.ThreeFlexItemBox, ThreeFlexItemBoxProps, uid,
                         queue, uid_to_comp, _init_dict_anno, inited)


PointerEventsProperties = Union[Literal["auto"], Literal["none"],
                                Literal["visiblePainted"],
                                Literal["visibleFill"],
                                Literal["visibleStroke"], Literal["visible"],
                                Literal["painted"], Literal["fill"],
                                Literal["stroke"], Literal["all"],
                                Literal["inherit"]]


@dataclasses.dataclass
class HtmlProps(Object3dBaseProps):
    prepend: Union[bool, Undefined] = undefined
    center: Union[bool, Undefined] = undefined
    fullscreen: Union[bool, Undefined] = undefined
    eps: Union[float, Undefined] = undefined
    distance_factor: Union[float, Undefined] = undefined
    sprite: Union[bool, Undefined] = undefined
    transform: Union[bool, Undefined] = undefined
    z_index_range: Union[List[Union[int, float]], Undefined] = undefined
    wrapper_class: Union[str, Undefined] = undefined
    pointer_events: Union[PointerEventsProperties, Undefined] = undefined
    occlude: Union[bool, Undefined] = undefined


class Html(ContainerBase[HtmlProps]):
    """we can use MUI components only in Html.
    """

    def __init__(self,
                 init_dict: Dict[str, MUIComponentType],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        _init_dict_anno: Optional[Dict[str, Union[Component,
                                                  BasicProps]]] = None
        if init_dict is not None:
            _init_dict_anno = {**init_dict}
        super().__init__(UIType.ThreeHtml, HtmlProps, uid, queue, uid_to_comp,
                         _init_dict_anno, inited)

    def get_state(self):
        state = super().get_state()
        state.update({
            "position": self.props.position,
            "rotation": self.props.rotation,
            "scale": self.props.scale,
            "up": self.props.up,
            "visible": self.props.visible,
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
            self.props.position = position
            upd["position"] = position
        if rotation is not None:
            self.props.rotation = rotation
            upd["rotation"] = rotation
        if up is not None:
            self.props.up = up
            upd["up"] = up
        if scale is not None:
            self.props.scale = scale
            upd["scale"] = scale
        if visible is not None:
            self.props.visible = visible
            upd["visible"] = visible
        await self.queue.put(self.create_update_event(upd))
