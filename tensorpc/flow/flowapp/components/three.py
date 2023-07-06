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
import json
import time
from typing import (Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
                    Iterable, List, Optional, Tuple, Type, TypeVar, Union)
import urllib.request
from tensorpc import compat
from tensorpc.core.httpservers.core import JS_MAX_SAFE_INT
from tensorpc.flow.flowapp.appcore import Event, EventDataType
from tensorpc.flow.jsonlike import DataClassWithUndefined
from typing_extensions import Literal

import tensorpc.core.dataclass_dispatch as dataclasses

import numpy as np
from typing_extensions import ParamSpec, TypeAlias

from tensorpc.utils.uniquename import UniqueNamePool

from ..core import (AppEvent, AppEventType, BasicProps, Component,
                    ContainerBase, ContainerBaseProps, EventHandler, SimpleEventType,
                    Fragment, FrontendEventType, NumberType, T_base_props,
                    T_child, T_container_props, TaskLoopEvent, UIEvent,
                    UIRunStatus, UIType, Undefined, ValueType, undefined)
from .mui import (FlexBoxProps, MUIFlexBoxProps, MUIComponentType,
                  MUIContainerBase, PointerEventsProperties,
                  Image as MUIImage)
from .common import handle_standard_event

Vector3Type: TypeAlias = Tuple[float, float, float]

_CORO_NONE: TypeAlias = Union[Coroutine[None, None, None], None]
_CORO_ANY: TypeAlias = Union[Coroutine[Any, None, None], Any]

CORO_NONE: TypeAlias = Union[Coroutine[None, None, None], None]

ThreeLayoutType: TypeAlias = Union[List["ThreeComponentType"],
                                   Dict[str, "ThreeComponentType"]]

P = ParamSpec('P')

class PyDanticConfigForNumpy:
    arbitrary_types_allowed = True


@dataclasses.dataclass
class ThreeBasicProps(BasicProps):
    pass


@dataclasses.dataclass
class R3FlexPropsBase(BasicProps):
    alignContent: Union[str, Undefined] = undefined
    alignItems: Union[str, Undefined] = undefined
    justifyContent: Union[str, Undefined] = undefined
    flexDirection: Union[str, Undefined] = undefined
    flexWrap: Union[str, Undefined] = undefined

    alignSelf: Union[str, Undefined] = undefined
    flexGrow: Union[str, Undefined] = undefined
    flexShrink: Union[str, Undefined] = undefined
    flexBasis: Union[str, Undefined] = undefined

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


class Side(enum.Enum):
    FrontSide = 0
    BackSide = 1
    DoubleSide = 2


SideType: TypeAlias = Literal[0, 1, 2]


class MeshMaterialType(enum.Enum):
    Basic = 0
    Depth = 1
    Lambert = 2
    Matcap = 3
    Normal = 4
    Phong = 5
    Physical = 6
    Standard = 7
    Toon = 8


@dataclasses.dataclass
class ThreeMaterialPropsBase(ThreeBasicProps):
    # deprecated, only works in MeshBasicMaterialV1 and MeshStandardMaterialV1
    materialType: int = 0
    transparent: Union[bool, Undefined] = undefined
    opacity: Union[NumberType, Undefined] = undefined
    depthTest: Union[bool, Undefined] = undefined
    depthWrite: Union[bool, Undefined] = undefined
    alphaTest: Union[NumberType, Undefined] = undefined
    visible: Union[bool, Undefined] = undefined
    side: Union[SideType, Undefined] = undefined


class ThreeComponentBase(Component[T_base_props, "ThreeComponentType"]):
    pass


class ThreeContainerBase(ContainerBase[T_container_props, T_child]):
    pass


class ThreeMaterialBase(ThreeComponentBase[T_base_props]):
    pass


class ThreeGeometryBase(ThreeComponentBase[T_base_props]):
    pass


@dataclasses.dataclass
class ThreeGeometryPropsBase(ThreeBasicProps):
    pass


T_material_prop = TypeVar("T_material_prop", bound=ThreeMaterialPropsBase)
T_geometry_prop = TypeVar("T_geometry_prop", bound=ThreeGeometryPropsBase)

ThreeComponentType = Union[ThreeComponentBase, ThreeContainerBase, Fragment]


@dataclasses.dataclass
class Object3dBaseProps(ThreeBasicProps):
    # position already exists in base flex props, so we use another name
    position: Union[Vector3Type, Undefined] = undefined
    rotation: Union[Vector3Type, Undefined] = undefined
    up: Union[Vector3Type, Undefined] = undefined
    scale: Union[Vector3Type, NumberType, Undefined] = undefined
    visible: Union[bool, Undefined] = undefined
    receiveShadow: Union[bool, Undefined] = undefined
    castShadow: Union[bool, Undefined] = undefined
    renderOrder: Union[NumberType, Undefined] = undefined


@dataclasses.dataclass
class Object3dContainerBaseProps(Object3dBaseProps, ContainerBaseProps):
    pass


T_o3d_prop = TypeVar("T_o3d_prop", bound=Object3dBaseProps)
T_o3d_container_prop = TypeVar("T_o3d_container_prop",
                               bound=Object3dContainerBaseProps)


class Object3dBase(ThreeComponentBase[T_o3d_prop]):
    def __init__(self, base_type: UIType, prop_cls: Type[T_o3d_prop],
                 allowed_events: Optional[Iterable[EventDataType]] = None) -> None:
        super().__init__(base_type, prop_cls, allowed_events)

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
        await self.send_and_wait(
            self.update_object3d_event(position, rotation, up, scale, visible))


class Object3dWithEventBase(Object3dBase[T_o3d_prop]):
    def __init__(self, base_type: UIType, prop_cls: Type[T_o3d_prop]) -> None:
        super().__init__(base_type, prop_cls, allowed_events=[
            FrontendEventType.Click.value,
            FrontendEventType.DoubleClick.value,
            FrontendEventType.Enter.value,
            FrontendEventType.Leave.value,
            FrontendEventType.Over.value,
            FrontendEventType.Out.value,
            FrontendEventType.Up.value,
            FrontendEventType.Down.value,
            FrontendEventType.ContextMenu.value,
            FrontendEventType.Change.value,
        ])
        # TODO we should remove event_change here.
        self.event_change = self._create_event_slot(FrontendEventType.Change)
        self.event_double_click = self._create_event_slot(FrontendEventType.DoubleClick)
        self.event_click = self._create_event_slot(FrontendEventType.Click)
        self.event_enter = self._create_event_slot(FrontendEventType.Enter)
        self.event_leave = self._create_event_slot(FrontendEventType.Leave)
        self.event_over = self._create_event_slot(FrontendEventType.Over)
        self.event_out = self._create_event_slot(FrontendEventType.Out)
        self.event_up = self._create_event_slot(FrontendEventType.Up)
        self.event_down = self._create_event_slot(FrontendEventType.Down)
        self.event_context_menu = self._create_event_slot(FrontendEventType.ContextMenu)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync)


class Object3dContainerBase(ThreeContainerBase[T_o3d_container_prop, T_child]):
    def __init__(self,
                 base_type: UIType,
                 prop_cls: Type[T_o3d_container_prop],
                 children: Dict[str, T_child],
                 inited: bool = False) -> None:
        super().__init__(base_type, prop_cls, children, inited)

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
        await self.send_and_wait(
            self.update_object3d_event(position, rotation, up, scale, visible))


class O3dContainerWithEventBase(Object3dContainerBase[T_o3d_container_prop,
                                                      T_child]):
    def __init__(self,
                 base_type: UIType,
                 prop_cls: Type[T_o3d_container_prop],
                 children: Dict[str, T_child],
                 inited: bool = False) -> None:
        super().__init__(base_type, prop_cls, children, inited)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync)


@dataclasses.dataclass(config=PyDanticConfigForNumpy)
class PointProps(ThreeBasicProps):
    limit: int = 0
    points: Union[np.ndarray, Undefined] = undefined
    colors: Union[np.ndarray, str, Undefined] = undefined
    attrs: Union[np.ndarray, Undefined] = undefined
    attrFields: Union[List[str], Undefined] = undefined
    sizeAttenuation: bool = False
    size: float = 3.0
    sizes: Union[np.ndarray, Undefined] = undefined
    encodeMethod: Union[Literal["none", "int16"], Undefined] = undefined 
    encodeScale: Union[NumberType, Undefined] = undefined

class PointsControlType(enum.Enum):
    SetColors = 0


class Points(ThreeComponentBase[PointProps]):
    def __init__(self, limit: int) -> None:
        super().__init__(UIType.ThreePoints, PointProps)
        self.props.points = np.zeros((0, 3), np.float32)
        self.props.limit = limit

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["points"] = self.props.points
        res["colors"] = self.props.colors
        res["sizes"] = self.props.sizes
        res["attrs"] = self.props.attrs
        res["attrFields"] = self.props.attrFields
        return res

    def validate_props(self, props: Dict[str, Any]):
        if "points" in props:
            return props["points"].shape[0] <= self.props.limit
        return False

    def _check_colors(self, colors, points: Optional[np.ndarray] = None):
        if isinstance(colors, np.ndarray):
            if colors.ndim == 1:
                assert colors.dtype == np.uint8, "when gray, must be uint8"
            else:
                assert colors.ndim == 2 and colors.shape[1] == 3
            if points is not None:
                assert points.shape[0] == colors.shape[0]

    async def set_colors_in_range(self, colors: Union[str, np.ndarray],
                                  begin: int, end: int):
        """
        Args: 
            cam2world: camera to world matrix, 4x4 ndaray or 16 list
            distance: camera orbit target distance.
        """
        assert begin >= 0 and end >= begin and end <= self.props.limit
        self._check_colors(colors)
        if isinstance(colors, np.ndarray):
            assert colors.shape[0] == end - begin
        return await self.send_and_wait(
            self.create_comp_event({
                "type": PointsControlType.SetColors.value,
                "offset": [begin, end],
                "colors": colors,
            }))

    async def clear(self):
        self.props.points = np.zeros((0, 3), np.float32)
        self.props.colors = undefined
        self.props.attrs = undefined
        self.props.attrFields = undefined
        self.props.sizes = undefined

        return await self.send_and_wait(
            self.update_event(points=self.props.points,
                              colors=undefined,
                              attrs=undefined,
                              attrFields=undefined,
                              sizes=undefined))

    async def update_points(self,
                            points: np.ndarray,
                            colors: Optional[Union[np.ndarray, str,
                                                   Undefined]] = None,
                            attrs: Optional[Union[np.ndarray,
                                                  Undefined]] = None,
                            attr_fields: Optional[List[str]] = None,
                            limit: Optional[int] = None,
                            sizes: Optional[Union[np.ndarray,
                                                  Undefined]] = None,
                            size_attenuation: bool = False,
                            size: Optional[Union[NumberType,
                                                 Undefined]] = None,
                            encode_method: Optional[Union[Literal["none", "int16"], Undefined]] = None, 
                            encode_scale: Optional[Union[NumberType, Undefined] ]= 50):
        # TODO better check, we must handle all errors before sent to frontend.
        assert points.ndim == 2 and points.shape[1] in [
            3, 4
        ], "only support 3 or 4 features for points"
        if limit is not None:
            assert points.shape[
                0] <= limit, f"your points size {points.shape[0]} must smaller than limit {limit}"
        else:
            assert points.shape[
                0] <= self.props.limit, f"your points size {points.shape[0]} must smaller than limit {self.props.limit}"
            
        assert points.dtype == np.float32, "only support fp32 points"
        if points.shape[1] == 4 and colors is None:
            colors = points[:, 3].astype(np.uint8)
            points = points[:, :3]
        self._check_colors(colors, points)
        if encode_method == "int16":
            upd: Dict[str, Any] = {
                "points": (points * encode_scale).astype(np.int16),
                "encodeMethod": "int16",
                "encodeScale": encode_scale,
                "sizeAttenuation": size_attenuation,
            }
        else:
            upd: Dict[str, Any] = {
                "points": points,
                "sizeAttenuation": size_attenuation,
            }
        if size is not None:
            upd["size"] = size
        if sizes is not None:
            if not isinstance(sizes, Undefined):
                assert sizes.shape[0] == points.shape[
                    0] and sizes.dtype == np.float32
            upd["sizes"] = sizes
            self.props.sizes = sizes

        if colors is not None:
            upd["colors"] = colors
            self.props.colors = colors
        if attrs is not None:
            self.props.attrs = attrs
            if not isinstance(attrs, Undefined):
                if attrs.ndim == 1:
                    attrs = attrs.reshape(-1, 1)
                if attr_fields is None:
                    attr_fields = [f"{i}" for i in range(attrs.shape[1])]
            upd["attrs"] = attrs
            upd["attrFields"] = attr_fields
            if attr_fields is not None:
                self.props.attrFields = attr_fields
        if limit is not None:
            assert limit > 0
            upd["limit"] = limit
            self.props.limit = limit
        self.props.points = points
        await self.send_and_wait(self.create_update_event(upd))

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass(config=PyDanticConfigForNumpy)
class SegmentsProps(ThreeBasicProps):
    limit: int = 0
    lines: Union[np.ndarray, Undefined] = undefined
    colors: Union[np.ndarray, Undefined] = undefined
    lineWidth: float = 1.0
    color: Union[str, Undefined] = undefined
    transparent: Union[bool, Undefined] = undefined
    opacity: Union[float, Undefined] = undefined


class Segments(ThreeComponentBase[SegmentsProps]):
    def __init__(self,
                 limit: int,
                 line_width: float = 1.0,
                 color: Union[str, Undefined] = undefined) -> None:
        super().__init__(UIType.ThreeSegments, SegmentsProps)
        self.props.lines = np.zeros((0, 2, 3), np.float32)
        self.props.lineWidth = line_width
        self.props.limit = limit
        self.props.colors = undefined
        self.props.color = color

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["lines"] = self.props.lines
        res["colors"] = self.props.colors
        return res

    def validate_props(self, props: Dict[str, Any]):
        if "lines" in props:
            return props["lines"].shape[0] <= self.props.limit
        return False

    async def clear(self):
        self.props.lines = np.zeros((0, 2, 3), np.float32)
        self.props.colors = undefined
        return self.send_and_wait(
            self.update_event(lines=undefined, colors=undefined))

    async def update_lines(self,
                           lines: np.ndarray,
                           colors: Optional[Union[np.ndarray,
                                                  Undefined]] = None,
                                                        limit: Optional[int] = None
):
        assert lines.ndim == 3 and lines.shape[1] == 2 and lines.shape[
            2] == 3, f"{lines.shape} lines must be [N, 2, 3]"
        if limit is not None:
            assert lines.shape[
                0] <= limit, f"your points size {lines.shape[0]} must smaller than limit {limit}"
        else:
            assert lines.shape[
                0] <= self.props.limit, f"your points size {lines.shape[0]} must smaller than limit {self.props.limit}"
        upd: Dict[str, Any] = {
            "lines": lines,
        }
        if colors is not None:
            if not isinstance(colors, Undefined):
                assert colors.shape[0] == lines.shape[
                    0], "color shape not valid"
            upd["colors"] = colors
            self.props.colors = colors
        if limit is not None:
            assert limit > 0
            upd["limit"] = limit
            self.props.limit = limit

        self.props.lines = lines.astype(np.float32)

        await self.send_and_wait(self.create_update_event(upd))

    async def update_mesh_lines(self, mesh: np.ndarray):
        mesh = mesh.reshape(-1, 3, 3)
        indexes = [0, 1, 1, 2, 2, 0]
        lines = np.stack([mesh[:, i] for i in indexes],
                         axis=1).reshape(-1, 2, 3)
        await self.update_lines(lines)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass(config=PyDanticConfigForNumpy)
class Boxes2DProps(Object3dBaseProps):
    centers: Union[np.ndarray, Undefined] = undefined
    dimensions: Union[np.ndarray, Undefined] = undefined
    colors: Union[np.ndarray, Undefined] = undefined
    attrs: Union[List[str], Undefined] = undefined
    color: Union[str, Undefined] = undefined
    alpha: Union[NumberType, Undefined] = undefined
    lineColor: Union[str, Undefined] = undefined
    lineWidth: Union[NumberType, Undefined] = undefined
    hoverLineColor: Union[str, Undefined] = undefined
    hoverLineWidth: Union[NumberType, Undefined] = undefined


class Boxes2D(Object3dWithEventBase[Boxes2DProps]):
    def __init__(self, limit: int) -> None:
        super().__init__(UIType.ThreeBoxes2D, Boxes2DProps)
        self.props.centers = np.zeros((0, 2), np.float32)
        self.props.dimensions = np.zeros((0, 2), np.float32)
        self.limit = limit

    def to_dict(self):
        res = super().to_dict()
        res["limit"] = self.limit
        return res

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["centers"] = self.props.centers
        res["dimensions"] = self.props.dimensions
        res["colors"] = self.props.colors
        return res

    def validate_props(self, props: Dict[str, Any]):
        if "centers" in props:
            res = props["centers"].shape[0] <= self.limit
        else:
            res = False
        return res

    async def update_boxes(self,
                           centers: Optional[np.ndarray] = None,
                           dimensions: Optional[np.ndarray] = None,
                           colors: Optional[Union[np.ndarray,
                                                  Undefined]] = None,
                           attrs: Optional[Union[List[str],
                                                 Undefined]] = None):
        # TODO check props in
        assert not isinstance(self.props.centers, Undefined)
        upd: Dict[str, Any] = {}
        if centers is not None:
            assert centers.shape[
                0] <= self.limit, f"your centers size must smaller than limit {self.limit}"
            self.props.centers = centers
            upd["centers"] = centers
        if dimensions is not None:
            if dimensions.ndim == 1:
                assert dimensions.shape[0] in [
                    1, 2
                ], "dimersion must be [1] or [2]"
            else:
                assert dimensions.shape[
                    0] <= self.limit, f"your dimensions size must smaller than limit {self.limit}"
            self.props.dimensions = dimensions
            if dimensions.shape != self.props.centers.shape:
                # check broadcastable
                np.broadcast_shapes(self.props.centers.shape, dimensions.shape)
            upd["dimensions"] = dimensions
        if colors is not None:
            if not isinstance(colors, Undefined):
                assert colors.shape[
                    0] <= self.limit, f"your colors size must smaller than limit {self.limit}"
            self.props.colors = colors
            upd["colors"] = colors
        if attrs is not None:
            self.props.attrs = attrs
            upd["attrs"] = attrs
        await self.send_and_wait(self.create_update_event(upd))

    def get_props(self):
        state = super().get_props()
        dims = self.props.dimensions
        centers = self.props.centers
        if not isinstance(dims, Undefined) and not isinstance(
                centers, Undefined):
            if dims.shape != centers.shape:
                dims = np.broadcast_to(dims, centers.shape)
        state.update({
            "colors": self.props.colors,
            "centers": self.props.centers,
            "dimensions": dims,
            "attrs": self.props.attrs,
        })
        return state

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class BoundingBoxProps(Object3dBaseProps):
    dimension: Union[Vector3Type, Undefined] = undefined
    edgeWidth: Union[float, Undefined] = undefined
    edgeColor: Union[str, Undefined] = undefined
    emissive: Union[str, Undefined] = undefined
    color: Union[str, Undefined] = undefined
    opacity: Union[float, Undefined] = undefined
    edgeOpacity: Union[float, Undefined] = undefined
    checked: bool = False
    add_cross: bool = True


class BoundingBox(Object3dWithEventBase[BoundingBoxProps]):
    def __init__(self,
                 dimension: Vector3Type,
                 edge_width: float = 1,
                 edge_color: str = "green",
                 emissive: str = "red",
                 color: str = "red",
                 opacity: float = 0.2,
                 edge_opacity: float = 0.5) -> None:
        super().__init__(UIType.ThreeBoundingBox, BoundingBoxProps)
        self.props.dimension = dimension
        self.props.edgeWidth = edge_width
        self.props.edgeColor = edge_color
        self.props.emissive = emissive
        self.props.color = color
        self.props.opacity = opacity
        self.props.edgeOpacity = edge_opacity
        

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["dimension"] = self.props.dimension
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
        await handle_standard_event(self, ev, is_sync=is_sync)

    def state_change_callback(
            self,
            data: bool,
            type: ValueType = FrontendEventType.Change.value):
        self.props.checked = data


@dataclasses.dataclass
class AxesHelperProps(Object3dBaseProps):
    length: NumberType = 10


class AxesHelper(ThreeComponentBase[AxesHelperProps]):
    def __init__(self, length: float) -> None:
        super().__init__(UIType.ThreeAxesHelper, AxesHelperProps)
        self.props.length = length

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class EdgesProps(ThreeBasicProps):
    threshold: Union[NumberType, Undefined] = undefined
    color: Union[ValueType, Undefined] = undefined
    scale: Union[NumberType, Undefined] = undefined


class Edges(ThreeComponentBase[EdgesProps]):
    def __init__(
        self,
        threshold: Union[NumberType, Undefined] = undefined,
        color: Union[ValueType, Undefined] = undefined,
        scale: Union[NumberType, Undefined] = undefined,
    ) -> None:
        super().__init__(UIType.ThreeEdges, EdgesProps)
        self.props.threshold = threshold
        self.props.color = color
        self.props.scale = scale

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class WireframeProps(ThreeBasicProps):
    fillOpacity: Union[NumberType, Undefined] = undefined
    fillMix: Union[NumberType, Undefined] = undefined
    strokeOpacity: Union[NumberType, Undefined] = undefined
    thickness: Union[NumberType, Undefined] = undefined
    colorBackfaces: Union[bool, Undefined] = undefined
    dashInvert: Union[bool, Undefined] = undefined
    dash: Union[bool, Undefined] = undefined
    dashRepeats: Union[NumberType, Undefined] = undefined
    dashLength: Union[NumberType, Undefined] = undefined
    squeeze: Union[bool, Undefined] = undefined
    squeezeMin: Union[NumberType, Undefined] = undefined
    squeezeMax: Union[NumberType, Undefined] = undefined
    stroke: Union[ValueType, Undefined] = undefined
    backfaceStroke: Union[ValueType, Undefined] = undefined
    fill: Union[ValueType, Undefined] = undefined


class Wireframe(ThreeComponentBase[WireframeProps]):
    """used in Mesh childs.
    """
    def __init__(self) -> None:
        super().__init__(UIType.ThreeWireframe, WireframeProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class InfiniteGridHelperProps(Object3dBaseProps):
    size1: Union[NumberType, Undefined] = undefined
    size2: Union[NumberType, Undefined] = undefined
    color: Union[str, Undefined] = undefined
    distance: Union[NumberType, Undefined] = undefined
    axes: Union[str, Undefined] = undefined


class InfiniteGridHelper(ThreeComponentBase[InfiniteGridHelperProps]):
    def __init__(self,
                 size1: float,
                 size2: float,
                 color: str,
                 distance: float = 8000) -> None:
        super().__init__(UIType.ThreeInfiniteGridHelper,
                         InfiniteGridHelperProps)
        self.props.size1 = size1
        self.props.size2 = size2
        self.props.color = color
        self.props.distance = distance

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class Group(Object3dContainerBase[Object3dContainerBaseProps,
                                  ThreeComponentType]):
    # TODO can/should group accept event?
    def __init__(self,
                 children: Union[Dict[str, ThreeComponentType],
                                 List[ThreeComponentType]],
                 inited: bool = False) -> None:
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.ThreeGroup, Object3dContainerBaseProps,
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
class ImageProps(Object3dBaseProps):
    image: bytes = dataclasses.field(default_factory=bytes)


class Image(Object3dWithEventBase[ImageProps]):
    def __init__(self) -> None:
        super().__init__(UIType.ThreeImage, ImageProps)

    async def show(self, image: np.ndarray):
        encoded = MUIImage.encode_image_bytes(image)
        self.props.image = encoded
        await self.send_and_wait(self.create_update_event({
            "image": encoded,
        }))

    async def clear(self):
        self.props.image = b''
        await self.send_and_wait(self.update_event(image=b''))

    async def show_raw(self, image_bytes: bytes, suffix: str):
        await self.send_and_wait(self.show_raw_event(image_bytes, suffix))

    def encode_raw_to_web(self, raw: bytes, suffix: str):
        return b'data:image/' + suffix.encode(
            "utf-8") + b';base64,' + base64.b64encode(raw)

    def show_raw_event(self, image_bytes: bytes, suffix: str):
        raw = b'data:image/' + suffix.encode(
            "utf-8") + b';base64,' + base64.b64encode(image_bytes)
        self.props.image = raw
        return self.create_update_event({
            "image": raw,
        })

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["image"] = self.props.image
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
class PerspectiveCameraProps(Object3dBaseProps):
    fov: Union[float, Undefined] = undefined
    aspect: Union[float, Undefined] = undefined
    near: Union[float, Undefined] = undefined
    far: Union[float, Undefined] = undefined


class PerspectiveCamera(Object3dBase[PerspectiveCameraProps]):
    def __init__(
        self,
        make_default: bool = True,
        fov: Union[float, Undefined] = undefined,
        aspect: Union[float, Undefined] = undefined,
        near: Union[float, Undefined] = undefined,
        far: Union[float, Undefined] = undefined,
        position: Vector3Type = (0, 0, 1),
        up: Vector3Type = (0, 0, 1)
    ) -> None:
        super().__init__(UIType.ThreePerspectiveCamera, PerspectiveCameraProps)
        self.props.fov = fov
        self.props.aspect = aspect
        self.props.near = near
        self.props.far = far
        self.props.position = position
        self.props.up = up
        self.make_default = make_default

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class OrthographicCameraProps(Object3dBaseProps):
    zoom: Union[float, Undefined] = undefined
    near: Union[float, Undefined] = undefined
    far: Union[float, Undefined] = undefined


class OrthographicCamera(Object3dBase[OrthographicCameraProps]):
    def __init__(
        self,
        make_default: bool = True,
        near: Union[float, Undefined] = undefined,
        far: Union[float, Undefined] = undefined,
        zoom: Union[float, Undefined] = undefined,
        position: Vector3Type = (0, 0, 1),
        up: Vector3Type = (0, 0, 1)
    ) -> None:
        super().__init__(UIType.ThreeOrthographicCamera,
                         OrthographicCameraProps)
        self.props.zoom = zoom
        self.props.near = near
        self.props.far = far
        self.props.position = position
        self.props.up = up
        self.make_default = make_default

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class OrbitControlProps(ThreeBasicProps):
    enableDamping: Union[bool, Undefined] = undefined
    dampingFactor: Union[NumberType, Undefined] = undefined
    minDistance: Union[NumberType, Undefined] = undefined
    maxDistance: Union[NumberType, Undefined] = undefined
    minPolarAngle: Union[NumberType, Undefined] = undefined
    maxPolarAngle: Union[NumberType, Undefined] = undefined
    minZoom: Union[NumberType, Undefined] = undefined
    maxZoom: Union[NumberType, Undefined] = undefined
    enableZoom: Union[bool, Undefined] = undefined
    zoomSpeed: Union[NumberType, Undefined] = undefined
    enableRotate: Union[bool, Undefined] = undefined
    rotateSpeed: Union[NumberType, Undefined] = undefined
    enablePan: Union[bool, Undefined] = undefined
    panSpeed: Union[NumberType, Undefined] = undefined
    keyPanSpeed: Union[NumberType, Undefined] = undefined
    makeDefault: Union[bool, Undefined] = undefined


@dataclasses.dataclass
class CameraControlProps(ThreeBasicProps):
    dampingFactor: Union[NumberType, Undefined] = undefined
    smoothTime: Union[NumberType, Undefined] = undefined
    draggingSmoothTime: Union[NumberType, Undefined] = undefined

    minDistance: Union[NumberType, Undefined] = undefined
    maxDistance: Union[NumberType, Undefined] = undefined
    minPolarAngle: Union[NumberType, Undefined] = undefined
    maxPolarAngle: Union[NumberType, Undefined] = undefined
    minZoom: Union[NumberType, Undefined] = undefined
    maxZoom: Union[NumberType, Undefined] = undefined
    polarRotateSpeed: Union[NumberType, Undefined] = undefined
    azimuthRotateSpeed: Union[NumberType, Undefined] = undefined
    truckSpeed: Union[NumberType, Undefined] = undefined
    dollySpeed: Union[NumberType, Undefined] = undefined
    verticalDragToForward: Union[bool, Undefined] = undefined
    keyboardFront: Union[bool, Undefined] = undefined
    keyboardMoveSpeed: Union[NumberType, Undefined] = undefined
    keyboardElevateSpeed: Union[NumberType, Undefined] = undefined

    infinityDolly: Union[bool, Undefined] = undefined
    makeDefault: Union[bool, Undefined] = undefined


class MapControl(ThreeComponentBase[OrbitControlProps]):
    def __init__(self) -> None:
        super().__init__(UIType.ThreeMapControl, OrbitControlProps)
        self.props.enableDamping = True
        self.props.dampingFactor = 0.25
        self.props.minDistance = 1
        self.props.maxDistance = 100

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class CameraUserControlType(enum.Enum):
    SetCamPose = 0
    SetLookAt = 1
    Reset = 2
    SetCamPoseRaw = 3


class CameraControl(ThreeComponentBase[CameraControlProps]):
    """default values: https://github.com/yomotsu/camera-controls#properties
    threejs camera default axes:
        x: right
        y: up
        z: negative forward
    """
    def __init__(self) -> None:
        super().__init__(UIType.ThreeCameraControl, CameraControlProps, [FrontendEventType.Change.value])

        # self.props.enableDamping = True
        # self.props.dampingFactor = 1
        self.props.draggingSmoothTime = 0
        self.props.smoothTime = 0
        # self.props.minDistance = 1
        # self.props.maxDistance = 100
        self.event_change = self._create_event_slot(FrontendEventType.Change)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        await handle_standard_event(self,
                                    ev,
                                    sync_state_after_change=False,
                                    is_sync=is_sync)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def set_cam2world(self,
                            cam2world: Union[List[float], np.ndarray],
                            distance: float,
                            fov_angle: float = -1,
                            update_now: bool = False):
        """
        TODO handle OrthographicCamera
        TODO currently we use a simple way to set cam2world, the
            rotation is limited by fixed camera up. so only cam2world[:, 2]
            is used in set_cam2world.
        Args: 
            cam2world: camera to world matrix, 4x4 ndaray or 16 list, R|T, not R/T
                the coordinate system is right hand, x right, y up, z negative forward
            distance: camera orbit target distance.
        """
        cam2world = np.array(cam2world, np.float32).reshape(4, 4)
        cam2world = cam2world.T  # R|T to R/T
        return await self.send_and_wait(
            self.create_comp_event({
                "type":
                CameraUserControlType.SetCamPose.value,
                "pose":
                list(map(float,
                         cam2world.reshape(-1).tolist())),
                "targetDistance":
                distance,
                "fov":
                fov_angle,
                "updateNow":
                update_now,
            }))


    async def set_lookat(self, origin: List[float], target: List[float]):
        """
        Args: 
            origin: camera position
            target: lookat target
        """
        return await self.send_and_wait(
            self.create_comp_event({
                "type": CameraUserControlType.SetLookAt.value,
                "lookat": origin + target,
            }))

    async def reset_camera(self):
        return await self.send_and_wait(
            self.create_comp_event({
                "type": CameraUserControlType.Reset.value,
            }))

    @staticmethod
    def fov_size_to_intrinsic(fov_angle: float, width: NumberType,
                              height: NumberType):
        size_wh = [int(width), int(height)]
        fov = (np.pi / 180) * fov_angle
        tanHalfFov = np.tan((fov / 2))
        f = size_wh[1] / 2 / tanHalfFov
        intrinsic = np.zeros((3, 3), np.float32)
        intrinsic[0, 0] = f
        intrinsic[1, 1] = f
        intrinsic[0, 2] = size_wh[0] / 2
        intrinsic[1, 2] = size_wh[1] / 2
        intrinsic[2, 2] = 1
        return intrinsic

    @staticmethod
    def intrinsic_size_to_fov(intrinsic: np.ndarray, width: NumberType,
                              height: NumberType):
        f = intrinsic[0][0]
        size_wh = [int(width), int(height)]
        tanHalfFov = size_wh[1] / 2 / f
        fov = np.arctan(tanHalfFov) * 2
        fov_angle = fov / (np.pi / 180)
        return fov_angle


class OrbitControl(ThreeComponentBase[OrbitControlProps]):
    def __init__(self) -> None:
        super().__init__(UIType.ThreeOrbitControl, OrbitControlProps)
        self.props.enableDamping = True
        self.props.dampingFactor = 0.25
        self.props.minDistance = 1
        self.props.maxDistance = 100

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class PointerLockControlProps(Object3dBaseProps):
    enabled: Union[bool, Undefined] = undefined
    minPolarAngle: Union[float, Undefined] = undefined
    maxPolarAngle: Union[float, Undefined] = undefined


class PointerLockControl(ThreeComponentBase[PointerLockControlProps]):
    def __init__(self,
                 enabled: Union[bool, Undefined] = undefined,
                 min_polar_angle: Union[float, Undefined] = undefined,
                 max_polar_angle: Union[float, Undefined] = undefined) -> None:
        super().__init__(UIType.ThreePointerLockControl,
                         PointerLockControlProps)
        self.props.enabled = enabled
        self.props.minPolarAngle = min_polar_angle
        self.props.maxPolarAngle = max_polar_angle

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class FirstPersonControlProps(ThreeBasicProps):
    enabled: Union[bool, Undefined] = undefined
    movementSpeed: Union[float, Undefined] = undefined
    autoForward: Union[bool, Undefined] = undefined
    lookSpeed: Union[float, Undefined] = undefined
    lookVertical: Union[bool, Undefined] = undefined
    activeLook: Union[bool, Undefined] = undefined
    heightSpeed: Union[bool, Undefined] = undefined
    heightCoef: Union[float, Undefined] = undefined
    heightMin: Union[float, Undefined] = undefined
    heightMax: Union[float, Undefined] = undefined
    constrainVertical: Union[bool, Undefined] = undefined
    verticalMin: Union[float, Undefined] = undefined
    verticalMax: Union[float, Undefined] = undefined
    mouseDragOn: Union[bool, Undefined] = undefined


class FirstPersonControl(ThreeComponentBase[FirstPersonControlProps]):
    def __init__(self) -> None:
        super().__init__(UIType.ThreeFirstPersonControl,
                         FirstPersonControlProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class FlexAutoReflow(ThreeComponentBase[ThreeBasicProps]):
    def __init__(self) -> None:
        super().__init__(UIType.ThreeFlexAutoReflow, ThreeBasicProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class FlexManualReflowProps(ThreeBasicProps):
    timestamp: str = ""


class FlexManualReflow(ThreeComponentBase[FlexManualReflowProps]):
    def __init__(self) -> None:
        super().__init__(UIType.ThreeFlexManualReflow, FlexManualReflowProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def reflow(self):
        await self.send_and_wait(self.update_event(timestamp=str(time.time())))


@dataclasses.dataclass
class ScreenShotProps(ThreeBasicProps):
    pass


class ScreenShot(ThreeComponentBase[ScreenShotProps]):
    """a special ui to get screen shot. steps:
    1. use trigger_screen_shot with userdata
    2. get image and userdata you provided from callback.
    currently impossible to get image from one function call.
    """
    def __init__(self, callback: Callable[[Tuple[str, Any]],
                                          _CORO_NONE]) -> None:
        super().__init__(UIType.ThreeScreenShot, ScreenShotProps,
                         allowed_events=[FrontendEventType.Change.value])
        self.register_event_handler(FrontendEventType.Change.value, callback)
        self.event_change = self._create_event_slot(FrontendEventType.Change)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def trigger_screen_shot(self, data: Optional[Any] = None):
        """when you provide a data, we will use image and 
        this data to call your callback
        """
        # check data is can be converted to json
        x = json.dumps(data)
        assert len(x) < 1000 * 1000
        await self.send_and_wait(
            self.create_comp_event({
                "type": 0,
                "data": data,
            }))

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self,
                                           ev,
                                           sync_status_first=True,
                                           is_sync=is_sync)


class _PendingState:
    def __init__(self,
                 ev: asyncio.Event,
                 result: Optional[Any] = None) -> None:
        self.ev = ev
        self.result = result


class ScreenShotSyncReturn(ThreeComponentBase[ScreenShotProps]):
    """a special ui to get screen shot. steps:
    1. use trigger_screen_shot with userdata
    2. get image and userdata you provided from callback.
    currently impossible to get image from one function call.
    """
    def __init__(self) -> None:
        super().__init__(UIType.ThreeScreenShot, ScreenShotProps, allowed_events=[FrontendEventType.Change.value])
        self.register_event_handler(FrontendEventType.Change.value,
                                    self._on_callback)
        self._pending_rpc: Dict[int, _PendingState] = {}
        self._uid_index = 0
        self.event_change = self._create_event_slot(FrontendEventType.Change)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def _on_callback(self, data: Tuple[str, Any]):
        img_data = data[0]
        uid = data[1]
        if uid in self._pending_rpc:
            self._pending_rpc[uid].ev.set()
            self._pending_rpc[uid].result = img_data

    async def get_screen_shot(self, timeout=2):
        uid = self._uid_index % JS_MAX_SAFE_INT
        await self.send_and_wait(
            self.create_comp_event({
                "type": 0,
                "data": uid,
            }))
        self._uid_index += 1
        ev = asyncio.Event()
        self._pending_rpc[uid] = _PendingState(ev, None)
        try:
            await asyncio.wait_for(ev.wait(), timeout=timeout)
            res = self._pending_rpc.pop(uid).result
            assert res is not None
            if isinstance(res, bytes):
                return res
            return urllib.request.urlopen(res).read()
        except:
            if uid in self._pending_rpc:
                self._pending_rpc.pop(uid)
            raise

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self,
                                           ev,
                                           sync_status_first=True,
                                           sync_state_after_change=False,
                                           is_sync=is_sync)


@dataclasses.dataclass
class ThreeCanvasProps(MUIFlexBoxProps):
    threeBackgroundColor: Union[str, Undefined] = undefined
    allowKeyboardEvent: Union[bool, Undefined] = undefined
    tabIndex: Union[int, Undefined] = undefined
    shadows: Union[bool, Undefined] = undefined
    enablePerf: Union[bool, Undefined] = undefined
    perfPosition: Union[Literal['top-right', 'top-left', 'bottom-right',
                                 'bottom-left'], Undefined] = undefined


class ThreeCanvas(MUIContainerBase[ThreeCanvasProps, ThreeComponentType]):
    def __init__(self,
                 children: Union[List[ThreeComponentType],
                                 Dict[str, ThreeComponentType]],
                 background: Union[str, Undefined] = undefined,
                 inited: bool = False) -> None:
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.ThreeCanvas, ThreeCanvasProps,
                         children, inited)
        self.props.threeBackgroundColor = background

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class TransformControlsProps(ContainerBaseProps):
    enabled: Union[bool, Undefined] = undefined
    axis: Union[str, Undefined] = undefined
    mode: Union[str, Undefined] = undefined
    translationSnap: Union[NumberType, Undefined] = undefined
    rotationSnap: Union[NumberType, Undefined] = undefined
    scaleSnap: Union[NumberType, Undefined] = undefined
    space: Union[str, Undefined] = undefined
    size: Union[NumberType, Undefined] = undefined
    showX: Union[bool, Undefined] = undefined
    showY: Union[bool, Undefined] = undefined
    showZ: Union[bool, Undefined] = undefined
    object3dUid: Union[str, Undefined] = undefined


class TransformControls(ThreeComponentBase[TransformControlsProps]):
    def __init__(self) -> None:
        super().__init__(UIType.ThreeTransformControl, TransformControlsProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class ThreeFlexProps(R3FlexPropsBase, ContainerBaseProps):
    size: Union[Vector3Type, Undefined] = undefined
    position: Union[Vector3Type, Undefined] = undefined
    direction: Union[str, Undefined] = undefined
    plane: Union[str, Undefined] = undefined
    scaleFactor: Union[int, Undefined] = undefined


class Flex(ThreeContainerBase[ThreeFlexProps, ThreeComponentType]):
    def __init__(self,
                 children: Dict[str, ThreeComponentType],
                 inited: bool = False) -> None:
        super().__init__(UIType.ThreeFlex, ThreeFlexProps,
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
class ThreeFlexItemBoxProps(R3FlexPropsBase, ContainerBaseProps):
    centerAnchor: Union[bool, Undefined] = undefined  # false


class ItemBox(ThreeContainerBase[ThreeFlexItemBoxProps, ThreeComponentType]):
    def __init__(self,
                 children: Dict[str, ThreeComponentType],
                 inited: bool = False) -> None:
        super().__init__(UIType.ThreeFlexItemBox, ThreeFlexItemBoxProps,
                         children, inited)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


def VBox(children: Dict[str, ThreeComponentType]):
    box = ItemBox(children)
    box.props.flexDirection = "column"
    return box


def HBox(children: Dict[str, ThreeComponentType]):
    box = ItemBox(children)
    box.props.flexDirection = "row"
    return box


def FlexItem(comp: ThreeComponentType):
    box = ItemBox({
        "c": comp,
    })
    return box


@dataclasses.dataclass
class HtmlProps(Object3dContainerBaseProps):
    prepend: Union[bool, Undefined] = undefined
    center: Union[bool, Undefined] = undefined
    fullscreen: Union[bool, Undefined] = undefined
    eps: Union[float, Undefined] = undefined
    distanceFactor: Union[float, Undefined] = undefined
    sprite: Union[bool, Undefined] = undefined
    transform: Union[bool, Undefined] = undefined
    zIndexRange: Union[List[Union[int, float]], Undefined] = undefined
    wrapperClass: Union[str, Undefined] = undefined
    pointerEvents: Union[PointerEventsProperties, Undefined] = undefined
    occlude: Union[bool, Undefined] = undefined
    insideFlex: Union[bool, Undefined] = undefined


class Html(Object3dContainerBase[HtmlProps, MUIComponentType]):
    """we can use MUI components only in Html.
    TODO reject invalid component
    """
    def __init__(self,
                 children: Dict[str, MUIComponentType],
                 inited: bool = False) -> None:
        super().__init__(UIType.ThreeHtml, HtmlProps, children,
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
class TextProps(Object3dBaseProps):
    value: str = ""
    characters: Union[str, Undefined] = undefined
    color: Union[str, Undefined] = undefined
    fontSize: Union[NumberType, Undefined] = undefined
    maxWidth: Union[NumberType, Undefined] = undefined
    lineHeight: Union[NumberType, Undefined] = undefined
    letterSpacing: Union[NumberType, Undefined] = undefined
    textAlign: Union[Literal["left", "right", "center", "justify"],
                      Undefined] = undefined
    font: Union[str, Undefined] = undefined
    anchorX: Union[NumberType, Literal["left", "center", "right"],
                    Undefined] = undefined
    anchorY: Union[NumberType, Literal["top", "top-baseline", "middle",
                                        "bottom-baseline", "bottom"],
                    Undefined] = undefined
    clipRect: Union[Tuple[NumberType, NumberType, NumberType, NumberType],
                     Undefined] = undefined
    depthOffset: Union[NumberType, Undefined] = undefined
    direction: Union[Literal["auto", "ltr", "rtl"], Undefined] = undefined
    overflowWrap: Union[Literal["normal", "break-word"],
                         Undefined] = undefined
    whiteSpace: Union[Literal['normal', 'nowrap'], Undefined] = undefined
    outlineWidth: Union[ValueType, Undefined] = undefined
    outlineOffsetX: Union[ValueType, Undefined] = undefined
    outlineOffsetY: Union[ValueType, Undefined] = undefined
    outlineBlur: Union[ValueType, Undefined] = undefined
    outlineColor: Union[str, Undefined] = undefined
    outlineOpacity: Union[NumberType, Undefined] = undefined
    strokeWidth: Union[ValueType, Undefined] = undefined
    strokeColor: Union[NumberType, Undefined] = undefined
    strokeOpacity: Union[NumberType, Undefined] = undefined
    fillOpacity: Union[NumberType, Undefined] = undefined


class Text(Object3dWithEventBase[TextProps]):
    def __init__(self, init: str) -> None:
        super().__init__(UIType.ThreeText, TextProps)
        self.props.value = init

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["value"] = self.props.value
        return res

    async def update_value(self, value: str):
        self.props.value = value
        upd: Dict[str, Any] = {"value": value}
        await self.send_and_wait(self.create_update_event(upd))

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class LineProps(Object3dBaseProps):
    points: List[Tuple[NumberType, NumberType,
                       NumberType]] = dataclasses.field(default_factory=list)
    color: Union[str, Undefined] = undefined
    dashed: Union[bool, Undefined] = undefined
    vertexColors: Union[Tuple[NumberType, NumberType, NumberType],
                         Undefined] = undefined
    lineWidth: Union[NumberType, Undefined] = undefined
    transparent: Union[bool, Undefined] = undefined
    opacity: Union[NumberType, Undefined] = undefined


class Line(Object3dWithEventBase[LineProps]):
    def __init__(
        self, points: Union[np.ndarray, List[Tuple[NumberType, NumberType,
                                                   NumberType]]]
    ) -> None:
        super().__init__(UIType.ThreeLine, LineProps)
        if isinstance(points, np.ndarray):
            assert points.ndim == 2 and points.shape[
                1] == 3, "must be [N, 3] arrayu"
            self.props.points = points.tolist()
        else:
            self.props.points = points

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["points"] = self.props.points
        return res

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class ArrowXYMeasure(Group):
    def __init__(self,
                 p1: Tuple[NumberType, NumberType],
                 p2: Tuple[NumberType, NumberType],
                 label: str,
                 label_size: float,
                 arrow_width: float,
                 arrow_height: float,
                 opacity: float = 1,
                 color: Optional[str] = None):
        p1n = np.array(p1)
        p2n = np.array(p2)
        if color is None:
            color = "black"
        transparent = opacity < 1
        unified_p = (p1n + p2n) / 2
        p1u = p1n - unified_p
        p2u = p2n - unified_p
        rot_yaw = np.arctan2(p1u[1], p1u[0])
        length = np.linalg.norm(p1u)
        assert length - label_size / 2 - arrow_height > 0, "your arrow too short"
        points = [
            (label_size / 2, 0, 0),
            (length, 0, 0),
            (length - arrow_height, arrow_width / 2, 0),
            (length - arrow_height, -arrow_width / 2, 0),
            (length, 0, 0),
        ]

        positions = (unified_p[0], unified_p[1], 0)
        rots = (0, 0, rot_yaw)
        rots2 = (0, 0, rot_yaw + np.pi)

        self.arrow0 = Line(points).prop(position=positions,
                                        rotation=rots,
                                        transparent=transparent,
                                        opacity=opacity,
                                        color=color)
        self.arrow1 = Line(points).prop(position=positions,
                                        rotation=rots2,
                                        transparent=transparent,
                                        opacity=opacity,
                                        color=color)
        self.text = Text(label).prop(fontSize=label_size,
                                     position=positions,
                                     color=color,
                                     strokeOpacity=opacity,
                                     fillOpacity=opacity)
        layout = {
            "a0": self.arrow0,
            "a1": self.arrow1,
            "t": self.text,
        }
        super().__init__(layout)


class GeometryType(enum.Enum):
    Box = 0
    Circle = 1
    Cone = 2
    Sphere = 3
    Plane = 4
    # Tube = 5
    Torus = 6
    TorusKnot = 7
    Tetrahedron = 8
    Ring = 9
    # Polyhedron = 10
    Icosahedron = 11
    Octahedron = 12
    Dodecahedron = 13
    Extrude = 14
    # Lathe = 15
    Capsule = 16
    Cylinder = 17


class PathOpType(enum.Enum):
    Move = 0
    Line = 1
    BezierCurve = 2
    QuadraticCurve = 3
    AbsArc = 4
    Arc = 5


@dataclasses.dataclass
class SimpleGeometryProps(ThreeGeometryPropsBase):
    shapeType: int = 0
    shapeArgs: Union[List[Union[int, float, bool]], Undefined] = undefined


@dataclasses.dataclass
class PathShapeProps(ThreeGeometryPropsBase):
    pathOps: List[Tuple[int, List[Union[float, bool]]]] = dataclasses.field(
        default_factory=list)
    curveSegments: Union[NumberType, Undefined] = undefined


class Shape:
    def __init__(self) -> None:
        self.ops: List[Tuple[int, List[Union[float, bool]]]] = []

    def move_to(self, x: float, y: float):
        self.ops.append((PathOpType.Move.value, [x, y]))

    def line_to(self, x: float, y: float):
        self.ops.append((PathOpType.Line.value, [x, y]))

    def absarc(self,
               x: float,
               y: float,
               radius: float,
               startAngle: float,
               endAngle: float,
               clockwise: bool = False):
        self.ops.append((PathOpType.AbsArc.value,
                         [x, y, radius, startAngle, endAngle, clockwise]))

    def arc(self,
            x: float,
            y: float,
            radius: float,
            startAngle: float,
            endAngle: float,
            clockwise: bool = False):
        self.ops.append((PathOpType.Arc.value,
                         [x, y, radius, startAngle, endAngle, clockwise]))

    def bezier_curve_to(self, cp1X: float, cp1Y: float, cp2X: float,
                        cp2Y: float, x: float, y: float):
        self.ops.append((PathOpType.Arc.value, [cp1X, cp1Y, cp2X, cp2Y, x, y]))

    def quadratic_curve_to(self, cpX: float, cpY: float, x: float, y: float):
        self.ops.append((PathOpType.QuadraticCurve.value, [cpX, cpY, x, y]))


class ShapeGeometry(ThreeGeometryBase[PathShapeProps]):
    def __init__(self, shape: Shape) -> None:
        super().__init__(UIType.ThreeShape, PathShapeProps)
        self.props.pathOps = shape.ops

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


def _rounded_shape(x: float, y: float, w: float, h: float, r: float):
    ctx = Shape()
    ctx.move_to(x, y + r)
    ctx.line_to(x, y + h - r)
    ctx.quadratic_curve_to(x, y + h, x + r, y + h)
    ctx.line_to(x + w - r, y + h)
    ctx.quadratic_curve_to(x + w, y + h, x + w, y + h - r)
    ctx.line_to(x + w, y + r)
    ctx.quadratic_curve_to(x + w, y, x + w - r, y)
    ctx.line_to(x + r, y)
    ctx.quadratic_curve_to(x, y, x, y + r)
    return ctx


def _rounded_shape_v2(x: float, y: float, w: float, h: float, r: float):
    ctx = Shape()
    eps = 1e-5
    r -= eps
    ctx.absarc(eps, eps, eps, -np.pi / 2, -np.pi, True)
    ctx.absarc(eps, h - r * 2, eps, np.pi, np.pi / 2, True)
    ctx.absarc(w - r * 2, h - r * 2, eps, np.pi / 2, 0, True)
    ctx.absarc(w - r * 2, eps, eps, 0, -np.pi / 2, True)
    return ctx


class RoundedRectGeometry(ShapeGeometry):
    def __init__(self, width: float, height: float, radius: float) -> None:
        shape = _rounded_shape(-width / 2, -height / 2, width, height, radius)
        super().__init__(shape)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class SimpleGeometry(ThreeGeometryBase[SimpleGeometryProps]):
    def __init__(self, type: GeometryType, args: List[Union[int, float,
                                                            bool]]) -> None:
        super().__init__(UIType.ThreeSimpleGeometry, SimpleGeometryProps)
        self.props.shapeType = type.value
        self.props.shapeArgs = args

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class BoxGeometry(SimpleGeometry):
    def __init__(self,
                 width: float = 1,
                 height: float = 1,
                 depth: float = 1,
                 width_segments: int = 1,
                 height_segments: int = 1,
                 depth_segments: int = 1) -> None:
        args: List[Union[int, float, bool]] = [
            width, height, depth, width_segments, height_segments,
            depth_segments
        ]
        super().__init__(GeometryType.Box, args)


class CapsuleGeometry(SimpleGeometry):
    def __init__(self,
                 radius: float = 1,
                 length: float = 1,
                 cap_subdivisions: int = 4,
                 radial_segments: int = 8) -> None:
        args: List[Union[int, float, bool]] = [
            radius, length, cap_subdivisions, radial_segments
        ]
        super().__init__(GeometryType.Capsule, args)


class PlaneGeometry(SimpleGeometry):
    def __init__(
        self,
        width: float = 1,
        height: float = 1,
        width_segments: int = 1,
        height_segments: int = 1,
    ) -> None:
        args: List[Union[int, float, bool]] = [
            width, height, width_segments, height_segments
        ]
        super().__init__(GeometryType.Plane, args)


class CircleGeometry(SimpleGeometry):
    def __init__(self,
                 radius: float = 1,
                 segments: int = 8,
                 theta_start: float = 0,
                 theta_length: float = np.pi * 2) -> None:
        args: List[Union[int, float, bool]] = [
            radius, segments, theta_start, theta_length
        ]
        super().__init__(GeometryType.Circle, args)


class ConeGeometry(SimpleGeometry):
    def __init__(self,
                 radius: float = 1,
                 height: float = 1,
                 radial_segments: int = 32,
                 height_segments: int = 1,
                 open_ended: bool = False,
                 theta_start: float = 0,
                 theta_length: float = np.pi * 2) -> None:
        args: List[Union[int, float, bool]] = [
            radius, height, radial_segments, height_segments, open_ended,
            theta_start, theta_length
        ]
        super().__init__(GeometryType.Cone, args)


class CylinderGeometry(SimpleGeometry):
    def __init__(self,
                 radius_top: float = 1,
                 radius_bottom: float = 1,
                 height: float = 1,
                 radial_segments: int = 32,
                 height_segments: int = 1,
                 open_ended: bool = False,
                 theta_start: float = 0,
                 theta_length: float = np.pi * 2) -> None:
        args: List[Union[int, float, bool]] = [
            radius_top, radius_bottom, height, radial_segments,
            height_segments, open_ended, theta_start, theta_length
        ]
        super().__init__(GeometryType.Cylinder, args)


class DodecahedronGeometry(SimpleGeometry):
    def __init__(self, radius: float = 1, detail: int = 0) -> None:
        args: List[Union[int, float, bool]] = [radius, detail]
        super().__init__(GeometryType.Dodecahedron, args)


class IcosahedronGeometry(SimpleGeometry):
    def __init__(self, radius: float = 1, detail: int = 0) -> None:
        args: List[Union[int, float, bool]] = [radius, detail]
        super().__init__(GeometryType.Icosahedron, args)


class OctahedronGeometry(SimpleGeometry):
    def __init__(self, radius: float = 1, detail: int = 0) -> None:
        args: List[Union[int, float, bool]] = [radius, detail]
        super().__init__(GeometryType.Octahedron, args)


class TetrahedronGeometry(SimpleGeometry):
    def __init__(self, radius: float = 1, detail: int = 0) -> None:
        args: List[Union[int, float, bool]] = [radius, detail]
        super().__init__(GeometryType.Tetrahedron, args)


class RingGeometry(SimpleGeometry):
    def __init__(self,
                 inner_radius: float = 0.5,
                 outer_radius: float = 1,
                 theta_segments: int = 32,
                 phi_segments: int = 1,
                 theta_start: float = 0,
                 theta_length: float = np.pi * 2) -> None:
        args: List[Union[int, float, bool]] = [
            inner_radius, outer_radius, theta_segments, phi_segments,
            theta_start, theta_length
        ]
        super().__init__(GeometryType.Ring, args)


class SphereGeometry(SimpleGeometry):
    def __init__(self,
                 radius: float = 1,
                 widthSegments: int = 32,
                 heightSegments: int = 16,
                 phi_start: float = 0,
                 phi_length: float = np.pi * 2,
                 theta_start: float = 0,
                 theta_length: float = np.pi) -> None:
        args: List[Union[int, float, bool]] = [
            radius, widthSegments, heightSegments, phi_start, phi_length,
            theta_start, theta_length
        ]
        super().__init__(GeometryType.Sphere, args)


class TorusGeometry(SimpleGeometry):
    def __init__(self,
                 radius: float = 1,
                 tube: float = 0.4,
                 radial_segments: int = 12,
                 tubular_segments: int = 48,
                 arc: float = np.pi * 2) -> None:
        args: List[Union[int, float, bool]] = [
            radius, tube, radial_segments, tubular_segments, arc
        ]
        super().__init__(GeometryType.Torus, args)


class TorusKnotGeometry(SimpleGeometry):
    def __init__(self,
                 radius: float = 1,
                 tube: float = 0.4,
                 tubular_segments: int = 64,
                 radial_segments: int = 8,
                 p: int = 2,
                 q: int = 3) -> None:
        args: List[Union[int, float, bool]] = [
            radius, tube, tubular_segments, radial_segments, p, q
        ]
        super().__init__(GeometryType.TorusKnot, args)


@dataclasses.dataclass
class MeshBasicMaterialProps(ThreeMaterialPropsBase):
    color: Union[str, Undefined] = undefined
    wireframe: Union[bool, Undefined] = undefined
    vertexColors: Union[bool, Undefined] = undefined
    fog: Union[bool, Undefined] = undefined


@dataclasses.dataclass
class MeshStandardMaterialProps(MeshBasicMaterialProps):
    emissive: Union[str, Undefined] = undefined
    roughness: Union[NumberType, Undefined] = undefined
    metalness: Union[NumberType, Undefined] = undefined
    flagShading: Union[bool, Undefined] = undefined


@dataclasses.dataclass
class MeshLambertMaterialProps(MeshBasicMaterialProps):
    emissive: Union[str, Undefined] = undefined


@dataclasses.dataclass
class MeshMatcapMaterialProps(ThreeMaterialPropsBase):
    flagShading: Union[bool, Undefined] = undefined
    color: Union[str, Undefined] = undefined


@dataclasses.dataclass
class MeshNormalMaterialProps(ThreeMaterialPropsBase):
    flagShading: Union[bool, Undefined] = undefined
    wireframe: Union[bool, Undefined] = undefined


@dataclasses.dataclass
class MeshDepthMaterialProps(ThreeMaterialPropsBase):
    wireframe: Union[bool, Undefined] = undefined


@dataclasses.dataclass
class MeshPhongMaterialProps(MeshBasicMaterialProps):
    reflectivity: Union[NumberType, Undefined] = undefined
    refractionRatio: Union[NumberType, Undefined] = undefined
    emissive: Union[str, Undefined] = undefined
    specular: Union[str, Undefined] = undefined
    shininess: Union[NumberType, Undefined] = undefined


@dataclasses.dataclass
class MeshPhysicalMaterialProps(MeshStandardMaterialProps):
    reflectivity: Union[NumberType, Undefined] = undefined
    clearcoat: Union[NumberType, Undefined] = undefined
    clearcoatRoughness: Union[NumberType, Undefined] = undefined
    metalness: Union[NumberType, Undefined] = undefined
    roughness: Union[NumberType, Undefined] = undefined
    sheen: Union[NumberType, Undefined] = undefined
    transmission: Union[NumberType, Undefined] = undefined
    ior: Union[NumberType, Undefined] = undefined
    attenuationColor: Union[str, NumberType, Undefined] = undefined
    attenuationDistance: Union[NumberType, Undefined] = undefined
    specularIntensity: Union[NumberType, Undefined] = undefined
    specularColor: Union[str, NumberType, Undefined] = undefined
    sheenRoughness: Union[NumberType, Undefined] = undefined
    sheenColor: Union[str, NumberType, Undefined] = undefined


@dataclasses.dataclass
class MeshToonMaterialProps(ThreeMaterialPropsBase):
    color: Union[str, Undefined] = undefined


@dataclasses.dataclass
class MeshTransmissionMaterialProps(MeshPhysicalMaterialProps):
    transmission: Union[NumberType, Undefined] = undefined
    thickness: Union[NumberType, Undefined] = undefined
    backsideThickness: Union[NumberType, Undefined] = undefined
    roughness: Union[NumberType, Undefined] = undefined
    chromaticAberration: Union[NumberType, Undefined] = undefined
    anisotropy: Union[NumberType, Undefined] = undefined
    distortion: Union[NumberType, Undefined] = undefined
    distortion_scale: Union[NumberType, Undefined] = undefined
    temporalDistortion: Union[NumberType, Undefined] = undefined
    transmission_sampler: Union[bool, Undefined] = undefined
    backside: Union[bool, Undefined] = undefined
    resolution: Union[NumberType, Undefined] = undefined
    backsideResolution: Union[NumberType, Undefined] = undefined
    samples: Union[NumberType, Undefined] = undefined


@dataclasses.dataclass
class MeshDiscardMaterialProps(ThreeBasicProps):
    pass


class MeshBasicMaterialV1(ThreeMaterialBase[MeshBasicMaterialProps]):
    def __init__(self) -> None:
        super().__init__(UIType.ThreeMeshMaterial, MeshBasicMaterialProps)
        self.props.materialType = MeshMaterialType.Basic.value

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshStandardMaterialV1(ThreeMaterialBase[MeshStandardMaterialProps]):
    def __init__(self) -> None:
        super().__init__(UIType.ThreeMeshMaterial, MeshStandardMaterialProps)
        self.props.materialType = MeshMaterialType.Standard.value

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshBasicMaterial(ThreeComponentBase[MeshBasicMaterialProps]):
    def __init__(self) -> None:
        super().__init__(UIType.ThreeMeshBasicMaterial, MeshBasicMaterialProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshStandardMaterial(ThreeMaterialBase[MeshStandardMaterialProps]):
    def __init__(self) -> None:
        super().__init__(UIType.ThreeMeshStandardMaterial,
                         MeshStandardMaterialProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshLambertMaterial(ThreeMaterialBase[MeshLambertMaterialProps]):
    def __init__(self, ) -> None:
        super().__init__(UIType.ThreeMeshLambertMaterial,
                         MeshLambertMaterialProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshMatcapMaterial(ThreeMaterialBase[MeshMatcapMaterialProps]):
    def __init__(self, ) -> None:
        super().__init__(UIType.ThreeMeshMatcapMaterial,
                         MeshMatcapMaterialProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshNormalMaterial(ThreeMaterialBase[MeshNormalMaterialProps]):
    def __init__(self, ) -> None:
        super().__init__(UIType.ThreeMeshNormalMaterial,
                         MeshNormalMaterialProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshDepthMaterial(ThreeMaterialBase[MeshDepthMaterialProps]):
    def __init__(self, ) -> None:
        super().__init__(UIType.ThreeMeshDepthMaterial, MeshDepthMaterialProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshPhongMaterial(ThreeMaterialBase[MeshPhongMaterialProps]):
    def __init__(self, ) -> None:
        super().__init__(UIType.ThreeMeshPhongMaterial, MeshPhongMaterialProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshPhysicalMaterial(ThreeMaterialBase[MeshPhysicalMaterialProps]):
    def __init__(self, ) -> None:
        super().__init__(UIType.ThreeMeshPhysicalMaterial,
                         MeshPhysicalMaterialProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshToonMaterial(ThreeMaterialBase[MeshToonMaterialProps]):
    def __init__(self, ) -> None:
        super().__init__(UIType.ThreeMeshToonMaterial, MeshToonMaterialProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshTransmissionMaterial(ThreeMaterialBase[MeshTransmissionMaterialProps]
                               ):
    def __init__(self, ) -> None:
        super().__init__(UIType.ThreeMeshTransmissionMaterial,
                         MeshTransmissionMaterialProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshDiscardMaterial(ThreeMaterialBase[MeshDiscardMaterialProps]):
    def __init__(self, ) -> None:
        super().__init__(UIType.ThreeMeshDiscardMaterial,
                         MeshDiscardMaterialProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


MeshChildType: TypeAlias = Union[ThreeMaterialBase, ThreeMaterialPropsBase,
                                 ThreeGeometryPropsBase, ThreeGeometryBase]


@dataclasses.dataclass
class PrimitiveMeshProps(Object3dContainerBaseProps):
    pass


@dataclasses.dataclass
class MeshProps(PrimitiveMeshProps):
    hoverColor: Union[str, Undefined] = undefined
    clickColor: Union[str, Undefined] = undefined
    toggleMode: Union[bool, Undefined] = undefined
    toggled: Union[bool, Undefined] = undefined


class Mesh(O3dContainerWithEventBase[PrimitiveMeshProps, ThreeComponentType]):
    def __init__(self, children: ThreeLayoutType) -> None:
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}

        super().__init__(UIType.ThreePrimitiveMesh, PrimitiveMeshProps,
                         children)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshV1(O3dContainerWithEventBase[MeshProps, ThreeComponentType]):
    def __init__(self,
                 geometry: ThreeGeometryBase,
                 material: ThreeMaterialBase,
                 inited: bool = False) -> None:
        self.geometry = geometry
        assert isinstance(geometry, ThreeGeometryBase)
        assert isinstance(material, ThreeMaterialBase)

        self.material = material
        children: Dict[str, ThreeComponentType] = {
            "geometry": geometry,
            "material": material,
        }
        super().__init__(UIType.ThreeMesh, MeshProps, children,
                         inited)
        self.props.toggled = False

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["toggled"] = self.props.toggled
        return res

    def state_change_callback(
            self,
            data: bool,
            type: ValueType = FrontendEventType.Change.value):
        self.props.toggled = data

    async def set_checked(self, checked: bool):
        ev = self.create_update_event({
            "toggled": checked,
        })
        await self.send_and_wait(ev)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class HudProps(ThreeFlexProps):
    renderPriority: Union[int, Undefined] = undefined


class Hud(ThreeContainerBase[HudProps, ThreeComponentType]):
    # TODO can/should group accept event?
    def __init__(self,
                 children: Dict[str, ThreeComponentType],
                 inited: bool = False) -> None:
        super().__init__(UIType.ThreeHud, HudProps, children,
                         inited)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class ShapeButton(Group):
    def __init__(self, name: str, shape: Shape, font_size: NumberType,
                 text_max_width: NumberType,
                 callback: Callable[[Any], _CORO_ANY]) -> None:
        material = MeshBasicMaterialV1()
        material.prop(color="#393939")
        mesh = MeshV1(ShapeGeometry(shape), material)
        mesh.register_event_handler(FrontendEventType.Click.value, callback, stop_propagation=True)
        mesh.prop(hoverColor="#222222", clickColor="#009A63")
        self.mesh = mesh
        text = Text(name)
        text.prop(fontSize=font_size,
                  color="white",
                  position=(0, 0, 0),
                  maxWidth=text_max_width)
        children = {
            "mesh": mesh,
            "text": text,
        }
        super().__init__(children)
        self.name = name
        # self.callback = callback
        self.event_click = self._create_event_slot(FrontendEventType.Click)

    def to_dict(self):
        res = super().to_dict()
        res["name"] = self.name
        return res

    async def headless_click(self):
        uiev = UIEvent(
            {self._flow_uid: (FrontendEventType.Click.value, self.name)})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return handle_standard_event(self, ev, is_sync)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class Button(Group):
    def __init__(self,
                 name: str,
                 width: float,
                 height: float,
                 callback: Callable[[Any], _CORO_ANY],
                 radius: Optional[float] = None,
                 font_size: Optional[float] = None) -> None:
        if radius is None:
            radius = min(width, height) * 0.25
        if font_size is None:
            font_size = min(width, height) * 0.5
        material = MeshBasicMaterialV1()
        material.prop(color="#393939")
        mesh = MeshV1(RoundedRectGeometry(width, height, radius), material)
        mesh.register_event_handler(FrontendEventType.Click.value, callback, stop_propagation=True)
        self.event_click = self._create_event_slot(FrontendEventType.Click)

        mesh.prop(hoverColor="#222222", clickColor="#009A63")
        self.mesh = mesh
        text = Text(name)
        text.prop(fontSize=font_size,
                  color="white",
                  position=(0, 0, 0),
                  maxWidth=width)
        children = {
            "mesh": mesh,
            "text": text,
        }
        super().__init__(children)
        self.name = name
        # self.callback = callback

    def to_dict(self):
        res = super().to_dict()
        res["name"] = self.name
        return res

    async def headless_click(self):
        uiev = UIEvent(
            {self._flow_uid: (FrontendEventType.Click.value, self.name)})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class ToggleButton(Group):
    def __init__(self,
                 name: str,
                 width: float,
                 height: float,
                 callback: Callable[[bool], _CORO_ANY],
                 radius: Optional[float] = None,
                 font_size: Optional[float] = None) -> None:
        if radius is None:
            radius = min(width, height) * 0.25
        if font_size is None:
            font_size = min(width, height) * 0.5
        material = MeshBasicMaterialV1()
        material.prop(color="#393939")
        mesh = MeshV1(RoundedRectGeometry(width, height, radius), material)
        mesh.register_event_handler(FrontendEventType.Click.value, callback, stop_propagation=True)
        self.event_click = self._create_event_slot(FrontendEventType.Click)

        mesh.prop(hoverColor="#222222",
                  clickColor="#009A63",
                  toggleMode=True)
        self.mesh = mesh

        text = Text(name)
        text.prop(fontSize=font_size,
                  color="white",
                  position=(0, 0, 0),
                  maxWidth=width)
        children = {
            "mesh": mesh,
            "text": text,
        }
        super().__init__(children)
        self.name = name

    @property
    def toggled(self):
        return self.mesh.props.toggled

    def to_dict(self):
        res = super().to_dict()
        res["name"] = self.name
        return res

    async def headless_toggle(self):
        uiev = UIEvent(
            {self._flow_uid: (FrontendEventType.Change.value, self.name)})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class PivotControlsProps(ContainerBaseProps):
    offset: Union[Vector3Type, Undefined] = undefined
    rotation: Union[Vector3Type, Undefined] = undefined

    scale: Union[NumberType, Undefined] = undefined
    lineWidth: Union[NumberType, Undefined] = undefined
    fixed: Union[bool, Undefined] = undefined
    matrix: Union[List[float], Undefined] = undefined
    anchor: Union[Vector3Type, Undefined] = undefined
    autoTransform: Union[bool, Undefined] = undefined
    activeAxes: Union[Tuple[bool, bool, bool], Undefined] = undefined
    axisColors: Union[Tuple[ValueType, ValueType, ValueType],
                       Undefined] = undefined
    hoveredColor: Union[ValueType, Undefined] = undefined
    depthTest: Union[bool, Undefined] = undefined
    opacity: Union[float, Undefined] = undefined
    visible: Union[bool, Undefined] = undefined
    annotations: Union[bool, Undefined] = undefined


class PivotControls(ThreeContainerBase[PivotControlsProps,
                                       ThreeComponentType]):
    def __init__(self,
                 children: Optional[Union[Dict[str, ThreeComponentType],
                                          List[ThreeComponentType]]] = None,
                 callback: Optional[Callable[[bool], _CORO_ANY]] = None,
                 debounce: float = 100) -> None:
        if children is None:
            children = []
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}

        super().__init__(UIType.ThreePivotControl,
                         PivotControlsProps,
                         allowed_events=[FrontendEventType.Change.value],
                         _children=children)
        if callback is not None:
            self.register_event_handler(FrontendEventType.Change.value,
                                        callback,
                                        debounce=debounce)

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
                                           sync_status_first=True,
                                           sync_state_after_change=False,
                                           is_sync=is_sync)


@dataclasses.dataclass
class PointLightProps(Object3dBaseProps):
    color: Union[NumberType, str, Undefined] = undefined
    intensity: Union[NumberType, Undefined] = undefined
    distance: Union[NumberType, Undefined] = undefined
    decay: Union[NumberType, Undefined] = undefined
    castShadow: Union[bool, Undefined] = undefined
    power: Union[NumberType, Undefined] = undefined


class PointLight(Object3dBase[PointLightProps]):
    def __init__(self,
                 position: Union[Vector3Type, Undefined] = undefined,
                 color: Union[NumberType, str, Undefined] = undefined,
                 intensity: Union[NumberType, Undefined] = undefined) -> None:
        super().__init__(UIType.ThreePointLight, PointLightProps)
        self.props.color = color
        self.props.intensity = intensity
        self.props.position = position

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class AmbientLightProps(Object3dBaseProps):
    color: Union[NumberType, str, Undefined] = undefined
    intensity: Union[NumberType, Undefined] = undefined


class AmbientLight(Object3dBase[AmbientLightProps]):
    def __init__(self,
                 color: Union[NumberType, str, Undefined] = undefined,
                 intensity: Union[NumberType, Undefined] = undefined) -> None:
        super().__init__(UIType.ThreeAmbientLight, AmbientLightProps)
        self.props.color = color
        self.props.intensity = intensity

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class HemisphereLightProps(Object3dBaseProps):
    color: Union[NumberType, str, Undefined] = undefined
    intensity: Union[NumberType, Undefined] = undefined
    groundColor: Union[NumberType, str, Undefined] = undefined


class HemisphereLight(Object3dBase[HemisphereLightProps]):
    def __init__(
            self,
            color: Union[NumberType, str, Undefined] = undefined,
            intensity: Union[NumberType, Undefined] = undefined,
            ground_color: Union[NumberType, str,
                                Undefined] = undefined) -> None:
        super().__init__(UIType.ThreeHemisphereLight, HemisphereLightProps)
        self.props.color = color
        self.props.intensity = intensity
        self.props.groundColor = ground_color

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class DirectionalLightProps(Object3dBaseProps):
    color: Union[NumberType, str, Undefined] = undefined
    intensity: Union[NumberType, Undefined] = undefined
    castShadow: Union[bool, Undefined] = undefined
    targetPosition: Union[Vector3Type, Undefined] = undefined
    helperColor: Union[NumberType, Undefined] = undefined
    helperSize: Union[NumberType, Undefined] = undefined


class DirectionalLight(Object3dBase[DirectionalLightProps]):
    def __init__(
            self,
            position: Union[Vector3Type, Undefined] = undefined,
            color: Union[NumberType, str, Undefined] = undefined,
            intensity: Union[NumberType, Undefined] = undefined,
            target_position: Union[Vector3Type,
                                   Undefined] = undefined) -> None:
        super().__init__(UIType.ThreeDirectionalLight, DirectionalLightProps)
        self.props.color = color
        self.props.intensity = intensity
        self.props.targetPosition = target_position
        self.props.position = position

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class SpotLightProps(Object3dBaseProps):
    color: Union[NumberType, str, Undefined] = undefined
    intensity: Union[NumberType, Undefined] = undefined
    distance: Union[NumberType, Undefined] = undefined
    decay: Union[NumberType, Undefined] = undefined
    castShadow: Union[bool, Undefined] = undefined
    angle: Union[NumberType, Undefined] = undefined
    penumbra: Union[NumberType, Undefined] = undefined
    power: Union[NumberType, Undefined] = undefined
    targetPosition: Union[Vector3Type, Undefined] = undefined
    helperColor: Union[NumberType, Undefined] = undefined


class SpotLight(Object3dBase[SpotLightProps]):
    def __init__(
            self,
            position: Union[Vector3Type, Undefined] = undefined,
            color: Union[NumberType, str, Undefined] = undefined,
            intensity: Union[NumberType, Undefined] = undefined,
            target_position: Union[Vector3Type,
                                   Undefined] = undefined) -> None:
        super().__init__(UIType.ThreeDirectionalLight, SpotLightProps)
        self.props.color = color
        self.props.intensity = intensity
        self.props.targetPosition = target_position
        self.props.position = position

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class BufferMeshControlType(enum.Enum):
    UpdateBuffers = 0
    CalculateVertexNormals = 1


@dataclasses.dataclass(config=PyDanticConfigForNumpy)
class BufferMeshProps(Object3dContainerBaseProps):
    initialBuffers: Union[Dict[str, np.ndarray], Undefined] = undefined
    initialIndex: Union[np.ndarray, Undefined] = undefined
    limit: Union[int, Undefined] = undefined
    initialCalcVertexNormals: Union[bool, Undefined] = undefined


@dataclasses.dataclass(config=PyDanticConfigForNumpy)
class BufferMeshUpdate(DataClassWithUndefined):
    data: np.ndarray
    offset: Union[int, Undefined] = undefined
    newCount: Union[int, Undefined] = undefined


class BufferMesh(O3dContainerWithEventBase[BufferMeshProps,
                                           ThreeComponentType]):
    def __init__(
            self,
            initial_buffers: Dict[str, np.ndarray],
            limit: int,
            children: ThreeLayoutType,
            initial_index: Union[np.ndarray, Undefined] = undefined) -> None:
        """initialIndex and initialBuffers must be specified in init,
        they can't be setted in update_event.
        WARNING: this element should only be used for advanced usage.
        if you use this with wrong inputs, the frontend may crash. 
        Args:
            initial_index: if undefined, user can't setted in update_buffers.
            initial_buffers: dict of threejs buffer attributes.
                if unsupported data format (for float, only f32 supported),
                will be casted to f32 implicitly.
        """
        first_dim = -1
        for k, v in initial_buffers.items():
            assert v.shape[0] <= limit, "initial buffer size exceeds limit"
            if first_dim == -1:
                first_dim = v.shape[0]
            else:
                assert first_dim == v.shape[0], "buffer size mismatch"
            if v.dtype == np.float16 or v.dtype == np.float64:
                initial_buffers[k] = v.astype(np.float32)
        # TODO children must be material or Edges
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}

        super().__init__(UIType.ThreeBufferMesh, BufferMeshProps, children)
        self.props.initialBuffers = initial_buffers
        self.props.limit = limit
        self.props.initialIndex = initial_index
        self.initial_buffers = initial_buffers
        self.initial_index = initial_index

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def calc_vertex_normals_in_frontend(self):
        res = {
            "type": BufferMeshControlType.CalculateVertexNormals.value,
        }
        return await self.send_and_wait(self.create_comp_event(res))

    async def update_buffers(self,
                             updates: Dict[str, BufferMeshUpdate],
                             update_bound: bool = False,
                             new_index: Optional[np.ndarray] = None):
        """
        Args: 
            updates: contains the updates for each buffer, the key must be in initialBuffers.
            update_bound: if true, the bound will be updated. user should update this when they 
                change the position.
            new_index: if not None, the index will be updated.
        """
        if isinstance(self.initial_index, Undefined):
            assert new_index is None, "new_index must be None"
        updates_dict = {}
        for k, v in updates.items():
            assert k in self.initial_buffers, "key not found"
            if v.data.dtype == np.float16 or v.data == np.float64:
                v.data = v.data.astype(np.float32)
            updates_dict[k] = v.get_dict()
        res = {
            "type": BufferMeshControlType.UpdateBuffers.value,
            "updates": updates_dict,
            "updateBound": update_bound,
        }
        if new_index is not None:
            res["newIndex"] = new_index
        return await self.send_and_wait(self.create_comp_event(res))


@dataclasses.dataclass(config=PyDanticConfigForNumpy)
class VoxelMeshProps(Object3dContainerBaseProps):
    size: Union[NumberType, Undefined] = undefined
    centers: Union[np.ndarray, Undefined] = undefined
    colors: Union[np.ndarray, Undefined] = undefined
    limit: Union[int, Undefined] = undefined


class VoxelMesh(O3dContainerWithEventBase[VoxelMeshProps, ThreeComponentType]):
    def __init__(self,
                 centers: np.ndarray,
                 size: float,
                 limit: int,
                 children: ThreeLayoutType,
                 colors: Union[np.ndarray, Undefined] = undefined) -> None:
        if not isinstance(colors, Undefined):
            assert centers.shape[0] == colors.shape[
                0], "centers and colors must have same length"
        assert centers.shape[0] <= limit
        if centers.dtype != np.float32:
            centers = centers.astype(np.float32)
        # TODO children must be material or Edges
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}

        super().__init__(UIType.ThreeVoxelMesh, VoxelMeshProps, children)
        self.props.limit = limit
        self.props.colors = colors
        self.props.size = size
        self.props.centers = centers

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass(config=PyDanticConfigForNumpy)
class InstancedMeshProps(Object3dContainerBaseProps):
    transforms: Union[np.ndarray, Undefined] = undefined
    scales: Union[np.ndarray, Undefined] = undefined
    colors: Union[np.ndarray, Undefined] = undefined
    limit: Union[int, Undefined] = undefined


class InstancedMesh(O3dContainerWithEventBase[InstancedMeshProps,
                                              ThreeComponentType]):
    def __init__(self,
                 transforms: np.ndarray,
                 limit: int,
                 children: ThreeLayoutType,
                 colors: Union[np.ndarray, Undefined] = undefined) -> None:
        if not isinstance(colors, Undefined):
            assert transforms.shape[0] == colors.shape[
                0], "centers and colors must have same length"
        assert transforms.shape[0] <= limit
        if transforms.dtype != np.float32:
            transforms = transforms.astype(np.float32)
        # TODO children must be material or Edges
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}

        super().__init__(UIType.ThreeInstancedMesh, InstancedMeshProps,
                         children)
        self.props.limit = limit
        self.props.colors = colors
        self.props.transforms = transforms

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class SkyProps(ThreeBasicProps):
    distance: Union[NumberType, Undefined] = undefined
    sunPosition: Union[Vector3Type, Undefined] = undefined
    inclination: Union[NumberType, Undefined] = undefined
    azimuth: Union[NumberType, Undefined] = undefined
    mieCoefficient: Union[NumberType, Undefined] = undefined
    mieDirectionalG: Union[NumberType, Undefined] = undefined
    rayleigh: Union[NumberType, Undefined] = undefined
    turbidity: Union[NumberType, Undefined] = undefined


class Sky(ThreeComponentBase[SkyProps]):
    def __init__(self) -> None:
        super().__init__(UIType.ThreeSky, SkyProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class EnvGround:
    radius: Union[NumberType, Undefined] = undefined
    height: Union[NumberType, Undefined] = undefined
    scale: Union[Vector3Type, Undefined] = undefined


@dataclasses.dataclass
class EnvironmentProps(ContainerBaseProps):
    files: Union[List[str], str, Undefined] = undefined
    resolution: Union[int, Undefined] = undefined
    background: Union[bool, Literal["only"], Undefined] = undefined
    blur: Union[int, Undefined] = undefined
    preset: Union[Literal["sunset", "dawn", "night", "warehouse", "forest",
                          "apartment", "studio", "city", "park", "lobby"],
                  Undefined] = undefined
    ground: Union[EnvGround, bool, Undefined] = undefined
    path: Union[str, Undefined] = undefined


class Environment(ThreeContainerBase[EnvironmentProps, ThreeComponentType]):
    def __init__(self, children: Optional[ThreeLayoutType] = None) -> None:
        if children is None:
            children = {}
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}

        super().__init__(UIType.ThreeEnvironment, EnvironmentProps,
                         {**children})

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

class URILoaderType(enum.IntEnum):
    GLTF = 0
    FBX = 1
    RGBE = 2
    TEXTURE = 3

@dataclasses.dataclass
class LoaderContextProps(ContainerBaseProps):
    uri: str = ""
    loaderType: URILoaderType = URILoaderType.GLTF
    dataKey: Union[str, Undefined] = undefined # default: URILoader


class URILoaderContext(ThreeContainerBase[LoaderContextProps, ThreeComponentType]):
    """create a context with template data.
    default dataKey: "" (empty), this means the data itself is passed to children
    """
    def __init__(self, type: URILoaderType, uri: str, children: Optional[ThreeLayoutType] = None) -> None:
        if children is None:
            children = {}
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}

        super().__init__(UIType.ThreeURILoaderContext, LoaderContextProps,
                         {**children})
        self.props.uri = uri
        self.props.loaderType = type

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


    
@dataclasses.dataclass
class CubeCameraProps(Object3dContainerBaseProps):
    frames: Union[int, Undefined] = undefined
    resolution: Union[NumberType, Undefined] = undefined
    near: Union[NumberType, Undefined] = undefined
    far: Union[NumberType, Undefined] = undefined
    dataKey: Union[str, Undefined] = undefined


class CubeCamera(Object3dContainerBase[CubeCameraProps, ThreeComponentType]):
    """create a context with template data. 
    default dataKey: CubeCameraTexture
    """
    def __init__(self, children: ThreeLayoutType) -> None:
        if children is None:
            children = {}
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        assert children, "CubeCamera must have children"
        super().__init__(UIType.ThreeCubeCamera, CubeCameraProps,
                         {**children})
    
    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

@dataclasses.dataclass
class ContactShadowsProps(Object3dBaseProps):
    opacity: Union[NumberType, Undefined] = undefined
    width: Union[NumberType, Undefined] = undefined
    height: Union[NumberType, Undefined] = undefined
    blur: Union[NumberType, Undefined] = undefined
    near: Union[NumberType, Undefined] = undefined
    far: Union[NumberType, Undefined] = undefined
    smooth: Union[bool, Undefined] = undefined
    resolution: Union[NumberType, Undefined] = undefined
    frames: Union[int, Undefined] = undefined
    scale: Union[NumberType, Tuple[NumberType, NumberType], Undefined] = undefined
    color: Union[str, NumberType, Undefined] = undefined
    depthWrite: Union[bool, Undefined] = undefined


class ContactShadows(ThreeComponentBase[ContactShadowsProps]):
    def __init__(self) -> None:
        super().__init__(UIType.ThreeContactShadows, ContactShadowsProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)
