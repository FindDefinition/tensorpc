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
                    T_container_props, Fragment, EventHandler)
from .mui import (FlexBoxProps, MUIComponentType, MUIContainerBase,
                  _encode_image_bytes, PointerEventsProperties)


Vector3Type: TypeAlias = Tuple[float, float, float]

_CORO_NONE: TypeAlias = Union[Coroutine[None, None, None], None]
_CORO_ANY: TypeAlias = Union[Coroutine[Any, None, None], Any]

CORO_NONE: TypeAlias = Union[Coroutine[None, None, None], None]
P = ParamSpec('P')


@dataclasses.dataclass
class ThreeBasicProps(BasicProps):
    pass


@dataclasses.dataclass
class R3FlexPropsBase(BasicProps):
    align_content: Union[str, Undefined] = undefined
    align_items: Union[str, Undefined] = undefined
    justify_content: Union[str, Undefined] = undefined
    flex_direction: Union[str, Undefined] = undefined
    flex_wrap: Union[str, Undefined] = undefined

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
    material_type: int = 0
    transparent: Union[bool, Undefined] = undefined
    opacity: Union[NumberType, Undefined] = undefined
    depth_test: Union[bool, Undefined] = undefined
    depth_write: Union[bool, Undefined] = undefined
    alpha_test: Union[NumberType, Undefined] = undefined
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


class PointerEventType(enum.Enum):
    # we don't support hover/move/missed
    # here because it make too much events.
    # TODO maybe we can use debounced event for hover?
    Click = 0
    DoubleClick = 1
    Enter = 2
    Leave = 3
    Over = 4
    Out = 5
    Up = 6
    Down = 7
    ContextMenu = 8
    # special method for three 2d UI.
    Change = 9


@dataclasses.dataclass
class Object3dBaseProps(ThreeBasicProps):
    # position already exists in base flex props, so we use another name
    position: Union[Vector3Type, Undefined] = undefined
    rotation: Union[Vector3Type, Undefined] = undefined
    up: Union[Vector3Type, Undefined] = undefined
    scale: Union[Vector3Type, Undefined] = undefined
    visible: Union[bool, Undefined] = undefined
    receive_shadow: Union[bool, Undefined] = undefined
    cast_shadow: Union[bool, Undefined] = undefined
    # event: {
    #   type
    #   stopPropagation
    # }


@dataclasses.dataclass
class Object3dContainerBaseProps(Object3dBaseProps, ContainerBaseProps):
    pass


T_o3d_prop = TypeVar("T_o3d_prop", bound=Object3dBaseProps)
T_o3d_container_prop = TypeVar("T_o3d_container_prop",
                               bound=Object3dContainerBaseProps)


class Object3dBase(ThreeComponentBase[T_o3d_prop]):

    def __init__(self,
                 base_type: UIType,
                 prop_cls: Type[T_o3d_prop],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, base_type, prop_cls, queue)

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
        await self.send_app_event_and_wait(
            self.update_object3d_event(position, rotation, up, scale, visible))


class Object3dWithEventBase(Object3dBase[T_o3d_prop]):

    def __init__(self,
                 base_type: UIType,
                 prop_cls: Type[T_o3d_prop],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(base_type, prop_cls, uid, queue)
        self._pointer_event_map: Dict[PointerEventType,
                                      Union[EventHandler, Undefined]] = {
                                          PointerEventType.Click: undefined,
                                          PointerEventType.DoubleClick:
                                          undefined,
                                          PointerEventType.Enter: undefined,
                                          PointerEventType.Leave: undefined,
                                          PointerEventType.Over: undefined,
                                          PointerEventType.Out: undefined,
                                          PointerEventType.Up: undefined,
                                          PointerEventType.Down: undefined,
                                          PointerEventType.ContextMenu:
                                          undefined,
                                          PointerEventType.Change: undefined,
                                      }

    def to_dict(self):
        res = super().to_dict()
        evs = []
        for k, v in self._pointer_event_map.items():
            if k == PointerEventType.Change:
                continue
            if not isinstance(v, Undefined):
                d = v.to_dict()
                d["type"] = k.value
                evs.append(d)
        res["props"]["usedEvents"] = evs
        return res

    def set_pointer_callback(
            self,
            on_click: Optional[Union[EventHandler, Undefined]] = None,
            on_double_click: Optional[Union[EventHandler,
                                            Undefined]] = None,
            on_enter: Optional[Union[EventHandler, Undefined]] = None,
            on_leave: Optional[Union[EventHandler, Undefined]] = None,
            on_over: Optional[Union[EventHandler, Undefined]] = None,
            on_out: Optional[Union[EventHandler, Undefined]] = None,
            on_up: Optional[Union[EventHandler, Undefined]] = None,
            on_down: Optional[Union[EventHandler, Undefined]] = None,
            on_context_menu: Optional[Union[EventHandler,
                                            Undefined]] = None,
            on_change: Optional[Union[EventHandler, Undefined]] = None):
        pointer_event_map = {
            PointerEventType.Click: on_click,
            PointerEventType.DoubleClick: on_double_click,
            PointerEventType.Enter: on_enter,
            PointerEventType.Leave: on_leave,
            PointerEventType.Over: on_over,
            PointerEventType.Out: on_out,
            PointerEventType.Up: on_up,
            PointerEventType.Down: on_down,
            PointerEventType.ContextMenu: on_context_menu,
            PointerEventType.Change: on_change,
        }
        for k, v in pointer_event_map.items():
            if v is not None:
                self._pointer_event_map[k] = v

    async def handle_event(self, ev: Any):
        # ev: [type, data]
        type, data = ev
        ev_type = PointerEventType(type)
        handler = self._pointer_event_map[ev_type]
        if isinstance(handler, Undefined):
            return
        if self.props.status == UIRunStatus.Running.value:
            # TODO send exception if ignored click
            print("IGNORE EVENT", self.props.status)
            return
        elif self.props.status == UIRunStatus.Stop.value:
            self.state_change_callback(data)

            def ccb(cb):
                return lambda: cb(data)

            self._task = asyncio.create_task(
                self.run_callback(ccb(handler.cb), True))


class Object3dContainerBase(ThreeContainerBase[T_o3d_container_prop, T_child]):

    def __init__(self,
                 base_type: UIType,
                 prop_cls: Type[T_o3d_container_prop],
                 children: Dict[str, T_child],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(base_type, prop_cls, uid, queue, uid_to_comp,
                         children, inited)

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
        await self.send_app_event_and_wait(
            self.update_object3d_event(position, rotation, up, scale, visible))


class O3dContainerWithEventBase(Object3dContainerBase[T_o3d_container_prop,
                                                      T_child]):

    def __init__(self,
                 base_type: UIType,
                 prop_cls: Type[T_o3d_container_prop],
                 children: Dict[str, T_child],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(base_type, prop_cls, children, uid, queue,
                         uid_to_comp, inited)
        self._pointer_event_map: Dict[PointerEventType,
                                      Union[EventHandler, Undefined]] = {
                                          PointerEventType.Click: undefined,
                                          PointerEventType.DoubleClick:
                                          undefined,
                                          PointerEventType.Enter: undefined,
                                          PointerEventType.Leave: undefined,
                                          PointerEventType.Over: undefined,
                                          PointerEventType.Out: undefined,
                                          PointerEventType.Up: undefined,
                                          PointerEventType.Down: undefined,
                                          PointerEventType.ContextMenu:
                                          undefined,
                                          PointerEventType.Change: undefined,
                                      }

    def to_dict(self):
        res = super().to_dict()
        evs = []
        for k, v in self._pointer_event_map.items():
            if k == PointerEventType.Change:
                continue
            if not isinstance(v, Undefined):
                evs.append({
                    "type": k.value,
                    "stopPropagation": v.stop_propagation
                })
        res["props"]["usedEvents"] = evs
        return res

    def set_pointer_callback(
            self,
            on_click: Optional[Union[EventHandler, Undefined]] = None,
            on_double_click: Optional[Union[EventHandler,
                                            Undefined]] = None,
            on_enter: Optional[Union[EventHandler, Undefined]] = None,
            on_leave: Optional[Union[EventHandler, Undefined]] = None,
            on_over: Optional[Union[EventHandler, Undefined]] = None,
            on_out: Optional[Union[EventHandler, Undefined]] = None,
            on_up: Optional[Union[EventHandler, Undefined]] = None,
            on_down: Optional[Union[EventHandler, Undefined]] = None,
            on_context_menu: Optional[Union[EventHandler,
                                            Undefined]] = None,
            on_change: Optional[Union[EventHandler, Undefined]] = None):
        pointer_event_map = {
            PointerEventType.Click: on_click,
            PointerEventType.DoubleClick: on_double_click,
            PointerEventType.Enter: on_enter,
            PointerEventType.Leave: on_leave,
            PointerEventType.Over: on_over,
            PointerEventType.Out: on_out,
            PointerEventType.Up: on_up,
            PointerEventType.Down: on_down,
            PointerEventType.ContextMenu: on_context_menu,
            PointerEventType.Change: on_change,
        }
        for k, v in pointer_event_map.items():
            if v is not None:
                self._pointer_event_map[k] = v

    async def handle_event(self, ev: Any):
        # ev: [type, data]
        type, data = ev
        ev_type = PointerEventType(type)
        handler = self._pointer_event_map[ev_type]
        if isinstance(handler, Undefined):
            return
        if self.props.status == UIRunStatus.Running.value:
            # TODO send exception if ignored click
            print("IGNORE EVENT", self.props.status)
            return
        elif self.props.status == UIRunStatus.Stop.value:
            if ev_type == PointerEventType.Change:
                self.state_change_callback(data)

            def ccb(cb):
                return lambda: cb(data)

            self._task = asyncio.create_task(
                self.run_callback(ccb(handler.cb), True, sync_first=False))


@dataclasses.dataclass
class PointProps(ThreeBasicProps):
    points: Union[np.ndarray, Undefined] = undefined
    intensity: Union[np.ndarray, Undefined] = undefined
    colors: Union[np.ndarray, Undefined] = undefined
    attrs: Union[np.ndarray, Undefined] = undefined
    attr_fields: Union[List[str], Undefined] = undefined
    size_attenuation: bool = False
    size: float = 3.0


class Points(ThreeComponentBase[PointProps]):

    def __init__(self,
                 limit: int,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreePoints, PointProps, queue)
        self.props.points = np.zeros((0, 3), np.float32)
        self.limit = limit

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["points"] = self.props.points
        res["intensity"] = self.props.intensity
        res["colors"] = self.props.colors
        res["attrs"] = self.props.attrs
        res["attr_fields"] = self.props.attr_fields
        return res

    def validate_props(self, props: Dict[str, Any]):
        if "points" in props:
            return props["points"].shape[0] <= self.limit
        return False

    def to_dict(self):
        res = super().to_dict()
        res["limit"] = self.limit
        return res

    async def update_points(self,
                            points: np.ndarray,
                            intensity: Optional[Union[np.ndarray,
                                                      Undefined]] = None,
                            colors: Optional[Union[np.ndarray,
                                                   Undefined]] = None,
                            attrs: Optional[Union[np.ndarray,
                                                  Undefined]] = None,
                            attr_fields: Optional[List[str]] = None):
        # TODO better check, we must handle all errors before sent to frontend.
        assert points.ndim == 2 and points.shape[1] in [
            3, 4
        ], "only support 3 or 4 features for points"
        assert points.shape[
            0] <= self.limit, f"your points size must smaller than limit {self.limit}"
        if points.shape[1] == 4 and intensity is None:
            intensity = points[:, 3]
            points = points[:, :3]

        upd: Dict[str, Any] = {
            "points": points,
        }
        if intensity is not None:
            upd["intensity"] = intensity
            self.props.intensity = intensity
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
            upd["attr_fields"] = attr_fields
            if attr_fields is not None:
                self.props.attr_fields = attr_fields
        self.props.points = points
        await self.send_app_event_and_wait(self.create_update_event(upd))

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class SegmentsProps(ThreeBasicProps):
    lines: Union[np.ndarray, Undefined] = undefined
    colors: Union[np.ndarray, Undefined] = undefined
    line_width: float = 1.0
    color: Union[str, Undefined] = undefined
    transparent: Union[bool, Undefined] = undefined
    opacity: Union[float, Undefined] = undefined


class Segments(ThreeComponentBase[SegmentsProps]):

    def __init__(self,
                 limit: int,
                 line_width: float = 1.0,
                 color: Union[str, Undefined] = undefined,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeSegments, SegmentsProps, queue)
        self.props.lines = np.zeros((0, 2, 3), np.float32)
        self.props.line_width = line_width
        self.limit = limit
        self.props.colors = undefined
        self.props.color = color

    def to_dict(self):
        res = super().to_dict()
        res["limit"] = self.limit
        return res

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["lines"] = self.props.lines
        res["colors"] = self.props.colors
        return res

    def validate_props(self, props: Dict[str, Any]):
        if "lines" in props:
            return props["lines"].shape[0] <= self.limit
        return False

    async def update_lines(self,
                           lines: np.ndarray,
                           colors: Optional[Union[np.ndarray,
                                                  Undefined]] = None):
        assert lines.ndim == 3 and lines.shape[1] == 2 and lines.shape[
            2] == 3, f"{lines.shape} lines must be [N, 2, 3]"
        assert lines.shape[
            0] <= self.limit, f"your line size must smaller than limit {self.limit}"
        upd: Dict[str, Any] = {
            "lines": lines,
        }
        if colors is not None:
            if not isinstance(colors, Undefined):
                assert colors.shape[0] == lines.shape[
                    0], "color shape not valid"
            upd["colors"] = colors
            self.props.colors = colors
        self.props.lines = lines.astype(np.float32)
        await self.send_app_event_and_wait(self.create_update_event(upd))

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class Boxes2DProps(Object3dBaseProps):
    centers: Union[np.ndarray, Undefined] = undefined
    dimensions: Union[np.ndarray, Undefined] = undefined
    colors: Union[np.ndarray, Undefined] = undefined
    attrs: Union[List[str], Undefined] = undefined
    color: Union[str, Undefined] = undefined
    alpha: Union[NumberType, Undefined] = undefined
    line_color: Union[str, Undefined] = undefined
    line_width: Union[NumberType, Undefined] = undefined
    hover_line_color: Union[str, Undefined] = undefined
    hover_line_width: Union[NumberType, Undefined] = undefined


class Boxes2D(Object3dWithEventBase[Boxes2DProps]):

    def __init__(self,
                 limit: int,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(UIType.ThreeBoxes2D, Boxes2DProps, uid, queue)
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
        await self.send_app_event_and_wait(self.create_update_event(upd))

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
    edge_width: Union[float, Undefined] = undefined
    edge_color: Union[str, Undefined] = undefined
    emissive: Union[str, Undefined] = undefined
    color: Union[str, Undefined] = undefined
    opacity: Union[float, Undefined] = undefined
    edge_opacity: Union[float, Undefined] = undefined


class BoundingBox(Object3dWithEventBase[BoundingBoxProps]):

    def __init__(self,
                 dimension: Vector3Type,
                 edge_width: float = 4,
                 edge_color: str = "green",
                 emissive: str = "red",
                 color: str = "red",
                 opacity: float = 0.5,
                 edge_opacity: float = 0.5,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(UIType.ThreeBoundingBox, BoundingBoxProps, uid, queue)
        self.props.dimension = dimension
        self.props.edge_width = edge_width
        self.props.edge_color = edge_color
        self.props.emissive = emissive
        self.props.color = color
        self.props.opacity = opacity
        self.props.edge_opacity = edge_opacity

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


@dataclasses.dataclass
class AxesHelperProps(Object3dBaseProps):
    length: NumberType = 10


class AxesHelper(ThreeComponentBase[AxesHelperProps]):

    def __init__(self,
                 length: float,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeAxesHelper, AxesHelperProps, queue)
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
class InfiniteGridHelperProps(Object3dBaseProps):
    size1: Union[NumberType, Undefined] = undefined
    size2: Union[NumberType, Undefined] = undefined
    color: Union[str, Undefined] = undefined
    distance: Union[NumberType, Undefined] = undefined


class InfiniteGridHelper(ThreeComponentBase[InfiniteGridHelperProps]):

    def __init__(self,
                 size1: float,
                 size2: float,
                 color: str,
                 distance: float = 8000,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeInfiniteGridHelper,
                         InfiniteGridHelperProps, queue)
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
                 children: Dict[str, ThreeComponentType],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.ThreeGroup, Object3dContainerBaseProps,
                         children, uid, queue, uid_to_comp, inited)

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

    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(UIType.ThreeImage, ImageProps, uid, queue)

    async def show(self, image: np.ndarray):
        encoded = _encode_image_bytes(image)
        self.props.image = encoded
        await self.send_app_event_and_wait(
            self.create_update_event({
                "image": encoded,
            }))

    async def show_raw(self, image_bytes: bytes, suffix: str):
        await self.send_app_event_and_wait(
            self.show_raw_event(image_bytes, suffix))

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

    def __init__(self,
                 makeDefault: bool,
                 fov: Union[float, Undefined] = undefined,
                 aspect: Union[float, Undefined] = undefined,
                 near: Union[float, Undefined] = undefined,
                 far: Union[float, Undefined] = undefined,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(UIType.ThreePerspectiveCamera, PerspectiveCameraProps,
                         uid, queue)
        self.props.fov = fov
        self.props.aspect = aspect
        self.props.near = near
        self.props.far = far
        self.makeDefault = makeDefault

    # TODO from camera matrix and intrinsics
    def to_dict(self):
        res = super().to_dict()
        res["makeDefault"] = self.makeDefault
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
class OrthographicCameraProps(Object3dBaseProps):
    zoom: Union[float, Undefined] = undefined
    near: Union[float, Undefined] = undefined
    far: Union[float, Undefined] = undefined


class OrthographicCamera(Object3dBase[OrthographicCameraProps]):

    def __init__(self,
                 makeDefault: bool,
                 near: Union[float, Undefined] = undefined,
                 far: Union[float, Undefined] = undefined,
                 zoom: Union[float, Undefined] = undefined,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(UIType.ThreeOrthographicCamera,
                         OrthographicCameraProps, uid, queue)
        self.props.zoom = zoom
        self.props.near = near
        self.props.far = far
        self.makeDefault = makeDefault

    # TODO from camera matrix and intrinsics
    def to_dict(self):
        res = super().to_dict()
        res["makeDefault"] = self.makeDefault
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
class OrbitControlProps(ThreeBasicProps):
    enable_damping: Union[bool, Undefined] = undefined
    damping_factor: Union[NumberType, Undefined] = undefined
    min_distance: Union[NumberType, Undefined] = undefined
    max_distance: Union[NumberType, Undefined] = undefined
    min_polar_angle: Union[NumberType, Undefined] = undefined
    max_polar_angle: Union[NumberType, Undefined] = undefined
    min_zoom: Union[NumberType, Undefined] = undefined
    max_zoom: Union[NumberType, Undefined] = undefined
    enable_zoom: Union[bool, Undefined] = undefined
    zoom_speed: Union[NumberType, Undefined] = undefined
    enable_rotate: Union[bool, Undefined] = undefined
    rotate_speed: Union[NumberType, Undefined] = undefined
    enable_pan: Union[bool, Undefined] = undefined
    pan_speed: Union[NumberType, Undefined] = undefined
    key_pan_speed: Union[NumberType, Undefined] = undefined


class MapControl(ThreeComponentBase[OrbitControlProps]):

    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeMapControl, OrbitControlProps, queue)
        self.props.enable_damping = True
        self.props.damping_factor = 0.25
        self.props.min_distance = 1
        self.props.max_distance = 100

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class OrbitControl(ThreeComponentBase[OrbitControlProps]):

    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeOrbitControl, OrbitControlProps,
                         queue)
        self.props.enable_damping = True
        self.props.damping_factor = 0.25
        self.props.min_distance = 1
        self.props.max_distance = 100

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
    min_polar_angle: Union[float, Undefined] = undefined
    max_polar_angle: Union[float, Undefined] = undefined


class PointerLockControl(ThreeComponentBase[PointerLockControlProps]):

    def __init__(self,
                 enabled: Union[bool, Undefined] = undefined,
                 min_polar_angle: Union[float, Undefined] = undefined,
                 max_polar_angle: Union[float, Undefined] = undefined,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreePointerLockControl,
                         PointerLockControlProps, queue)
        self.props.enabled = enabled
        self.props.min_polar_angle = min_polar_angle
        self.props.max_polar_angle = max_polar_angle

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
        super().__init__(uid, UIType.ThreeFirstPersonControl,
                         FirstPersonControlProps, queue)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class FlexAutoReflow(ThreeComponentBase[ThreeBasicProps]):

    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeFlexAutoReflow, ThreeBasicProps,
                         queue)

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

    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeFlexManualReflow, FlexManualReflowProps,
                         queue)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def reflow(self):
        await self.send_app_event_and_wait(self.update_event(timestamp=str(time.time())))

class ThreeCanvas(MUIContainerBase[ContainerBaseProps, ThreeComponentType]):

    def __init__(self,
                 children: Dict[str, ThreeComponentType],
                 background: Union[str, Undefined] = undefined,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.ThreeCanvas, ContainerBaseProps, uid, queue,
                         uid_to_comp, children, inited)
        self.background = background

    def to_dict(self):
        res = super().to_dict()
        if not isinstance(self.background, Undefined):
            res["backgroundColor"] = self.background
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
class TransformControlsProps(ContainerBaseProps):
    enabled: Union[bool, Undefined] = undefined
    axis: Union[str, Undefined] = undefined
    mode: Union[str, Undefined] = undefined
    translation_snap: Union[NumberType, Undefined] = undefined
    rotation_snap: Union[NumberType, Undefined] = undefined
    scale_snap: Union[NumberType, Undefined] = undefined
    space: Union[str, Undefined] = undefined
    size: Union[NumberType, Undefined] = undefined
    show_x: Union[bool, Undefined] = undefined
    show_y: Union[bool, Undefined] = undefined
    show_z: Union[bool, Undefined] = undefined
    object3d_uid: Union[str, Undefined] = undefined


class TransformControls(ThreeComponentBase[TransformControlsProps]):

    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeTransformControl,
                         TransformControlsProps, queue)

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
    scale_factor: Union[int, Undefined] = undefined


class Flex(ThreeContainerBase[ThreeFlexProps, ThreeComponentType]):

    def __init__(self,
                 children: Dict[str, ThreeComponentType],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.ThreeFlex, ThreeFlexProps, uid, queue,
                         uid_to_comp, children, inited)

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
    center_anchor: Union[bool, Undefined] = undefined  # false


class ItemBox(ThreeContainerBase[ThreeFlexItemBoxProps, ThreeComponentType]):

    def __init__(self,
                 children: Dict[str, ThreeComponentType],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.ThreeFlexItemBox, ThreeFlexItemBoxProps, uid,
                         queue, uid_to_comp, children, inited)

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
    box.props.flex_direction = "column"
    return box


def HBox(children: Dict[str, ThreeComponentType]):
    box = ItemBox(children)
    box.props.flex_direction = "row"
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
    distance_factor: Union[float, Undefined] = undefined
    sprite: Union[bool, Undefined] = undefined
    transform: Union[bool, Undefined] = undefined
    z_index_range: Union[List[Union[int, float]], Undefined] = undefined
    wrapper_class: Union[str, Undefined] = undefined
    pointer_events: Union[PointerEventsProperties, Undefined] = undefined
    occlude: Union[bool, Undefined] = undefined
    inside_flex: Union[bool, Undefined] = undefined


class Html(Object3dContainerBase[HtmlProps, MUIComponentType]):
    """we can use MUI components only in Html.
    """

    def __init__(self,
                 children: Dict[str, MUIComponentType],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.ThreeHtml, HtmlProps, children, uid, queue,
                         uid_to_comp, inited)

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
    font_size: Union[NumberType, Undefined] = undefined
    max_width: Union[NumberType, Undefined] = undefined
    line_height: Union[NumberType, Undefined] = undefined
    letter_spacing: Union[NumberType, Undefined] = undefined
    text_align: Union[Literal["left", "right", "center", "justify"],
                      Undefined] = undefined
    font: Union[str, Undefined] = undefined
    anchor_x: Union[NumberType, Literal["left", "center", "right"],
                    Undefined] = undefined
    anchor_y: Union[NumberType, Literal["top", "top-baseline", "middle",
                                        "bottom-baseline", "bottom"],
                    Undefined] = undefined
    clip_rect: Union[Tuple[NumberType, NumberType, NumberType, NumberType],
                     Undefined] = undefined
    depth_offset: Union[NumberType, Undefined] = undefined
    direction: Union[Literal["auto", "ltr", "rtl"], Undefined] = undefined
    overflow_wrap: Union[Literal["normal", "break-word"],
                         Undefined] = undefined
    white_space: Union[Literal['normal', 'nowrap'],
                       Undefined] = undefined
    outline_width: Union[ValueType, Undefined] = undefined
    outline_offset_x: Union[ValueType, Undefined] = undefined
    outline_offset_y: Union[ValueType, Undefined] = undefined
    outline_blur: Union[ValueType, Undefined] = undefined
    outline_color: Union[str, Undefined] = undefined
    outline_opacity: Union[NumberType, Undefined] = undefined
    stroke_width: Union[ValueType, Undefined] = undefined
    stroke_color: Union[NumberType, Undefined] = undefined
    stroke_opacity: Union[NumberType, Undefined] = undefined
    fill_opacity: Union[NumberType, Undefined] = undefined


class Text(Object3dWithEventBase[TextProps]):
    """we can use MUI components only in Html.
    """

    def __init__(self,
                 init: str,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(UIType.ThreeText, TextProps, uid, queue)
        self.props.value = init

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["value"] = self.props.value
        return res

    async def update_value(self, value: str):
        self.props.value = value
        upd: Dict[str, Any] = {"value": value}
        await self.send_app_event_and_wait(self.create_update_event(upd))

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
    vertex_colors: Union[Tuple[NumberType, NumberType, NumberType],
                         Undefined] = undefined
    line_width: Union[NumberType, Undefined] = undefined
    transparent: Union[bool, Undefined] = undefined
    opacity: Union[NumberType, Undefined] = undefined


class Line(Object3dWithEventBase[LineProps]):
    """we can use MUI components only in Html.
    """

    def __init__(self,
                 points: Union[np.ndarray, List[Tuple[NumberType, NumberType,
                                                      NumberType]]],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(UIType.ThreeLine, LineProps, uid, queue)
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
        self.text = Text(label).prop(font_size=label_size,
                                     position=positions,
                                     color=color,
                                     stroke_opacity=opacity,
                                     fill_opacity=opacity)
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
    Tube = 5
    Torus = 6
    TorusKnot = 7
    Tetrahedron = 8
    Ring = 9
    Polyhedron = 10
    Icosahedron = 11
    Octahedron = 12
    Dodecahedron = 13
    Extrude = 14
    Lathe = 15
    Capsule = 16


class PathOpType(enum.Enum):
    Move = 0
    Line = 1
    BezierCurve = 2
    QuadraticCurve = 3
    AbsArc = 4
    Arc = 5


@dataclasses.dataclass
class SimpleGeometryProps(ThreeGeometryPropsBase):
    shape_type: int = 0
    shape_args: Union[List[Union[int, float, bool]], Undefined] = undefined


@dataclasses.dataclass
class PathShapeProps(ThreeGeometryPropsBase):
    path_ops: List[Tuple[int, List[Union[float, bool]]]] = dataclasses.field(
        default_factory=list)
    curve_segments: Union[NumberType, Undefined] = undefined


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

    def __init__(self,
                 shape: Shape,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeShape, PathShapeProps, queue)
        self.props.path_ops = shape.ops

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

    def __init__(self,
                 width: float,
                 height: float,
                 radius: float,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        shape = _rounded_shape(-width / 2, -height / 2, width, height, radius)
        super().__init__(shape, uid, queue)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class SimpleGeometry(ThreeGeometryBase[SimpleGeometryProps]):

    def __init__(self,
                 type: GeometryType,
                 args: List[Union[int, float, bool]],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeSimpleGeometry, SimpleGeometryProps,
                         queue)
        self.props.shape_type = type.value
        self.props.shape_args = args

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
                 depth_segments: int = 1,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        args: List[Union[int, float, bool]] = [
            width, height, depth, width_segments, height_segments,
            depth_segments
        ]
        super().__init__(GeometryType.Box, args, uid, queue)


class CapsuleGeometry(SimpleGeometry):

    def __init__(self,
                 radius: float = 1,
                 length: float = 1,
                 cap_subdivisions: int = 4,
                 radial_segments: int = 8,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        args: List[Union[int, float, bool]] = [
            radius, length, cap_subdivisions, radial_segments
        ]
        super().__init__(GeometryType.Capsule, args, uid, queue)


class CircleGeometry(SimpleGeometry):

    def __init__(self,
                 radius: float = 1,
                 segments: int = 8,
                 theta_start: float = 0,
                 theta_length: float = np.pi * 2,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        args: List[Union[int, float, bool]] = [
            radius, segments, theta_start, theta_length
        ]
        super().__init__(GeometryType.Circle, args, uid, queue)


class ConeGeometry(SimpleGeometry):

    def __init__(self,
                 radius: float = 1,
                 height: float = 1,
                 radial_segments: int = 8,
                 height_segments: int = 1,
                 open_ended: bool = False,
                 theta_start: float = 0,
                 theta_length: float = np.pi * 2,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        args: List[Union[int, float, bool]] = [
            radius, height, radial_segments, height_segments, open_ended,
            theta_start, theta_length
        ]
        super().__init__(GeometryType.Cone, args, uid, queue)


@dataclasses.dataclass
class MeshBasicMaterialProps(ThreeMaterialPropsBase):
    color: Union[str, Undefined] = undefined
    wireframe: Union[bool, Undefined] = undefined
    vertex_colors: Union[bool, Undefined] = undefined
    fog: Union[bool, Undefined] = undefined


@dataclasses.dataclass
class MeshStandardMaterialProps(MeshBasicMaterialProps):
    emissive: Union[str, Undefined] = undefined
    roughness: Union[NumberType, Undefined] = undefined
    metalness: Union[NumberType, Undefined] = undefined
    flag_shading: Union[bool, Undefined] = undefined


@dataclasses.dataclass
class MeshLambertMaterialProps(MeshBasicMaterialProps):
    emissive: Union[str, Undefined] = undefined


@dataclasses.dataclass
class MeshMatcapMaterialProps(ThreeMaterialPropsBase):
    flag_shading: Union[bool, Undefined] = undefined
    color: Union[str, Undefined] = undefined


@dataclasses.dataclass
class MeshNormalMaterialProps(ThreeMaterialPropsBase):
    flag_shading: Union[bool, Undefined] = undefined
    wireframe: Union[bool, Undefined] = undefined


@dataclasses.dataclass
class MeshPhongMaterialProps(MeshBasicMaterialProps):
    reflectivity: Union[NumberType, Undefined] = undefined
    refraction_ratio: Union[NumberType, Undefined] = undefined
    emissive: Union[str, Undefined] = undefined
    specular: Union[str, Undefined] = undefined
    shininess: Union[NumberType, Undefined] = undefined


@dataclasses.dataclass
class MeshPhysicalMaterialProps(MeshStandardMaterialProps):
    reflectivity: Union[NumberType, Undefined] = undefined
    clearcoat: Union[NumberType, Undefined] = undefined
    clearcoat_roughness: Union[NumberType, Undefined] = undefined


@dataclasses.dataclass
class MeshToonMaterialProps(ThreeMaterialPropsBase):
    color: Union[str, Undefined] = undefined


class MeshBasicMaterial(ThreeMaterialBase[MeshBasicMaterialProps]):

    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeMeshMaterial, MeshBasicMaterialProps,
                         queue)
        self.props.material_type = MeshMaterialType.Basic.value

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshStandardMaterial(ThreeMaterialBase[MeshStandardMaterialProps]):

    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeMeshMaterial,
                         MeshStandardMaterialProps, queue)
        self.props.material_type = MeshMaterialType.Standard.value

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshLambertMaterial(ThreeMaterialBase[MeshLambertMaterialProps]):

    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeMeshMaterial,
                         MeshLambertMaterialProps, queue)
        self.props.material_type = MeshMaterialType.Lambert.value

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshMatcapMaterial(ThreeMaterialBase[MeshMatcapMaterialProps]):

    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeMeshMaterial,
                         MeshMatcapMaterialProps, queue)
        self.props.material_type = MeshMaterialType.Matcap.value

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshNormalMaterial(ThreeMaterialBase[MeshNormalMaterialProps]):

    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeMeshMaterial,
                         MeshNormalMaterialProps, queue)
        self.props.material_type = MeshMaterialType.Normal.value

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshPhongMaterial(ThreeMaterialBase[MeshPhongMaterialProps]):

    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeMeshMaterial, MeshPhongMaterialProps,
                         queue)
        self.props.material_type = MeshMaterialType.Phong.value

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshPhysicalMaterial(ThreeMaterialBase[MeshPhysicalMaterialProps]):

    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeMeshMaterial,
                         MeshPhysicalMaterialProps, queue)
        self.props.material_type = MeshMaterialType.Physical.value

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshToonMaterial(ThreeMaterialBase[MeshToonMaterialProps]):

    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeMeshMaterial, MeshToonMaterialProps,
                         queue)
        self.props.material_type = MeshMaterialType.Toon.value

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
class MeshProps(Object3dContainerBaseProps):
    hover_color: Union[str, Undefined] = undefined
    click_color: Union[str, Undefined] = undefined
    toggle_mode: Union[bool, Undefined] = undefined
    toggled: Union[bool, Undefined] = undefined


class Mesh(O3dContainerWithEventBase[MeshProps, ThreeComponentType]):

    def __init__(self,
                 geometry: ThreeGeometryBase,
                 material: ThreeMaterialBase,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        self.geometry = geometry
        self.material = material
        children: Dict[str, ThreeComponentType] = {
            "geometry": geometry,
            "material": material,
        }
        super().__init__(UIType.ThreeMesh, MeshProps, children, uid, queue,
                         uid_to_comp, inited)
        self.props.toggled = False

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["toggled"] = self.props.toggled
        return res

    def state_change_callback(self, data: bool):
        self.props.toggled = data

    async def set_checked(self, checked: bool):
        ev = self.create_update_event({
            "toggled": checked,
        })
        await self.send_app_event_and_wait(ev)

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
    render_priority: Union[int, Undefined] = undefined


class Hud(ThreeContainerBase[HudProps, ThreeComponentType]):
    # TODO can/should group accept event?
    def __init__(self,
                 children: Dict[str, ThreeComponentType],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.ThreeHud, HudProps, uid, queue, uid_to_comp,
                         children, inited)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class ShapeButton(Group):

    def __init__(self,
                 name: str,
                 shape: Shape,
                 font_size: NumberType,
                 text_max_width: NumberType,
                 callback: Callable[[Any], _CORO_ANY],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        material = MeshBasicMaterial()
        material.prop(color="#393939")
        mesh = Mesh(ShapeGeometry(shape), material)
        mesh.set_pointer_callback(on_click=EventCallback(callback, True))
        mesh.prop(hover_color="#222222", click_color="#009A63")
        self.mesh = mesh
        text = Text(name)
        text.prop(font_size=font_size,
                  color="white",
                  position=(0, 0, 0),
                  max_width=text_max_width)
        children = {
            "mesh": mesh,
            "text": text,
        }
        super().__init__(children, uid, queue)
        self.name = name
        # self.callback = callback

    def to_dict(self):
        res = super().to_dict()
        res["name"] = self.name
        return res

    async def headless_click(self):
        uiev = UIEvent({self._flow_uid: [PointerEventType.Click, self.name]})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    def get_callback(self):
        res = self.mesh._pointer_event_map[PointerEventType.Click]
        assert not isinstance(res, Undefined)
        return res.cb

    def set_callback(self, val: Any):
        self.mesh.set_pointer_callback(on_click=EventCallback(val, True))
        # self.callback = val

    async def handle_event(self, ev: Any):
        if self.props.status == UIRunStatus.Running.value:
            # TODO send exception if ignored click
            print("IGNORE EVENT", self.props.status)
            return
        elif self.props.status == UIRunStatus.Stop.value:
            cb2 = self.get_callback()
            self._task = asyncio.create_task(
                self.run_callback(lambda: cb2(ev)))

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
                 font_size: Optional[float] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        if radius is None:
            radius = min(width, height) * 0.25
        if font_size is None:
            font_size = min(width, height) * 0.5
        material = MeshBasicMaterial()
        material.prop(color="#393939")
        mesh = Mesh(RoundedRectGeometry(width, height, radius), material)
        mesh.set_pointer_callback(on_click=EventCallback(callback, True))
        mesh.prop(hover_color="#222222", click_color="#009A63")
        self.mesh = mesh
        text = Text(name)
        text.prop(font_size=font_size,
                  color="white",
                  position=(0, 0, 0),
                  max_width=width)
        children = {
            "mesh": mesh,
            "text": text,
        }
        super().__init__(children, uid, queue)
        self.name = name
        # self.callback = callback

    def to_dict(self):
        res = super().to_dict()
        res["name"] = self.name
        return res

    async def headless_click(self):
        uiev = UIEvent({self._flow_uid: [PointerEventType.Click, self.name]})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    def get_callback(self):
        res = self.mesh._pointer_event_map[PointerEventType.Click]
        assert not isinstance(res, Undefined)
        return res.cb

    def set_callback(self, val: Any):
        self.mesh.set_pointer_callback(on_click=EventCallback(val, True))
        # self.callback = val

    # async def handle_event(self, ev: Any):
    #     if self.props.status == UIRunStatus.Running.value:
    #         # TODO send exception if ignored click
    #         print("IGNORE EVENT", self.props.status)
    #         return
    #     elif self.props.status == UIRunStatus.Stop.value:
    #         cb2 = self.get_callback()
    #         self._task = asyncio.create_task(self.run_callback(lambda: cb2(ev)))
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
                 font_size: Optional[float] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        if radius is None:
            radius = min(width, height) * 0.25
        if font_size is None:
            font_size = min(width, height) * 0.5
        material = MeshBasicMaterial()
        material.prop(color="#393939")
        mesh = Mesh(RoundedRectGeometry(width, height, radius), material)
        mesh.set_pointer_callback(on_change=EventCallback(callback, True))

        mesh.prop(hover_color="#222222",
                  click_color="#009A63",
                  toggle_mode=True)
        self.mesh = mesh

        text = Text(name)
        text.prop(font_size=font_size,
                  color="white",
                  position=(0, 0, 0),
                  max_width=width)
        children = {
            "mesh": mesh,
            "text": text,
        }
        super().__init__(children, uid, queue)
        self.name = name

    @property 
    def toggled(self):
        return self.mesh.props.toggled

    def to_dict(self):
        res = super().to_dict()
        res["name"] = self.name
        return res

    async def headless_toggle(self):
        uiev = UIEvent({self._flow_uid: [PointerEventType.Change, self.name]})
        return await self.put_app_event(
            AppEvent("", {AppEventType.UIEvent: uiev}))

    def get_callback(self):
        res = self.mesh._pointer_event_map[PointerEventType.Change]
        assert not isinstance(res, Undefined)
        return res.cb

    def set_callback(self, val: Any):
        self.mesh.set_pointer_callback(on_change=EventCallback(val, True))

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)
