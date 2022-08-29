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
from tensorpc import compat
from typing import (Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
                    Iterable, List, Optional, Tuple, Type, TypeVar, Union)

if compat.Python3_8AndLater:
    from typing import Literal
else:
    from typing_extensions import Literal
from typing_extensions import TypeAlias

import numpy as np
from tensorpc.utils.uniquename import UniqueNamePool
import dataclasses
from ..core import AppEvent, BasicProps, Component, NumberType, T_child, TaskLoopEvent, UIEvent, UIRunStatus, UIType, ContainerBase, Undefined, ValueType, undefined, ComponentBaseProps, T_base_props
from .mui import FlexBoxProps, MUIContainerBase, _encode_image_bytes, MUIComponentType, _handle_button_event

Vector3Type: TypeAlias = Tuple[float, float, float]

_CORO_NONE: TypeAlias = Union[Coroutine[None, None, None], None]


@dataclasses.dataclass
class ThreeBasicProps(BasicProps):
    pass


@dataclasses.dataclass
class ThreeFlexPropsBase(FlexBoxProps):
    pass


@dataclasses.dataclass
class ThreeFlexItemPropsBase(ComponentBaseProps):
    pass


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


class ThreeContainerBase(ContainerBase[T_base_props, T_child]):
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

ThreeComponentType = Union[ThreeComponentBase, ThreeContainerBase,
                           ThreeBasicProps, ThreeFlexPropsBase,
                           ThreeFlexItemPropsBase]

ThreeMaterialType = Union[ThreeMaterialBase[T_base_props],
                          ThreeMaterialPropsBase]


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



T_o3d_prop = TypeVar("T_o3d_prop", bound=Object3dBaseProps)


class Object3dBase(ThreeComponentBase[T_o3d_prop]):
    def __init__(self,
                 base_type: UIType,
                 prop_cls: Type[T_o3d_prop],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, base_type, prop_cls, queue)

    def get_state(self):
        state = super().get_state()
        if not isinstance(self.props.position, Undefined):
            state["position"] = self.props.position
        if not isinstance(self.props.rotation, Undefined):
            state["rotation"] = self.props.rotation
        if not isinstance(self.props.scale, Undefined):
            state["scale"] = self.props.scale
        if not isinstance(self.props.up, Undefined):
            state["up"] = self.props.up
        if not isinstance(self.props.visible, Undefined):
            state["visible"] = self.props.visible
        if not isinstance(self.props.receive_shadow, Undefined):
            state["receive_shadow"] = self.props.receive_shadow
        if not isinstance(self.props.cast_shadow, Undefined):
            state["cast_shadow"] = self.props.cast_shadow
        return state

    def set_state(self, state: Dict[str, Any]):
        super().set_state(state)
        if "position" in state:
            self.props.position = state["position"]
        if "rotation" in state:
            self.props.rotation = state["rotation"]
        if "scale" in state:
            self.props.scale = state["scale"]
        if "up" in state:
            self.props.up = state["up"]
        if "visible" in state:
            self.props.visible = state["visible"]
        if "visible" in state:
            self.props.visible = state["visible"]
        if "receive_shadow" in state:
            self.props.receive_shadow = state["receive_shadow"]
        if "cast_shadow" in state:
            self.props.cast_shadow = state["cast_shadow"]

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


class EventCallback:
    def __init__(self,
                 cb: Callable[[Any], _CORO_NONE],
                 stop_propagation: bool = False) -> None:
        self.cb = cb
        self.stop_propagation = stop_propagation


PointerEventCBType: TypeAlias = EventCallback


class Object3dWithEventBase(Object3dBase[T_o3d_prop]):
    def __init__(self,
                 base_type: UIType,
                 prop_cls: Type[T_o3d_prop],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(base_type, prop_cls, uid, queue)
        self._pointer_event_map: Dict[PointerEventType,
                                      Union[PointerEventCBType, Undefined]] = {
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
                                          PointerEventType.Change:
                                          undefined,
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
        res["usedEvents"] = evs
        return res

    def set_pointer_callback(self,
                     on_click: Optional[Union[PointerEventCBType,
                                              Undefined]] = None,
                     on_double_click: Optional[Union[PointerEventCBType,
                                                     Undefined]] = None,
                     on_enter: Optional[Union[PointerEventCBType,
                                              Undefined]] = None,
                     on_leave: Optional[Union[PointerEventCBType,
                                              Undefined]] = None,
                     on_over: Optional[Union[PointerEventCBType,
                                             Undefined]] = None,
                     on_out: Optional[Union[PointerEventCBType,
                                            Undefined]] = None,
                     on_up: Optional[Union[PointerEventCBType,
                                           Undefined]] = None,
                     on_down: Optional[Union[PointerEventCBType,
                                             Undefined]] = None,
                     on_context_menu: Optional[Union[PointerEventCBType,
                                                     Undefined]] = None,
                     on_change: Optional[Union[PointerEventCBType,
                                                     Undefined]] = None):
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
        if self._status == UIRunStatus.Running:
            # TODO send exception if ignored click
            print("IGNORE EVENT", self._status)
            return
        elif self._status == UIRunStatus.Stop:
            self.state_change_callback(data)

            def ccb(cb):
                return lambda: cb(data)

            self._task = asyncio.create_task(
                self.run_callback(ccb(handler.cb), True))


class Object3dContainerBase(ThreeContainerBase[T_o3d_prop, T_child]):
    def __init__(self,
                 base_type: UIType,
                 prop_cls: Type[T_o3d_prop],
                 init_dict: Dict[str, T_child],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(base_type, prop_cls, uid, queue, uid_to_comp,
                         init_dict, inited)

    def get_state(self):
        state = super().get_state()
        if not isinstance(self.props.position, Undefined):
            state["position"] = self.props.position
        if not isinstance(self.props.rotation, Undefined):
            state["rotation"] = self.props.rotation
        if not isinstance(self.props.scale, Undefined):
            state["scale"] = self.props.scale
        if not isinstance(self.props.up, Undefined):
            state["up"] = self.props.up
        if not isinstance(self.props.visible, Undefined):
            state["visible"] = self.props.visible
        if not isinstance(self.props.receive_shadow, Undefined):
            state["receive_shadow"] = self.props.receive_shadow
        if not isinstance(self.props.cast_shadow, Undefined):
            state["cast_shadow"] = self.props.cast_shadow
        return state

    def set_state(self, state: Dict[str, Any]):
        super().set_state(state)
        if "position" in state:
            self.props.position = state["position"]
        if "rotation" in state:
            self.props.rotation = state["rotation"]
        if "scale" in state:
            self.props.scale = state["scale"]
        if "up" in state:
            self.props.up = state["up"]
        if "visible" in state:
            self.props.visible = state["visible"]
        if "visible" in state:
            self.props.visible = state["visible"]
        if "receive_shadow" in state:
            self.props.receive_shadow = state["receive_shadow"]
        if "cast_shadow" in state:
            self.props.cast_shadow = state["cast_shadow"]

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


class O3dContainerWithEventBase(Object3dContainerBase[T_o3d_prop, T_child]):
    def __init__(self,
                 base_type: UIType,
                 prop_cls: Type[T_o3d_prop],
                 init_dict: Dict[str, T_child],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(base_type, prop_cls, init_dict, uid, queue,
                         uid_to_comp, inited)
        self._pointer_event_map: Dict[PointerEventType,
                                      Union[PointerEventCBType, Undefined]] = {
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
                                          PointerEventType.Change:
                                          undefined,
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
        res["usedEvents"] = evs
        return res

    def set_pointer_callback(self,
                     on_click: Optional[Union[PointerEventCBType,
                                              Undefined]] = None,
                     on_double_click: Optional[Union[PointerEventCBType,
                                                     Undefined]] = None,
                     on_enter: Optional[Union[PointerEventCBType,
                                              Undefined]] = None,
                     on_leave: Optional[Union[PointerEventCBType,
                                              Undefined]] = None,
                     on_over: Optional[Union[PointerEventCBType,
                                             Undefined]] = None,
                     on_out: Optional[Union[PointerEventCBType,
                                            Undefined]] = None,
                     on_up: Optional[Union[PointerEventCBType,
                                           Undefined]] = None,
                     on_down: Optional[Union[PointerEventCBType,
                                             Undefined]] = None,
                     on_context_menu: Optional[Union[PointerEventCBType,
                                                     Undefined]] = None,
                     on_change: Optional[Union[PointerEventCBType,
                                                     Undefined]] = None):
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
        if self._status == UIRunStatus.Running:
            # TODO send exception if ignored click
            print("IGNORE EVENT", self._status)
            return
        elif self._status == UIRunStatus.Stop:
            self.state_change_callback(data)

            def ccb(cb):
                return lambda: cb(data)

            self._task = asyncio.create_task(
                self.run_callback(ccb(handler.cb), True))


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
        if points.shape[1] == 4 and intensity is None:
            intensity = points[:, 3]
            points = points[:, :3]

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
        await self.send_app_event_and_wait(self.create_update_event(upd))

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

    def set_state(self, state: Dict[str, Any]):
        super().set_state(state)
        if "points" in state:
            if state["points"].shape[0] <= self.limit:
                self.points = state["points"]
                if "intensity" in state:
                    self.intensity = state["intensity"]
                if "colors" in state:
                    self.colors = state["colors"]
                if "attrs" in state:
                    self.attrs = state["attrs"]
                if "attrFields" in state:
                    self.attr_fields = state["attrFields"]
                if "size" in state:
                    self.size = state["size"]
                if "sizeAttenuation" in state:
                    self.sizeAttenuation = state["sizeAttenuation"]


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
        await self.send_app_event_and_wait(self.create_update_event(upd))

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

    def set_state(self, state: Dict[str, Any]):
        super().set_state(state)
        if "lines" in state:
            if state["lines"].shape[0] <= self.limit:
                self.lines = state["lines"]
                self.lineWidth = state["lineWidth"]
                self.opacity = state["opacity"]
                self.transparent = state["transparent"]
                self.color = state["color"]
                if "colors" in state:
                    self.colors = state["colors"]


class Boxes2D(Object3dWithEventBase[Object3dBaseProps]):
    def __init__(self,
                 limit: int,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(UIType.ThreeBoxes2D, Object3dBaseProps, uid, queue)
        self.centers = np.zeros((0, 2), np.float32)
        self.dimensions = np.zeros((0, 2), np.float32)
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
                           dimensions: Optional[np.ndarray] = None,
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

        if dimensions is not None:
            assert dimensions.shape[
                0] <= self.limit, f"your dimensions size must smaller than limit {self.limit}"
            self.dimensions = dimensions
            if dimensions.shape != self.centers.shape:
                dimensions = np.broadcast_to(dimensions, self.centers.shape)
            upd["dimensions"] = dimensions


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

        await self.send_app_event_and_wait(self.create_update_event(upd))

    def get_state(self):
        state = super().get_state()
        dims = self.dimensions
        if self.dimensions.shape != self.centers.shape:
            dims = np.broadcast_to(dims, self.centers.shape)
        state.update({
            "colors": self.colors,
            "color": self.color,
            "centers": self.centers,
            "dimensions": dims,
            "attrs": self.attrs,
            "alpha": self.alpha,
        })
        return state

    def set_state(self, state: Dict[str, Any]):
        super().set_state(state)
        if "centers" in state:
            if state["centers"].shape[0] <= self.limit:
                self.centers = state["centers"]
                self.dimensions = state["dimensions"]
                self.alpha = state["alpha"] if "alpha" in state else undefined
                self.opacity = state[
                    "opacity"] if "opacity" in state else undefined
                self.color = state["color"] if "color" in state else undefined
                self.colors = state[
                    "colors"] if "colors" in state else undefined
                self.attrs = state["attrs"] if "attrs" in state else undefined


class BoundingBox(Object3dWithEventBase[Object3dBaseProps]):
    def __init__(self,
                 dimension: Vector3Type,
                 edgeWidth: float = 4,
                 edgeColor: str = "green",
                 emissive: str = "red",
                 color: str = "red",
                 opacity: float = 0.5,
                 edgeOpacity: float = 0.5,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(UIType.ThreeBoundingBox, Object3dBaseProps, uid,
                         queue)
        self.dimension = dimension
        self.edgeWidth = edgeWidth
        self.edgeColor = edgeColor
        self.emissive = emissive
        self.color = color
        self.opacity = opacity
        self.edgeOpacity = edgeOpacity

    def get_state(self):
        state = super().get_state()
        state.update({
            "dimension": self.dimension,
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


class Group(Object3dContainerBase[Object3dBaseProps, ThreeComponentType]):
    # TODO can/should group accept event?
    def __init__(self,
                 init_dict: Dict[str, ThreeComponentType],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.ThreeGroup, Object3dBaseProps, init_dict, uid,
                         queue, uid_to_comp, inited)


class Image(Object3dWithEventBase[Object3dBaseProps]):
    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(UIType.ThreeImage, Object3dBaseProps, uid, queue)
        self.image_str: bytes = b""

    async def show(self, image: np.ndarray):
        encoded = _encode_image_bytes(image)
        self.image_str = encoded
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
        self.image_str = raw
        return self.create_update_event({
            "image": raw,
        })

    def get_state(self):
        state = super().get_state()
        state["image"] = self.image_str
        return state


class PerspectiveCamera(Object3dBase[Object3dBaseProps]):
    def __init__(self,
                 makeDefault: bool,
                 fov: Union[float, Undefined] = undefined,
                 aspect: Union[float, Undefined] = undefined,
                 near: Union[float, Undefined] = undefined,
                 far: Union[float, Undefined] = undefined,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(UIType.ThreePerspectiveCamera, Object3dBaseProps, uid,
                         queue)
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
        await self.send_app_event_and_wait(self.create_update_event(upd))

    def get_state(self):
        state = super().get_state()
        state.update({
            "fov": self.fov,
            "aspect": self.aspect,
            "near": self.near,
            "far": self.far,
        })
        return state


class OrthographicCamera(Object3dBase[Object3dBaseProps]):
    def __init__(self,
                 makeDefault: bool,
                 near: Optional[float] = None,
                 far: Optional[float] = None,
                 zoom: Optional[float] = None,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(UIType.ThreeOrthographicCamera, Object3dBaseProps,
                         uid, queue)
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
        await self.send_app_event_and_wait(self.create_update_event(upd))

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
        await self.send_app_event_and_wait(self.create_update_event(upd))

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
        await self.send_app_event_and_wait(self.create_update_event(upd))

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
        if not isinstance(self.enabled, Undefined):
            res["enabled"] = self.enabled
        if not isinstance(self.minPolarAngle, Undefined):
            res["minPolarAngle"] = self.minPolarAngle
        if not isinstance(self.maxPolarAngle, Undefined):
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
        super().__init__(uid, UIType.ThreeFirstPersonControl,
                         FirstPersonControlProps, queue)


class FlexAutoReflow(ThreeComponentBase[ThreeBasicProps]):
    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeFlexAutoReflow, ThreeBasicProps,
                         queue)


class ThreeCanvas(MUIContainerBase[BasicProps, ThreeComponentType]):
    def __init__(self,
                 init_dict: Dict[str, ThreeComponentType],
                 background: Union[str, Undefined] = undefined,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.ThreeCanvas, ThreeBasicProps, uid, queue,
                         uid_to_comp, init_dict, inited)
        self.background = background

    def to_dict(self):
        res = super().to_dict()
        if not isinstance(self.background, Undefined):
            res["backgroundColor"] = self.background
        return res


@dataclasses.dataclass
class ThreeFlexProps(ThreeFlexPropsBase):
    size: Union[Vector3Type, Undefined] = undefined
    position: Union[Vector3Type, Undefined] = undefined
    direction: Union[str, Undefined] = undefined
    plane: Union[str, Undefined] = undefined
    scale_factor: Union[int, Undefined] = undefined


class Flex(ThreeContainerBase[ThreeFlexProps, ThreeComponentType]):
    def __init__(self,
                 init_dict: Dict[str, ThreeComponentType],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.ThreeFlex, ThreeFlexProps, uid, queue,
                         uid_to_comp, init_dict, inited)


@dataclasses.dataclass
class ThreeFlexItemBoxProps(ThreeFlexItemPropsBase):
    center_anchor: Union[bool, Undefined] = undefined  # false


class ItemBox(ThreeContainerBase[ThreeFlexItemBoxProps, ThreeComponentType]):
    def __init__(self,
                 init_dict: Dict[str, ThreeComponentType],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.ThreeFlexItemBox, ThreeFlexItemBoxProps, uid,
                         queue, uid_to_comp, init_dict, inited)

    def get_state(self):
        state = super().get_state()
        print(state)
        return state


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
    enable_reflow: Union[bool, Undefined] = undefined


class Html(Object3dContainerBase[HtmlProps, MUIComponentType]):
    """we can use MUI components only in Html.
    """
    def __init__(self,
                 init_dict: Dict[str, MUIComponentType],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.ThreeHtml, HtmlProps, init_dict, uid, queue,
                         uid_to_comp, inited)


@dataclasses.dataclass
class TextProps(Object3dBaseProps):
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
    white_space: Union[Literal['normal', 'overflowWrap'],
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
        self.value = init

    def get_state(self):
        state = super().get_state()
        state.update({
            "value": self.value,
        })
        return state

    async def update_value(self, value: str):
        self.value = value
        upd: Dict[str, Any] = {"value": value}
        await self.send_app_event_and_wait(self.create_update_event(upd))


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


class MeshStandardMaterial(ThreeMaterialBase[MeshStandardMaterialProps]):
    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeMeshMaterial,
                         MeshStandardMaterialProps, queue)
        self.props.material_type = MeshMaterialType.Standard.value


class MeshLambertMaterial(ThreeMaterialBase[MeshLambertMaterialProps]):
    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeMeshMaterial,
                         MeshLambertMaterialProps, queue)
        self.props.material_type = MeshMaterialType.Lambert.value


class MeshMatcapMaterial(ThreeMaterialBase[MeshMatcapMaterialProps]):
    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeMeshMaterial,
                         MeshMatcapMaterialProps, queue)
        self.props.material_type = MeshMaterialType.Matcap.value


class MeshNormalMaterial(ThreeMaterialBase[MeshNormalMaterialProps]):
    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeMeshMaterial,
                         MeshNormalMaterialProps, queue)
        self.props.material_type = MeshMaterialType.Normal.value


class MeshPhongMaterial(ThreeMaterialBase[MeshPhongMaterialProps]):
    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeMeshMaterial, MeshPhongMaterialProps,
                         queue)
        self.props.material_type = MeshMaterialType.Phong.value


class MeshPhysicalMaterial(ThreeMaterialBase[MeshPhysicalMaterialProps]):
    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeMeshMaterial,
                         MeshPhysicalMaterialProps, queue)
        self.props.material_type = MeshMaterialType.Physical.value


class MeshToonMaterial(ThreeMaterialBase[MeshToonMaterialProps]):
    def __init__(self,
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None) -> None:
        super().__init__(uid, UIType.ThreeMeshMaterial, MeshToonMaterialProps,
                         queue)
        self.props.material_type = MeshMaterialType.Toon.value


MeshChildType: TypeAlias = Union[ThreeMaterialBase, ThreeMaterialPropsBase,
                                 ThreeGeometryPropsBase, ThreeGeometryBase]

@dataclasses.dataclass
class MeshProps(Object3dBaseProps):
    hover_color: Union[str, Undefined] = undefined 
    click_color: Union[str, Undefined] = undefined
    toggle_mode: Union[bool, Undefined] = undefined

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
        init_dict: Dict[str, ThreeComponentType] = {
            "geometry": geometry,
            "material": material,
        }
        super().__init__(UIType.ThreeMesh, MeshProps, init_dict, uid,
                         queue, uid_to_comp, inited)


@dataclasses.dataclass
class HudProps(ThreeFlexProps):
    render_priority: Union[int, Undefined] = undefined


class Hud(ThreeContainerBase[HudProps, ThreeComponentType]):
    # TODO can/should group accept event?
    def __init__(self,
                 init_dict: Dict[str, ThreeComponentType],
                 uid: str = "",
                 queue: Optional[asyncio.Queue] = None,
                 uid_to_comp: Optional[Dict[str, Component]] = None,
                 inited: bool = False) -> None:
        super().__init__(UIType.ThreeHud, HudProps, uid, queue, uid_to_comp,
                         init_dict, inited)

class Button(Group):

    def __init__(self,
                 name: str,
                 width: float,
                 height: float, 
                 callback: Callable[[Any], _CORO_NONE],
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
        text.prop(font_size=font_size, color="white", position=(0, 0, 0), max_width=width)
        init_dict = {
            "mesh": mesh,
            "text": text,
        }
        super().__init__(init_dict, uid, queue)
        self.name = name
        # self.callback = callback

    def to_dict(self):
        res = super().to_dict()
        res["name"] = self.name
        return res

    async def headless_click(self):
        return await self.queue.put(UIEvent({self.uid: [PointerEventType.Click, self.name]}))

    def get_callback(self):
        res = self.mesh._pointer_event_map[PointerEventType.Click]
        assert not isinstance(res, Undefined)
        return res.cb

    def set_callback(self, val: Any):
        self.mesh.set_pointer_callback(on_click=EventCallback(val, True))
        # self.callback = val

    async def handle_event(self, ev: Any):
        if self._status == UIRunStatus.Running:
            # TODO send exception if ignored click
            print("IGNORE EVENT", self._status)
            return
        elif self._status == UIRunStatus.Stop:
            cb2 = self.get_callback()
            self._task = asyncio.create_task(self.run_callback(lambda: cb2(ev)))

class ToggleButton(Group):

    def __init__(self,
                 name: str,
                 width: float,
                 height: float, 
                 callback: Callable[[Any], _CORO_NONE],
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

        mesh.prop(hover_color="#222222", click_color="#009A63", toggle_mode=True)
        self.mesh = mesh
    
        text = Text(name)
        text.prop(font_size=font_size, color="white", position=(0, 0, 0), max_width=width)
        init_dict = {
            "mesh": mesh,
            "text": text,
        }
        super().__init__(init_dict, uid, queue)
        self.name = name

    def to_dict(self):
        res = super().to_dict()
        res["name"] = self.name
        return res

    async def headless_toggle(self):
        return await self.queue.put(UIEvent({self.uid: [PointerEventType.Change, self.name]}))

    def get_callback(self):
        res = self.mesh._pointer_event_map[PointerEventType.Change]
        assert not isinstance(res, Undefined)
        return res.cb

    def set_callback(self, val: Any):
        self.mesh.set_pointer_callback(on_click=EventCallback(val, True))
