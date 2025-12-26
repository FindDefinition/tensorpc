# Copyright 2024 Yan Yan
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

from collections.abc import Mapping
import enum
from typing import (TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, Coroutine, Dict,
                    Iterable, List, Optional, Sequence, Tuple, Type, TypeVar, Union)
from tensorpc.core.annolib import DataclassType
from tensorpc.dock.components.three.event import HudLayoutChangeEvent, KeyboardHoldEvent, PointerEvent, PointerMissedEvent, PoseChangeEvent, ViewportChangeEvent
from tensorpc.dock.core.appcore import Event, EventDataType
from typing_extensions import Literal
import tensorpc.core.dataclass_dispatch as dataclasses

import numpy as np
from typing_extensions import ParamSpec, TypeAlias, Annotated

from ....core.datamodel.typemetas import RangedFloat, RangedInt, Vector3Type, NumberType, ValueType
from ...core.component import (BasicProps, Component,
                    ContainerBase, ContainerBaseProps, EventHandler, MatchCase,
                    SimpleEventType, Fragment, FrontendEventType,
                    T_base_props, T_child, T_container_props, TaskLoopEvent,
                    UIEvent, UIRunStatus, UIType, Undefined,
                    undefined)
from ..mui import (FlexBoxProps, MUIFlexBoxProps, MUIComponentType,
                  MUIContainerBase, MenuItem)
from ...core.common import handle_standard_event
from tensorpc.core.datamodel import typemetas
if TYPE_CHECKING:
    from .uikit import Root, Fullscreen

_CORO_NONE: TypeAlias = Union[Coroutine[None, None, None], None]
_CORO_ANY: TypeAlias = Union[Coroutine[Any, None, None], Any]

CORO_NONE: TypeAlias = Union[Coroutine[None, None, None], None]

ThreeLayoutType: TypeAlias = Union[Sequence["ThreeComponentType"],
                                   Mapping[str, "ThreeComponentType"]]
ThreeEffectType: TypeAlias = Union[Sequence["ThreeEffectBase"],
                                   Mapping[str, "ThreeEffectBase"]]

P = ParamSpec('P')

class PyDanticConfigForNumpy:
    arbitrary_types_allowed = True


@dataclasses.dataclass
class ThreeBasicProps(BasicProps):
    pass

class Side(enum.Enum):
    FrontSide = 0
    BackSide = 1
    DoubleSide = 2


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


class SideType(enum.IntEnum):
    Front = 0
    Back = 1
    Double = 2

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

class BlendFunction(enum.IntEnum):
    SKIP = 0
    SET = 1
    ADD = 2
    ALPHA = 3
    AVERAGE = 4
    COLOR = 5
    COLOR_BURN = 6
    COLOR_DODGE = 7
    DARKEN = 8
    DIFFERENCE = 9
    DIVIDE = 10
    DST = 11
    EXCLUSION = 12
    HARD_LIGHT = 13
    HARD_MIX = 14
    HUE = 15
    INVERT = 16
    INVERT_RGB = 17
    LIGHTEN = 18
    LINEAR_BURN = 19
    LINEAR_DODGE = 20
    LINEAR_LIGHT = 21
    LUMINOSITY = 22
    MULTIPLY = 23
    NEGATION = 24
    NORMAL = 25
    OVERLAY = 26
    PIN_LIGHT = 27
    REFLECT = 28
    SATURATION = 29
    SCREEN = 30
    SOFT_LIGHT = 31
    SRC = 32
    SUBTRACT = 33
    VIVID_LIGHT = 34


class ToneMapppingMode(enum.IntEnum):
    REINHARD = 0
    REINHARD2 = 1
    REINHARD2_ADAPTIVE = 2
    OPTIMIZED_CINEON = 3
    ACES_FILMIC = 4
    UNCHARTED2 = 5

@dataclasses.dataclass
class ThreeMaterialPropsBaseProps:
    # deprecated, only works in MeshBasicMaterialV1 and MeshStandardMaterialV1
    # materialType: int = 0
    transparent: Union[bool, Undefined] = undefined
    opacity: Annotated[Union[NumberType, Undefined],
                       typemetas.CommonObject(default=1.0)] = undefined
    depthTest: Union[bool, Undefined] = undefined
    depthWrite: Union[bool, Undefined] = undefined
    alphaTest: Union[NumberType, Undefined] = undefined
    visible: Union[bool, Undefined] = undefined
    side: Union[SideType, Undefined] = undefined


@dataclasses.dataclass
class ThreeMaterialPropsBase(ThreeMaterialPropsBaseProps, ThreeBasicProps):
    pass

class ThreeComponentBase(Component[T_base_props, "ThreeComponentType"]):
    pass


class ThreeContainerBase(ContainerBase[T_container_props, T_child]):
    pass


class ThreeEffectBase(Component[T_base_props, "ThreeComponentType"]):
    pass


class ThreeMaterialBase(ThreeComponentBase[T_base_props]):
    pass

class ThreeMaterialContainerBase(ThreeContainerBase[T_container_props, T_child]):
    pass

class ThreeGeometryBase(ThreeComponentBase[T_base_props]):
    pass


@dataclasses.dataclass
class ThreeGeometryPropsBase(ThreeBasicProps):
    pass


T_material_prop = TypeVar("T_material_prop", bound=ThreeMaterialPropsBase)
T_geometry_prop = TypeVar("T_geometry_prop", bound=ThreeGeometryPropsBase)

ThreeComponentType: TypeAlias = Union[ThreeComponentBase, ThreeContainerBase, Fragment, MatchCase, "DataPortal"]
ThreeComponentTypeForCanvas: TypeAlias = Union[ThreeComponentBase, ThreeContainerBase, Fragment, MatchCase, "DataPortal", "Root", "Fullscreen"]


def is_three_component(obj: Component):
    return isinstance(obj, (ThreeComponentBase, ThreeContainerBase, Fragment,
                            Canvas, ThreeEffectBase))


@dataclasses.dataclass
class DataPortalProps(ContainerBaseProps):
    comps: List[Component] = dataclasses.field(default_factory=list)

class DataPortal(ThreeContainerBase[DataPortalProps, ThreeComponentType]):
    def __init__(self, source: Component, children: Optional[ThreeLayoutType] = None) -> None:
        if children is not None and isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}
        allowed_comp_types = {UIType.DataModel, UIType.ThreeURILoaderContext, UIType.ThreeCubeCamera}
        
        sources = [source]
        for comp in sources:
            assert comp._flow_comp_type in allowed_comp_types, "DataPortal only support DataModel and resource loaders."
        assert len(sources) == 1, "DataPortal only support one source."
        super().__init__(UIType.DataPortal, DataPortalProps, children, allowed_events=[])
        self.prop(comps=sources)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

@dataclasses.dataclass
class Object3dBaseProps(ThreeBasicProps):
    # position already exists in base flex props, so we use another name
    position: Annotated[Union[Vector3Type, Undefined],
                        typemetas.Vector3(0.01)] = undefined
    rotation: Annotated[Union[Vector3Type, Undefined],
                        typemetas.Vector3(0.1)] = undefined
    up: Union[Vector3Type, Undefined] = undefined
    scale: Annotated[
        Union[Vector3Type, NumberType, Undefined],
        typemetas.RangedVector3(0.01, 10.0, 0.01, default=(1, 1,
                                                           1))] = undefined
    visible: Annotated[Union[bool, Undefined],
                       typemetas.CommonObject(default=True)] = undefined
    receiveShadow: Union[bool, Undefined] = undefined
    castShadow: Union[bool, Undefined] = undefined
    renderOrder: Union[int, Undefined] = undefined
    layers: Annotated[Union[int, Undefined],
                      typemetas.RangedInt(0, 31, 1, default=0)] = undefined


@dataclasses.dataclass
class Object3dContainerBaseProps(Object3dBaseProps, ContainerBaseProps):
    pass


T_o3d_prop = TypeVar("T_o3d_prop", bound=Object3dBaseProps)
T_o3d_container_prop = TypeVar("T_o3d_container_prop",
                               bound=Object3dContainerBaseProps)


class Object3dBase(ThreeComponentBase[T_o3d_prop]):

    def __init__(
            self,
            base_type: UIType,
            prop_cls: Type[T_o3d_prop],
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

    def __init__(
            self,
            base_type: UIType,
            prop_cls: Type[T_o3d_prop],
            allowed_events: Optional[Iterable[EventDataType]] = None) -> None:
        if allowed_events is None:
            allowed_events = []

        super().__init__(base_type,
                         prop_cls,
                         allowed_events=[
                             FrontendEventType.Click.value,
                             FrontendEventType.DoubleClick.value,
                             FrontendEventType.Enter.value,
                             FrontendEventType.Leave.value,
                             FrontendEventType.Over.value,
                             FrontendEventType.Out.value,
                             FrontendEventType.Up.value,
                             FrontendEventType.Down.value,
                             FrontendEventType.Move.value,
                             FrontendEventType.Missed.value,
                             FrontendEventType.ContextMenu.value,
                             FrontendEventType.Wheel.value,
                         ] + list(allowed_events))
        self.event_double_click = self._create_event_slot_noarg(
            FrontendEventType.DoubleClick, lambda x: PointerEvent(**x))
        self.event_click = self._create_event_slot_noarg(FrontendEventType.Click, lambda x: PointerEvent(**x))
        self.event_enter = self._create_event_slot(FrontendEventType.Enter, lambda x: PointerEvent(**x))
        self.event_leave = self._create_event_slot(FrontendEventType.Leave, lambda x: PointerEvent(**x))
        self.event_over = self._create_event_slot(FrontendEventType.Over, lambda x: PointerEvent(**x))
        self.event_out = self._create_event_slot(FrontendEventType.Out, lambda x: PointerEvent(**x))
        self.event_up = self._create_event_slot(FrontendEventType.Up, lambda x: PointerEvent(**x))
        self.event_down = self._create_event_slot(FrontendEventType.Down, lambda x: PointerEvent(**x))
        self.event_move = self._create_event_slot(FrontendEventType.Move, lambda x: PointerEvent(**x))
        self.event_missed = self._create_event_slot(FrontendEventType.Missed, lambda x: PointerMissedEvent(**x))
        self.event_wheel = self._create_event_slot(FrontendEventType.Wheel, lambda x: PointerEvent(**x))

        self.event_context_menu = self._create_event_slot(
            FrontendEventType.ContextMenu)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync)


class Object3dContainerBase(ThreeContainerBase[T_o3d_container_prop, T_child]):

    def __init__(
            self,
            base_type: UIType,
            prop_cls: Type[T_o3d_container_prop],
            children: Union[Mapping[str, T_child], DataclassType],
            allowed_events: Optional[Iterable[EventDataType]] = None) -> None:
        super().__init__(base_type,
                         prop_cls,
                         children,
                         allowed_events=allowed_events)

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

    def __init__(
            self,
            base_type: UIType,
            prop_cls: Type[T_o3d_container_prop],
            children: Union[Mapping[str, T_child], DataclassType],
            allowed_events: Optional[Iterable[EventDataType]] = None) -> None:
        if allowed_events is None:
            allowed_events = []
        super().__init__(base_type,
                         prop_cls,
                         children,
                         allowed_events=[
                             FrontendEventType.Click.value,
                             FrontendEventType.DoubleClick.value,
                             FrontendEventType.Enter.value,
                             FrontendEventType.Leave.value,
                             FrontendEventType.Over.value,
                             FrontendEventType.Out.value,
                             FrontendEventType.Up.value,
                             FrontendEventType.Down.value,
                             FrontendEventType.Move.value,
                             FrontendEventType.Missed.value,
                             FrontendEventType.ContextMenu.value,
                             FrontendEventType.Wheel.value,
                         ] + list(allowed_events))
        self.event_double_click = self._create_event_slot_noarg(
            FrontendEventType.DoubleClick, lambda x: PointerEvent(**x))
        self.event_click = self._create_event_slot_noarg(FrontendEventType.Click, lambda x: PointerEvent(**x))
        self.event_enter = self._create_event_slot(FrontendEventType.Enter, lambda x: PointerEvent(**x))
        self.event_leave = self._create_event_slot(FrontendEventType.Leave, lambda x: PointerEvent(**x))
        self.event_over = self._create_event_slot(FrontendEventType.Over, lambda x: PointerEvent(**x))
        self.event_out = self._create_event_slot(FrontendEventType.Out, lambda x: PointerEvent(**x))
        self.event_up = self._create_event_slot(FrontendEventType.Up, lambda x: PointerEvent(**x))
        self.event_down = self._create_event_slot(FrontendEventType.Down, lambda x: PointerEvent(**x))
        self.event_move = self._create_event_slot(FrontendEventType.Move, lambda x: PointerEvent(**x))
        self.event_missed = self._create_event_slot(FrontendEventType.Missed, lambda x: PointerMissedEvent(**x))
        self.event_context_menu = self._create_event_slot(
            FrontendEventType.ContextMenu)
        self.event_wheel = self._create_event_slot(FrontendEventType.Wheel, lambda x: PointerEvent(**x))

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync)

@dataclasses.dataclass
class PerspectiveCameraProps(Object3dContainerBaseProps):
    makeDefault: Union[bool, Undefined] = undefined
    fov: Union[float, Undefined] = undefined
    aspect: Union[float, Undefined] = undefined
    near: Union[float, Undefined] = undefined
    far: Union[float, Undefined] = undefined

@dataclasses.dataclass
class ThreeCanvasProps(MUIFlexBoxProps):
    threeBackgroundColor: Union[str, Undefined] = undefined
    allowKeyboardEvent: Union[bool, Undefined] = undefined
    tabIndex: Union[int, Undefined] = undefined
    shadows: Union[bool, Undefined] = undefined
    enablePerf: Union[bool, Undefined] = undefined
    perfPosition: Union[Literal['top-right', 'top-left', 'bottom-right',
                                'bottom-left'], Undefined] = undefined
    flat: Union[bool, Undefined] = undefined
    linear: Union[bool, Undefined] = undefined
    dpr: Union[Tuple[int, int], Undefined] = undefined
    raycastLayerMask: Union[int, Undefined] = undefined
    isViewMode: Union[bool, Undefined] = undefined
    menuItems: Union[List[MenuItem], Undefined] = undefined
    localClippingEnabled: Union[bool, Undefined] = undefined
    cameraProps: Union[PerspectiveCameraProps, Undefined] = undefined

@dataclasses.dataclass
class ThreeViewProps(MUIFlexBoxProps):
    threeBackgroundColor: Union[str, Undefined] = undefined
    allowKeyboardEvent: Union[bool, Undefined] = undefined
    tabIndex: Union[int, Undefined] = undefined
    enablePerf: Union[bool, Undefined] = undefined
    perfPosition: Union[Literal['top-right', 'top-left', 'bottom-right',
                                'bottom-left'], Undefined] = undefined
    elementId: Union[str, Undefined] = undefined
    className: Union[str, Undefined] = undefined
    visible: Union[bool, Undefined] = undefined
    index: Union[int, Undefined] = undefined
    frames: Union[int, Undefined] = undefined
    menuItems: Union[List[MenuItem], Undefined] = undefined

class Canvas(MUIContainerBase[ThreeCanvasProps, ThreeComponentTypeForCanvas]):

    def __init__(self,
                 children: Union[List[ThreeComponentTypeForCanvas],
                                 Dict[str, ThreeComponentTypeForCanvas]],
                 background: Union[str, Undefined] = undefined) -> None:
        if isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.ThreeCanvas,
                         ThreeCanvasProps,
                         children,
                         allowed_events=[
                             FrontendEventType.ContextMenuSelect.value,
                             FrontendEventType.KeyHold.value,
                             FrontendEventType.CanvasViewportChange.value,
                         ])
        self.props.threeBackgroundColor = background
        self.event_context_menu = self._create_event_slot(
            FrontendEventType.ContextMenuSelect)
        self.event_keyboard_hold = self._create_event_slot(
            FrontendEventType.KeyHold, lambda x: KeyboardHoldEvent(**x))
        self.event_viewport_change = self._create_event_slot(
            FrontendEventType.CanvasViewportChange, lambda x: ViewportChangeEvent(**x))

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync)

class View(MUIContainerBase[ThreeViewProps, ThreeComponentType]):

    def __init__(
        self, children: ThreeLayoutType
    ) -> None:
        if isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.ThreeView,
                         ThreeViewProps,
                         children,
                         allowed_events=[
                            FrontendEventType.ContextMenuSelect.value,
                            FrontendEventType.KeyHold.value,
                            FrontendEventType.CanvasViewportChange.value,
                         ])
        self.event_context_menu = self._create_event_slot(
            FrontendEventType.ContextMenuSelect)
        self.event_keyboard_hold = self._create_event_slot(
            FrontendEventType.KeyHold, lambda x: KeyboardHoldEvent(**x))
        self.event_viewport_change = self._create_event_slot(
            FrontendEventType.CanvasViewportChange, lambda x: ViewportChangeEvent(**x))

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync)


class ViewCanvas(MUIContainerBase[ThreeCanvasProps, MUIComponentType]):

    def __init__(self,
                 children: Union[List[MUIComponentType],
                                 Dict[str, MUIComponentType]],
                 background: Union[str, Undefined] = undefined) -> None:
        if isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.ThreeCanvas, ThreeCanvasProps, children)
        self.props.threeBackgroundColor = background
        self.props.isViewMode = True

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class PivotControlsCommonProps:
    scale: Union[NumberType, Undefined] = undefined
    lineWidth: Union[NumberType, Undefined] = undefined
    fixed: Union[bool, Undefined] = undefined
    anchor: Union[Vector3Type, Undefined] = undefined
    activeAxes: Union[Tuple[bool, bool, bool], Undefined] = undefined
    axisColors: Union[Tuple[ValueType, ValueType, ValueType],
                      Undefined] = undefined
    hoveredColor: Union[ValueType, Undefined] = undefined
    depthTest: Union[bool, Undefined] = undefined
    opacity: Annotated[Union[NumberType, Undefined],
                       typemetas.CommonObject(default=1.0)] = undefined
    visible: Union[bool, Undefined] = undefined
    annotations: Union[bool, Undefined] = undefined


@dataclasses.dataclass
class InteractableProps:
    # used for events. for example, if you set userdata
    # in a mesh inside a group container, when you add handler
    # in group and click mesh, the group will receive
    # event with userdata of this mesh.
    userData: Union[Undefined, Any] = undefined
    toggled: Union[bool, Undefined] = undefined
    # mesh have four state: normal, hover, click (point down), toggled (selected)
    # each state can have different override props.
    enableHover: Union[bool, Undefined] = undefined
    enableClick: Union[bool, Undefined] = undefined
    enableToggle: Union[bool, Undefined] = undefined
    # you must use select context to enable select.
    # to enable outline support, you must also put outline
    # into select context.
    enableSelect: Union[bool, Undefined] = undefined
    enableSelectOutline: Union[bool, Undefined] = undefined
    # follow https://docs.pmnd.rs/react-three-fiber/api/objects#piercing-into-nested-properties
    # to set nested props of a mesh.
    hoverOverrideProps: Union[Undefined, Dict[str, Any]] = undefined
    clickOverrideProps: Union[Undefined, Dict[str, Any]] = undefined
    toggleOverrideProps: Union[Undefined, Dict[str, Any]] = undefined
    selectOverrideProps: Union[Undefined, Dict[str, Any]] = undefined

    enablePivotControl: Union[bool, Undefined] = undefined
    enablePivotOnSelected: Union[bool, Undefined] = undefined
    pivotControlProps: PivotControlsCommonProps = dataclasses.field(
        default_factory=PivotControlsCommonProps)
    pivotDebounce: Union[NumberType, Undefined] = undefined

@dataclasses.dataclass
class GroupProps(Object3dContainerBaseProps):
    userData: Union[Any, Undefined] = undefined
    enableSelect: Union[bool, Undefined] = undefined
    variant: Union[Literal["default", "faceToCamera", "relativeToCamera"],
                   Undefined] = undefined

class Group(O3dContainerWithEventBase[GroupProps, ThreeComponentType]):
    # TODO can/should group accept event?
    def __init__(
        self, children: ThreeLayoutType
    ) -> None:
        if isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.ThreeGroup, GroupProps, children)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

@dataclasses.dataclass
class HudGroupProps(Object3dContainerBaseProps):
    userData: Union[Any, Undefined] = undefined
    # auto (keep aspect ratio, width) by default.
    alignContent: Union[Literal["auto", "width", "height", "stretch"], bool, Undefined] = undefined
    # fully controlled content width/height in three unit.
    # we won't measure the content size because
    # it's impossible to trigger re-measure when child
    # size changed.
    childWidth: Union[NumberType, Undefined] = undefined
    childHeight: Union[NumberType, Undefined] = undefined
    childWidthScale: Union[NumberType, Undefined] = undefined
    childHeightScale: Union[NumberType, Undefined] = undefined

    # uv sign. default: 1, -1
    xSign: Union[NumberType, Undefined] = undefined
    ySign: Union[NumberType, Undefined] = undefined
    # pixel or percent, not threejs units
    top: Union[ValueType, Undefined] = undefined
    left: Union[ValueType, Undefined] = undefined
    right: Union[ValueType, Undefined] = undefined
    bottom: Union[ValueType, Undefined] = undefined
    padding: Union[ValueType, Undefined] = undefined
    width: Union[ValueType, Undefined] = undefined
    height: Union[ValueType, Undefined] = undefined
    borderColor: Union[ValueType, Undefined] = undefined
    borderRadius: Union[NumberType, Undefined] = undefined
    borderWidth: Union[NumberType, Undefined] = undefined
    # fully controlled scroll values (unified, 0-1), used to implement 
    # scroll bar via DataModel.
    scrollValueX: Union[NumberType, Undefined] = undefined
    scrollValueY: Union[NumberType, Undefined] = undefined
    alwaysPortal: Union[bool, Undefined] = undefined
    anchor: Union[tuple[int, int], Undefined] = undefined

class HudGroup(O3dContainerWithEventBase[HudGroupProps, ThreeComponentType]):
    """Group that used in HUD (must be child of camera)
    support pixel or percent position (top/left/right/bottom), width/height and padding.

    don't support flex layout, only absolute layout. all elements inside this group
    should located in z = 0 plane.
    """
    def __init__(
        self, children: ThreeLayoutType
    ) -> None:
        if isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.ThreeHudGroup, HudGroupProps, children, allowed_events=[
            FrontendEventType.HudGroupLayoutChange.value,
        ])
        self.event_hud_layout_change = self._create_event_slot(
            FrontendEventType.HudGroupLayoutChange, lambda x: HudLayoutChangeEvent(**x))

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

# @dataclasses.dataclass
# class MeshUserData:
#     enableSelect: Union[bool, Undefined] = undefined
#     data: Union[Undefined, Any] = undefined


@dataclasses.dataclass
class PrimitiveMeshProps(Object3dContainerBaseProps, InteractableProps):
    # used for events. for example, if you set userdata
    # in a mesh inside a group container, when you add handler
    # in group and click mesh, the group will receive
    # event with userdata of this mesh.
    userData: Union[Undefined, Any] = undefined


@dataclasses.dataclass
class MeshProps(PrimitiveMeshProps):
    hoverColor: Union[str, Undefined] = undefined
    clickColor: Union[str, Undefined] = undefined
    toggleMode: Union[bool, Undefined] = undefined
    toggled: Union[bool, Undefined] = undefined


@dataclasses.dataclass
class MeshChangeData:
    toggled: Union[bool, Undefined] = undefined
    matrix: Union[Undefined, List[float]] = undefined
    position: Union[Undefined, Vector3Type] = undefined
    rotation: Union[Undefined, Vector3Type] = undefined


class Mesh(O3dContainerWithEventBase[PrimitiveMeshProps, ThreeComponentType]):
    """standard three mesh.
    mesh itself don't have much props, but you can use 
    dash-case format to set nested object prop:
    ```Python
    mesh.update_raw_props({
        "material-color": "red"
    })
    ```
    this also works for hover/click/toggle override props.

    see https://docs.pmnd.rs/react-three-fiber/api/objects#piercing-into-nested-properties
    """

    def __init__(self, children: ThreeLayoutType) -> None:
        if isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}

        super().__init__(UIType.ThreePrimitiveMesh,
                         PrimitiveMeshProps,
                         children,
                         allowed_events=[
                             FrontendEventType.MeshPoseChange.value,
                         ])
        self.event_pose_change = self._create_event_slot(FrontendEventType.MeshPoseChange, lambda x: PoseChangeEvent(**x))

    async def handle_event(self, ev: Event, is_sync: bool = False):
        await handle_standard_event(self,
                                    ev,
                                    sync_state_after_change=False,
                                    is_sync=is_sync)

    def state_change_callback(
            self,
            value: dict,
            type: ValueType = FrontendEventType.Change.value):
        if "toggled" in value:
            self.props.toggled = value["toggled"]
        else:
            assert "position" in value
            assert "rotation" in value
            self.props.position = value["position"]
            self.props.rotation = value["rotation"]

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

@dataclasses.dataclass
class DataListGroupProps(Object3dContainerBaseProps):
    userData: Union[Any, Undefined] = undefined
    idKey: Union[str, Undefined] = undefined
    dataList: List[Any] = dataclasses.field(default_factory=list)


class DataListGroup(O3dContainerWithEventBase[DataListGroupProps, ThreeComponentType]):
    """ flex box that use data list and data model component to render
    list of data with same UI components.
    """

    @dataclasses.dataclass
    class ChildDef:
        component: Component

    def __init__(self,
                 children: Component,
                 init_data_list: Optional[List[Any]] = None) -> None:
        super().__init__(
            UIType.ThreeDataListGroup,
            DataListGroupProps,
            DataListGroup.ChildDef(children))
        # backend events
        if init_data_list is not None:
            self.props.dataList = init_data_list

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

