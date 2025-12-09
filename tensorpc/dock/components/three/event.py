from types import NoneType
import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.annolib import Undefined, undefined
from typing import Any, TypeAlias, Union, Optional
from tensorpc.core.datamodel.typemetas import (NumberType, Vector2Type, Vector3Type, ColorRGB, ColorRGBA)
from tensorpc.dock.components.mui.event import KeyboardEvent, KeyboardHoldEvent
EulerTuple: TypeAlias = Union[tuple[float, float, float], tuple[float, float, float, str]]

@dataclasses.dataclass
class Ray:
    origin: Vector3Type
    direction: Vector3Type 

@dataclasses.dataclass
class Face:
    a: NumberType
    b: NumberType
    c: NumberType
    normal: Vector3Type
    materialIndex: int 

@dataclasses.dataclass
class PointerMissedEvent:
    offset: Vector2Type

@dataclasses.dataclass
class _PointerWheel:
    deltaX: NumberType
    deltaY: NumberType
    deltaZ: NumberType
    deltaMode: int


@dataclasses.dataclass
class PointerEvent:
    distance: NumberType 
    pointer: Vector2Type
    unprojectedPoint: Vector3Type
    ray: Ray
    offset: Vector2Type
    point: Vector3Type
    pointLocal: Vector3Type

    distanceToRay: Union[Undefined, NumberType] = undefined
    index: Union[Undefined, int] = undefined
    face: Optional[Union[Face, Undefined]] = undefined
    faceIndex: Union[Undefined, int] = undefined
    uv: Union[Undefined, Vector2Type] = undefined
    instanceId: Union[Undefined, int] = undefined
    userData: Union[Undefined, Any] = undefined
    wheel: Union[Undefined, _PointerWheel] = undefined
    # for pointer capture
    numIntersections: int = 0
    dataIndexes: Union[Undefined, list[int]] = undefined


@dataclasses.dataclass
class CameraEvent:
    position: Vector3Type
    rotation: Vector3Type
    matrixWorld: list[float]
    size: Vector2Type
    fov: Union[Undefined, NumberType] = undefined
    aspect: Union[Undefined, NumberType] = undefined

@dataclasses.dataclass
class ViewportChangeEvent:
    width: NumberType
    height: NumberType
    aspect: NumberType
    dpr: NumberType
    factor: NumberType
    distance: NumberType

@dataclasses.dataclass
class HudLayoutChangeEvent:
    positionX: Optional[NumberType] = None 
    positionY: Optional[NumberType] = None
    scaleX: Optional[NumberType] = None
    scaleY: Optional[NumberType] = None
    innerPositionX: NumberType = 0.0
    innerPositionY: NumberType = 0.0
    innerSizeX: NumberType = 1.0
    innerSizeY: NumberType = 1.0
    scaledChildSizeX: Optional[NumberType] = None
    scaledChildSizeY: Optional[NumberType] = None
    # container / child
    scrollFactorX: NumberType = 1.0
    scrollFactorY: NumberType = 1.0
    # child / container
    overflowFactorX: NumberType = 1.0
    overflowFactorY: NumberType = 1.0

@dataclasses.dataclass
class PoseChangeEvent:
    positionWorld: Vector3Type
    positionOffsetWorld: Vector3Type
    positionLocal: Vector3Type
    positionOffsetLocal: Vector3Type
    rotation: Union[Undefined, EulerTuple] = undefined
