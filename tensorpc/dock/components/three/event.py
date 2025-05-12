import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.annolib import Undefined, undefined
from typing import Any, Union, Optional
from tensorpc.core.datamodel.typemetas import (NumberType, Vector2Type, Vector3Type, ColorRGB, ColorRGBA)


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
class PointerEvent:
    distance: NumberType 
    pointer: Vector2Type
    unprojectedPoint: Vector3Type
    ray: Ray
    offset: Vector2Type

    distanceToRay: Union[Undefined, NumberType] = undefined
    point: Union[Undefined, Vector3Type] = undefined
    index: Union[Undefined, int] = undefined
    face: Union[Face, Undefined] = undefined
    faceIndex: Union[Undefined, int] = undefined
    uv: Union[Undefined, Vector2Type] = undefined
    instanceId: Union[Undefined, int] = undefined
    userData: Union[Undefined, Any] = undefined

@dataclasses.dataclass
class CameraEvent:
    position: Vector3Type
    rotation: Vector3Type
    matrixWorld: list[float]
    size: Vector2Type
    fov: Union[Undefined, NumberType] = undefined
    aspect: Union[Undefined, NumberType] = undefined

@dataclasses.dataclass
class KeyboardEvent:
    code: str 
    altKey: bool
    ctrlKey: bool
    metaKey: bool
    shiftKey: bool

@dataclasses.dataclass
class KeyboardHoldEvent(KeyboardEvent):
    deltaTime: NumberType
    elapsedTime: NumberType
