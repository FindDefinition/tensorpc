from types import NoneType
import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.annolib import Undefined, undefined
from typing import Any, TypeAlias, Union, Optional
from tensorpc.core.datamodel.typemetas import NumberType

@dataclasses.dataclass
class PointerEvent:
    clientX: NumberType
    clientY: NumberType
    screenX: NumberType
    screenY: NumberType
    offsetX: NumberType
    offsetY: NumberType
    offsetWidth: NumberType
    offsetHeight: NumberType
    clientWidth: NumberType
    clientHeight: NumberType
    button: int
    id: Optional[str] = None
    deltaMode: Optional[int] = None
    deltaX: Optional[NumberType] = None
    deltaY: Optional[NumberType] = None
    deltaZ: Optional[NumberType] = None
    movementX: Optional[NumberType] = None
    movementY: Optional[NumberType] = None

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
