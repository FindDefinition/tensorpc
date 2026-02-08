import inspect
from typing import Callable, List, Optional, Tuple, Any, Union
from typing_extensions import Annotated
from tensorpc import compat
from tensorpc.core.dataclass_dispatch import dataclass
from typing_extensions import TypeAlias, get_type_hints
from dataclasses import Field, make_dataclass, field
import enum 

ValueType: TypeAlias = Union[int, float, str]
NumberType: TypeAlias = Union[int, float]
Vector2Type: TypeAlias = Tuple[float, float]
Vector3Type: TypeAlias = Tuple[float, float, float]

@dataclass
class BaseObject:
    alias: Optional[str] = None
    tooltip: Optional[str] = None

@dataclass
class CommonObject(BaseObject):
    default: Optional[Any] = None

@dataclass
class Enum(BaseObject):
    excludes: Optional[List[Any]] = None

@dataclass
class DynamicEnum(BaseObject):
    pass

@dataclass(kw_only=True)
class RangedInt(BaseObject):
    lo: int
    hi: int
    step: Optional[int] = None
    default: Optional[int] = None


@dataclass(kw_only=True)
class RangedFloat(BaseObject):
    lo: float
    hi: float
    step: Optional[float] = None
    default: Optional[float] = None


@dataclass
class ColorRGB(BaseObject):
    value_is_string: bool = True
    default: Optional[Union[int, str]] = None


@dataclass
class ColorRGBA(BaseObject):
    value_is_string: bool = True
    default: Optional[Union[int, str]] = None


@dataclass(kw_only=True)
class RangedVector3(BaseObject):
    lo: float
    hi: float
    step: Optional[float] = None
    default: Optional[Tuple[float, float, float]] = None

@dataclass(kw_only=True)
class RangedVector2(BaseObject):
    lo: float
    hi: float
    step: Optional[float] = None
    default: Optional[Tuple[float, float]] = None

@dataclass
class Vector3(BaseObject):
    step: Optional[float] = None
    default: Optional[Tuple[float, float, float]] = None

@dataclass
class Vector2(BaseObject):
    step: Optional[float] = None
    default: Optional[Tuple[float, float]] = None
