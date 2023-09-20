import inspect
from typing import Callable, List, Optional, Tuple, Any, Union
from typing_extensions import Annotated
from tensorpc.core.dataclass_dispatch import dataclass
from typing_extensions import TypeAlias, get_type_hints
from dataclasses import Field, make_dataclass, field

@dataclass
class CommonObject:
    alias: Optional[str] = None 
    default: Optional[Any] = None

@dataclass
class RangedInt:
    lo: int
    hi: int
    step: Optional[int] = None
    alias: Optional[str] = None 
    default: Optional[int] = None

@dataclass
class RangedFloat:
    lo: float
    hi: float
    step: Optional[float] = None
    alias: Optional[str] = None 
    default: Optional[float] = None

@dataclass
class ColorRGB:
    value_is_string: bool = True
    default: Optional[Union[int, str]] = None

@dataclass
class ColorRGBA:
    value_is_string: bool = True
    default: Optional[Union[int, str]] = None

@dataclass
class RangedVector3:
    lo: float
    hi: float
    step: Optional[float] = None
    alias: Optional[str] = None 
    default: Optional[Tuple[float, float, float]] = None

@dataclass
class Vector3:
    step: Optional[float] = None
    alias: Optional[str] = None 
    default: Optional[Tuple[float, float, float]] = None


Vector2Type: TypeAlias = Tuple[float, float]

Vector3Type: TypeAlias = Tuple[float, float, float]


def annotated_function_to_dataclass(func: Callable):
    annos = get_type_hints(func, include_extras=True)
    specs = inspect.signature(func)
    name_to_parameter = {p.name: p for p in specs.parameters.values()}
    fields: List[Tuple[str, Any, Field]] = []
    for name, anno in annos.items():
        param = name_to_parameter[name]
        assert param.default is not inspect.Parameter.empty, "annotated function arg must have default value"
        fields.append((name, anno, field(default=param.default)))
    return make_dataclass(func.__name__, fields)
