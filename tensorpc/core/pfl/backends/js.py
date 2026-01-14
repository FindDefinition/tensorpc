import dataclasses 
import math
import random
import struct
import time
from typing import Any, Callable, Optional, TypeVar, Union, overload
from ..core import PFLCompileFuncMeta, mark_meta_infer, mark_pfl_compilable, register_backend, PFLParseConfig
from ..pfl_reg import register_pfl_std
import numpy as np 
# implement all math func in javascript Math 
from typing_extensions import Self
from tensorpc.utils import perfetto_colors

register_backend("js", PFLParseConfig(
    allow_var_union=False,
    allow_kw=False,
    allow_nd_slice=False,
    allow_slice=True,
    allow_new_var_after_if=True,
    tuple_assign_must_be_homogeneous=True,
    allow_custom_class=False,
    allow_dynamic_container_literal=True,
    allow_partial_in_slice=True,
    allow_remove_optional_based_on_cond=False,
))

@register_pfl_std(mapped_name="len", backend="js", mapped=len)
def len_func(x: Any) -> int:
    return len(x)

@register_pfl_std(mapped_name="print", backend="js", mapped=print)
def print_func(*x: Any) -> None:
    return print(*x)

@register_pfl_std(mapped_name="int", backend="js", mapped=int)
def int_func(x: Any) -> int:
    return int(x)

@register_pfl_std(mapped_name="float", backend="js", mapped=float)
def float_func(x: Any) -> float:
    return float(x)

@register_pfl_std(mapped_name="bool", backend="js", mapped=bool)
def bool_func(x: Any) -> bool:
    return bool(x)

@register_pfl_std(mapped_name="str", backend="js", mapped=str)
def str_func(x: Any) -> str:
    return str(x)

@register_pfl_std(mapped_name="range", backend="js", mapped=range)
def range_func(start: int, stop: Optional[int] = None, step: Optional[int] = None) -> range:
    if stop is None and step is None:
        return range(start)
    elif step is None and stop is not None:
        return range(start, stop)
    else:
        assert stop is not None and step is not None, "stop and step must be provided together"
        return range(start, stop, step) 

_T_math = TypeVar("_T_math", int, float)

@register_pfl_std(mapped_name="Math", backend="js")
@dataclasses.dataclass
class Math:
    @staticmethod 
    def abs(x: float) -> float:
        return abs(x)

    @staticmethod 
    def acos(x: float) -> float:
        return math.acos(x)

    @staticmethod
    def asin(x: float) -> float:
        return math.asin(x)

    @staticmethod
    def atan(x: float) -> float:
        return math.atan(x)

    @staticmethod
    def atan2(y: float, x: float) -> float:
        return math.atan2(y, x)

    @staticmethod
    def ceil(x: float) -> float:
        return math.ceil(x)

    @staticmethod
    def cos(x: float) -> float:
        return math.cos(x)

    @staticmethod
    def exp(x: float) -> float:
        return math.exp(x)

    @staticmethod
    def floor(x: float) -> float:
        return math.floor(x)

    @staticmethod
    def log(x: float) -> float:
        return math.log(x)

    @staticmethod
    def max(*args: float) -> float:
        return max(args)

    @staticmethod
    def min(*args: float) -> float:
        return min(args)

    @staticmethod
    def pow(x: float, y: float) -> float:
        return math.pow(x, y)

    @staticmethod
    def random() -> float:
        return random.random()

    @staticmethod
    def round(x: float) -> float:
        return round(x)

    @staticmethod
    def sign(x: float) -> int:
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    @staticmethod
    def sin(x: float) -> float:
        return math.sin(x)

    @staticmethod
    def sqrt(x: float) -> float:
        return math.sqrt(x)

    @staticmethod
    def tan(x: float) -> float:
        return math.tan(x)

    @staticmethod
    def trunc(x: float) -> float:
        return math.trunc(x)

    @staticmethod
    def cbrt(x: float) -> float:
        return math.copysign(abs(x) ** (1/3), x)

    @staticmethod
    def clz32(x: int) -> int:
        return 32 - len(bin(x & 0xffffffff)[2:]) if x != 0 else 32

    @staticmethod
    def imul(a: int, b: int) -> int:
        return (a * b) & 0xffffffff if (a * b) >= 0 else ((a * b) + 0x100000000) & 0xffffffff

    @staticmethod
    def fround(x: float) -> float:
        return struct.unpack('f', struct.pack('f', x))[0]

    @staticmethod
    def log10(x: float) -> float:
        return math.log10(x)

    @staticmethod
    def log2(x: float) -> float:
        return math.log2(x)

    @staticmethod
    def log1p(x: float) -> float:
        return math.log1p(x)

    @staticmethod
    def expm1(x: float) -> float:
        return math.expm1(x)

    @staticmethod
    def hypot(*args: float) -> float:
        return math.hypot(*args)

    @staticmethod
    def sinh(x: float) -> float:
        return math.sinh(x)

    @staticmethod
    def cosh(x: float) -> float:
        return math.cosh(x)

    @staticmethod
    def tanh(x: float) -> float:
        return math.tanh(x)

    @staticmethod
    def asinh(x: float) -> float:
        return math.asinh(x)

    @staticmethod
    def acosh(x: float) -> float:
        return math.acosh(x)

    @staticmethod
    def atanh(x: float) -> float:
        return math.atanh(x)

    @staticmethod
    def to_degrees(x: float) -> float:
        return math.degrees(x)

    @staticmethod
    def to_radians(x: float) -> float:
        return math.radians(x)

    # Constants
    E: float = math.e
    LN10: float = math.log(10)
    LN2: float = math.log(2)
    LOG2E: float = math.log2(math.e)
    LOG10E: float = math.log10(math.e)
    PI: float = math.pi
    SQRT1_2: float = math.sqrt(0.5)
    SQRT2: float = math.sqrt(2)


@register_pfl_std(mapped_name="TypedArray", backend="js")
@dataclasses.dataclass
class TypedArray:
    def __getitem__(self, key: int) -> float: ...
    def __setitem__(self, key: int, val: float) -> None: ...

    @property 
    def length(self) -> int: ...

@register_pfl_std(mapped_name="BinpackResult", backend="js")
@dataclasses.dataclass
class BinpackResult:
    result: list[tuple[float, float]]
    width: float
    height: float 
    fill: float

@register_pfl_std(mapped_name="PerfUtil", backend="js", mapped=time)
@dataclasses.dataclass
class PerfUtil:
    @staticmethod
    def time() -> float: ...

@register_pfl_std(mapped_name="MathUtil", backend="js")
@dataclasses.dataclass
class MathUtil:
    @staticmethod
    def clamp(x: float, min_val: float, max_val: float) -> float:
        return max(min(x, max_val), min_val)

    @staticmethod
    def getTypedArray(x: np.ndarray) -> TypedArray: ...

    @staticmethod
    @overload
    def binpack(boxes: list[tuple[float, float]]) -> BinpackResult: ...

    @staticmethod
    @overload
    def binpack(boxes: list[tuple[float, float]], containerWidth: float) -> BinpackResult: ...

    @staticmethod
    def binpack(boxes: list[tuple[float, float]], containerWidth: Optional[float] = None) -> BinpackResult:
        raise NotImplementedError

@register_pfl_std(mapped_name="ColorUtil", backend="js")
@dataclasses.dataclass
class ColorUtil:
    @staticmethod
    def getPerfettoColorRGB(color: str) -> tuple[float, float, float]: 
        res = perfetto_colors.perfetto_string_to_color(color).base.rgb
        return res[0], res[1], res[2]

    @staticmethod
    def getPerfettoSliceColorRGB(color: str) -> tuple[float, float, float]:
        res = perfetto_colors.perfetto_slice_to_color(color).base.rgb
        return res[0], res[1], res[2]

    @staticmethod
    def getPerfettoVariantColorRGB(color: str) -> tuple[float, float, float]:
        res = perfetto_colors.perfetto_slice_to_color(color).variant.rgb
        return res[0], res[1], res[2]

    @staticmethod
    def getPerfettoVariantSliceColorRGB(color: str) -> tuple[float, float, float]:
        res = perfetto_colors.perfetto_slice_to_color(color).variant.rgb
        return res[0], res[1], res[2]

@register_pfl_std(mapped_name="Common", backend="js")
@dataclasses.dataclass
class Common:
    @staticmethod
    def getItemPath(obj: Any, attrs: Optional[list[Any]]) -> Any:
        if attrs is None:
            return None
        for attr in attrs:
            obj = obj[attr]
        return obj

@register_pfl_std(mapped_name="NdArray", mapped=np.ndarray, backend="js")
@dataclasses.dataclass
class NdArray:
    shape: list[int]
    dtype: int
    ndim: int
    def __getitem__(self, key: int) -> Self: ...
    def tolist(self) -> list[Any]: ...
    def size(self) -> int: ...
    def reshape(self, new_shape: list[int]) -> Self: ...

@register_pfl_std(mapped_name="Numpy", mapped=np, backend="js")
@dataclasses.dataclass
class Numpy:
    float32: int = 0
    float64: int = 4
    int8: int = 3
    int16: int = 2
    int32: int = 1
    int64: int = 8
    uint8: int = 6
    uint16: int = 9
    uint32: int = 10
    uint64: int = 11
    bool_: int = 5
    
    @staticmethod
    def array(data: list[Any]) -> np.ndarray: 
        return np.array(data)

    @staticmethod
    def zeros(shape: list[int], dtype: int) -> np.ndarray: 
        return np.zeros(shape, dtype=_JS_DTYPE_TO_NP[dtype])

    @staticmethod
    def ones(shape: list[int], dtype: int) -> np.ndarray: 
        return np.ones(shape, dtype=_JS_DTYPE_TO_NP[dtype])

    @staticmethod
    def empty(shape: list[int], dtype: int) -> np.ndarray: 
        return np.zeros(shape, dtype=_JS_DTYPE_TO_NP[dtype])

    @staticmethod
    def full(shape: list[int], val: Union[int, float], dtype: int) -> np.ndarray: 
        return np.full(shape, val, dtype=_JS_DTYPE_TO_NP[dtype])

    @staticmethod
    def zeros_like(x: np.ndarray) -> np.ndarray: 
        return np.zeros_like(x)

    @staticmethod
    def empty_like(x: np.ndarray) -> np.ndarray: 
        return np.empty_like(x)

_JS_DTYPE_TO_NP = {
    0: np.float32,
    4: np.float64,
    3: np.int8,
    2: np.int16,
    1: np.int32,
    8: np.int64,
    6: np.uint8,
    9: np.uint16,
    10: np.uint32,
    11: np.uint64,
    5: np.bool_,
}

T = TypeVar("T")

@overload
def mark_js_compilable(fn: T) -> T: ...

@overload
def mark_js_compilable(fn: None = None, *, is_template: bool = False, 
        always_inline: bool = False, meta: Optional[PFLCompileFuncMeta] = None) -> Callable[[T], T]: ...

@register_pfl_std(mapped_name="compiler_mark_js_compilable", backend=None, _internal_disable_type_check=True)
def mark_js_compilable(fn: Optional[Any] = None, *, is_template: bool = False, 
        always_inline: bool = False, meta: Optional[PFLCompileFuncMeta] = None) -> Union[Any, Callable[[Any], Any]]:
    return mark_pfl_compilable(fn, backends=["js"], is_template=is_template,
        always_inline=always_inline, meta=meta)

