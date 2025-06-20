from functools import partial
import inspect
import math

import triton
from tensorpc.apps.ppcl.tsim import get_tensorsim_context_checked
from tensorpc.apps.ppcl.tsim.core import get_tensorsim_context
from tensorpc.apps.ppcl.tsim.tensor import SimTensor
from tensorpc.core import pfl
from typing_extensions import Self
import numpy as np 
import dataclasses 
from typing import Annotated, Any, Callable, ClassVar, Optional, Type, TypeAlias, TypeVar, Union, cast, overload
from tensorpc.apps.ppcl import tsim 
from tensorpc.apps.ppcl.tsim import DTypeEnum
import triton.language as tl

pfl.register_backend("triton", pfl.PFLParseConfig(
    allow_var_union=False,
    allow_kw=True,
    allow_nd_slice=True,
    allow_slice=True,
    allow_new_var_after_if=True,
))

@pfl.register_pfl_std(mapped_name="int", backend="triton", mapped=int)
def int_func(x: Any) -> int:
    return int(x)

@pfl.register_pfl_std(mapped_name="float", backend="triton", mapped=float)
def float_func(x: Any) -> float:
    return float(x)

@pfl.register_pfl_std(mapped_name="bool", backend="triton", mapped=bool)
def bool_func(x: Any) -> bool:
    return bool(x)

@pfl.register_pfl_std(mapped_name="range", backend="triton", mapped=range)
def range_func(start: int, stop: Optional[int] = None, step: Optional[int] = None) -> range:
    if stop is None and step is None:
        return range(start)
    elif step is None and stop is not None:
        return range(start, stop)
    else:
        assert stop is not None and step is not None, "stop and step must be provided together"
        return range(start, stop, step) 

@pfl.register_pfl_std(mapped_name="print", backend="triton", mapped=print)
def print_func(*args: Any) -> None:
    print(*args)

@pfl.register_pfl_std(mapped_name="TritonConstexpr", backend="triton", mapped=tl.constexpr)
@dataclasses.dataclass
class ConstExpr:
    pass

@pfl.register_pfl_std(mapped_name="TritonBlockTensor", backend="triton")
@dataclasses.dataclass
class Tensor:
    _wrapped: tsim.SimTensor
    def _replace_wrapped(self, new_tensor: tsim.SimTensor) -> Self:
        return dataclasses.replace(self, _wrapped=new_tensor)
    def __repr__(self) -> str:
        return self._wrapped.__repr__()
    @property 
    def shape(self) -> list[int]:
        return self._wrapped.shape 
    @property 
    def dtype(self) -> int:
        return self._wrapped.dtype 
    @property 
    def ndim(self) -> int: 
        return len(self.shape)
    @property 
    def T(self) -> Self: 
        return self._replace_wrapped(self._wrapped.T)
        
    def __getitem__(self, key: Any) -> Self:
        return self._replace_wrapped(self._wrapped[key])

    def to(self, dtype: int) -> Self:
        return self._replace_wrapped(self._wrapped.to(dtype))

    @staticmethod 
    def _binary_infer(fn: Callable, this: pfl.PFLExprInfo, other: pfl.PFLExprInfo) -> pfl.PFLMetaInferResult:
        assert this.has_metadata(Tensor, PointerTensor)
        if other.has_metadata(Tensor, PointerTensor):
            # metadata is Tensor or PointerTensor
            return pfl.PFLMetaInferResult(fn(this.metadata_checked, other.metadata_checked))
        assert other.type == pfl.PFLExprType.NUMBER 
        # when binary operation with Tensor, result won't be constexpr, so we create a new dummy number.
        return pfl.PFLMetaInferResult(fn(this.metadata_checked, other.get_origin_type_checked()(1)))

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __add__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped + other._wrapped)
        return self._replace_wrapped(self._wrapped + other)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __iadd__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            self._wrapped += other._wrapped
            return self
        self._wrapped += other
        return self

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __radd__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(other._wrapped + self._wrapped)
        return self._replace_wrapped(other + self._wrapped)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __sub__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped - other._wrapped)
        return self._replace_wrapped(self._wrapped - other)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __isub__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            self._wrapped -= other._wrapped
            return self
        self._wrapped -= other
        return self

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __rsub__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(other._wrapped - self._wrapped)
        return self._replace_wrapped(other - self._wrapped)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __mul__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped * other._wrapped)
        return self._replace_wrapped(self._wrapped * other)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __imul__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            self._wrapped *= other._wrapped
            return self
        self._wrapped *= other
        return self

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __rmul__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(other._wrapped * self._wrapped)
        return self._replace_wrapped(other * self._wrapped)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __truediv__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped / other._wrapped)
        return self._replace_wrapped(self._wrapped / other)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __rtruediv__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(other._wrapped / self._wrapped)
        return self._replace_wrapped(other / self._wrapped)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __itruediv__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            self._wrapped /= other._wrapped
            return self
        self._wrapped /= other
        return self

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __floordiv__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped // other._wrapped)
        return self._replace_wrapped(self._wrapped // other)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __rfloordiv__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(other._wrapped // self._wrapped)
        return self._replace_wrapped(other // self._wrapped)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __ifloordiv__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            self._wrapped //= other._wrapped
            return self
        self._wrapped //= other
        return self

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __mod__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped % other._wrapped)
        return self._replace_wrapped(self._wrapped % other)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __rmod__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(other._wrapped % self._wrapped)
        return self._replace_wrapped(other % self._wrapped)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __imod__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            self._wrapped %= other._wrapped
            return self
        self._wrapped %= other
        return self

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __and__(self, other: Self) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped & other._wrapped)
        return self._replace_wrapped(self._wrapped & other)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __rand__(self, other: Self) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(other._wrapped & self._wrapped)
        return self._replace_wrapped(other & self._wrapped)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __iand__(self, other: Self) -> Self:
        if isinstance(other, Tensor):
            self._wrapped &= other._wrapped
            return self
        self._wrapped &= other
        return self
    
    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __or__(self, other: Self) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped | other._wrapped)
        return self._replace_wrapped(self._wrapped | other)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __ror__(self, other: Self) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(other._wrapped | self._wrapped)
        return self._replace_wrapped(other | self._wrapped)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __ior__(self, other: Self) -> Self:
        if isinstance(other, Tensor):
            self._wrapped |= other._wrapped
            return self
        self._wrapped |= other
        return self

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __xor__(self, other: Self) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped ^ other._wrapped)
        return self._replace_wrapped(self._wrapped ^ other)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __rxor__(self, other: Self) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(other._wrapped ^ self._wrapped)
        return self._replace_wrapped(other ^ self._wrapped)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __ixor__(self, other: Self) -> Self:
        if isinstance(other, Tensor):
            self._wrapped ^= other._wrapped
            return self
        self._wrapped ^= other
        return self

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __lshift__(self, other: Union[Self, int]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped << other._wrapped)
        return self._replace_wrapped(self._wrapped << other)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __rlshift__(self, other: Union[Self, int]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(other._wrapped << self._wrapped)
        return self._replace_wrapped(other << self._wrapped)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __ilshift__(self, other: Union[Self, int]) -> Self:
        if isinstance(other, Tensor):
            self._wrapped <<= other._wrapped
            return self
        self._wrapped <<= other
        return self

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __rshift__(self, other: Union[Self, int]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped >> other._wrapped)
        return self._replace_wrapped(self._wrapped >> other)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __rrshift__(self, other: Union[Self, int]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(other._wrapped >> self._wrapped)
        return self._replace_wrapped(other >> self._wrapped)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __irshift__(self, other: Union[Self, int]) -> Self:
        if isinstance(other, Tensor):
            self._wrapped >>= other._wrapped
            return self
        self._wrapped >>= other
        return self

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __lpow__(self, other: Union[Self, int]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped ** other._wrapped)
        return self._replace_wrapped(self._wrapped ** other)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __rpow__(self, other: Union[Self, int]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(other._wrapped ** self._wrapped)
        return self._replace_wrapped(other ** self._wrapped)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __ipow__(self, other: Union[Self, int]) -> Self:
        if isinstance(other, Tensor):
            self._wrapped **= other._wrapped
            return self
        self._wrapped **= other
        return self

    def __neg__(self) -> Self:
        return self._replace_wrapped(-self._wrapped)

    def __pos__(self) -> Self:
        return self._replace_wrapped(+self._wrapped)

    def __invert__(self) -> Self:
        return self._replace_wrapped(~self._wrapped)

    def __lt__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped < other._wrapped)
        return self._replace_wrapped(self._wrapped < other)


    def __le__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped <= other._wrapped)
        return self._replace_wrapped(self._wrapped <= other)
    def __ge__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped >= other._wrapped)
        return self._replace_wrapped(self._wrapped >= other)
    def __gt__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped > other._wrapped)
        return self._replace_wrapped(self._wrapped > other)

@pfl.register_pfl_std(mapped_name="TritonPointerTensor", backend="triton")
@dataclasses.dataclass
class PointerTensor:
    _wrapped: tsim.SimPointerTensor
    def _replace_wrapped(self, new_tensor: tsim.SimPointerTensor) -> Self:
        return dataclasses.replace(self, _wrapped=new_tensor)
    def __repr__(self) -> str:
        return self._wrapped.__repr__()
    @property 
    def shape(self) -> list[int]:
        return self._wrapped.shape 
    @property 
    def dtype(self) -> int:
        return self._wrapped.dtype 
    @property 
    def ndim(self) -> int: 
        return len(self.shape)
    @property 
    def T(self) -> Self:
        return self._replace_wrapped(self._wrapped.T)
    def __getitem__(self, key: Any) -> Self:
        return self._replace_wrapped(self._wrapped[key])

    # pointer tensor only support add and sub operations
    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __add__(self, other: Union[Tensor, int]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped + other._wrapped)
        return self._replace_wrapped(self._wrapped + other)

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __iadd__(self, other: Union[Tensor, int]) -> Self:
        if isinstance(other, Tensor):
            self._wrapped += other._wrapped
            return self
        self._wrapped += other
        return self

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __radd__(self, other: Union[Tensor, int]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(other._wrapped + self._wrapped)
        return self._replace_wrapped(other + self._wrapped)

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __sub__(self, other: Union[Tensor, int]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped - other._wrapped)
        return self._replace_wrapped(self._wrapped - other)

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __isub__(self, other: Union[Tensor, int]) -> Self:
        if isinstance(other, Tensor):
            self._wrapped -= other._wrapped
            return self
        self._wrapped -= other
        return self

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __rsub__(self, other: Union[Tensor, int]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(other._wrapped - self._wrapped)

        return self._replace_wrapped(other - self._wrapped)

@pfl.register_pfl_std(mapped_name="TritonPointerScalarFloat", backend="triton")
@dataclasses.dataclass
class PointerScalarFloat:
    _wrapped: tsim.SimPointerScalarFloat
    def _replace_wrapped(self, new_tensor: tsim.SimPointerScalarFloat) -> Self:
        return dataclasses.replace(self, _wrapped=new_tensor)
    def __repr__(self) -> str:
        return self._wrapped.__repr__()
    @property 
    def shape(self) -> list[int]:
        return self._wrapped.shape 
    @property 
    def dtype(self) -> int:
        return self._wrapped.dtype 
    @property 
    def ndim(self) -> int: 
        return len(self.shape)
    @property 
    def T(self) -> Self:
        return self._replace_wrapped(self._wrapped.T)
    def __getitem__(self, key: Any) -> Self:
        return self._replace_wrapped(self._wrapped[key])

    # pointer tensor only support add and sub operations
    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __add__(self, other: Union[Tensor, int]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped + other._wrapped)
        return self._replace_wrapped(self._wrapped + other)

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __iadd__(self, other: Union[Tensor, int]) -> Self:
        if isinstance(other, Tensor):
            self._wrapped += other._wrapped
            return self
        self._wrapped += other
        return self

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __radd__(self, other: Union[Tensor, int]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(other._wrapped + self._wrapped)
        return self._replace_wrapped(other + self._wrapped)

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __sub__(self, other: Union[Tensor, int]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped - other._wrapped)
        return self._replace_wrapped(self._wrapped - other)

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __isub__(self, other: Union[Tensor, int]) -> Self:
        if isinstance(other, Tensor):
            self._wrapped -= other._wrapped
            return self
        self._wrapped -= other
        return self

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __rsub__(self, other: Union[Tensor, int]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(other._wrapped - self._wrapped)

        return self._replace_wrapped(other - self._wrapped)

@pfl.register_pfl_std(mapped_name="TritonPointerScalarInt", backend="triton")
@dataclasses.dataclass
class PointerScalarInt:
    _wrapped: tsim.SimPointerScalarInt
    def _replace_wrapped(self, new_tensor: tsim.SimPointerScalarInt) -> Self:
        return dataclasses.replace(self, _wrapped=new_tensor)
    def __repr__(self) -> str:
        return self._wrapped.__repr__()
    @property 
    def shape(self) -> list[int]:
        return self._wrapped.shape 
    @property 
    def dtype(self) -> int:
        return self._wrapped.dtype 
    @property 
    def ndim(self) -> int: 
        return len(self.shape)
    @property 
    def T(self) -> Self:
        return self._replace_wrapped(self._wrapped.T)
    def __getitem__(self, key: Any) -> Self:
        return self._replace_wrapped(self._wrapped[key])

    # pointer tensor only support add and sub operations
    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __add__(self, other: Union[Tensor, int]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped + other._wrapped)
        return self._replace_wrapped(self._wrapped + other)

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __iadd__(self, other: Union[Tensor, int]) -> Self:
        if isinstance(other, Tensor):
            self._wrapped += other._wrapped
            return self
        self._wrapped += other
        return self

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __radd__(self, other: Union[Tensor, int]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(other._wrapped + self._wrapped)
        return self._replace_wrapped(other + self._wrapped)

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __sub__(self, other: Union[Tensor, int]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped - other._wrapped)
        return self._replace_wrapped(self._wrapped - other)

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __isub__(self, other: Union[Tensor, int]) -> Self:
        if isinstance(other, Tensor):
            self._wrapped -= other._wrapped
            return self
        self._wrapped -= other
        return self

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __rsub__(self, other: Union[Tensor, int]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(other._wrapped - self._wrapped)

        return self._replace_wrapped(other - self._wrapped)

def _none_infer(fn: Callable, *args, **kwargs) -> Optional[pfl.PFLMetaInferResult]:
    return None 
    
def _global_unary_infer(fn: Callable, x: pfl.PFLExprInfo) -> Optional[pfl.PFLMetaInferResult]:
    if x.type == pfl.PFLExprType.NUMBER:
        if x.has_metadata():
            return pfl.PFLMetaInferResult(fn(x))
        return None
    return pfl.PFLMetaInferResult(fn(x.metadata_checked))

def _global_binary_infer(fn: Callable, this: pfl.PFLExprInfo, other: pfl.PFLExprInfo) -> Optional[pfl.PFLMetaInferResult]:
    # binary only do const eval if all operands are number and have metadata.
    if this.type == pfl.PFLExprType.NUMBER and other.type == pfl.PFLExprType.NUMBER:
        # only calc metadata (consteval) if all operands have metadata (const value)
        if this.has_metadata() and other.has_metadata():
            return pfl.PFLMetaInferResult(fn(this.metadata_checked, other.metadata_checked))
        return None
    if not this.has_metadata(Tensor) and other.has_metadata(Tensor):
        assert this.type == pfl.PFLExprType.NUMBER 
        return pfl.PFLMetaInferResult(fn(other.get_origin_type_checked()(1), other.metadata_checked))
    elif this.has_metadata(Tensor) and not other.has_metadata(Tensor):
        assert other.type == pfl.PFLExprType.NUMBER 
        return pfl.PFLMetaInferResult(fn(this.metadata_checked, other.get_origin_type_checked()(1)))
    elif this.has_metadata(Tensor) and other.has_metadata(Tensor):
        return pfl.PFLMetaInferResult(fn(this.metadata_checked, other.metadata_checked))
    else:
        raise NotImplementedError(f"Unsupported binary operation between {this.type} and {other.type}")

@overload
def min_fn(x: int, y: int) -> int: ...
@overload
def min_fn(x: float, y: float) -> float: ...
@overload
def min_fn(x: Tensor, y: Union[int, float]) -> Tensor: ...
@overload
def min_fn(x: Union[int, float], y: Tensor) -> Tensor: ...

@pfl.register_pfl_std(mapped_name="min", backend="triton", mapped=min)
@pfl.configure_std_func(meta_infer=_global_binary_infer)
def min_fn(x: Union[Tensor, int, float], y: Union[Tensor, int, float]) -> Union[Tensor, int, float]:
    if not isinstance(x, Tensor) and not isinstance(y, Tensor):
        return min(x, y)
    x_wrapped = x._wrapped if isinstance(x, Tensor) else x
    y_wrapped = y._wrapped if isinstance(y, Tensor) else y
    return Tensor(tsim.minimum(x_wrapped, y_wrapped))

@overload
def max_fn(x: int, y: int) -> int: ...
@overload
def max_fn(x: float, y: float) -> float: ...
@overload
def max_fn(x: Tensor, y: Union[int, float]) -> Tensor: ...
@overload
def max_fn(x: Union[int, float], y: Tensor) -> Tensor: ...

@pfl.register_pfl_std(mapped_name="max", backend="triton", mapped=max)
@pfl.configure_std_func(meta_infer=_global_binary_infer)
def max_fn(x: Union[Tensor, int, float], y: Union[Tensor, int, float]) -> Union[Tensor, int, float]:
    if not isinstance(x, Tensor) and not isinstance(y, Tensor):
        return max(x, y)
    x_wrapped = x._wrapped if isinstance(x, Tensor) else x
    y_wrapped = y._wrapped if isinstance(y, Tensor) else y
    return Tensor(tsim.minimum(x_wrapped, y_wrapped))

_T_math = TypeVar("_T_math", Tensor, int, float)
_T_math_fp = TypeVar("_T_math_fp", Tensor, float)

@pfl.register_pfl_std(mapped_name="tl", backend="triton", mapped=tl)
@dataclasses.dataclass
class triton_std:
    # # subtypes using ClassVar
    # TensorX: TypeAlias = Tensor
    # PointerTensor: ClassVar[Type["PointerTensor"]] = PointerTensor


    float32: DTypeEnum = DTypeEnum.float32
    float64: DTypeEnum = DTypeEnum.float64
    int8: DTypeEnum = DTypeEnum.int8
    int16: DTypeEnum = DTypeEnum.int16
    int32: DTypeEnum = DTypeEnum.int32
    int64: DTypeEnum = DTypeEnum.int64
    uint8: DTypeEnum = DTypeEnum.uint8
    uint16: DTypeEnum = DTypeEnum.uint16
    uint32: DTypeEnum = DTypeEnum.uint32
    uint64: DTypeEnum = DTypeEnum.uint64
    bool_: DTypeEnum = DTypeEnum.bool_
    float16: DTypeEnum = DTypeEnum.float16
    bfloat16 : DTypeEnum = DTypeEnum.bfloat16

    float8e5 : DTypeEnum = DTypeEnum.float8e5
    float8e5b16 : DTypeEnum = DTypeEnum.float8e5b16
    float8e4nv : DTypeEnum = DTypeEnum.float8e4nv
    float8e4b8 : DTypeEnum = DTypeEnum.float8e4b8
    float8e4b15 : DTypeEnum = DTypeEnum.float8e4b15

    @staticmethod
    def _program_id_infer(fn: Callable, axis: pfl.PFLExprInfo) -> Optional[pfl.PFLMetaInferResult]:
        return None

    @staticmethod
    @pfl.configure_std_func(meta_infer=_program_id_infer)
    def program_id(axis: int) -> int: 
        ctx = get_tensorsim_context_checked()
        return ctx.grid_id[axis]

    # @staticmethod
    # def _compiler_hints_infer(fn: Callable, x: pfl.PFLExprInfo, y: pfl.PFLExprInfo) -> Optional[pfl.PFLMetaInferResult]:
    #     return pfl.PFLMetaInferResult(x.metadata)

    @staticmethod
    # @pfl.configure_std_func(meta_infer=_compiler_hints_infer)
    def multiple_of(x: _T_math, y: Any) -> _T_math: 
        return x

    @staticmethod
    @pfl.configure_std_func(meta_infer=_program_id_infer)
    def num_programs(axis: int) -> int: 
        ctx = get_tensorsim_context_checked()
        return ctx.grid_size[axis]

    @staticmethod
    def zeros(shape: list[int], dtype: int) -> Tensor: 
        return Tensor(tsim.zeros(shape, dtype))

    @staticmethod
    def ones(shape: list[int], dtype: int) -> Tensor:
        return Tensor(tsim.ones(shape, dtype))

    @staticmethod
    def arange(start: int, end: int) -> Tensor:
        return Tensor(tsim.arange(start, end))

    @staticmethod 
    def _load_infer(fn: Callable, pointer: pfl.PFLExprInfo, mask: Optional[pfl.PFLExprInfo] = None, other: Optional[pfl.PFLExprInfo] = None) -> Optional[pfl.PFLMetaInferResult]:
        if other is None:
            other_value = None 
        else:
            if not other.has_metadata(Tensor):
                if other.type == pfl.PFLExprType.NUMBER:
                    other_value = other.get_origin_type_checked()(1)
                else:
                    raise NotImplementedError(f"Unsupported type {other.type} for load operation")
            else:
                other_value = other.metadata_checked
        if mask is None:
            mask_value = None 
        else:
            mask_value = mask.metadata_checked if mask.has_metadata(Tensor) else None
        return pfl.PFLMetaInferResult(fn(pointer.metadata_checked, mask_value, other_value))
    
    @staticmethod
    @overload
    def load(pointer: PointerScalarFloat, mask: Optional[bool] = None, other: Optional[Union[int, float]] = None) -> float: ...
    @staticmethod
    @overload
    def load(pointer: PointerScalarInt, mask: Optional[bool] = None, other: Optional[Union[int, float]] = None) -> int: ...
    @staticmethod
    @overload
    def load(pointer: PointerTensor, mask: Optional[Tensor] = None, other: Optional[Union[int, float, Tensor]] = None) -> Tensor: ...

    @staticmethod
    @pfl.configure_std_func(meta_infer=_load_infer)
    def load(pointer: Union[PointerTensor, PointerScalarFloat, PointerScalarInt], mask: Optional[Union[Tensor, bool]] = None, other: Optional[Union[int, float, Tensor]] = None) -> Union[Tensor, int, float]:
        mask_wrapped = mask._wrapped if isinstance(mask, Tensor) else mask 
        other_wrapped = other._wrapped if isinstance(other, Tensor) else other
        if isinstance(pointer, PointerScalarFloat):
            assert isinstance(mask_wrapped, bool)
            assert not isinstance(other_wrapped, tsim.SimTensor)
            return pointer._wrapped.load(mask_wrapped, other_wrapped)
        elif isinstance(pointer, PointerScalarInt):
            assert isinstance(mask_wrapped, bool)
            assert not isinstance(other_wrapped, tsim.SimTensor)
            return pointer._wrapped.load(mask_wrapped, other_wrapped)
        return Tensor(pointer._wrapped.load(mask_wrapped, other_wrapped))

    @staticmethod
    @overload
    def store(pointer: PointerScalarFloat, value: float, mask: Optional[bool] = None) -> None: ...
    @staticmethod
    @overload
    def store(pointer: PointerScalarInt, value: int, mask: Optional[bool] = None) -> None: ...
    @staticmethod
    @overload
    def store(pointer: PointerTensor, value: Tensor, mask: Optional[Tensor] = None) -> None: ...

    @staticmethod
    @pfl.configure_std_func(meta_infer=_none_infer)
    def store(pointer: Union[PointerTensor, PointerScalarFloat, PointerScalarInt], value: Union[int, float, Tensor], mask: Optional[Union[Tensor, bool]] = None) -> None:
        mask_wrapped = mask._wrapped if isinstance(mask, Tensor) else mask 
        value_wrapped = value._wrapped if isinstance(value, Tensor) else value
        if isinstance(pointer, PointerScalarFloat):
            assert isinstance(mask_wrapped, bool)
            assert not isinstance(value_wrapped, tsim.SimTensor)
            return pointer._wrapped.store(value_wrapped, mask_wrapped)
        elif isinstance(pointer, PointerScalarInt):
            assert isinstance(mask_wrapped, bool)
            assert not isinstance(value_wrapped, tsim.SimTensor)
            return pointer._wrapped.store(value_wrapped, mask_wrapped)
        pointer._wrapped.store(value_wrapped, mask_wrapped)

    @staticmethod
    def range(start: int, stop: Optional[int] = None, step: Optional[int] = None, num_stages: Optional[int] = None) -> range:
        if stop is not None:
            if step is not None:
                return range(start, stop, step)
            return range(start, stop)
        return range(start)
        mask_wrapped = mask._wrapped if mask else None 
        pointer._wrapped.store(value._wrapped, mask_wrapped)

    @staticmethod
    @overload
    def cdiv(x: Tensor, div: Union[Tensor, int, float]) -> Tensor: ...
    
    @staticmethod
    @overload
    def cdiv(x: int, div: int) -> int: ...

    @staticmethod
    @overload
    def cdiv(x: float, div: float) -> float: ...

    @staticmethod
    @pfl.configure_std_func(meta_infer=_global_binary_infer)
    def cdiv(x: Union[Tensor, int, float], div: Union[Tensor, int, float]) -> Union[int, float, Tensor]: 
        x_wrapped = x._wrapped if isinstance(x, Tensor) else x 
        div_wrapped = div._wrapped if isinstance(div, Tensor) else div 
        res = (x_wrapped + div_wrapped - 1) // div_wrapped
        if isinstance(res, tsim.SimTensor):
            return Tensor(res)
        return res

    @staticmethod 
    def _dot_infer(fn: Callable, x: pfl.PFLExprInfo, y: pfl.PFLExprInfo, acc: Optional[pfl.PFLExprInfo] = None, **kwargs_dontcare) -> Optional[pfl.PFLMetaInferResult]:
        if acc is None:
            acc_value = None 
        else:
            acc_value = acc.metadata_checked
        return pfl.PFLMetaInferResult(fn(x.metadata_checked, y.metadata_checked, acc_value))

    @staticmethod
    @pfl.configure_std_func(meta_infer=_dot_infer)
    def dot(x: Tensor, y: Tensor, acc: Optional[Tensor] = None, allow_tf32: bool = True) -> Tensor:
        res_wrapped = x._wrapped @ y._wrapped
        if acc is not None:
            acc._wrapped[:] += res_wrapped
            return acc
        return Tensor(res_wrapped)
    
    @staticmethod
    def argmax(input: Tensor, axis: int, tie_break_left: bool = True, keep_dims: bool = False) -> Tensor:
        assert isinstance(axis, int)
        return Tensor(input._wrapped.argmax(axis, keep_dims))

    @staticmethod
    def argmin(input: Tensor, axis: int, tie_break_left: bool = True, keep_dims: bool = False) -> Tensor:
        assert isinstance(axis, int)
        return Tensor(input._wrapped.argmin(axis, keep_dims))

    @staticmethod
    def max(input: Tensor, axis: int, keep_dims: bool = False) -> Tensor:
        assert isinstance(axis, int)
        return Tensor(input._wrapped.max(axis, keep_dims))

    @staticmethod
    def min(input: Tensor, axis: int, keep_dims: bool = False) -> Tensor:
        assert isinstance(axis, int)
        return Tensor(input._wrapped.min(axis, keep_dims))


    # @staticmethod
    # def max_with_indices(input: Tensor, axis: int, return_indices_tie_break_left: bool = True, keep_dims: bool = False) -> Tensor: ...

    # @staticmethod
    # def min_with_indices(input: Tensor, axis: int, return_indices_tie_break_left: bool = True, keep_dims: bool = False) -> Tensor: ...

    @staticmethod
    def sum(input: Tensor, axis: int, keep_dims: bool = False, dtype: Optional[int] = None) -> Tensor:
        assert isinstance(axis, int)
        return Tensor(input._wrapped.sum(axis, keep_dims))

    @staticmethod
    def where(cond: Tensor, x: Union[Tensor, int, float], y: Union[Tensor, int, float]) -> Tensor: 
        x_wrapped = x._wrapped if isinstance(x, Tensor) else x 
        y_wrapped = y._wrapped if isinstance(y, Tensor) else y
        return Tensor(tsim.where(cond._wrapped, x_wrapped, y_wrapped))

    @staticmethod
    @pfl.configure_std_func(meta_infer=_global_unary_infer)
    def abs(x: _T_math) -> _T_math:
        if isinstance(x, Tensor):
            return Tensor(tsim.abs(x._wrapped))
        else:
            res = abs(x)
            if isinstance(x, int):
                return int(res)
            else:
                return res

    @staticmethod
    @pfl.configure_std_func(meta_infer=_global_unary_infer)
    def ceil(x: _T_math_fp) -> _T_math_fp:
        if isinstance(x, Tensor):
            return Tensor(tsim.ceil(x._wrapped))
        else:
            return math.ceil(x)

    @staticmethod
    @pfl.configure_std_func(meta_infer=_global_unary_infer)
    def floor(x: _T_math_fp) -> _T_math_fp:
        if isinstance(x, Tensor):
            return Tensor(tsim.floor(x._wrapped))
        else:
            return math.floor(x)

    @staticmethod
    @pfl.configure_std_func(meta_infer=_global_unary_infer)
    def cos(x: _T_math_fp) -> _T_math_fp:
        if isinstance(x, Tensor):
            return Tensor(tsim.cos(x._wrapped))
        else:
            return math.cos(x)

    @staticmethod
    @pfl.configure_std_func(meta_infer=_global_unary_infer)
    def sin(x: _T_math_fp) -> _T_math_fp:
        if isinstance(x, Tensor):
            return Tensor(tsim.sin(x._wrapped))
        else:
            return math.sin(x)

    # @staticmethod
    # def div_rn(x: Union[Tensor, int, float], y: Union[Tensor, int, float]) -> Tensor: ...

    # @staticmethod
    # def fdiv(x: Union[Tensor, int, float], y: Union[Tensor, int, float]) -> Tensor: ...

    # @staticmethod
    # def fma(x: Union[Tensor, int, float], y: Union[Tensor, int, float], z: Union[Tensor, int, float]) -> Tensor: ...

    @staticmethod
    @pfl.configure_std_func(meta_infer=_global_unary_infer)
    def exp(x: _T_math_fp) -> _T_math_fp:
        if isinstance(x, Tensor):
            return Tensor(tsim.exp(x._wrapped))
        else:
            return math.exp(x)

    @staticmethod
    @pfl.configure_std_func(meta_infer=_global_unary_infer)
    def exp2(x: _T_math_fp) -> _T_math_fp:
        if isinstance(x, Tensor):
            return Tensor(tsim.exp2(x._wrapped))
        else:
            return math.exp2(x)


    @staticmethod
    @pfl.configure_std_func(meta_infer=_global_unary_infer)
    def log(x: _T_math_fp) -> _T_math_fp:
        if isinstance(x, Tensor):
            return Tensor(tsim.log(x._wrapped))
        else:
            return math.log(x)

    @staticmethod
    @pfl.configure_std_func(meta_infer=_global_unary_infer)
    def log2(x: _T_math_fp) -> _T_math_fp:
        if isinstance(x, Tensor):
            return Tensor(tsim.log2(x._wrapped))
        else:
            return math.log2(x)

    @staticmethod
    @pfl.configure_std_func(meta_infer=_global_unary_infer)
    def rsqrt(x: _T_math_fp) -> _T_math_fp: 
        if isinstance(x, Tensor):
            return Tensor(tsim.rsqrt(x._wrapped))
        else:
            return 1.0 / math.sqrt(x)

    @staticmethod
    @pfl.configure_std_func(meta_infer=_global_unary_infer)
    def sqrt(x: _T_math_fp) -> _T_math_fp:
        if isinstance(x, Tensor):
            return Tensor(tsim.sqrt(x._wrapped))
        else:
            return math.sqrt(x)

    # @staticmethod
    # def sqrt_rn(x: Union[Tensor, int, float]) -> Tensor: ...

    @staticmethod
    @pfl.configure_std_func(meta_infer=_global_unary_infer)
    def sigmoid(x: _T_math_fp) -> _T_math_fp:
        if isinstance(x, Tensor):
            return Tensor(tsim.sigmoid(x._wrapped))
        else:
            return math.sqrt(x)


    @staticmethod
    def softmax(x: Tensor, axis: Optional[int] = None) -> Tensor:
        return Tensor(tsim.softmax(x._wrapped, axis))


    @staticmethod
    @overload
    def _internal_binary_anno(x: Tensor, y: float) -> Tensor: ...

    @staticmethod
    @overload
    def _internal_binary_anno(x: float, y: Tensor) -> Tensor: ...

    @staticmethod
    @overload
    def _internal_binary_anno(x: float, y: float) -> float: ...

    @staticmethod
    @overload
    def _internal_binary_anno(x: Tensor, y: Tensor) -> Tensor: ...

    @staticmethod
    def _internal_binary_anno(x: Union[Tensor, float], y: Union[Tensor, float]) -> Union[Tensor, float]:
        raise NotImplementedError("shouldn't be used directly.")

    @staticmethod
    @pfl.configure_std_func(take_overloads_fn=_internal_binary_anno, meta_infer=_global_binary_infer)
    def maximum(x: Union[Tensor, float], y: Union[Tensor, float]) -> Tensor:
        x_wrapped = x._wrapped if isinstance(x, Tensor) else x 
        y_wrapped = y._wrapped if isinstance(y, Tensor) else y
        res_wrapped = tsim.maximum(x_wrapped, y_wrapped)
        if isinstance(res_wrapped, SimTensor):
            return Tensor(res_wrapped)
        else:
            return res_wrapped

    @staticmethod
    @pfl.configure_std_func(take_overloads_fn=_internal_binary_anno, meta_infer=_global_binary_infer)
    def minimum(x: Union[Tensor, float], y: Union[Tensor, float]) -> Tensor:
        x_wrapped = x._wrapped if isinstance(x, Tensor) else x 
        y_wrapped = y._wrapped if isinstance(y, Tensor) else y
        res_wrapped = tsim.minimum(x_wrapped, y_wrapped)
        if isinstance(res_wrapped, SimTensor):
            return Tensor(res_wrapped)
        else:
            return res_wrapped

T = TypeVar("T")


def _triton_anno_transform(inferred: pfl.PFLExprInfo, anno_in_ast: Any) -> pfl.PFLExprInfo:
    print(anno_in_ast, anno_in_ast is tl.constexpr, inferred)
    if anno_in_ast is tl.constexpr:
        inferred = dataclasses.replace(inferred)
        inferred.anno_metadatas_ext.append(ConstExpr())
    return inferred

@dataclasses.dataclass 
class TritonSimInfo:
    grid_size: tuple[int, int, int]
    ref_results: dict[str, Any]

def _validate_and_convert_triton_kwargs(kwargs: dict[str, Any], meta_only: bool) -> dict[str, Any]:
    new_kwargs: dict[str, Any] = {}
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            # check v is contiguous
            v_dtype_tsim = tsim.NP_DTYPE_TO_PPCL[v.dtype.type]
            if meta_only:
                v_ptr = PointerTensor(tsim.create_pointer_tensor_meta(v_dtype_tsim, []))
            else:
                mem = tsim.create_sim_memory(k, v)
                v_ptr = PointerTensor(tsim.create_pointer_tensor(v_dtype_tsim, 0, mem))
            new_kwargs[k] = v_ptr 
        elif isinstance(v, PointerTensor):
            if meta_only:
                new_kwargs[k] = dataclasses.replace(v, _wrapped=v._wrapped.to_meta_tensor())
            else:
                new_kwargs[k] = v
        else:
            # TODO add support for string constexpr
            assert isinstance(v, (int, float, bool))
            new_kwargs[k] = v
    
    return new_kwargs

def _handle_triton_inline_data(inline_run_env_fn: Callable[[], pfl.PFLInlineRunEnv]):
    env = inline_run_env_fn()
    # convert test data to triton sim
    new_kwargs: dict[str, Any] = _validate_and_convert_triton_kwargs(env.kwargs, False)
    env.kwargs = new_kwargs 
    return env 

@overload
def mark_triton_compilable(fn: T) -> T: ...

@overload
def mark_triton_compilable(fn: None = None, *, inline_run_env_fn: Optional[Callable[[], pfl.PFLInlineRunEnv]] = None, is_template: bool = False) -> Callable[[T], T]: ...

@pfl.register_pfl_std(mapped_name="triton_compiler_mark_pfl_compilable", backend=None, _internal_disable_type_check=True)
def mark_triton_compilable(fn: Optional[T] = None, *, inline_run_env_fn: Optional[Callable[[], pfl.PFLInlineRunEnv]] = None, 
                is_template: bool = False) -> Union[T, Callable[[T], T]]:
    def wrapper(fn_wrapped: T) -> T:
        prev_meta: Optional[pfl.PFLCompileFuncMeta] = getattr(fn_wrapped, pfl.PFL_COMPILE_META_ATTR, None)
        inline_run_env_fn_ = inline_run_env_fn
        if inline_run_env_fn_ is not None:
            inline_run_env_fn_ = partial(_handle_triton_inline_data, inline_run_env_fn_)
        if prev_meta is None:
            prev_meta = pfl.PFLCompileFuncMeta(["triton"], inline_run_env_fn_, is_template=is_template, anno_transform=_triton_anno_transform)
            setattr(fn_wrapped, pfl.PFL_COMPILE_META_ATTR, prev_meta)
        else:
            prev_meta.backends = ["triton"]
            prev_meta.inline_run_env_fn = inline_run_env_fn_
            prev_meta.is_template = is_template
            prev_meta.anno_transform = _triton_anno_transform

        return cast(T, fn_wrapped)
    if fn is None:
        return wrapper
    else:
        return wrapper(fn)

def parse_triton_compilable_to_runner(fn: triton.JITFunction, do_meta_eval: bool = True) -> pfl.PFLAsyncRunner:
    fn_unwrapped = inspect.unwrap(fn.fn)
    fn_metadata = pfl.get_compilable_meta(fn_unwrapped)
    assert fn_metadata is not None 
    inline_run_env_fn = fn_metadata.inline_run_env_fn
    assert inline_run_env_fn is not None 
    env = inline_run_env_fn()
    external_annos = {k: type(v) for k, v in env.kwargs.items()}
    meta_args = _validate_and_convert_triton_kwargs(env.kwargs, True)
    lib = pfl.parse_func_to_pfl_library(fn, backend="triton", external_anno=(external_annos, None), func_unwrapper=lambda fn: fn.fn)
    if do_meta_eval:
        evaluator = pfl.PFLStaticEvaluator.meta_evaulator(lib)
        evaluator.eval_total_tree(fn.fn, meta_args)
    return pfl.PFLAsyncRunner(lib)