from functools import partial
import math
import concurrent
import triton
from tensorpc.apps.mls.tsim import get_tensorsim_context_checked
from tensorpc.apps.mls.tsim.tensor import SimTensor
from tensorpc.core import pfl
from typing_extensions import Literal, Self
import numpy as np
import dataclasses
from typing import Annotated, Any, Callable, ClassVar, Optional, Type, TypeAlias, TypeVar, Union, cast, get_type_hints, overload
from tensorpc.apps.mls import tsim
from tensorpc.apps.mls.tsim import DTypeEnum
import triton.language as tl
from triton.experimental.gluon import language as gl

from tensorpc.core.annolib import Undefined

pfl.register_backend(
    "triton",
    pfl.PFLParseConfig(
        allow_var_union=False,
        allow_kw=True,
        allow_nd_slice=True,
        allow_slice=True,
        allow_new_var_after_if=True,
    ))
pfl.register_backend(
    "gluon",
    pfl.PFLParseConfig(
        allow_var_union=False,
        allow_kw=True,
        allow_nd_slice=True,
        allow_slice=True,
        allow_new_var_after_if=True,
    ))

# @pfl.register_pfl_std(mapped_name="int", backend="triton", mapped=int)
# def int_func(x: Any) -> int:
#     return int(x)

# @pfl.register_pfl_std(mapped_name="float", backend="triton", mapped=float)
# def float_func(x: Any) -> float:
#     return float(x)

# @pfl.register_pfl_std(mapped_name="bool", backend="triton", mapped=bool)
# def bool_func(x: Any) -> bool:
#     return bool(x)

@pfl.register_pfl_builtin_proxy(backend_cfg={
    "triton": ("int", int),
    "gluon": ("int", int),
})
@dataclasses.dataclass
class _IntProxy:

    @staticmethod
    def __pfl_proxy_init__(x: Any) -> int:
        return int(x)

@pfl.register_pfl_builtin_proxy(backend_cfg={
    "triton": ("float", float),
    "gluon": ("float", float),
})
@dataclasses.dataclass
class _FloatProxy:

    @staticmethod
    def __pfl_proxy_init__(x: Any) -> float:
        return float(x)

@pfl.register_pfl_builtin_proxy(backend_cfg={
    "triton": ("bool", bool),
    "gluon": ("bool", bool),
})
@dataclasses.dataclass
class _BoolProxy:

    @staticmethod
    def __pfl_proxy_init__(x: Any) -> bool:
        return bool(x)


@pfl.register_pfl_std(backend_cfg={
    "triton": ("range", range),
    "gluon": ("range", range),
})
def range_func(start: int,
               stop: Optional[int] = None,
               step: Optional[int] = None) -> range:
    if stop is None and step is None:
        return range(start)
    elif step is None and stop is not None:
        return range(start, stop)
    else:
        assert stop is not None and step is not None, "stop and step must be provided together"
        return range(start, stop, step)


def _print_meta_infer(fn: Callable, *args: pfl.PFLExprInfo):
    print(
        f"[pfl.staticanalysis]Types: {args}, metadatas: {[x.metadata for x in args]}, "
        f"constexprs: {[x._constexpr_data for x in args]}"
    )
    return None


@pfl.register_pfl_std(backend_cfg={
    "triton": ("print", print),
    "gluon": ("print", print),
})
@pfl.configure_std_func(meta_infer=_print_meta_infer, force_meta_infer=True)
def print_func(*args: Any) -> None:
    print(*args)

@pfl.register_pfl_std(backend_cfg={
    "triton": ("TritonConstexpr", tl.constexpr),
    "gluon": ("TritonConstexpr", gl.constexpr),
})
@dataclasses.dataclass
class ConstExpr:
    pass


@pfl.register_pfl_std(backend_cfg={
    "triton": ("TritonPointerType", tl.pointer_type),
    "gluon": ("TritonPointerType", gl.pointer_type),
})
@dataclasses.dataclass
class pointer_type:
    element_ty: int


@pfl.register_pfl_std(backend_cfg={
    "triton": ("TritonBlockTensor", None),
    "gluon": ("TritonBlockTensor", None),
})
@dataclasses.dataclass
class Tensor:
    _wrapped: tsim.SimTensor

    def _replace_wrapped(self, new_tensor: tsim.SimTensor) -> Self:
        return dataclasses.replace(self, _wrapped=new_tensor)

    def __repr__(self) -> str:
        return self._wrapped.__repr__()

    def _clone(self) -> Self:
        return self._replace_wrapped(self._wrapped.clone())

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

    @overload
    def to(self, dtype: int) -> Self:
        ...

    @overload
    def to(self, dtype: pointer_type) -> "PointerTensor":
        ...

    def to(self, dtype: Union[int,
                              pointer_type]) -> Union[Self, "PointerTensor"]:
        if isinstance(dtype, int):
            return self._replace_wrapped(self._wrapped.to(dtype))
        else:
            if self._wrapped.storage is None:
                return PointerTensor(
                    tsim.create_pointer_tensor_meta(dtype.element_ty,
                                                    self._wrapped.shape))
            ctx = tsim.get_tensorsim_context_checked()
            assert ctx.global_mem is not None, "pointer of pointer must have global memory set."
            return PointerTensor(
                tsim.create_pointer_tensor(
                    dtype.element_ty,
                    self._wrapped.get_storage_checked().data, ctx.global_mem))

    @staticmethod
    def _binary_infer(fn: Callable, this: pfl.PFLExprInfo,
                      other: pfl.PFLExprInfo) -> pfl.PFLMetaInferResult:
        assert this.has_metadata(Tensor, PointerTensor, PointerScalarFloat,
                                 PointerScalarInt)
        if other.has_metadata(Tensor, PointerTensor, PointerScalarFloat,
                              PointerScalarInt):
            # metadata is Tensor or PointerTensor
            return pfl.PFLMetaInferResult(
                fn(this.metadata_checked, other.metadata_checked))
        assert other.type == pfl.PFLExprType.NUMBER or other.type == pfl.PFLExprType.BOOL, "other must be a number or a boolean"
        # when binary operation with Tensor, result won't be constexpr, so we create a new dummy number.
        return pfl.PFLMetaInferResult(
            fn(this.metadata_checked,
               other.get_origin_type_checked()(1)))

    @staticmethod
    def _reshape_permute_meta_infer(
            fn: Callable, x: pfl.PFLExprInfo, *shapes:
        pfl.PFLExprInfo) -> Optional[pfl.PFLMetaInferResult]:
        if len(shapes) == 0:
            # for triton.trans
            return pfl.PFLMetaInferResult(fn(x.metadata_checked))
        shape = shapes[0]
        shapes = shapes[1:]
        if shape.type == pfl.PFLExprType.ARRAY or shape.type == pfl.PFLExprType.TUPLE:
            assert len(shapes) == 0
            assert shape.has_metadata(
            ), "shape must have metadata for reshape operation"
            metadata = shape.metadata_checked
            assert isinstance(
                metadata,
                (list, tuple)), "shape must be a list or tuple of integers"
            for dim in metadata:
                assert not isinstance(
                    dim, Undefined), "element of shape must defined"
        else:
            assert shape.type == pfl.PFLExprType.NUMBER, "shape must be a number or an array of numbers"
            for s in shapes:
                assert s.type == pfl.PFLExprType.NUMBER, "shape must be a number or an array of numbers"
        return pfl.PFLMetaInferResult(
            fn(x.metadata_checked, shape.metadata_checked,
               *[s.metadata_checked for s in shapes]))

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
            return self._replace_wrapped(self._wrapped**other._wrapped)
        return self._replace_wrapped(self._wrapped**other)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __rpow__(self, other: Union[Self, int]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(other._wrapped**self._wrapped)
        return self._replace_wrapped(other**self._wrapped)

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

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __lt__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped < other._wrapped)
        return self._replace_wrapped(self._wrapped < other)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __le__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped <= other._wrapped)
        return self._replace_wrapped(self._wrapped <= other)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __ge__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped >= other._wrapped)
        return self._replace_wrapped(self._wrapped >= other)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __gt__(self, other: Union[Self, int, float]) -> Self:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped > other._wrapped)
        return self._replace_wrapped(self._wrapped > other)

    @overload
    def __eq__(self, other: Self) -> Self:
        ...

    @overload
    def __eq__(self, other: Union[int, float]) -> Self:
        ...

    @overload
    def __ne__(self, other: Self) -> Self:
        ...

    @overload
    def __ne__(self, other: Union[int, float]) -> Self:
        ...

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __eq__(self, other: Any) -> Any:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped == other._wrapped)
        return self._replace_wrapped(self._wrapped == other)

    @pfl.configure_std_func(meta_infer=_binary_infer)
    def __ne__(self, other: Any) -> Any:
        if isinstance(other, Tensor):
            return self._replace_wrapped(self._wrapped != other._wrapped)
        return self._replace_wrapped(self._wrapped != other)

    @overload
    def reshape(self, shape: int, *shapes: int) -> Self:
        ...

    @overload
    def reshape(self, shape: Union[list[int], tuple[int, ...]], *shapes:
                int) -> Self:
        ...

    @pfl.configure_std_func(meta_infer=_reshape_permute_meta_infer)
    def reshape(self, shape: Union[list[int], tuple[int, ...], int], *shapes:
                int) -> Self:
        return self._replace_wrapped(self._wrapped.reshape(shape, *shapes))

    @overload
    def permute(self, shape: int, *shapes: int) -> Self:
        ...

    @overload
    def permute(self, shape: Union[list[int], tuple[int, ...]], *shapes:
                int) -> Self:
        ...

    @pfl.configure_std_func(meta_infer=_reshape_permute_meta_infer)
    def permute(self, shape: Union[list[int], tuple[int, ...], int], *shapes:
                int) -> Self:
        return self._replace_wrapped(self._wrapped.permute(shape, *shapes))

    def split(self) -> "tuple[Tensor, Tensor]":
        last_dim = self.shape[-1]
        assert last_dim == 2, "triton split only support last dimension of 2"
        assert last_dim % 2 == 0, "split only support even first dimension"
        # last_dim_div_2 = last_dim // 2
        # res = self._replace_wrapped(self._wrapped[..., :last_dim_div_2]), self._replace_wrapped(self._wrapped[..., last_dim_div_2:])
        res = self._replace_wrapped(
            self._wrapped[..., 0]), self._replace_wrapped(self._wrapped[...,
                                                                        1])
        return res

    def cumsum(self, axis: int) -> Self:
        return self._replace_wrapped(self._wrapped.cumsum(axis))

    # @staticmethod
    # def max_with_indices(input: Tensor, axis: int, return_indices_tie_break_left: bool = True, keep_dims: bool = False) -> Tensor: ...

    # @staticmethod
    # def min_with_indices(input: Tensor, axis: int, return_indices_tie_break_left: bool = True, keep_dims: bool = False) -> Tensor: ...

    def sum(self,
            axis: int,
            keep_dims: bool = False,
            dtype: Optional[int] = None) -> Self:
        assert isinstance(axis, int)
        return self._replace_wrapped(self._wrapped.sum(axis, keep_dims))

@pfl.register_pfl_std(backend_cfg={
    "triton": ("TritonPointerTensor", None),
    "gluon": ("TritonPointerTensor", None),
})
@dataclasses.dataclass
class PointerTensor:
    _wrapped: tsim.SimPointerTensor

    def clone(self) -> Self:
        # create a new pointer tensor with the same metadata
        return self._replace_wrapped(self._wrapped.clone())

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


if hasattr(tl, "_experimental_tensor_descriptor"):
    _TRITON_TD_CLS = tl._experimental_tensor_descriptor
elif hasattr(tl, "tensor_descriptor"):
    _TRITON_TD_CLS = tl.tensor_descriptor
else:
    _TRITON_TD_CLS = None

@pfl.register_pfl_std(mapped_name="TritonTensorDesc",
                      backend="triton",
                      mapped=_TRITON_TD_CLS)
@dataclasses.dataclass
class TensorDescriptor:
    _wrapped: tsim.SimTensorBlockPointer

    def clone(self) -> Self:
        # create a new pointer tensor with the same metadata
        return dataclasses.replace(self, _wrapped=self._wrapped.clone())

    @staticmethod
    def _load_infer(fn: Callable, this: pfl.PFLExprInfo,
                    offset: pfl.PFLExprInfo) -> pfl.PFLMetaInferResult:
        assert this.has_metadata(
            TensorDescriptor
        ), f"this must have metadata of TensorDescriptor, got {this.metadata}"
        return pfl.PFLMetaInferResult(this.metadata_checked.load([0, 0]))

    @pfl.configure_std_func(meta_infer=_load_infer)
    def load(self, offset: list[int]) -> Tensor:
        return Tensor(self._wrapped.load(offset, other=0))

    def store(self, offset: list[int], value: Union[int, float,
                                                    Tensor]) -> None:
        value_wrapped = value._wrapped if isinstance(value, Tensor) else value
        self._wrapped.store(offset, value_wrapped)

    def atomic_add(self, offset: list[int], value: Tensor) -> Tensor:
        res = self._wrapped.store(offset, value._wrapped, atomic_op="add")
        assert res is not None
        return Tensor(res)

@pfl.register_pfl_std(backend_cfg={
    "triton": ("TritonBlockPointer", None),
    "gluon": ("TritonBlockPointer", None),
})
@dataclasses.dataclass
class BlockPointer:
    _wrapped: tsim.SimTensorBlockPointer

    def advance(self, offset: Union[list[int], tuple[int, ...]]) -> Self:
        return self.__class__(self._wrapped.advance(list(offset)))


@pfl.register_pfl_std(backend_cfg={
    "triton": ("TritonPointerScalarFloat", None),
    "gluon": ("TritonPointerScalarFloat", None),
})
@dataclasses.dataclass
class PointerScalarFloat:
    _wrapped: tsim.SimPointerScalar

    def _replace_wrapped(self, new_tensor: tsim.SimPointerScalar) -> Self:
        return dataclasses.replace(self, _wrapped=new_tensor)

    def clone(self) -> Self:
        # create a new pointer tensor with the same metadata
        return self._replace_wrapped(self._wrapped.clone())

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

    @overload
    def __add__(self, other: Tensor) -> PointerTensor:
        ...

    @overload
    def __add__(self, other: int) -> Self:
        ...

    # pointer tensor only support add and sub operations
    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __add__(self, other: Union[Tensor, int]) -> Union[Self, PointerTensor]:
        if isinstance(other, Tensor):
            res = self._wrapped + other._wrapped
            return PointerTensor(res)
        return self._replace_wrapped(self._wrapped + other)

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __iadd__(self, other: int) -> Self:
        self._wrapped += other
        return self

    @overload
    def __radd__(self, other: Tensor) -> PointerTensor:
        ...

    @overload
    def __radd__(self, other: int) -> Self:
        ...

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __radd__(self, other: Union[Tensor,
                                    int]) -> Union[Self, PointerTensor]:
        if isinstance(other, Tensor):
            return PointerTensor(other._wrapped + self._wrapped)
        return self._replace_wrapped(other + self._wrapped)

    @overload
    def __sub__(self, other: Tensor) -> PointerTensor:
        ...

    @overload
    def __sub__(self, other: int) -> Self:
        ...

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __sub__(self, other: Union[Tensor, int]) -> Union[Self, PointerTensor]:
        if isinstance(other, Tensor):
            return PointerTensor(self._wrapped - other._wrapped)
        return self._replace_wrapped(self._wrapped - other)

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __isub__(self, other: int) -> Self:
        self._wrapped -= other
        return self

    @overload
    def __rsub__(self, other: Tensor) -> PointerTensor:
        ...

    @overload
    def __rsub__(self, other: int) -> Self:
        ...

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __rsub__(self, other: Union[Tensor,
                                    int]) -> Union[Self, PointerTensor]:
        if isinstance(other, Tensor):
            return PointerTensor(other._wrapped - self._wrapped)

        return self._replace_wrapped(other - self._wrapped)


@pfl.register_pfl_std(backend_cfg={
    "triton": ("TritonPointerScalarInt", None),
    "gluon": ("GluonPointerScalarInt", None),
})
@dataclasses.dataclass
class PointerScalarInt:
    _wrapped: tsim.SimPointerScalar

    def _replace_wrapped(self, new_tensor: tsim.SimPointerScalar) -> Self:
        return dataclasses.replace(self, _wrapped=new_tensor)

    def clone(self) -> Self:
        # create a new pointer tensor with the same metadata
        return self._replace_wrapped(self._wrapped.clone())

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

    @overload
    def __add__(self, other: Tensor) -> PointerTensor:
        ...

    @overload
    def __add__(self, other: int) -> Self:
        ...

    @overload
    def __radd__(self, other: Tensor) -> PointerTensor:
        ...

    @overload
    def __radd__(self, other: int) -> Self:
        ...

    # pointer tensor only support add and sub operations
    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __add__(self, other: Union[Tensor, int]) -> Union[Self, PointerTensor]:
        if isinstance(other, Tensor):
            return PointerTensor(self._wrapped + other._wrapped)
        return self._replace_wrapped(self._wrapped + other)

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __iadd__(self, other: int) -> Self:
        self._wrapped += other
        return self

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __radd__(self, other: Union[Tensor,
                                    int]) -> Union[Self, PointerTensor]:
        if isinstance(other, Tensor):
            return PointerTensor(other._wrapped + self._wrapped)
        return self._replace_wrapped(other + self._wrapped)

    @overload
    def __sub__(self, other: Tensor) -> PointerTensor:
        ...

    @overload
    def __sub__(self, other: int) -> Self:
        ...

    @overload
    def __rsub__(self, other: Tensor) -> PointerTensor:
        ...

    @overload
    def __rsub__(self, other: int) -> Self:
        ...

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __sub__(self, other: Union[Tensor, int]) -> Union[Self, PointerTensor]:
        if isinstance(other, Tensor):
            return PointerTensor(self._wrapped - other._wrapped)
        return self._replace_wrapped(self._wrapped - other)

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __isub__(self, other: int) -> Self:
        self._wrapped -= other
        return self

    @pfl.configure_std_func(meta_infer=Tensor._binary_infer)
    def __rsub__(self, other: Union[Tensor,
                                    int]) -> Union[Self, PointerTensor]:
        if isinstance(other, Tensor):
            return PointerTensor(other._wrapped - self._wrapped)

        return self._replace_wrapped(other - self._wrapped)


def _none_infer(fn: Callable, *args,
                **kwargs) -> Optional[pfl.PFLMetaInferResult]:
    return None

def _global_unary_infer(
        fn: Callable, x: pfl.PFLExprInfo) -> Optional[pfl.PFLMetaInferResult]:
    if x.type == pfl.PFLExprType.NUMBER:
        if x.has_metadata():
            return pfl.PFLMetaInferResult(fn(x))
        return None
    return pfl.PFLMetaInferResult(fn(x.metadata_checked))


def _global_binary_infer(
        fn: Callable, this: pfl.PFLExprInfo,
        other: pfl.PFLExprInfo) -> Optional[pfl.PFLMetaInferResult]:
    # binary only do const eval if all operands are number and have metadata.
    if this.type == pfl.PFLExprType.NUMBER and other.type == pfl.PFLExprType.NUMBER:
        # only calc metadata (consteval) if all operands have metadata (const value)
        if this.has_metadata() and other.has_metadata():
            return pfl.PFLMetaInferResult(
                fn(this.metadata_checked, other.metadata_checked))
        return None
    if not this.has_metadata(Tensor) and other.has_metadata(Tensor):
        assert this.type == pfl.PFLExprType.NUMBER
        return pfl.PFLMetaInferResult(
            fn(other.get_origin_type_checked()(1), other.metadata_checked))
    elif this.has_metadata(Tensor) and not other.has_metadata(Tensor):
        assert other.type == pfl.PFLExprType.NUMBER
        return pfl.PFLMetaInferResult(
            fn(this.metadata_checked,
               other.get_origin_type_checked()(1)))
    elif this.has_metadata(Tensor) and other.has_metadata(Tensor):
        return pfl.PFLMetaInferResult(
            fn(this.metadata_checked, other.metadata_checked))
    else:
        raise NotImplementedError(
            f"Unsupported binary operation between {this.type} and {other.type}"
        )


@overload
def min_fn(x: int, y: int) -> int:
    ...


@overload
def min_fn(x: float, y: float) -> float:
    ...


@overload
def min_fn(x: Tensor, y: Union[int, float]) -> Tensor:
    ...


@overload
def min_fn(x: Union[int, float], y: Tensor) -> Tensor:
    ...


@pfl.register_pfl_std(backend_cfg={
    "triton": ("min", min),
    "gluon": ("min", min),
})
@pfl.configure_std_func(meta_infer=_global_binary_infer)
def min_fn(x: Union[Tensor, int, float],
           y: Union[Tensor, int, float]) -> Union[Tensor, int, float]:
    if not isinstance(x, Tensor) and not isinstance(y, Tensor):
        return min(x, y)
    x_wrapped = x._wrapped if isinstance(x, Tensor) else x
    y_wrapped = y._wrapped if isinstance(y, Tensor) else y
    return Tensor(tsim.minimum(x_wrapped, y_wrapped))


@overload
def max_fn(x: int, y: int) -> int:
    ...


@overload
def max_fn(x: float, y: float) -> float:
    ...


@overload
def max_fn(x: Tensor, y: Union[int, float]) -> Tensor:
    ...


@overload
def max_fn(x: Union[int, float], y: Tensor) -> Tensor:
    ...


@pfl.register_pfl_std(backend_cfg={
    "triton": ("max", max),
    "gluon": ("max", max),
})
@pfl.configure_std_func(meta_infer=_global_binary_infer)
def max_fn(x: Union[Tensor, int, float],
           y: Union[Tensor, int, float]) -> Union[Tensor, int, float]:
    if not isinstance(x, Tensor) and not isinstance(y, Tensor):
        return max(x, y)
    x_wrapped = x._wrapped if isinstance(x, Tensor) else x
    y_wrapped = y._wrapped if isinstance(y, Tensor) else y
    return Tensor(tsim.minimum(x_wrapped, y_wrapped))


_T_math = TypeVar("_T_math", Tensor, int, float)
_T_math_fp = TypeVar("_T_math_fp", Tensor, float)
_T_any = TypeVar("_T_any", Tensor, int, float, PointerTensor,
                 PointerScalarFloat, PointerScalarInt)
_T_all_tensor = TypeVar("_T_all_tensor", Tensor, PointerTensor)


@pfl.register_pfl_std(mapped_name="tl_math", backend="triton", mapped=tl.math)
@dataclasses.dataclass
class triton_std_math:

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
    def _internal_binary_anno(x: Tensor, y: float) -> Tensor:
        ...

    @staticmethod
    @overload
    def _internal_binary_anno(x: float, y: Tensor) -> Tensor:
        ...

    @staticmethod
    @overload
    def _internal_binary_anno(x: float, y: float) -> float:
        ...

    @staticmethod
    @overload
    def _internal_binary_anno(x: Tensor, y: Tensor) -> Tensor:
        ...

    @staticmethod
    def _internal_binary_anno(x: Union[Tensor, float],
                              y: Union[Tensor, float]) -> Union[Tensor, float]:
        raise NotImplementedError("shouldn't be used directly.")

    @staticmethod
    @pfl.configure_std_func(take_overloads_fn=_internal_binary_anno,
                            meta_infer=_global_binary_infer)
    def maximum(x: Union[Tensor, float], y: Union[Tensor, float]) -> Tensor:
        x_wrapped = x._wrapped if isinstance(x, Tensor) else x
        y_wrapped = y._wrapped if isinstance(y, Tensor) else y
        res_wrapped = tsim.maximum(x_wrapped, y_wrapped)
        if isinstance(res_wrapped, SimTensor):
            return Tensor(res_wrapped)
        else:
            return res_wrapped

    @staticmethod
    @pfl.configure_std_func(take_overloads_fn=_internal_binary_anno,
                            meta_infer=_global_binary_infer)
    def minimum(x: Union[Tensor, float], y: Union[Tensor, float]) -> Tensor:
        x_wrapped = x._wrapped if isinstance(x, Tensor) else x
        y_wrapped = y._wrapped if isinstance(y, Tensor) else y
        res_wrapped = tsim.minimum(x_wrapped, y_wrapped)
        if isinstance(res_wrapped, SimTensor):
            return Tensor(res_wrapped)
        else:
            return res_wrapped

    @staticmethod
    def _clamp_infer(fn: Callable, x: pfl.PFLExprInfo, min: pfl.PFLExprInfo,
                     max: pfl.PFLExprInfo) -> Optional[pfl.PFLMetaInferResult]:
        if x.has_metadata():
            return pfl.PFLMetaInferResult(x.metadata_checked)
        return None 

    @staticmethod
    @pfl.configure_std_func(meta_infer=_clamp_infer)
    def clamp(x: Tensor, min: Union[Tensor, int, float],
              max: Union[Tensor, int, float]) -> Tensor:
        min_wrapped = min._wrapped if isinstance(min, Tensor) else min
        max_wrapped = max._wrapped if isinstance(max, Tensor) else max
        return Tensor(tsim.clamp(x._wrapped, min_wrapped, max_wrapped))


@dataclasses.dataclass
class triton_gluon_std(triton_std_math):
    # we use inherit here to import all methods from triton_std_math
    # we can also use ClassVar. e.g. abs: ClassVar = staticmethod(triton_std_math.abs)

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
    bfloat16: DTypeEnum = DTypeEnum.bfloat16

    float8e5: DTypeEnum = DTypeEnum.float8e5
    float8e5b16: DTypeEnum = DTypeEnum.float8e5b16
    float8e4nv: DTypeEnum = DTypeEnum.float8e4nv
    float8e4b8: DTypeEnum = DTypeEnum.float8e4b8
    float8e4b15: DTypeEnum = DTypeEnum.float8e4b15

    @staticmethod
    def _program_id_infer(
            fn: Callable,
            axis: pfl.PFLExprInfo) -> Optional[pfl.PFLMetaInferResult]:
        return None

    @staticmethod
    @pfl.configure_std_func(meta_infer=_program_id_infer)
    def program_id(axis: int) -> int:
        ctx = get_tensorsim_context_checked()
        return ctx.grid_id[axis]

    @staticmethod
    def _compiler_hints_infer(
            fn: Callable, x: pfl.PFLExprInfo,
            y: pfl.PFLExprInfo) -> Optional[pfl.PFLMetaInferResult]:
        return pfl.PFLMetaInferResult(x.metadata)

    @staticmethod
    @pfl.configure_std_func(meta_infer=_compiler_hints_infer)
    def max_contiguous(x: _T_any, y: Any) -> _T_any:
        return x

    @staticmethod
    @pfl.configure_std_func(meta_infer=_compiler_hints_infer)
    def multiple_of(x: _T_any, y: Any) -> _T_any:
        return x

    @staticmethod
    @pfl.configure_std_func(meta_infer=_program_id_infer)
    def num_programs(axis: int) -> int:
        ctx = get_tensorsim_context_checked()
        return ctx.grid_size[axis]
    
    @staticmethod
    def zeros(shape: Union[list[int], tuple[int, ...]], dtype: int) -> Tensor:
        return Tensor(tsim.zeros(shape, dtype))

    @staticmethod
    def arange(start: int, end: int) -> Tensor:
        return Tensor(tsim.arange(start, end))

    @staticmethod
    def _load_infer(
        fn: Callable,
        pointer: pfl.PFLExprInfo,
        mask: Optional[pfl.PFLExprInfo] = None,
        other: Optional[pfl.PFLExprInfo] = None,
        *args,
        **kwargs
    ) -> Optional[pfl.PFLMetaInferResult]:
        if isinstance(pointer.metadata_checked,
                      (PointerScalarFloat, PointerScalarInt)):
            return None
        if other is None:
            other_value = None
        else:
            if not other.has_metadata(Tensor):
                if other.type == pfl.PFLExprType.NUMBER:
                    other_value = other.get_origin_type_checked()(1)
                else:
                    raise NotImplementedError(
                        f"Unsupported type {other.type} for load operation")
            else:
                other_value = other.metadata_checked
        if mask is None:
            mask_value = None
        else:
            mask_value = mask.metadata_checked if mask.has_metadata(
                Tensor) else None
        return pfl.PFLMetaInferResult(
            fn(pointer.metadata_checked, mask_value, other_value))

    @staticmethod
    @overload
    def load(pointer: PointerScalarFloat,
             mask: Optional[bool] = None,
             other: Optional[Union[int, float]] = None,
             boundary_check: Optional[tuple[int, ...]] = None,
             padding_option: Optional[str] = None,
             cache_modifier: Optional[str] = None,
             eviction_policy: Optional[str] = None) -> float:
        ...

    @staticmethod
    @overload
    def load(pointer: PointerScalarInt,
             mask: Optional[bool] = None,
             other: Optional[Union[int, float]] = None,
             boundary_check: Optional[tuple[int, ...]] = None,
             padding_option: Optional[str] = None,
             cache_modifier: Optional[str] = None,
             eviction_policy: Optional[str] = None) -> int:
        ...

    @staticmethod
    @overload
    def load(pointer: PointerTensor,
             mask: Optional[Tensor] = None,
             other: Optional[Union[int, float, Tensor]] = None,
             boundary_check: Optional[tuple[int, ...]] = None,
             padding_option: Optional[str] = None,
             cache_modifier: Optional[str] = None,
             eviction_policy: Optional[str] = None) -> Tensor:
        ...

    @staticmethod
    @overload
    def load(pointer: BlockPointer,
             mask: Optional[Tensor] = None,
             other: Optional[Union[int, float, Tensor]] = None,
             boundary_check: Optional[tuple[int, ...]] = None,
             padding_option: Optional[str] = None,
             cache_modifier: Optional[str] = None,
             eviction_policy: Optional[str] = None) -> Tensor:
        ...

    @staticmethod
    @pfl.configure_std_func(meta_infer=_load_infer)
    def load(
        pointer: Union[PointerTensor, PointerScalarFloat, PointerScalarInt,
                       BlockPointer],
        mask: Optional[Union[Tensor, bool]] = None,
        other: Optional[Union[int, float, Tensor]] = None,
        boundary_check: Optional[tuple[int, ...]] = None,
        padding_option: Optional[str] = None,
        cache_modifier: Optional[str] = None,
        eviction_policy: Optional[str] = None
    ) -> Union[Tensor, int, float]:
        mask_wrapped = mask._wrapped if isinstance(mask, Tensor) else mask
        other_wrapped = other._wrapped if isinstance(other, Tensor) else other
        if isinstance(pointer, PointerScalarFloat):
            if mask_wrapped is not None:
                assert isinstance(mask_wrapped, bool)
            assert not isinstance(other_wrapped, tsim.SimTensor)
            return pointer._wrapped.load(mask_wrapped, other_wrapped)
        elif isinstance(pointer, PointerScalarInt):
            if mask_wrapped is not None:
                assert isinstance(mask_wrapped, bool)
            assert not isinstance(other_wrapped, tsim.SimTensor)
            return pointer._wrapped.load(mask_wrapped, other_wrapped)
        elif isinstance(pointer, BlockPointer):
            assert mask is None, "BlockPointer does not support (don't need) mask"
            assert other is None, "BlockPointer does not support other"
            other_for_block = None 
            if padding_option == "zero":
                other_for_block = 0 
            elif padding_option == "nan":
                other_for_block = float("nan")
            return Tensor(pointer._wrapped.load([0, 0], other_for_block))
        return Tensor(pointer._wrapped.load(mask_wrapped, other_wrapped))

    @staticmethod
    @overload
    def store(pointer: PointerScalarFloat,
              value: float,
              mask: Optional[bool] = None,
              boundary_check: Optional[tuple[int, ...]] = None,
             cache_modifier: Optional[str] = None,
             eviction_policy: Optional[str] = None) -> None:
        ...

    @staticmethod
    @overload
    def store(pointer: PointerScalarInt,
              value: int,
              mask: Optional[bool] = None,
              boundary_check: Optional[tuple[int, ...]] = None,
             cache_modifier: Optional[str] = None,
             eviction_policy: Optional[str] = None) -> None:
        ...

    @staticmethod
    @overload
    def store(pointer: PointerTensor,
              value: Union[int, float, Tensor],
              mask: Optional[Tensor] = None,
              boundary_check: Optional[tuple[int, ...]] = None,
             cache_modifier: Optional[str] = None,
             eviction_policy: Optional[str] = None) -> None:
        ...

    @staticmethod
    @overload
    def store(pointer: BlockPointer, value: Union[int, float, Tensor],
              mask: Optional[Tensor] = None,
              boundary_check: Optional[tuple[int, ...]] = None,
             cache_modifier: Optional[str] = None,
             eviction_policy: Optional[str] = None) -> None:
        ...

    @staticmethod
    @pfl.configure_std_func(meta_infer=_none_infer)
    def store(pointer: Union[PointerTensor, PointerScalarFloat,
                             PointerScalarInt, BlockPointer],
              value: Union[int, float, Tensor],
              mask: Optional[Union[Tensor, bool]] = None,
              boundary_check: Optional[tuple[int, ...]] = None,
             cache_modifier: Optional[str] = None,
             eviction_policy: Optional[str] = None) -> None:
        mask_wrapped = mask._wrapped if isinstance(mask, Tensor) else mask
        value_wrapped = value._wrapped if isinstance(value, Tensor) else value
        if isinstance(pointer, PointerScalarFloat):
            assert mask_wrapped is None or isinstance(mask_wrapped, bool)
            assert not isinstance(value_wrapped, tsim.SimTensor)
            pointer._wrapped.store(value_wrapped, mask_wrapped)
            return
        elif isinstance(pointer, PointerScalarInt):
            assert mask_wrapped is None or isinstance(mask_wrapped, bool)
            assert not isinstance(value_wrapped, tsim.SimTensor)
            pointer._wrapped.store(value_wrapped, mask_wrapped)
            return
        elif isinstance(pointer, BlockPointer):
            assert mask is None, "BlockPointer does not support (don't need) mask"
            assert isinstance(value_wrapped, tsim.SimTensor)
            pointer._wrapped.store([0, 0], value_wrapped)
            return
        pointer._wrapped.store(value_wrapped, mask_wrapped)
    
    @staticmethod
    def static_range(start: int,
              stop: Optional[int] = None,
              step: Optional[int] = None) -> range:
        if stop is not None:
            if step is not None:
                return range(start, stop, step)
            return range(start, stop)
        return range(start)

    @staticmethod
    def range(start: int,
              stop: Optional[int] = None,
              step: Optional[int] = None,
              num_stages: Optional[int] = None,
              warp_specialize: Optional[int] = None,
              flatten: bool = False) -> range:
        if stop is not None:
            if step is not None:
                return range(start, stop, step)
            return range(start, stop)
        return range(start)


    @staticmethod
    @overload
    def cdiv(x: Tensor, div: Union[Tensor, int, float]) -> Tensor:
        ...

    @staticmethod
    @overload
    def cdiv(x: int, div: int) -> int:
        ...

    @staticmethod
    @overload
    def cdiv(x: float, div: float) -> float:
        ...

    @staticmethod
    @pfl.configure_std_func(meta_infer=_global_binary_infer)
    def cdiv(x: Union[Tensor, int, float],
             div: Union[Tensor, int, float]) -> Union[int, float, Tensor]:
        x_wrapped = x._wrapped if isinstance(x, Tensor) else x
        div_wrapped = div._wrapped if isinstance(div, Tensor) else div
        res = (x_wrapped + div_wrapped - 1) // div_wrapped
        if isinstance(res, tsim.SimTensor):
            return Tensor(res)
        return res


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
    def sum(input: Tensor,
            axis: int,
            keep_dims: bool = False,
            dtype: Optional[int] = None) -> Tensor:
        assert isinstance(axis, int)
        return Tensor(input._wrapped.sum(axis, keep_dims))

    @staticmethod
    def _where_infer(fn: Callable, cond: pfl.PFLExprInfo, x: pfl.PFLExprInfo,
                     y: pfl.PFLExprInfo) -> Optional[pfl.PFLMetaInferResult]:
        if x.type == pfl.PFLExprType.NUMBER:
            x_val = x.get_origin_type_checked()(1)
        else:
            x_val = x.metadata_checked
        if y.type == pfl.PFLExprType.NUMBER:
            y_val = y.get_origin_type_checked()(1)
        else:
            y_val = y.metadata_checked
        return pfl.PFLMetaInferResult(fn(cond.metadata_checked, x_val, y_val))

    @staticmethod
    @pfl.configure_std_func(meta_infer=_where_infer)
    def where(cond: Tensor, x: Union[Tensor, int, float],
              y: Union[Tensor, int, float]) -> Tensor:
        x_wrapped = x._wrapped if isinstance(x, Tensor) else x
        y_wrapped = y._wrapped if isinstance(y, Tensor) else y
        return Tensor(tsim.where(cond._wrapped, x_wrapped, y_wrapped))


    @staticmethod
    @overload
    def reshape(x: _T_all_tensor, shape: int, *shapes: int) -> _T_all_tensor:
        ...

    @staticmethod
    @overload
    def reshape(x: _T_all_tensor, shape: Union[list[int], tuple[int, ...]],
                *shapes: int) -> _T_all_tensor:
        ...

    @staticmethod
    @pfl.configure_std_func(meta_infer=Tensor._reshape_permute_meta_infer)
    def reshape(x: _T_all_tensor, shape: Union[list[int], tuple[int, ...],
                                               int], *shapes:
                int) -> _T_all_tensor:
        if isinstance(x, Tensor):
            return Tensor(x._wrapped.reshape(shape, *shapes))
        else:
            assert isinstance(
                x, PointerTensor
            ), "reshape only support Tensor and PointerTensor type"
            return PointerTensor(x._wrapped.reshape(shape, *shapes))

    @staticmethod
    @overload
    def permute(x: _T_all_tensor, shape: int, *shapes: int) -> _T_all_tensor:
        ...

    @staticmethod
    @overload
    def permute(x: _T_all_tensor, shape: Union[list[int], tuple[int, ...]],
                *shapes: int) -> _T_all_tensor:
        ...

    @staticmethod
    @pfl.configure_std_func(meta_infer=Tensor._reshape_permute_meta_infer)
    def permute(x: _T_all_tensor, shape: Union[list[int], tuple[int, ...],
                                               int], *shapes:
                int) -> _T_all_tensor:
        if isinstance(x, Tensor):
            return Tensor(x._wrapped.permute(shape))
        else:
            assert isinstance(
                x, PointerTensor
            ), "reshape only support Tensor and PointerTensor type"
            return PointerTensor(x._wrapped.permute(shape))


    @staticmethod
    def split(x: _T_all_tensor) -> tuple[_T_all_tensor, _T_all_tensor]:
        last_dim = x.shape[-1]
        assert last_dim == 2, "triton split only support last dimension of size 2"
        # assert last_dim % 2 == 0, "split only support even first dimension"
        # last_dim_div_2 = last_dim // 2

        if isinstance(x, Tensor):
            return Tensor(x._wrapped[..., 0]), Tensor(x._wrapped[..., 1])
        else:
            assert isinstance(
                x, PointerTensor
            ), "split only support Tensor and PointerTensor type"
            return PointerTensor(x._wrapped[...,
                                            0]), PointerTensor(x._wrapped[...,
                                                                          1])

    @staticmethod
    def join(x: _T_all_tensor, y: _T_all_tensor) -> _T_all_tensor:
        if isinstance(x, Tensor) and isinstance(y, Tensor):
            return Tensor(x._wrapped.stack([y._wrapped], axis=-1))
        else:
            assert isinstance(x, PointerTensor) and isinstance(
                y, PointerTensor
            ), "join only support Tensor and PointerTensor type"
            return PointerTensor(x._wrapped.stack([y._wrapped], axis=-1))


    @staticmethod
    def _static_assert_static_infer(x: pfl.PFLExprInfo):
        assert x.has_constexpr_data(
        ), "static_assert only support constexpr data"
        assert x._constexpr_data, "static_assert condition must be True"

    @staticmethod
    def static_assert(x: bool) -> None:
        assert x

    @staticmethod
    def advance(base: BlockPointer,
                offset: Union[list[int], tuple[int, ...]]) -> BlockPointer:
        return base.advance(offset)

    @staticmethod
    def _atomic_meta_infer(
            fn, pointer: pfl.PFLExprInfo, *args, **kwargs) -> Optional[pfl.PFLMetaInferResult]:
        if not pointer.has_metadata():
            return None 
        pointer_meta = pointer.metadata_checked
        if isinstance(pointer_meta, (PointerScalarFloat, PointerScalarInt)):
            return None 
        assert isinstance(pointer_meta, (PointerTensor, BlockPointer)), \
            "pointer must be PointerTensor or BlockPointer"
        _unused = 1
        if isinstance(pointer_meta, PointerTensor):
            store_res = pointer_meta._wrapped.store(_unused)
        else:
            store_res = pointer_meta._wrapped.store([0, 0], _unused)
        assert isinstance(store_res, tsim.SimTensor)
        return pfl.PFLMetaInferResult(Tensor(store_res))

    @staticmethod
    @overload
    def atomic_add(pointer: PointerScalarFloat,
              value: float, 
              mask: Optional[bool] = None,
              sem: Optional[str] = None,
              scope: Optional[str] = None) -> float:
        ...

    @staticmethod
    @overload
    def atomic_add(pointer: PointerScalarInt,
              value: int, 
              mask: Optional[bool] = None,
              sem: Optional[str] = None,
              scope: Optional[str] = None) -> int:
        ...

    @staticmethod
    @overload
    def atomic_add(pointer: PointerTensor,
              value: Tensor, 
              mask: Optional[Tensor] = None,
              sem: Optional[str] = None,
              scope: Optional[str] = None) -> Tensor:
        ...


    @staticmethod
    @pfl.configure_std_func(meta_infer=_atomic_meta_infer)
    def atomic_add(pointer: Union[PointerTensor, PointerScalarFloat,
                             PointerScalarInt, BlockPointer],
              value: Union[int, float, Tensor],
              mask: Optional[Union[Tensor, bool]] = None,
              sem: Optional[str] = None,
              scope: Optional[str] = None) -> Union[Tensor, int, float]:
        value_wrapped = value._wrapped if isinstance(value, Tensor) else value
        mask_wrapped = mask._wrapped if isinstance(mask, Tensor) else mask
        if isinstance(pointer, PointerScalarFloat):
            assert mask_wrapped is None or isinstance(mask_wrapped, bool)
            assert not isinstance(value_wrapped, tsim.SimTensor)
            res = pointer._wrapped.store(value_wrapped, mask_wrapped, atomic_op="add")
        elif isinstance(pointer, PointerScalarInt):
            assert mask_wrapped is None or isinstance(mask_wrapped, bool)
            assert not isinstance(value_wrapped, tsim.SimTensor)
            res = pointer._wrapped.store(value_wrapped, mask_wrapped, atomic_op="add")
        elif isinstance(pointer, BlockPointer):
            assert mask is None, "BlockPointer does not support (don't need) mask"
            assert isinstance(value_wrapped, tsim.SimTensor)
            res1 = pointer._wrapped.store([0, 0], value_wrapped, atomic_op="add")
            assert isinstance(res1, tsim.SimTensor)
            res = Tensor(res1)
        else:
            res1 = pointer._wrapped.store(value_wrapped, mask_wrapped, atomic_op="add")
            assert isinstance(res1, tsim.SimTensor), str(type(res1))
            res = Tensor(res1)
        assert res is not None 
        return res



@pfl.register_pfl_std(mapped_name="tl", backend="triton", mapped=tl)
@dataclasses.dataclass
class triton_std(triton_gluon_std):
    @staticmethod
    def _make_block_pointer_meta_infer(
            base_cls: Union[Type[TensorDescriptor],
                            Type[BlockPointer]], fn, base: pfl.PFLExprInfo,
            shape: pfl.PFLExprInfo, strides: pfl.PFLExprInfo,
            offsets: pfl.PFLExprInfo, block_shape: pfl.PFLExprInfo, *args,
            **kwargs) -> Optional[pfl.PFLMetaInferResult]:
        assert base.has_metadata() and shape.has_metadata(
        ) and strides.has_metadata() and block_shape.has_metadata(
        ) and offsets.has_metadata(
        ), "base, shape, strides and block_shape must have metadata"
        shape_val = shape.metadata_checked  # may be list of int or list of undefined.
        strides_val = strides.metadata_checked
        block_shape_val = block_shape.metadata_checked
        for val in block_shape_val:
            assert isinstance(val, int), "block_shape must be constexpr"
        assert (
            len(shape_val) == len(strides_val) == len(block_shape_val)
        ), "Shape, strides, block_shape and offset must have the same length"
        return pfl.PFLMetaInferResult(
            base_cls(
                tsim.create_tensor_block_pointer_meta(
                    base.metadata_checked._wrapped, len(shape_val),
                    block_shape_val)))

    @staticmethod
    @pfl.configure_std_func(
        meta_infer=partial(_make_block_pointer_meta_infer, TensorDescriptor))
    def make_tensor_descriptor(base: Union[PointerScalarFloat,
                                           PointerScalarInt], shape: list[int],
                               strides: list[int],
                               block_shape: list[int]) -> TensorDescriptor:
        return TensorDescriptor(
            tsim.create_tensor_block_pointer(base._wrapped, shape, strides,
                                             block_shape))

    @staticmethod
    @pfl.configure_std_func(
        meta_infer=partial(_make_block_pointer_meta_infer, BlockPointer))
    def make_block_ptr(
            base: Union[PointerScalarFloat,
                        PointerScalarInt], shape: Union[list[int], tuple[int,
                                                                         ...]],
            strides: Union[list[int],
                           tuple[int, ...]], offsets: Union[list[int],
                                                            tuple[int, ...]],
            block_shape: Union[list[int], tuple[int, ...]],
            order: Union[list[int], tuple[int, ...]]) -> BlockPointer:
        return BlockPointer(
            tsim.create_tensor_block_pointer(base._wrapped, list(shape), list(strides),
                                             list(block_shape), list(offsets)))
    
    @staticmethod
    @overload
    def trans(x: _T_all_tensor, order: int, *orders: int) -> _T_all_tensor:
        ...

    @staticmethod
    @overload
    def trans(x: _T_all_tensor,
              order: Optional[Union[list[int], tuple[int, ...]]] = None,
              *orders: int) -> _T_all_tensor:
        ...

    @staticmethod
    @pfl.configure_std_func(meta_infer=Tensor._reshape_permute_meta_infer)
    def trans(x: _T_all_tensor,
              order: Optional[Union[list[int], tuple[int, ...], int]] = None,
              *orders: int) -> _T_all_tensor:
        if order is None:
            assert len(orders) == 0
            order = list(range(len(x.shape) - 1, -1, -1))
        if isinstance(x, Tensor):
            return Tensor(x._wrapped.permute(order))
        else:
            assert isinstance(
                x, PointerTensor
            ), "reshape only support Tensor and PointerTensor type"
            return PointerTensor(x._wrapped.permute(order))
    
    @staticmethod
    def _cast_meta_infer(
            fn: Callable, x: pfl.PFLExprInfo,
            dtype: pfl.PFLExprInfo) -> Optional[pfl.PFLMetaInferResult]:
        dtype_val = dtype.constexpr_data_checked
        if isinstance(dtype_val, pointer_type):
            if x.type == pfl.PFLExprType.NUMBER:
                assert isinstance(dtype._constexpr_data, pointer_type)
                res = tsim.create_pointer_scalar_meta(dtype_val.element_ty)
                if res.is_floating():
                    return pfl.PFLMetaInferResult(PointerScalarFloat(res))
                else:
                    return pfl.PFLMetaInferResult(PointerScalarInt(res))
        if x.type == pfl.PFLExprType.NUMBER:
            return None  # currently scalar don't have metadata
        assert x.has_metadata(Tensor), "cast only support Tensor type"
        x_meta = x.get_metadata_checked(Tensor)
        return pfl.PFLMetaInferResult(x_meta.to(dtype_val))

    @staticmethod
    def _cast_static_infer(x: pfl.PFLExprInfo, dtype: pfl.PFLExprInfo):
        assert dtype.has_constexpr_data(
        ), "dtype must have constexpr data for cast operation"
        if isinstance(dtype._constexpr_data, pointer_type):
            dtype_enum = tsim.DTypeEnum(dtype._constexpr_data.element_ty)
            if x.type == pfl.PFLExprType.NUMBER:
                if dtype_enum.is_floating_type():
                    return PointerScalarFloat
                else:
                    assert dtype_enum.is_integer_type()
                    return PointerScalarInt
            else:
                return PointerTensor
        else:
            dtype_v = DTypeEnum(dtype._constexpr_data)
            if x.get_origin_type_checked() is Tensor:
                return Tensor
            else:
                if dtype_v.is_floating_type():
                    return float
                elif dtype_v.is_integer_type():
                    return int
                else:
                    raise NotImplementedError(
                        f"Unsupported dtype {dtype_v} for cast operation")

    @staticmethod
    @pfl.configure_std_func(static_type_infer=_cast_static_infer,
                            meta_infer=_cast_meta_infer,
                            force_meta_infer=True)
    def cast(x: Union[int, float, Tensor], dtype: Union[int,
                                                        pointer_type]) -> Any:
        dtype_val = dtype
        if isinstance(dtype_val, pointer_type):
            if isinstance(x, (int, float)):
                assert isinstance(dtype_val, pointer_type)
                ctx = tsim.get_tensorsim_context_checked()
                assert ctx.global_mem is not None, "pointer of pointer must have global memory set."
                res = tsim.create_pointer_scalar(dtype_val.element_ty, int(x),
                                                 ctx.global_mem)
                if res.is_floating():
                    return PointerScalarFloat(res)
                else:
                    return PointerScalarInt(res)
            else:
                raise NotImplementedError(
                    "don't support Tensor to pointer cast yet.")
        else:
            if isinstance(x, Tensor):
                return x.to(dtype_val)
            else:
                dtype_v = DTypeEnum(dtype_val)
                if dtype_v.is_floating_type():
                    return float(x)
                elif dtype_v.is_integer_type():
                    return int(x)
                else:
                    raise NotImplementedError(
                        f"Unsupported dtype {dtype_v} for cast operation")
    
    @staticmethod
    def _dot_infer(fn: Callable,
                   x: pfl.PFLExprInfo,
                   y: pfl.PFLExprInfo,
                   acc: Optional[pfl.PFLExprInfo] = None,
                   **kwargs_dontcare) -> Optional[pfl.PFLMetaInferResult]:
        if acc is None:
            acc_value = None
        else:
            acc_value = acc.metadata_checked
        return pfl.PFLMetaInferResult(
            fn(x.metadata_checked, y.metadata_checked, acc_value))

    @staticmethod
    @pfl.configure_std_func(meta_infer=_dot_infer)
    def dot(x: Tensor,
            y: Tensor,
            acc: Optional[Tensor] = None,
            allow_tf32: bool = True) -> Tensor:
        res_wrapped = x._wrapped @ y._wrapped
        if acc is not None:
            acc._wrapped[:] += res_wrapped
            return acc
        return Tensor(res_wrapped)

    @staticmethod
    def argmax(input: Tensor,
               axis: int,
               tie_break_left: bool = True,
               keep_dims: bool = False) -> Tensor:
        assert isinstance(axis, int)
        return Tensor(input._wrapped.argmax(axis, keep_dims))

    @staticmethod
    def argmin(input: Tensor,
               axis: int,
               tie_break_left: bool = True,
               keep_dims: bool = False) -> Tensor:
        assert isinstance(axis, int)
        return Tensor(input._wrapped.argmin(axis, keep_dims))

    @staticmethod
    def cumsum(input: Tensor, axis: int) -> Tensor:
        return Tensor(input._wrapped.cumsum(axis))

    @staticmethod
    def debug_barrier() -> None:
        return None 

