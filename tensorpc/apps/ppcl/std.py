from functools import partial
from tensorpc.apps.ppcl.tsim import get_tensorsim_context_checked
from tensorpc.core import pfl
from typing_extensions import Self
import numpy as np 
import dataclasses 
from typing import Any, ClassVar, Optional, Type, TypeAlias, TypeVar, Union, overload
from .tensorinfer import compare_op_infer, getitem_infer, bin_op_infer, matrix_transpose_infer, tensor_create_infer
from .core import NP_DTYPE_TO_PPCL, DTypeEnum, TensorMeta
a = np.zeros([1])

pfl.register_backend("ppcl", pfl.PFLParseConfig(
    allow_var_union=False,
    allow_kw=True,
    allow_nd_slice=True,
    allow_slice=True,
    allow_new_var_after_if=True,
))

@pfl.register_pfl_std(mapped_name="int", backend="ppcl")
def int_func(x: Any) -> int:
    return int(x)

@pfl.register_pfl_std(mapped_name="float", backend="ppcl")
def float_func(x: Any) -> float:
    return float(x)

@pfl.register_pfl_std(mapped_name="bool", backend="ppcl")
def bool_func(x: Any) -> bool:
    return bool(x)

@pfl.register_pfl_std(mapped_name="range", backend="ppcl")
def range_func(start: int, stop: Optional[int] = None, step: Optional[int] = None) -> range:
    if stop is None and step is None:
        return range(start)
    elif step is None and stop is not None:
        return range(start, stop)
    else:
        assert stop is not None and step is not None, "stop and step must be provided together"
        return range(start, stop, step) 


@pfl.register_pfl_std(mapped_name="PPCLTensor", backend="ppcl")
@dataclasses.dataclass
class Tensor:
    shape: list[int]
    dtype: int
    @property 
    def ndim(self) -> int: 
        return len(self.shape)
    @property 
    def T(self) -> Self: ...
    def __getitem__(self, key: Any) -> Self: ...
    def size(self) -> int: ...
    def to(self, dtype: int) -> Self: ...

    def __add__(self, other: Union[Self, int, float]) -> Self: ...
    def __iadd__(self, other: Union[Self, int, float]) -> Self: ...
    def __radd__(self, other: Union[Self, int, float]) -> Self: ...

    def __sub__(self, other: Union[Self, int, float]) -> Self: ...
    def __isub__(self, other: Union[Self, int, float]) -> Self: ...
    def __rsub__(self, other: Union[Self, int, float]) -> Self: ...

    def __mul__(self, other: Union[Self, int, float]) -> Self: ...
    def __imul__(self, other: Union[Self, int, float]) -> Self: ...

    def __rmul__(self, other: Union[Self, int, float]) -> Self: ...

    def __truediv__(self, other: Union[Self, int, float]) -> Self: ...
    def __rtruediv__(self, other: Union[Self, int, float]) -> Self: ...
    def __itruediv__(self, other: Union[Self, int, float]) -> Self: ...

    def __floordiv__(self, other: Union[Self, int, float]) -> Self: ...
    def __rfloordiv__(self, other: Union[Self, int, float]) -> Self: ...
    def __ifloordiv__(self, other: Union[Self, int, float]) -> Self: ...

    def __mod__(self, other: Union[Self, int, float]) -> Self: ...
    def __rmod__(self, other: Union[Self, int, float]) -> Self: ...
    def __imod__(self, other: Union[Self, int, float]) -> Self: ...

    def __lt__(self, other: Union[Self, int, float]) -> Self: ...
    def __le__(self, other: Union[Self, int, float]) -> Self: ...
    def __ge__(self, other: Union[Self, int, float]) -> Self: ...
    def __gt__(self, other: Union[Self, int, float]) -> Self: ...

    def __eq__(self, other: Any) -> bool: ...
    def __ne__(self, other: Any) -> bool: ...

    def __and__(self, other: Self) -> Self: ...
    def __xor__(self, other: Self) -> Self: ...
    def __or__(self, other: Self) -> Self: ...

pfl.mark_meta_infer(Tensor.__getitem__)(getitem_infer)
pfl.mark_meta_infer(Tensor.__add__)(bin_op_infer)
pfl.mark_meta_infer(Tensor.__radd__)(bin_op_infer)
pfl.mark_meta_infer(Tensor.__iadd__)(bin_op_infer)

pfl.mark_meta_infer(Tensor.__sub__)(bin_op_infer)
pfl.mark_meta_infer(Tensor.__rsub__)(bin_op_infer)
pfl.mark_meta_infer(Tensor.__isub__)(bin_op_infer)

pfl.mark_meta_infer(Tensor.__mul__)(bin_op_infer)
pfl.mark_meta_infer(Tensor.__rmul__)(bin_op_infer)
pfl.mark_meta_infer(Tensor.__imul__)(bin_op_infer)

pfl.mark_meta_infer(Tensor.__truediv__)(bin_op_infer)
pfl.mark_meta_infer(Tensor.__rtruediv__)(bin_op_infer)
pfl.mark_meta_infer(Tensor.__itruediv__)(bin_op_infer)

pfl.mark_meta_infer(Tensor.__floordiv__)(bin_op_infer)
pfl.mark_meta_infer(Tensor.__rfloordiv__)(bin_op_infer)
pfl.mark_meta_infer(Tensor.__ifloordiv__)(bin_op_infer)

pfl.mark_meta_infer(Tensor.__mod__)(bin_op_infer)
pfl.mark_meta_infer(Tensor.__rmod__)(bin_op_infer)
pfl.mark_meta_infer(Tensor.__imod__)(bin_op_infer)

pfl.mark_meta_infer(Tensor.__and__)(bin_op_infer)
pfl.mark_meta_infer(Tensor.__xor__)(bin_op_infer)
pfl.mark_meta_infer(Tensor.__or__)(bin_op_infer)

pfl.mark_meta_infer(Tensor.__lt__)(compare_op_infer)
pfl.mark_meta_infer(Tensor.__le__)(compare_op_infer)
pfl.mark_meta_infer(Tensor.__ge__)(compare_op_infer)
pfl.mark_meta_infer(Tensor.__gt__)(compare_op_infer)
# pfl.mark_meta_infer(Tensor.__eq__)(compare_op_infer_eqne)
# pfl.mark_meta_infer(Tensor.__ne__)(compare_op_infer_eqne)


pfl.mark_meta_infer(Tensor.T)(matrix_transpose_infer)

@pfl.mark_meta_infer(Tensor.to)
def __to_infer(this: pfl.PFLExprInfo, dtype: pfl.PFLExprInfo) -> Optional[pfl.PFLMetaInferResult]:
    new_meta = dataclasses.replace(this.metadata_checked, dtype=dtype.metadata_checked)
    return pfl.PFLMetaInferResult(new_meta)

@pfl.register_pfl_std(mapped_name="PPCLPointerTensor", backend="ppcl")
@dataclasses.dataclass
class PointerTensor:
    shape: list[int]
    # pointer tensor dtype indicates the type of the pointer, not the data type
    dtype: int
    @property 
    def ndim(self) -> int: 
        return len(self.shape)
    @property 
    def T(self) -> Self: ...
    def __getitem__(self, key: Any) -> Self: ...
    def size(self) -> int: ...
    # pointer tensor only support add and sub operations
    def __add__(self, other: Union[Tensor, int]) -> Self: ...
    def __sub__(self, other: Union[Tensor, int]) -> Self: ...
    def __radd__(self, other: Union[Tensor, int]) -> Self: ...
    def __rsub__(self, other: Union[Tensor, int]) -> Self: ...
    def __iadd__(self, other: Union[Tensor, int]) -> Self: ...
    def __isub__(self, other: Union[Tensor, int]) -> Self: ...

pfl.mark_meta_infer(PointerTensor.__add__)(bin_op_infer)
pfl.mark_meta_infer(PointerTensor.__sub__)(bin_op_infer)
pfl.mark_meta_infer(PointerTensor.__radd__)(bin_op_infer)
pfl.mark_meta_infer(PointerTensor.__rsub__)(bin_op_infer)
pfl.mark_meta_infer(PointerTensor.__iadd__)(bin_op_infer)
pfl.mark_meta_infer(PointerTensor.__isub__)(bin_op_infer)

pfl.mark_meta_infer(PointerTensor.__getitem__)(getitem_infer)
pfl.mark_meta_infer(PointerTensor.T)(matrix_transpose_infer)


# pfl.mark_meta_infer(Pointer.__add__)(bin_op_infer)
# pfl.mark_meta_infer(Pointer.__sub__)(bin_op_infer)
# def bin_op_infer(this: pfl.PFLExprInfo, data: pfl.PFLExprInfo) -> Optional[pfl.PFLMetaInferResult]:
#     if this.has_metadata():
#         if not data.has_metadata() and data == this:
#             return
#         this_meta = this.get_metadata_checked(TensorMetaBase)
#         shape_this = this_meta.shape
#         if data == this:
#             data_meta = data.get_metadata_checked(TensorMetaBase)
#             shape_other = data_meta.shape
#             dtype_other = data_meta.dtype
#         else:
#             assert data.type == pfl.PFLExprType.NUMBER or data.type == pfl.PFLExprType.BOOL
#             shape_other = []
#             dtype_other = this_meta.get_default_dtype_from_pfl(data)
#         shape_res = np.broadcast_shapes(shape_this, shape_other)
#         res_meta = this_meta.__class__(
#             shape=list(shape_res),
#             dtype=this_meta.dtype_promotion(this_meta.dtype, dtype_other)
#         )
#         return pfl.PFLMetaInferResult(res_meta)

@overload
def min_fn(x: int, y: int) -> int: ...
@overload
def min_fn(x: float, y: float) -> float: ...
@overload
def min_fn(x: Tensor, y: Union[int, float]) -> Tensor: ...
@overload
def min_fn(x: Union[int, float], y: Tensor) -> Tensor: ...

@pfl.register_pfl_std(mapped_name="min", backend="ppcl")
def min_fn(x: Union[Tensor, int, float], y: Union[Tensor, int, float]) -> Union[Tensor, int, float]: ...

@overload
def max_fn(x: int, y: int) -> int: ...
@overload
def max_fn(x: float, y: float) -> float: ...
@overload
def max_fn(x: Tensor, y: Union[int, float]) -> Tensor: ...
@overload
def max_fn(x: Union[int, float], y: Tensor) -> Tensor: ...

@pfl.register_pfl_std(mapped_name="max", backend="ppcl")
def max_fn(x: Union[Tensor, int, float], y: Union[Tensor, int, float]) -> Union[Tensor, int, float]: ...


_T_math = TypeVar("_T_math", bound=Union[Tensor, int, float])

@pfl.register_pfl_std(mapped_name="ppcl", backend="ppcl")
@dataclasses.dataclass
class ppcl:
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
    def program_id(axis: int) -> int: 
        ctx = get_tensorsim_context_checked()
        return ctx.grid_size[axis]

    @staticmethod
    def array(data: list[Any]) -> Tensor: ...

    @staticmethod
    def zeros(shape: list[int], dtype: int) -> Tensor: ...

    @staticmethod
    def ones(shape: list[int], dtype: int) -> Tensor: ...

    @staticmethod
    def arange(start: int, end: int) -> Tensor: ...

    @staticmethod
    def load(pointer: PointerTensor, mask: Optional[Union[Tensor, bool]] = None, other: Optional[Union[int, float, Tensor]] = None) -> Tensor: ...
    
    @staticmethod
    def store(pointer: PointerTensor, value: Tensor, mask: Optional[Tensor] = None, other: Optional[Tensor] = None) -> None: ...

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
    def cdiv(x: Union[Tensor, int, float], div: Union[Tensor, int, float]) -> Union[int, float, Tensor]: ...

    @staticmethod
    def dot(x: Tensor, y: Tensor, acc: Optional[Tensor] = None, allow_tf32: bool = True) -> Tensor: ...
    
    @staticmethod
    def argmax(input: Tensor, axis: int, tie_break_left: bool = True, keep_dims: bool = False) -> Tensor: ...

    @staticmethod
    def argmin(input: Tensor, axis: int, tie_break_left: bool = True, keep_dims: bool = False) -> Tensor: ...

    @staticmethod
    def max(input: Tensor, axis: int, keep_dims: bool = False) -> Tensor: ...

    @staticmethod
    def min(input: Tensor, axis: int, keep_dims: bool = False) -> Tensor: ...

    # @staticmethod
    # def max_with_indices(input: Tensor, axis: int, return_indices_tie_break_left: bool = True, keep_dims: bool = False) -> Tensor: ...

    # @staticmethod
    # def min_with_indices(input: Tensor, axis: int, return_indices_tie_break_left: bool = True, keep_dims: bool = False) -> Tensor: ...

    @staticmethod
    def sum(input: Tensor, axis: int, keep_dims: bool = False, dtype: Optional[int] = None) -> Tensor: ...

    @staticmethod
    def where(cond: Tensor, x: Union[Tensor, int, float], y: Union[Tensor, int, float]) -> Tensor: ...

    @staticmethod
    def abs(x: Union[Tensor, int, float]) -> Tensor: ...

    @staticmethod
    def ceil(x: Union[Tensor, int, float]) -> Tensor: ...

    @staticmethod
    def cos(x: Union[Tensor, int, float]) -> Tensor: ...

    @staticmethod
    def erf(x: Union[Tensor, int, float]) -> Tensor: ...

    @staticmethod
    def div_rn(x: Union[Tensor, int, float], y: Union[Tensor, int, float]) -> Tensor: ...

    @staticmethod
    def fdiv(x: Union[Tensor, int, float], y: Union[Tensor, int, float]) -> Tensor: ...

    @staticmethod
    def fma(x: Union[Tensor, int, float], y: Union[Tensor, int, float], z: Union[Tensor, int, float]) -> Tensor: ...

    @staticmethod
    def exp(x: Union[Tensor, int, float]) -> Tensor: ...

    @staticmethod
    def exp2(x: Union[Tensor, int, float]) -> Tensor: ...

    @staticmethod
    def floor(x: Union[Tensor, int, float]) -> Tensor: ...

    @staticmethod
    def log(x: Union[Tensor, int, float]) -> Tensor: ...

    @staticmethod
    def log2(x: Union[Tensor, int, float]) -> Tensor: ...

    @staticmethod
    def rsqrt(x: Union[Tensor, int, float]) -> Tensor: ...

    @staticmethod
    def sqrt(x: Union[Tensor, int, float]) -> Tensor: ...

    @staticmethod
    def sqrt_rn(x: Union[Tensor, int, float]) -> Tensor: ...

    @staticmethod
    def sigmoid(x: Union[Tensor, int, float]) -> Tensor: ...

    @staticmethod
    def sin(x: Union[Tensor, int, float]) -> Tensor: ...

    @staticmethod
    def softmax(x: Union[Tensor, int, float]) -> Tensor: ...

    @staticmethod
    def maximum(x: Union[Tensor, int, float], y: Union[Tensor, int, float]) -> Tensor: ...

    @staticmethod
    def minimim(x: Union[Tensor, int, float], y: Union[Tensor, int, float]) -> Tensor: ...

# pfl.mark_meta_infer(ppcl.array)(partial(tensor_create_infer, base_cls=TensorMeta))
pfl.mark_meta_infer(ppcl.zeros)(partial(tensor_create_infer, base_cls=TensorMeta))
pfl.mark_meta_infer(ppcl.ones)(partial(tensor_create_infer, base_cls=TensorMeta))

# @pfl.mark_meta_infer(ppcl.array)
# def __array_infer(data: pfl.PFLExprInfo) -> Optional[pfl.PFLMetaInferResult]:
#     res = np.array(data.metadata_checked)
#     return pfl.PFLMetaInferResult(TensorMeta(
#         shape=list(res.shape),
#         dtype=NP_DTYPE_TO_PPCL[res.dtype],
#     ))

@pfl.mark_meta_infer(ppcl.arange)
def __arange_infer(start: pfl.PFLExprInfo, end: pfl.PFLExprInfo) -> Optional[pfl.PFLMetaInferResult]:
    return pfl.PFLMetaInferResult(TensorMeta(
        shape=[end.metadata_checked - start.metadata_checked],
        dtype=DTypeEnum.int32,
        is_pointer=False,
    ))

@pfl.mark_meta_infer(ppcl.load)
def __load_infer(pointer: pfl.PFLExprInfo, mask: pfl.PFLExprInfo, other: pfl.PFLExprInfo) -> Optional[pfl.PFLMetaInferResult]:
    meta = pointer.get_metadata_checked(TensorMeta)
    assert meta.num_element == 1, "not implemented"
    return pfl.PFLMetaInferResult(TensorMeta(
        shape=meta.shape,
        dtype=meta.dtype,
        is_pointer=False,
    ))

@pfl.mark_meta_infer(ppcl.store)
def __store_infer(pointer: pfl.PFLExprInfo, value: pfl.PFLExprInfo, mask: Optional[pfl.PFLExprInfo] = None, other: Optional[pfl.PFLExprInfo] = None) -> Optional[pfl.PFLMetaInferResult]:
    return 

