from collections.abc import Sequence
import enum
from functools import partial
from typing import Any, Optional, Type, Union, cast, overload
import dataclasses
import numpy as np 
from tensorpc.core import pfl
import contextlib 
import contextvars
from typing_extensions import Self

from tensorpc.core.pfl.pfl_ast import BinOpType, CompareType, UnaryOpType
from tensorpc.apps.ppcl.tsim.core import DTypeEnum, PPCL_TO_NP_DTYPE, get_default_base_dtype, get_default_float_dtype, get_default_int_dtype, get_tensorsim_context, NumpyReduceType, NP_DTYPE_TO_PPCL

@dataclasses.dataclass
class SimTensorStorage:
    data: np.ndarray
    # multi-level io. e.g. load from global and store to shared memory, 
    # then load from shared memory.
    # both global indices and shared indices will be stored here.
    indices: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    
    def __post_init__(self):
        assert not isinstance(self.data, np.number)

    def getitem(self, inds: Any) -> Self:
        new_data = self.data[inds]
        new_storage = dataclasses.replace(self, data=new_data)
        # handle indices
        if isinstance(inds, tuple):
            # only allow one ellipsis
            ellipsis_found = False
            none_cnt = 0
            for item in inds:
                if item is ...:
                    if ellipsis_found:
                        raise ValueError("only one ellipsis is allowed in indices")
                    ellipsis_found = True
                if item is None:
                    none_cnt += 1

            new_storage_inds = {}
            for k, indices in new_storage.indices.items():
                new_slices = [slice(None)] * (indices.ndim - self.data.ndim)
                if ellipsis_found:
                    new_inds = (*inds, *new_slices) 
                else:
                    new_inds = (*inds, ..., *new_slices) 
                new_storage_inds[k] = indices[new_inds]
            new_storage.indices = new_storage_inds
        else:
            new_storage.indices = {
                k: x[inds] for k, x in new_storage.indices.items()
            }
        return new_storage

    def setitem(self, inds: Any, value: Union[Self, int, float, bool]):
        if isinstance(value, SimTensorStorage):
            self.data[inds] = value.data
        else:
            # impossible to clear part of indices
            self.data[inds] = value

        if isinstance(inds, tuple):
            # only allow one ellipsis
            ellipsis_found = False
            none_cnt = 0
            for item in inds:
                if item is ...:
                    if ellipsis_found:
                        raise ValueError("only one ellipsis is allowed in indices")
                    ellipsis_found = True
                if item is None:
                    none_cnt += 1

            for k, indices in self.indices.items():
                new_slices = [slice(None)] * (indices.ndim - self.data.ndim)
                if ellipsis_found:
                    new_inds = (*inds, *new_slices) 
                else:
                    new_inds = (*inds, ..., *new_slices) 
                if isinstance(value, SimTensor):
                    assert value.storage is not None 
                    self.indices[k][new_inds] = value.storage.indices[k]
                else:
                    self.indices[k][new_inds] = -1
        else:
            for k, indices in self.indices.items():
                if isinstance(value, SimTensor):
                    assert value.storage is not None 
                    self.indices[k][inds] = value.storage.indices[k]
                else:
                    self.indices[k][inds] = -1

    def broadcast(self, new_data: np.ndarray, other_storage: Self) -> Self:
        new_storage = dataclasses.replace(self, data=new_data) 
        assert len(self.indices) == len(other_storage.indices), \
            "Cannot broadcast storage with different number of indices"
        new_storage.indices = {}
        for k, indices in self.indices.items():
            other_indices = other_storage.indices[k]
            if indices.ndim < other_indices.ndim:
                indices = indices[tuple([...] + [None] * (other_indices.ndim - indices.ndim))]
            elif indices.ndim > other_indices.ndim:
                other_indices = other_indices[tuple([...] + [None] * (indices.ndim - other_indices.ndim))]
            res_indices = np.broadcast(indices, other_indices)
            new_storage.indices[k] = (cast(np.ndarray, res_indices))
        return new_storage

    def reduce(self, new_data: np.ndarray, axes: list[int], keepdims: bool) -> Self:
        new_storage = dataclasses.replace(self, data=new_data) 
        new_storage.indices = {}
        new_shape: list[int] = []
        permute_inds: list[int] = []
        for i, dim in enumerate(self.data.shape):
            if i in axes:
                if keepdims:
                    new_shape.append(1)
            else:
                new_shape.append(dim)
                permute_inds.append(i)
        old_ndim = self.data.ndim
        for k, indices in self.indices.items():
            permute_inds_cur = permute_inds.copy()
            pure_inds_ndim = self.data.ndim - indices.ndim
            permute_inds_cur.extend(c + old_ndim for c in range(pure_inds_ndim))
            permute_inds_cur.extend(axes)
            new_indices = indices.transpose(permute_inds_cur)
            new_indices_shape = list(new_data.shape) + list(new_indices.shape)[-len(axes) - pure_inds_ndim:]
            new_indices = new_indices.reshape(new_indices_shape) 
            new_storage.indices[k] = (cast(np.ndarray, new_indices))
        new_storage.data = new_data
        return new_storage

@dataclasses.dataclass
class SimTensorBase:
    """
    A CPU/Meta Tensor that can be used for computing simulation or metadata inference.
    
    """
    shape: list[int]
    dtype: int
    # if storage is None, only meta inference is supported.
    storage: Optional[SimTensorStorage] = None

    def get_storage_checked(self) -> SimTensorStorage:
        assert self.storage is not None 
        return self.storage 

    @staticmethod
    def dtype_promotion(*args: int):
        return DTypeEnum.dtype_promotion(*args)

    def is_floating(self) -> bool:
        return DTypeEnum(self.dtype).is_floating_type()

    def is_unsigned(self) -> bool:
        return DTypeEnum(self.dtype).is_unsigned_type()

    def is_integer(self) -> bool:
        return DTypeEnum(self.dtype).is_integer_type()

    def is_boolean(self) -> bool:
        return DTypeEnum(self.dtype).is_boolean_type()

    def is_pointer(self) -> bool:
        return False

    def get_pointer_num_elements(self) -> int:
        return 1

    def bit_size(self) -> int:
        return DTypeEnum(self.dtype).bit_size()

    @staticmethod
    def dtype_to_np(dtype: int) -> np.dtype:
        return PPCL_TO_NP_DTYPE[DTypeEnum(dtype)]

    def is_scalar(self) -> bool:
        return len(self.shape) == 0

    def _replace_data(self, new_data: np.ndarray) -> Self:
        assert self.storage is not None, "Cannot replace data of a tensor without storage"
        assert list(new_data.shape) == self.shape, \
            f"New data shape {new_data.shape} does not match tensor shape {self.shape}"
        assert new_data.dtype == self.dtype_to_np(self.dtype), \
            f"New data dtype {new_data.dtype} does not match tensor dtype {self.dtype_to_np(self.dtype)}" 
        new_storage = dataclasses.replace(self.storage, data=new_data)
        res = dataclasses.replace(self, shape=list(map(int, new_data.shape)), storage=new_storage)
        return res

    def __getitem__(self, inds: Any) -> Self:
        if self.storage is None:
            if not isinstance(inds, tuple):
                inds = (inds,)
            # tuple of slices
            # from pytorch
            dim = 0
            specified_dims = 0
            for item in inds:
                if item is None or item is ...:
                    specified_dims += 1
            res_shape = self.shape.copy()
            for item in inds:
                if item is ...:
                    dim += len(self.shape) - specified_dims
                elif isinstance(item, slice):
                    slice_obj = item
                    start = 0 if slice_obj.start is None else slice_obj.start
                    stop = res_shape[dim] if slice_obj.stop is None else slice_obj.stop
                    step = 1 if slice_obj.step is None else slice_obj.step
                    step_abs = abs(step)
                    if (start < 0):
                        start += res_shape[dim]
                    if (stop < 0):
                        stop += res_shape[dim]
                    length = stop - start 
                    res_dim = (length + step_abs - 1) // step_abs
                    res_shape[dim] = res_dim
                    dim += 1
                elif isinstance(item, int):
                    res_shape.pop(dim)
                elif item is None:
                    res_shape.insert(dim, 1)
                    dim += 1
                else:
                    raise NotImplementedError(f"Unsupported slice type: {type(item)}")
            res = dataclasses.replace(self, shape=res_shape)
            return res 
        else:
            new_storage = self.storage.getitem(inds) 
            res = dataclasses.replace(self, shape=list(map(int, new_storage.data.shape)))
            res.storage = new_storage
            return res 

    def __setitem__(self, inds: Any, value: Union[Self, int, float, bool]):
        # only needed when storage is not None 
        if self.storage is not None:
            if isinstance(value, SimTensorBase):
                assert value.storage is not None, "value must have storage"
                self.storage.setitem(inds, value.storage)
            else:
                self.storage.setitem(inds, value)

    @property 
    def T(self) -> Self:
        if self.storage is None:
            # only meta inference is supported
            new_shape = self.shape[::-1]
            res = dataclasses.replace(self, shape=new_shape)
            return res
        else:
            new_data = self.storage.data.T
            new_storage = dataclasses.replace(self.storage, data=new_data)
            res = dataclasses.replace(self, shape=list(map(int, new_data.shape)))
            res.storage = new_storage
            return res

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def to(self, dtype: int) -> Self:
        if self.storage is None:
            return dataclasses.replace(self, dtype=dtype)
        if self.dtype == dtype:
            return dataclasses.replace(self)
        new_data = self.storage.data.astype(PPCL_TO_NP_DTYPE[DTypeEnum(dtype)])
        new_storage = dataclasses.replace(self.storage, data=new_data)
        res = dataclasses.replace(self, shape=list(map(int, new_data.shape)), dtype=dtype)
        res.storage = new_storage
        return res

    def reshape(self, new_shape: Union[Sequence[int], int], *shapes: int) -> Self:
        if isinstance(new_shape, int):
            new_shape_list = [new_shape, *shapes]
        else:
            new_shape_list = list(new_shape)
        # calc new shape, find -1 first
        found_idx = -1
        total_prod = 1
        total_prod_no_minus_one = 1
        for i, dim in enumerate(new_shape_list):
            if dim == -1:
                if found_idx >= 0:
                    raise ValueError("Only one -1 is allowed in new shape")
                found_idx = i
            else:
                total_prod_no_minus_one *= dim
            total_prod *= dim 
        if found_idx >= 0:
            assert total_prod_no_minus_one > 0, "don't support reshape with zero fornow."
            assert total_prod % total_prod_no_minus_one == 0, \
                f"Cannot reshape tensor with shape {self.shape} to {new_shape_list}, "
            new_shape_list[found_idx] = total_prod // total_prod_no_minus_one
        if self.storage is None:
            return dataclasses.replace(self, shape=list(map(int, new_shape_list)))
        new_data = self.storage.data.reshape(new_shape)
        new_storage = dataclasses.replace(self.storage, data=new_data)
        res = dataclasses.replace(self, shape=list(map(int, new_data.shape)))
        res.storage = new_storage
        return res

    def _unary_base(self, op_type: UnaryOpType) -> Self:
        if self.storage is None:
            return dataclasses.replace(self, shape=list(map(int, self.shape)))
        if op_type == UnaryOpType.UADD:
            new_data = +self.storage.data
        elif op_type == UnaryOpType.USUB:
            new_data = -self.storage.data
        elif op_type == UnaryOpType.NOT:
            new_data = ~self.storage.data
        elif op_type == UnaryOpType.INVERT:
            new_data = np.invert(self.storage.data)
        else:
            raise ValueError(f"Unsupported unary operation: {op_type}")
        new_storage = dataclasses.replace(self.storage, data=new_data)
        res = dataclasses.replace(self, shape=list(map(int, new_data.shape)))
        res.storage = new_storage
        return res

    def __pos__(self) -> Self:
        return self._unary_base(UnaryOpType.UADD)
        
    def __neg__(self) -> Self:
        return self._unary_base(UnaryOpType.USUB)

    def __invert__(self) -> Self:
        return self._unary_base(UnaryOpType.INVERT)

    def __not__(self) -> Self:
        return self._unary_base(UnaryOpType.NOT)

    def _binary_base(self, other: Union[Self, int, float, bool], op_type: BinOpType, is_reversed: bool, is_inplace: bool = False) -> Self:
        # TODO numpy dtype promotion is different from triton?
        is_pointer = self.is_pointer()
        pointer_dtype = self.dtype
        res_replace_tgt = self
        if isinstance(other, SimTensorBase):
            assert not (self.is_pointer() and other.is_pointer()), "Cannot perform binary operation between two pointer tensors"
            if self.is_pointer():
                assert other.is_integer() or other.is_unsigned(), "Pointer tensor can only be operated with integer or unsigned tensor"
            is_pointer |= other.is_pointer()
            if other.is_pointer():
                pointer_dtype = other.dtype
                res_replace_tgt = other
                assert self.is_integer() or self.is_unsigned(), "Pointer tensor can only be operated with integer or unsigned tensor"

        if is_pointer:
            assert op_type in (BinOpType.ADD, BinOpType.SUB), "Pointer tensors can only be added or subtracted"
        if self.storage is None:
            if isinstance(other, SimTensorBase):
                if op_type == BinOpType.MATMUL:
                    assert len(self.shape) >= 2 and len(other.shape) >= 2
                    new_shape_no_mm = np.broadcast_shapes(self.shape[:-2], other.shape[:-2])
                    assert self.shape[-1] == other.shape[-2], "matmul shape mismatch"
                    new_shape = list(new_shape_no_mm) + [self.shape[-2], other.shape[-1]]
                else:
                    new_shape = np.broadcast_shapes(self.shape, other.shape)
                if is_pointer:
                    new_dtype = pointer_dtype
                else:
                    new_dtype = self.dtype_promotion(self.dtype, other.dtype)
            else:
                # TODO : handle scalar dtype
                if isinstance(other, int):
                    other_dtype = get_default_int_dtype()
                elif isinstance(other, int):
                    other_dtype = get_default_float_dtype()
                elif isinstance(other, bool):
                    other_dtype = DTypeEnum.bool_
                else:
                    raise NotImplementedError
                new_shape = self.shape
                new_dtype = self.dtype_promotion(self.dtype, other_dtype)
            return dataclasses.replace(res_replace_tgt, shape=list(map(int, new_shape)), dtype=new_dtype)
        if isinstance(other, SimTensorBase):
            assert other.storage is not None 
            other_data = other.storage.data
            # new_dtype = self.dtype_promotion(self.dtype, other.dtype)
        else:
            # new_dtype = self.dtype
            other_data = other
        self_data = self.storage.data
        if is_reversed:
            self_data, other_data = other_data, self_data
        assert isinstance(self_data, np.ndarray) or isinstance(other_data, np.ndarray), f"{type(self_data)}, {type(other_data)}"
        if op_type == BinOpType.ADD:
            new_data = self_data + other_data 
        elif op_type == BinOpType.SUB:
            new_data = self_data - other_data
        elif op_type == BinOpType.MULT:
            new_data = self_data * other_data
        elif op_type == BinOpType.DIV:
            new_data = self_data / other_data
        elif op_type == BinOpType.FLOOR_DIV:
            new_data = self_data // other_data
        elif op_type == BinOpType.POW:
            new_data = np.power(self_data, other_data)
        elif op_type == BinOpType.MOD:
            new_data = self_data % other_data
        elif op_type == BinOpType.LSHIFT:
            new_data = np.left_shift(self_data, other_data)
        elif op_type == BinOpType.RSHIFT:
            new_data = np.right_shift(self_data, other_data)
        elif op_type == BinOpType.MATMUL:
            assert isinstance(self_data, np.ndarray) and isinstance(other_data, np.ndarray)
            new_data = self_data @ other_data
        elif op_type == BinOpType.BIT_AND:
            new_data = self_data & other_data # type: ignore
        elif op_type == BinOpType.BIT_OR:
            new_data = self_data | other_data # type: ignore
        elif op_type == BinOpType.BIT_XOR:
            new_data = self_data ^ other_data # type: ignore
        else:
            raise ValueError(f"Unsupported binary operation: {op_type}")
        if isinstance(new_data, np.number):
            new_data = np.array(new_data)
        assert isinstance(new_data, np.ndarray)
        if is_inplace:
            assert isinstance(self_data, np.ndarray)
            self_data[:] = new_data
            new_data = self_data
            return self 
        else:
            if is_pointer:
                new_storage = dataclasses.replace(res_replace_tgt.get_storage_checked(), data=new_data.astype(np.int64), indices={}) 
                res = dataclasses.replace(res_replace_tgt, shape=list(map(int, new_data.shape)), dtype=pointer_dtype, storage=new_storage)
            else:
                new_storage = dataclasses.replace(self.storage, data=new_data, indices={})
                res = dataclasses.replace(self, shape=list(map(int, new_data.shape)), dtype=NP_DTYPE_TO_PPCL[new_data.dtype.type])
                res.storage = new_storage
            return res

    def _compare_base(self, other: Union[Self, int, float, bool], op_type: CompareType) -> Self:
        if self.storage is None:
            if isinstance(other, SimTensorBase):
                new_shape = np.broadcast_shapes(self.shape, other.shape)
                new_dtype = DTypeEnum.bool_
            else:
                # TODO : handle scalar dtype
                new_shape = self.shape
                new_dtype = self.dtype
            return dataclasses.replace(self, shape=list(map(int, new_shape)), dtype=new_dtype)
        if isinstance(other, SimTensorBase):
            assert other.storage is not None 
            new_shape = np.broadcast_shapes(self.shape, other.shape)
            other_data = other.storage.data
            # new_dtype = self.dtype_promotion(self.dtype, other.dtype)
        else:
            new_shape = self.shape
            # new_dtype = self.dtype
            other_data = other
        self_data = self.storage.data
        assert isinstance(self_data, np.ndarray) or isinstance(other_data, np.ndarray)
        if op_type == CompareType.EQUAL:
            new_data = self_data == other_data 
        elif op_type == CompareType.NOT_EQUAL:
            new_data = self_data != other_data
        elif op_type == CompareType.GREATER:
            new_data = self_data > other_data
        elif op_type == CompareType.GREATER_EQUAL:
            new_data = self_data >= other_data
        elif op_type == CompareType.LESS:
            new_data = self_data < other_data
        elif op_type == CompareType.LESS_EQUAL:
            new_data = self_data <= other_data
        else:
            raise ValueError(f"Unsupported compare operation: {op_type}")
        assert isinstance(new_data, np.ndarray)
        new_storage = dataclasses.replace(self.storage, data=new_data)
        res = dataclasses.replace(self, shape=list(map(int, new_data.shape)), dtype=NP_DTYPE_TO_PPCL[new_data.dtype.type])
        res.storage = new_storage
        return res

@dataclasses.dataclass
class SimTensor(SimTensorBase):
    """
    A CPU/Meta Tensor that can be used for computing simulation or metadata inference.
    
    """

    def _reduce_meta_only(self, axes: Optional[Sequence[int]] = None, keepdims: bool = False) -> Self:
        if axes is None:
            axes = list(range(len(self.shape)))
        # only meta inference is supported
        new_shape: list[int] = []
        for i, dim in enumerate(self.shape):
            if i not in axes:
                new_shape.append(dim)
            elif keepdims:
                new_shape.append(1)
        res = dataclasses.replace(self, shape=new_shape)
        return res

    def _reduce_base(self, rtype: NumpyReduceType, axis: Optional[Union[Sequence[int], int]] = None, keepdims: bool = False) -> Self:
        if rtype == NumpyReduceType.ARGMAX or rtype == NumpyReduceType.ARGMIN:
            assert isinstance(axis, int), "axis must be an int for argmax/argmin"
        axis_is_none = False
        if axis is None:
            axis_is_none = True
            axis = tuple(range(len(self.shape)))
        elif isinstance(axis, int):
            axis = tuple([axis])
        elif isinstance(axis, Sequence):
            axis = tuple(axis)
        if self.storage is None:
            return self._reduce_meta_only(axis, keepdims)
        else:
            if rtype == NumpyReduceType.SUM:
                new_data = self.storage.data.sum(axis=axis, keepdims=keepdims)
            elif rtype == NumpyReduceType.MEAN:
                new_data = self.storage.data.mean(axis=axis, keepdims=keepdims)
            elif rtype == NumpyReduceType.MAX:
                new_data = self.storage.data.max(axis=axis, keepdims=keepdims)
            elif rtype == NumpyReduceType.MIN:
                new_data = self.storage.data.min(axis=axis, keepdims=keepdims)
            elif rtype == NumpyReduceType.PROD:
                new_data = self.storage.data.prod(axis=axis, keepdims=keepdims)
            elif rtype == NumpyReduceType.ARGMAX or rtype == NumpyReduceType.ARGMIN:
                if axis_is_none:
                    if rtype == NumpyReduceType.ARGMAX:
                        new_data = self.storage.data.reshape(-1).argmax(keepdims=keepdims) 
                    else:
                        new_data = self.storage.data.reshape(-1).argmin(keepdims=keepdims)
                else:
                    if rtype == NumpyReduceType.ARGMAX:
                        new_data = self.storage.data.argmax(axis=axis[0], keepdims=keepdims) 
                    else:
                        new_data = self.storage.data.argmin(axis=axis[0], keepdims=keepdims)
            else:
                raise ValueError(f"Unsupported reduce type: {rtype}")
            if isinstance(new_data, np.number):
                new_data = np.array(new_data)
            new_storage = self.storage.reduce(new_data, list(axis), keepdims)
            res = dataclasses.replace(self, shape=list(map(int, new_data.shape)))
            res.storage = new_storage
            return res

    def sum(self, axis: Optional[Union[Sequence[int], int]] = None, keepdims: bool = False) -> Self:
        return self._reduce_base(NumpyReduceType.SUM, axis, keepdims)

    def mean(self, axis: Optional[Union[Sequence[int], int]] = None, keepdims: bool = False) -> Self:
        return self._reduce_base(NumpyReduceType.MEAN, axis, keepdims)

    def max(self, axis: Optional[Union[Sequence[int], int]] = None, keepdims: bool = False) -> Self:
        return self._reduce_base(NumpyReduceType.MAX, axis, keepdims)

    def min(self, axis: Optional[Union[Sequence[int], int]] = None, keepdims: bool = False) -> Self:
        return self._reduce_base(NumpyReduceType.MIN, axis, keepdims)

    def prod(self, axis: Optional[Union[Sequence[int], int]] = None, keepdims: bool = False) -> Self:
        return self._reduce_base(NumpyReduceType.PROD, axis, keepdims)

    def argmax(self, axis: Optional[int] = None, keepdims: bool = False) -> Self:
        return self._reduce_base(NumpyReduceType.ARGMAX, axis, keepdims)
        
    def argmin(self, axis: Optional[int] = None, keepdims: bool = False) -> Self:
        return self._reduce_base(NumpyReduceType.ARGMIN, axis, keepdims)
        
    def __lt__(self, other: Union[Self, int, float]) -> Self:
        return self._compare_base(other, CompareType.LESS)
    def __le__(self, other: Union[Self, int, float]) -> Self:
        return self._compare_base(other, CompareType.LESS_EQUAL)
    def __ge__(self, other: Union[Self, int, float]) -> Self:
        return self._compare_base(other, CompareType.GREATER_EQUAL)
    def __gt__(self, other: Union[Self, int, float]) -> Self:
        return self._compare_base(other, CompareType.GREATER)

    @overload
    def __eq__(self, other: Self) -> Self: ...
    @overload
    def __eq__(self, other: Union[int, float]) -> Self: ...

    @overload
    def __ne__(self, other: Self) -> Self: ...
    @overload
    def __ne__(self, other: Union[int, float]) -> Self: ...

    def __eq__(self, other: Any) -> Any:
        assert  isinstance(other, (SimTensorBase, int, float))
        return self._compare_base(cast(Self, other), CompareType.EQUAL)
    
    def __ne__(self, other: Any) -> Any:
        assert  isinstance(other, (SimTensorBase, int, float))
        return self._compare_base(cast(Self, other), CompareType.NOT_EQUAL)

    def __add__(self, other: Union[Self, int, float]) -> Self: 
        return self._binary_base(other, BinOpType.ADD, False)
    def __iadd__(self, other: Union[Self, int, float]) -> Self: 
        return self._binary_base(other, BinOpType.ADD, False, True)
    def __radd__(self, other: Union[Self, int, float]) -> Self:
        return self._binary_base(other, BinOpType.ADD, True)

    def __sub__(self, other: Union[Self, int, float]) -> Self:
        return self._binary_base(other, BinOpType.SUB, False)
    def __isub__(self, other: Union[Self, int, float]) -> Self:
        return self._binary_base(other, BinOpType.SUB, False, True)
    def __rsub__(self, other: Union[Self, int, float]) -> Self:
        return self._binary_base(other, BinOpType.SUB, True)

    def __mul__(self, other: Union[Self, int, float]) -> Self:
        return self._binary_base(other, BinOpType.MULT, False)
    def __imul__(self, other: Union[Self, int, float]) -> Self:
        return self._binary_base(other, BinOpType.MULT, False, True)

    def __rmul__(self, other: Union[Self, int, float]) -> Self:
        return self._binary_base(other, BinOpType.MULT, True)

    def __truediv__(self, other: Union[Self, int, float]) -> Self:
        return self._binary_base(other, BinOpType.DIV, False)
    def __rtruediv__(self, other: Union[Self, int, float]) -> Self:
        return self._binary_base(other, BinOpType.DIV, True)
    def __itruediv__(self, other: Union[Self, int, float]) -> Self:
        return self._binary_base(other, BinOpType.DIV, False, True)

    def __floordiv__(self, other: Union[Self, int, float]) -> Self:
        return self._binary_base(other, BinOpType.FLOOR_DIV, False)
    def __rfloordiv__(self, other: Union[Self, int, float]) -> Self:
        return self._binary_base(other, BinOpType.FLOOR_DIV, True)
    def __ifloordiv__(self, other: Union[Self, int, float]) -> Self:
        return self._binary_base(other, BinOpType.FLOOR_DIV, False, True)

    def __mod__(self, other: Union[Self, int, float]) -> Self:
        return self._binary_base(other, BinOpType.MOD, False)
    def __rmod__(self, other: Union[Self, int, float]) -> Self:
        return self._binary_base(other, BinOpType.MOD, True)
    def __imod__(self, other: Union[Self, int, float]) -> Self:
        return self._binary_base(other, BinOpType.MOD, False, True)


    def __and__(self, other: Self) -> Self:
        return self._binary_base(other, BinOpType.BIT_AND, False)
    def __iand__(self, other: Self) -> Self:
        return self._binary_base(other, BinOpType.BIT_AND, False, True)
    def __rand__(self, other: Self) -> Self:
        return self._binary_base(other, BinOpType.BIT_AND, True)

    def __xor__(self, other: Self) -> Self:
        return self._binary_base(other, BinOpType.BIT_XOR, False)
    def __ixor__(self, other: Self) -> Self:
        return self._binary_base(other, BinOpType.BIT_XOR, False, True)
    def __rxor__(self, other: Self) -> Self:
        return self._binary_base(other, BinOpType.BIT_XOR, True)
    def __or__(self, other: Self) -> Self:
        return self._binary_base(other, BinOpType.BIT_OR, False)
    def __ior__(self, other: Self) -> Self:
        return self._binary_base(other, BinOpType.BIT_OR, False, True)
    def __ror__(self, other: Self) -> Self:
        return self._binary_base(other, BinOpType.BIT_OR, True)

    def __lshift__(self, other: Union[Self, int]) -> Self:
        return self._binary_base(other, BinOpType.LSHIFT, False)
    def __rlshift__(self, other: Union[Self, int]) -> Self:
        return self._binary_base(other, BinOpType.LSHIFT, True)
    def __ilshift__(self, other: Union[Self, int]) -> Self:
        return self._binary_base(other, BinOpType.LSHIFT, False, True)
    
    def __rshift__(self, other: Union[Self, int]) -> Self:
        return self._binary_base(other, BinOpType.RSHIFT, False)
    def __rrshift__(self, other: Union[Self, int]) -> Self:
        return self._binary_base(other, BinOpType.RSHIFT, True)
    def __irshift__(self, other: Union[Self, int]) -> Self:
        return self._binary_base(other, BinOpType.RSHIFT, False, True)

    def __pow__(self, other: Union[Self, int, float]) -> Self:
        return self._binary_base(other, BinOpType.POW, False)
    def __rpow__(self, other: Union[Self, int, float]) -> Self:
        return self._binary_base(other, BinOpType.POW, True)
    def __ipow__(self, other: Union[Self, int, float]) -> Self:
        return self._binary_base(other, BinOpType.POW, False, True)

    def __matmul__(self, other: Union[Self, int, float]) -> Self:
        return self._binary_base(other, BinOpType.MATMUL, False)
    def __rmatmul__(self, other: Union[Self, int, float]) -> Self:
        return self._binary_base(other, BinOpType.MATMUL, True)
    def __imatmul__(self, other: Union[Self, int, float]) -> Self:
        return self._binary_base(other, BinOpType.MATMUL, False, True)

def zeros(shape: Sequence[int], dtype: int) -> SimTensor:
    """
    Create a tensor filled with zeros.
    """
    parse_ctx = pfl.get_parse_context()
    meta_only = False
    if parse_ctx is not None:
        meta_only = True 
    if not meta_only:
        ctx = get_tensorsim_context()
        if ctx is not None and ctx.cfg.meta_only:
            meta_only = True 
    if meta_only:
        # only meta inference is supported
        return SimTensor(shape=list(map(int, shape)), dtype=dtype, storage=None)
    data = np.zeros(shape, dtype=PPCL_TO_NP_DTYPE[DTypeEnum(dtype)])
    storage = SimTensorStorage(data=data)
    return SimTensor(shape=list(map(int, shape)), dtype=dtype, storage=storage)

def ones(shape: Sequence[int], dtype: int) -> SimTensor:
    """
    Create a tensor filled with ones.
    """
    parse_ctx = pfl.get_parse_context()
    meta_only = False
    if parse_ctx is not None:
        meta_only = True 
    if not meta_only:
        ctx = get_tensorsim_context()
        if ctx is not None and ctx.cfg.meta_only:
            meta_only = True 
    if meta_only:
        # only meta inference is supported
        return SimTensor(shape=list(map(int, shape)), dtype=dtype, storage=None)
    data = np.ones(shape, dtype=PPCL_TO_NP_DTYPE[DTypeEnum(dtype)])
    storage = SimTensorStorage(data=data)
    return SimTensor(shape=list(map(int, shape)), dtype=dtype, storage=storage)

def empty(shape: Sequence[int], dtype: int) -> SimTensor:
    """
    Create an empty tensor with uninitialized data.
    """
    parse_ctx = pfl.get_parse_context()
    meta_only = False
    if parse_ctx is not None:
        meta_only = True 
    if not meta_only:
        ctx = get_tensorsim_context()
        if ctx is not None and ctx.cfg.meta_only:
            meta_only = True 
    if meta_only:
        # only meta inference is supported
        return SimTensor(shape=list(map(int, shape)), dtype=dtype, storage=None)
    data = np.empty(shape, dtype=PPCL_TO_NP_DTYPE[DTypeEnum(dtype)])
    storage = SimTensorStorage(data=data)
    return SimTensor(shape=list(map(int, shape)), dtype=dtype, storage=storage)

def broadcast_to(tensor: SimTensor, shape: Sequence[int]) -> SimTensor:
    """
    Broadcast a tensor to a new shape.
    """
    if tensor.storage is None:
        # only meta inference is supported
        new_shape = np.broadcast_shapes(tensor.shape, shape)
        return dataclasses.replace(tensor, shape=list(map(int, new_shape)))
    new_data = np.broadcast_to(tensor.storage.data, shape)
    new_storage = dataclasses.replace(tensor.storage, data=new_data)
    return dataclasses.replace(tensor, shape=list(map(int, new_data.shape)), storage=new_storage)

def arange(start: int, stop: Optional[int] = None, step: int = 1, dtype: int = DTypeEnum.int64) -> SimTensor:
    """
    Create a tensor with a range of values.
    """
    parse_ctx = pfl.get_parse_context()
    meta_only = False
    if parse_ctx is not None:
        meta_only = True 
    if not meta_only:
        ctx = get_tensorsim_context()
        if ctx is not None and ctx.cfg.meta_only:
            meta_only = True 
    if meta_only:
        # only meta inference is supported
        if stop is None:
            stop = start
            start = 0
        shape = [(stop - start + step - 1) // step]
        return SimTensor(shape=list(map(int, shape)), dtype=dtype, storage=None)
    if stop is None:
        stop = start
        start = 0
    data = np.arange(start, stop, step, dtype=PPCL_TO_NP_DTYPE[DTypeEnum(dtype)])
    storage = SimTensorStorage(data=data)
    return SimTensor(shape=list(map(int, data.shape)), dtype=dtype, storage=storage)

@dataclasses.dataclass(kw_only=True)
class SimMemoryStorage(SimTensorStorage):
    name: str
    def _boundry_check(self, pointer: Union["SimPointerTensor", "SimPointerScalarBase"], mask: Optional[SimTensor] = None):
        if pointer.storage is None:
            return 
        if mask is not None:
            if mask.storage is None:
                return None 
            assert mask.is_boolean(), "Mask must be boolean tensor"
            # check mask shape is broadcastable with pointer shape
            np.broadcast_shapes(mask.shape, pointer.shape)
            assert mask.ndim <= pointer.ndim
            mask_shape_rev = mask.shape[::-1]
            pointer_shape_rev = pointer.shape[::-1]
            # pointer can't be broadcasted.
            for j in range(mask.ndim):
                assert mask_shape_rev[j] <= pointer_shape_rev[j]

        data_raw = self.data.view(pointer.dtype_to_np(pointer.dtype)).reshape(-1, pointer.num_element)
        if mask is not None and mask.storage is not None:
            pointer_data = pointer.storage.data[mask.storage.data]
        else:
            pointer_data = pointer.storage.data
        pointer_data_max = pointer_data.max()
        pointer_data_min = pointer_data.min()
        if pointer_data_min < 0 or pointer_data_max >= data_raw.shape[0]:
            raise IndexError(f"Pointer data {pointer_data}({pointer_data_max}) out of bounds for memory size {data_raw.shape[0]}(shape={self.data.shape})")


    def load(self, pointer: Union["SimPointerTensor", "SimPointerScalarBase"], mask: Optional[Union[SimTensor, bool]] = None, other: Optional[Union[SimTensor, float, int, bool]] = None) -> SimTensor:
        if pointer.storage is None:
            assert not isinstance(pointer, SimPointerScalarBase)
            return SimTensor(shape=pointer.shape, dtype=pointer.dtype)
        if PPCL_TO_NP_DTYPE[DTypeEnum(pointer.dtype)] != self.data.dtype:
            raise NotImplementedError
        if mask is not None:
            if isinstance(mask, bool):
                mask_ten = zeros([], DTypeEnum.bool_)
                mask_ten.get_storage_checked().data[:] = mask 
                mask = mask_ten
            mask = broadcast_to(mask, pointer.shape)
        self._boundry_check(pointer, mask)
        output = empty([*pointer.shape, pointer.num_element], pointer.dtype)
        output_data = output.get_storage_checked().data
        data_raw = self.data.view(pointer.dtype_to_np(pointer.dtype)).reshape(-1, pointer.num_element)
        pointer_data = pointer.storage.data.reshape(-1)
        loaded_indices = pointer.storage.data[..., None] * pointer.num_element + np.arange(pointer.num_element, dtype=np.int32)
        if mask is None:
            output_data[:] = data_raw[pointer_data].reshape(*pointer.shape, pointer.num_element)
        else:
            mask_view = mask.get_storage_checked().data.reshape(-1)
            output_data.reshape(-1, pointer.num_element)[mask_view] = data_raw[pointer_data[mask_view]]
            if other is not None:
                if isinstance(other, SimTensor):
                    output_data.reshape(-1, pointer.num_element)[~mask_view] = other.get_storage_checked().data.reshape(-1, pointer.num_element)[~mask_view]
                else:
                    output_data.reshape(-1, pointer.num_element)[~mask_view] = other
            loaded_indices.reshape(-1, pointer.num_element)[~mask_view] = -1
        if pointer.num_element == 1:
            loaded_indices = loaded_indices[..., 0]
            output = output[..., 0]
        res_indices: dict[str, np.ndarray] = output.get_storage_checked().indices
        for k, inds in self.indices.items():
            pure_inds_shape = inds.shape[len(self.data.shape):]
            inds = inds.reshape(-1, *pure_inds_shape)
            if mask is None:
                inds = inds[pointer_data]
            else:
                mask_view = mask.get_storage_checked().data.reshape(-1)

                new_inds = np.full((*pointer.shape, pointer.num_element, *pure_inds_shape), -1, dtype=np.int32)
                new_inds_flatten = new_inds.reshape(-1, *pure_inds_shape)
                new_inds_flatten[mask_view] = inds[pointer_data[mask_view]]
                inds = new_inds
            if pointer.num_element == 1:
                inds = inds.reshape(*pointer.shape, *pure_inds_shape)
            else:
                inds = inds.reshape(*pointer.shape, pointer.num_element, *pure_inds_shape)
            res_indices[k] = (inds)
        res_indices[self.name] = (loaded_indices)
        return output


    def store(self, pointer: Union["SimPointerTensor", "SimPointerScalarBase"], value: Union[SimTensor, int, float], mask: Optional[Union[SimTensor, bool]] = None):
        assert pointer.num_element == 1, "Pointer must be a single element pointer for now"
        if pointer.storage is None:
            return 
        if PPCL_TO_NP_DTYPE[DTypeEnum(pointer.dtype)] != self.data.dtype:
            raise NotImplementedError
        if isinstance(value, SimTensor):
            # WARNING: only write indices when value is Tensor.
            # otherwise unchanged.
            if value.storage is None:
                return 
            if not self.indices:
                # lazy create indices based on first store value
                for k, value_inds in value.storage.indices.items():
                    pure_inds_shape = value_inds.shape[len(value.shape):]
                    self.indices[k] = (np.full([*self.data.shape, *pure_inds_shape], -1, dtype=np.int32)) 
            else:
                # validate indices
                if len(self.indices) != len(value.storage.indices):
                    raise ValueError("Cannot store value with different number of indices")
                for k, value_inds in value.storage.indices.items():
                    pure_inds_shape = value_inds.shape[len(value.shape):]
                    pure_this_shape = self.indices[k].shape[len(self.data.shape):]
                    if pure_inds_shape != pure_this_shape:
                        raise ValueError(f"Indices shape mismatch: {pure_inds_shape} vs {pure_this_shape}")
        
        if isinstance(mask, bool):
            mask_ten = zeros([], DTypeEnum.bool_)
            mask_ten.get_storage_checked().data[:] = mask 
            mask = mask_ten
        if not isinstance(value, SimTensor):
            value_ten = zeros([], get_default_base_dtype(type(value)))
            value_ten.get_storage_checked().data[:] = value 
            value = value_ten
        assert value.storage is not None 
        self._boundry_check(pointer, mask)
        data_raw = self.data.view(pointer.dtype_to_np(pointer.dtype)).reshape(-1)
        pointer_data = pointer.storage.data.reshape(-1)
        stored_data_raw = value.storage.data.reshape(-1)
        if mask is None:
            data_raw[pointer_data] = stored_data_raw
        else:
            mask_view = mask.get_storage_checked().data.reshape(-1)
            data_raw[pointer_data[mask_view]] = stored_data_raw[mask_view]
        for k, inds in self.indices.items():
            value_inds = value.storage.indices[k]
            inds = inds.reshape(-1, *(inds.shape[len(self.data.shape):]))
            if mask is None:
                inds[pointer_data] = value_inds.reshape(-1)
            else:
                mask_view = mask.get_storage_checked().data.reshape(-1)
                inds[pointer_data[mask_view]] = value_inds.reshape(-1)[mask_view]


def create_sim_memory(name: str, data: np.ndarray):
    return SimMemoryStorage(data, name=name)

@dataclasses.dataclass
class SimPointerTensor(SimTensorBase):
    # pointer tensor is always int64*, dtype is element type.
    # pointer of pointer is unsupported currently.
    # num_element: for vector load, only used when we directly target to raw backend, e.g. CUDA.
    num_element: int = 1
    memory_storage: Optional[SimMemoryStorage] = None
    def __post_init__(self):
        if self.storage is not None:
            assert self.storage.data.dtype == np.int64, "Pointer tensor storage must be of int64 type"

    def to_meta_tensor(self):
        return dataclasses.replace(self, storage=None, memory_storage=None)

    def is_pointer(self) -> bool:
        return True

    def get_memory_storage_checked(self) -> SimMemoryStorage:
        if self.memory_storage is None:
            raise ValueError("Pointer tensor does not have a memory storage")
        return self.memory_storage

    def load(self, mask: Optional[Union[bool, SimTensor]] = None, other: Optional[Union[SimTensor, float, int, bool]] = None) -> SimTensor:
        if self.memory_storage is None:
            # create a meta tensor from self
            return SimTensor(shape=self.shape, dtype=self.dtype, storage=None)
        return self.memory_storage.load(self, mask, other)

    def store(self, value: Union[SimTensor, int, float], mask: Optional[Union[bool, SimTensor]] = None):
        if self.memory_storage is None:
            return 
        return self.memory_storage.store(self, value, mask)

    def __add__(self, other: Union[SimTensor, int]) -> Self: 
        res = self._binary_base(cast(Self, other), BinOpType.ADD, False)
        assert isinstance(res, SimPointerTensor)
        return res
    def __iadd__(self, other: Union[SimTensor, int]) -> Self: 
        res = self._binary_base(cast(Self, other), BinOpType.ADD, False, True)
        assert isinstance(res, SimPointerTensor)
        return res

    def __radd__(self, other: Union[SimTensor, int]) -> Self:
        res = self._binary_base(cast(Self, other), BinOpType.ADD, True)
        assert isinstance(res, SimPointerTensor)
        return res

    def __sub__(self, other: Union[SimTensor, int]) -> Self:
        res =  self._binary_base(cast(Self, other), BinOpType.SUB, False)
        assert isinstance(res, SimPointerTensor)
        return res
    def __isub__(self, other: Union[SimTensor, int]) -> Self:
        res =  self._binary_base(cast(Self, other), BinOpType.SUB, False, True)
        assert isinstance(res, SimPointerTensor)
        return res
    def __rsub__(self, other: Union[SimTensor, int]) -> Self:
        res =  self._binary_base(cast(Self, other), BinOpType.SUB, True)
        assert isinstance(res, SimPointerTensor)
        return res

@dataclasses.dataclass
class SimPointerScalarBase(SimTensorBase):
    # pointer tensor is always int64*, dtype is element type.
    # pointer of pointer is unsupported currently.
    num_element: int = 1
    memory_storage: Optional[SimMemoryStorage] = None
    def __post_init__(self):
        assert self.is_scalar()
        if self.storage is not None:
            assert self.storage.data.dtype == np.int64, "Pointer tensor storage must be of int64 type"

    def to_meta_tensor(self):
        return dataclasses.replace(self, storage=None, memory_storage=None)

    def is_pointer(self) -> bool:
        return True

    def get_memory_storage_checked(self) -> SimMemoryStorage:
        if self.memory_storage is None:
            raise ValueError("Pointer tensor does not have a memory storage")
        return self.memory_storage

    def __add__(self, other: Union[SimTensor, int]) -> Self: 
        res = self._binary_base(cast(Self, other), BinOpType.ADD, False)
        assert isinstance(res, SimPointerTensor)
        return res

    def __iadd__(self, other: Union[SimTensor, int]) -> Self: 
        res = self._binary_base(cast(Self, other), BinOpType.ADD, False, True)
        assert isinstance(res, SimPointerTensor)
        return res

    def __radd__(self, other: Union[SimTensor, int]) -> Self:
        res = self._binary_base(cast(Self, other), BinOpType.ADD, True)
        assert isinstance(res, SimPointerTensor)
        return res

    def __sub__(self, other: Union[SimTensor, int]) -> Self:
        res =  self._binary_base(cast(Self, other), BinOpType.SUB, False)
        assert isinstance(res, SimPointerTensor)
        return res

    def __isub__(self, other: Union[SimTensor, int]) -> Self:
        res =  self._binary_base(cast(Self, other), BinOpType.SUB, False, True)
        assert isinstance(res, SimPointerTensor)
        return res

    def __rsub__(self, other: Union[SimTensor, int]) -> Self:
        res =  self._binary_base(cast(Self, other), BinOpType.SUB, True)
        assert isinstance(res, SimPointerTensor)
        return res


@dataclasses.dataclass
class SimPointerScalarFloat(SimPointerScalarBase):
    def load(self, mask: Optional[bool] = None, other: Optional[Union[float, int]] = None) -> float:
        if self.memory_storage is None or self.storage is None:
            raise ValueError("can't load scalar meta")
        res = self.memory_storage.load(self, mask, other)
        assert res.is_scalar()
        assert res.storage is not None
        return res.storage.data.item()

    def store(self, value: Union[float, int], mask: Optional[bool] = None):
        if self.memory_storage is None:
            return 
        return self.memory_storage.store(self, value, mask)

@dataclasses.dataclass
class SimPointerScalarInt(SimPointerScalarBase):
    def load(self, mask: Optional[bool] = None, other: Optional[Union[float, int]] = None) -> int:
        if self.memory_storage is None or self.storage is None:
            raise ValueError("can't load scalar meta")
        res = self.memory_storage.load(self, mask, other)
        assert res.is_scalar()
        assert res.storage is not None
        return res.storage.data.item()

    def store(self, value: Union[float, int], mask: Optional[bool] = None):
        if self.memory_storage is None:
            return 
        return self.memory_storage.store(self, value, mask)

def create_pointer_tensor_meta(dtype: int, shape: list[int], num_element: int = 1) -> SimPointerTensor:
    """ Create a pointer tensor with a single scalar value.
    """
    return SimPointerTensor(shape=shape, dtype=dtype, num_element=num_element, storage=None)

def create_pointer_scalar_meta(dtype: int, num_element: int = 1) -> Union[SimPointerScalarInt, SimPointerScalarFloat]:
    """ Create a pointer tensor with a single scalar value.
    """
    if DTypeEnum(dtype).is_floating_type():
        return SimPointerScalarFloat(shape=[], dtype=dtype, num_element=num_element)
    else:
        return SimPointerScalarInt(shape=[], dtype=dtype, num_element=num_element)

def create_pointer_tensor(dtype: int, ptr: Union[np.ndarray, int], memory: SimMemoryStorage, num_element: int = 1) -> SimPointerTensor:
    """ Create a pointer tensor with a single scalar value.
    """
    if not isinstance(ptr, np.ndarray):
        assert ptr >= 0
    assert memory is not None, "Memory storage must be provided for pointer tensors in sim mode"
    if isinstance(ptr, np.ndarray):
        assert ptr.dtype == np.int64
        data = ptr
    else:
        data = np.array(ptr, dtype=np.int64)
    storage = SimTensorStorage(data=data)
    ptr_dtype_np = PPCL_TO_NP_DTYPE[DTypeEnum(dtype)]
    assert ptr_dtype_np == memory.data.dtype.type, "Pointer tensor dtype must match memory storage dtype"
    return SimPointerTensor(shape=list(data.shape), dtype=dtype, num_element=num_element, storage=storage, memory_storage=memory)

def create_pointer_scalar(dtype: int, ptr: int, memory: SimMemoryStorage, num_element: int = 1) -> Union[SimPointerScalarInt, SimPointerScalarFloat]:
    """ Create a pointer tensor with a single scalar value.
    """
    assert ptr >= 0
    assert memory is not None, "Memory storage must be provided for pointer tensors in sim mode"
    data = np.array(ptr, dtype=np.int64)
    storage = SimTensorStorage(data=data)
    ptr_dtype_np = PPCL_TO_NP_DTYPE[DTypeEnum(dtype)]
    assert ptr_dtype_np == memory.data.dtype.type, "Pointer tensor dtype must match memory storage dtype"
    if DTypeEnum(dtype).is_floating_type():
        return SimPointerScalarFloat(shape=[], dtype=dtype, num_element=num_element, storage=storage, memory_storage=memory)
    else:
        return SimPointerScalarInt(shape=[], dtype=dtype, num_element=num_element, storage=storage, memory_storage=memory)

def get_may_tensor_dtype(x: Union[int, float, bool, SimTensor]):
    if isinstance(x, bool):
        return DTypeEnum.bool_
    elif isinstance(x, int):
        return get_default_int_dtype()
    elif isinstance(x, float):
        return get_default_float_dtype()
    else:
        return DTypeEnum(x.dtype)


