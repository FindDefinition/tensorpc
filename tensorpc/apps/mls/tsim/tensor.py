from collections.abc import Sequence
import enum
from functools import partial
import io
from typing import Any, Optional, Type, Union, cast, overload
import dataclasses
import numpy as np 
from tensorpc.core import pfl
import contextlib 
import contextvars
from typing_extensions import Self

from tensorpc.core.pfl.pfl_ast import BinOpType, CompareType, UnaryOpType
from tensorpc.apps.mls.tsim.core import DTypeEnum, get_default_base_dtype, get_default_float_dtype, get_default_int_dtype, get_tensorsim_context, NumpyReduceType
from .core import get_sim_mode, TensorSimMode, get_tensorsim_context_checked

@dataclasses.dataclass
class SimTensorStorage:
    data: np.ndarray
    # multi-level io. e.g. load from global and store to shared memory, 
    # then load from shared memory.
    # both global indices and shared indices will be stored here.
    indices: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

    def __repr__(self):
        if self.data.dtype == np.bool_ and self.data.ndim == 2 and self.data.shape[0] <= 128 and self.data.shape[1] <= 128:
            ss = io.StringIO()
            for i, row in enumerate(self.data):
                row_str = f"{i:03d}"
                print(f"[{row_str}] ", end="", file=ss)
                for bit in row:
                    print(int(bit), end="", file=ss)  # Print 0 or 1 without a newline
                print(file=ss)  # Print a newline after each row
            return f"{self.__class__.__name__}:\n{ss.getvalue()}"

        return f"{self.__class__.__name__}:\n{self.data}"

    def clone(self) -> Self:
        return dataclasses.replace(self, data=self.data.copy(), indices={k: v.copy() for k, v in self.indices.items()}) 

    def __post_init__(self):
        assert not isinstance(self.data, np.number)

    def getItem(self, inds: Any) -> Self:
        new_data = self.data[inds]
        if isinstance(new_data, np.number):
            new_data = np.array(new_data)
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
            for k in new_storage.indices.keys():
                indices, element_shape = self.get_unflatten_inds(k)

                new_slices = [slice(None)] * (indices.ndim - self.data.ndim)
                if ellipsis_found:
                    new_inds = (*inds, *new_slices) 
                else:
                    new_inds = (*inds, ..., *new_slices) 
                new_storage_inds[k] = indices[new_inds].reshape(-1, *element_shape)
            new_storage.indices = new_storage_inds
        else:
            new_storage_inds = {}
            for k, indices in new_storage.indices.items():
                indices, element_shape = self.get_unflatten_inds(k)
                new_storage_inds[k] = indices[inds].reshape(-1, *element_shape)
            new_storage.indices = new_storage_inds
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

            for k in self.indices.keys():
                indices, element_shape = self.get_unflatten_inds(k)
                new_slices = [slice(None)] * (indices.ndim - self.data.ndim)
                if ellipsis_found:
                    new_inds = (*inds, *new_slices) 
                else:
                    new_inds = (*inds, ..., *new_slices) 
                if isinstance(value, SimTensor):
                    assert value.storage is not None 
                    indices[new_inds] = value.storage.indices[k]
                else:
                    indices[new_inds] = -1
                self.indices[k] = indices.reshape(-1, *element_shape)
        else:
            for k in self.indices.keys():
                indices, element_shape = self.get_unflatten_inds(k)
                if isinstance(value, SimTensor):
                    assert value.storage is not None 
                    indices[inds] = value.storage.indices[k]
                else:
                    indices[inds] = -1
                self.indices[k] = indices.reshape(-1, *element_shape)

    @staticmethod 
    def get_reshaped_shape(data_shape: Sequence[int], new_shape: Sequence[int]) -> Sequence[int]:
        """
        Calculate the new shape after reshaping.
        If -1 is found, it will be replaced with the product of the remaining dimensions.
        """
        new_shape_list = list(new_shape)
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
                f"Cannot reshape tensor with shape {data_shape} to {new_shape_list}, "
            new_shape_list[found_idx] = total_prod // total_prod_no_minus_one
        return new_shape_list

    def reshape(self, shape: Sequence[int]):
        new_shape_list = self.get_reshaped_shape(self.data.shape, shape)
        new_data = self.data.reshape(new_shape_list)
        return dataclasses.replace(self, data=new_data)

    @staticmethod 
    def get_permuted_shape(data_shape: Sequence[int], new_order: Sequence[int]) -> Sequence[int]:
        """
        Calculate the new shape after permuting the dimensions.
        """
        assert len(new_order) == len(data_shape)
        return [data_shape[i] for i in new_order]

    def permute(self, new_order: Sequence[int]):
        assert len(new_order) == self.data.ndim
        new_data = self.data.transpose(new_order)
        new_inds = {}
        for k in self.indices.keys():
            indices, element_shape = self.get_unflatten_inds(k)
            new_indices = indices.transpose(list(new_order) + list(range(len(new_order), len(new_order) + len(element_shape))))
            new_inds[k] = new_indices.reshape(-1, *element_shape)
        new_storage = dataclasses.replace(self, data=new_data, indices=new_inds)
        return new_storage

    def get_unflatten_inds(self, k: str) -> tuple[np.ndarray, Sequence[int]]:
        """
        Get the unflattened indices for a given key.
        """
        indices = self.indices[k]
        element_shape = indices.shape[1:]
        return indices.reshape(*self.data.shape, *element_shape), element_shape

    def reduce(self, new_data: np.ndarray, axes: list[int], keepdims: bool) -> Self:
        new_storage = dataclasses.replace(self, data=new_data) 
        new_storage.indices = {}
        new_shape: list[int] = []
        permute_inds: list[int] = []
        permute_shape: list[int] = []
        for i, dim in enumerate(self.data.shape):
            if i in axes:
                if keepdims:
                    new_shape.append(1)
            else:
                new_shape.append(dim)
                permute_inds.append(i)
                permute_shape.append(dim)
        # old_ndim = self.data.ndim
        # for k in self.indices.keys():
        #     indices, element_shape = self.get_unflatten_inds(k)
        #     permute_inds_cur = permute_inds.copy()
        #     pure_inds_ndim = self.data.ndim - indices.ndim
        #     permute_inds_cur.extend(c + old_ndim for c in range(pure_inds_ndim))
        #     permute_inds_cur.extend(axes)
        #     new_indices = indices.transpose(permute_inds_cur)
        #     new_indices_shape = list(new_data.shape) + list(new_indices.shape)[-len(axes) - pure_inds_ndim:]
        #     print("!!!", k, element_shape, indices.shape, permute_inds, new_indices.shape, new_indices_shape)
        #     new_indices = new_indices.reshape(new_indices_shape) 
        #     new_storage.indices[k] = (cast(np.ndarray, new_indices)).reshape(-1, *permute_shape)
        #     new_storage.get_unflatten_inds(k)
        # print("WTFWTF", new_data.shape, self.data.shape, {k: v.shape for k, v in new_storage.indices.items()})
        new_storage.data = new_data
        return new_storage

    def concat(self, other_list: list[Self], axis: int) -> Self:
        new_data = np.concatenate((self.data, *[o.data for o in other_list]), axis=axis)
        # TODO use full-element-trace instead of indices system. no need to implement indices here.
        new_storage = dataclasses.replace(self, data=new_data, indices={})
        return new_storage

    def broadcast_to(self, new_shape: Sequence[int]) -> Self:
        new_data = np.broadcast_to(self.data, new_shape)
        new_storage = dataclasses.replace(self, data=new_data)
        new_storage.indices = {}
        for k in self.indices.keys():
            indices, element_shape = self.get_unflatten_inds(k)
            new_indices = np.broadcast_to(indices, (*new_shape, *element_shape))
            new_storage.indices[k] = new_indices.reshape(-1, *element_shape)
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

    def __repr__(self):
        shape_str = ",".join(map(str, self.shape))
        return f"{self.__class__.__name__}([{shape_str}|{DTypeEnum(self.dtype).name}|{self.storage}])"

    def get_storage_checked(self) -> SimTensorStorage:
        assert self.storage is not None 
        return self.storage 

    def clone(self) -> Self:
        new_storage = self.storage.clone() if self.storage is not None else None
        # memory storage is not cloned, it is shared.
        return dataclasses.replace(self, storage=new_storage) 

    @staticmethod
    def dtype_promotion(*args: int):
        return DTypeEnum.dtype_promotion(*args)

    def is_logicsim_ignore(self) -> bool:
        return get_sim_mode() == TensorSimMode.LOGIC_ONLY and (DTypeEnum(self.dtype).is_floating_type() and not self.is_pointer())

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
        return DTypeEnum(dtype).to_numpy_dtype()

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
            new_storage = self.storage.getItem(inds) 
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
        return self.permute(list(range(self.ndim - 1, -1, -1)))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def to(self, dtype: int) -> Self:
        if self.storage is None:
            return dataclasses.replace(self, dtype=dtype)
        if self.dtype == dtype:
            return dataclasses.replace(self)
        new_data = self.storage.data.astype(DTypeEnum(dtype).to_numpy_dtype())
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
        new_shape_list = SimTensorStorage.get_reshaped_shape(self.shape, new_shape_list)
        if self.storage is None:
            return dataclasses.replace(self, shape=list(map(int, new_shape_list)))
        new_storage = self.storage.reshape(new_shape_list)
        res = dataclasses.replace(self, shape=list(map(int, new_storage.data.shape)))
        res.storage = new_storage
        return res

    def permute(self, new_order: Union[Sequence[int], int], *orders: int) -> Self:
        if isinstance(new_order, int):
            new_order_list = [new_order, *orders]
        else:
            new_order_list = list(new_order)
        # calc new shape, find -1 first
        new_shape_list = SimTensorStorage.get_permuted_shape(self.shape, new_order_list)
        if self.storage is None:
            return dataclasses.replace(self, shape=list(map(int, new_shape_list)))
        new_storage = self.storage.permute(new_order_list)
        res = dataclasses.replace(self, shape=list(map(int, new_storage.data.shape)))
        res.storage = new_storage
        return res

    def cumsum(self, dim: int):
        if self.storage is None or self.is_logicsim_ignore():
            return dataclasses.replace(self)
        new_data = np.cumsum(self.storage.data, axis=dim)
        new_storage = self.storage.clone()
        new_storage.data = new_data
        res = dataclasses.replace(self, shape=list(map(int, new_data.shape)))
        res.storage = new_storage
        return res

    def _unary_base(self, op_type: UnaryOpType) -> Self:
        if self.storage is None:
            return dataclasses.replace(self, shape=list(map(int, self.shape)))
        if self.is_logicsim_ignore():
            new_data = self.storage.data
        else:
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
        if self.storage is None or self.is_logicsim_ignore():
            if isinstance(other, SimTensorBase):
                if self.is_logicsim_ignore():
                    assert other.is_logicsim_ignore()
                if op_type == BinOpType.MATMUL:
                    assert len(self.shape) >= 2 and len(other.shape) >= 2
                    new_shape_no_mm = np.broadcast_shapes(self.shape[:-2], other.shape[:-2])
                    assert self.shape[-1] == other.shape[-2], f"matmul shape mismatch, {self.shape} != {other.shape}"
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
                elif isinstance(other, float):
                    other_dtype = get_default_float_dtype()
                elif isinstance(other, bool):
                    other_dtype = DTypeEnum.bool_
                else:
                    raise NotImplementedError
                new_shape = self.shape
                new_dtype = self.dtype_promotion(self.dtype, other_dtype)
            if get_sim_mode() == TensorSimMode.LOGIC_ONLY:
                ctx = get_tensorsim_context_checked()
                new_data = ctx._cached_empty(new_shape, dtype=self.dtype_to_np(new_dtype))
                new_storage = SimTensorStorage(data=new_data, indices={})
            else:
                new_storage = None
            return dataclasses.replace(res_replace_tgt, shape=list(map(int, new_shape)), dtype=new_dtype, storage=new_storage)
        if isinstance(other, SimTensorBase):
            assert not other.is_logicsim_ignore()
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
                res = dataclasses.replace(self, shape=list(map(int, new_data.shape)), dtype=DTypeEnum.from_numpy_dtype(new_data.dtype))
                res.storage = new_storage
            return res

    def _compare_base(self, other: Union[Self, int, float, bool], op_type: CompareType) -> Self:
        # TODO how to support compare mask in logic only mode?
        if get_sim_mode() == TensorSimMode.LOGIC_ONLY:
            assert not self.is_floating()
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
            if get_sim_mode() == TensorSimMode.LOGIC_ONLY:
                assert not other.is_floating()

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
        res = dataclasses.replace(self, shape=list(map(int, new_data.shape)), dtype=DTypeEnum.from_numpy_dtype(new_data.dtype))
        res.storage = new_storage
        return res

    def concat(self, tensors: Sequence[Self], axis: int = 0) -> Self:
        """
        Concatenate a sequence of tensors along a specified axis.
        """
        tensors = [self, *tensors]
        if len(tensors) == 0:
            raise ValueError("Cannot concatenate an empty sequence of tensors")
        if axis < 0:
            axis += len(tensors[0].shape)
        assert all(isinstance(t, SimTensorBase) for t in tensors), "All inputs must be SimTensor instances"
        if any(t.storage is None for t in tensors):
            # meta inference. validate shapes
            first_shape_no_cat = list(tensors[0].shape)
            first_shape_no_cat[axis] = 0  # set the concatenation axis to 0
            total_sum_axis = 0
            for t in tensors:
                t_shape_no_cat = list(t.shape)
                total_sum_axis += t_shape_no_cat[axis]
                t_shape_no_cat[axis] = 0
                assert t_shape_no_cat == first_shape_no_cat, "All tensors must have the same shape except axis for concat"
                assert t.dtype == tensors[0].dtype, "All tensors must have the same dtype for concat"
            first_shape_no_cat[axis] = total_sum_axis
            return dataclasses.replace(self, shape=first_shape_no_cat, dtype=tensors[0].dtype, storage=None)
        new_storage = tensors[0].get_storage_checked().concat([t.get_storage_checked() for t in tensors], axis)
        return dataclasses.replace(self, shape=list(map(int, new_storage.data.shape)), dtype=tensors[0].dtype, storage=new_storage)

    def unsqueeze(self, axis: Optional[Union[int, Sequence[int]]] = None) -> Self:
        """
        Add a new dimension to the tensor at the specified axis.
        If axis is None, add a new dimension at the end.
        """
        if self.storage is None:
            if axis is None:
                new_shape = self.shape + [1]
            elif isinstance(axis, int):
                new_shape = self.shape[:axis] + [1] + self.shape[axis:]
            else:
                new_shape = list(self.shape)
                for a in sorted(axis, reverse=True):
                    new_shape.insert(a, 1)
            return dataclasses.replace(self, shape=new_shape)
        else:
            if axis is None:
                new_shape = self.shape + [1]
            elif isinstance(axis, int):
                new_shape = self.shape[:axis] + [1] + self.shape[axis:]
            else:
                new_shape = list(self.shape)
                for a in sorted(axis, reverse=True):
                    new_shape.insert(a, 1)
            new_storage = self.storage.reshape(new_shape)
            res = dataclasses.replace(self, shape=list(map(int, new_storage.data.shape)), storage=new_storage)
            return res

    def squeeze(self, axis: Optional[Union[int, Sequence[int]]] = None) -> Self:
        """
        Remove dimensions of size 1 from the tensor.
        If axis is specified, only remove those dimensions.
        """
        if self.storage is None:
            if axis is None:
                new_shape = [dim for dim in self.shape if dim != 1]
            elif isinstance(axis, int):
                new_shape = self.shape[:axis] + self.shape[axis + 1:] if self.shape[axis] == 1 else self.shape
            else:
                new_shape = list(self.shape)
                for a in sorted(axis, reverse=True):
                    if new_shape[a] == 1:
                        new_shape.pop(a)
            return dataclasses.replace(self, shape=new_shape)
        else:
            if axis is None:
                new_shape = [dim for dim in self.shape if dim != 1]
                axes_to_remove = [i for i, dim in enumerate(self.shape) if dim == 1]
            elif isinstance(axis, int):
                new_shape = self.shape[:axis] + self.shape[axis + 1:] if self.shape[axis] == 1 else self.shape
                axes_to_remove = [axis] if self.shape[axis] == 1 else []
            else:
                new_shape = list(self.shape)
                axes_to_remove = [a for a in axis if new_shape[a] == 1]
                for a in sorted(axes_to_remove, reverse=True):
                    new_shape.pop(a)
            new_storage = self.storage.reduce(self.storage.data, axes_to_remove, keepdims=False)
            res = dataclasses.replace(self, shape=list(map(int, new_storage.data.shape)), storage=new_storage)
            return res

    def stack(self, tensors: Sequence[Self], axis: int = 0) -> Self:
        """
        Stack a sequence of tensors along a new axis.
        """
        # use concat to implement stack

        tensors = [self, *tensors]
        if len(tensors) == 0:
            raise ValueError("Cannot stack an empty sequence of tensors")
        axis_unsq = axis
        if axis_unsq < 0:
            axis_unsq += len(tensors[0].shape) + 1
        tensors_unsqueezed = [t.unsqueeze(axis_unsq) for t in tensors]
        return tensors_unsqueezed[0].concat(tensors_unsqueezed[1:], axis)


@dataclasses.dataclass
class SimTensor(SimTensorBase):
    """
    A CPU/Meta Tensor that can be used for computing simulation or metadata inference.
    
    """
    def __repr__(self):
        return super().__repr__()

    def _reduce_meta_only(self, axes: Optional[Sequence[int]] = None, keepdims: bool = False) -> Self:
        if axes is None:
            axes = list(range(len(self.shape)))
        else:
            axes = list(axes)
        for i in range(len(axes)):
            # handle negative axes
            if axes[i] < 0:
                axes[i] += len(self.shape)
            assert 0 <= axes[i] < len(self.shape), f"Axis {axes[i]} is out of bounds for tensor of dimension {len(self.shape)}"
        # only meta inference is supported
        new_shape: list[int] = []
        for i, dim in enumerate(self.shape):
            if i not in axes:
                new_shape.append(dim)
            elif keepdims:
                new_shape.append(1)
        if self.is_logicsim_ignore():
            new_data = np.empty(new_shape, dtype=self.dtype_to_np(self.dtype))  # type: ignore
            new_storage = SimTensorStorage(data=new_data, indices={})
        else:
            new_storage = None
        res = dataclasses.replace(self, shape=new_shape, storage=new_storage)
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
        if self.storage is None or self.is_logicsim_ignore():
            # logic only mode don't support argmax/argmin.
            assert rtype != NumpyReduceType.ARGMAX and rtype != NumpyReduceType.ARGMIN, "argmax/argmin is not supported in logic only mode"
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

    def clone(self) -> Self:
        """
        Create a clone of the tensor.
        """
        if self.storage is None:
            return dataclasses.replace(self, storage=None)
        new_data = self.storage.data.copy()
        new_storage = dataclasses.replace(self.storage, data=new_data)
        return dataclasses.replace(self, shape=list(map(int, new_data.shape)), dtype=self.dtype, storage=new_storage)

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
    data = np.zeros(shape, dtype=DTypeEnum(dtype).to_numpy_dtype())
    storage = SimTensorStorage(data=data)
    return SimTensor(shape=list(map(int, shape)), dtype=dtype, storage=storage)

def full(shape: Sequence[int], value: Union[int, float, bool], dtype: int) -> SimTensor:
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
    data = np.full(shape, value, dtype=DTypeEnum(dtype).to_numpy_dtype())
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
    data = np.ones(shape, dtype=DTypeEnum(dtype).to_numpy_dtype())
    storage = SimTensorStorage(data=data)
    return SimTensor(shape=list(map(int, shape)), dtype=dtype, storage=storage)

def empty(shape: Sequence[int], dtype: int, *, _internal_cached_empty: bool = False) -> SimTensor:
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
    if _internal_cached_empty:
        ctx = get_tensorsim_context_checked()
        data = ctx._cached_empty(shape, dtype=DTypeEnum(dtype).to_numpy_dtype())
    else:
        data = np.empty(shape, dtype=DTypeEnum(dtype).to_numpy_dtype())
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
    new_storage = tensor.storage.broadcast_to(shape)
    return dataclasses.replace(tensor, shape=list(map(int, new_storage.data.shape)), storage=new_storage)

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
    data = np.arange(start, stop, step, dtype=DTypeEnum(dtype).to_numpy_dtype())
    storage = SimTensorStorage(data=data)
    return SimTensor(shape=list(map(int, data.shape)), dtype=dtype, storage=storage)


def get_may_tensor_dtype(x: Union[int, float, bool, SimTensor]):
    if isinstance(x, bool):
        return DTypeEnum.bool_
    elif isinstance(x, int):
        return get_default_int_dtype()
    elif isinstance(x, float):
        return get_default_float_dtype()
    else:
        return DTypeEnum(x.dtype)


def from_numpy(arr: np.ndarray) -> SimTensor:
    """
    Create a SimTensor from a numpy array.
    """
    dtype = DTypeEnum.from_numpy_dtype(arr.dtype)
    assert isinstance(arr, np.ndarray), "Input must be a numpy array"
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
        return SimTensor(shape=list(map(int, arr.shape)), dtype=dtype, storage=None)
    storage = SimTensorStorage(data=arr)
    return SimTensor(shape=list(map(int, arr.shape)), dtype=dtype, storage=storage)

