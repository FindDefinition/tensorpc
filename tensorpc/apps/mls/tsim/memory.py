from collections.abc import Sequence
from functools import partial
from typing import Any, Callable, Optional, Type, Union, cast, overload
import dataclasses
import numpy as np
from typing_extensions import Literal, Self

from tensorpc.core import pfl
from tensorpc.core.pfl.evaluator import get_pfl_runner_state
from tensorpc.core.pfl.pfl_ast import BinOpType, CompareType, UnaryOpType
from tensorpc.apps.mls.tsim.core import (
    DTypeEnum,
    TensorSimIoMatrixInfo,
    TensorSimIoOp,
    TensorSimMode,
    get_default_base_dtype,
    get_sim_mode,
    get_tensorsim_context,
    get_tensorsim_context_checked,
)
from .tensor import (
    SimTensor,
    SimTensorBase,
    SimTensorStorage,
    empty,
    full,
    zeros,
    broadcast_to,
    from_numpy,
)


def _align_up(a: int, b: int):
    return ((a + b - 1) // b) * b


@dataclasses.dataclass(kw_only=True)
class MemoryBlockDesc:
    size: int
    dtype: DTypeEnum
    byte_offset: int = -1
    byte_offset_with_hole = -1
    data_view: Optional[np.ndarray] = None
    indices: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    write_cnt: int = 0

    def is_written(self) -> bool:
        return self.write_cnt > 0

    def __repr__(self):
        return f"Memory[{self.size}|{self.dtype.name}|{self.offset}~{self.offset + self.size}|{self.offset_with_hole}~{self.offset_with_hole + self.size}]"

    def get_data_view_checked(self):
        assert self.data_view is not None
        return self.data_view

    def get_byte_size(self) -> int:
        """Get the byte size of the memory block."""
        return self.size * self.dtype.bit_size() // 8

    @property
    def offset(self):
        return self.byte_offset // self.dtype.byte_size()

    @property
    def offset_with_hole(self):
        return self.byte_offset_with_hole // self.dtype.byte_size()

    def get_unflatten_inds(self, k: str) -> tuple[np.ndarray, Sequence[int]]:
        """
        Get the unflattened indices for a given key.
        """
        indices = self.indices[k]
        element_shape = indices.shape[1:]
        return indices.reshape(*self.get_data_view_checked().shape,
                               *element_shape), element_shape


def assign_random_memory_block_offset(
    descs: dict[str, MemoryBlockDesc],
    memory_hole_range: Optional[tuple[int, int]] = None,
):
    # if memory_distance_range is None, use max size of descs
    if memory_hole_range is None:
        # 2x~3x
        max_byte_size = max(d.get_byte_size() for d in descs.values())
        memory_hole_range = (2 * max_byte_size, 3 * max_byte_size)
    start = 0
    start_with_hole = 0
    for v in descs.values():
        v.byte_offset = start
        v.byte_offset_with_hole = start_with_hole
        hole_size = np.random.randint(*memory_hole_range)
        start += v.get_byte_size()
        start = _align_up(start, 128)
        start_with_hole += v.get_byte_size() + hole_size
        start_with_hole = _align_up(start_with_hole, 128)
    return start


@dataclasses.dataclass
class _SimPointerMapResult:
    mapped: np.ndarray
    block_name: str
    is_partial: bool
    all_masked: bool
    origin_pointer_with_mask: np.ndarray


@dataclasses.dataclass(kw_only=True)
class SimMemoryStorage(SimTensorStorage):
    memory_blocks: dict[str, MemoryBlockDesc]

    def _map_pointer_data(self, pointer_data: np.ndarray,
                          pointer_item_size: int):
        if pointer_data.size == 0:
            raise NotImplementedError
        pointer_data_max = pointer_data.max() * pointer_item_size
        pointer_data_min = pointer_data.min() * pointer_item_size
        is_partial: bool = False
        # no need to use binary search because amount of memory blocks is small.
        for k, desc in self.memory_blocks.items():
            min_in_range = (pointer_data_min >= desc.byte_offset_with_hole
                            and pointer_data_min < desc.byte_offset_with_hole +
                            desc.get_byte_size())
            max_in_range = (pointer_data_max >= desc.byte_offset_with_hole
                            and pointer_data_max < desc.byte_offset_with_hole +
                            desc.get_byte_size())
            if min_in_range or max_in_range:
                is_partial = not (max_in_range and min_in_range)
                pointer_data_offset = (
                    desc.byte_offset_with_hole) // desc.dtype.byte_size()
                return pointer_data_offset, k, is_partial
        block_ranges = [
            f"{k}: {v.byte_offset_with_hole} ~ {v.byte_offset_with_hole + v.get_byte_size()}"
            for k, v in self.memory_blocks.items()
        ]
        raise ValueError(
            f"can't find your pointer {pointer_item_size} in any memory block, may "
            f"out of range: {pointer_data_min} {pointer_data_max} not in {block_ranges}."
        )

    def _map_pointer(
        self,
        pointer: Union["SimPointerTensor", "SimPointerScalarBase"],
        mask: Optional[SimTensor] = None,
    ):
        assert pointer.storage is not None
        origin_with_mask = np.empty_like(pointer.storage.data)

        if mask is not None and mask.storage is not None:
            pointer_data = pointer.storage.data[mask.storage.data]
            origin_with_mask[~mask.storage.data] = -1
        else:
            pointer_data = pointer.storage.data

        if pointer_data.size == 0:
            return _SimPointerMapResult(
                mapped=np.array([], dtype=np.int64),
                block_name="",
                is_partial=True,
                all_masked=True,
                origin_pointer_with_mask=origin_with_mask,
            )
            # return pointer_data, "", True, True
        offset, k, is_partial = self._map_pointer_data(
            pointer_data,
            DTypeEnum(pointer.dtype).byte_size())
        if mask is not None and mask.storage is not None:
            origin_with_mask[mask.storage.data] = pointer_data - offset
        return _SimPointerMapResult(
            mapped=pointer.storage.data - offset,
            block_name=k,
            is_partial=is_partial,
            all_masked=False,
            origin_pointer_with_mask=origin_with_mask,
        )

    def _boundry_check_and_map_pointers(
        self,
        pointer: Union["SimPointerTensor", "SimPointerScalarBase"],
        mask: Optional[SimTensor] = None,
    ) -> _SimPointerMapResult:
        assert pointer.storage is not None
        if mask is not None:
            assert mask.storage is not None
            assert mask.is_boolean(), "Mask must be boolean tensor"
            # check mask shape is broadcastable with pointer shape
            np.broadcast_shapes(mask.shape, pointer.shape)
            assert mask.ndim <= pointer.ndim
            mask_shape_rev = mask.shape[::-1]
            pointer_shape_rev = pointer.shape[::-1]
            # pointer can't be broadcasted.
            for j in range(mask.ndim):
                assert mask_shape_rev[j] <= pointer_shape_rev[j]

        res = self._map_pointer(pointer, mask)
        if not res.all_masked:
            assert not res.is_partial, "your pointer (with mask filtering) is out of range."
        return res

    def load(
        self,
        pointer: Union["SimPointerTensor", "SimPointerScalarBase"],
        mask: Optional[Union[SimTensor, bool]] = None,
        other: Optional[Union[SimTensor, float, int, bool]] = None,
        matrix_info: Optional[TensorSimIoMatrixInfo] = None,
    ) -> SimTensor:
        if pointer.storage is None:
            assert not isinstance(pointer, SimPointerScalarBase)
            return SimTensor(shape=pointer.shape, dtype=pointer.dtype)
        if mask is not None:
            if isinstance(mask, bool):
                mask_ten = zeros([], DTypeEnum.bool_)
                mask_ten.get_storage_checked().data[:] = mask
                mask = mask_ten
            mask = broadcast_to(mask, pointer.shape)
        map_result = self._boundry_check_and_map_pointers(pointer, mask)
        mapped_pointer_data = map_result.mapped
        block_name = map_result.block_name
        all_masked = map_result.all_masked
        # assert not all_masked
        if all_masked:
            # print("WARNING: all masked load detected.", pointer)
            out = empty(pointer.shape, pointer.dtype)
            output_data = out.get_storage_checked().data
            if other is not None:
                if isinstance(other, SimTensor):
                    output_data[:] = other.get_storage_checked().data
                else:
                    output_data[:] = other
            # if all masked, return empty tensor
            return out
        block_desc = self.memory_blocks[block_name]
        block_data = block_desc.get_data_view_checked()
        if DTypeEnum(pointer.dtype).to_numpy_dtype() != block_data.dtype:
            raise NotImplementedError

        if get_sim_mode() == TensorSimMode.LOGIC_ONLY and pointer.is_floating():
            output = empty([*pointer.shape, pointer.num_element], pointer.dtype, _internal_cached_empty=True)
            if pointer.num_element == 1:
                output = output[..., 0]
            return output
        else:
            output = empty([*pointer.shape, pointer.num_element], pointer.dtype)
        
        output_data = output.get_storage_checked().data
        data_raw = block_data.view(pointer.dtype_to_np(pointer.dtype)).reshape(
            -1, pointer.num_element)
        pointer_data = mapped_pointer_data.reshape(-1)
        loaded_indices = mapped_pointer_data[
            ..., None] * pointer.num_element + np.arange(pointer.num_element,
                                                         dtype=np.int32)
        # js don't support int64, so we use int32
        loaded_indices = loaded_indices.astype(np.int32)
        runner_ctx = get_pfl_runner_state()
        tensor_sim_ctx = get_tensorsim_context()
        if tensor_sim_ctx is not None and runner_ctx is not None:
            assert isinstance(
                runner_ctx.cur_expr,
                pfl.PFLCall), f"shouldn't happen, {type(runner_ctx.cur_expr)}"
            indices_for_record = loaded_indices.reshape(
                -1, pointer.num_element).copy()
            if mask is not None:
                indices_for_record[~mask.get_storage_checked().data.
                                   reshape(-1)] = -1
            tensor_sim_ctx._recorded_io_ops.append(
                TensorSimIoOp(True,
                              block_name,
                              indices_for_record.reshape(-1),
                              runner_ctx.cur_expr,
                              pointer.shape,
                              matrix_info=matrix_info))
        if mask is None:
            output_data[:] = data_raw[pointer_data].reshape(
                *pointer.shape, pointer.num_element)
        else:
            mask_view = mask.get_storage_checked().data.reshape(-1)
            output_data.reshape(-1, pointer.num_element)[mask_view] = data_raw[
                pointer_data[mask_view]]
            if other is not None:
                if isinstance(other, SimTensor):
                    output_data.reshape(
                        -1, pointer.num_element
                    )[~mask_view] = other.get_storage_checked().data.reshape(
                        -1, pointer.num_element)[~mask_view]
                else:
                    output_data.reshape(
                        -1, pointer.num_element)[~mask_view] = other
            loaded_indices.reshape(-1, pointer.num_element)[~mask_view] = -1
        loaded_indices = loaded_indices.reshape(-1, pointer.num_element)
        # print(block_desc, block_data, pointer_data, output_data)
        if pointer.num_element == 1:
            loaded_indices = loaded_indices[..., 0]
            output = output[..., 0]
        res_indices: dict[str,
                          np.ndarray] = output.get_storage_checked().indices
        block_indices = block_desc.indices
        for k in block_indices.keys():
            _, element_shape = block_desc.get_unflatten_inds(k)
            inds = block_indices[k]
            if mask is None:
                inds = inds[pointer_data]
            else:
                mask_view = mask.get_storage_checked().data.reshape(-1)
                new_inds = np.full(
                    (*pointer.shape, pointer.num_element, *element_shape),
                    -1,
                    dtype=np.int32,
                )
                new_inds_flatten = new_inds.reshape(-1, *element_shape)
                new_inds_flatten[mask_view] = inds[pointer_data[mask_view]]
                inds = new_inds
            if pointer.num_element == 1:
                inds = inds.reshape(*pointer.shape, *element_shape)
            else:
                inds = inds.reshape(*pointer.shape, pointer.num_element,
                                    *element_shape)
            res_indices[k] = inds.reshape(-1, *element_shape)
        res_indices[block_name] = loaded_indices
        return output

    def store(
        self,
        pointer: Union["SimPointerTensor", "SimPointerScalarBase"],
        value: Union[SimTensor, int, float],
        mask: Optional[Union[SimTensor, bool]] = None,
        matrix_info: Optional[TensorSimIoMatrixInfo] = None,
        atomic_op: Optional[Literal["add", "and", "cas", "max", "min", "or",
                                    "xor", "xchg"]] = None,
        atomic_cas_cmp: Optional[Union[SimTensor, int, float]] = None,
    ) -> Optional[SimTensor]:
        assert (pointer.num_element == 1
                ), "Pointer must be a single element pointer for now"
        if pointer.storage is None:
            if atomic_op is not None:
                old = SimTensor(pointer.shape, pointer.dtype)
                return old
            return
        if mask is not None:
            if isinstance(mask, bool):
                mask_ten = zeros([], DTypeEnum.bool_)
                mask_ten.get_storage_checked().data[:] = mask
                mask = mask_ten
            mask = broadcast_to(mask, pointer.shape)
        map_result = self._boundry_check_and_map_pointers(pointer, mask)
        mapped_pointer_data = map_result.mapped
        block_name = map_result.block_name
        all_masked = map_result.all_masked
        if all_masked:
            # return undefined.
            if atomic_op is not None:
                old = empty(pointer.shape, pointer.dtype)
                return old
            return
        block_desc = self.memory_blocks[block_name]
        # TODO use total byte instead of cnt.
        block_desc.write_cnt += 1
        block_data = block_desc.get_data_view_checked()
        runner_ctx = get_pfl_runner_state()
        tensor_sim_ctx = get_tensorsim_context()
        if tensor_sim_ctx is not None and runner_ctx is not None:
            assert isinstance(
                runner_ctx.cur_expr,
                pfl.PFLCall), f"shouldn't happen, {type(runner_ctx.cur_expr)}"
            indices_for_record = mapped_pointer_data.reshape(
                -1, pointer.num_element).copy()
            if mask is not None:
                indices_for_record[~mask.get_storage_checked().data.
                                   reshape(-1)] = -1
            tensor_sim_ctx._recorded_io_ops.append(
                TensorSimIoOp(False,
                              block_name,
                              indices_for_record.reshape(-1),
                              runner_ctx.cur_expr,
                              pointer.shape,
                              matrix_info=matrix_info,
                              atomic_op=atomic_op))

        if DTypeEnum(pointer.dtype).to_numpy_dtype() != block_data.dtype:
            raise NotImplementedError
        if isinstance(value, SimTensor):
            # WARNING: only write indices when value is Tensor.
            # otherwise unchanged.
            if value.shape != pointer.shape:
                value = broadcast_to(value, pointer.shape)
            if value.storage is None:
                return
            if not block_desc.indices:
                # lazy create indices based on first store value
                for k in value.storage.indices.keys():
                    value_inds, element_shape = value.storage.get_unflatten_inds(
                        k)
                    block_desc.indices[k] = np.full(
                        [*block_data.shape, *element_shape],
                        -1,
                        dtype=np.int32).reshape(-1, *element_shape)
            else:
                # validate indices
                if len(block_desc.indices) != len(value.storage.indices):
                    raise ValueError(
                        "Cannot store value with different number of indices")
                for k in value.storage.indices.keys():
                    _, element_shape = value.storage.get_unflatten_inds(k)
                    _, stored_element_shape = block_desc.get_unflatten_inds(k)
                    if element_shape != stored_element_shape:
                        raise ValueError(
                            f"Indices shape mismatch: {element_shape} vs {stored_element_shape}"
                        )
        if not isinstance(value, SimTensor):
            value_ten = full(pointer.shape, value, pointer.dtype)
            value = value_ten
        assert value.storage is not None
        data_raw = block_data.view(pointer.dtype_to_np(
            pointer.dtype)).reshape(-1)
        pointer_data = mapped_pointer_data.reshape(-1)
        stored_data_raw = value.storage.data.reshape(-1)
        old = zeros(pointer.shape, pointer.dtype)

        if mask is None:
            pointer_data_masked = pointer_data
            stored_data_raw_masked = stored_data_raw
            old_data = data_raw[pointer_data].reshape(pointer.shape)
            old.get_storage_checked().data[:] = old_data
        else:
            mask_view = mask.get_storage_checked().data.reshape(-1)
            pointer_data_masked = pointer_data[mask_view]
            stored_data_raw_masked = stored_data_raw[mask_view]
            old_data = old.get_storage_checked().data.reshape(-1)
            old_data[mask_view] = data_raw[pointer_data_masked]

        if atomic_op == "add":
            data_raw[pointer_data_masked] += stored_data_raw_masked
        elif atomic_op == "and":
            data_raw[pointer_data_masked] &= stored_data_raw_masked
        elif atomic_op == "cas":
            assert atomic_cas_cmp is not None, "CAS operation requires atomic_cas_cmp"
            if isinstance(atomic_cas_cmp, SimTensor):
                atomic_cas_cmp_val = atomic_cas_cmp.get_storage_checked(
                ).data
            else:
                atomic_cas_cmp_val = atomic_cas_cmp
            store_cas = np.where(
                data_raw[pointer_data_masked] == atomic_cas_cmp_val,
                stored_data_raw_masked, data_raw[pointer_data_masked])
            data_raw[pointer_data_masked] = store_cas
        elif atomic_op == "max":
            data_raw[pointer_data_masked] = np.maximum(data_raw[pointer_data_masked],
                                                stored_data_raw_masked)
        elif atomic_op == "min":
            data_raw[pointer_data_masked] = np.minimum(data_raw[pointer_data_masked],
                                                stored_data_raw_masked)
        elif atomic_op == "or":
            data_raw[pointer_data_masked] |= stored_data_raw_masked
        elif atomic_op == "xor":
            data_raw[pointer_data_masked] ^= stored_data_raw_masked
        elif atomic_op == "xchg":
            # exchange operation, swap the data
            tmp = data_raw[pointer_data_masked].copy()
            data_raw[pointer_data_masked] = stored_data_raw_masked
            stored_data_raw_masked[:] = tmp
        else:
            data_raw[pointer_data_masked] = stored_data_raw_masked
        # all inds are stored in flatten array.
        for k, inds in block_desc.indices.items():
            if k in value.storage.indices:
                value_inds = value.storage.indices[k]
                if mask is None:
                    inds[pointer_data] = value_inds.reshape(-1)
                else:
                    mask_view = mask.get_storage_checked().data.reshape(-1)
                    inds[pointer_data[mask_view]] = value_inds.reshape(
                        -1)[mask_view]
        return old


def create_sim_memory_single(name: str, data: np.ndarray):
    return create_sim_memory({name: data})


def create_sim_memory(data_dict: dict[str, np.ndarray]):
    block_dict: dict[str, MemoryBlockDesc] = {}
    for k, v in data_dict.items():
        block_dict[k] = MemoryBlockDesc(size=v.size,
                                        dtype=DTypeEnum.from_numpy_dtype(
                                            v.dtype))
    total_contiguous_size = assign_random_memory_block_offset(block_dict)
    total_data = np.empty(total_contiguous_size, dtype=np.uint8)
    for k, v in data_dict.items():
        desc = block_dict[k]
        data_view = total_data[desc.byte_offset:desc.byte_offset +
                               desc.get_byte_size()]
        data_view[:] = v.view(np.uint8).reshape(-1)
        desc.data_view = data_view.view(v.dtype).reshape(v.shape)
    return SimMemoryStorage(total_data, memory_blocks=block_dict)


@dataclasses.dataclass
class SimPointerTensor(SimTensorBase):
    # pointer tensor is always int64*, dtype is element type.
    # pointer of pointer is unsupported currently.
    # num_element: for vector load, only used when we directly target to raw backend, e.g. CUDA.
    num_element: int = 1
    memory_storage: Optional[SimMemoryStorage] = None

    def __post_init__(self):
        if self.storage is not None:
            assert (self.storage.data.dtype == np.int64
                    ), "Pointer tensor storage must be of int64 type"

    def to_meta_tensor(self):
        return dataclasses.replace(self, storage=None, memory_storage=None)

    def is_pointer(self) -> bool:
        return True

    def get_memory_storage_checked(self) -> SimMemoryStorage:
        if self.memory_storage is None:
            raise ValueError("Pointer tensor does not have a memory storage")
        return self.memory_storage

    def load(
        self,
        mask: Optional[Union[bool, SimTensor]] = None,
        other: Optional[Union[SimTensor, float, int, bool]] = None,
        matrix_info: Optional[TensorSimIoMatrixInfo] = None,
    ) -> SimTensor:
        if self.memory_storage is None:
            # create a meta tensor from self
            return SimTensor(shape=self.shape, dtype=self.dtype, storage=None)
        return self.memory_storage.load(self,
                                        mask,
                                        other,
                                        matrix_info=matrix_info)

    def store(
        self,
        value: Union[SimTensor, int, float],
        mask: Optional[Union[bool, SimTensor]] = None,
        matrix_info: Optional[TensorSimIoMatrixInfo] = None,
        atomic_op: Optional[Literal["add", "and", "cas", "max", "min", "or",
                                    "xor", "xchg"]] = None,
        atomic_cas_cmp: Optional[Union[SimTensor, int, float]] = None,
    ):
        if self.memory_storage is None:
            old = SimTensor(shape=self.shape, dtype=self.dtype)
            return old
        return self.memory_storage.store(self,
                                         value,
                                         mask,
                                         matrix_info=matrix_info,
                                         atomic_op=atomic_op,
                                         atomic_cas_cmp=atomic_cas_cmp)

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
        res = self._binary_base(cast(Self, other), BinOpType.SUB, False)
        assert isinstance(res, SimPointerTensor)
        return res

    def __isub__(self, other: Union[SimTensor, int]) -> Self:
        res = self._binary_base(cast(Self, other), BinOpType.SUB, False, True)
        assert isinstance(res, SimPointerTensor)
        return res

    def __rsub__(self, other: Union[SimTensor, int]) -> Self:
        res = self._binary_base(cast(Self, other), BinOpType.SUB, True)
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
            assert (self.storage.data.dtype == np.int64
                    ), "Pointer tensor storage must be of int64 type"

    def _to_pointer_tensor(self) -> SimPointerTensor:
        return SimPointerTensor(
            shape=self.shape,
            dtype=self.dtype,
            storage=self.storage,
            num_element=self.num_element,
            memory_storage=self.memory_storage,
        )

    def to_meta_tensor(self):
        return dataclasses.replace(self, storage=None, memory_storage=None)

    def is_pointer(self) -> bool:
        return True

    def get_memory_storage_checked(self) -> SimMemoryStorage:
        if self.memory_storage is None:
            raise ValueError("Pointer tensor does not have a memory storage")
        return self.memory_storage

    def _scalar_bin_op(self, other: int, op_type: BinOpType) -> Self:
        if self.storage is None:
            return dataclasses.replace(self)
        lfs = self.storage.data.item()
        if op_type == BinOpType.ADD:
            new_res = lfs + other
        elif op_type == BinOpType.SUB:
            new_res = lfs - other
        else:
            raise ValueError(f"Unsupported binary operation: {op_type}")
        new_storage = dataclasses.replace(self.storage,
                                          data=np.array(new_res,
                                                        dtype=np.int64))
        return dataclasses.replace(self, storage=new_storage)

    @overload
    def __add__(self, other: SimTensor) -> SimPointerTensor:
        ...

    @overload
    def __add__(self, other: int) -> Self:
        ...

    def __add__(self, other: Union[SimTensor,
                                   int]) -> Union[Self, SimPointerTensor]:
        if isinstance(other, int):
            return self._scalar_bin_op(other, BinOpType.ADD)
        res = self._to_pointer_tensor() + other
        return res

    def __iadd__(self, other: int) -> Self:
        return self._scalar_bin_op(other, BinOpType.ADD)

    @overload
    def __radd__(self, other: SimTensor) -> SimPointerTensor:
        ...

    @overload
    def __radd__(self, other: int) -> Self:
        ...

    def __radd__(
            self, other: Union[SimTensor,
                               int]) -> Union[Self, SimPointerTensor]:
        if isinstance(other, int):
            return self._scalar_bin_op(other, BinOpType.ADD)
        res = other + self._to_pointer_tensor()
        return res

    @overload
    def __sub__(self, other: SimTensor) -> SimPointerTensor:
        ...

    @overload
    def __sub__(self, other: int) -> Self:
        ...

    def __sub__(self, other: Union[SimTensor,
                                   int]) -> Union[Self, SimPointerTensor]:
        if isinstance(other, int):
            return self._scalar_bin_op(other, BinOpType.SUB)
        res = self._to_pointer_tensor() - other
        return res

    def __isub__(self, other: int) -> Self:
        return self._scalar_bin_op(other, BinOpType.SUB)

    @overload
    def __rsub__(self, other: SimTensor) -> SimPointerTensor:
        ...

    @overload
    def __rsub__(self, other: int) -> Self:
        ...

    def __rsub__(
            self, other: Union[SimTensor,
                               int]) -> Union[Self, SimPointerTensor]:
        if isinstance(other, int):
            return self._scalar_bin_op(other, BinOpType.SUB)
        res = other - self._to_pointer_tensor()
        return res


@dataclasses.dataclass
class SimPointerScalar(SimPointerScalarBase):

    def load(self,
             mask: Optional[bool] = None,
             other: Optional[Union[float, int]] = None) -> Union[float, int]:
        if self.memory_storage is None or self.storage is None:
            raise ValueError("can't load scalar meta")
        res = self.memory_storage.load(self, mask, other)
        assert res.is_scalar()
        assert res.storage is not None
        return res.storage.data.item()

    def store(
        self,
        value: Union[float, int],
        mask: Optional[bool] = None,
        atomic_op: Optional[Literal["add", "and", "cas", "max", "min", "or",
                                    "xor", "xchg"]] = None,
        atomic_cas_cmp: Optional[Union[SimTensor, int, float]] = None,
    ):
        if self.memory_storage is None:
            return
        res = self.memory_storage.store(self,
                                        value,
                                        mask,
                                        atomic_op=atomic_op,
                                        atomic_cas_cmp=atomic_cas_cmp)
        assert isinstance(res, SimTensor)
        return res.get_storage_checked().data.item()


def create_pointer_tensor_meta(dtype: int,
                               shape: list[int],
                               num_element: int = 1) -> SimPointerTensor:
    """Create a pointer tensor with a single scalar value."""
    return SimPointerTensor(shape=shape,
                            dtype=dtype,
                            num_element=num_element,
                            storage=None)


def create_pointer_scalar_meta(dtype: int,
                               num_element: int = 1) -> SimPointerScalar:
    """Create a pointer tensor with a single scalar value."""
    return SimPointerScalar(shape=[], dtype=dtype, num_element=num_element)


def create_pointer_tensor(
    dtype: int,
    ptr: Union[np.ndarray, int],
    memory: SimMemoryStorage,
    num_element: int = 1,
) -> SimPointerTensor:
    """Create a pointer tensor with a single scalar value."""
    if not isinstance(ptr, np.ndarray):
        assert ptr >= 0
    assert (
        memory is not None
    ), "Memory storage must be provided for pointer tensors in sim mode"
    if isinstance(ptr, np.ndarray):
        assert ptr.dtype == np.int64
        data = ptr
    else:
        data = np.array(ptr, dtype=np.int64)
    storage = SimTensorStorage(data=data)
    mapped_data, mapped_key, _ = memory._map_pointer_data(
        data,
        DTypeEnum(dtype).byte_size())
    assert (DTypeEnum(dtype) == memory.memory_blocks[mapped_key].dtype
            ), "Pointer tensor dtype must match memory storage dtype"
    return SimPointerTensor(
        shape=list(data.shape),
        dtype=dtype,
        num_element=num_element,
        storage=storage,
        memory_storage=memory,
    )


def create_pointer_scalar(dtype: int,
                          ptr: int,
                          memory: SimMemoryStorage,
                          num_element: int = 1) -> SimPointerScalar:
    """Create a pointer tensor with a single scalar value."""
    assert ptr >= 0
    assert (
        memory is not None
    ), "Memory storage must be provided for pointer tensors in sim mode"
    data = np.array(ptr, dtype=np.int64)
    storage = SimTensorStorage(data=data)
    mapped_data, mapped_key, _ = memory._map_pointer_data(
        data,
        DTypeEnum(dtype).byte_size())
    assert (DTypeEnum(dtype) == memory.memory_blocks[mapped_key].dtype
            ), "Pointer tensor dtype must match memory storage dtype"
    return SimPointerScalar(
        shape=[],
        dtype=dtype,
        num_element=num_element,
        storage=storage,
        memory_storage=memory,
    )


@dataclasses.dataclass
class SimTensorBlockPointer:
    # like triton tensor descriptor
    base: SimPointerScalar
    shape: list[int]
    strides: list[int]
    block_shape: list[int]
    # state
    offset: list[int]

    def __repr__(self):
        return (f"SimTensorBlockPointer(shape={self.shape}, "
                f"strides={self.strides}, block_shape={self.block_shape}, "
                f"offset={self.offset})")

    def __post_init__(self):
        assert (
            len(self.shape) == len(self.strides) == len(
                self.block_shape) == len(self.offset)
        ), "base ndim must match shape, strides, block_shape and offset length"

    def to_meta_tensor(self):
        return dataclasses.replace(self, base=self.base.to_meta_tensor())

    def clone(self) -> Self:
        """Clone the SimTensorBlockPointer."""
        return dataclasses.replace(
            self,
            base=self.base.clone(),
            offset=list(self.offset),
        )

    def advance(self, offsets: list[int]) -> Self:
        if self.base.storage is None:
            # meta advance
            return dataclasses.replace(self)
        assert len(offsets) == len(
            self.shape), "Offsets must match shape length"
        new_offset = [
            self.offset[i] + offsets[i] for i in range(len(self.offset))
        ]
        return dataclasses.replace(self, offset=new_offset)

    def get_current_pointer_tensor_and_mask(
        self,
        offsets: Optional[list[int]] = None
    ) -> tuple[SimPointerTensor, SimTensor]:
        pointer_base = self.base
        if offsets is None:
            offsets = self.offset
        else:
            assert len(offsets) == len(
                self.offset), "Offsets must match current offset length"
            offsets = [
                self.offset[i] + offsets[i] for i in range(len(self.offset))
            ]
        if get_sim_mode() == TensorSimMode.LOGIC_ONLY:
            ctx = get_tensorsim_context_checked()
            if tuple(self.block_shape) not in ctx._logic_only_cached_offsets:
                offsets_blocks = np.meshgrid(
                    *[
                        np.arange(0, s, dtype=np.int64)
                        for s in self.block_shape
                    ],
                    indexing="ij",
                )

                ctx._logic_only_cached_offsets[tuple(self.block_shape)] = offsets_blocks
            else:
                offsets_blocks = ctx._logic_only_cached_offsets[tuple(self.block_shape)]
        else: 
            offsets_blocks = np.meshgrid(
                *[
                    np.arange(0, s, dtype=np.int64)
                    for s in self.block_shape
                ],
                indexing="ij",
            )
        offset_data = cast(
            np.ndarray,
            sum((ob + o) * s for ob, o, s in zip(offsets, offsets_blocks, self.strides)))
        mask_arr = np.logical_and.reduce([
            np.logical_and(o >= 0, o < s)
            for o, s in zip(offsets_blocks, self.shape)
        ])
        pointer_tensor = pointer_base + from_numpy(offset_data)
        return pointer_tensor, from_numpy(mask_arr)

    def load(
        self,
        offsets: list[int],
        other: Optional[Union[SimTensor, float, int, bool]] = None,
    ) -> SimTensor:
        if self.base.storage is None:
            # meta load
            return SimTensor(shape=self.block_shape, dtype=self.base.dtype)
        cur_pointer_tensor, mask_tensor = self.get_current_pointer_tensor_and_mask(
            offsets)
        matrix_info = None
        if len(self.shape) == 2:
            matrix_info = TensorSimIoMatrixInfo(
                offsets=(offsets[0] + self.offset[0],
                         offsets[1] + self.offset[1]),
                shape=(self.block_shape[0], self.block_shape[1]),
            )
        return cur_pointer_tensor.load(mask=mask_tensor,
                                       other=other,
                                       matrix_info=matrix_info)

    def store(
        self,
        offsets: list[int],
        value: Union[SimTensor, int, float],
        atomic_op: Optional[Literal["add", "and", "cas", "max", "min", "or",
                                    "xor", "xchg"]] = None,
        atomic_cas_cmp: Optional[Union[SimTensor, int, float]] = None,
    ):
        if self.base.storage is None:
            # meta load
            old = SimTensor(shape=self.block_shape,
                            dtype=self.base.dtype,
                            storage=None)
            return old
        cur_pointer_tensor, mask_tensor = self.get_current_pointer_tensor_and_mask(
            offsets)
        matrix_info = None
        if len(self.shape) == 2:
            matrix_info = TensorSimIoMatrixInfo(
                offsets=(offsets[0] + self.offset[0],
                         offsets[1] + self.offset[1]),
                shape=(self.block_shape[0], self.block_shape[1]),
            )
        return cur_pointer_tensor.store(mask=mask_tensor,
                                        value=value,
                                        matrix_info=matrix_info,
                                        atomic_op=atomic_op,
                                        atomic_cas_cmp=atomic_cas_cmp)


def create_tensor_block_pointer(
    base: SimPointerScalar,
    shape: list[int],
    strides: list[int],
    block_shape: list[int],
    offset: Optional[list[int]] = None,
) -> SimTensorBlockPointer:
    """Create a tensor block pointer."""
    if offset is None:
        offset = [0] * len(shape)
    assert (
        len(shape) == len(strides) == len(block_shape) == len(offset)
    ), "Shape, strides, block_shape and offset must have the same length"
    return SimTensorBlockPointer(base=base,
                                 shape=shape,
                                 strides=strides,
                                 block_shape=block_shape,
                                 offset=offset)


def create_tensor_block_pointer_meta(
    base: SimPointerScalar,
    ndim: int,
    block_shape: list[int],
) -> SimTensorBlockPointer:
    """Create a tensor block pointer."""
    assert base.storage is None, "Base pointer must be a meta pointer"
    shape = [1] * ndim
    strides = [1] * ndim
    assert len(block_shape) == ndim, "Block shape must match ndim"
    offset = [0] * ndim
    return SimTensorBlockPointer(base=base,
                                 shape=shape,
                                 strides=strides,
                                 block_shape=block_shape,
                                 offset=offset)
