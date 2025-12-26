from collections.abc import Sequence
import enum
from functools import partial
from typing import TYPE_CHECKING, Any, Optional, Type, Union, cast, overload
import dataclasses
import numpy as np 
from tensorpc.core import pfl
import contextlib 
import contextvars
from typing_extensions import Self

from tensorpc.core.pfl.pfl_ast import BinOpType, CompareType, UnaryOpType
if TYPE_CHECKING:
    from .memory import SimMemoryStorage

class NumpyReduceType(enum.IntEnum):
    SUM = 0
    MEAN = 1
    MAX = 2
    MIN = 3
    PROD = 4
    ARGMAX = 5
    ARGMIN = 6


class DTypeClassEnum(enum.IntEnum):
    floating = 0
    integer = 1
    unsigned = 2
    boolean = 3

class DTypeEnum(enum.IntEnum):
    # js/cumm/triton supported types
    float32 = 0
    float64 = 4
    int8 = 3
    int16 = 2
    int32 = 1
    int64 = 8
    uint8 = 6
    uint16 = 9
    uint32 = 10
    uint64 = 11
    bool_ = 5
    # cumm/triton supported types
    float16 = 7
    bfloat16 = 12

    # triton supported types
    float8e5 = 13
    float8e5b16 = 14
    float8e4nv = 15
    float8e4b8 = 16
    float8e4b15 = 17

    @staticmethod
    def dtype_promotion(*args: int):
        max_priority = -1
        max_dtype = -1
        for arg in args:
            dtype = DTypeEnum(arg)
            priority = _calcuate_dtype_priority(dtype)
            if priority > max_priority:
                max_priority = priority
                max_dtype = dtype
        if max_dtype == -1:
            raise ValueError("No valid dtype provided for promotion")
        return max_dtype

    def is_floating_type(self) -> bool:
        dtype_cls = _DTYPE_TO_DTYPE_CLS[self]
        return dtype_cls == DTypeClassEnum.floating

    def is_unsigned_type(self) -> bool:
        dtype_cls = _DTYPE_TO_DTYPE_CLS[self]
        return dtype_cls == DTypeClassEnum.unsigned

    def is_integer_type(self) -> bool:
        dtype_cls = _DTYPE_TO_DTYPE_CLS[self]
        return dtype_cls == DTypeClassEnum.integer

    def is_boolean_type(self) -> bool:
        dtype_cls = _DTYPE_TO_DTYPE_CLS[self]
        return dtype_cls == DTypeClassEnum.boolean

    def bit_size(self) -> int:
        return _DTYPE_TO_NUM_BITS[self]

    def byte_size(self) -> int:
        assert self.bit_size() % 8 == 0
        return self.bit_size() // 8

    @classmethod
    def from_numpy_dtype(cls, np_dtype: np.dtype) -> 'DTypeEnum':
        """
        Convert a numpy dtype to a DTypeEnum.
        """
        if np_dtype not in NP_DTYPE_TO_PPCL:
            raise ValueError(f"Unsupported numpy dtype: {np_dtype}")
        return NP_DTYPE_TO_PPCL[np_dtype]

    def to_numpy_dtype(self) -> np.dtype:
        """
        Convert a DTypeEnum to a numpy dtype.
        """
        dtype_mapping = get_sim_dtype_mapping()
        if self in dtype_mapping:
            return dtype_mapping[self]
        if self not in PPCL_TO_NP_DTYPE:
            raise ValueError(f"Unsupported DTypeEnum: {self}")
        return PPCL_TO_NP_DTYPE[self]

_DTYPE_TO_DTYPE_CLS = {
    DTypeEnum.float64: DTypeClassEnum.floating,
    DTypeEnum.float32: DTypeClassEnum.floating,
    DTypeEnum.float16: DTypeClassEnum.floating,
    DTypeEnum.bfloat16: DTypeClassEnum.floating,
    DTypeEnum.float8e5: DTypeClassEnum.floating,
    DTypeEnum.float8e5b16: DTypeClassEnum.floating,
    DTypeEnum.float8e4nv: DTypeClassEnum.floating,
    DTypeEnum.float8e4b8: DTypeClassEnum.floating,
    DTypeEnum.float8e4b15: DTypeClassEnum.floating,
    DTypeEnum.int64: DTypeClassEnum.integer,
    DTypeEnum.int32: DTypeClassEnum.integer,
    DTypeEnum.int16: DTypeClassEnum.integer,
    DTypeEnum.int8: DTypeClassEnum.integer,
    DTypeEnum.uint64: DTypeClassEnum.unsigned,
    DTypeEnum.uint32: DTypeClassEnum.unsigned,
    DTypeEnum.uint16: DTypeClassEnum.unsigned,
    DTypeEnum.uint8: DTypeClassEnum.unsigned,
    DTypeEnum.bool_: DTypeClassEnum.boolean,
}

_DTYPE_TO_NUM_BITS = {
    DTypeEnum.float64: 64,
    DTypeEnum.float32: 32,
    DTypeEnum.float16: 16,
    DTypeEnum.bfloat16: 16,
    DTypeEnum.float8e5: 8,
    DTypeEnum.float8e5b16: 8,

    DTypeEnum.float8e4nv: 8,
    DTypeEnum.float8e4b8: 8,
    DTypeEnum.float8e4b15: 8,
    DTypeEnum.int64: 64,
    DTypeEnum.int32: 32,
    DTypeEnum.int16: 16,
    DTypeEnum.int8: 8,
    DTypeEnum.uint64: 64,
    DTypeEnum.uint32: 32,
    DTypeEnum.uint16: 16,
    DTypeEnum.uint8: 8,
    DTypeEnum.bool_: 8,  # bool is often represented as 8 bits in many systems
}

# follow triton's promotion rules
_DTYPE_TO_PROMOTION_FLOAT_PRIORITY = {
    DTypeEnum.float64: 9,
    DTypeEnum.float32: 8,
    DTypeEnum.float16: 7,
    DTypeEnum.bfloat16: 6, 

    DTypeEnum.float8e5: 5,
    DTypeEnum.float8e5b16: 4,
    DTypeEnum.float8e4nv: 3,
    DTypeEnum.float8e4b8: 2,
    DTypeEnum.float8e4b15: 1,
}

_DTYPE_TO_PROMOTION_SIGNED_PRIORITY = {
    DTypeEnum.int64: 9,
    DTypeEnum.int32: 8,
    DTypeEnum.int16: 7,
    DTypeEnum.int8: 6, 
}

_DTYPE_TO_PROMOTION_UNSIGNED_PRIORITY = {
    DTypeEnum.uint64: 9,
    DTypeEnum.uint32: 8,
    DTypeEnum.uint16: 7,
    DTypeEnum.uint8: 6, 
}

_DTYPE_CLS_PROMOTION_PRIORITY = {
    DTypeClassEnum.floating: 9,
    DTypeClassEnum.unsigned: 8,
    DTypeClassEnum.integer: 7,
    DTypeClassEnum.boolean: 6,
}

_DTYPE_CLS_TO_PROMOTION_PRIORITY_DICT = {
    DTypeClassEnum.floating: _DTYPE_TO_PROMOTION_FLOAT_PRIORITY,
    DTypeClassEnum.unsigned: _DTYPE_TO_PROMOTION_UNSIGNED_PRIORITY,
    DTypeClassEnum.integer: _DTYPE_TO_PROMOTION_SIGNED_PRIORITY,
    DTypeClassEnum.boolean: {
        DTypeEnum.bool_: 1,
    },
}


NP_DTYPE_TO_PPCL = {
    np.dtype(np.float32): DTypeEnum.float32,
    np.dtype(np.float64): DTypeEnum.float64,
    np.dtype(np.int8): DTypeEnum.int8,
    np.dtype(np.int16): DTypeEnum.int16,
    np.dtype(np.int32): DTypeEnum.int32,
    np.dtype(np.int64): DTypeEnum.int64,
    np.dtype(np.float16): DTypeEnum.float16,

    np.dtype(np.uint8): DTypeEnum.uint8,
    np.dtype(np.uint16): DTypeEnum.uint16,
    np.dtype(np.uint32): DTypeEnum.uint32,
    np.dtype(np.uint64): DTypeEnum.uint64,
    np.dtype(np.bool_): DTypeEnum.bool_,
}

PPCL_TO_NP_DTYPE: dict[DTypeEnum, np.dtype] = {v: k for k, v in NP_DTYPE_TO_PPCL.items()}

def _calcuate_dtype_priority(dtype: DTypeEnum) -> int:
    dtype_cls = _DTYPE_TO_DTYPE_CLS[dtype]
    cls_priority = _DTYPE_CLS_PROMOTION_PRIORITY[dtype_cls] * 100
    dtype_priority_dict = _DTYPE_CLS_TO_PROMOTION_PRIORITY_DICT[dtype_cls]
    dtype_priority = dtype_priority_dict[dtype]
    return cls_priority + dtype_priority

def _get_default_sim_dtype_mapping() -> dict[DTypeEnum, np.dtype]:
    """
    Returns the default dtype mapping for tensor simulation.
    This is used to override the dtype run in numpy.
    """
    return {
        DTypeEnum.float8e5: np.dtype(np.float32),
        DTypeEnum.float8e5b16: np.dtype(np.float32),
        DTypeEnum.float8e4nv: np.dtype(np.float32),
        DTypeEnum.float8e4b8: np.dtype(np.float32),
        DTypeEnum.float8e4b15: np.dtype(np.float32),
        DTypeEnum.float16: np.dtype(np.float32),
        DTypeEnum.bfloat16: np.dtype(np.float32),
    }

class TensorSimMode(enum.IntEnum):
    FULL = 0
    # only run logic (integer/boolean) ops, skip floating point ops such as matmul.
    # raise error if you convert float to int/boolean.
    # WARNING: we will reuse float buffer if possible, user shouldn't rely on float values.
    LOGIC_ONLY = 1
    # only run shape/dtype inference.
    META_ONLY = 2

@dataclasses.dataclass 
class TensorSimConfig:
    default_int_dtype: DTypeEnum = DTypeEnum.int64
    default_float_dtype: DTypeEnum = DTypeEnum.float32
    record_memory: bool = False
    mode: TensorSimMode = TensorSimMode.FULL
    # used to override dtype runned in numpy.
    # e.g. {DTypeEnum.float8: np.float32} can be used
    # to run float8 as float32 in numpy.
    dtype_mapping: dict[DTypeEnum, np.dtype] = dataclasses.field(
        default_factory=_get_default_sim_dtype_mapping)

    @property 
    def meta_only(self):
        return self.mode == TensorSimMode.META_ONLY

@dataclasses.dataclass 
class TensorSimIoMatrixInfo:
    # only used for better debug info.
    # block ptr may load a block matrix
    offsets: tuple[int, int]
    shape: tuple[int, int]


@dataclasses.dataclass 
class TensorSimIoOp:
    is_load: bool
    name: str
    io_indices: np.ndarray
    ast_node: pfl.PFLCall
    shape: list[int]
    matrix_info: Optional[TensorSimIoMatrixInfo] = None
    atomic_op: Optional[str] = None

class TensorSimContext:
    def __init__(self, grid_id: Sequence[int], grid_size: Sequence[int], 
            simd_group_id: Optional[Sequence[int]] = None, simd_group_size: Optional[Sequence[int]] = None, 
            cfg: Optional[TensorSimConfig] = None, global_mem: Optional["SimMemoryStorage"] = None):
        self.grid_size = grid_size
        self.simd_group_size = simd_group_size
        self.grid_id = grid_id
        self.simd_group_id = simd_group_id
        # required for ptr of ptr.
        self.global_mem = global_mem

        if cfg is None:
            cfg = TensorSimConfig()
        self.cfg = cfg

        self._recorded_io_ops: list[TensorSimIoOp] = []

        self._logic_only_cached_memory: dict[tuple[tuple[int, ...], np.dtype], Any] = {}
        self._logic_only_cached_offsets: dict[tuple[int, ...], Any] = {}

    def get_flatted_grid_id(self) -> int:
        """
        Returns the flatted grid id, which is the linear index of the grid.
        """
        assert len(self.grid_id) == len(self.grid_size), \
            f"Grid id {self.grid_id} and size {self.grid_size} must have same length."
        res = 0
        for idx, size in zip(self.grid_id, self.grid_size):
            res = res * size + idx
        return res

    def set_grid_id(self, grid_id: Sequence[int]):
        # TODO simd group id set
        assert len(grid_id) == len(self.grid_size), \
            f"Grid id {grid_id} and size {self.grid_size} must have same length."
        self.grid_id = grid_id

    def _cached_empty(self, shape: Sequence[int], dtype: np.dtype):
        key = (tuple(shape), dtype)
        if key not in self._logic_only_cached_memory:
            self._logic_only_cached_memory[key] = np.zeros(shape, dtype=dtype)
        return self._logic_only_cached_memory[key]

_TENSOR_SIM_CONTEXT: contextvars.ContextVar[
    Optional[TensorSimContext]] = contextvars.ContextVar("TensorSimContext",
                                                        default=None)


@contextlib.contextmanager
def enter_tensorsim_context(grid_id: Sequence[int], grid_size: Sequence[int], 
            simd_group_id: Optional[Sequence[int]] = None, simd_group_size: Optional[Sequence[int]] = None, 
            cfg: Optional[TensorSimConfig] = None, global_mem: Optional["SimMemoryStorage"] = None):
    for idx, size in zip(grid_id, grid_size):
        assert idx < size, f"Grid index {idx} must be less than grid size {size}"
    ctx = TensorSimContext(
        grid_id=grid_id,
        grid_size=grid_size,
        simd_group_id=simd_group_id,
        simd_group_size=simd_group_size,
        cfg=cfg,
        global_mem=global_mem,
    )
    token = _TENSOR_SIM_CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _TENSOR_SIM_CONTEXT.reset(token)


def get_tensorsim_context_checked():
    ctx = _TENSOR_SIM_CONTEXT.get()
    if ctx is None:
        raise ValueError("not in parse context")
    return ctx


def get_tensorsim_context():
    ctx = _TENSOR_SIM_CONTEXT.get()
    return ctx


def get_default_int_dtype():
    ctx = _TENSOR_SIM_CONTEXT.get()
    if ctx is None:
        return DTypeEnum.int64
    return ctx.cfg.default_int_dtype

def get_default_float_dtype():
    ctx = _TENSOR_SIM_CONTEXT.get()
    if ctx is None:
        return DTypeEnum.float32
    return ctx.cfg.default_float_dtype

def get_default_base_dtype(type: Union[Type[int], Type[float], Type[bool]]):
    ctx = _TENSOR_SIM_CONTEXT.get()
    if issubclass(type, bool):
        return DTypeEnum.bool_ 
    elif issubclass(type, int):
        if ctx is None:
            return DTypeEnum.int64
        return ctx.cfg.default_int_dtype
    else:
        if ctx is None:
            return DTypeEnum.float32
        return ctx.cfg.default_float_dtype

def get_sim_dtype_mapping():
    ctx = _TENSOR_SIM_CONTEXT.get()
    if ctx is None:
        return {}
    if ctx.cfg.dtype_mapping is None:
        return {}
    return ctx.cfg.dtype_mapping


def get_flush_sim_io_ops() -> list[TensorSimIoOp]:
    ctx = _TENSOR_SIM_CONTEXT.get()
    if ctx is None:
        return []
    res = ctx._recorded_io_ops.copy()
    ctx._recorded_io_ops.clear()
    return res

def get_sim_mode() -> TensorSimMode:
    ctx = _TENSOR_SIM_CONTEXT.get()
    if ctx is None:
        return TensorSimMode.FULL
    return ctx.cfg.mode