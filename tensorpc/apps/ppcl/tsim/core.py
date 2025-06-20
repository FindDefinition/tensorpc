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
    np.float32: DTypeEnum.float32,
    np.float64: DTypeEnum.float64,
    np.int8: DTypeEnum.int8,
    np.int16: DTypeEnum.int16,
    np.int32: DTypeEnum.int32,
    np.int64: DTypeEnum.int64,
    np.float16: DTypeEnum.float16,

    np.uint8: DTypeEnum.uint8,
    np.uint16: DTypeEnum.uint16,
    np.uint32: DTypeEnum.uint32,
    np.uint64: DTypeEnum.uint64,
    np.bool_: DTypeEnum.bool_,
}

PPCL_TO_NP_DTYPE: dict[DTypeEnum, np.dtype] = {v: k for k, v in NP_DTYPE_TO_PPCL.items()}

def _calcuate_dtype_priority(dtype: DTypeEnum) -> int:
    dtype_cls = _DTYPE_TO_DTYPE_CLS[dtype]
    cls_priority = _DTYPE_CLS_PROMOTION_PRIORITY[dtype_cls] * 100
    dtype_priority_dict = _DTYPE_CLS_TO_PROMOTION_PRIORITY_DICT[dtype_cls]
    dtype_priority = dtype_priority_dict[dtype]
    return cls_priority + dtype_priority

@dataclasses.dataclass 
class TensorSimConfig:
    meta_only: bool = False
    default_int_dtype: DTypeEnum = DTypeEnum.int64
    default_float_dtype: DTypeEnum = DTypeEnum.float32
    record_memory: bool = False

class TensorSimContext:
    def __init__(self, grid_id: Sequence[int], grid_size: Sequence[int], 
            simd_group_id: Optional[Sequence[int]] = None, simd_group_size: Optional[Sequence[int]] = None, 
            cfg: Optional[TensorSimConfig] = None):
        self.grid_size = grid_size
        self.simd_group_size = simd_group_size
        self.grid_id = grid_id
        self.simd_group_id = simd_group_id

        if cfg is None:
            cfg = TensorSimConfig()
        self.cfg = cfg

_TENSOR_SIM_CONTEXT: contextvars.ContextVar[
    Optional[TensorSimContext]] = contextvars.ContextVar("TensorSimContext",
                                                        default=None)


@contextlib.contextmanager
def enter_tensorsim_context(grid_id: Sequence[int], grid_size: Sequence[int], 
            simd_group_id: Optional[Sequence[int]] = None, simd_group_size: Optional[Sequence[int]] = None, 
            cfg: Optional[TensorSimConfig] = None):
    ctx = TensorSimContext(
        grid_id=grid_id,
        grid_size=grid_size,
        simd_group_id=simd_group_id,
        simd_group_size=simd_group_size,
        cfg=cfg
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

