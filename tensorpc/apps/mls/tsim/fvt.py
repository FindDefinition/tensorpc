"""FVT (full variable tracing)

trace tensor math in symbolic way.

some op isn't supported, e.g. histogram.
"""

import dataclasses
from typing import Union
import numpy as np 
import enum 

class FVTOpType(enum.IntEnum):
    LOAD = 0
    CONSTANT = 1
    ELEMENTWISE = 2 # unary, binary, etc
    INPLACE_ELEMENTWISE = 3
    VAL_REDUCE = 4 # sum, mean, etc
    # TODO do we need this lind of reduce?
    CHOICE_REDUCE = 5 # max, min, etc

    DOT = 6
    # TODO: we don't need op that don't modify a element.
    # WHERE = 7
    # GATHER = 8 # sort is a kind of gather
    
    # SPLIT = 9
    # GETITEM = 10
    # CONCAT = 11
    # STACK = 12

@dataclasses.dataclass 
class FVTTensorInfo:
    op_ids: np.ndarray
    indices: np.ndarray

@dataclasses.dataclass 
class FVTOpInfoBase:
    id: int
    type: FVTOpType
    args: list["FVTOpInfoBase"]

@dataclasses.dataclass 
class FVTOpLoad(FVTOpInfoBase):
    mem_key: str

@dataclasses.dataclass 
class FVTOpConstant(FVTOpInfoBase):
    # assume no array constant in triton.
    value: Union[int, float, bool]

@dataclasses.dataclass 
class FVTOpElementWise(FVTOpInfoBase):
    element_wise_type: str

@dataclasses.dataclass 
class FVTOpInplaceElementWise(FVTOpInfoBase):
    element_wise_type: str

@dataclasses.dataclass 
class FVTOpValReduce(FVTOpInfoBase):
    reduce_type: str
    axes: list[int]

@dataclasses.dataclass 
class FVTOpChoiceReduce(FVTOpInfoBase):
    reduce_type: str
    axes: list[int]

@dataclasses.dataclass 
class FVTOpDot(FVTOpInfoBase):
    pass

@dataclasses.dataclass 
class FVTOpContainer:
    op: FVTOpInfoBase
    args: list[FVTTensorInfo]
    results: list[FVTTensorInfo]


class FVTGraph:
    def __init__(self):
        self.ops: list[FVTOpContainer] = []

    def append_op(self, op: FVTOpInfoBase, args: list[FVTTensorInfo], results: list[FVTTensorInfo]) -> None:
        self.ops.append(FVTOpContainer(op=op, args=args, results=results))

    def add_constant(self, value: Union[int, float, bool]):
        op = FVTOpConstant(id=len(self.ops), type=FVTOpType.CONSTANT, value=value, args=[])
        args = []
        results = []
        self.append_op(op, args, results)
        return op 

    def add_constant(self, value: Union[int, float, bool]):
        op = FVTOpConstant(id=len(self.ops), type=FVTOpType.CONSTANT, value=value, args=[])
        args = []
        results = []
        self.append_op(op, args, results)
        return op 
