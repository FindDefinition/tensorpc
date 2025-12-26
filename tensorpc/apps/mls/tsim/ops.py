import enum
from typing import Optional, Union
import dataclasses
import numpy as np

from tensorpc.apps.mls.tsim.core import DTypeEnum
from tensorpc.core.pfl.pfl_ast import BinOpType
from .tensor import SimTensor, get_may_tensor_dtype

class _ExtendBinOpType(enum.IntEnum):
    MAXIMUM = 0
    MINIMUM = 1

def where(cond: SimTensor, x: Union[int, float, bool, SimTensor], y: Union[int, float, bool, SimTensor]):
    x_dtype = get_may_tensor_dtype(x)
    y_dtype = get_may_tensor_dtype(y)
    assert x_dtype == y_dtype, f"where(, x, y) dtype of x, y must be same, get {x_dtype.name} and {y_dtype.name}."

    if cond.storage is None:
        shapes = [cond.shape]
        if isinstance(x, SimTensor):
            shapes.append(x.shape)
        if isinstance(y, SimTensor):
            shapes.append(y.shape)
        final_shape = np.broadcast_shapes(*shapes)
        return dataclasses.replace(cond, dtype=x_dtype, shape=list(final_shape))
    if isinstance(x, SimTensor):
        assert x.storage is not None 
        x_data = x.storage.data 
    else:
        x_data = x 
    if isinstance(y, SimTensor):
        assert y.storage is not None 
        y_data = y.storage.data 
    else:
        y_data = y 
    # print("WHERE DEBUG", x_data, y_data)

    res_data = np.where(cond.storage.data, x_data, y_data).astype(x_dtype.to_numpy_dtype())
    res_storage = dataclasses.replace(cond.storage, data=res_data)
    return dataclasses.replace(cond, dtype=DTypeEnum.from_numpy_dtype(res_data.dtype), shape=list(res_data.shape), storage=res_storage)

def _extend_bin_op(type: _ExtendBinOpType, x: Union[int, float, bool, SimTensor], y: Union[int, float, bool, SimTensor]):
    tgt_simten: Optional[SimTensor] = None
    if isinstance(x, SimTensor):
        if x.storage is None:
            if isinstance(y, SimTensor):
                assert y.storage is None, "If x is SimTensor without storage, y must also be SimTensor without storage."
            # use add for dtype and shape inference
            return x._binary_base(y, BinOpType.ADD, False, False)
        tgt_simten = x
        x_data = x.storage.data 
    else:
        x_data = x 
    if isinstance(y, SimTensor):
        if y.storage is None:
            if isinstance(x, SimTensor):
                assert x.storage is None, "If y is SimTensor without storage, x must also be SimTensor without storage."
            # use add for dtype and shape inference
            return y._binary_base(x, BinOpType.ADD, True, False)
        y_data = y.storage.data 
        tgt_simten = y
    else:
        y_data = y 
    if type == _ExtendBinOpType.MAXIMUM:
        res_data = np.maximum(x_data, y_data)
    else:
        res_data = np.minimum(x_data, y_data)
    if tgt_simten is not None:
        assert tgt_simten.storage is not None 
        if not isinstance(res_data, np.ndarray):
            # np maximum return scalar if all operand is scalar
            res_data = np.array(res_data)
        res_storage = dataclasses.replace(tgt_simten.storage, data=res_data)
        return dataclasses.replace(tgt_simten, dtype=DTypeEnum.from_numpy_dtype(res_data.dtype), shape=list(res_data.shape), storage=res_storage)
    else:
        return res_data.item()

def maximum(x: Union[int, float, bool, SimTensor], y: Union[int, float, bool, SimTensor]):
    return _extend_bin_op(_ExtendBinOpType.MAXIMUM, x, y)

def minimum(x: Union[int, float, bool, SimTensor], y: Union[int, float, bool, SimTensor]):
    return _extend_bin_op(_ExtendBinOpType.MINIMUM, x, y)

def clamp(x: SimTensor, min: Union[int, float, SimTensor], max: Union[int, float, SimTensor]) -> SimTensor:
    return maximum(minimum(x, max), min)