from typing import Optional, Union
import numpy as np
from .tensor import SimTensor
from .core import get_sim_mode, TensorSimMode

def abs(tensor: SimTensor) -> SimTensor:
    if tensor.storage is None:
        return tensor
    if get_sim_mode() == TensorSimMode.LOGIC_ONLY and tensor.is_floating():
        return tensor
    return tensor._replace_data(np.abs(tensor.storage.data))

def ceil(tensor: SimTensor) -> SimTensor:
    if tensor.storage is None:
        return tensor
    if get_sim_mode() == TensorSimMode.LOGIC_ONLY and tensor.is_floating():
        return tensor
    return tensor._replace_data(np.ceil(tensor.storage.data))

def floor(tensor: SimTensor) -> SimTensor:
    if tensor.storage is None:
        return tensor
    if get_sim_mode() == TensorSimMode.LOGIC_ONLY and tensor.is_floating():
        return tensor
    return tensor._replace_data(np.floor(tensor.storage.data))

def sin(tensor: SimTensor) -> SimTensor:
    if tensor.storage is None:
        return tensor
    if get_sim_mode() == TensorSimMode.LOGIC_ONLY and tensor.is_floating():
        return tensor
    return tensor._replace_data(np.sin(tensor.storage.data))

def cos(tensor: SimTensor) -> SimTensor:
    if tensor.storage is None:
        return tensor
    if get_sim_mode() == TensorSimMode.LOGIC_ONLY and tensor.is_floating():
        return tensor
    return tensor._replace_data(np.cos(tensor.storage.data))

def exp(tensor: SimTensor) -> SimTensor:
    if tensor.storage is None:
        return tensor
    if get_sim_mode() == TensorSimMode.LOGIC_ONLY and tensor.is_floating():
        return tensor
    return tensor._replace_data(np.exp(tensor.storage.data))

def exp2(tensor: SimTensor) -> SimTensor:
    if tensor.storage is None:
        return tensor
    if get_sim_mode() == TensorSimMode.LOGIC_ONLY and tensor.is_floating():
        return tensor
    return tensor._replace_data(np.exp2(tensor.storage.data))

def log(tensor: SimTensor) -> SimTensor:
    if tensor.storage is None:
        return tensor
    if get_sim_mode() == TensorSimMode.LOGIC_ONLY and tensor.is_floating():
        return tensor
    return tensor._replace_data(np.log(tensor.storage.data))

def log2(tensor: SimTensor) -> SimTensor:
    if tensor.storage is None:
        return tensor
    if get_sim_mode() == TensorSimMode.LOGIC_ONLY and tensor.is_floating():
        return tensor
    return tensor._replace_data(np.log2(tensor.storage.data))

def sqrt(tensor: SimTensor) -> SimTensor:
    if tensor.storage is None:
        return tensor
    if get_sim_mode() == TensorSimMode.LOGIC_ONLY and tensor.is_floating():
        return tensor
    return tensor._replace_data(np.sqrt(tensor.storage.data))   

def rsqrt(tensor: SimTensor) -> SimTensor:
    if tensor.storage is None:
        return tensor
    if get_sim_mode() == TensorSimMode.LOGIC_ONLY and tensor.is_floating():
        return tensor
    return tensor._replace_data(np.reciprocal(np.sqrt(tensor.storage.data)))

def clamp(tensor: SimTensor, min_value: Union[SimTensor, int, float], max_value: Union[SimTensor, int, float]) -> SimTensor:
    if tensor.storage is None:
        return tensor
    if get_sim_mode() == TensorSimMode.LOGIC_ONLY and tensor.is_floating():
        return tensor
    if isinstance(min_value, SimTensor):
        assert min_value.storage is not None, "min_value tensor must have storage"
        min_value_np = min_value.storage.data
    else:
        min_value_np = min_value
    if isinstance(max_value, SimTensor):
        assert max_value.storage is not None, "max_value tensor must have storage"
        max_value_np = max_value.storage.data
    else:
        max_value_np = max_value
    data = tensor.storage.data
    data = np.clip(data, min_value_np, max_value_np)
    return tensor._replace_data(data)

def sigmoid(tensor: SimTensor) -> SimTensor:
    if tensor.storage is None:
        return tensor
    if get_sim_mode() == TensorSimMode.LOGIC_ONLY and tensor.is_floating():
        return tensor
    x = tensor.storage.data
    data = 1 / (1 + np.exp(-x))
    return tensor._replace_data(data)   

def softmax(x: SimTensor, axis: Optional[int] = None) -> SimTensor:
    if x.storage is None:
        return x
    if get_sim_mode() == TensorSimMode.LOGIC_ONLY and x.is_floating():
        return x
    x_data = x.storage.data
    x_max = x_data.max(axis=axis, keepdims=True)
    z = x_data - x_max
    x_exp = np.exp(z)
    res = x_exp / x_exp.sum(axis=axis, keepdims=True)
    return x._replace_data(res)   


