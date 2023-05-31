

from typing import Any, Optional
from tensorpc.flow.jsonlike import TensorType
from tensorpc.core.moduleid import get_qualname_of_type
import numpy as np 

def _try_cast_tensor_dtype(obj: Any) -> Optional[np.dtype]:
    try:
        if isinstance(obj, np.ndarray):
            return obj.dtype
        elif get_qualname_of_type(type(obj)) == TensorType.TVTensor.value:
            from cumm.dtypes import get_npdtype_from_tvdtype
            return get_npdtype_from_tvdtype(obj.dtype)
        elif get_qualname_of_type(type(obj)) == TensorType.TorchTensor.value:
            import torch
            _TORCH_DTYPE_TO_NP = {
                torch.float32: np.dtype(np.float32),
                torch.float64: np.dtype(np.float64),
                torch.float16: np.dtype(np.float16),
                torch.int32: np.dtype(np.int32),
                torch.int64: np.dtype(np.int64),
                torch.int8: np.dtype(np.int8),
                torch.int16: np.dtype(np.int16),
                torch.uint8: np.dtype(np.uint8),
            }
            return _TORCH_DTYPE_TO_NP[obj.dtype]
    except:
        return None

def _get_tensor_type(obj):
    if isinstance(obj, np.ndarray):
        return TensorType.NpArray
    elif get_qualname_of_type(type(obj)) == TensorType.TVTensor.value:
        return TensorType.TVTensor
    elif get_qualname_of_type(type(obj)) == TensorType.TorchTensor.value:
        return TensorType.TorchTensor
    else:
        return TensorType.Unknown

def _cast_tensor_to_np(obj: Any) -> Optional[np.ndarray]:
    if isinstance(obj, np.ndarray):
        return obj
    elif get_qualname_of_type(type(obj)) == TensorType.TVTensor.value:
        if obj.device == 0:
            return obj.cpu().numpy()
        return obj.numpy()

    elif get_qualname_of_type(type(obj)) == TensorType.TorchTensor.value:
        if obj.is_cuda:
            return obj.detach().cpu().numpy()
        return obj.numpy()
    return None

class TensorContainer:
    def __init__(self, obj: Any, type: TensorType, dtype: np.dtype) -> None:
        self.type = type
        self.dtype = dtype
        self._obj = obj

    def numpy(self):
        res = _cast_tensor_to_np(self._obj)
        assert res is not None 
        return res 
    
    @property 
    def shape(self):
        return list(self._obj.shape)

def get_tensor_container(obj) -> Optional[TensorContainer]:
    type = _get_tensor_type(obj)
    if type == TensorType.Unknown:
        return None 
    dtype = _try_cast_tensor_dtype(obj)
    assert dtype is not None 
    return TensorContainer(obj, type, dtype)