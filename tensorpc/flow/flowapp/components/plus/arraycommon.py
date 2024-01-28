from typing import Any, Optional

from tensorpc.core.moduleid import get_qualname_of_type
from .common import CommonQualNames
import numpy as np 

def can_cast_to_np_array(obj: Any):
    if isinstance(obj, np.ndarray):
        return True 
    elif get_qualname_of_type(type(obj)) == CommonQualNames.TorchTensor:
        return True
    elif get_qualname_of_type(type(obj)) == CommonQualNames.TVTensor:
        return True 
    return False 

def try_cast_to_np_array(obj: Any) -> Optional[np.ndarray]:
    if isinstance(obj, np.ndarray):
        return obj
    elif get_qualname_of_type(type(obj)) == CommonQualNames.TorchTensor:
        return obj.detach().cpu().numpy()
    elif get_qualname_of_type(type(obj)) == CommonQualNames.TVTensor:
        return obj.cpu().numpy() 
    return None 