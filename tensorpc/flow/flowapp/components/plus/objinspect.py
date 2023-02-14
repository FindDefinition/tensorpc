from tensorpc.flow.flowapp.components import mui, three
from typing import Dict, Iterable, Optional, Union, List
import numpy as np 
from tensorpc.utils.moduleid import get_qualname_of_type

TORCH_TENSOR_NAME = "torch.Tensor"
TV_TENSOR_NAME = "cumm.core_cc.tensorview_bind.Tensor"

BASE_OBJ_TO_TYPE = {
    int: mui.JsonLikeType.Int,
    float: mui.JsonLikeType.Float,
    complex: mui.JsonLikeType.Complex,
    bool: mui.JsonLikeType.Bool,
    str: mui.JsonLikeType.String,
}



def parse_obj(obj, name: str, id: str):
    obj_type = type(obj)
    if obj is None or obj is Ellipsis:
        return mui.JsonLikeNode(id, name, mui.JsonLikeType.Constant.value, value=str(obj)) 
    elif isinstance(obj, (int, float, complex, bool, str)):
        return mui.JsonLikeNode(id, name, BASE_OBJ_TO_TYPE[type(obj)].value, value=str(obj)) 
    elif isinstance(obj, (list, dict, tuple, set)):
        t = mui.JsonLikeType.List
        if isinstance(obj, list):
            t = mui.JsonLikeType.List 
        elif isinstance(obj, dict):
            t = mui.JsonLikeType.Dict 
        elif isinstance(obj, tuple):
            t = mui.JsonLikeType.Tuple 
        elif isinstance(obj, set):
            t = mui.JsonLikeType.Set 
        else:
            raise NotImplementedError
        return mui.JsonLikeNode(id, name, t.value, lazyExpandCount=len(obj)) 
    elif isinstance(obj, np.ndarray):
        t = mui.JsonLikeType.Tensor
        return mui.JsonLikeNode(id, name, t.value, typeStr="np.ndarray", value=f"{obj.shape}|{obj.dtype}") 
    elif get_qualname_of_type(obj_type) == TORCH_TENSOR_NAME:
        t = mui.JsonLikeType.Tensor
        return mui.JsonLikeNode(id, name, t.value, typeStr="torch.Tensor", value=f"{obj.shape}|{obj.dtype}") 

    elif get_qualname_of_type(obj_type) == TV_TENSOR_NAME:
        t = mui.JsonLikeType.Tensor
        return mui.JsonLikeNode(id, name, t.value, typeStr="tv.Tensor", value=f"{obj.shape}|{obj.dtype}") 
    else:
        return mui.JsonLikeNode(id, name, t.value, typeStr="tv.Tensor", value=f"{obj.shape}|{obj.dtype}") 

class ObjectInspector(mui.FlexBox):
    def __init__(self) -> None:
        
        self.tree = mui.JsonLikeTree()
        self.viewer_container = mui.VBox([]).prop(flex=3)
        super().__init__([
            self.tree.prop(flex=1),
        ])
        self.prop(flex_flow="row")

    