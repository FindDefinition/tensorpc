from functools import partial
import types
from tensorpc.flow.flowapp.components import mui, three
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple, Type, Union, List
import numpy as np
from tensorpc.flow.flowapp.core import FrontendEventType 
from tensorpc.utils.moduleid import get_qualname_of_type
import inspect 
import enum
from tensorpc.core.inspecttools import get_members

TORCH_TENSOR_NAME = "torch.Tensor"
TV_TENSOR_NAME = "cumm.core_cc.tensorview_bind.Tensor"

BASE_OBJ_TO_TYPE = {
    int: mui.JsonLikeType.Int,
    float: mui.JsonLikeType.Float,
    complex: mui.JsonLikeType.Complex,
    bool: mui.JsonLikeType.Bool,
    str: mui.JsonLikeType.String,
}

STRING_LENGTH_LIMIT = 2000

def parse_obj_item(obj, name: str, id: str, checker: Callable[[Type], bool]):
    obj_type = type(obj)
    if obj is None or obj is Ellipsis:
        return mui.JsonLikeNode(id, name, mui.JsonLikeType.Constant.value, value=str(obj)) 
    elif isinstance(obj, enum.Enum):
        return mui.JsonLikeNode(id, name, mui.JsonLikeType.Enum.value, "enum", value=str(obj)) 
    elif isinstance(obj, (int)):
        return mui.JsonLikeNode(id, name, mui.JsonLikeType.Int.value, value=str(obj)) 
    elif isinstance(obj, (float)):
        return mui.JsonLikeNode(id, name, mui.JsonLikeType.Float.value, value=str(obj)) 
    elif isinstance(obj, (complex)):
        return mui.JsonLikeNode(id, name, mui.JsonLikeType.Complex.value, value=str(obj)) 
    elif isinstance(obj, (bool)):
        return mui.JsonLikeNode(id, name, mui.JsonLikeType.Bool.value, value=str(obj)) 
    elif isinstance(obj, str):
        if len(obj) > STRING_LENGTH_LIMIT:
            value = obj[:STRING_LENGTH_LIMIT] + "..."
        else:
            value = obj
        return mui.JsonLikeNode(id, name, mui.JsonLikeType.String.value, value=value) 

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
        t = mui.JsonLikeType.Object
        obj_dict = _get_obj_dict(obj, checker)
        return mui.JsonLikeNode(id, name, t.value, typeStr=obj_type.__qualname__, lazyExpandCount=len(obj_dict)) 

def parse_obj_dict(obj_dict: Dict[str, Any], ns: str, checker: Callable[[Type], bool]):
    res_node: List[mui.JsonLikeNode] = []
    for k, v in obj_dict.items():
        node = parse_obj_item(v, k, f"{ns}-{k}", checker)
        res_node.append(node )
    return res_node 

def _check_is_valid(obj_type, cared_types: Set[Type], ignored_types: Set[Type]):
    valid = True 
    if len(cared_types) != 0:
        valid &= obj_type in cared_types
    if len(ignored_types) != 0:
        valid &= obj_type not in ignored_types
    return valid

def _get_obj_dict(obj, checker: Callable[[Type], bool], check_obj: bool = True):
    res: Dict[str, Any] = {}
    if isinstance(obj, (list, tuple, set)):
        obj_list = list(obj)
        return {str(i): obj_list[i] for i in range(len(obj))}
    elif isinstance(obj, dict):
        return {str(k): v for k, v in obj.items()}
    if inspect.isbuiltin(obj):
        return {}
    if not checker(obj) and check_obj:
        return {}
    if isinstance(obj, types.ModuleType):
        return {}
    # if isinstance(obj, mui.Component):
    #     return {}
    members = get_members(obj, no_parent=False)
    member_keys = set([m[0] for m in members])
    for k in dir(obj):
        if k.startswith("__"):
            continue 
        if k in member_keys:
            continue 
        try:
            v = getattr(obj, k)
        except:
            continue
        if not (checker(v)):
            continue
        if isinstance(v, types.ModuleType):
            continue
        if inspect.isfunction(v) or inspect.ismethod(v) or inspect.isbuiltin(v):
            continue 
        res[k] = v
    return res 

def _get_obj_by_uid(obj, uid: str, checker: Callable[[Type], bool]):
    parts = uid.split("-")
    if len(parts) == 1:
        return obj
    # uid contains root, remove it at first.
    return _get_obj_by_uid_resursive(obj, parts[1:], checker)
    
def _get_obj_by_uid_resursive(obj, parts: List[str], checker: Callable[[Type], bool]) :
    key = parts[0]
    if isinstance(obj, (list, tuple, set)):
        obj_list = list(obj)
        key_index = int(key)
        child_obj = obj_list[key_index]
    elif isinstance(obj, dict):
        obj_dict = obj 
        assert key in obj_dict
        child_obj = obj_dict[key]
    else:
        obj_dict = _get_obj_dict(obj, checker)
        assert key in obj_dict
        child_obj = obj_dict[key]
    if len(parts) == 1:
        return child_obj
    else:
        return _get_obj_by_uid_resursive(child_obj, parts[1:], checker)


def _get_root_tree(obj, checker: Callable[[Type], bool]):
    obj_dict = _get_obj_dict(obj, checker)
    root_node = parse_obj_item(obj, "root", "root", checker)
    root_node.children = parse_obj_dict(obj_dict, "root", checker)
    # root_node.lazyExpandCount = len(obj_dict)
    return root_node

class ObjectInspector(mui.FlexBox):
    def __init__(self, init: Optional[Any] = None, cared_types: Optional[Set[Type]] = None, ignored_types: Optional[Set[Type]] = None) -> None:
        self.tree = mui.JsonLikeTree()
        self.viewer_container = mui.VBox([]).prop(flex=3)
        super().__init__([
            self.tree.prop(flex=1),
        ])
        self.prop(flex_flow="row", overflow="auto")
        self._uid_to_node: Dict[str, mui.JsonLikeNode] = {}
        if cared_types is None:
            cared_types = set()
        if ignored_types is None:
            ignored_types = set()
        self._cared_types = cared_types
        self._ignored_types = ignored_types
        if init is None:
            self.tree.props.tree = mui.JsonLikeNode("root", "root", mui.JsonLikeType.Constant.value, value="None")
            self.root = None
        else:
            self.tree.props.tree = _get_root_tree(init, self._valid_checker)
            self.root = init

        # print(self.tree.props.tree)
        # inspect.isbuiltin()
        self.tree.register_event_handler(FrontendEventType.TreeLazyExpand.value, self._on_expand)

    def _checker(self, obj):
        return _check_is_valid(type(obj), self._cared_types, self._ignored_types)

    def _valid_checker(self, obj):
        return True

    @property 
    def _objinspect_root(self):
        return self.tree.props.tree

    async def _on_expand(self, uid: str):
        node = self._objinspect_root._get_node_by_uid(uid)
        obj = _get_obj_by_uid(self.root, uid, self._valid_checker)
        obj_dict = _get_obj_dict(obj, self._checker)
        tree = parse_obj_dict(obj_dict, node.id, self._checker)
        node.children = tree 
        upd = self.tree.update_event(tree=self._objinspect_root)
        await self.tree.send_and_wait(upd)

    async def set_object(self, obj):
        self.root = obj 
        self.tree.props.tree = _get_root_tree(obj, self._valid_checker)
        await self.tree.send_and_wait(self.tree.update_event(tree=self.tree.props.tree))


