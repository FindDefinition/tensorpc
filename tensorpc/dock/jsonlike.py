import abc
import copy
import enum
import json
import re
from collections.abc import MutableMapping
from dataclasses import is_dataclass
from functools import partial
from typing import (Any, Callable, Dict, Generic, Hashable, List, Optional,
                    Tuple, Type, TypeVar, Union)

import numpy as np
from pydantic import GetCoreSchemaHandler
from pydantic_core import PydanticCustomError, core_schema
from typing_extensions import (Concatenate, Literal, ParamSpec, Protocol, Self,
                               TypeAlias)

import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.annolib import BackendOnlyProp, Undefined, undefined
from tensorpc.core.datamodel.asdict import (
    as_dict_no_undefined, as_dict_no_undefined_no_deepcopy,
    asdict_flatten_field_only, asdict_flatten_field_only_no_undefined,
    undefined_dict_factory, undefined_dict_factory_with_exclude)
from tensorpc.core.moduleid import get_qualname_of_type
from tensorpc.core.tree_id import UniqueTreeId, UniqueTreeIdForTree
from .core.uitypes import MenuItem
ValueType: TypeAlias = Union[int, float, str]
NumberType: TypeAlias = Union[int, float]

STRING_LENGTH_LIMIT = 500
T = TypeVar("T")
Tsrc = TypeVar("Tsrc")

def flatten_dict(d: MutableMapping,
                 parent_key: str = '',
                 sep: str = '.') -> MutableMapping:
    items: List[Any] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def camel_to_snake(name: str):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('__([A-Z])', r'_\1', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()


def snake_to_camel(name: str):
    if "_" not in name:
        return name
    res = ''.join(word.title() for word in name.split('_'))
    res = res[0].lower() + res[1:]
    return res


def split_props_to_undefined(props: Dict[str, Any]):
    res = {}
    res_und = []
    for res_camel, val in props.items():
        if not isinstance(val, BackendOnlyProp):
            if isinstance(val, Undefined):
                res_und.append(res_camel)
            else:
                res[res_camel] = val
    return res, res_und

class CommonQualNames:
    TorchTensor = "torch.Tensor"
    TVTensor = "cumm.core_cc.tensorview_bind.Tensor"
    TorchParameter = "torch.nn.parameter.Parameter"
    TorchDTensor = "torch.distributed.tensor.DTensor"

def _get_torch_dtensor_placements(ten: Any):
    from torch.distributed.tensor import (Partial, Replicate,  # type: ignore
                                          Shard)
    p_strs = []
    for p in ten.placements:
        if isinstance(p, Shard):
            p_strs.append(f"S({p.dim})")
        elif isinstance(p, Partial):
            p_strs.append(f"P")
        elif isinstance(p, Replicate):
            p_strs.append(f"R")
    placements_str = ",".join(p_strs)
    return placements_str

class TensorType(enum.Enum):
    Unknown = ""
    NpArray = "numpy.ndarray"
    TorchTensor = "torch.Tensor"
    TVTensor = "cumm.core_cc.tensorview_bind.Tensor"
    TorchParameter = "torch.nn.parameter.Parameter"


class JsonLikeType(enum.Enum):
    Int = 0
    Float = 1
    Bool = 2
    Constant = 3
    String = 4
    List = 5
    Dict = 6
    Tuple = 7
    Set = 8
    Tensor = 9
    Object = 10
    Complex = 11
    Enum = 12
    Layout = 13
    ListFolder = 14
    DictFolder = 15
    Function = 16


def _div_up(x: int, y: int):
    return (x + y - 1) // y


_FOLDER_TYPES = {JsonLikeType.ListFolder.value, JsonLikeType.DictFolder.value}


@dataclasses.dataclass(eq=True)
class IconButtonData:
    id: ValueType
    icon: int
    tooltip: Union[Undefined, str] = undefined


@dataclasses.dataclass(eq=True)
class JsonLikeNode:
    id: UniqueTreeIdForTree
    # must be id.split(SPLIT)[-1] for child of list/dict
    name: str
    type: int
    typeStr: Union[Undefined, str] = undefined
    value: Union[Undefined, str] = undefined
    cnt: int = 0
    children: "List[JsonLikeNode]" = dataclasses.field(default_factory=list)
    drag: Union[Undefined, bool] = undefined
    iconBtns: Union[Undefined, List[IconButtonData]] = undefined
    realId: Union[Undefined, UniqueTreeIdForTree] = undefined
    start: Union[Undefined, int] = undefined
    # name color
    color: Union[Undefined, str] = undefined
    menus: Union[Undefined, List[MenuItem]] = undefined
    edit: Union[Undefined, bool] = undefined
    userdata: Union[Undefined, Any] = undefined
    alias: Union[Undefined, str] = undefined
    fixedIconBtns: Union[Undefined, List[IconButtonData]] = undefined

    # backend only props, not used in frontend
    dictKey: Union[Undefined, BackendOnlyProp[Hashable]] = undefined
    keys: Union[Undefined, BackendOnlyProp[List[str]]] = undefined

    def last_part(self):
        return self.id.parts[-1]

    def is_folder(self):
        return self.type in _FOLDER_TYPES

    def get_dict_key(self):
        if not isinstance(self.dictKey, Undefined):
            return self.dictKey.data
        return undefined

    def _get_node_by_uid(self, uid: str, split: str = ":"):
        """TODO if dict key contains split word, this function will
        produce wrong result.
        """
        uid_object = UniqueTreeIdForTree(uid, 1)
        parts = uid_object.parts
        if len(parts) == 1:
            return self
        # uid contains root, remove it at first.
        return self._get_node_by_uid_resursive(parts[1:])

    def _get_node_by_uid_resursive(self, parts: List[str]) -> "JsonLikeNode":
        key = parts[0]
        node: Optional[JsonLikeNode] = None
        for c in self.children:
            if c.last_part() == key:
                node = c
                break
        assert node is not None, f"{key} missing"
        if len(parts) == 1:
            return node
        else:
            return node._get_node_by_uid_resursive(parts[1:])

    def _get_node_by_uid_trace(self, uid_parts: List[str]):
        parts = uid_parts
        if len(parts) == 1:
            return [self]
        # uid contains root, remove it at first.
        nodes, found = self._get_node_by_uid_resursive_trace(
            parts[1:], check_missing=True)
        assert found
        return [self] + nodes

    def _get_node_by_uid_trace_found(self, uid_parts: List[str], check_missing: bool = False):
        parts = uid_parts
        if len(parts) == 1:
            return [self], True
        # uid contains root, remove it at first.
        res = self._get_node_by_uid_resursive_trace(parts[1:], check_missing)
        return [self] + res[0], res[1]

    def _get_node_by_uid_resursive_trace(
            self,
            parts: List[str],
            check_missing: bool = False) -> Tuple[List["JsonLikeNode"], bool]:
        key = parts[0]
        node: Optional[JsonLikeNode] = None
        for c in self.children:
            if c.last_part() == key:
                node = c
                break
        if check_missing:
            assert node is not None, f"{key} missing"
        if node is None:
            return [], False
        if len(parts) == 1:
            return [node], True
        else:
            res = node._get_node_by_uid_resursive_trace(
                parts[1:], check_missing)
            return [node] + res[0], res[1]

    def _is_divisible(self, divisor: int):
        return self.cnt > divisor

    def _get_divided_tree(self, divisor: int, start: int, split: str = "::"):
        num_child = _div_up(self.cnt, divisor)
        if num_child > divisor:
            tmp = num_child
            num_child = divisor
            divisor = tmp
        count = 0
        total = self.cnt
        res: List[JsonLikeNode] = []
        if self.type in _FOLDER_TYPES:
            real_id = self.realId
        else:
            real_id = self.id
        if self.type == JsonLikeType.List.value or self.type == JsonLikeType.ListFolder.value:
            for i in range(num_child):
                this_cnt = min(total - count, divisor)
                node = JsonLikeNode(self.id.append_part(f"{i}"),
                                    f"{i}",
                                    JsonLikeType.ListFolder.value,
                                    cnt=this_cnt,
                                    realId=real_id,
                                    start=start + count)
                res.append(node)
                count += this_cnt
        if self.type == JsonLikeType.Dict.value or self.type == JsonLikeType.DictFolder.value:
            assert not isinstance(self.keys, Undefined)
            keys = self.keys.data
            for i in range(num_child):
                this_cnt = min(total - count, divisor)
                keys_child = keys[count:count + this_cnt]
                node = JsonLikeNode(self.id.append_part(f"{i}"),
                                    f"{i}",
                                    JsonLikeType.DictFolder.value,
                                    cnt=this_cnt,
                                    realId=real_id,
                                    start=start + count,
                                    keys=BackendOnlyProp(keys_child))
                res.append(node)
                count += this_cnt
        return res
    
    @classmethod 
    def create_dummy(cls):
        return cls(UniqueTreeIdForTree.from_parts(["root"]), "root",
                    JsonLikeType.Object.value, "Object", undefined, 0, [])

    @staticmethod 
    def create_dummy_dict():
        return {
            "id": UniqueTreeIdForTree.from_parts(["root"]).uid_encoded,
            "name": "root",
            "type": JsonLikeType.Object.value,
            "children": [],
        }

    @staticmethod 
    def create_dummy_dict_binary():
        return json.dumps(JsonLikeNode.create_dummy_dict()).encode()
        
    def get_userdata_typed(self, type: Type[T]) -> T:
        assert isinstance(self.userdata, type)
        return self.userdata

    def get_all_tree_ids(self) -> List[UniqueTreeIdForTree]:
        """get all tree ids in this node, including self.
        """
        res = [self.id]
        for c in self.children:
            res.extend(c.get_all_tree_ids())
        return res

    def get_all_tree_container_ids(self) -> List[UniqueTreeIdForTree]:
        """get all tree ids in this node, including self.
        """
        if self.children:
            res = [self.id]
            for c in self.children:
                res.extend(c.get_all_tree_ids())
            return res
        else:
            return []

def parse_obj_to_jsonlike(obj, name: str, id: UniqueTreeIdForTree):
    obj_type = type(obj)
    if obj is None or obj is Ellipsis:
        return JsonLikeNode(id,
                            name,
                            JsonLikeType.Constant.value,
                            value=str(obj))
    elif isinstance(obj, JsonLikeNode):
        obj_copy = dataclasses.replace(obj)
        obj_copy.name = name
        obj_copy.id = id
        obj_copy.drag = False
        return obj_copy
    elif isinstance(obj, enum.Enum):
        return JsonLikeNode(id,
                            name,
                            JsonLikeType.Enum.value,
                            "enum",
                            value=str(obj))
    elif isinstance(obj, (bool)):
        # bool is inherit from int, so we must check bool first.
        return JsonLikeNode(id, name, JsonLikeType.Bool.value, value=str(obj))
    elif isinstance(obj, (int)):
        return JsonLikeNode(id, name, JsonLikeType.Int.value, value=str(obj))
    elif isinstance(obj, (float)):
        return JsonLikeNode(id, name, JsonLikeType.Float.value, value=str(obj))
    elif isinstance(obj, (complex)):
        return JsonLikeNode(id,
                            name,
                            JsonLikeType.Complex.value,
                            value=str(obj))
    elif isinstance(obj, str):
        if len(obj) > STRING_LENGTH_LIMIT:
            value = obj[:STRING_LENGTH_LIMIT] + "..."
        else:
            value = obj
        return JsonLikeNode(id, name, JsonLikeType.String.value, value=value)

    elif isinstance(obj, (list, dict, tuple, set)):
        t = JsonLikeType.List
        if isinstance(obj, list):
            t = JsonLikeType.List
        elif isinstance(obj, dict):
            t = JsonLikeType.Dict
        elif isinstance(obj, tuple):
            t = JsonLikeType.Tuple
        elif isinstance(obj, set):
            t = JsonLikeType.Set
        else:
            raise NotImplementedError
        # TODO suppert nested view
        return JsonLikeNode(id, name, t.value, cnt=len(obj), drag=False)
    elif isinstance(obj, np.ndarray):
        t = JsonLikeType.Tensor
        shape_short = ",".join(map(str, obj.shape))
        return JsonLikeNode(id,
                            name,
                            t.value,
                            typeStr="np.ndarray",
                            value=f"[{shape_short}]{obj.dtype}",
                            drag=True)
    else:
        qname = get_qualname_of_type(obj_type)
        if qname == CommonQualNames.TorchTensor:
            t = JsonLikeType.Tensor
            shape_short = ",".join(map(str, obj.shape))
            return JsonLikeNode(id,
                                name,
                                t.value,
                                typeStr="torch.Tensor",
                                value=f"[{shape_short}]{obj.dtype}",
                                drag=True)
        elif qname == CommonQualNames.TorchParameter:
            t = JsonLikeType.Tensor
            shape_short = ",".join(map(str, obj.data.shape))
            return JsonLikeNode(id,
                                name,
                                t.value,
                                typeStr="torch.Parameter",
                                value=f"[{shape_short}]{obj.data.dtype}",
                                drag=True)
        elif qname == CommonQualNames.TorchDTensor:
            t = JsonLikeType.Tensor
            shape_short = ",".join(map(str, obj._local_tensor.shape))
            shape_local_short = ",".join(map(str, obj._local_tensor.shape))
            return JsonLikeNode(id,
                                name,
                                t.value,
                                typeStr="torch.DTensor",
                                value=f"[{shape_short}]({shape_local_short})<{_get_torch_dtensor_placements(obj)}>{obj._local_tensor.dtype}",
                                drag=True)
        elif qname == CommonQualNames.TVTensor:
            t = JsonLikeType.Tensor
            shape_short = ",".join(map(str, obj.shape))
            return JsonLikeNode(id,
                                name,
                                t.value,
                                typeStr="tv.Tensor",
                                value=f"[{shape_short}]{obj.dtype}",
                                drag=True)
        elif qname.startswith("torch"):
            import torch 
            if isinstance(obj, torch.Tensor):
                t = JsonLikeType.Tensor
                shape_short = ",".join(map(str, obj.shape))
                return JsonLikeNode(id,
                                    name,
                                    t.value,
                                    typeStr="torch.Tensor",
                                    value=f"{obj_type.__name__}[{shape_short}]{obj.dtype}",
                                    drag=True)
            else:
                t = JsonLikeType.Object
                value = undefined
                return JsonLikeNode(id,
                                    name,
                                    t.value,
                                    value=value,
                                    typeStr=obj_type.__qualname__)
        else:
            t = JsonLikeType.Object
            value = undefined
            return JsonLikeNode(id,
                                name,
                                t.value,
                                value=value,
                                typeStr=obj_type.__qualname__)

class TreeItem(abc.ABC):

    @abc.abstractmethod
    async def get_child_desps(
            self, parent_ns: UniqueTreeIdForTree) -> Dict[str, JsonLikeNode]:
        raise NotImplementedError

    @abc.abstractmethod
    async def get_child(self, key: str) -> Any:
        raise NotImplementedError

    def get_json_like_node(self, id: UniqueTreeIdForTree) -> Optional[JsonLikeNode]:
        """name and id is determined by parent, only root node use name provided
        by this method.
        """
        return None

    async def handle_button(self, button_key: str) -> Optional[bool]:
        return None

    async def handle_lazy_expand(self) -> Any:
        return None

    async def handle_child_button(self, button_key: str,
                                  child_key: str) -> Optional[bool]:
        return None

    async def handle_context_menu(self, userdata: Dict[str,
                                                       Any]) -> Optional[bool]:
        return None

    async def handle_child_context_menu(
            self, child_key: str, userdata: Dict[str, Any]) -> Optional[bool]:
        return None

    async def handle_child_rename(self, child_key: str,
                                  newname: str) -> Optional[bool]:
        return None

    def default_expand(self) -> bool:
        return True

def merge_props_not_undefined(dst: T, src: T):
    assert is_dataclass(dst)
    assert is_dataclass(src)
    for src_field in dataclasses.fields(src):
        src_field_value = getattr(src, src_field.name)
        if not isinstance(src_field_value, Undefined):
            setattr(dst, src_field.name, src_field_value)
        if is_dataclass(src_field_value):
            merge_props_not_undefined(getattr(dst, src_field.name), src_field_value)

