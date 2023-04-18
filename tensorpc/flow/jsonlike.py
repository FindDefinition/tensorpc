import enum

from typing import Any, Dict, Generic, Hashable, List, Optional, TypeVar, Union, Tuple
import dataclasses
import re
import numpy as np
from tensorpc.core.moduleid import get_qualname_of_type
from typing_extensions import (Concatenate, Literal, ParamSpec, Protocol, Self,
                               TypeAlias)

ValueType: TypeAlias = Union[int, float, str]
NumberType: TypeAlias = Union[int, float]

STRING_LENGTH_LIMIT = 500
T = TypeVar("T")


class Undefined:

    def __repr__(self) -> str:
        return "undefined"


class BackendOnlyProp(Generic[T]):
    """when wrap a property with this class, it will be ignored when serializing to frontend
    """

    def __init__(self, data: T) -> None:
        super().__init__()
        self.data = data

    def __repr__(self) -> str:
        return "BackendOnlyProp"


# DON'T MODIFY THIS VALUE!!!
undefined = Undefined()


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


def _undefined_dict_factory(x: List[Tuple[str, Any]]):
    res: Dict[str, Any] = {}
    for k, v in x:
        if not isinstance(v, (Undefined, BackendOnlyProp)):
            res[k] = v
    return res


@dataclasses.dataclass
class _DataclassSer:
    obj: Any


def as_dict_no_undefined(obj: Any):
    return dataclasses.asdict(_DataclassSer(obj),
                              dict_factory=_undefined_dict_factory)["obj"]


@dataclasses.dataclass
class DataClassWithUndefined:

    def get_dict_and_undefined(self, state: Dict[str, Any]):
        this_type = type(self)
        res = {}
        # we only support update in first-level dict,
        # so we ignore all undefined in childs.
        ref_dict = dataclasses.asdict(self,
                                      dict_factory=_undefined_dict_factory)
        res_und = []
        for field in dataclasses.fields(this_type):
            if field.name in state:
                continue
            res_camel = snake_to_camel(field.name)
            val = ref_dict[field.name]
            if isinstance(val, Undefined):
                res_und.append(res_camel)
            else:
                res[res_camel] = val
        return res, res_und

    def get_dict(self):
        this_type = type(self)
        res = {}
        ref_dict = dataclasses.asdict(self,
                                      dict_factory=_undefined_dict_factory)
        for field in dataclasses.fields(this_type):
            res_camel = snake_to_camel(field.name)
            if field.name not in ref_dict:
                val = undefined
            else:
                val = ref_dict[field.name]
            res[res_camel] = val
        return res


class CommonQualNames:
    TorchTensor = "torch.Tensor"
    TVTensor = "cumm.core_cc.tensorview_bind.Tensor"


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

@dataclasses.dataclass
class IconButtonData:
    id: ValueType
    icon: int
    tooltip: Union[Undefined, str] = undefined

@dataclasses.dataclass
class ContextMenuData:
    title: str
    id: Union[Undefined, ValueType] = undefined
    icon: Union[Undefined, int] = undefined
    userdata: Union[Undefined, Any] = undefined


@dataclasses.dataclass
class JsonLikeNode:
    id: str
    name: str
    type: int
    typeStr: Union[Undefined, str] = undefined
    value: Union[Undefined, str] = undefined
    cnt: int = 0
    children: "List[JsonLikeNode]" = dataclasses.field(default_factory=list)
    drag: Union[Undefined, bool] = undefined
    iconBtns: Union[Undefined, List[IconButtonData]] = undefined
    realId: Union[Undefined, str] = undefined
    start: Union[Undefined, int] = undefined
    # name color
    color: Union[Undefined, str] = undefined
    dictKey: Union[Undefined, BackendOnlyProp[Hashable]] = undefined
    keys: Union[Undefined, BackendOnlyProp[List[str]]] = undefined
    menus: Union[Undefined, List[ContextMenuData]] = undefined
    edit: Union[Undefined, bool] = undefined
    userdata: Union[Undefined, Any] = undefined

    def last_part(self, split: str = "::"):
        return self.id[self.id.rfind(split) + len(split):]

    def is_folder(self):
        return self.type in _FOLDER_TYPES

    def get_dict_key(self):
        if not isinstance(self.dictKey, Undefined):
            return self.dictKey.data
        return undefined

    def _get_node_by_uid(self, uid: str, split: str = "::"):
        """TODO if dict key contains split word, this function will
        produce wrong result.
        """
        parts = uid.split(split)
        if len(parts) == 1:
            return self
        # uid contains root, remove it at first.
        return self._get_node_by_uid_resursive(parts[1:])

    def _get_node_by_uid_resursive(self, parts: List[str]) -> "JsonLikeNode":
        key = parts[0]
        node: Optional[JsonLikeNode] = None
        for c in self.children:
            # TODO should we use id.split[-1] instead of name?
            if c.last_part() == key:
                node = c
                break
        assert node is not None, f"{key} missing"
        if len(parts) == 1:
            return node
        else:
            return node._get_node_by_uid_resursive(parts[1:])

    def _get_node_by_uid_trace(self, uid: str, split: str = "::"):
        parts = uid.split(split)
        if len(parts) == 1:
            return [self]
        # uid contains root, remove it at first.
        nodes, found = self._get_node_by_uid_resursive_trace(
            parts[1:], check_missing=True)
        assert found
        return [self] + nodes

    def _get_node_by_uid_trace_found(self, uid: str, split: str = "::"):
        parts = uid.split(split)
        if len(parts) == 1:
            return [self], True
        # uid contains root, remove it at first.
        res = self._get_node_by_uid_resursive_trace(parts[1:])
        return [self] + res[0], res[1]

    def _get_node_by_uid_resursive_trace(
            self,
            parts: List[str],
            check_missing: bool = False) -> Tuple[List["JsonLikeNode"], bool]:
        key = parts[0]
        node: Optional[JsonLikeNode] = None
        for c in self.children:
            # TODO should we use id.split[-1] instead of name?
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
                node = JsonLikeNode(self.id + f"{split}{i}",
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
                node = JsonLikeNode(self.id + f"{split}{i}",
                                    f"{i}",
                                    JsonLikeType.DictFolder.value,
                                    cnt=this_cnt,
                                    realId=real_id,
                                    start=start + count,
                                    keys=BackendOnlyProp(keys_child))
                res.append(node)
                count += this_cnt
        return res


def parse_obj_to_jsonlike(obj, name: str, id: str):
    obj_type = type(obj)
    if obj is None or obj is Ellipsis:
        return JsonLikeNode(id,
                            name,
                            JsonLikeType.Constant.value,
                            value=str(obj))
    elif isinstance(obj, JsonLikeNode):
        # TODO should we check obj name/id?
        return obj
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
    elif get_qualname_of_type(obj_type) == CommonQualNames.TorchTensor:
        t = JsonLikeType.Tensor
        shape_short = ",".join(map(str, obj.shape))
        return JsonLikeNode(id,
                            name,
                            t.value,
                            typeStr="torch.Tensor",
                            value=f"[{shape_short}]{obj.dtype}",
                            drag=True)
    elif get_qualname_of_type(obj_type) == CommonQualNames.TVTensor:
        t = JsonLikeType.Tensor
        shape_short = ",".join(map(str, obj.shape))
        return JsonLikeNode(id,
                            name,
                            t.value,
                            typeStr="tv.Tensor",
                            value=f"[{shape_short}]{obj.dtype}",
                            drag=True)
    else:
        t = JsonLikeType.Object
        value = undefined
        return JsonLikeNode(id,
                            name,
                            t.value,
                            value=value,
                            typeStr=obj_type.__qualname__)
