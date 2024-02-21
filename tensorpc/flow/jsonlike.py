import enum

from typing import Any, Callable, Dict, Generic, Hashable, List, Optional, TypeVar, Union, Tuple
import tensorpc.core.dataclass_dispatch as dataclasses
import re
import numpy as np
from tensorpc.core.moduleid import get_qualname_of_type
from typing_extensions import (Concatenate, Literal, ParamSpec, Protocol, Self,
                               TypeAlias)
import abc
from collections.abc import MutableMapping
import copy

from pydantic_core import PydanticCustomError, core_schema
from pydantic import (
    GetCoreSchemaHandler, )

from tensorpc.core.tree_id import UniqueTreeId, UniqueTreeIdForTree

ValueType: TypeAlias = Union[int, float, str]
NumberType: TypeAlias = Union[int, float]

STRING_LENGTH_LIMIT = 500
T = TypeVar("T")


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


class Undefined:
    def __repr__(self) -> str:
        return "undefined"

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any,
                                     _handler: GetCoreSchemaHandler):
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.any_schema(),
        )

    @classmethod
    def validate(cls, v):
        if not isinstance(v, Undefined):
            raise ValueError('undefined required, but get', type(v))
        return v

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Undefined)

    def __ne__(self, o: object) -> bool:
        return not isinstance(o, Undefined)

    def __hash__(self) -> int:
        # for python 3.11
        return 0


class BackendOnlyProp(Generic[T]):
    """when wrap a property with this class, it will be ignored when serializing to frontend
    """
    def __init__(self, data: T) -> None:
        super().__init__()
        self.data = data

    def __repr__(self) -> str:
        return "BackendOnlyProp"

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any,
                                     _handler: GetCoreSchemaHandler):
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.any_schema(),
        )

    @classmethod
    def validate(cls, v):
        if not isinstance(v, BackendOnlyProp):
            raise ValueError('BackendOnlyProp required')
        return cls(v.data)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, BackendOnlyProp):
            return o.data == self.data
        else:
            return o == self.data

    def __ne__(self, o: object) -> bool:
        if isinstance(o, BackendOnlyProp):
            return o.data != self.data
        else:
            return o != self.data


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


def undefined_dict_factory(x: List[Tuple[str, Any]]):
    res: Dict[str, Any] = {}
    for k, v in x:
        if isinstance(v, UniqueTreeId):
            res[k] = v.uid_encoded
        elif not isinstance(v, (Undefined, BackendOnlyProp)):
            res[k] = v
    return res


@dataclasses.dataclass
class _DataclassSer:
    obj: Any


def as_dict_no_undefined(obj: Any):
    return dataclasses.asdict(_DataclassSer(obj),
                              dict_factory=undefined_dict_factory)["obj"]


def asdict_field_only(obj,
                      *,
                      dict_factory: Callable[[List[Tuple[str, Any]]],
                                             Dict[str, Any]] = dict):
    "(list[tuple[str, Any]]) -> dict[str, Any]"
    """same as dataclasses.asdict except that this function
    won't recurse into nested container.
    """
    if not dataclasses.is_dataclass(obj):
        raise TypeError("asdict() should be called on dataclass instances")
    return _asdict_field_only_inner(obj, dict_factory)


def _asdict_field_only_inner(obj, dict_factory):
    if dataclasses.is_dataclass(obj):
        result = []
        for f in dataclasses.fields(obj):
            value = _asdict_field_only_inner(getattr(obj, f.name),
                                             dict_factory)
            result.append((f.name, value))
        return dict_factory(result)
    else:
        return copy.deepcopy(obj)


def asdict_flatten_field_only(obj,
                              *,
                              dict_factory: Callable[[List[Tuple[str, Any]]],
                                                     Dict[str, Any]] = dict):
    """same as dataclasses.asdict except that this function
    won't recurse into nested container.
    """
    if not dataclasses.is_dataclass(obj):
        raise TypeError("asdict() should be called on dataclass instances")
    return _asdict_flatten_field_only(obj, dict_factory)


def _asdict_flatten_field_only(obj,
                               dict_factory,
                               parent_key: str = '',
                               sep: str = '.'):
    result = []
    for f in dataclasses.fields(obj):
        obj_child = getattr(obj, f.name)
        new_key = parent_key + sep + f.name if parent_key else f.name
        if dataclasses.is_dataclass(obj_child):
            result.extend(
                _asdict_flatten_field_only(obj_child,
                                           dict_factory,
                                           new_key,
                                           sep=sep).items())
        else:
            result.append((new_key, obj_child))
    return dict_factory(result)


def asdict_no_deepcopy(obj,
                       *,
                       dict_factory: Callable[[List[Tuple[str, Any]]],
                                              Dict[str, Any]] = dict,
                       obj_factory: Optional[Callable[[Any], Any]] = None):
    """Return the fields of a dataclass instance as a new dictionary mapping
    field names to field values.

    Example usage:

      @dataclass
      class C:
          x: int
          y: int

      c = C(1, 2)
      assert asdict(c) == {'x': 1, 'y': 2}

    If given, 'dict_factory' will be used instead of built-in dict.
    The function applies recursively to field values that are
    dataclass instances. This will also look into built-in containers:
    tuples, lists, and dicts.
    """
    if not dataclasses.is_dataclass(obj):
        raise TypeError("asdict() should be called on dataclass instances")
    return _asdict_inner(obj, dict_factory, obj_factory)


def _asdict_inner(obj, dict_factory, obj_factory=None):
    if dataclasses.is_dataclass(obj):
        result = []
        for f in dataclasses.fields(obj):
            value = _asdict_inner(getattr(obj, f.name), dict_factory,
                                  obj_factory)
            result.append((f.name, value))
        return dict_factory(result)
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        # obj is a namedtuple.  Recurse into it, but the returned
        # object is another namedtuple of the same type.  This is
        # similar to how other list- or tuple-derived classes are
        # treated (see below), but we just need to create them
        # differently because a namedtuple's __init__ needs to be
        # called differently (see bpo-34363).

        # I'm not using namedtuple's _asdict()
        # method, because:
        # - it does not recurse in to the namedtuple fields and
        #   convert them to dicts (using dict_factory).
        # - I don't actually want to return a dict here.  The main
        #   use case here is json.dumps, and it handles converting
        #   namedtuples to lists.  Admittedly we're losing some
        #   information here when we produce a json list instead of a
        #   dict.  Note that if we returned dicts here instead of
        #   namedtuples, we could no longer call asdict() on a data
        #   structure where a namedtuple was used as a dict key.

        return type(obj)(
            *[_asdict_inner(v, dict_factory, obj_factory) for v in obj])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(_asdict_inner(v, dict_factory, obj_factory)
                         for v in obj)
    elif isinstance(obj, dict):
        return type(obj)((_asdict_inner(k, dict_factory, obj_factory),
                          _asdict_inner(v, dict_factory, obj_factory))
                         for k, v in obj.items())
    else:
        if obj_factory is not None:
            obj = obj_factory(obj)
        return obj


@dataclasses.dataclass
class DataClassWithUndefined:
    def get_dict_and_undefined(
            self,
            state: Dict[str, Any],
            dict_factory: Callable[[List[Tuple[str, Any]]],
                                   Dict[str, Any]] = undefined_dict_factory,
            obj_factory: Optional[Callable[[Any], Any]] = None):
        this_type = type(self)
        res = {}
        # we only support update in first-level dict,
        # so we ignore all undefined in childs.
        ref_dict = asdict_no_deepcopy(self,
                                      dict_factory=dict_factory,
                                      obj_factory=obj_factory)
        assert isinstance(ref_dict, dict)
        # ref_dict = dataclasses.asdict(self,
        #                               dict_factory=undefined_dict_factory)
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

    def get_dict(self,
                 to_camel: bool = True,
                 dict_factory: Callable[[List[Tuple[str, Any]]],
                                        Dict[str,
                                             Any]] = undefined_dict_factory,
                 obj_factory: Optional[Callable[[Any], Any]] = None):
        this_type = type(self)
        res = {}
        ref_dict = asdict_no_deepcopy(self,
                                      dict_factory=dict_factory,
                                      obj_factory=obj_factory)
        assert isinstance(ref_dict, dict)
        # ref_dict = dataclasses.asdict(self,
        #                               dict_factory=undefined_dict_factory)
        for field in dataclasses.fields(this_type):
            if to_camel:
                res_camel = snake_to_camel(field.name)
            else:
                res_camel = field.name
            if field.name not in ref_dict:
                val = undefined
            else:
                val = ref_dict[field.name]
            res[res_camel] = val
        return res

    def get_flatten_dict(
        self,
        dict_factory: Callable[[List[Tuple[str, Any]]],
                               Dict[str, Any]] = undefined_dict_factory):
        this_type = type(self)
        res = {}
        ref_dict = asdict_flatten_field_only(self, dict_factory=dict_factory)
        for field in dataclasses.fields(this_type):
            res_camel = field.name
            if field.name not in ref_dict:
                val = undefined
            else:
                val = ref_dict[field.name]
            res[res_camel] = val
        return res


class CommonQualNames:
    TorchTensor = "torch.Tensor"
    TVTensor = "cumm.core_cc.tensorview_bind.Tensor"


class TensorType(enum.Enum):
    Unknown = ""
    NpArray = "numpy.ndarray"
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


@dataclasses.dataclass(eq=True)
class IconButtonData:
    id: ValueType
    icon: int
    tooltip: Union[Undefined, str] = undefined


@dataclasses.dataclass(eq=True)
class ContextMenuData:
    title: str
    id: Union[Undefined, ValueType] = undefined
    icon: Union[Undefined, int] = undefined
    userdata: Union[Undefined, Any] = undefined


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
    dictKey: Union[Undefined, BackendOnlyProp[Hashable]] = undefined
    keys: Union[Undefined, BackendOnlyProp[List[str]]] = undefined
    menus: Union[Undefined, List[ContextMenuData]] = undefined
    edit: Union[Undefined, bool] = undefined
    userdata: Union[Undefined, Any] = undefined
    alias: Union[Undefined, str] = undefined
    fixedIconBtns: Union[Undefined, List[IconButtonData]] = undefined

    # @staticmethod
    # def decode_uid(uid: str, split_length: int = 2):
    #     index = uid.find("|")
    #     lengths = list(map(int, uid[:index].split(",")))
    #     res: List[str] = []
    #     start = index + 1
    #     for l in lengths:
    #         end = start + l
    #         res.append(uid[start:end])
    #         start = end + split_length
    #     return res

    # @staticmethod
    # def encode_uid(parts: List[str], split: str = ":"):
    #     lengths = [str(len(p)) for p in parts]
    #     lengths_str = f",".join(lengths)
    #     return f"{lengths_str}|{split.join(parts)}"

    @staticmethod
    def decode_uid_legacy(uid: str, split: str = ":"):
        return uid.split(split)

    def last_part(self, split: str = ":"):
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
        uid_object = UniqueTreeId(uid, 1)
        parts = uid_object.parts
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

    def _get_node_by_uid_trace(self, uid_parts: List[str]):
        parts = uid_parts
        if len(parts) == 1:
            return [self]
        # uid contains root, remove it at first.
        nodes, found = self._get_node_by_uid_resursive_trace(
            parts[1:], check_missing=True)
        assert found
        return [self] + nodes

    def _get_node_by_uid_trace_found(self, uid_parts: List[str]):
        parts = uid_parts
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


class TreeItem(abc.ABC):
    @abc.abstractmethod
    async def get_child_desps(self) -> Dict[str, JsonLikeNode]:
        raise NotImplementedError

    @abc.abstractmethod
    async def get_child(self, key: str) -> Any:
        raise NotImplementedError

    def get_json_like_node(self) -> Optional[JsonLikeNode]:
        """name and id is determined by parent, only root node use name provided
        by this method.
        """
        return None

    async def handle_button(self, button_key: str) -> Optional[bool]:
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
