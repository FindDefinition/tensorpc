
from collections.abc import Mapping, Sequence
import copy
import dataclasses
from typing import Any, AsyncGenerator, Callable, ClassVar, Dict, List, Optional, Set, Tuple, Type, TypeVar, TypedDict, Union, Generic
from typing_extensions import Literal, Annotated, NotRequired, Protocol, get_origin, get_args, get_type_hints
from dataclasses import dataclass
from dataclasses import Field, make_dataclass, field
import inspect
import sys 
import typing 
import types
from pydantic_core import PydanticCustomError, core_schema
from pydantic import (
    GetCoreSchemaHandler, )


from tensorpc import compat
from tensorpc.core.tree_id import UniqueTreeId

class DataclassType(Protocol):
    # as already noted in comments, checking for this attribute is currently
    # the most reliable way to ascertain that something is a dataclass
    __dataclass_fields__: ClassVar[Dict[str, Any]]

if sys.version_info < (3, 10):

    def origin_is_union(tp: Optional[Type[Any]]) -> bool:
        return tp is typing.Union

else:

    def origin_is_union(tp: Optional[Type[Any]]) -> bool:
        return tp is typing.Union or tp is types.UnionType  # noqa: E721

def lenient_issubclass(cls: Any,
                       class_or_tuple: Any) -> bool:  # pragma: no cover
    return isinstance(cls, type) and issubclass(cls, class_or_tuple)

def is_annotated(ann_type: Any) -> bool:
    # https://github.com/pydantic/pydantic/blob/35144d05c22e2e38fe093c533ff3a05ce9a30116/pydantic/_internal/_typing_extra.py#L99C1-L104C1
    origin = get_origin(ann_type)
    return origin is not None and lenient_issubclass(origin, Annotated)

def is_not_required(ann_type: Any) -> bool:
    # https://github.com/pydantic/pydantic/blob/35144d05c22e2e38fe093c533ff3a05ce9a30116/pydantic/_internal/_typing_extra.py#L99C1-L104C1
    origin = get_origin(ann_type)
    return origin is not None and origin is NotRequired

def is_optional(ann_type: Any) -> bool:
    origin = get_origin(ann_type)
    return origin is not None and origin_is_union(origin) and type(None) in get_args(ann_type)

def is_async_gen(ann_type: Any) -> bool:
    # https://github.com/pydantic/pydantic/blob/35144d05c22e2e38fe093c533ff3a05ce9a30116/pydantic/_internal/_typing_extra.py#L99C1-L104C1
    origin = get_origin(ann_type)
    return origin is not None and lenient_issubclass(origin, AsyncGenerator)

_DCLS_GET_TYPE_HINTS_CACHE: dict[Any, dict[str, Any]] = {}

def get_type_hints_with_cache(cls, include_extras: bool = False):
    if cls not in _DCLS_GET_TYPE_HINTS_CACHE:
        _DCLS_GET_TYPE_HINTS_CACHE[cls] = get_type_hints(cls, include_extras=include_extras)
    return _DCLS_GET_TYPE_HINTS_CACHE[cls]

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

    def bool(self):
        return False

T = TypeVar("T")

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

@dataclass
class AnnotatedArg:
    name: str 
    param: Optional[inspect.Parameter] 
    type: Any 
    annometa: Optional[Tuple[Any, ...]] = None 

@dataclass
class AnnotatedReturn:
    type: Any 
    annometa: Optional[Tuple[Any, ...]] = None 

def extract_annotated_type_and_meta(ann_type: Any) -> Tuple[Any, Optional[Any]]:
    if is_annotated(ann_type):
        annometa = ann_type.__metadata__
        ann_type = get_args(ann_type)[0]
        return ann_type, annometa
    return ann_type, None

@dataclass
class AnnotatedType:
    origin_type: Any
    child_types: list[Any]
    annometa: Optional[Tuple[Any, ...]] = None
    is_optional: bool = False
    is_undefined: bool = False

    def is_any_type(self) -> bool:
        return self.origin_type is Any

    def is_union_type(self) -> bool:
        return origin_is_union(self.origin_type)

    def is_dataclass_type(self) -> bool:
        return dataclasses.is_dataclass(self.origin_type)

    def is_dict_type(self) -> bool:
        return issubclass(self.origin_type, dict)

    def is_list_type(self) -> bool:
        return issubclass(self.origin_type, list)

    def is_sequence_type(self) -> bool:
        return issubclass(self.origin_type, Sequence) and not issubclass(self.origin_type, str)

    def is_mapping_type(self) -> bool:
        return issubclass(self.origin_type, Mapping)

    def get_dict_key_anno_type(self) -> "AnnotatedType":
        assert self.is_dict_type() and len(self.child_types) == 2
        # we forward all optional or undefined to child types to make sure user know its a optional path.
        return parse_type_may_optional_undefined(self.child_types[0], is_optional=self.is_optional, is_undefined=self.is_undefined)

    def get_dict_value_anno_type(self) -> "AnnotatedType":
        assert self.is_dict_type() and len(self.child_types) == 2
        return parse_type_may_optional_undefined(self.child_types[1], is_optional=self.is_optional, is_undefined=self.is_undefined)

    def get_mapping_value_anno_type(self) -> "AnnotatedType":
        assert self.is_mapping_type() and len(self.child_types) == 2
        return parse_type_may_optional_undefined(self.child_types[1], is_optional=self.is_optional, is_undefined=self.is_undefined)

    def get_list_value_anno_type(self) -> "AnnotatedType":
        assert self.is_list_type()
        return parse_type_may_optional_undefined(self.child_types[0], is_optional=self.is_optional, is_undefined=self.is_undefined)

    def get_seq_value_anno_type(self) -> "AnnotatedType":
        assert self.is_sequence_type()
        return parse_type_may_optional_undefined(self.child_types[0], is_optional=self.is_optional, is_undefined=self.is_undefined)

    def get_child_annotated_type(self, index: int) -> "AnnotatedType":
        return parse_type_may_optional_undefined(self.child_types[index], is_optional=self.is_optional, is_undefined=self.is_undefined)

    def get_dataclass_field_annotated_types(self) -> dict[str, "AnnotatedType"]:
        assert self.is_dataclass_type()
        type_hints = get_type_hints_with_cache(self.origin_type, include_extras=True)
        return {field.name: parse_type_may_optional_undefined(type_hints[field.name], 
            is_optional=self.is_optional, is_undefined=self.is_undefined) for field in dataclasses.fields(self.origin_type)}

    @staticmethod 
    def get_any_type():
        return AnnotatedType(Any, [])

def parse_type_may_optional_undefined(ann_type: Any, is_optional: Optional[bool] = None, is_undefined: Optional[bool] = None) -> AnnotatedType:
    """Parse a type. If is union, return its non-optional and non-undefined type list.
    else return the type itself.
    """
    ann_type, ann_meta = extract_annotated_type_and_meta(ann_type)
    # check ann_type is Union
    ty_origin = get_origin(ann_type)
    if ty_origin is not None:
        if origin_is_union(ty_origin):
            ty_args = get_args(ann_type)
            is_optional = is_optional or False
            is_undefined = is_undefined or False
            for ty in ty_args:
                if ty is type(None):
                    is_optional = True
                elif ty is Undefined:
                    is_undefined = True
            ty_args = [ty for ty in ty_args if ty is not type(None) and ty is not Undefined]
            if len(ty_args) == 1:
                res = parse_type_may_optional_undefined(ty_args[0])
                res.is_optional = is_optional
                res.is_undefined = is_undefined
                return res
            assert inspect.isclass(ty_origin), f"origin type must be a class, but get {ty_origin}"
            return AnnotatedType(ty_origin, ty_args, ann_meta, is_optional, is_undefined)
        else:
            ty_args = get_args(ann_type)
            assert inspect.isclass(ty_origin), f"origin type must be a class, but get {ty_origin}"
            return AnnotatedType(ty_origin, list(ty_args), ann_meta)
    return AnnotatedType(ann_type, [], ann_meta)

def child_type_generator(t: type):
    yield t
    args = get_args(t)
    for arg in args:
        yield from child_type_generator(arg)

def child_type_generator_with_dataclass(t: type):
    yield t
    if dataclasses.is_dataclass(t):
        type_hints = get_type_hints_with_cache(t, include_extras=True)
        for field in dataclasses.fields(t):
            yield from child_type_generator(type_hints[field.name])
    else:
        args = get_args(t)
        for arg in args:
            yield from child_type_generator(arg)

def parse_annotated_function(func: Callable, is_dynamic_class: bool = False) -> Tuple[List[AnnotatedArg], Optional[AnnotatedReturn]]:
    if compat.Python3_10AndLater:
        annos = get_type_hints(func, include_extras=True)
    else:
        annos = get_type_hints(func, include_extras=True, globalns={} if is_dynamic_class else None)
    
    specs = inspect.signature(func)
    name_to_parameter = {p.name: p for p in specs.parameters.values()}
    anno_args: List[AnnotatedArg] = []
    return_anno: Optional[AnnotatedReturn] = None
    for name, anno in annos.items():
        if name == "return":
            anno, annotated_metas = extract_annotated_type_and_meta(anno)
            return_anno = AnnotatedReturn(anno, annotated_metas)
        else:
            param = name_to_parameter[name]
            anno, annotated_metas = extract_annotated_type_and_meta(anno)

            arg_anno = AnnotatedArg(name, param, anno, annotated_metas)
            anno_args.append(arg_anno)
    for name, param in name_to_parameter.items():
        if name not in annos and param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            anno_args.append(AnnotatedArg(name, param, Any))
    return anno_args, return_anno

def annotated_function_to_dataclass(func: Callable, is_dynamic_class: bool = False):
    if compat.Python3_10AndLater:
        annos = get_type_hints(func, include_extras=True)
    else:
        annos = get_type_hints(func, include_extras=True, globalns={} if is_dynamic_class else None)
    specs = inspect.signature(func)
    name_to_parameter = {p.name: p for p in specs.parameters.values()}
    fields: List[Tuple[str, Any, Field]] = []
    for name, anno in annos.items():
        param = name_to_parameter[name]
        assert param.default is not inspect.Parameter.empty, "annotated function arg must have default value"
        fields.append((name, anno, field(default=param.default)))
    return make_dataclass(func.__name__, fields)

def _main():

    class WTF(TypedDict):
        pass 

    class WTF2(WTF):
        c: int

    class A:
        def add(self, a: int, b: int) -> WTF2:
            return WTF2(c=a + b)

        @staticmethod
        def add_stc(a: int, b: int) -> int:
            return a + b
    a = A()
    print(issubclass(WTF2,dict))
    print(dir(WTF2))
    print(parse_annotated_function(a.add))
    print(parse_annotated_function(a.add_stc))
    print(is_optional(Optional[int]))
    print(is_async_gen(AsyncGenerator[int, None]))
    print(is_not_required(NotRequired[int])) # type: ignore
    print(is_not_required(Optional[int]))
    @dataclass
    class Model:
        a: dict 
        b: dict[str, Any]
        c: Optional[dict[str, Any]]
        d: list 
        e: list[int]
        f: List[int]
        g: Sequence[int]

    for field in dataclasses.fields(Model):
        at = parse_type_may_optional_undefined(field.type)
        print(at, at.is_list_type(), at.is_sequence_type(), at.is_mapping_type(), at.is_dict_type())

if __name__ == "__main__":
    _main()