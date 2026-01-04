"""
https://github.com/wyfo/apischema/blob/master/apischema/typing.py
https://github.com/pydantic/pydantic/blob/main/pydantic/_internal/_typing_extra.py
"""

import collections.abc
from collections.abc import Mapping, Sequence
import copy
import dataclasses
from functools import partial
from typing import Any, AsyncGenerator, Callable, ClassVar, Dict, Generator, List, Optional, Set, Tuple, Type, TypeVar, TypedDict, Union, Generic
from typing_extensions import Literal, Annotated, NotRequired, Protocol, get_origin, get_args, get_type_hints, TypeGuard, TypeIs, Self, ParamSpec, ParamSpecArgs
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
from tensorpc.core.moduleid import get_qualname_of_type
from tensorpc.core.tree_id import UniqueTreeId


class DataclassType(Protocol):
    # as already noted in comments, checking for this attribute is currently
    # the most reliable way to ascertain that something is a dataclass
    __dataclass_fields__: ClassVar[Dict[str, Any]]

T_dataclass = TypeVar("T_dataclass", bound=DataclassType)

if sys.version_info < (3, 10):

    def origin_is_union(tp: Optional[Type[Any]]) -> bool:
        return tp is typing.Union

else:

    def origin_is_union(tp: Optional[Type[Any]]) -> bool:
        return tp is typing.Union or tp is types.UnionType  # noqa: E721


def lenient_issubclass(cls: Any,
                       class_or_tuple: Any) -> bool:  # pragma: no cover
    return isinstance(cls, type) and issubclass(cls, class_or_tuple)


def is_annotated(ann_type: Any) -> TypeGuard[Annotated]:
    # https://github.com/pydantic/pydantic/blob/35144d05c22e2e38fe093c533ff3a05ce9a30116/pydantic/_internal/_typing_extra.py#L99C1-L104C1
    origin = get_origin(ann_type)
    return origin is not None and lenient_issubclass(origin, Annotated)


def is_not_required(ann_type: Any) -> bool:
    # https://github.com/pydantic/pydantic/blob/35144d05c22e2e38fe093c533ff3a05ce9a30116/pydantic/_internal/_typing_extra.py#L99C1-L104C1
    origin = get_origin(ann_type)
    return origin is not None and origin is NotRequired


def is_optional(ann_type: Any) -> bool:
    origin = get_origin(ann_type)
    return origin is not None and origin_is_union(origin) and type(
        None) in get_args(ann_type)


def is_async_gen(ann_type: Any) -> bool:
    # https://github.com/pydantic/pydantic/blob/35144d05c22e2e38fe093c533ff3a05ce9a30116/pydantic/_internal/_typing_extra.py#L99C1-L104C1
    origin = get_origin(ann_type)
    return origin is not None and lenient_issubclass(origin, AsyncGenerator)


_DCLS_GET_TYPE_HINTS_CACHE: dict[Any, dict[str, Any]] = {}


def get_type_hints_with_cache(cls, include_extras: bool = False):
    if cls not in _DCLS_GET_TYPE_HINTS_CACHE:
        _DCLS_GET_TYPE_HINTS_CACHE[cls] = get_type_hints(
            cls, include_extras=include_extras)
    return _DCLS_GET_TYPE_HINTS_CACHE[cls]

_DCLS_RESOLVE_TYPE_HINTS_CACHE: dict[Any, dict[str, Any]] = {}

def resolve_type_hints_with_cache(cls):
    if cls not in _DCLS_RESOLVE_TYPE_HINTS_CACHE:
        _DCLS_RESOLVE_TYPE_HINTS_CACHE[cls] = resolve_type_hints(
            cls)
    return _DCLS_RESOLVE_TYPE_HINTS_CACHE[cls]

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

    def __bool__(self):
        return False

# DON'T MODIFY THIS VALUE!!!
undefined = Undefined()

def is_undefined(val: object) -> TypeIs[Undefined]:
    return isinstance(val, Undefined)

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

if sys.version_info >= (3, 13):
    from typing import _collect_type_parameters
elif sys.version_info >= (3, 11):
    from typing import _collect_parameters as _collect_type_parameters  # type: ignore
else:
    from typing import _collect_type_vars as _collect_type_parameters


def _generic_mro(result, tp):
    origin = get_origin(tp)
    if origin is None:
        origin = tp
    result[origin] = tp
    if hasattr(origin, "__orig_bases__"):
        parameters = _collect_type_parameters(origin.__orig_bases__)
        substitution = dict(zip(parameters, get_args(tp)))
        for base in origin.__orig_bases__:
            if get_origin(base) in result:
                continue
            base_parameters = getattr(base, "__parameters__", ())
            if base_parameters:
                base = base[tuple(substitution.get(p, p) for p in base_parameters)]
            _generic_mro(result, base)


# sentinel value to avoid to subscript Generic and Protocol
BASE_GENERIC_MRO = {Generic: Generic, Protocol: Protocol}


def generic_mro(tp):
    origin = get_origin(tp)
    if origin is None and not hasattr(tp, "__orig_bases__"):
        if not isinstance(tp, type):
            raise TypeError(f"{tp!r} is not a type or a generic alias")
        return tp.__mro__
    result = BASE_GENERIC_MRO.copy()
    _generic_mro(result, tp)
    cls = origin if origin is not None else tp
    return tuple(result.get(sub_cls, sub_cls) for sub_cls in cls.__mro__)


def resolve_type_hints(obj: Any) -> Dict[str, Any]:
    """Wrap get_type_hints to resolve type vars in case of generic inheritance.

    `obj` can also be a parametrized generic class.

    Copied from https://github.com/wyfo/apischema/blob/master/apischema/typing.py
    """
    origin_or_obj = get_origin(obj) or obj
    if isinstance(origin_or_obj, type):
        hints = {}
        for base in reversed(generic_mro(obj)):
            base_origin = get_origin(base) or base
            base_annotations = getattr(base_origin, "__dict__", {}).get(
                "__annotations__", {}
            )
            substitution = dict(
                zip(getattr(base_origin, "__parameters__", ()), get_args(base))
            )
            for name, hint in get_type_hints(base_origin, include_extras=True).items():
                if name not in base_annotations:
                    continue
                if isinstance(hint, TypeVar):
                    hints[name] = substitution.get(hint, hint) # type: ignore
                elif getattr(hint, "__parameters__", ()):
                    hints[name] = (Union if origin_is_union(hint) else hint)[
                        tuple(substitution.get(p, p) for p in hint.__parameters__)
                    ]
                else:
                    hints[name] = hint
        return hints
    else:
        return get_type_hints(obj, include_extras=True)


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


def extract_annotated_type_and_meta(
        ann_type: Any) -> Tuple[Any, Optional[Any]]:
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
    raw_type: Optional[Any] = None
    # currently only used for tuple type
    is_homogeneous: bool = False

    def get_optional_undefined_removed(self):
        return dataclasses.replace(
            self,
            is_optional=False,
            is_undefined=False)

    def is_any_type(self) -> bool:
        return self.origin_type is Any

    def is_type_var(self) -> bool:
        return isinstance(self.origin_type, TypeVar)

    def is_param_spec(self) -> bool:
        return isinstance(self.origin_type, ParamSpec)

    def is_param_spec_args(self) -> bool:
        return isinstance(self.origin_type, ParamSpecArgs)

    def is_tuple_type(self) -> bool:
        return self.origin_type is tuple or self.origin_type is Tuple

    def is_union_type(self) -> bool:
        return origin_is_union(self.origin_type)

    def is_number_type(self) -> bool:
        if inspect.isclass(self.origin_type) and issubclass(self.origin_type, (int, float)):
            return True
        if not self.is_union_type():
            return False
        for ty in self.child_types:
            if not issubclass(ty, (int, float)):
                return False
        return True

    def is_bool_type(self) -> bool:
        if inspect.isclass(self.origin_type) and issubclass(self.origin_type, bool):
            return True
        return False

    def is_dataclass_type(self) -> bool:
        return dataclasses.is_dataclass(self.origin_type)

    def _is_non_class_base_type(self):
        return self.is_union_type() or self.is_any_type() or self.is_tuple_type() or self.is_type_var()

    def is_dict_type(self) -> bool:
        if self._is_non_class_base_type():
            return False
        return issubclass(self.origin_type, dict)

    def is_callable(self) -> bool:
        if self._is_non_class_base_type():
            return False
        return self.origin_type is Callable

    def is_list_type(self) -> bool:
        if self._is_non_class_base_type():
            return False
        return issubclass(self.origin_type, list)

    def is_sequence_type(self) -> bool:
        if self._is_non_class_base_type():
            return False
        assert inspect.isclass(
            self.origin_type
        ), f"origin type must be a class, but get {self.origin_type} {type(self.origin_type)}"
        return issubclass(self.origin_type,
                          Sequence) and not issubclass(self.origin_type, str)

    def is_mapping_type(self) -> bool:
        if self._is_non_class_base_type():
            return False
        assert inspect.isclass(
            self.origin_type
        ), f"origin type must be a class, but get {self.origin_type}"
        return issubclass(self.origin_type, Mapping)

    def get_dict_key_anno_type(self) -> "AnnotatedType":
        assert self.is_dict_type() and len(self.child_types) == 2
        # we forward all optional or undefined to child types to make sure user know its a optional path.
        return parse_type_may_optional_undefined(
            self.child_types[0],
            is_optional=self.is_optional,
            is_undefined=self.is_undefined)

    def get_dict_value_anno_type(self) -> "AnnotatedType":
        assert self.is_dict_type() and len(self.child_types) == 2
        return parse_type_may_optional_undefined(
            self.child_types[1],
            is_optional=self.is_optional,
            is_undefined=self.is_undefined)

    def get_mapping_value_anno_type(self) -> "AnnotatedType":
        assert self.is_mapping_type() and len(self.child_types) == 2
        return parse_type_may_optional_undefined(
            self.child_types[1],
            is_optional=self.is_optional,
            is_undefined=self.is_undefined)

    def get_list_value_anno_type(self) -> "AnnotatedType":
        assert self.is_list_type()
        return parse_type_may_optional_undefined(
            self.child_types[0],
            is_optional=self.is_optional,
            is_undefined=self.is_undefined)

    def get_seq_value_anno_type(self) -> "AnnotatedType":
        assert self.is_sequence_type()
        return parse_type_may_optional_undefined(
            self.child_types[0],
            is_optional=self.is_optional,
            is_undefined=self.is_undefined)

    def get_child_annotated_type(self, index: int) -> "AnnotatedType":
        return parse_type_may_optional_undefined(
            self.child_types[index],
            is_optional=self.is_optional,
            is_undefined=self.is_undefined)

    def get_dataclass_field_annotated_types(
            self) -> dict[str, "AnnotatedType"]:
        assert self.is_dataclass_type()
        type_hints = get_type_hints_with_cache(self.origin_type,
                                               include_extras=True)
        return {
            field.name:
            parse_type_may_optional_undefined(type_hints[field.name],
                                              is_optional=self.is_optional,
                                              is_undefined=self.is_undefined)
            for field in dataclasses.fields(self.origin_type)
        }

    def get_dataclass_fields_and_annotated_types(
            self) -> dict[str, tuple["AnnotatedType", Field]]:
        assert self.is_dataclass_type()
        return self.get_dataclass_fields_and_annotated_types_static(self.origin_type)

    @staticmethod
    def get_dataclass_fields_and_annotated_types_static(
            type: Any) -> dict[str, tuple["AnnotatedType", Field]]:
        assert inspect.isclass(type) and dataclasses.is_dataclass(type), \
            f"type must be a dataclass, but get {type}"
        type_hints = get_type_hints_with_cache(type,
                                               include_extras=True)
        res: dict[str, tuple["AnnotatedType", Field]] = {}
        for field in dataclasses.fields(type):
            field_type = type_hints[field.name]
            field_annotype = parse_type_may_optional_undefined(field_type)
            res[field.name] = (field_annotype, field)
        return res


    @staticmethod
    def get_any_type():
        return AnnotatedType(Any, [])

    def get_annometa(self, metatype: Type[T]) -> Optional[T]:
        """Get the annometa of the specified type, if not found, return None."""
        if self.annometa is None:
            return None
        for meta in self.annometa:
            if isinstance(meta, metatype):
                return meta
        return None

def parse_type_may_optional_undefined(
        ann_type: Any,
        is_optional: Optional[bool] = None,
        is_undefined: Optional[bool] = None,
        self_type: Optional[AnnotatedType] = None) -> AnnotatedType:
    """Parse a type. If is union, return its non-optional and non-undefined type list.
    else return the type itself.

    WARNING: use Annotated[Optional], don't use Optional[Annotated]
    """
    raw_type = ann_type
    ann_type, ann_meta = extract_annotated_type_and_meta(ann_type)
    if isinstance(ann_type, (TypeVar, ParamSpec, ParamSpecArgs)):
        return AnnotatedType(ann_type, [], ann_meta, False,
                                 False, ann_type)
    # check ann_type is Union
    assert not isinstance(ann_type, str), "you must evaluate your annotation"
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
            ty_args = [
                ty for ty in ty_args
                if ty is not type(None) and ty is not Undefined
            ]
            if len(ty_args) == 1:
                res = parse_type_may_optional_undefined(ty_args[0])
                res.is_optional = is_optional
                res.is_undefined = is_undefined
                # use Annotated[Optional], Optional[Annotated] won't work
                res.annometa = ann_meta
                return res
            # assert inspect.isclass(
            #     ty_origin), f"origin type must be a class, but get {ty_origin}"
            for i in range(len(ty_args)):
                cur_ty_arg = ty_args[i]
                if cur_ty_arg is Self:
                    assert self_type is not None, "Self type must be provided when parsing Self type"
                    ty_args[i] = self_type.raw_type
            return AnnotatedType(ty_origin, ty_args, ann_meta, is_optional,
                                 is_undefined, raw_type)
        else:
            ty_args = get_args(ann_type)
            if ty_origin is tuple or ty_origin is Tuple:
                # tuple type, we need to check if it is homogeneous
                if len(ty_args) == 2 and ty_args[1] is Ellipsis:
                    # Tuple[T, ...] is a homogeneous tuple type
                    ty_args = [ty_args[0]]
                    return AnnotatedType(ty_origin, ty_args, ann_meta,
                                         raw_type=raw_type,
                                         is_homogeneous=True)
            elif ty_origin is Callable:
                # flat args
                assert len(ty_args) == 2
                if isinstance(ty_args[0], Sequence):
                    ty_args = list(ty_args[0]) + [ty_args[1]]
                for arg in ty_args:
                    assert not isinstance(arg, (ParamSpec, ParamSpecArgs))
            # assert inspect.isclass(
            #     ty_origin), f"origin type must be a class, but get {ty_origin}"
            return AnnotatedType(ty_origin, list(ty_args), ann_meta, raw_type=raw_type)
    return AnnotatedType(ann_type, [], ann_meta, raw_type=raw_type)


def child_type_generator(t: type):
    yield t
    args = get_args(t)
    if is_annotated(t):
        # avoid yield meta object in annotated
        yield from child_dataclass_type_generator(args[0])
    else:
        for arg in args:
            yield from child_dataclass_type_generator(arg)


def child_type_generator_with_dataclass(t: type):
    yield t
    if dataclasses.is_dataclass(t):
        type_hints = get_type_hints_with_cache(t, include_extras=True)
        for field in dataclasses.fields(t):
            yield from child_type_generator_with_dataclass(type_hints[field.name])
    else:
        args = get_args(t)
        if is_annotated(t):
            yield from child_type_generator_with_dataclass(args[0])
        else:
            for arg in args:
                yield from child_type_generator_with_dataclass(arg)

def _child_dataclass_type_generator_recursive(t: type, visited: set[Any]) -> Generator[type[DataclassType], None, None]:
    if dataclasses.is_dataclass(t) and inspect.isclass(t):
        if t in visited:
            return
        visited.add(t)
        yield t
        type_hints = get_type_hints_with_cache(t, include_extras=True)
        for field in dataclasses.fields(t):
            yield from _child_dataclass_type_generator_recursive(type_hints[field.name], visited)
    else:
        args = get_args(t)
        if is_annotated(t):
            yield from _child_dataclass_type_generator_recursive(args[0], visited)
        else:
            for arg in args:
                yield from _child_dataclass_type_generator_recursive(arg, visited)



def child_dataclass_type_generator(t: type) -> Generator[type[DataclassType], None, None]:
    visited = set()
    yield from _child_dataclass_type_generator_recursive(t, visited)

def dataclass_flatten_fields_generator(t: type) -> Generator[tuple[dataclasses.Field, str, Any, Any], None, None]:
    visited = set()
    yield from _dataclass_flatten_fields_generator_recursive(t, "", visited)

def _dataclass_flatten_fields_generator_recursive(t: type, qname: str, visited: set[Any]) -> Generator[tuple[dataclasses.Field, str, Any, Any], None, None]:
    assert dataclasses.is_dataclass(t) and inspect.isclass(t)
    type_hints = get_type_hints_with_cache(t, include_extras=True)
    for field in dataclasses.fields(t):
        field_type = type_hints[field.name]
        field_annometa = None
        qname_cur = f"{qname}.{field.name}" if qname else field.name
        if is_annotated(t):
            args = get_args(t)
            field_type = args[0]
            field_annometa = t.__metadata__
            # field_annometa = 
        yield field, qname_cur, field_type, field_annometa
        if dataclasses.is_dataclass(field_type) and inspect.isclass(field_type):
            yield from _dataclass_flatten_fields_generator_recursive(field_type, qname_cur, visited)


def parse_annotated_function(
    func: Callable,
    is_dynamic_class: bool = False
) -> Tuple[List[AnnotatedArg], Optional[AnnotatedReturn]]:
    if compat.Python3_10AndLater:
        annos = get_type_hints(func, include_extras=True)
    else:
        annos = get_type_hints(func,
                               include_extras=True,
                               globalns={} if is_dynamic_class else None)

    specs = inspect.signature(func)
    name_to_parameter = {p.name: p for p in specs.parameters.values()}
    # print(name_to_parameter, annos.keys())
    anno_args: List[AnnotatedArg] = []
    anno_args_map: dict[str, AnnotatedArg] = {}
    return_anno: Optional[AnnotatedReturn] = None
    for name, param in name_to_parameter.items():
        anno_args.append(AnnotatedArg(name, param, Any))
        anno_args_map[name] = anno_args[-1]
    for name, anno in annos.items():
        if name == "return":
            anno, annotated_metas = extract_annotated_type_and_meta(anno)
            return_anno = AnnotatedReturn(anno, annotated_metas)
        else:
            param = name_to_parameter[name]
            anno, annotated_metas = extract_annotated_type_and_meta(anno)

            # arg_anno = AnnotatedArg(name, param, anno, annotated_metas)
            anno_args_map[name].type = anno
            anno_args_map[name].annometa = annotated_metas
    # for name, param in name_to_parameter.items():
    #     if name not in annos and param.kind in (
    #             inspect.Parameter.POSITIONAL_ONLY,
    #             inspect.Parameter.POSITIONAL_OR_KEYWORD):
    #         anno_args.append(AnnotatedArg(name, param, Any))
    return anno_args, return_anno


def annotated_function_to_dataclass(func: Callable,
                                    is_dynamic_class: bool = False):
    if compat.Python3_10AndLater:
        annos = get_type_hints(func, include_extras=True)
    else:
        annos = get_type_hints(func,
                               include_extras=True,
                               globalns={} if is_dynamic_class else None)
    specs = inspect.signature(func)
    name_to_parameter = {p.name: p for p in specs.parameters.values()}
    fields: List[Tuple[str, Any, Field]] = []
    for name, anno in annos.items():
        param = name_to_parameter[name]
        assert param.default is not inspect.Parameter.empty, "annotated function arg must have default value"
        fields.append((name, anno, field(default=param.default)))
    return make_dataclass(func.__name__, fields)


@dataclasses.dataclass
class AnnotatedFieldMeta:
    name: str
    annotype: AnnotatedType
    field_id: int
    field: dataclasses.Field


def _recursive_get_field_meta_dict(annotype: AnnotatedType, field_meta_dict: dict[int, AnnotatedFieldMeta]):
    if annotype.child_types:
        # generic dataclass
        type_hints = resolve_type_hints(annotype.origin_type[tuple(annotype.child_types)])
    else:
        type_hints = resolve_type_hints(annotype.origin_type)

    for field in dataclasses.fields(annotype.origin_type):
        field_type = type_hints[field.name]
        field_annotype = parse_type_may_optional_undefined(field_type)
        field_meta = AnnotatedFieldMeta(field.name, field_annotype, id(field), field)
        if id(field) not in field_meta_dict:
            field_meta_dict[id(field)] = field_meta
        else:
            # avoid nested check
            continue
        if field_annotype.is_dataclass_type():
            _recursive_get_field_meta_dict(field_annotype, field_meta_dict)
        else:
            for t in child_dataclass_type_generator(field_type):
                _recursive_get_field_meta_dict(parse_type_may_optional_undefined(t), field_meta_dict)
    return 

def get_dataclass_field_meta_dict(model_type: type[T_dataclass]) -> dict[int, AnnotatedFieldMeta]:
    # TODO: currently model_type must not be generic, you must remove all generic args via inherit.
    field_meta_dict: dict[int, AnnotatedFieldMeta] = {}
    _recursive_get_field_meta_dict(parse_type_may_optional_undefined(model_type), field_meta_dict)
    new_field_meta_dict: dict[int, AnnotatedFieldMeta] = {**field_meta_dict}
    return new_field_meta_dict

def unparse_type_expr(expr: Any, get_type_str: Callable[[Any], str] = get_qualname_of_type) -> str:
    if isinstance(expr, list):
        return "[" + ", ".join([unparse_type_expr(e) for e in expr]) + "]"
    if expr is Ellipsis:
        return "..."
    module = expr.__module__
    # TODO support TypeVar, ParamSpec, etc.
    is_typing = module == "typing" or module == "typing_extensions"
    if (is_typing or (module == "builtins")):
        origin = get_origin(expr)
        if origin is None:
            # should be builtin
            if module == "builtins":
                return get_qualname_of_type(expr)
            else:
                return get_type_str(expr)
        if origin == collections.abc.Callable:
            origin = Callable
        if origin == Literal:
            origin_str = get_type_str(Literal)
            args = get_args(expr)
            arg_strs = [repr(arg) for arg in args]
            return f"{origin_str}[{', '.join(arg_strs)}]"
        if is_annotated(expr):
            # don't support unparse annotated.
            return unparse_type_expr(get_args(expr)[0], get_type_str)
        if module == "builtins":
            origin_str = get_qualname_of_type(expr)
        else:
            origin_str = get_type_str(expr)

        args = get_args(expr)
        arg_strs = [unparse_type_expr(arg, get_type_str) for arg in args]
        return f"{origin_str}[{', '.join(arg_strs)}]"
    else:
        return get_type_str(expr)

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
    print(issubclass(WTF2, dict))
    print(dir(WTF2))
    print(parse_annotated_function(a.add))
    print(parse_annotated_function(a.add_stc))
    print(is_optional(Optional[int]))
    print(is_async_gen(AsyncGenerator[int, None]))
    print(is_not_required(NotRequired[int]))  # type: ignore
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
        print(at, at.is_list_type(), at.is_sequence_type(),
              at.is_mapping_type(), at.is_dict_type())

def _main_test():
    print(parse_type_may_optional_undefined(tuple[int, ...]))
    print(unparse_type_expr(Callable[[Union[list[int], str, Literal["wtf"]]], dict[str, int]]))
if __name__ == "__main__":
    _main_test()
