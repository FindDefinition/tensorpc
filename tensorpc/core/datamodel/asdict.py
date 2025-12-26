from functools import partial
from typing import Any, Callable, Dict, Generic, Hashable, Optional, Type, TypeVar, Union, cast
from tensorpc.core.core_io import JsonSpecialData
from tensorpc.core.tree_id import UniqueTreeId, UniqueTreeIdForTree

from tensorpc.core.annolib import Undefined, BackendOnlyProp, undefined
import tensorpc.core.dataclass_dispatch as dataclasses
import copy 

def undefined_dict_factory(x: list[tuple[str, Any]]):
    res: Dict[str, Any] = {}
    for k, v in x:
        if isinstance(v, UniqueTreeId):
            res[k] = v.uid_encoded
        elif not isinstance(v, (Undefined, BackendOnlyProp)):
            res[k] = v
    return res

def undefined_dict_factory_with_field(x: list[tuple[str, Any, Any]]):
    res: Dict[str, Any] = {}
    for k, v, f in x:
        if isinstance(v, UniqueTreeId):
            res[k] = v.uid_encoded
        elif not isinstance(v, (Undefined, BackendOnlyProp)):
            res[k] = v
    return res

def undefined_dict_factory_with_exclude(x: list[tuple[str, Any, Any]], exclude_field_ids: set[int]):
    res: Dict[str, Any] = {}
    for k, v, f in x:
        if id(f) in exclude_field_ids:
            continue
        if isinstance(v, UniqueTreeId):
            res[k] = v.uid_encoded
        elif not isinstance(v, (Undefined, BackendOnlyProp)):
            res[k] = v
    return res

@dataclasses.dataclass
class _DataclassSer:
    obj: Any


def as_dict_no_undefined_v1(obj: Any):
    return dataclasses.asdict(_DataclassSer(obj),
                              dict_factory=undefined_dict_factory)["obj"]


def as_dict_no_undefined(obj: Any):
    return _asdict_inner(_DataclassSer(obj),
                              dict_factory=undefined_dict_factory)["obj"]

def asdict_field_only(obj,
                      *,
                      dict_factory: Callable[[list[tuple[str, Any]]],
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
                              dict_factory: Callable[[list[tuple[str, Any]]],
                                                     Dict[str, Any]] = dict):
    """same as dataclasses.asdict except that this function
    won't recurse into nested container.
    """
    if not dataclasses.is_dataclass(obj):
        raise TypeError("asdict() should be called on dataclass instances")
    return _asdict_flatten_field_only(obj, dict_factory)


def asdict_flatten_field_only_no_undefined(obj):
    """same as dataclasses.asdict except that this function
    won't recurse into nested container.
    """
    if not dataclasses.is_dataclass(obj):
        raise TypeError("asdict() should be called on dataclass instances")
    return _asdict_flatten_field_only(obj, undefined_dict_factory)


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
                       dict_factory: Callable[[list[tuple[str, Any]]],
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

def _default_dict_factory_with_field(val: list[tuple[str, Any, Any]]):
    return dict((x[0], x[1]) for x in val)

def asdict_no_deepcopy_with_field(obj,
                       *,
                       dict_factory_with_field: Callable[[list[tuple[str, Any, Any]]],
                                              Dict[str, Any]] = _default_dict_factory_with_field,
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
    return _asdict_inner_with_field(obj, dict_factory_with_field, obj_factory)

def _asdict_inner(obj, dict_factory, obj_factory=None) -> Any:
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
    elif isinstance(obj, JsonSpecialData):
        return obj.replace_data(_asdict_inner(obj.data, dict_factory, obj_factory))
    else:
        if obj_factory is not None:
            obj = obj_factory(obj)
        return obj

def _asdict_inner_with_field(obj, dict_factory, obj_factory=None) -> Any:
    if dataclasses.is_dataclass(obj):
        result = []
        for f in dataclasses.fields(obj):
            value = _asdict_inner_with_field(getattr(obj, f.name), dict_factory,
                                  obj_factory)
            result.append((f.name, value, f))
        return dict_factory(result)
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        return type(obj)(
            *[_asdict_inner_with_field(v, dict_factory, obj_factory) for v in obj])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(_asdict_inner_with_field(v, dict_factory, obj_factory)
                         for v in obj)
    elif isinstance(obj, dict):
        return type(obj)((_asdict_inner_with_field(k, dict_factory, obj_factory),
                          _asdict_inner_with_field(v, dict_factory, obj_factory))
                         for k, v in obj.items())
    elif isinstance(obj, JsonSpecialData):
        return obj.replace_data(_asdict_inner_with_field(obj.data, dict_factory, obj_factory))
    else:
        if obj_factory is not None:
            obj = obj_factory(obj)
        return obj


def as_dict_no_undefined_no_deepcopy(obj,
                                     *,
                                     obj_factory: Optional[Callable[[Any], Any]] = None):
    if not dataclasses.is_dataclass(obj):
        raise TypeError("asdict() should be called on dataclass instances")
    res = asdict_no_deepcopy(obj,
                              dict_factory=undefined_dict_factory,
                              obj_factory=obj_factory)
    assert isinstance(res, dict)
    return res

@dataclasses.dataclass
class DataClassWithUndefined:

    def get_dict_and_undefined(
            self,
            state: Dict[str, Any],
            dict_factory: Callable[[list[tuple[str, Any]]],
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
            field_name = field.name
            val = ref_dict[field_name]
            if isinstance(val, Undefined):
                res_und.append(field_name)
            else:
                res[field_name] = val
        return res, res_und

    def get_dict(self,
                 dict_factory: Callable[[list[tuple[str, Any]]],
                                        Dict[str,
                                             Any]] = undefined_dict_factory,
                 obj_factory: Optional[Callable[[Any], Any]] = None):
        this_type = type(self)
        res = {}
        ref_dict = asdict_no_deepcopy(self,
                                      dict_factory=dict_factory,
                                      obj_factory=obj_factory)
        assert isinstance(ref_dict, dict)
        for field in dataclasses.fields(this_type):
            field_name = field.name
            if field.name not in ref_dict:
                val = undefined
            else:
                val = ref_dict[field_name]
            res[field_name] = val
        return res

    def get_dict_with_fields(self,
                 dict_factory: Callable[[list[tuple[str, Any, Any]]],
                                        Dict[str,
                                             Any]] = undefined_dict_factory_with_field,
                 obj_factory: Optional[Callable[[Any], Any]] = None):
        this_type = type(self)
        res = {}
        ref_dict = asdict_no_deepcopy_with_field(self,
                                      dict_factory_with_field=dict_factory,
                                      obj_factory=obj_factory)
        assert isinstance(ref_dict, dict)
        for field in dataclasses.fields(this_type):
            field_name = field.name
            if field.name not in ref_dict:
                val = undefined
            else:
                val = ref_dict[field_name]
            res[field_name] = val
        return res

    def get_flatten_dict(
        self,
        dict_factory: Callable[[list[tuple[str, Any]]],
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


def as_dict_no_undefined_with_exclude(obj: Any, exclude_field_ids: set[int]):
    dict_fact = partial(undefined_dict_factory_with_exclude, exclude_field_ids=exclude_field_ids)
    res = asdict_no_deepcopy_with_field(_DataclassSer(obj),
                              dict_factory_with_field=dict_fact)
    return cast(dict, res)["obj"]
