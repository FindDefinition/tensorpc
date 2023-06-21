import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Type, Union 

import types


def isclassmethod(method):
    # https://stackoverflow.com/questions/19227724/check-if-a-function-uses-classmethod
    bound_to = getattr(method, '__self__', None)
    if not isinstance(bound_to, type):
        # must be bound to a class
        return False
    name = method.__name__
    for cls in bound_to.__mro__:
        descriptor = vars(cls).get(name)
        if descriptor is not None:
            return isinstance(descriptor, classmethod)
    return False


def isproperty(method):
    return isinstance(method, property)


def isstaticmethod(cls, method_name: str):
    method_static = inspect.getattr_static(cls, method_name)
    return isinstance(method_static, staticmethod)


def get_members_by_type(obj_type: Any, no_parent: bool = True):
    """this function return member functions that keep def order.
    """
    this_cls = obj_type
    if not no_parent:
        res = inspect.getmembers(this_cls, inspect.isfunction)
        return res
    parents = inspect.getmro(this_cls)[1:]
    parents_methods = set()
    for parent in parents:
        members = inspect.getmembers(parent, predicate=inspect.isfunction)
        parents_methods.update(members)

    child_methods = set(
        inspect.getmembers(this_cls, predicate=inspect.isfunction))
    child_only_methods = child_methods - parents_methods
    res = list(child_only_methods)
    # res.sort(key=lambda x: inspect.getsourcelines(x[1])[1])
    # inspect.getsourcelines need to read file, so .__code__.co_firstlineno
    # is greatly faster than it.
    res.sort(key=lambda x: x[1].__code__.co_firstlineno)
    return res


def get_all_members_by_type(obj_type: Any):
    """this function return all member functions
    """
    this_cls = obj_type
    child_methods = inspect.getmembers(this_cls, predicate=inspect.isfunction)
    return child_methods


def get_members(obj: Any, no_parent: bool = True):
    return get_members_by_type(type(obj), no_parent)


def chehck_obj_is_pybind(obj):
    # TODO better way to check a type is a pybind11 type
    obj_type = type(obj)
    obj_type_dir = dir(obj_type)

    return "__dict__" not in obj_type_dir and "__weakref__" not in obj_type_dir


def get_obj_userdefined_properties(obj: Any) -> Set[str]:
    res: Set[str] = set()
    is_pybind = chehck_obj_is_pybind(obj)
    if is_pybind:
        # all properties in pybind object is property object
        # so we must check setter.
        # pybind object attrs defined by def_readwrite or def_readonly
        # will have setter with type "instancemethod"
        # otherwise is "builtin_function_or_method"
        # for common case.
        obj_type = type(obj)
        for key in dir(obj_type):
            if not key.startswith("__") and not key.endswith("__"):
                class_attr = getattr(obj_type, key)
                if isinstance(class_attr, property):
                    if class_attr.fget is not None and type(
                            class_attr.fget).__name__ != "instancemethod":
                        res.add(key)
    else:
        obj_type = type(obj)
        for key in dir(obj_type):
            if not key.startswith("__") and not key.endswith("__"):
                class_attr = getattr(obj_type, key)
                if isinstance(class_attr, property):
                    res.add(key)
    return res

def is_obj_builtin_or_module(v):
    if isinstance(v, types.ModuleType):
        return True 
    if inspect.isfunction(v) or inspect.ismethod(v) or inspect.isbuiltin(
            v):
        return True 
    return False 

def filter_local_vars(local_var: Mapping[str, Any]) -> Mapping[str, Any]:
    new_local_vars: Dict[str, Any] = {}
    for k, v in local_var.items():
        if not is_obj_builtin_or_module(v) and k != "__class__":
            new_local_vars[k] = v 

    return new_local_vars

def get_function_defined_type(func: Callable):
    func = inspect.unwrap(func)
    mod = inspect.getmodule(func)
    if mod is None:
        return None 
    if mod.__name__.startswith("tensorpc") and not mod.__name__.startswith("tensorpc.flow.sampleapp"):
        # ignore all tensorpc type
        return None
    func_qname = func.__qualname__
    func_qname_parts = func_qname.split(".")
    res: Union[Type, types.ModuleType] = mod
    cur_obj = mod.__dict__
    for part in func_qname_parts[:-1]:
        cur_obj = cur_obj[part]
        res = cur_obj
    return res
        