import inspect 
from typing import Any, Dict, List, Optional

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
        # inspect.getsourcelines need to read file, so .__code__.co_firstlineno
        # is greatly faster than it.
        # res.sort(key=lambda x: inspect.getsourcelines(x[1])[1])
        res.sort(key=lambda x: x[1].__code__.co_firstlineno)
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
    res.sort(key=lambda x: x[1].__code__.co_firstlineno)
    return res

def get_all_members_by_type(obj_type: Any):
    """this function return all member functions
    """
    this_cls = obj_type
    child_methods = inspect.getmembers(this_cls, predicate=inspect.isfunction)
    return child_methods


def get_members(obj: Any, no_parent: bool = True):
    return get_members_by_type( type(obj), no_parent)
