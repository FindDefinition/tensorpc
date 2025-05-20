
from typing import Any, Callable, Optional, Type, Union, TypeVar, cast

from tensorpc.core.annolib import DataclassType, T_dataclass
import tensorpc.core.dataclass_dispatch as dataclasses
import inspect 

T = TypeVar("T")

class StdRegistry:

    def __init__(self):
        self.global_dict: dict[str, Type[DataclassType]] = {}

    def register(
            self,
            func=None,
            *,
            js_name: Optional[str] = None):

        def wrapper(func: Type[T]) -> Type[T]:
            assert inspect.isclass(func) or inspect.isfunction(
                func
            ), "register_compute_node should be used on class or function"
            assert inspect.isclass(func) and dataclasses.is_dataclass(func), "std object must be a dataclass"
            assert js_name is not None 
            key_ = js_name
            self.global_dict[key_] = func
            return cast(Type[T], func)

        if func is None:
            return wrapper
        else:
            return wrapper(func)

    def __contains__(self, key: str):
        return key in self.global_dict

    def __getitem__(self, key: str):
        return self.global_dict[key]

    def items(self):
        yield from self.global_dict.items()

    def has_dcls_type(self, dcls: Type[T_dataclass]) -> bool:
        for _, dcls_ in self.global_dict.items():
            if dcls_ is dcls:
                return True
        return False

STD_REGISTRY = StdRegistry()

def register_df_std(
        func=None,
        *,
        js_name: Optional[str] = None,
        ):
    return STD_REGISTRY.register(func, js_name=js_name)

