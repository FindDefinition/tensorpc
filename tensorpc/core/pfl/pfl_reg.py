
from types import ModuleType
from typing import Any, Callable, Optional, Type, Union, TypeVar, cast

from tensorpc.core.annolib import DataclassType, T_dataclass, Undefined
import tensorpc.core.dataclass_dispatch as dataclasses
import inspect 
import dataclasses

@dataclasses.dataclass
class StdRegistryItem:
    dcls: Type[DataclassType]
    mapped_name: str
    mapped: Optional[Union[ModuleType, Type]] = None
    backend: str = "js"
    is_temp: bool = False

T = TypeVar("T")

class StdRegistry:

    def __init__(self):
        self.global_dict: dict[str, StdRegistryItem] = {}

    def register(
            self,
            func=None,
            *,
            mapped_name: Optional[str] = None,
            mapped: Optional[Union[ModuleType, Type]] = None,
            backend: str = "js",
            ):

        def wrapper(func: Type[T]) -> Type[T]:
            assert inspect.isclass(func) or inspect.isfunction(
                func
            ), "register_compute_node should be used on class or function"
            assert inspect.isclass(func) and dataclasses.is_dataclass(func), "std object must be a dataclass"
            assert mapped_name is not None 
            key_ = mapped_name
            self.global_dict[key_] = StdRegistryItem(
                dcls=func,
                mapped_name=mapped_name,
                mapped=mapped,
                backend=backend,
            )
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

    def get_item_by_dcls(self, dcls: Any, backend: str = "js") -> Optional[StdRegistryItem]:
        for _, item in self.global_dict.items():
            if item.backend != backend:
                continue 
            if item.dcls is dcls:
                return item
            if item.mapped is not None and item.mapped is dcls:
                return item
        return None

    def get_dcls_item_by_mapped_type(self, mapped_type: Any, backend: str = "js") -> Optional[StdRegistryItem]:
        for _, item in self.global_dict.items():
            if item.backend != backend:
                continue 
            if item.mapped is not None and item.mapped is mapped_type:
                return item
        return None

STD_REGISTRY = StdRegistry()

def register_pfl_std(
        func=None,
        *,
        mapped_name: Optional[str] = None,
        mapped: Optional[Union[ModuleType, Type]] = None,
        backend: str = "js",
        ):
    return STD_REGISTRY.register(func, mapped_name=mapped_name, mapped=mapped, backend=backend)

