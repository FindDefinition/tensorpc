from types import ModuleType
from typing import Any, Callable, ClassVar, Optional, Type, Union, TypeVar, cast

from tensorpc.core.annolib import DataclassType
import tensorpc.core.dataclass_dispatch as dataclasses
import inspect
import dataclasses

from tensorpc.core.pfl.constants import PFL_BUILTIN_PROXY_INIT_FN


@dataclasses.dataclass
class StdRegistryItem:
    dcls: Union[Type[DataclassType], Callable]
    mapped_name: str
    mapped: Optional[Union[ModuleType, Type, Callable]] = None
    # if backend is None, it means all backends share this item
    backend: Optional[str] = "js"
    is_temp: bool = False
    is_func: bool = False
    is_builtin: bool = False
    # if a dataclass can't be created from auto-generated dataclass __init__ in pfl, we must set this.
    disable_dcls_ctor: bool = False
    namespace_aliases: dict[str, Type[DataclassType]] = dataclasses.field(
        default_factory=dict)
    # used to register some system std function used in decorator (disable type check)
    _internal_disable_type_check: bool = False


T = TypeVar("T")


class StdRegistry:

    def __init__(self):
        self.global_dict: dict[tuple[str, Optional[str]], StdRegistryItem] = {}
        self._mapped_backend_to_item: dict[tuple[Any, Optional[str]], StdRegistryItem] = {}
        self._type_backend_to_item: dict[tuple[Any, Optional[str]], StdRegistryItem] = {}

    def register(
        self,
        func=None,
        *,
        mapped_name: Optional[str] = None,
        mapped: Optional[Union[ModuleType, Type, Callable]] = None,
        backend: Optional[str] = "js",
        disable_dcls_ctor: bool = False,
        _internal_disable_type_check: bool = False,
    ):

        def wrapper(func: T) -> T:
            assert inspect.isclass(func) or inspect.isfunction(
                func
            ), "register_compute_node should be used on class or function"
            namespace_aliases: dict[str, Type[DataclassType]] = {}
            if inspect.isclass(func):
                assert dataclasses.is_dataclass(
                    func
                ), "std object must be a dataclass if it isn't a global function"
                # iterate class vars of this dataclasses since we use it as namespace alias.
                # the value of classvar must be registered dataclass.
                for attr, cls in inspect.get_annotations(func).items():
                    if cls is ClassVar:
                        value = getattr(func, attr)
                        if inspect.isclass(
                            value
                        ):
                            assert inspect.isclass(
                                value
                            ) and dataclasses.is_dataclass(
                                value
                            ), "classvar (used as namespace alias) must be a dataclass class"
                            registered_item = self.get_item_by_dcls(value)
                            if registered_item is None:
                                raise ValueError(
                                    f"ClassVar {attr} of {func.__name__} must be registered as a std object."
                                )
                            namespace_aliases[attr] = cast(Type[DataclassType],
                                                        registered_item.dcls)
            assert mapped_name is not None
            key_ = mapped_name
            assert (
                key_, backend
            ) not in self.global_dict, f"Duplicate registration for {key_} with backend {backend}"
            item = StdRegistryItem(
                dcls=func,
                mapped_name=mapped_name,
                mapped=mapped,
                backend=backend,
                is_func=inspect.isfunction(func),
                disable_dcls_ctor=disable_dcls_ctor,
                namespace_aliases=namespace_aliases,
                _internal_disable_type_check=_internal_disable_type_check,
            )

            if mapped is not None:
                assert (mapped, backend) not in self._mapped_backend_to_item, f"Duplicate mapped type {mapped} for {key_} with backend {backend}"
                self._mapped_backend_to_item[(mapped, backend)] = item
                self._type_backend_to_item[(mapped, backend)] = item
            self.global_dict[(key_, backend)] = item
            self._type_backend_to_item[(func, backend)] = item
            return cast(T, func)

        if func is None:
            return wrapper
        else:
            return wrapper(func)

    def register_builtin_proxy(
        self,
        func=None,
        *,
        mapped_name: Optional[str] = None,
        mapped: Optional[Type] = None,
        backend: Optional[str] = "js",
        _internal_disable_type_check: bool = False,
    ):

        def wrapper(func: T) -> T:
            assert inspect.isclass(func) and dataclasses.is_dataclass(func), "builtin only support class (dataclass)"
            init_fn = inspect.getattr_static(func, PFL_BUILTIN_PROXY_INIT_FN, None)
            assert init_fn is not None and isinstance(init_fn, staticmethod), "your builtin proxy class must have a __pfl_proxy_init__ staticmethod."
            assert mapped_name is not None
            key_ = mapped_name
            assert (
                key_, backend
            ) not in self.global_dict, f"Duplicate registration for {key_} with backend {backend}"

            self.global_dict[(key_, backend)] = StdRegistryItem(
                dcls=func,
                mapped_name=mapped_name,
                mapped=mapped,
                backend=backend,
                is_func=False,
                is_builtin=True,
                _internal_disable_type_check=_internal_disable_type_check,
            )

            return cast(T, func)

        if func is None:
            return wrapper
        else:
            return wrapper(func)

    def __contains__(self, key: tuple[str, str]):
        return key in self.global_dict

    def __getitem__(self, key: tuple[str, str]):
        return self.global_dict[key]

    def items(self):
        yield from self.global_dict.items()

    def get_item_by_dcls(
        self,
        dcls: Any,
        backend: str = "js",
        external: Optional[dict[tuple[str, Optional[str]],
                                StdRegistryItem]] = None
    ) -> Optional[StdRegistryItem]:
        if external is not None:
            check_items = {**self.global_dict, **external}
        else:
            check_items = self.global_dict
        for _, item in check_items.items():
            if item.backend is not None and item.backend != backend:
                continue
            if item.dcls is dcls:
                return item
            if item.mapped is not None and item.mapped is dcls:
                return item
        return None

    def get_dcls_item_by_mapped_type(
        self,
        mapped_type: Any,
        backend: str = "js",
        external: Optional[dict[tuple[str, Optional[str]],
                                StdRegistryItem]] = None,
        _builtin_only: bool = False
    ) -> Optional[StdRegistryItem]:
        if external is not None:
            check_items = {**self.global_dict, **external}
        else:
            return self._mapped_backend_to_item.get((mapped_type, backend), None)
            # check_items = self.global_dict
        for _, item in check_items.items():
            if _builtin_only and not item.is_builtin:
                continue
            if item.backend is not None and item.backend != backend:
                continue
            if item.mapped is not None and item.mapped is mapped_type:
                return item
        return None

    def get_proxy_dcls_by_mapped_type(
        self,
        mapped_type: Any,
        backend: str = "js",
        external: Optional[dict[tuple[str, Optional[str]],
                                StdRegistryItem]] = None,
    ) -> Optional[StdRegistryItem]:
        return self.get_dcls_item_by_mapped_type(
            mapped_type,
            backend=backend,
            external=external,
            _builtin_only=True,
        )

STD_REGISTRY = StdRegistry()


def register_pfl_std(
    func=None,
    *,
    mapped_name: Optional[str] = None,
    mapped: Optional[Union[ModuleType, Type, Callable]] = None,
    backend: Optional[str] = "js",
    disable_dcls_ctor: bool = False,
    _internal_disable_type_check: bool = False,
):
    return STD_REGISTRY.register(
        func,
        mapped_name=mapped_name,
        mapped=mapped,
        backend=backend,
        disable_dcls_ctor=disable_dcls_ctor,
        _internal_disable_type_check=_internal_disable_type_check)

def register_pfl_builtin_proxy(
    func=None,
    *,
    mapped_name: Optional[str] = None,
    mapped: Optional[Type] = None,
    backend: Optional[str] = "js",
    _internal_disable_type_check: bool = False,
):
    return STD_REGISTRY.register_builtin_proxy(
        func,
        mapped_name=mapped_name,
        mapped=mapped,
        backend=backend,
        _internal_disable_type_check=_internal_disable_type_check)

# compile-time system functions
@register_pfl_std(mapped_name="compiler_print_type", backend=None)
def compiler_print_type(x: Any) -> Any:
    raise NotImplementedError("can't be called directly.")


@register_pfl_std(mapped_name="compiler_print_metadata", backend=None)
def compiler_print_metadata(x: Any) -> Any:
    raise NotImplementedError("can't be called directly.")

@register_pfl_std(mapped_name="compiler_isinstance", backend=None, mapped=isinstance)
def compiler_isinstance(x: Any, cls: Any) -> bool:
    raise NotImplementedError("can't be called directly.")

@register_pfl_std(mapped_name="compiler_remove_optional", backend=None)
def compiler_remove_optional(x: Any) -> Any:
    raise NotImplementedError("can't be called directly.")


ALL_COMPILE_TIME_FUNCS = {
    compiler_print_type,
    compiler_print_metadata,
    compiler_isinstance,
    compiler_remove_optional,
}
