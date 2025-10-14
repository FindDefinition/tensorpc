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
    # only available in dcls.
    partial_constexpr_fields: Optional[set[str]] = None
    # wben some args is constexpr in constructor,
    # we can create a partial-constexpr dataclass.
    constexpr_infer: Optional[Callable[..., Any]] = None


T = TypeVar("T")


class StdRegistry:

    def __init__(self):
        self.global_dict: dict[tuple[str, Optional[str]], StdRegistryItem] = {}
        self._mapped_backend_to_item: dict[tuple[Any, Optional[str]], StdRegistryItem] = {}
        self._type_backend_to_item: dict[tuple[Any, Optional[str]], StdRegistryItem] = {}
        self._type_to_item: dict[Any, StdRegistryItem] = {}

    def register(
        self,
        func=None,
        *,
        mapped_name: Optional[str] = None,
        mapped: Optional[Union[ModuleType, Type, Callable]] = None,
        backend: Optional[str] = "js",
        backend_cfg: Optional[dict[str, tuple[str, Optional[Union[ModuleType, Type, Callable]]]]] = None,
        disable_dcls_ctor: bool = False,
        partial_constexpr_fields: Optional[set[str]] = None,
        constexpr_infer: Optional[Callable[..., Any]] = None,
        _internal_disable_type_check: bool = False,
        _is_register_builtin_proxy: bool = False,
    ):

        def wrapper(func: T) -> T:
                
            backends: list[Optional[str]] = []
            mapped_names: list[str] = []
            mappeds: dict[Optional[str], Union[ModuleType, Type, Callable]] = {}
            if backend_cfg is not None:
                for backend_, (mapped_name_, mapped_) in backend_cfg.items():
                    backends.append(backend_)
                    mapped_names.append(mapped_name_)
                    if mapped_ is not None:
                        mappeds[backend_] = mapped_
            else:
                assert mapped_name is not None
                backends = [backend]
                mapped_names = [mapped_name]
                if mapped is not None:
                    mappeds = {backend: mapped}
                else:
                    mappeds = {}
            namespace_aliases: dict[str, Type[DataclassType]] = {}
            if _is_register_builtin_proxy:
                assert inspect.isclass(func) and dataclasses.is_dataclass(func), "builtin only support class (dataclass)"
                init_fn = inspect.getattr_static(func, PFL_BUILTIN_PROXY_INIT_FN, None)
                assert init_fn is not None and isinstance(init_fn, staticmethod), "your builtin proxy class must have a __pfl_proxy_init__ staticmethod."
            else:
                assert inspect.isclass(func) or inspect.isfunction(
                    func
                ), "register_compute_node should be used on class or function"
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
                else:
                    assert partial_constexpr_fields is None
            for backend_, mapped_name_ in zip(backends, mapped_names):
                if not mappeds:
                    mapped_ = None 
                else:
                    mapped_ = mappeds[backend_]
                assert mapped_name_ is not None
                key_ = mapped_name_
                assert (
                    key_, backend_
                ) not in self.global_dict, f"Duplicate registration for {key_} with backend {backend_}"
                item = StdRegistryItem(
                    dcls=func,
                    mapped_name=mapped_name_,
                    mapped=mapped_,
                    backend=backend_,
                    is_func=inspect.isfunction(func),
                    disable_dcls_ctor=disable_dcls_ctor,
                    namespace_aliases=namespace_aliases,
                    partial_constexpr_fields=partial_constexpr_fields,
                    constexpr_infer=constexpr_infer,
                    is_builtin=_is_register_builtin_proxy,
                    _internal_disable_type_check=_internal_disable_type_check,
                )

                if mapped is not None:
                    assert (mapped_, backend_) not in self._mapped_backend_to_item, f"Duplicate mapped type {mapped_} for {key_} with backend {backend_}"
                    self._mapped_backend_to_item[(mapped_, backend_)] = item
                    self._type_backend_to_item[(mapped_, backend_)] = item
                self.global_dict[(key_, backend_)] = item
                self._type_backend_to_item[(func, backend_)] = item
                self._type_to_item[func] = item
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

    def get_item_by_key(
        self,
        type_or_fn: Any,
    ) -> Optional[StdRegistryItem]:
        return self._type_to_item.get(type_or_fn, None)

    def get_item_by_dcls(
        self,
        dcls: Any,
        backend: str = "js",
    ) -> Optional[StdRegistryItem]:
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
        _builtin_only: bool = False
    ) -> Optional[StdRegistryItem]:
        check_items = self.global_dict
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
    ) -> Optional[StdRegistryItem]:
        return self.get_dcls_item_by_mapped_type(
            mapped_type,
            backend=backend,
            _builtin_only=True,
        )

STD_REGISTRY = StdRegistry()


def register_pfl_std(
    func=None,
    *,
    mapped_name: Optional[str] = None,
    mapped: Optional[Union[ModuleType, Type, Callable]] = None,
    backend: Optional[str] = "js",
    backend_cfg: Optional[dict[str, tuple[str, Optional[Union[ModuleType, Type, Callable]]]]] = None,
    disable_dcls_ctor: bool = False,
    partial_constexpr_fields: Optional[set[str]] = None,
    constexpr_infer: Optional[Callable[..., Any]] = None,
    _internal_disable_type_check: bool = False,
):
    return STD_REGISTRY.register(
        func,
        mapped_name=mapped_name,
        mapped=mapped,
        backend=backend,
        backend_cfg=backend_cfg,
        disable_dcls_ctor=disable_dcls_ctor,
        partial_constexpr_fields=partial_constexpr_fields,
        constexpr_infer=constexpr_infer,
        _internal_disable_type_check=_internal_disable_type_check)

def register_pfl_builtin_proxy(
    func=None,
    *,
    mapped_name: Optional[str] = None,
    mapped: Optional[Union[ModuleType, Type, Callable]] = None,
    backend: Optional[str] = "js",
    backend_cfg: Optional[dict[str, tuple[str, Optional[Union[ModuleType, Type, Callable]]]]] = None,
    _internal_disable_type_check: bool = False,
):
    return STD_REGISTRY.register(
        func,
        mapped_name=mapped_name,
        mapped=mapped,
        backend=backend,
        backend_cfg=backend_cfg,
        _internal_disable_type_check=_internal_disable_type_check,
        _is_register_builtin_proxy=True)

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

@register_pfl_std(mapped_name="compiler_cast", backend=None, mapped=cast)
def compiler_cast(x: Any, cls: Any) -> Any:
    raise NotImplementedError("can't be called directly.")

ALL_COMPILE_TIME_FUNCS = {
    compiler_print_type,
    compiler_print_metadata,
    compiler_isinstance,
    compiler_remove_optional,
    compiler_cast,
}
