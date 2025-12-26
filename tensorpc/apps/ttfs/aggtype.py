import abc
import contextvars
from typing import (
    Generic,
    Iterable,
    Optional,
    Union,
    cast,
    dataclass_transform,
    TypeVar,
    ParamSpec,
    overload,
)
import torch
from typing_extensions import (
    Literal,
    Annotated,
    NotRequired,
    Protocol,
    get_origin,
    get_args,
    get_type_hints,
    TypeGuard,
    TypeIs,
    Self,
    ParamSpec,
    ParamSpecArgs,
)
from tensorpc.core.tree_id import UniqueTreeId
import triton.language as tl
import inspect
import typing
import triton
import dataclasses

from typing import Any, Callable
import typing
import triton
from triton.runtime.jit import constexpr_function as _constexpr_function_gluon
from triton.experimental import gluon
from triton.language.core import base_value, JITCallable, ir, _aggregate_type
from triton.experimental.gluon.language._core import builtin as gluon_builtin
from triton.language.core import builtin as triton_builtin
from triton.compiler.code_generator import BoundJITMethod
import ast 
if typing.TYPE_CHECKING:
    from _typeshed import DataclassInstance as StandardDataclass
    from triton.compiler import CompiledKernel

TRITON_VERSION = tuple(map(int, triton.__version__.split(".")[:2]))

if TRITON_VERSION <= (3, 5):
    # monkey fix for a triton bug
    def mangle(self):
        return "_T" + "_".join(ty.mangle() for ty in self.types) + "_T"

    triton.language.core.tuple_type.mangle = mangle
    del mangle

_T = TypeVar("_T")

TRITON_AGG_FLATTEN_META_FIELD = "__ttfs_triton_agg_flatten_meta__"
TRITON_AGG_META_FIELD = "__ttfs_triton_agg_meta__"


def constexpr_function(fn: _T) -> _T:
    def wrapper(fn_wrapped: _T) -> _T:
        return cast(_T, _constexpr_function_gluon(fn_wrapped))

    return wrapper(fn)


def lenient_issubclass(cls: Any, class_or_tuple: Any) -> bool:  # pragma: no cover
    return isinstance(cls, type) and issubclass(cls, class_or_tuple)


def is_annotated(ann_type: Any) -> TypeGuard[Annotated]:
    # https://github.com/pydantic/pydantic/blob/35144d05c22e2e38fe093c533ff3a05ce9a30116/pydantic/_internal/_typing_extra.py#L99C1-L104C1
    origin = get_origin(ann_type)
    return origin is not None and lenient_issubclass(origin, Annotated)


def extract_annotated_type_and_meta(ann_type: Any) -> tuple[Any, Optional[Any]]:
    if is_annotated(ann_type):
        annometa = ann_type.__metadata__
        ann_type = get_args(ann_type)[0]
        return ann_type, annometa
    return ann_type, None

class TritonAggFieldAccessor(abc.ABC):
    """only support leaf fields."""
    
    @abc.abstractmethod
    def get_custom_fields(self) -> list[tuple[str, bool]]: 
        """get all custom fields and is_constexpr flag.
        """
        ...

    @abc.abstractmethod
    def get_load_call_node(self, meta: "TritonAggFlatField", root_arg_name: str, fn_node: ast.Call) -> ast.Call: 
        ...

    @abc.abstractmethod
    def get_flatten_agg_to_kwarg_lines(self, meta: "TritonAggFlatField", root_arg_name: str, desc_path: str) -> list[str]: 
        ...


@dataclasses.dataclass
class FieldMeta:
    # is_kernel_arg: used in jitx (host side aggregate)
    # if False, this field will not be passed as kernel argument, must have default value.
    is_kernel_arg: bool = True
    _internal_is_constexpr: bool = False
    accessor: Optional[TritonAggFieldAccessor] = None

    @property 
    def is_constexpr(self) -> bool:
        return self._internal_is_constexpr

    @property 
    def is_kernel_argument(self) -> bool:
        return self.is_kernel_arg


def _get_annotated_triton_fields(cls_dcls):
    fields = dataclasses.fields(cls_dcls)
    field_types = get_type_hints(cls_dcls, include_extras=True)
    # remove type because it's defined in base_value
    field_types.pop("type")
    field_metas: dict[str, FieldMeta] = {}
    for field in fields:
        field_type, annometa = extract_annotated_type_and_meta(field_types[field.name])
        field_types[field.name] = field_type
        field_meta = FieldMeta()
        if field_type is tl.constexpr:
            field_meta._internal_is_constexpr = True
        elif annometa is not None:
            for meta in annometa:
                if meta is tl.constexpr:
                    field_meta._internal_is_constexpr = True
                elif isinstance(meta, FieldMeta):
                    if not meta.is_kernel_argument:
                        assert field.default is not dataclasses.MISSING or field.default_factory is not dataclasses.MISSING, \
                            f"Field '{field.name}' is marked as non-kernel-arg field, it must have a default value."
                    field_meta.is_kernel_arg = meta.is_kernel_arg
                    field_meta.accessor = meta.accessor
                elif inspect.isclass(meta) and issubclass(meta, FieldMeta):
                    raise ValueError("FieldMeta should be used as instance, not class.")
        field_metas[field.name] = field_meta
    return field_types, field_metas


def _get_annotated_field_types(cls_dcls):
    field_types, field_metas = _get_annotated_triton_fields(cls_dcls)
    constexpr_fields: set[str] = set()
    for name, meta in field_metas.items():
        if meta._internal_is_constexpr:
            constexpr_fields.add(name)
    return field_types, constexpr_fields


@dataclasses.dataclass
class TritonAggFlatField:
    path: str
    type: Any
    meta: FieldMeta
    parent_dcls: tuple[Any, ...] = ()
    parent_field_metas: tuple[FieldMeta, ...] = ()
    @property 
    def is_constexpr(self) -> bool:
        return self.meta._internal_is_constexpr

    def mangle_path(self, root_name: str, tail_field: str = "") -> str:
        # convert parts to "_ttfs_part0_part1_part2X[len(part0)]_[len(part1)]_[len(part2)]"
        parts = self.path.split(".")
        if tail_field != "":
            parts.append(tail_field)
        lengths = [str(len(p)) for p in parts]
        right = "_".join(lengths)
        return f"{root_name}_" + "_".join(parts) + f"X{right}"
        

def _get_agg_fields(dcls,
                    dict_factory,
                    parent_key: str = '',
                    sep: str = '.'):
    result = []
    field_types, field_metas = _get_annotated_triton_fields(dcls)
    for f in dataclasses.fields(dcls):
        f_type = field_types[f.name]
        new_key = parent_key + sep + f.name if parent_key else f.name
        meta = field_metas[f.name]
        result.append((new_key, TritonAggFlatField(new_key, f_type, meta)))
    return dict_factory(result)


def _flatten_agg_field_only(dcls,
                            parent_dcls = (),
                            parent_field_metas = (),
                               parent_key: str = '',
                               sep: str = '.'):
    # TODO avoid recursion
    # if hasattr(dcls, TRITON_AGG_FLATTEN_META_FIELD):
    #     return getattr(dcls, TRITON_AGG_FLATTEN_META_FIELD)
    result = {}
    field_types, field_metas = _get_annotated_triton_fields(dcls)
    new_parent_dcls = parent_dcls + (dcls,)
    for f in dataclasses.fields(dcls):
        f_type = field_types[f.name]
        new_key = parent_key + sep + f.name if parent_key else f.name
        meta = field_metas[f.name]
        if not meta._internal_is_constexpr and inspect.isclass(f_type) and dataclasses.is_dataclass(f_type) and issubclass(f_type, base_aggregate_value):
            if meta.accessor is not None:
                result[new_key] = (TritonAggFlatField(new_key, f_type, meta, parent_dcls, parent_field_metas))

                # metas_local = meta.accessor.get_flatten_field_metas(
                #     meta, f_type)
                # metas_local_new = {}
                # for k, v in metas_local.items():
                #     new_key_local = new_key + sep + k
                #     v = dataclasses.replace(v, path=new_key_local,)
                #     metas_local_new[new_key_local] = v
                # result.update(metas_local_new)
            else:
                result.update(
                    _flatten_agg_field_only(f_type,
                                            new_parent_dcls,
                                            parent_field_metas + (meta,),
                                            new_key,
                                            sep=sep))
        else:
            result[new_key] = (TritonAggFlatField(new_key, f_type, meta, parent_dcls, parent_field_metas))
    return result

def is_aggregate_type(ann_type: Any) -> bool:
    return inspect.isclass(ann_type) and dataclasses.is_dataclass(ann_type) and issubclass(ann_type, base_aggregate_value)

class base_aggregate_value(base_value):
    __triton_builtin__ = True
    __triton_aggregate__ = True

    @classmethod
    def _get_instance(cls):
        return super().__new__(cls)

    def __overrided_init__(
        self, cls, original_init, *args, _semantic=None, _generator=None, **kwargs
    ):
        assert inspect.isclass(cls) and dataclasses.is_dataclass(
            cls
        ), f"{cls} must be a dataclass"
        extra_kwargs = {}
        if isinstance(original_init, JITCallable):
            # raise ValueError(f"{cls.__name__}.__init__ cannot be a @triton.jit function")
            pass
        else:
            if "_semantic" in inspect.signature(original_init).parameters:
                extra_kwargs["_semantic"] = _semantic
            if "_generator" in inspect.signature(original_init).parameters:
                extra_kwargs["_generator"] = _generator
        original_init(self, *args, **extra_kwargs, **kwargs)
        # Require that the user-defined constructor initialized all fields.
        for field in dataclasses.fields(cls):
            if not hasattr(self, field.name):
                raise AttributeError(
                    f"constructor for {cls.__name__} did not initialize attribute '{field.name}'"
                )

    # Only allow setting attributes defined in the class annotations.
    def __setattr__(self, name, value):
        field_types, constexpr_fields = _get_annotated_field_types(self.__class__)
        if name not in field_types:
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")
        value_unwrapped = tl.core._unwrap_if_constexpr(value)
        field_type = field_types[name]
        if field_type is not typing.Any:
            if issubclass(field_type, tl.tensor):
                # TODO we can't assign int/float/bool (constexpr) to
                # tensor.
                # TODO better way to handle torch.Tensor, it's only used on host side.
                # TODO Optional type check (often Optional[tl.tensor])?
                allowed_cls = (tl.tensor, int, float, bool, torch.Tensor)
            else:
                allowed_cls = (field_type,)
            if (
                not isinstance(value_unwrapped, allowed_cls)
                and field_type is not tl.constexpr
            ):
                raise TypeError(
                    f"Expected {field_type} for attribute '{name}', got {type(value_unwrapped)}"
                )
        if name in constexpr_fields:
            # TODO should we check that value is a compile-time constant?
            value = tl.constexpr(value)
        super().__setattr__(name, value)

    def _flatten_ir(self, handles: list[ir.value]) -> None:
        field_types, _ = _get_annotated_field_types(self.__class__)
        for name in field_types:
            getattr(self, name)._flatten_ir(handles)

    @property
    def type(self):
        field_types, _ = _get_annotated_field_types(self.__class__)
        return _aggregate_type(
            self.__class__,
            [(name, getattr(self, name).type) for name in field_types.keys()],
        )


@dataclass_transform()
@overload
def aggregate(
    cls: type[_T], *, init: bool = True, kw_only: bool = ...
) -> type["StandardDataclass"]: ...


@dataclass_transform()
@overload
def aggregate(
    *, init: bool = True, kw_only: bool = ...
) -> Callable[[type[_T]], type["StandardDataclass"]]: ...


@dataclass_transform()
def aggregate(
    cls: Optional[type[_T]] = None, *, init: bool = True, kw_only: bool = False
) -> Union[type["StandardDataclass"], Callable[[type[_T]], type["StandardDataclass"]]]:
    # Define the wrapped Triton value type.
    def wrapper(cls: type[_T]) -> type["StandardDataclass"]:
        bases = cls.__bases__
        base_value_has_base_agg = any(
            issubclass(base, base_aggregate_value) for base in bases
        )
        if not base_value_has_base_agg:
            cls_dcls = dataclasses.dataclass(cls, init=init, kw_only=kw_only)

            @dataclasses.dataclass(init=init, kw_only=kw_only)
            class cls_with_base_agg(base_aggregate_value, cls_dcls):
                pass

            cls_with_base_agg.__name__ = cls.__name__
            cls_with_base_agg.__module__ = cls.__module__
            cls_with_base_agg.__qualname__ = cls.__qualname__
            cls_with_base_agg.__doc__ = cls.__doc__
            res_cls = cls_with_base_agg
        else:
            res_cls = cast(
                type["StandardDataclass"],
                dataclasses.dataclass(cls, init=init, kw_only=kw_only),
            )
        if inspect.getfile(cls.__init__) == "<string>":
            # __init__ is generated by dataclasses
            hash_attrs = []
        else:
            hash_attrs = [cls.__init__]
        if hasattr(cls, "__post_init__"):
            hash_attrs.append(getattr(cls, "__post_init__"))
        for name, member in inspect.getmembers(cls):
            if (
                inspect.isfunction(member)
                or inspect.ismethod(member)
                or isinstance(member, JITCallable)
            ):
                if name != "__init__":
                    hash_attrs.append(member)
        res_cls.hash_attrs = hash_attrs
        setattr(res_cls, TRITON_AGG_FLATTEN_META_FIELD,
                _flatten_agg_field_only(res_cls))
        setattr(res_cls, TRITON_AGG_META_FIELD, _get_agg_fields(res_cls, dict))

        if not isinstance(cls.__init__, JITCallable):
            # patch __init__ to support _semantic and _generator kwargs
            original_init = res_cls.__init__
            res_cls.__init__ = lambda self, *args, **kwargs: res_cls.__overrided_init__(
                self, res_cls, original_init, *args, **kwargs
            )
        return res_cls

    if cls is None:
        return wrapper
    else:
        return wrapper(cls)


@gluon_builtin
def aggregate_replace_gluon(obj: _T, _semantic=None, **changes) -> _T:
    return dataclasses.replace(obj, **changes)


@tl.core.builtin
def aggregate_replace_triton(obj: _T, _semantic=None, **changes) -> _T:
    return dataclasses.replace(obj, **changes)


@gluon_builtin
def aggregate_super_method_gluon(
    method: BoundJITMethod, _semantic=None
) -> BoundJITMethod:
    self = method.__self__
    name = method.__func__.__name__
    cls = self.__class__
    bases = cls.__bases__
    assert (
        len(bases) == 1
    ), "Only single inheritance is supported for aggregate_super_method_gluon"
    base_cls = bases[0]
    super_method = getattr(base_cls, name)
    new_method = BoundJITMethod(self, super_method)
    return new_method


@tl.core.builtin
def aggregate_super_method_triton(
    method: BoundJITMethod, _semantic=None
) -> BoundJITMethod:
    self = method.__self__
    name = method.__func__.__name__
    cls = self.__class__
    bases = cls.__bases__
    assert (
        len(bases) == 1
    ), "Only single inheritance is supported for aggregate_super_method_gluon"
    base_cls = bases[0]
    super_method = getattr(base_cls, name)
    new_method = BoundJITMethod(self, super_method)
    return new_method


PS = ParamSpec("PS")


class TritonJitFunctionAnno(Generic[PS]):

    def __call__(self, *args: PS.args, **kwargs: PS.kwargs) -> "CompiledKernel": ...


class TritonJitFunctionWrappedAnno(Generic[PS]):

    def __getitem__(self, grid: Any) -> TritonJitFunctionAnno[PS]: ...


def gluon_jit(fn: Callable[PS, _T]) -> Callable[PS, _T]:
    if isinstance(fn, staticmethod):
        fn = fn.__func__
    return cast(Callable[PS, _T], gluon.jit(fn))


def gluon_jit_kernel(fn: Callable[PS, Any]) -> TritonJitFunctionWrappedAnno[PS]:
    return cast(TritonJitFunctionWrappedAnno[PS], gluon.jit(fn))


def triton_jit(fn: Callable[PS, _T]) -> Callable[PS, _T]:
    if isinstance(fn, staticmethod):
        fn = fn.__func__
    return cast(Callable[PS, _T], triton.jit(fn))


def triton_jit_kernel(fn: Callable[PS, Any]) -> TritonJitFunctionWrappedAnno[PS]:
    return cast(TritonJitFunctionWrappedAnno[PS], triton.jit(fn))

