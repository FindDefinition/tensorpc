from .aggtype import TRITON_AGG_FLATTEN_META_FIELD, TRITON_AGG_META_FIELD, gluon_jit, gluon_builtin, is_aggregate_type, triton_builtin, constexpr_function, aggregate, triton_jit
from triton.experimental.gluon import language as gl
import triton.language as tl

import dataclasses
import torch
import triton
import inspect
from typing import get_type_hints, Any, TypeVar, Union

_T = TypeVar("_T")

@gluon_jit
def get_all_fields_with_type_gluon_v1(obj, agg_type: gl.constexpr) -> tl.tuple:
    all_field_keys: gl.constexpr = get_all_field_keys_with_type(obj, agg_type)
    # gl.static_print("!", all_field_keys)
    return tl.tuple([_get_field_gluon(obj, k) for k in tl.tuple(all_field_keys)])

@gluon_builtin
def get_all_fields_with_type_gluon(obj, agg_type, _semantic=None, ) -> tl.tuple:
    all_field_keys = __get_all_field_keys_with_type(obj, agg_type)
    # gl.static_print("!", all_field_keys)
    return tl.tuple([getattr(obj, k.value) for k in all_field_keys])

@triton_builtin
def get_all_fields_with_type_triton(obj, agg_type, _semantic=None, ) -> tl.tuple:
    all_field_keys = __get_all_field_keys_with_type(obj, agg_type)
    # gl.static_print("!", all_field_keys)
    return tl.tuple([getattr(obj, k.value) for k in all_field_keys])


@gluon_builtin
def replace_all_fields_with_type_gluon(obj: _T, agg_type, fields: tl.tuple, _semantic=None) -> _T:
    all_field_keys = __get_all_field_keys_with_type(obj, agg_type)
    kv = {k.value: v for k, v in zip(all_field_keys, fields.values)}
    return dataclasses.replace(obj, **kv)

@triton_builtin
def replace_all_fields_with_type_triton(obj: _T, agg_type, fields: tl.tuple, _semantic=None) -> _T:
    all_field_keys = __get_all_field_keys_with_type(obj, agg_type)
    kv = {k.value: v for k, v in zip(all_field_keys, fields.values)}
    return dataclasses.replace(obj, **kv)


@constexpr_function
def _get_field_is_constexpr(obj, field):
    field = tl.core._unwrap_if_constexpr(field)

    res = getattr(obj, field)
    res_is_constexpr = isinstance(res, tl.constexpr)

    return res_is_constexpr

@constexpr_function
def has_field(obj, field):
    field = tl.core._unwrap_if_constexpr(field)
    return hasattr(obj, field)

@constexpr_function
def all_fields_is_constexpr(obj, fields):
    fields = tl.core._unwrap_if_constexpr(fields)
    for field in fields:
        res = getattr(obj, field)
        res_is_constexpr = isinstance(res, tl.constexpr)
        if not res_is_constexpr:
            return False 

    return True


def __get_all_field_keys_with_type(obj, agg_type):
    all_types = get_type_hints(type(obj), include_extras=True)
    res: list[gl.constexpr] = []
    for k, v in all_types.items():
        if inspect.isclass(v) and issubclass(v, agg_type):
            res.append(gl.constexpr(k))
    return tuple(res)

__get_all_field_keys_with_type.__triton_builtin__ = True

@constexpr_function
def get_all_field_keys_with_type(obj, agg_type):
    return __get_all_field_keys_with_type(obj, agg_type)


@constexpr_function
def filter_constexpr_fields(obj, fields):
    fields = tl.core._unwrap_if_constexpr(fields)
    res_list: list[str] = []
    for field in fields:
        res = getattr(obj, field)
        res_is_constexpr = isinstance(res, tl.constexpr)
        if res_is_constexpr:
            res_list.append(field)

    return res_list

@constexpr_function
def filter_non_constexpr_fields(obj, fields):
    fields = tl.core._unwrap_if_constexpr(fields)
    res_list: list[str] = []
    for field in fields:
        res = getattr(obj, field)
        res_is_constexpr = isinstance(res, tl.constexpr)
        if not res_is_constexpr:
            res_list.append(field)

    return res_list


@constexpr_function
def is_constexpr(obj):
    res_is_constexpr = isinstance(obj, (tl.constexpr, str, int, float, bool))
    return res_is_constexpr

@gluon_builtin
def _get_field_gluon(obj, field, *, _semantic=None):
    field = tl.core._unwrap_if_constexpr(field)
    res = getattr(obj, field)
    res_is_constexpr = isinstance(res, tl.constexpr)
    if res_is_constexpr:
        return tl.constexpr(res)
    else:
        return res

@gluon_builtin
def _get_field_type_gluon(dcls, field, *, _semantic=None):
    field = tl.core._unwrap_if_constexpr(field)
    agg_fields = getattr(dcls, TRITON_AGG_FLATTEN_META_FIELD)
    return tl.constexpr(agg_fields[field].type)


@tl.core.builtin
def _get_field_type_triton(dcls, field, *, _semantic=None):
    field = tl.core._unwrap_if_constexpr(field)
    agg_fields = getattr(dcls, TRITON_AGG_FLATTEN_META_FIELD)
    return tl.constexpr(agg_fields[field].type)



@aggregate
class TritonField:
    value: Any
    is_constexpr: tl.constexpr = False

@aggregate
class TritonConstexprField:
    value: tl.constexpr
    is_constexpr: tl.constexpr = True

@gluon_jit
def dispatch_value(value):
    if is_constexpr(value):
        return TritonConstexprField(value, True)
    else:
        return TritonField(value, False)

@gluon_jit
def get_field_gluon(obj, field: gl.constexpr) -> Any:
    res_is_c: tl.constexpr = _get_field_is_constexpr(obj, field)

    if res_is_c:
        res: tl.constexpr  = _get_field_gluon(obj, field)
        return TritonConstexprField(res, True)
    else:
        res  = _get_field_gluon(obj, field)
        return TritonField(res, False)

@triton_jit
def get_field_triton(obj, field: tl.constexpr) -> Any:
    res_is_c: tl.constexpr = _get_field_is_constexpr(obj, field)

    if res_is_c:
        res: tl.constexpr  = _get_field_triton(obj, field)
        return TritonConstexprField(res, True)
    else:
        res  = _get_field_triton(obj, field)
        return TritonField(res, False)


@gluon_jit
def get_field(obj, field: gl.constexpr) -> Union[TritonConstexprField, TritonField]:
    res_is_c: tl.constexpr = _get_field_is_constexpr(obj, field)

    if res_is_c:
        if is_gluon():
            res: tl.constexpr  = _get_field_gluon(obj, field)
        else:
            res: tl.constexpr  = _get_field_triton(obj, field)
        return TritonConstexprField(res, True)
    else:
        if is_gluon():
            res  = _get_field_gluon(obj, field)
        else:
            res  = _get_field_triton(obj, field)
        return TritonField(res, False)


_TORCH_TO_GL_DTYPE: dict[torch.dtype, gl.dtype] = {
    torch.int8: gl.int8,
    torch.uint8: gl.uint8,
    torch.float16: gl.float16,
    torch.bfloat16: gl.bfloat16,
    torch.float32: gl.float32,
    torch.float8_e4m3fn: gl.float8e4nv,
    torch.float8_e5m2: gl.float8e5,
    torch.float8_e4m3fnuz: gl.float8e4b8,
}

def torch_dtype_to_gluon(dtype: torch.dtype) -> gl.dtype:
    if dtype not in _TORCH_TO_GL_DTYPE:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return _TORCH_TO_GL_DTYPE[dtype]

@tl.core.builtin
def is_gluon(_semantic=None):
    # https://github.com/sublinear-systems/triton-utils/blob/master/utils.py#L106
    return isinstance(_semantic, triton.experimental.gluon.language._semantic.GluonSemantic)

@tl.core.builtin
def get_num_warps(_semantic=None, _generator=None):
    """
    Returns the number of warps that execute the current context, including in warp-specialized regions.
    """
    if _generator.caller_context is not None:
        # assert isinstance(generator.caller_context, GluonCallerContext)
        return gl.constexpr(_generator.caller_context.num_warps)
    return gl.constexpr(_semantic.builder.options.num_warps)

@constexpr_function
def get_field_type(dcls, field):
    field = tl.core._unwrap_if_constexpr(field)
    assert isinstance(field, str)
    field_parts = field.split(".")
    for part in field_parts:
        # assert is_aggregate_type(dcls), f"Expected aggregate type but got {dcls}"
        agg_fields = getattr(dcls, TRITON_AGG_META_FIELD)
        dcls = agg_fields[part].type
    return dcls
