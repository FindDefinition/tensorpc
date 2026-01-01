import ast
from collections.abc import Mapping, Sequence
import contextlib
import contextvars
from dataclasses import is_dataclass
import enum
from functools import partial
import inspect
import sys
from typing import (TYPE_CHECKING, Any, Callable, ForwardRef, Generic, Optional, Type, TypeAlias, TypeVar, Union,
                    cast, overload)

import pydantic
from pydantic_core import PydanticUndefined
from typing_extensions import Literal, Self, get_overloads, ParamSpec, ParamSpecArgs

import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.annolib import (AnnotatedType, DataclassType, T_dataclass,
                                   Undefined, get_type_hints_with_cache, is_optional, is_undefined,
                                   parse_type_may_optional_undefined, resolve_type_hints_with_cache,
                                   undefined)
from tensorpc.core.inspecttools import unwrap_fn_static_cls_property
from tensorpc.core.pfl.constants import PFL_BUILTIN_PROXY_INIT_FN, PFL_COMPILE_META_ATTR, PFL_STDLIB_FUNC_META_ATTR, PFL_FUNC_ANNO_META_ATTR
from tensorpc.core.moduleid import get_module_id_of_type, get_qualname_of_type
from tensorpc.core.tree_id import UniqueTreeId
from .typedefs import (BoolOpType, BinOpType, CompareType, UnaryOpType)

from .pfl_reg import STD_REGISTRY, StdRegistryItem, register_pfl_std

from tensorpc.utils.rich_logging import get_logger
if TYPE_CHECKING:
    from .pfl_ast import PFLFunc

PFL_LOGGER = get_logger("pfl")

_T = TypeVar("_T")

@dataclasses.dataclass
class PFLMetaInferResult:
    data: Any

@dataclasses.dataclass
class PFLVariableMeta:
    data: Any
    meta_infer: Optional[Callable[..., PFLMetaInferResult]] = None

class PFLExprType(enum.IntEnum):
    UNKNOWN = -1
    NUMBER = 0
    BOOL = 1
    STRING = 2
    ARRAY = 3
    OBJECT = 4
    NDARRAY = 5
    FUNCTION = 6
    NONE_TYPE = 7
    UNDEFINED_TYPE = 8
    DATACLASS_TYPE = 9
    ANY = 10
    DATACLASS_OBJECT = 11
    RANGE = 12
    # union is only allowed in function argument, variable/function return can't be union.
    # e.g. cpp function overload don't support return type overload.
    UNION = 13
    TUPLE = 14
    SLICE = 15
    ELLIPSIS = 16
    # typevar
    GENERIC_TYPE = 17
    # paramspec. only used to infer template function type
    # from a call. e.g. call_fn(template_fn, args)
    # args must be tuple, we can use types in args to infer
    # params of template_fn.
    # support of paramspec kwargs isn't planned.
    GENERIC_PARAM_SPEC = 18
    GENERIC_PARAM_ARGS = 19

_BASE_TYPE_TO_STRING = {
    PFLExprType.UNKNOWN: "unknown",
    PFLExprType.NUMBER: "number",
    PFLExprType.BOOL: "bool",
    PFLExprType.STRING: "string",
    PFLExprType.NONE_TYPE: "null",
    PFLExprType.ELLIPSIS: "...",

    PFLExprType.UNDEFINED_TYPE: "undefined",
    PFLExprType.ANY: "any",
    PFLExprType.RANGE: "range",
    PFLExprType.SLICE: "slice",

}
_TYPE_CAN_CAST_TO_BOOL = {
    PFLExprType.NUMBER,
    PFLExprType.BOOL,
    PFLExprType.STRING,
    PFLExprType.ARRAY,
    PFLExprType.OBJECT,
    PFLExprType.DATACLASS_OBJECT,
}

_TYPE_SUPPORT_BINARY_OP = {
    PFLExprType.NUMBER,
    PFLExprType.BOOL,
}

_TYPE_SUPPORT_UNARY_OP = {
    PFLExprType.NUMBER,
    PFLExprType.BOOL,
}

_TYPE_SUPPORT_COMPARE_OP = {
    PFLExprType.NUMBER,
    PFLExprType.BOOL,
    PFLExprType.STRING,
    PFLExprType.ARRAY,
    PFLExprType.OBJECT,
}

BASE_ANNO_TYPE_TO_PFLSTATIC_TYPE = {
    int: PFLExprType.NUMBER,
    float: PFLExprType.NUMBER,
    bool: PFLExprType.BOOL,
    str: PFLExprType.STRING,
    type(None): PFLExprType.NONE_TYPE,
    type(Ellipsis): PFLExprType.ELLIPSIS,
    Undefined: PFLExprType.UNDEFINED_TYPE,
    range: PFLExprType.RANGE,
}

@dataclasses.dataclass
class _DummyPydantic:
    pass 

def is_dcls_init_defined_by_user(dcls: Type[DataclassType]):
    assert dataclasses.is_dataclass(dcls)
    if dataclasses.is_pydantic_dataclass(dcls):
        return dcls.__init__ is not _DummyPydantic.__init__
    else:
        file_path = inspect.getsourcefile(dcls.__init__)
        return file_path is not None and file_path != "<string>"

def is_dcls_post_init_defined_by_user(dcls: Type[DataclassType]):
    assert dataclasses.is_dataclass(dcls)
    return hasattr(dcls, "__post_init__")

@dataclasses.dataclass
class PFLTemplateFnSpecMeta:
    """used for template function specification based on 
    function call.

    WARNING: fn annotated with `PFLTemplateFnSpecMeta` must
    be scalar/list/dict of template functions,
    args annotated with `PFLTemplateFnSpecArgsMeta`
    must be tuple.
    e.g. 
    ```Python
    def fn(fn1: Annotated[Any, PFLTemplateFnSpecMeta("1")], 
           args: Annotated[Any, PFLTemplateFnSpecArgsMeta("1")]):
        pass

    def template_fn(a, b):
        return a + b
    # template_fn is specificated via types in args
    fn(template_fn, (val1, val2))
    ```
    """
    key: str

@dataclasses.dataclass
class PFLTemplateFnSpecArgsMeta:
    key: str

@dataclasses.dataclass
class PFLParseConfig:
    # TODO: currently we don't support variable with union type except
    # number type (int | float)
    allow_var_union: bool = False 
    # some language (e.g. js) don't support keyword argument,
    allow_kw: bool = False
    # a[1:2:1, ..., None]
    allow_nd_slice: bool = False
    # a[1:2:1]
    allow_slice: bool = False
    # if True, new variable CREATED IN ALL BRANCH can be used after if statement.
    # otherwise new variable can only be used in the branch scope it is created.
    # when we want to generate cpp-like code, we need to set this to False,
    allow_new_var_after_if: bool = True
    # allow isinstance. this op is supported in py/js, not supported in cpp.
    # currently isinstance is compile-time, since we don't support var union.
    allow_isinstance: bool = True
    # if the test value is constexpr, we can only parse one branch to avoid
    # type error. keep in mind that new variables created in the branch
    # are visible in parent block.
    # WARNING: inline if is only available in function marked with "template"
    inline_constexpr_if: bool = True
    # js have cpp-style variable declaration, when we use
    # tuple assign, assigned variable must all-exist in scope or
    # not created yet.
    tuple_assign_must_be_homogeneous: bool = False
    allow_custom_class: bool = False
    # enable partial type infer, for string expr parsing.
    # WARNING: unknown attr call isn't allowed.
    allow_partial_type_infer: bool = False
    # if True, allow empty container literal to be any type.
    # e.g. `[]` will have type `list[Any]`
    allow_dynamic_container_literal: bool = False
    # allow type in slice to be unknown/any.
    allow_partial_in_slice: bool = False
    # if True, var type can be overridden by later assignment.
    allow_var_type_override: bool = False
    # if True, parser will try to remove optional from variable type based on condition
    # e.g. xxx is not None, WARNING: only support if block and simple cond for now.
    allow_remove_optional_based_on_cond: bool = False 

@dataclasses.dataclass
class StaticEvalConfig:
    # meta eval support two feature: custom infer function and partial run.
    # 1. allow user assign a meta infer function for each 
    # overloaded operator, functions and methods.
    # if not set, user should use proxy object instead of meta func
    # for custom object.
    # 2. partial call: some operands of meta infer func can have
    # no eval result.
    prefer_meta_eval: bool = False
    # when allow partial, if any argument of a op is undefined, the result of
    # this op is undefined. otherwise raise error.
    allow_partial: bool = True

class PFLErrorFormatContext:
    def __init__(self,
                 lines: list[str],
                 num_line_neighbor: int = 1):
        self.lines = lines
        self.num_line_neighbor = num_line_neighbor

    def format_error_from_lines_node(self, node: Any):
        from tensorpc.core.pfl.pfl_ast import PFLAstNodeBase
        if isinstance(node, ast.AST):
            if hasattr(node, "lineno") and hasattr(node, "col_offset"):
                lineno = node.lineno  # type: ignore
                col_offset = node.col_offset  # type: ignore
                end_col_offset = col_offset + 1
                end_lineno = lineno
                if hasattr(node, "end_col_offset"):
                    end_col_offset = node.end_col_offset # type: ignore
                if hasattr(node, "end_lineno"):
                    end_lineno = node.end_lineno # type: ignore
            else:
                return 
        elif isinstance(node, PFLAstNodeBase):
            lineno = node.source_loc[0]
            col_offset = node.source_loc[1]
            end_col_offset = col_offset + 1
            end_lineno = lineno
            if node.source_loc[2] is not None:
                end_lineno = node.source_loc[2]
            if node.source_loc[3] is not None:
                end_col_offset = node.source_loc[3]
        else:
            raise NotImplementedError
        start_line = max(lineno - self.num_line_neighbor, 1)

        min_length = max(1, end_col_offset - col_offset)
        end_line = min(end_lineno + self.num_line_neighbor, len(self.lines))
        error_lines = self.lines[start_line - 1:end_line].copy()
        if error_lines:
            indicate_line = f"{' ' * col_offset}{'^' * min_length}"
            error_lines.insert(end_lineno - start_line + 1, indicate_line)
            max_line_length = max(map(len, error_lines))
            error_lines.insert(0, "*" * max_line_length)
            error_lines.append("*" * max_line_length)
            return "\n".join(error_lines)
        return ""

class PFLParseCache:
    def __init__(self, backend: str, var_preproc: Callable[[Any], "PFLProcessedVarMeta"], 
            temp_std_items: Optional[dict[Type, StdRegistryItem]] = None):
        self._func_parse_result_cache: dict[Callable, PFLExprInfo] = {}
        self._user_dcls_parse_result_cache: dict[Type[DataclassType], PFLExprInfo] = {}

        self._annotype_cache: dict[Any, AnnotatedType] = {}
        self._std_item_cache: dict[Type, Optional[StdRegistryItem]] = {}
        if temp_std_items is not None:
            self._std_item_cache.update(temp_std_items)
        self._mapped_proxy_cache: dict[Type, Optional[StdRegistryItem]] = {}

        self._mapped_type_cache: dict[Type, StdRegistryItem] = {}
        self._backend = backend
        self._var_preproc = var_preproc
        self._local_cls_fn_cache: dict[str, dict[int, int]] = {}
        self._local_cls_fn_cnt_cache: dict[str, dict[int, int]] = {}

    def cached_parse_to_annotype(self,
                          type: Any) -> AnnotatedType:
        if type in self._annotype_cache:
            return self._annotype_cache[type]
        res = parse_type_may_optional_undefined(type)
        self._annotype_cache[type] = res
        return res

    def cached_parse_dcls(self,
                          dcls: Type[DataclassType],
                          external_local_ids: Optional[list[int]] = None) -> "PFLExprInfo":
        if dcls in self._user_dcls_parse_result_cache:
            return self._user_dcls_parse_result_cache[dcls]
        res = PFLExprInfo.from_dcls_type(dcls,
            external_local_ids=external_local_ids)
        has_user_init = is_dcls_init_defined_by_user(dcls)
        if has_user_init:
            init_fn = self.cached_parse_func(dcls.__init__,
                self_type=parse_type_may_optional_undefined(dcls))
            res.func_info = init_fn.get_func_info_checked()
            res.func_info.func_uid = get_module_id_of_type(dcls.__init__)
        else:
            assert res.dcls_info is not None
            res.func_info = dataclasses.replace(res.dcls_info)
        self._user_dcls_parse_result_cache[dcls] = res
        return res

    def cached_parse_func(self,
                          func: Callable,
                          is_bound_method: bool = False,
                          self_type: Optional[AnnotatedType] = None,
                          disable_type_check: bool = False,
                          ext_preproc_res: Optional["PFLProcessedVarMeta"] = None,
                          external_local_ids: Optional[list[int]] = None) -> "PFLExprInfo":
        preproc_res = self._var_preproc(func)
        func = preproc_res.value
        if ext_preproc_res is not None:
            compilable_meta = ext_preproc_res.compilable_meta
        else:
            compilable_meta = preproc_res.compilable_meta
        if func in self._func_parse_result_cache:
            return self._func_parse_result_cache[func]
        name = func.__name__
        if disable_type_check:
            # use (...Any) -> Any sig
            sig = inspect.Signature([varparam_fn("x", Any)], return_annotation=Any)
            return PFLExprInfo.from_signature(name, sig, raw_func=func)
        else:
            meta: Optional[PFLStdlibFuncMeta] = getattr(func, PFL_STDLIB_FUNC_META_ATTR, None)
            
            if meta is not None and meta.take_overloads_fn is not None:
                overload_fn = meta.take_overloads_fn
            else:
                overload_fn = func
            overloads = get_overloads(overload_fn)
            if overloads:
                sig = inspect.signature(overloads[0], eval_str=True)
                overload_sigs = [inspect.signature(o, eval_str=True) for o in overloads[1:]]
            else:
                sig = inspect.signature(func, eval_str=True)
                overload_sigs = None
            delay_parse_args = False 
            if compilable_meta is not None and compilable_meta.need_delayed_processing():
                delay_parse_args = True 
            res = PFLExprInfo.from_signature(name, sig, is_bound_method=is_bound_method, self_type=self_type, 
                overload_sigs=overload_sigs, raw_func=func, compilable_meta=compilable_meta,
                delay_parse_args=delay_parse_args)
            assert res.func_info is not None
            # generate function uid
            func_uid_base = get_module_id_of_type(func)
            if "<locals>" in func_uid_base:
                if external_local_ids is None:
                    external_local_ids = []
                local_nested_depth = func_uid_base.count("<locals>")
                if local_nested_depth == len(external_local_ids):
                    res.func_info._locals_ids = external_local_ids
                else:
                    cnt = PFLExprFuncInfo._get_local_defined_cls_fn_cnts(func, func_uid_base)
                    res.func_info._locals_ids = external_local_ids + [cnt]
            res.func_info.func_uid = func_uid_base
            return res 

    def cached_parse_std_item(self, item: StdRegistryItem) -> "PFLExprInfo":
        if item.is_func:
            res = self.cached_parse_func(item.dcls, disable_type_check=item._internal_disable_type_check)
        elif item.is_builtin:
            assert item.mapped is not None
            res = PFLExprInfo.from_annotype(
                parse_type_may_optional_undefined(item.dcls), is_type=True, parse_cache=self, proxy_dcls=cast(Type[DataclassType], item.dcls))
        else:
            res = PFLExprInfo.from_annotype(
                parse_type_may_optional_undefined(item.dcls), is_type=True, parse_cache=self)
            res.disable_dcls_ctor = item.disable_dcls_ctor
        res.is_stdlib = True
        return res 

    def cached_get_std_item(self, dcls: Type[T_dataclass]):
        if dcls in self._std_item_cache:
            return self._std_item_cache[dcls]
        item = STD_REGISTRY.get_item_by_dcls(dcls, self._backend)
        self._std_item_cache[dcls] = item
        return item

    def cached_get_dcls_by_mapped_type(self, usercls: Any):
        if usercls in self._mapped_type_cache:
            return self._mapped_type_cache[usercls]
        item = STD_REGISTRY.get_dcls_item_by_mapped_type(
            usercls, self._backend)
        if item is None:
            raise ValueError(
                f"can't find your mapped type {get_qualname_of_type(usercls)} from std registry."
            )
        self._mapped_type_cache[usercls] = item
        return item

    def cached_try_get_proxy_dcls_by_mapped_type(self, usercls: Any):
        if usercls in self._mapped_proxy_cache:
            return self._mapped_proxy_cache[usercls]
        item = STD_REGISTRY.get_proxy_dcls_by_mapped_type(
            usercls, self._backend)
        self._mapped_proxy_cache[usercls] = item
        return item

    @staticmethod
    def get_dcls_by_mapped_type(usercls: Any, backend: str):
        item = STD_REGISTRY.get_dcls_item_by_mapped_type(
            usercls, backend)
        if item is None:
            raise ValueError(
                f"can't find your mapped type {get_qualname_of_type(usercls)} from std registry."
            )
        return item

    @staticmethod
    def get_std_item(dcls: Type[T_dataclass], backend: str):
        item = STD_REGISTRY.get_item_by_dcls(dcls, backend)
        if item is None:
            raise ValueError(
                f"can't find your type {get_qualname_of_type(dcls)} from std registry."
            )
        return item


class PFLParseContext(PFLErrorFormatContext):

    def __init__(self,
                 lines: list[str],
                 func_globals: Any,
                 var_preproc: Callable[[Any], "PFLProcessedVarMeta"],
                 backend: str = "js",
                 temp_std_items: Optional[dict[Type, StdRegistryItem]] = None,
                 cfg: Optional[PFLParseConfig] = None,
                 eval_cfg: Optional[StaticEvalConfig] = None,
                 node_to_constants: Optional[dict[ast.AST, "PFLCompileConstant"]] = None,
                 compile_req: Optional["PFLCompileReq"] = None,
                 func_need_to_compile: Optional[list["PFLCompileReq"]] = None,
                 allow_inline_expand: bool = False):
        super().__init__(lines)
        # local states
        self.compile_req = compile_req
        self.anno_evaluate_globals = func_globals
        if node_to_constants is None:
            node_to_constants = {}
        self.node_to_constants = node_to_constants
        self.depend_compilables: list[str] = []
        # global states
        self._backend = backend
        self.cache = PFLParseCache(backend, var_preproc, temp_std_items)
        if cfg is None:
            cfg = PFLParseConfig()
        self.cfg = cfg
        if eval_cfg is None:
            eval_cfg = StaticEvalConfig()
        self.eval_cfg = eval_cfg
        self._disable_type_check: bool = False
        if func_need_to_compile is None:
            func_need_to_compile = []
        self._func_need_to_compile: list[PFLCompileReq] = func_need_to_compile
        self._allow_inline_expand = allow_inline_expand

    def get_compile_req_checked(self) -> "PFLCompileReq":
        if self.compile_req is None:
            raise ValueError("compile_req is None, please set it before use.")
        return self.compile_req

    @classmethod 
    def from_outer_ctx(cls, ctx: Self, lines: list[str], func_globals: Any, 
                 node_to_constants: Optional[dict[ast.AST, "PFLCompileConstant"]] = None,
                 compile_req: Optional["PFLCompileReq"] = None, allow_inline_expand: bool = False):
        assert not ctx._disable_type_check, "not supported inside decorator list"
        new_ctx = cls(lines, func_globals, ctx.cache._var_preproc, ctx._backend, cfg=ctx.cfg, 
            eval_cfg=ctx.eval_cfg,
            node_to_constants=node_to_constants,
            compile_req=compile_req,
            allow_inline_expand=allow_inline_expand)
        # cache and func_need_to_compile is shared.
        new_ctx.cache = ctx.cache
        new_ctx._func_need_to_compile = ctx._func_need_to_compile
        return new_ctx

    @classmethod 
    def create_root_ctx(cls, backend: str, 
            var_preproc: Callable[[Any], "PFLProcessedVarMeta"],
            cfg: Optional[PFLParseConfig] = None, 
            func_need_to_compile: Optional[list["PFLCompileReq"]] = None):
        return cls([], {}, var_preproc, backend, cfg=cfg, func_need_to_compile=func_need_to_compile) 

    def get_compile_req(self, func: Callable, info: Optional["PFLExprFuncInfo"], meta: Optional["PFLCompileFuncMeta"], args_from_call: Optional[tuple[list["PFLExprInfo"], dict[str, "PFLExprInfo"]]] = None,
                            self_type: Optional["PFLExprInfo"] = None,
                            is_prop: bool = False, is_method_def: bool = False,
                            bound_self: Optional[Any] = None,
                            is_dcls: bool = False,
                            local_ids: Optional[list[int]] = None) -> "PFLCompileReq":
        # args_from_call: used for template compile
        preproc_res = self.cache._var_preproc(func)
        func = preproc_res.value
        if meta is None:
            if info is None or info.compilable_meta is None:
                meta = PFLCompileFuncMeta([self._backend])
            else:
                meta = info.compilable_meta
        if info is None:
            func_uid = get_module_id_of_type(func)
        else:
            func_uid = info.func_uid
            if local_ids is None:
                local_ids = info._locals_ids
        # always enqueue compile request, compiler will check if it is needed.
        req = PFLCompileReq(func, func_uid, meta, info, args_from_call,
            self_type, is_prop, is_method_def, bound_self=bound_self,
            is_dcls=is_dcls, local_ids=local_ids)
        return req 

    def enqueue_func_compile(self, func: Callable, info: "PFLExprFuncInfo", args_from_call: Optional[tuple[list["PFLExprInfo"], dict[str, "PFLExprInfo"]]] = None,
                            self_type: Optional["PFLExprInfo"] = None,
                            is_prop: bool = False, is_method_def: bool = False,
                            bound_self: Optional[Any] = None,
                            local_ids: Optional[list[int]] = None):
        req = self.get_compile_req(func, info, None, args_from_call,
            self_type=self_type,
            is_prop=is_prop, is_method_def=is_method_def,
            bound_self=bound_self,
            local_ids=local_ids)
        self._func_need_to_compile.append(req)
        self.depend_compilables.append(info.func_uid)
        return req

    def enqueue_dcls_compile(self, func: Callable, info: "PFLExprFuncInfo", args_from_call: Optional[tuple[list["PFLExprInfo"], dict[str, "PFLExprInfo"]]] = None,
            self_type: Optional["PFLExprInfo"] = None):
        req = self.get_compile_req(func, info, None, args_from_call, is_dcls=True, self_type=self_type)
        self._func_need_to_compile.append(req)
        return req

_PFLPARSE_CONTEXT: contextvars.ContextVar[
    Optional[PFLParseContext]] = contextvars.ContextVar("PFLParseContext",
                                                        default=None)


@contextlib.contextmanager
def enter_parse_context(ctx: PFLParseContext):
    token = _PFLPARSE_CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _PFLPARSE_CONTEXT.reset(token)


def get_parse_context_checked():
    ctx = _PFLPARSE_CONTEXT.get()
    if ctx is None:
        raise ValueError("not in parse context")
    return ctx

def get_parse_cache_checked():
    ctx = _PFLPARSE_CONTEXT.get()
    if ctx is None:
        raise ValueError("not in parse context")
    return ctx.cache


def get_parse_context():
    ctx = _PFLPARSE_CONTEXT.get()
    return ctx

def get_parse_cache():
    ctx = _PFLPARSE_CONTEXT.get()
    if ctx is None:
        return None 
    return ctx.cache

def get_eval_cfg_in_parse_ctx():
    ctx = _PFLPARSE_CONTEXT.get()
    if ctx is None:
        return None 
    return ctx.eval_cfg


def has_parse_context():
    ctx = _PFLPARSE_CONTEXT.get()
    return ctx is not None

class PFLFuncArgFlag(enum.IntEnum):
    IS_VAARGS = 1 << 0
    IS_KW_VAARGS = 1 << 1
    IS_TYPE_PARSED = 1 << 2
    IS_TEMPLATE_FN_ARG_SPEC = 1 << 3

@dataclasses.dataclass(eq=False)
class PFLExprFuncArgInfo:
    name: str
    type: 'PFLExprInfo'
    default: Union[Undefined, Any] = undefined
    default_type: Union[Undefined, "PFLExprInfo"] = undefined
    flag: int = int(PFLFuncArgFlag.IS_TYPE_PARSED)
    # is_vaargs: bool = False
    # # for lazy-parsed dcls field.
    # is_type_parsed: bool = True
    # is_kw_vaargs: bool = False
    # is_template_fn_arg_spec: bool = False 
    template_fn_arg_spec_idx: Optional[int] = None

    @property 
    def is_vaargs(self) -> bool:
        return (self.flag & PFLFuncArgFlag.IS_VAARGS) != 0

    @is_vaargs.setter
    def is_vaargs(self, val: bool):
        if val:
            self.flag |= PFLFuncArgFlag.IS_VAARGS
        else:
            self.flag &= ~PFLFuncArgFlag.IS_VAARGS

    @property
    def is_kw_vaargs(self) -> bool:
        return (self.flag & PFLFuncArgFlag.IS_KW_VAARGS) != 0

    @is_kw_vaargs.setter
    def is_kw_vaargs(self, val: bool):
        if val:
            self.flag |= PFLFuncArgFlag.IS_KW_VAARGS
        else:
            self.flag &= ~PFLFuncArgFlag.IS_KW_VAARGS

    @property
    def is_type_parsed(self) -> bool:
        return (self.flag & PFLFuncArgFlag.IS_TYPE_PARSED) != 0

    @is_type_parsed.setter
    def is_type_parsed(self, val: bool):
        if val:
            self.flag |= PFLFuncArgFlag.IS_TYPE_PARSED
        else:
            self.flag &= ~PFLFuncArgFlag.IS_TYPE_PARSED

    @property
    def is_template_fn_arg_spec(self) -> bool:
        return (self.flag & PFLFuncArgFlag.IS_TEMPLATE_FN_ARG_SPEC) != 0

    @is_template_fn_arg_spec.setter
    def is_template_fn_arg_spec(self, val: bool):
        if val:
            self.flag |= PFLFuncArgFlag.IS_TEMPLATE_FN_ARG_SPEC
        else:
            self.flag &= ~PFLFuncArgFlag.IS_TEMPLATE_FN_ARG_SPEC

    def typevar_substitution(self, typevar_map: Mapping[TypeVar, "PFLExprInfo"]) -> Self:
        # we assume new type don't contains typevar.
        return dataclasses.replace(self, type=self.type.typevar_substitution(typevar_map))

    def __repr__(self) -> str:
        if not is_undefined(self.default):
            return f"{self.name}:{self.type}={self.default}"
        return f"{self.name}:{self.type}"

    def to_dict(self):
        res = {
            "name": self.name,
            "type": self.type.to_dict(),
        }
        if self.flag > 0:
            res["flag"] = self.flag
        return res 

_T = TypeVar("_T")

@dataclasses.dataclass
class FuncMatchResult(Generic[_T]):
    args: list[tuple[PFLExprFuncArgInfo, Union[_T, Undefined]]]
    vararg: Optional[tuple[PFLExprFuncArgInfo, list[_T]]] = None
    var_kwarg: Optional[tuple[PFLExprFuncArgInfo, dict[str, _T]]] = None

    def __post_init__(self):
        for a, _ in self.args:
            assert not a.is_vaargs and not a.is_kw_vaargs, "args should not contain vaargs or kw vaargs"
        if self.vararg is not None:
            assert self.vararg[0].is_vaargs, "vararg should be vaargs"
        if self.var_kwarg is not None:
            assert self.var_kwarg[0].is_kw_vaargs, "var_kwarg should be kw vaargs"

class PFLFuncInfoFlag(enum.IntEnum):
    IS_METHOD = 1 << 0
    IS_PROPERTY = 1 << 1
    IS_DCLS = 1 << 2
    IS_USER_DCLS = 1 << 3


@dataclasses.dataclass(eq=False)
class PFLExprFuncInfo:
    name: str
    return_type: "PFLExprInfo"
    args: list[PFLExprFuncArgInfo] = dataclasses.field(default_factory=list)
    # is_method: bool = False
    # is_property: bool = False
    flag: int = 0
    raw_func: Optional[Callable] = None
    # overload: when you define a function with overloads, the signature of origin function
    # will be ignored, and the overloads will be used instead.
    # keep in mind that if your f has three overloads, the first overload will be saved in main PFLExprInfo.
    # other overloads will be saved in `overloads`.
    overloads: Optional[list["PFLExprFuncInfo"]] = None

    compilable_meta: Optional["PFLCompileFuncMeta"] = None
    func_uid: str = ""
    # is_dcls: bool = False
    # is_user_dcls: bool = False
    # when some function is called as a bound method, this field is set to the type of the 'Self'.
    bound_self_type: Optional["PFLExprInfo"] = None
    has_template_fn_spec: bool = False
    _arg_name_to_idx: dict[str, int] = dataclasses.field(default_factory=dict)
    _vararg_pos: int = -1
    # for local defined functions
    _locals_ids: list[int] = dataclasses.field(default_factory=list)

    @property
    def is_method(self) -> bool:
        return (self.flag & PFLFuncInfoFlag.IS_METHOD) != 0    

    @is_method.setter
    def is_method(self, val: bool):
        if val:
            self.flag |= PFLFuncInfoFlag.IS_METHOD
        else:
            self.flag &= ~PFLFuncInfoFlag.IS_METHOD

    @property
    def is_property(self) -> bool:
        return (self.flag & PFLFuncInfoFlag.IS_PROPERTY) != 0

    @is_property.setter
    def is_property(self, val: bool):
        if val:
            self.flag |= PFLFuncInfoFlag.IS_PROPERTY
        else:
            self.flag &= ~PFLFuncInfoFlag.IS_PROPERTY

    @property
    def is_dcls(self) -> bool:
        return (self.flag & PFLFuncInfoFlag.IS_DCLS) != 0

    @is_dcls.setter
    def is_dcls(self, val: bool):
        if val:
            self.flag |= PFLFuncInfoFlag.IS_DCLS
        else:
            self.flag &= ~PFLFuncInfoFlag.IS_DCLS

    @property
    def is_user_dcls(self) -> bool:
        return (self.flag & PFLFuncInfoFlag.IS_USER_DCLS) != 0

    @is_user_dcls.setter
    def is_user_dcls(self, val: bool):
        if val:
            self.flag |= PFLFuncInfoFlag.IS_USER_DCLS
        else:
            self.flag &= ~PFLFuncInfoFlag.IS_USER_DCLS

    def __post_init__(self):
        for i, a in enumerate(self.args):
            self._arg_name_to_idx[a.name] = i
            if a.is_vaargs:
                self._vararg_pos = i
    
    def __repr__(self) -> str:
        args_str = []
        for arg in self.args:
            if arg.is_vaargs:
                args_str.append(f"...{arg}")
            else:
                if not is_undefined(arg.default):
                    prefix = f"{arg.type}[{arg.default}]"
                else:
                    prefix = str(arg.type)
                if self.is_dcls:
                    prefix = f"{arg.name}:{prefix}"
                if not isinstance(arg.type._constexpr_data, Undefined):
                    args_str.append(f"{prefix}={arg.type._constexpr_data}")
                else:
                    args_str.append(prefix)
        args = ", ".join(args_str)
        if self.is_dcls:
            res = f"{self.return_type}[{args}]"
        else:
            res = f"({args}) => {self.return_type}"
        return res

    def to_dict(self):
        args = [arg.to_dict() for arg in self.args]
        res: dict[str, Any] = {
            "args": args,
        }
        if self.flag > 0:
            res["flag"] = self.flag
        # if self.is_method:
        #     res["is_method"] = True
        # if self.is_property:
        #     res["is_property"] = True
        # if self.is_user_dcls:
        #     res["is_user_dcls"] = True
        if not self.is_dcls:
            # FIXME
            res.update({
                "return_type": self.return_type.to_dict() if self.return_type is not None else None,
            })
        return res
    
    def shallow_copy(self) -> "PFLExprFuncInfo":
        # if is dcls, args contains some mutable data, so we must copy
        # it. other fields are safe (immutable).
        new_args = [dataclasses.replace(arg) for arg in self.args]
        return dataclasses.replace(self, args=new_args)

    def is_template(self):
        if self.compilable_meta is None:
            return False 
        return self.compilable_meta.is_template

    def is_always_inline(self):
        if self.compilable_meta is None:
            return False 
        return self.compilable_meta.always_inline

    def need_delayed_processing(self):
        return self.is_template() or self.is_always_inline()

    def typevar_substitution(self, typevar_map: Mapping[TypeVar, "PFLExprInfo"]) -> Self:
        # we assume new type don't contains typevar.
        ret = self.return_type 
        if ret is not None:
            ret = ret.typevar_substitution(typevar_map)
        return dataclasses.replace(self, args=[c.typevar_substitution(typevar_map) for c in self.args], return_type=ret)

    def delayed_init_set_field_type(self, field_name: str, field_type: "PFLExprInfo"):
        """set field type by arguments.
        used for template user dataclass.
        """
        assert field_name in self._arg_name_to_idx, f"field {field_name} not found in args"
        field = self.args[self._arg_name_to_idx[field_name]]
        # print("delayed_init_set_field_type", field_name, field.is_type_parsed)
        if field.is_type_parsed:
            assert field.type.is_equal_type(field_type), f"field {field_name} type {field.type} is not equal to {field_type}"
        else:
            field.type = dataclasses.replace(field_type)
            field.is_type_parsed = True

    def get_field(self, field_name: str):
        return self.args[self._arg_name_to_idx[field_name]]

    def get_parsed_field(self, field_name: str):
        # print("get_parsed_field", field_name, self.is_template(), self.raw_func)
        field = self.args[self._arg_name_to_idx[field_name]]
        if field.is_type_parsed:
            return field
        else:
            if self.is_template():
                raise ValueError(f"field({field_name}) must be set by delayed_init_set_field_type before get attr.")
            assert self.raw_func is not None 
            # type_hints = get_type_hints_with_cache(self.raw_func, include_extras=True)
            type_hints = resolve_type_hints_with_cache(self.raw_func)
            anno = type_hints[field.name]
            annotype = parse_type_may_optional_undefined(anno)
            arg_type = PFLExprInfo.from_annotype(annotype,
                                            is_type=False,
                                            allow_union=False,
                                            allow_type_var=False)
            field.is_type_parsed = True 
            field.type = arg_type
            return field

    @classmethod
    def from_dcls_type(cls,
                       dcls: Type[DataclassType],
                       external_annos: Optional[dict[str, Any]] = None,
                       delay_parse_field: bool = False,
                       external_local_ids: Optional[list[int]] = None) -> Self:
        # type_hints = get_type_hints_with_cache(dcls, include_extras=True)
        type_hints = resolve_type_hints_with_cache(dcls)

        args: list[PFLExprFuncArgInfo] = []
        # type_hints = resolve_type_hints(anno_type.origin_type)

        for f in dataclasses.fields(dcls):
            if delay_parse_field:
                annotype = None
                arg_type = PFLExprInfo(PFLExprType.UNKNOWN)
            else:
                if external_annos is None or f.name not in external_annos:
                    anno = type_hints[f.name]
                else:
                    anno = external_annos[f.name]
                annotype = parse_type_may_optional_undefined(anno)
                arg_type = PFLExprInfo.from_annotype(annotype,
                                                is_type=False,
                                                allow_union=False,
                                                allow_type_var=False)
            arg = PFLExprFuncArgInfo(f.name, arg_type)
            # TODO: currently we skip all default_factory.
            if f.default is not dataclasses.MISSING:
                # f**k pydantic
                if isinstance(f.default, pydantic.fields.FieldInfo):
                    finfo = f.default
                    if finfo.default_factory is None and finfo.default is not PydanticUndefined:
                        arg.default = f.default
                        arg.default_type = PFLExprInfo.from_value(f.default)
                else:
                    arg.default = f.default
                    arg.default_type = PFLExprInfo.from_value(f.default)
            arg.is_type_parsed = annotype is not None
            args.append(arg)
        return_type = PFLExprInfo(
            PFLExprType.DATACLASS_OBJECT, 
            annotype=parse_type_may_optional_undefined(dcls),
        )
        # TODO do we need this?
        parse_cache = get_parse_cache()
        if parse_cache is None:
            meta = get_compilable_meta(dcls)
        else:
            preproc_res = parse_cache._var_preproc(dcls)
            meta = preproc_res.compilable_meta
        res = cls(name=dcls.__name__, args=args, raw_func=dcls, return_type=return_type, compilable_meta=meta)
        res.is_user_dcls = True
        res.is_dcls = True
        func_uid = get_module_id_of_type(dcls)
        if "<locals>" in func_uid:
            if external_local_ids is None:
                external_local_ids = []
            local_nested_depth = func_uid.count("<locals>")
            if local_nested_depth == len(external_local_ids):
                res._locals_ids = external_local_ids
            else:
                cnt = PFLExprFuncInfo._get_local_defined_cls_fn_cnts(dcls, func_uid)
                res._locals_ids = external_local_ids + [cnt]
        res.func_uid = func_uid
        return_type.dcls_info = res
        return res 

    @staticmethod 
    def _get_local_defined_cls_fn_uid(obj: Any, func_uid: str):
        cnt = PFLExprFuncInfo._get_local_defined_cls_fn_cnts(obj, func_uid)
        func_uid = func_uid.replace("<locals>", f"<locals>-{cnt}") 
        return func_uid

    @staticmethod 
    def _get_local_defined_cls_fn_cnts(obj: Any, func_uid: str):
        locals_cache = get_parse_cache_checked()._local_cls_fn_cache
        if func_uid not in locals_cache:
            locals_cache[func_uid] = {}
        dcls_id = id(obj)
        if dcls_id in locals_cache[func_uid]:
            cnt = locals_cache[func_uid][dcls_id]
        else:
            cnt = len(locals_cache[func_uid])
            locals_cache[func_uid][dcls_id] = cnt
        return cnt

    @classmethod
    def from_signature(cls,
                       name: str,
                       sig: inspect.Signature,
                       is_bound_method: bool = False,
                       self_type: Optional[AnnotatedType] = None,
                       overload_sigs: Optional[list[inspect.Signature]] = None,
                       raw_func: Optional[Callable] = None,
                       delay_parse_args: bool = False) -> Self:
        cnt = 0
        args: list[PFLExprFuncArgInfo] = []
        is_method = self_type is not None
        template_spec_arg_meta_idx: dict[str, int] = {}
        template_spec_meta_idx: dict[str, int] = {}
        for param in sig.parameters.values():
            is_template_fn_arg_spec: bool = False 
            if is_bound_method and cnt == 0:
                continue
            if is_method and cnt == 0:
                # first param is self, use self type
                annotype = self_type
            else:
                if delay_parse_args:
                    annotype = None
                else:
                    assert param.annotation is not inspect.Parameter.empty, f"param {param.name} must have annotation"
                    if param.annotation is Self:
                        assert self_type is not None 
                        annotype = self_type
                    else:
                        anno = param.annotation
                        annotype = parse_type_may_optional_undefined(anno, self_type=self_type)
            if annotype is not None:
                metadatas = annotype.annometa
                if metadatas is not None:
                    for m in metadatas:
                        if isinstance(m, (PFLTemplateFnSpecMeta)):
                            template_spec_meta_idx[m.key] = cnt
                        if isinstance(m, PFLTemplateFnSpecArgsMeta):
                            is_template_fn_arg_spec = True
                            assert m.key not in template_spec_arg_meta_idx, 'only one arg spec allowed for each key.'
                            template_spec_arg_meta_idx[m.key] = cnt
                arg_type = PFLExprInfo.from_annotype(annotype,
                                                is_type=False,
                                                allow_union=True,
                                                allow_type_var=True,
                                                allow_param_spec=True)
            else:
                arg_type = PFLExprInfo(PFLExprType.UNKNOWN)

            arg = PFLExprFuncArgInfo(param.name, arg_type)
            arg.is_template_fn_arg_spec = is_template_fn_arg_spec
            arg.is_type_parsed = annotype is not None

            args.append(arg)
            if param.default is not inspect.Parameter.empty:
                assert not is_template_fn_arg_spec, "arg with default value can't be marked with `PFLTemplateFnSpecArgsMeta`"
                arg.default = param.default
                arg.default_type = PFLExprInfo.from_value(param.default)
            arg.is_vaargs = param.kind == inspect.Parameter.VAR_POSITIONAL
            cnt += 1
        for k in template_spec_arg_meta_idx.keys():
            assert k in template_spec_meta_idx, f"PFLTemplateFnSpecMeta {k} not found"
        for k, idx in template_spec_meta_idx.items():
            assert k in template_spec_arg_meta_idx, f"PFLTemplateFnSpecArgsMeta {k} not found"
            arg = args[idx]
            arg.template_fn_arg_spec_idx = template_spec_arg_meta_idx[k]
        if sig.return_annotation is not inspect.Parameter.empty:
            if sig.return_annotation is not None:
                if sig.return_annotation is Self:
                    assert self_type is not None 
                    annotype = self_type 
                else:
                    ret_anno = sig.return_annotation
                    annotype = parse_type_may_optional_undefined(
                        ret_anno, self_type=self_type)
                return_type = PFLExprInfo.from_annotype(annotype,
                                                            is_type=False,
                                                            allow_type_var=True)
            else:
                return_type = PFLExprInfo(PFLExprType.NONE_TYPE)
        else:
            if delay_parse_args:
                return_type = PFLExprInfo(PFLExprType.UNKNOWN)
            else:
                return_type = PFLExprInfo(PFLExprType.NONE_TYPE)
        res = cls(name=name, args=args, raw_func=raw_func, return_type=return_type,
            has_template_fn_spec=len(template_spec_meta_idx) > 0)
        res.is_method = is_method
        if overload_sigs is not None:
            assert not delay_parse_args, "template functions shouldn't contain any overload."
            res.overloads = [cls.from_signature(
                name, s, is_bound_method=is_bound_method, self_type=self_type) for s in overload_sigs]
        return res

    def match_args_to_sig(self, args_from_call: tuple[list[_T], dict[str, _T]]) -> FuncMatchResult[_T]:
        args, kwargs = args_from_call
        vararg_pos = self._vararg_pos
        bound_arg_inc = int(self.bound_self_type is not None)
        num_query_args_fake = len(args) + bound_arg_inc
        if vararg_pos < 0:
            max_num_args = len(self.args)
            if self.args:
                max_num_args -= int(self.args[-1].is_kw_vaargs)
            assert num_query_args_fake <= max_num_args, f"func {self} expect at most {max_num_args} pos args, but got {len(args)}"
        match_args: list[tuple[PFLExprFuncArgInfo, Union[_T, Undefined]]] = []
        match_vararg: Optional[tuple[PFLExprFuncArgInfo, list[_T]]] = None
        match_var_kwarg: Optional[tuple[PFLExprFuncArgInfo, dict[str, _T]]] = None
        if self._vararg_pos >= 0:
            match_vararg = (self.args[self._vararg_pos], [])
        if self.args and self.args[-1].is_kw_vaargs:
            match_var_kwarg = (self.args[-1], {})
        for i, a in enumerate(self.args):
            # remove vaargs and va_kwargs later.
            match_args.append((a, undefined))
        for i, a in enumerate(args):
            arg_idx = i + bound_arg_inc
            if vararg_pos < 0:
                match_args[arg_idx] = (self.args[arg_idx], a)
            else:
                if arg_idx < vararg_pos:
                    match_args[arg_idx] = (self.args[arg_idx], a)
                else:
                    assert match_vararg is not None
                    match_vararg[1].append(a)
        for name, a in kwargs.items():
            arg_idx = self._arg_name_to_idx[name]
            arg_info = self.args[arg_idx]
            if arg_info.is_kw_vaargs:
                assert match_var_kwarg is not None
                match_var_kwarg[1][name] = a
            else:
                match_args[arg_idx] = (arg_info, a)
        # remove bound
        if self.bound_self_type is not None:
            assert len(match_args) > 0
            match_args = match_args[1:]
        # finally remove var args/kwargs from match_args
        match_args = [a for a in match_args if not a[0].is_vaargs and not a[0].is_kw_vaargs]
        res: FuncMatchResult[_T] = FuncMatchResult(match_args, match_vararg, match_var_kwarg)
        return res

    def get_bounded_type(self, bound_self: "PFLExprInfo"):
        new_ovs = None 
        if self.overloads is not None:
            new_ovs = [o.get_bounded_type(bound_self) for o in self.overloads]
        return dataclasses.replace(self, bound_self_type=bound_self, overloads=new_ovs)

class PFLExprInfoFlags(enum.IntEnum):
    HAS_OPTIONAL = 1 << 0
    HAS_UNDEFINED = 1 << 1
    IS_STD_LIB = 1 << 2

@dataclasses.dataclass(eq=False)
class PFLExprInfo:
    type: PFLExprType
    childs: list['PFLExprInfo'] = dataclasses.field(default_factory=list)
    flag: int = 0
    # has_optional: bool = False
    # has_undefined: bool = False
    proxy_dcls: Optional[Any] = None
    disable_dcls_ctor: bool = False
    # for custom dataclass
    mapped: str = ""
    # for container and dataclass
    annotype: Optional[AnnotatedType] = None
    anno_metadatas_ext: list[Any] = dataclasses.field(default_factory=list)
    # when this expr is function, this may be set. (dataclass ctor is lazy-parsed).
    func_info: Optional[PFLExprFuncInfo] = None
    dcls_info: Optional[PFLExprFuncInfo] = None

    # indicate this function/operator result use a compiled function, not stdlib function.
    # we need to get inside when evaluation
    compiled_uid: Optional[str] = None
    # currently only used by subscript, requires two functions (__getitem__ and __setitem__)
    additional_compiled_uid: Optional[str] = None
    delayed_compile_req: Optional[Any] = None
    # for dataclass in function arg
    # is_stdlib: bool = False
    # for meta call (type validation, shape inference, etc)
    # TODO should it be sent to frontend?
    _metadata: Union[Undefined, Any] = undefined
    _meta_infer: Optional[Callable[..., Optional[PFLMetaInferResult]]] = None
    _force_meta_infer: bool = False
    _static_type_infer: Optional[Callable[..., Any]] = None
    _constexpr_data: Union[Undefined, Any] = undefined

    @property 
    def has_optional(self) -> bool:
        return (self.flag & PFLExprInfoFlags.HAS_OPTIONAL) != 0

    @has_optional.setter
    def has_optional(self, val: bool):
        if val:
            self.flag |= PFLExprInfoFlags.HAS_OPTIONAL
        else:
            self.flag &= ~PFLExprInfoFlags.HAS_OPTIONAL

    @property
    def has_undefined(self) -> bool:
        return (self.flag & PFLExprInfoFlags.HAS_UNDEFINED) != 0

    @has_undefined.setter
    def has_undefined(self, val: bool):
        if val:
            self.flag |= PFLExprInfoFlags.HAS_UNDEFINED
        else:
            self.flag &= ~PFLExprInfoFlags.HAS_UNDEFINED

    @property
    def is_stdlib(self) -> bool:
        return (self.flag & PFLExprInfoFlags.IS_STD_LIB) != 0

    @is_stdlib.setter
    def is_stdlib(self, val: bool):
        if val:
            self.flag |= PFLExprInfoFlags.IS_STD_LIB
        else:
            self.flag &= ~PFLExprInfoFlags.IS_STD_LIB

    def to_dict(self):
        childs = [c.to_dict() for c in self.childs]
        res: dict[str, Any] = {
            "type": self.type,
        }
        if self.compiled_uid is not None:
            res["compiled_uid"] = self.compiled_uid
        if self.additional_compiled_uid is not None:
            res["additional_compiled_uid"] = self.additional_compiled_uid
        if self.childs:
            res["childs"] = childs
        if self.flag != 0:
            res["flag"] = self.flag
        if self.mapped:
            res["mapped"] = self.mapped
        if not isinstance(self.metadata, Undefined):
            res["metadata"] = self.metadata
        if self.func_info is not None:
            res["func_info"] = self.func_info.to_dict()
        if self.dcls_info is not None:
            res["dcls_info"] = self.dcls_info.to_dict()
        return res

    def __repr__(self) -> str:
        if self.type == PFLExprType.ARRAY:
            child_repr = str(self.childs[0])
            res = f"{child_repr}[]"
        elif self.type == PFLExprType.NUMBER:
            if self.annotype is not None:
                res = f"number<{get_qualname_of_type(self.annotype.origin_type)}>"
            else:
                res = f"number<unknown>"
        elif self.type == PFLExprType.OBJECT:
            child_repr = str(self.childs[0])
            res = f"Record<string, {child_repr}>"
        elif self.type == PFLExprType.DATACLASS_TYPE:
            assert self.annotype is not None
            res = str(self.annotype.origin_type.__name__)
        elif self.type == PFLExprType.DATACLASS_OBJECT:
            assert self.annotype is not None
            res = str(self.annotype.origin_type.__name__) #  + "()"
        elif self.type == PFLExprType.UNION:
            child_reprs = [str(c) for c in self.childs]
            res = f"{' | '.join(child_reprs)}"
        elif self.type == PFLExprType.TUPLE:
            child_reprs = [str(c) for c in self.childs]
            res = f"[{', '.join(child_reprs)}]"
        elif self.type == PFLExprType.SLICE:
            res = f"slice"
        elif self.type == PFLExprType.GENERIC_TYPE:
            res = f"~T"
        elif self.type == PFLExprType.FUNCTION:
            if self.func_info is None and self.delayed_compile_req is not None:
                # TODO better repr for template function
                res = f"template_func_no_spec(...)"
            else:
                assert self.func_info is not None 
                res = str(self.func_info)
        else:
            res = _BASE_TYPE_TO_STRING[self.type]
        # if self.mapped:
        #     res += f"<{self.mapped}>"

        if self.has_optional:
            res += " | null"
        if self.has_undefined:
            res += " | undefined"
        return res

    def get_origin_type_checked(self):
        assert self.annotype is not None
        return self.annotype.origin_type

    def get_func_info_checked(self) -> PFLExprFuncInfo:
        assert self.func_info is not None, "func_info is None"
        return self.func_info

    def get_dcls_info_checked(self) -> PFLExprFuncInfo:
        assert self.dcls_info is not None, "dcls_info is None"
        return self.dcls_info

    def get_optional_undefined_removed(self) -> Self:
        annotype = self.annotype
        if annotype is not None:
            annotype = annotype.get_optional_undefined_removed()
        res = dataclasses.replace(self, annotype=annotype)
        res.has_optional = False
        res.has_undefined = False
        return res

    @property 
    def is_user_dcls(self):
        return self.dcls_info is not None and self.dcls_info.is_user_dcls

    def get_bounded_type(self, bound_self: Self):
        assert self.type == PFLExprType.FUNCTION
        func_info = self.get_func_info_checked()
        new_func_info = func_info.get_bounded_type(bound_self)
        return dataclasses.replace(self, func_info=new_func_info)

    @classmethod
    def from_annotype(cls,
                      annotype: AnnotatedType,
                      is_type: bool = False,
                      allow_union: bool = False,
                      allow_type_var: bool = False,
                      parse_cache: Optional[PFLParseCache] = None,
                      proxy_dcls: Optional[Type[DataclassType]] = None,
                      allow_param_spec: bool = False,
                      external_local_ids: Optional[list[int]] = None) -> Self:
        # nested union/typevar isn't supported
        if annotype.origin_type in BASE_ANNO_TYPE_TO_PFLSTATIC_TYPE:
            res = cls(BASE_ANNO_TYPE_TO_PFLSTATIC_TYPE[annotype.origin_type])
            # set metadata directly for const value types
            if res.type == PFLExprType.NONE_TYPE:
                res.metadata = None 
            elif res.type == PFLExprType.ELLIPSIS:
                res.metadata = ... 
        elif annotype.is_type_var():
            if allow_type_var:
                res = cls(PFLExprType.GENERIC_TYPE)
            else:
                raise NotImplementedError(
                    f"TypeVar only supported in function argument and return anno, got {annotype}"
                )
        elif annotype.is_param_spec() or annotype.is_param_spec_args():
            if allow_param_spec:
                if annotype.is_param_spec():
                    res = cls(PFLExprType.GENERIC_PARAM_SPEC)
                else:
                    res = cls(PFLExprType.GENERIC_PARAM_ARGS)
            else:
                raise NotImplementedError(
                    "ParamSpec only supported in function argument anno."
                )
        elif annotype.is_number_type():
            # here we support user subclassed base type
            res = cls(PFLExprType.NUMBER)
        elif annotype.is_union_type():
            if allow_union:
                res = cls(PFLExprType.UNION, [
                    PFLExprInfo.from_annotype(
                        parse_type_may_optional_undefined(x), is_type)
                    for x in annotype.child_types
                ])
            else:
                raise NotImplementedError(
                    "Union only supported in function argument (cpp-style overload)."
                )
        elif annotype.is_list_type():
            value_anno_type = annotype.get_list_value_anno_type()
            res = cls(PFLExprType.ARRAY,
                      [PFLExprInfo.from_annotype(value_anno_type, is_type)])
        elif annotype.is_dict_type():
            value_anno_type = annotype.get_dict_value_anno_type()
            res = cls(PFLExprType.OBJECT,
                      [PFLExprInfo.from_annotype(value_anno_type, is_type)])
        elif annotype.is_dataclass_type():
            res = cls(PFLExprType.DATACLASS_TYPE if is_type else PFLExprType.
                      DATACLASS_OBJECT)
            if parse_cache is None:
                parse_cache = get_parse_cache_checked()
            item = parse_cache.cached_get_std_item(
                annotype.origin_type)
            if item is None:
                # isn't stdlib
                res.is_stdlib = False
                # TODO if marked as compilable, parse fields here.
            else:
                res.mapped = item.mapped_name
                res.is_stdlib = True
            if proxy_dcls is None:
                # we dont care fields of proxy dcls.
                res.dcls_info = PFLExprFuncInfo.from_dcls_type(
                    annotype.origin_type, delay_parse_field=True,
                    external_local_ids=external_local_ids)
                res.dcls_info.is_dcls = True 
                res.dcls_info.is_user_dcls = not res.is_stdlib
            if is_type:
                res.proxy_dcls = proxy_dcls
        elif annotype.is_tuple_type():
            res = cls(PFLExprType.TUPLE, [
                PFLExprInfo.from_annotype(
                    parse_type_may_optional_undefined(x), is_type, allow_type_var=True)
                for x in annotype.child_types
            ])
        elif annotype.is_any_type():

            res = cls(PFLExprType.ANY, [])
        elif annotype.is_callable():
            # first_child = annotype.child_types[0]
            # assert first_child is not Ellipsis, "Callable[..., ReturnType] isn't supported, use full spec or use ParamSpec."
            
            # if isinstance(first_child, ParamSpec):
            #     pass 
            raise NotImplementedError

        elif annotype.origin_type is slice:
            # we only support slice(number?).
            res = cls(PFLExprType.SLICE, [])
        else:
            mapped_item = get_parse_cache_checked(
            ).cached_get_dcls_by_mapped_type(annotype.origin_type)
            if mapped_item is None:
                raise ValueError(f"not support annotype {annotype}")
            res = cls(PFLExprType.DATACLASS_TYPE if is_type else PFLExprType.
                      DATACLASS_OBJECT)
            res.mapped = mapped_item.mapped_name
            annotype = parse_type_may_optional_undefined(mapped_item.dcls)
            res.dcls_info = PFLExprFuncInfo.from_dcls_type(
                annotype.origin_type, delay_parse_field=True,
                external_local_ids=external_local_ids)
            res.dcls_info.is_dcls = True 
            res.dcls_info.is_user_dcls = False

        res.annotype = annotype
        res.has_optional = annotype.is_optional
        res.has_undefined = annotype.is_undefined
        if not is_type and get_parse_context() is not None:
            builtin_proxy = get_parse_cache_checked().cached_try_get_proxy_dcls_by_mapped_type(annotype.origin_type)
            res.proxy_dcls = cast(Type[DataclassType], builtin_proxy.dcls) if builtin_proxy is not None else None
        return res


    @classmethod
    def from_value(cls,
                    value: Any) -> Self:
        # only support scalar values (int/float/bool/str/None) and tuple of them.
        if isinstance(value, bool):
            st = cls(PFLExprType.BOOL)
        elif isinstance(value, int):
            st = cls(PFLExprType.NUMBER)
        elif isinstance(value, float):
            st = cls(PFLExprType.NUMBER)
        elif isinstance(value, str):
            st = cls(PFLExprType.STRING)
        elif value is None:
            st = cls(PFLExprType.NONE_TYPE)
        elif value is ...:
            st = cls(PFLExprType.ELLIPSIS)
        elif isinstance(value, Undefined):
            st = cls(PFLExprType.UNDEFINED_TYPE)
        elif isinstance(value, tuple):
            st = cls(PFLExprType.TUPLE, [cls.from_value(v) for v in value])
        elif dataclasses.is_dataclass(value) and not inspect.isclass(value):
            # TODO check dcls don't have typevar
            st = cls.from_dcls_type(type(value))
        else:
            raise ValueError(f"Unsupported constant value type: {type(value)}")
        st.annotype = parse_type_may_optional_undefined(type(value))
        return st 

    @classmethod
    def from_signature(cls,
                          name: str,
                       sig: inspect.Signature,
                       is_bound_method: bool = False,
                       self_type: Optional[AnnotatedType] = None,
                       overload_sigs: Optional[list[inspect.Signature]] = None,
                       raw_func: Optional[Callable] = None,
                       compilable_meta: Optional["PFLCompileFuncMeta"] = None,
                       delay_parse_args: bool = False) -> Self:
        res_info = PFLExprFuncInfo.from_signature(name, sig, is_bound_method=is_bound_method, 
            self_type=self_type, overload_sigs=overload_sigs,
            delay_parse_args=delay_parse_args)
        if raw_func is not None:
            res_info.raw_func = raw_func
        if compilable_meta is not None:
            res_info.compilable_meta = compilable_meta
        res = cls(PFLExprType.FUNCTION, func_info=res_info)
        return res

    @classmethod
    def from_dcls_type(cls,
                       dcls: Type[DataclassType],
                       external_annos: Optional[dict[str, Any]] = None,
                       compilable_meta: Optional["PFLCompileFuncMeta"] = None,
                       delay_parse_field: bool = False,
                       external_local_ids: Optional[list[int]] = None) -> Self:
        res_info = PFLExprFuncInfo.from_dcls_type(dcls, external_annos, delay_parse_field,
            external_local_ids=external_local_ids)
        if compilable_meta is not None:
            res_info.compilable_meta = compilable_meta
        res = cls(PFLExprType.DATACLASS_TYPE, dcls_info=res_info, annotype=parse_type_may_optional_undefined(dcls))
        return res

    def is_equal_type(self, other, check_nested: bool = True):
        if not isinstance(other, PFLExprInfo):
            return False
        if self.type != other.type:
            return False
        # if self.is_optional() != other.is_optional():
        #     return False
        if self.type == PFLExprType.NUMBER:
            if self.annotype is not None and other.annotype is not None:
                self_origin_type = self.get_origin_type_checked()
                other_origin_type = other.get_origin_type_checked()
                return self_origin_type is other_origin_type
            else:
                return False
        if self.type == PFLExprType.DATACLASS_OBJECT:
            assert self.annotype is not None
            return self.get_origin_type_checked(
            ) is other.get_origin_type_checked()
        if len(self.childs) != len(other.childs):
            return False
        if check_nested:
            for i in range(len(self.childs)):
                if not self.childs[i].is_equal_type(other.childs[i]):
                    return False
        return True

    def can_cast_to_bool(self):
        # TODO custom type?
        return self.type in _TYPE_CAN_CAST_TO_BOOL

    def is_optional(self):
        return self.has_optional or self.has_undefined

    def support_bool_op(self):
        if self.type == PFLExprType.DATACLASS_OBJECT:
            op_func = inspect.getattr_static(self.get_origin_type_checked(), "__bool__", None)
            return op_func is not None
        return self.type in _TYPE_SUPPORT_BINARY_OP

    def support_binary_op(self):
        return self.type in _TYPE_SUPPORT_BINARY_OP and not self.is_optional()

    def check_support_compare_op(self, op: CompareType, other_st: Self, msg: str = ""):
        if self.type == PFLExprType.STRING and other_st.type == PFLExprType.STRING:
            assert op in [CompareType.EQUAL, CompareType.NOT_EQUAL], "only support ==/!= for string type"
            return 
        left_support = self.support_binary_op()
        right_support = other_st.support_binary_op()
        assert left_support and right_support, f"not support binary op for {self} vs {other_st}, {msg}"

    def check_support_binary_op(self, op: BinOpType, other_st: Self, msg: str = ""):
        if self.type == PFLExprType.STRING and other_st.type == PFLExprType.STRING:
            assert op == BinOpType.ADD, "only support + for string type"
            return 
        left_support = self.support_binary_op()
        right_support = other_st.support_binary_op()
        assert left_support and right_support, f"not support binary op for {self} vs {other_st}, {msg}"
    
    def check_support_unary_op(self, msg: str = ""):
        assert self.support_binary_op(), f"not support unary op for {self}, {msg}"

    def is_all_child_same(self):
        if len(self.childs) > 0:
            first = self.childs[0]
            for c in self.childs[1:]:
                if not c.is_equal_type(first):
                    return False 
            return True 
        return False

    def _get_base_number_type_priority(self, ty: Any):
        if issubclass(ty, bool):
            return 0
        elif issubclass(ty, int):
            return 1
        elif issubclass(ty, float):
            return 2
        else:
            raise NotImplementedError

    def check_support_binary_op_and_promotion(self, op: BinOpType, other: Self) -> Optional[AnnotatedType]:
        self.check_support_binary_op(op, other)
        support_prompt = [PFLExprType.NUMBER, PFLExprType.BOOL]
        if self.type in support_prompt and other.type in support_prompt:
            if self.annotype is not None and other.annotype is not None:
                self_priority = self._get_base_number_type_priority(self.annotype.origin_type if not self.annotype.is_union_type() else float)
                other_priority = self._get_base_number_type_priority(other.annotype.origin_type if not other.annotype.is_union_type() else float)
                if self_priority >= other_priority:
                    return self.annotype
                else:
                    return other.annotype

    def try_merge_two_info(self, other: Self) -> Self:
        support_merge = [PFLExprType.NUMBER, PFLExprType.BOOL]
        if self.type in support_merge and other.type in support_merge:
            if self.annotype is not None and other.annotype is not None:
                self_priority = self._get_base_number_type_priority(self.annotype.origin_type if not self.annotype.is_union_type() else float)
                other_priority = self._get_base_number_type_priority(other.annotype.origin_type if not other.annotype.is_union_type() else float)
                if self_priority >= other_priority:
                    return dataclasses.replace(self)
                else:
                    return dataclasses.replace(other)
        assert self.is_equal_type(other), f"can't merge {self} and {other}, they are not same type."
        return dataclasses.replace(self)

    def support_aug_assign(self):
        return self.type in _TYPE_SUPPORT_BINARY_OP

    def is_convertable(self, tgt: "PFLExprInfo"):
        assert self.type != PFLExprType.UNKNOWN, "source type is UNKNOWN"
        assert tgt.type != PFLExprType.UNKNOWN, "target type is UNKNOWN"
        if tgt.type == PFLExprType.ANY:
            return True
        if not tgt.is_optional() and self.is_optional():
            return False
        if tgt.type == PFLExprType.TUPLE and self.type == PFLExprType.TUPLE:
            if tgt.annotype is not None:
                if tgt.annotype.is_homogeneous:
                    return all(c.is_convertable(tgt.childs[0]) for c in self.childs)
                else:
                    if len(tgt.childs) != len(self.childs):
                        return False
                    return all(c.is_convertable(tgt_child) for c, tgt_child in zip(self.childs, tgt.childs))
        if tgt.type == PFLExprType.NUMBER or tgt.type == PFLExprType.BOOL:
            return self.type == PFLExprType.NUMBER or self.type == PFLExprType.BOOL
        elif tgt.type == PFLExprType.ARRAY or tgt.type == PFLExprType.OBJECT:
            if tgt.type == self.type:
                return self.childs[0].is_convertable(tgt.childs[0])
            return False
        elif tgt.type == PFLExprType.UNION:
            res = [self.is_convertable(tgt_child) for tgt_child in tgt.childs]
            return any(res)
        elif self.type == tgt.type == PFLExprType.DATACLASS_OBJECT:
            return issubclass(self.get_origin_type_checked(), tgt.get_origin_type_checked())
        return self.is_equal_type(tgt)

    def check_convertable(self, tgt: "PFLExprInfo", desc: str):
        if not self.is_convertable(tgt):
            raise ValueError(f"{desc} is not convertable from {self} to {tgt}")

    @property
    def metadata(self):
        if self.annotype is not None:
            var_meta = self.annotype.get_annometa(PFLVariableMeta)
            if var_meta is not None:
                return var_meta.data
        return self._metadata

    @property
    def meta_infer(self):
        if self.annotype is not None:
            var_meta = self.annotype.get_annometa(PFLVariableMeta)
            if var_meta is not None and var_meta.meta_infer is not None:
                return var_meta.meta_infer
        return self._meta_infer

    @property
    def metadata_checked(self):
        res = self.metadata
        if isinstance(res, Undefined):
            raise ValueError("metadata is not set")
        return res

    def has_metadata(self, *ty: Type[Any]):
        if not ty:
            return not isinstance(self.metadata, Undefined)
        else:
            return isinstance(self.metadata, ty)

    def has_constexpr_data(self):
        return not isinstance(self._constexpr_data, Undefined)

    def get_constexpr_checked(self, ty: Type[_T]) -> _T:
        res = self._constexpr_data
        assert isinstance(self._constexpr_data, ty), f"constexpr type {type(self._constexpr_data)} is not {ty}"
        return cast(_T, res)

    @property
    def constexpr_data_checked(self):
        res = self._constexpr_data
        if isinstance(res, Undefined):
            raise ValueError("constexpr_data is not set")
        return res

    @metadata.setter
    def metadata(self, value: Union[Undefined, Any]):
        from .pfl_ast import PFLAstNodeBase
        assert not isinstance(value, PFLAstNodeBase)
        if self.annotype is not None:
            var_meta = self.annotype.get_annometa(PFLVariableMeta)
            if var_meta is not None:
                # when metadata exists via user annassign, we skip assign.
                return
        self._metadata = value

    def get_eval_metadata_from_anno(self):
        if self.annotype is not None:
            var_meta = self.annotype.get_annometa(PFLVariableMeta)
            if var_meta is not None:
                return var_meta.data
        return None

    def get_anno_metadatas(self):
        res: list[Any] = self.anno_metadatas_ext.copy()
        if self.annotype is not None and self.annotype.annometa is not None:
            res.extend(self.annotype.annometa)
        return res

    def get_anno_metadata(self, ty: Type[_T]) -> Optional[_T]:
        candidate = self.get_anno_metadatas()
        for c in candidate:
            if isinstance(c, ty):
                return c
        return None

    def get_metadata(self, default: Any = None):
        res = self.metadata
        if isinstance(self.metadata, Undefined):
            return default 
        return res

    def get_metadata_checked(self, ty: Type[_T]) -> _T:
        res = self.metadata
        assert isinstance(self.metadata, ty), f"metadata type {type(self.metadata)} is not {ty}"
        return cast(_T, res)

    def typevar_substitution(self, typevar_map: Mapping[TypeVar, Self]) -> Self:
        # we assume new type don't contains typevar.
        if self.type == PFLExprType.GENERIC_TYPE:
            assert self.annotype is not None, "typevar must have annotype"
            if self.annotype.origin_type in typevar_map:
                new_annotype = typevar_map[self.annotype.origin_type]
                return dataclasses.replace(new_annotype)
            else:
                raise ValueError(
                    f"can't find typevar {self.annotype.origin_type} in typevar_map: {typevar_map}")
        else:
            fn_info: Optional[PFLExprFuncInfo] = None 
            if self.func_info is not None:
                fn_info = self.func_info.typevar_substitution(typevar_map)
            return dataclasses.replace(self, childs=[
                c.typevar_substitution(typevar_map) for c in self.childs
            ], func_info=fn_info)

    def shallow_copy(self) -> Self:
        # func info won't be copied to due with template dcls.
        return dataclasses.replace(self, childs=self.childs.copy())

    def _check_equal_type_with_unk_any_type_promption(self, other: Self, msg: str = "") -> Self:
        unk_or_any_types = [PFLExprType.UNKNOWN, PFLExprType.ANY]
        self_is_unk_or_any = self.type in unk_or_any_types
        other_is_unk_or_any = other.type in unk_or_any_types
        if self_is_unk_or_any:
            return other 
        elif other_is_unk_or_any:
            return self
        else:
            assert self.is_equal_type(other, check_nested=False), f"type not match: {self} vs {other}, {msg}"
            new_childs = []
            for c1, c2 in zip(self.childs, other.childs):
                new_childs.append(c1._check_equal_type_with_unk_any_type_promption(c2))
            return dataclasses.replace(self, childs=new_childs)

    def _remove_optional(self):
        # TODO deal with undefined
        if self.has_optional:
            annotype = self.annotype
            if annotype is not None:
                annotype = dataclasses.replace(annotype, is_optional=False)
            res = dataclasses.replace(self, annotype=annotype)
            res.has_optional = False
            return res
        return self

def param_fn(name: str, anno: Any, default: Any = inspect.Parameter.empty):
    # since js don't support keyword, we don't need to care about kw.
    return inspect.Parameter(name,
                             inspect.Parameter.POSITIONAL_OR_KEYWORD,
                             annotation=anno,
                             default=default)


def varparam_fn(name: str, anno: Any):
    # since js don't support keyword, we don't need to care about kw.
    return inspect.Parameter(name,
                             inspect.Parameter.VAR_POSITIONAL,
                             annotation=anno)


@dataclasses.dataclass
class PFLStdlibFuncMeta:
    # for type meta infer, e.g. calc all static ndarray shape and dtype
    # to generate cpp code.
    meta_infer: Optional[Callable[..., Optional[PFLMetaInferResult]]] = None
    # currently only at least one argument has metadata, meta_infer will be called.
    # if some argument has constexpr data, meta_infer can't be called.
    force_meta_infer: bool = False
    # used to simplify std annotation code
    take_overloads_fn: Optional[Callable] = None
    # if any stdlib func define this, we will use this to infer type instead of annotation.
    static_type_infer: Optional[Callable[..., Any]] = None
    # wben some args is constexpr in func call,
    # we can create a partial-constexpr data.
    constexpr_infer: Optional[Callable[..., Any]] = None

    
T_callable = TypeVar("T_callable", bound=Callable[..., Optional[PFLMetaInferResult]])


def mark_meta_infer(fn: Union[Callable, property]):
    if isinstance(fn, property):
        assert fn.fget is not None 
        fn_func = fn.fget
    else:
        fn_func = fn

    # if isinstance(fn, staticmethod):
    #     fn_func = fn.__func__
    # else:
    #     fn_func = fn
    def wrapper(meta_infer: T_callable) -> T_callable:
        prev_meta = getattr(fn_func, PFL_STDLIB_FUNC_META_ATTR, None)
        if prev_meta is None:
            prev_meta = PFLStdlibFuncMeta()
            setattr(fn_func, PFL_STDLIB_FUNC_META_ATTR, prev_meta)
        prev_meta.meta_infer = meta_infer
        return cast(T_callable, meta_infer)

    return wrapper

T = TypeVar("T")
T_base_callable = TypeVar("T_base_callable", bound=Callable)

BACKEND_CONFIG_REGISTRY: dict[str, PFLParseConfig] = {}

def register_backend(backend: str, config: PFLParseConfig):
    BACKEND_CONFIG_REGISTRY[backend] = config

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class PFLInlineRunEnv:
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    # if not exists, we use annotations from kwargs.
    annotations: Optional[dict[str, Any]] = None
    contexts: list[contextlib.AbstractContextManager] = dataclasses.field(default_factory=list)
    userdata: Optional[Any] = None

    def get_userdata_typed(self, ty: Type[T]) -> T:
        assert isinstance(self.userdata, ty)
        return self.userdata


@dataclasses.dataclass
class PFLCompileFuncMeta:
    # indicate a function or class (TODO) can be compiled.
    backends: Optional[list[str]] = None
    # used by debugger/simulator.
    inline_run_env_fn: Optional[Callable[..., PFLInlineRunEnv]] = None
    is_template: bool = False
    always_inline: bool = False
    userdata: Optional[Any] = None
    # there is no template variable in python, so we can use this to indicate
    # args that can be constexpr (not reqiured).
    # if set, is_template must be True.
    # for dataclass, this indicates fields that can be constexpr (TODO unused for now).
    constexpr_args: Optional[set[str]] = None
    def need_delayed_processing(self):
        return self.is_template or self.always_inline

    def __post_init__(self):
        return self.validate()

    def validate(self):
        # if self.constexpr_args is not None:
        #     assert self.is_template, "constexpr_args can only be set when is_template is True"
        pass


@dataclasses.dataclass
class PFLCompileReq:
    func_or_dcls: Callable
    uid: str
    meta: PFLCompileFuncMeta
    info: Optional[PFLExprFuncInfo] = None
    args_from_call: Optional[tuple[list["PFLExprInfo"], dict[str, "PFLExprInfo"]]] = None
    self_type: Optional[Union[PFLExprInfo, AnnotatedType]] = None
    is_prop: bool = False 
    is_method_def: bool = False
    # jit compile may need this.
    external_anno: Optional[tuple[dict[str, Any], Any]] = None
    # bounded object if func is a method.
    # usually used for meta-programming (use self.xxx as constant.)
    bound_self: Optional[Any] = None
    # if some arguments are constexpr, we will use this to inline expand if.
    constexpr_args: Optional[dict[str, Any]] = None
    is_dcls: bool = False
    dcls_infer_field_type: bool = False
    local_ids: Optional[list[int]] = None

    def __post_init__(self):
        if self.is_method_def:
            assert self.self_type is not None 

    def __repr__(self):
        if self.is_dcls:
            prefix = f"[dcls]"
            if self.meta.is_template:
                prefix += "[template]"
            return f"{prefix} {self.uid}"
        else:
            prefix = "[func]"
            if self.is_method_def:
                prefix = "[method]"
            if self.bound_self is not None:
                prefix = "[bound-method]"
            if self.meta.is_template:
                prefix += "[template]"
            elif self.meta.always_inline:
                prefix += "[always-inline]"
            return f"{prefix} {self.uid}"

    def get_func_compile_uid(self, delayed_info: Optional[PFLExprFuncInfo] = None) -> str:
        local_ids = self.local_ids if self.local_ids is not None else []
        local_ids_strs = []
        if local_ids:
            local_ids_strs = ["-".join([str(i) for i in local_ids])]
        if self.is_dcls:
            flag = 0
            desc = "cls"
            if self.meta.is_template:
                flag |= PFLCompilableFlags.IS_TEMPLATE
                assert self.info is not None 
                desc = str(self.info)
            return UniqueTreeId.from_parts([self.uid, str(PFLCompilableType.DATACLASS), str(flag), desc, *local_ids_strs]).uid_encoded
        if self.bound_self is None and not self.meta.is_template:
            return UniqueTreeId.from_parts([self.uid, str(PFLCompilableType.FUNCTION), str(0), "fn", *local_ids_strs]).uid_encoded
        if self.bound_self is not None:
            flag = PFLCompilableFlags.IS_BOUND
            assert not self.meta.is_template, "currently bound method can't be template"
            return UniqueTreeId.from_parts([self.uid, str(PFLCompilableType.FUNCTION), str(flag), f"fn-{hex(id(self.bound_self))}", *local_ids_strs]).uid_encoded
        else:
            # use signature as template func uid
            if delayed_info is not None:
                info = delayed_info
            else:
                assert self.info is not None 
                info = self.info
            flag = PFLCompilableFlags.IS_TEMPLATE
            return UniqueTreeId.from_parts([self.uid, str(PFLCompilableType.FUNCTION), str(flag), f"fn-{info}", *local_ids_strs]).uid_encoded

    def is_bound_method(self) -> bool:
        return self.bound_self is not None

@overload
def mark_pfl_compilable(fn: T) -> T: ...

@overload
def mark_pfl_compilable(fn: None = None, *, backends: Optional[list[str]] = None, 
        inline_run_env_fn: Optional[Callable[[], PFLInlineRunEnv]] = None, is_template: bool = False, 
        always_inline: bool = False, meta: Optional[PFLCompileFuncMeta] = None,
        constexpr_args: Optional[Sequence[str]] = None,
        userdata: Optional[dict[str, Any]] = None) -> Callable[[T], T]: ...

@register_pfl_std(mapped_name="compiler_mark_pfl_compilable", backend=None, _internal_disable_type_check=True)
def mark_pfl_compilable(fn: Optional[T] = None, *, backends: Optional[list[str]] = None, 
        inline_run_env_fn: Optional[Callable[[], PFLInlineRunEnv]] = None, is_template: bool = False, 
        always_inline: bool = False, meta: Optional[PFLCompileFuncMeta] = None,
        constexpr_args: Optional[Sequence[str]] = None,
        userdata: Optional[dict[str, Any]] = None) -> Union[T, Callable[[T], T]]:
    def wrapper(fn_wrapped: T) -> T:
        prev_meta: Optional[PFLCompileFuncMeta] = getattr(fn_wrapped, PFL_COMPILE_META_ATTR, None)
        constexpr_args_ = constexpr_args
        if constexpr_args_ is not None:
            constexpr_args_set = set(constexpr_args_)
        else:
            constexpr_args_set = None
        if meta is not None:
            setattr(fn_wrapped, PFL_COMPILE_META_ATTR, meta)
        else:
            if prev_meta is None:
                prev_meta = PFLCompileFuncMeta(backends, inline_run_env_fn, is_template=is_template, always_inline=always_inline,
                    constexpr_args=constexpr_args_set, userdata=userdata)
                setattr(fn_wrapped, PFL_COMPILE_META_ATTR, prev_meta)
            else:
                prev_meta.backends = backends
                prev_meta.inline_run_env_fn = inline_run_env_fn
                prev_meta.is_template = is_template
                prev_meta.always_inline = always_inline
                prev_meta.constexpr_args = constexpr_args_set
                prev_meta.userdata = userdata
                prev_meta.validate()
        return cast(T, fn_wrapped)
    if fn is None:
        return wrapper
    else:
        return wrapper(fn)

def get_compilable_meta(fn: Callable) -> Optional[PFLCompileFuncMeta]:
    meta: Optional[PFLCompileFuncMeta] = getattr(fn, PFL_COMPILE_META_ATTR, None)
    if meta is None:
        return None
    return meta


def configure_std_func(*, take_overloads_fn: Optional[Callable] = None, meta_infer: Optional[Callable[..., Optional[PFLMetaInferResult]]] = None,
                       static_type_infer: Optional[Callable[..., Any]] = None,
                       force_meta_infer: bool = False,
                       constexpr_infer: Optional[Callable[..., Any]] = None) -> Callable[[T_base_callable], T_base_callable]:
    def wrapper(fn_wrapped: T_base_callable) -> T_base_callable:
        fn_unwrapped = unwrap_fn_static_cls_property(fn_wrapped)

        take_overloads_fn_ = take_overloads_fn
        if take_overloads_fn_ is not None:
            take_overloads_fn_ = unwrap_fn_static_cls_property(take_overloads_fn_)
        prev_meta: Optional[PFLStdlibFuncMeta] = getattr(fn_unwrapped, PFL_FUNC_ANNO_META_ATTR, None)
        if meta_infer is not None:

            meta_infer_set_first_arg = partial(meta_infer, fn_unwrapped)
        else:
            meta_infer_set_first_arg = None
        if prev_meta is None:
            prev_meta = PFLStdlibFuncMeta(take_overloads_fn=take_overloads_fn, meta_infer=meta_infer_set_first_arg,
                                          static_type_infer=static_type_infer, force_meta_infer=force_meta_infer,
                                          constexpr_infer=constexpr_infer)
            setattr(fn_wrapped, PFL_STDLIB_FUNC_META_ATTR, prev_meta)
        else:
            if take_overloads_fn_ is not None:
                prev_meta.take_overloads_fn = take_overloads_fn_
            if meta_infer_set_first_arg is not None:
                prev_meta.meta_infer = meta_infer_set_first_arg
            if static_type_infer is not None:
                prev_meta.static_type_infer = static_type_infer
            if constexpr_infer is not None:
                prev_meta.constexpr_infer = constexpr_infer
            prev_meta.force_meta_infer = force_meta_infer

        return cast(T_base_callable, fn_wrapped)
    return wrapper

def evaluate_annotation_expr(annotation: Union[ast.expr, str]):
    if not isinstance(annotation, str):
        ann_str = ast.unparse(annotation)
    else:
        ann_str = annotation
    ann_fref = ForwardRef(ann_str,
                            is_argument=True,
                            is_class=False)
    if sys.version_info < (3, 9):
        ann_res = ann_fref._evaluate(
            get_parse_context_checked().anno_evaluate_globals,
            {})
    else:
        ann_res = ann_fref._evaluate(
            get_parse_context_checked().anno_evaluate_globals,
            {},
            recursive_guard=set())  # type: ignore
    return ann_res

class PFLCompileConstantType(enum.IntEnum):
    BUILTIN_VALUE = 0
    # global (qual) function, maybe stdlib
    FUNCTION = 1
    # dataclass, maybe stdlib
    DATACLASS_TYPE = 2
    # global value, must be dataclass instalce
    GLOBAL_VALUE = 3

@dataclasses.dataclass
class PFLProcessedVarMeta:
    value: Any
    is_static: bool = False 
    is_classmethod: bool = False
    is_property: bool = False
    compilable_meta: Optional[PFLCompileFuncMeta] = None


@dataclasses.dataclass
class PFLCompileConstant:
    type: PFLCompileConstantType
    value: Any
    var_meta: PFLProcessedVarMeta
    is_stdlib: bool = False 
    bound_self: Optional[Any] = None


class PFLCompilableType(enum.IntEnum):
    FUNCTION = 0
    DATACLASS = 1

class PFLCompilableFlags(enum.IntFlag):
    IS_BOUND = enum.auto()
    IS_TEMPLATE = enum.auto()
