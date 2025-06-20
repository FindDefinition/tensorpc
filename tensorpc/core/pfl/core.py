import ast
from collections.abc import Mapping
import contextlib
import contextvars
import enum
from functools import partial
import inspect
import sys
from typing import (TYPE_CHECKING, Any, Callable, ForwardRef, Optional, Type, TypeVar, Union,
                    cast, overload)

from typing_extensions import Literal, Self, get_overloads

import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.annolib import (AnnotatedType, DataclassType, T_dataclass,
                                   Undefined,
                                   parse_type_may_optional_undefined,
                                   undefined)
from tensorpc.core.inspecttools import unwrap_fn_static_cls_property
from tensorpc.core.pfl.constants import PFL_COMPILE_META_ATTR, PFL_STDLIB_FUNC_META_ATTR, PFL_FUNC_ANNO_META_ATTR
from tensorpc.core.moduleid import get_qualname_of_type

from .pfl_reg import STD_REGISTRY, StdRegistryItem, register_pfl_std

from tensorpc.utils.rich_logging import get_logger
if TYPE_CHECKING:
    from .pfl_ast import PFLFunc

PFL_LOGGER = get_logger("tensorpc.pfl")

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
    def __init__(self, backend: str, temp_std_items: Optional[dict[Type, StdRegistryItem]] = None):
        self._func_parse_result_cache: dict[Callable, PFLExprInfo] = {}
        self._annotype_cache: dict[Any, AnnotatedType] = {}
        self._std_item_cache: dict[Type, StdRegistryItem] = {}
        if temp_std_items is not None:
            self._std_item_cache.update(temp_std_items)

        self._mapped_type_cache: dict[Type, StdRegistryItem] = {}
        self._temp_dcls_dict: dict[tuple[str, Optional[str]], StdRegistryItem] = {}
        self._backend = backend

    def cached_parse_to_annotype(self,
                          type: Any) -> AnnotatedType:
        if type in self._annotype_cache:
            return self._annotype_cache[type]
        res = parse_type_may_optional_undefined(type)
        self._annotype_cache[type] = res
        return res

    def cached_parse_func(self,
                          func: Callable,
                          ignore_self: bool = False,
                          self_type: Optional[AnnotatedType] = None,
                          disable_type_check: bool = False) -> "PFLExprInfo":
        if func in self._func_parse_result_cache:
            return self._func_parse_result_cache[func]
        if disable_type_check:
            # use (...Any) -> Any sig
            sig = inspect.Signature([varparam_fn("x", Any)], return_annotation=Any)
            return PFLExprInfo.from_signature(sig, raw_func=func)
        else:
            meta: Optional[PFLStdlibFuncMeta] = getattr(func, PFL_STDLIB_FUNC_META_ATTR, None)
            if meta is not None and meta.take_overloads_fn is not None:
                overload_fn = meta.take_overloads_fn
            else:
                overload_fn = func
            overloads = get_overloads(overload_fn)
            if overloads:
                sig = inspect.signature(overloads[0])
                overload_sigs = [inspect.signature(o) for o in overloads[1:]]
            else:
                sig = inspect.signature(func)
                overload_sigs = None
            return PFLExprInfo.from_signature(sig, ignore_self=ignore_self, self_type=self_type, overload_sigs=overload_sigs, raw_func=func)

    def cached_parse_std_item(self, item: StdRegistryItem) -> "PFLExprInfo":
        if item.is_func:
            return self.cached_parse_func(item.dcls, disable_type_check=item._internal_disable_type_check)
        else:
            return PFLExprInfo.from_annotype(
                parse_type_may_optional_undefined(item.dcls), is_type=True, parse_cache=self)

    def cached_get_std_item(self, dcls: Type[T_dataclass]):
        if dcls in self._std_item_cache:
            return self._std_item_cache[dcls]
        item = STD_REGISTRY.get_item_by_dcls(dcls, self._backend, external=self._temp_dcls_dict)
        if item is None:
            raise ValueError(
                f"can't find your type {get_qualname_of_type(dcls)} from std registry."
            )
        self._std_item_cache[dcls] = item
        return item

    def cached_get_dcls_by_mapped_type(self, usercls: Any):
        if usercls in self._mapped_type_cache:
            return self._mapped_type_cache[usercls]
        item = STD_REGISTRY.get_dcls_item_by_mapped_type(
            usercls, self._backend, external=self._temp_dcls_dict)
        if item is None:
            raise ValueError(
                f"can't find your mapped type {get_qualname_of_type(usercls)} from std registry."
            )
        self._mapped_type_cache[usercls] = item
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
                 backend: str = "js",
                 temp_std_items: Optional[dict[Type, StdRegistryItem]] = None,
                 cfg: Optional[PFLParseConfig] = None,
                 eval_cfg: Optional[StaticEvalConfig] = None,
                 node_to_std_item: Optional[dict[ast.AST, StdRegistryItem]] = None,
                 node_to_compilable: Optional[dict[ast.AST, "PFLCompilable"]] = None):
        super().__init__(lines)
        # local states
        self.anno_evaluate_globals = func_globals
        if node_to_std_item is None:
            node_to_std_item = {}
        self.node_to_std_item = node_to_std_item
        if node_to_compilable is None:
            node_to_compilable = {}
        self.node_to_compilable = node_to_compilable
        self.depend_compilables: list[str] = []
        # global states
        self._backend = backend
        self.cache = PFLParseCache(backend, temp_std_items)
        if cfg is None:
            cfg = PFLParseConfig()
        self.cfg = cfg
        if eval_cfg is None:
            eval_cfg = StaticEvalConfig()
        self.eval_cfg = eval_cfg
        self._disable_type_check: bool = False

    @classmethod 
    def from_outer_ctx(cls, ctx: Self, lines: list[str], func_globals: Any, 
                node_to_std_item: Optional[dict[ast.AST, StdRegistryItem]] = None,
                 node_to_compilable: Optional[dict[ast.AST, Any]] = None):
        assert not ctx._disable_type_check, "not supported inside decorator list"
        new_ctx = cls(lines, func_globals, ctx._backend, cfg=ctx.cfg, 
            eval_cfg=ctx.eval_cfg, node_to_std_item=node_to_std_item,
            node_to_compilable=node_to_compilable)
        new_ctx.cache = ctx.cache
        return new_ctx

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

def get_eval_cfg_in_parse_ctx():
    ctx = _PFLPARSE_CONTEXT.get()
    if ctx is None:
        return None 
    return ctx.eval_cfg

def has_parse_context():
    ctx = _PFLPARSE_CONTEXT.get()
    return ctx is not None


@dataclasses.dataclass(eq=False)
class PFLExprInfo:
    type: PFLExprType
    childs: list['PFLExprInfo'] = dataclasses.field(default_factory=list)
    has_optional: bool = False
    has_undefined: bool = False
    # for custom dataclass
    mapped: str = ""
    # for container and dataclass
    annotype: Optional[AnnotatedType] = None
    anno_metadatas_ext: list[Any] = dataclasses.field(default_factory=list)
    # for function
    arg_name: Optional[str] = None
    return_type: Optional["PFLExprInfo"] = None
    default: Union[Undefined, Any] = undefined
    is_vaargs: bool = False
    is_method: bool = False
    is_property: bool = False
    raw_func: Optional[Callable] = None
    # indicate this function is a compiled function, not stdlib function.
    # we need to get inside when evaluation
    compiled_uid: Optional[str] = None
    # overload: when you define a function with overloads, the signature of origin function
    # will be ignored, and the overloads will be used instead.
    # keep in mind that if your f has three overloads, the first overload will be saved in main PFLExprInfo.
    # other overloads will be saved in `overloads`.
    overloads: Optional[list["PFLExprInfo"]] = None
    # for dataclass in function arg
    is_temp: bool = False
    # for meta call (type validation, shape inference, etc)
    # TODO should it be sent to frontend?
    _metadata: Union[Undefined, Any] = undefined
    _meta_infer: Optional[Callable[..., Optional[PFLMetaInferResult]]] = None
    def to_dict(self):
        childs = [c.to_dict() for c in self.childs]
        res: dict[str, Any] = {
            "type": self.type,
        }
        if self.childs:
            res["childs"] = childs
        if self.has_optional:
            res["has_optional"] = self.has_optional
        if self.has_undefined:
            res["has_undefined"] = self.has_undefined
        if self.is_vaargs:
            res["is_vaargs"] = self.is_vaargs
        if self.mapped:
            res["mapped"] = self.mapped
        if not isinstance(self.metadata, Undefined):
            res["metadata"] = self.metadata
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
            res = str(self.annotype.origin_type.__name__) + "()"
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
            args_str = []
            for arg in self.childs:
                if arg.is_vaargs:
                    args_str.append(f"...{arg}")
                else:
                    args_str.append(str(arg))
            args = ", ".join(args_str)
            res = f"({args}) => {self.return_type}"
        else:
            res = _BASE_TYPE_TO_STRING[self.type]
        if self.mapped:
            res += f"<{self.mapped}>"

        if self.has_optional:
            res += " | null"
        if self.has_undefined:
            res += " | undefined"
        return res

    def get_origin_type_checked(self):
        assert self.annotype is not None
        return self.annotype.origin_type

    @classmethod
    def from_annotype(cls,
                      annotype: AnnotatedType,
                      is_type: bool = False,
                      allow_union: bool = False,
                      allow_type_var: bool = False,
                      parse_cache: Optional[PFLParseCache] = None) -> Self:
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
                    "Union only supported in function argument and return anno."
                )
        elif annotype.is_number_type():
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
            res.mapped = item.mapped_name
            res.is_temp = item.is_temp
        elif annotype.is_tuple_type():
            res = cls(PFLExprType.TUPLE, [
                PFLExprInfo.from_annotype(
                    parse_type_may_optional_undefined(x), is_type)
                for x in annotype.child_types
            ])
        elif annotype.is_any_type():
            res = cls(PFLExprType.ANY, [])
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

        res.annotype = annotype
        res.has_optional = annotype.is_optional
        res.has_undefined = annotype.is_undefined
        return res

    @classmethod
    def from_signature(cls,
                       sig: inspect.Signature,
                       ignore_self: bool = False,
                       self_type: Optional[AnnotatedType] = None,
                       overload_sigs: Optional[list[inspect.Signature]] = None,
                       raw_func: Optional[Callable] = None) -> Self:
        res = cls(PFLExprType.FUNCTION)
        res.raw_func = raw_func
        if overload_sigs is not None:
            res.overloads = [PFLExprInfo.from_signature(
                s, ignore_self=ignore_self, self_type=self_type) for s in overload_sigs]
        cnt = 0
        for param in sig.parameters.values():
            if ignore_self and cnt == 0 and param.name == "self":
                res.is_method = True
                continue
            assert param.annotation is not inspect.Parameter.empty, f"param {param.name} must have annotation"
            if param.annotation is Self:
                assert self_type is not None 
                annotype = self_type
            else:
                annotype = parse_type_may_optional_undefined(param.annotation, self_type=self_type)
            
            arg = PFLExprInfo.from_annotype(annotype,
                                            is_type=False,
                                            allow_union=True,
                                            allow_type_var=True)
            res.childs.append(arg)
            if param.default is not inspect.Parameter.empty:
                arg.default = param.default
            arg.is_vaargs = param.kind == inspect.Parameter.VAR_POSITIONAL
            arg.arg_name = param.name
            cnt += 1
        if sig.return_annotation is not inspect.Parameter.empty:
            if sig.return_annotation is not None:
                if sig.return_annotation is Self:
                    assert self_type is not None 
                    annotype = self_type 
                else:
                    ret_anno = sig.return_annotation
                    annotype = parse_type_may_optional_undefined(
                        ret_anno, self_type=self_type)
                res.return_type = PFLExprInfo.from_annotype(annotype,
                                                            is_type=False,
                                                            allow_type_var=True)
            else:
                res.return_type = PFLExprInfo(PFLExprType.NONE_TYPE)
        return res

    # def __eq__(self, other):
    #     assert isinstance(other, PFLExprInfo)
    #     return self.is_equal_type(other)

    # def __ne__(self, other):
    #     assert isinstance(other, PFLExprInfo)
    #     return other.is_equal_type(self)

    def is_equal_type(self, other):
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

    def check_support_binary_op(self, msg: str = ""):
        assert self.support_binary_op(
        ), f"not support binary op for {self.type}, {msg}"

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

    def check_support_binary_op_and_promotion(self, other: Self) -> Optional[AnnotatedType]:
        self.check_support_binary_op()
        other.check_support_binary_op()
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
        assert self == other, f"can't merge {self} and {other}, they are not same type."
        return dataclasses.replace(self)

    def check_support_compare_op(self, msg: str = ""):
        assert self.support_binary_op(
        ), f"not support binary op for {self.type}, {msg}"

    def support_aug_assign(self):
        return self.type in _TYPE_SUPPORT_BINARY_OP

    def is_convertable(self, tgt: "PFLExprInfo"):
        if not tgt.is_optional() and self.is_optional():
            return False
        if tgt.type == PFLExprType.ANY:
            return True
        if tgt.type == PFLExprType.NUMBER or tgt.type == PFLExprType.BOOL:
            return self.type == PFLExprType.NUMBER or self.type == PFLExprType.BOOL
        elif tgt.type == PFLExprType.ARRAY or tgt.type == PFLExprType.OBJECT:
            if tgt.type == self.type:
                return self.childs[0].is_convertable(tgt.childs[0])
            return False
        elif tgt.type == PFLExprType.UNION:
            res = [self.is_convertable(tgt_child) for tgt_child in tgt.childs]
            return any(res)
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

    @metadata.setter
    def metadata(self, value: Union[Undefined, Any]):
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
            ret = self.return_type 
            if ret is not None:
                ret = ret.typevar_substitution(typevar_map)
            return dataclasses.replace(self, childs=[
                c.typevar_substitution(typevar_map) for c in self.childs
            ], return_type=ret)

    

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
    # used to simplify std annotation code
    take_overloads_fn: Optional[Callable] = None

    
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
    inline_run_env_fn: Optional[Callable[[], PFLInlineRunEnv]] = None
    is_template: bool = False
    # anno_transform: (infered_anno, original_anno) -> new_anno
    # used for convert third-party annotations such as tl.constexpr in triton.jit
    anno_transform: Optional[Callable[[PFLExprInfo, Any], PFLExprInfo]] = None 

@dataclasses.dataclass
class PFLCompilable:
    func: Callable
    uid: str
    meta: PFLCompileFuncMeta



@overload
def mark_pfl_compilable(fn: T) -> T: ...

@overload
def mark_pfl_compilable(fn: None = None, *, backends: Optional[list[str]] = None, 
        inline_run_env_fn: Optional[Callable[[], PFLInlineRunEnv]] = None, is_template: bool = False, 
        meta: Optional[PFLCompileFuncMeta] = None) -> Callable[[T], T]: ...

@register_pfl_std(mapped_name="compiler_mark_pfl_compilable", backend=None, _internal_disable_type_check=True)
def mark_pfl_compilable(fn: Optional[T] = None, *, backends: Optional[list[str]] = None, 
        inline_run_env_fn: Optional[Callable[[], PFLInlineRunEnv]] = None, is_template: bool = False, 
        meta: Optional[PFLCompileFuncMeta] = None) -> Union[T, Callable[[T], T]]:
    def wrapper(fn_wrapped: T) -> T:
        prev_meta: Optional[PFLCompileFuncMeta] = getattr(fn_wrapped, PFL_COMPILE_META_ATTR, None)
        if meta is not None:
            setattr(fn_wrapped, PFL_COMPILE_META_ATTR, meta)
        else:
            if prev_meta is None:
                prev_meta = PFLCompileFuncMeta(backends, inline_run_env_fn, is_template=is_template)
                setattr(fn_wrapped, PFL_COMPILE_META_ATTR, prev_meta)
            else:
                prev_meta.backends = backends
                prev_meta.inline_run_env_fn = inline_run_env_fn
                prev_meta.is_template = is_template
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

def configure_std_func(*, take_overloads_fn: Optional[Callable] = None, meta_infer: Optional[Callable[..., Optional[PFLMetaInferResult]]] = None) -> Callable[[T_base_callable], T_base_callable]:
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
            prev_meta = PFLStdlibFuncMeta(take_overloads_fn=take_overloads_fn, meta_infer=meta_infer_set_first_arg)
            setattr(fn_wrapped, PFL_STDLIB_FUNC_META_ATTR, prev_meta)
        else:
            if take_overloads_fn_ is not None:
                prev_meta.take_overloads_fn = take_overloads_fn_
            if meta_infer_set_first_arg is not None:
                prev_meta.meta_infer = meta_infer_set_first_arg
        return cast(T_base_callable, fn_wrapped)
    return wrapper
