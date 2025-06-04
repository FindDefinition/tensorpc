import ast
import contextlib
import contextvars
import enum
import inspect
import sys
from typing import (Any, Callable, ForwardRef, Optional, Type, TypeVar, Union,
                    cast)

from typing_extensions import Self

import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.annolib import (AnnotatedType, DataclassType, T_dataclass,
                                   Undefined,
                                   parse_type_may_optional_undefined,
                                   undefined)
from tensorpc.core.pfl.constants import PFL_FUNC_META_ATTR
from tensorpc.core.pfl.typemetas import PFLVariableMeta
from tensorpc.core.moduleid import get_qualname_of_type

from .pfl_reg import STD_REGISTRY, StdRegistryItem


@dataclasses.dataclass
class PFLMetaInferResult:
    data: Any


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


_BASE_TYPE_TO_STRING = {
    PFLExprType.UNKNOWN: "unknown",
    PFLExprType.NUMBER: "number",
    PFLExprType.BOOL: "bool",
    PFLExprType.STRING: "string",
    PFLExprType.NONE_TYPE: "null",
    PFLExprType.UNDEFINED_TYPE: "undefined",
    PFLExprType.ANY: "any",
    PFLExprType.RANGE: "range",
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
    Undefined: PFLExprType.UNDEFINED_TYPE,
    range: PFLExprType.RANGE,
}


class PFLParseContext:

    def __init__(self,
                 lines: list[str],
                 func_globals: Any,
                 backend: str = "js",
                 temp_std_items: Optional[dict[Type, StdRegistryItem]] = None):
        self.lines = lines

        self.error_node: Optional[ast.AST] = None
        self._backend = backend
        self.anno_evaluate_globals = func_globals

        self._std_item_cache: dict[Type, StdRegistryItem] = {}
        if temp_std_items is not None:
            self._std_item_cache.update(temp_std_items)
        self._func_parse_result_cache: dict[Callable, PFLExprInfo] = {}
        self._mapped_type_cache: dict[Type, StdRegistryItem] = {}

    def format_error_from_lines_node(self, node: ast.AST):
        if hasattr(node, "lineno") and hasattr(node, "col_offset"):
            lineno = node.lineno  # type: ignore
            col_offset = node.col_offset  # type: ignore
            if lineno > 0 and lineno <= len(self.lines):
                line = self.lines[lineno - 1]
                return f"{line}\n{' ' * col_offset}^"
        return ""

    def cached_parse_func(self,
                          func: Callable,
                          ignore_self: bool = False) -> "PFLExprInfo":
        if func in self._func_parse_result_cache:
            return self._func_parse_result_cache[func]
        sig = inspect.signature(func)
        return PFLExprInfo.from_signature(sig, ignore_self=ignore_self)

    def cached_get_std_item(self, dcls: Type[T_dataclass]):
        if dcls in self._std_item_cache:
            return self._std_item_cache[dcls]
        item = STD_REGISTRY.get_item_by_dcls(dcls, self._backend)
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
            usercls, self._backend)
        if item is None:
            raise ValueError(
                f"can't find your mapped type {get_qualname_of_type(usercls)} from std registry."
            )
        self._mapped_type_cache[usercls] = item
        return item


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


@dataclasses.dataclass
class PFLExprInfo:
    type: PFLExprType
    childs: list['PFLExprInfo'] = dataclasses.field(default_factory=list)
    has_optional: bool = False
    has_undefined: bool = False
    # for custom dataclass
    mapped: str = ""
    # for container and dataclass
    annotype: Optional[AnnotatedType] = None
    # for function
    return_type: Optional["PFLExprInfo"] = None
    default: Union[Undefined, Any] = undefined
    is_vaargs: bool = False
    is_method: bool = False
    # for dataclass in function arg
    is_temp: bool = False
    # for meta call (type validation, shape inference, etc)
    # TODO should it be sent to frontend?
    _metadata: Union[Undefined, Any] = undefined
    meta_infer: Optional[Callable] = None

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
            res = f"[{', '.join(child_reprs)}]"
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
                      allow_union: bool = False) -> Self:
        if annotype.origin_type in BASE_ANNO_TYPE_TO_PFLSTATIC_TYPE:
            res = cls(BASE_ANNO_TYPE_TO_PFLSTATIC_TYPE[annotype.origin_type])
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
            item = get_parse_context_checked().cached_get_std_item(
                annotype.origin_type)
            res.mapped = item.mapped_name
            res.is_temp = item.is_temp
        elif annotype.is_tuple_type():
            # TODO add real tuple support
            first_child_type = annotype.child_types[0]
            assert all(
                c == first_child_type
                for c in annotype.child_types), "tuple must be same type"
            res = cls(PFLExprType.ARRAY, [
                PFLExprInfo.from_annotype(
                    parse_type_may_optional_undefined(first_child_type),
                    is_type)
            ])
        elif annotype.is_any_type():
            res = cls(PFLExprType.ANY, [])
        else:
            mapped_item = get_parse_context_checked(
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
                       ignore_self: bool = False) -> Self:
        res = cls(PFLExprType.FUNCTION)
        cnt = 0
        for param in sig.parameters.values():
            if ignore_self and cnt == 0 and param.name == "self":
                res.is_method = True
                continue
            assert param.annotation is not inspect.Parameter.empty, f"param {param.name} must have annotation"
            annotype = parse_type_may_optional_undefined(param.annotation)
            arg = PFLExprInfo.from_annotype(annotype,
                                            is_type=False,
                                            allow_union=True)
            res.childs.append(arg)
            if param.default is not inspect.Parameter.empty:
                arg.default = param.default
            arg.is_vaargs = param.kind == inspect.Parameter.VAR_POSITIONAL
            cnt += 1
        if sig.return_annotation is not inspect.Parameter.empty:
            if sig.return_annotation is not None:
                annotype = parse_type_may_optional_undefined(
                    sig.return_annotation)
                res.return_type = PFLExprInfo.from_annotype(annotype,
                                                            is_type=False)
            else:
                res.return_type = PFLExprInfo(PFLExprType.NONE_TYPE)
        return res

    def __eq__(self, other):
        if not isinstance(other, PFLExprInfo):
            return False
        if other.type == PFLExprType.ANY or self.type == PFLExprType.ANY:
            return True
        if self.type != other.type:
            return False
        if self.type == PFLExprType.DATACLASS_OBJECT:
            assert self.annotype is not None
            return self.get_origin_type_checked(
            ) is other.get_origin_type_checked()
        if len(self.childs) != len(other.childs):
            return False
        for i in range(len(self.childs)):
            if self.childs[i] != other.childs[i]:
                return False
        return True

    def can_cast_to_bool(self):
        return self.type in _TYPE_CAN_CAST_TO_BOOL

    def is_optional(self):
        return self.has_optional or self.has_undefined

    def support_bool_op(self):
        return self.type in _TYPE_SUPPORT_BINARY_OP

    def support_binary_op(self):
        return self.type in _TYPE_SUPPORT_BINARY_OP and not self.is_optional()

    def check_support_binary_op(self, msg: str = ""):
        assert self.support_binary_op(
        ), f"not support binary op for {self.type}, {msg}"

    def check_support_compare_op(self, msg: str = ""):
        assert self.support_binary_op(
        ), f"not support binary op for {self.type}, {msg}"

    def support_aug_assign(self):
        return self.type in _TYPE_SUPPORT_BINARY_OP

    def is_convertable(self, tgt: "PFLExprInfo"):
        if tgt.type == PFLExprType.NUMBER or tgt.type == PFLExprType.BOOL:
            return self.type == PFLExprType.NUMBER or self.type == PFLExprType.BOOL
        elif tgt.type == PFLExprType.ARRAY or tgt.type == PFLExprType.OBJECT:
            if tgt.type == self.type:
                return self.childs[0].is_convertable(tgt.childs[0])
            return False
        elif tgt.type == PFLExprType.UNION:
            res = [self.is_convertable(tgt_child) for tgt_child in tgt.childs]
            return any(res)
        return self == tgt

    def check_convertable(self, tgt: "PFLExprInfo", desc: str):
        assert self.is_convertable(
            tgt), f"{desc} is not convertable from {self} to {tgt}"

    @property
    def metadata(self):
        if self.annotype is not None:
            var_meta = self.annotype.get_annometa(PFLVariableMeta)
            if var_meta is not None:
                return var_meta.data
        return self._metadata

    @property
    def metadata_checked(self):
        res = self.metadata
        if isinstance(res, Undefined):
            raise ValueError("metadata is not set")
        return res

    def has_metadata(self):
        return not isinstance(self.metadata, Undefined)

    @metadata.setter
    def metadata(self, value: Union[Undefined, Any]):
        if self.annotype is not None:
            var_meta = self.annotype.get_annometa(PFLVariableMeta)
            if var_meta is not None:
                # when metadata exists via user annassign, we skip assign.
                return
        self._metadata = value

    def get_metadata_from_anno(self):
        if self.annotype is not None:
            var_meta = self.annotype.get_annometa(PFLVariableMeta)
            if var_meta is not None:
                return var_meta.data
        return None


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
class PFLFuncMeta:
    # for type meta infer, e.g. calc all static ndarray shape and dtype
    # to generate cpp code.
    meta_infer: Optional[Callable[..., PFLMetaInferResult]] = None
    meta_assign_check: Optional[Callable[[PFLExprInfo, PFLExprInfo],
                                         PFLMetaInferResult]] = None


T_callable = TypeVar("T_callable", bound=Callable)


def register_meta_infer(fn: Callable):
    fn_func = fn

    # if isinstance(fn, staticmethod):
    #     fn_func = fn.__func__
    # else:
    #     fn_func = fn
    def wrapper(meta_infer: T_callable) -> T_callable:
        prev_meta = getattr(fn_func, PFL_FUNC_META_ATTR, None)
        if prev_meta is None:
            prev_meta = PFLFuncMeta()
            setattr(fn_func, PFL_FUNC_META_ATTR, prev_meta)
        prev_meta.meta_infer = meta_infer
        return cast(T_callable, meta_infer)

    return wrapper


def register_meta_assign_check(dcls: Type[DataclassType]):

    def wrapper(meta_infer: T_callable) -> T_callable:
        prev_meta = getattr(dcls, PFL_FUNC_META_ATTR, None)
        if prev_meta is None:
            prev_meta = PFLFuncMeta()
            setattr(dcls, PFL_FUNC_META_ATTR, prev_meta)
        prev_meta.meta_infer = meta_infer
        return cast(T_callable, meta_infer)

    return wrapper
