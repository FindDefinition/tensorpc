import ast
import bisect
from collections.abc import Sequence
import enum
import inspect
import traceback
from typing import Any, Callable, ClassVar, Generic, Optional, Type, TypeAlias, TypeVar, Union, cast
from typing_extensions import TypeVarTuple, Unpack
import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core import inspecttools
from tensorpc.core.annolib import (AnnotatedType, Undefined, is_undefined,
                                   parse_type_may_optional_undefined,
                                   undefined)
from tensorpc.core.pfl.constants import PFL_BUILTIN_PROXY_INIT_FN, PFL_STDLIB_FUNC_META_ATTR
from tensorpc.core.moduleid import get_qualname_of_type
from tensorpc.core.pfl.pfl_reg import STD_REGISTRY
from tensorpc.core.tree_id import UniqueTreeId

from .core import (PFL_LOGGER, FuncMatchResult, PFLCompileFuncMeta, PFLCompileReq, PFLExprFuncArgInfo, PFLExprFuncInfo, PFLExprInfo, PFLExprType, PFLStdlibFuncMeta, PFLMetaInferResult, get_eval_cfg_in_parse_ctx, get_parse_cache_checked,
                   get_parse_context_checked, param_fn, varparam_fn)
from .typedefs import (BoolOpType, BinOpType, CompareType, UnaryOpType)

_PFLTYPE_TO_SUPPORTED_METHODS = {
    PFLExprType.STRING: {
        "startswith":
        inspect.Signature([param_fn("prefix", str)],
                          return_annotation=bool),  # startsWith
        "endswith":
        inspect.Signature([param_fn("suffix", str)],
                          return_annotation=bool),  # endsWith
        "find":
        inspect.Signature([param_fn("sub", str),
                           param_fn("start", int, 0)],
                          return_annotation=int),  # indexOf
        "rfind":
        inspect.Signature([param_fn("sub", str),
                           param_fn("start", int, 0)],
                          return_annotation=int),  # indexOf
        "replace":
        inspect.Signature(
            [param_fn("old", str), param_fn("new", str)],
            return_annotation=str),  # replace
        "split":
        inspect.Signature(
            [param_fn("sep", str, None),
             param_fn("maxsplit", int, -1)],
            return_annotation=list[str]),  # split
        "join":
        inspect.Signature([param_fn("iterable", list[str])],
                          return_annotation=str),  # join
        "strip":
        inspect.Signature([],
                          return_annotation=str),  # strip
        "rstrip":
        inspect.Signature([],
                          return_annotation=str),  # rstrip
        "lstrip":
        inspect.Signature([],
                          return_annotation=str),  # rstrip
        "lower":
        inspect.Signature([],
                          return_annotation=str),  # rstrip
        "upper":
        inspect.Signature([],
                          return_annotation=str),  # rstrip

    },
}


def _dftype_with_gen_to_supported_methods(vt: Any):
    return {
        PFLExprType.ARRAY: {
            "append":
            inspect.Signature([param_fn("value", vt)],
                              return_annotation=None),  # push
            "extend":
            inspect.Signature([param_fn("iterable", list[vt])],
                              return_annotation=None),  # extend
            "insert":
            inspect.Signature([param_fn("index", int),
                               param_fn("value", vt)],
                              return_annotation=None),  # insert
            "remove":
            inspect.Signature([param_fn("value", vt)],
                              return_annotation=None),  # remove
            "pop":
            inspect.Signature([param_fn("index", int, -1)],
                              return_annotation=vt),  # pop
            "clear":
            inspect.Signature([], return_annotation=None),  # clear
        },
        PFLExprType.OBJECT: {
            "update":
            inspect.Signature([param_fn("iterable", dict[str, vt])],
                              return_annotation=None),  # update
            "remove":
            inspect.Signature([param_fn("key", str)],
                              return_annotation=None),  # remove
            "pop":
            inspect.Signature([param_fn("key", str)],
                              return_annotation=vt),  # pop
            # "clear": inspect.Signature([], return_annotation=None), # clear
            # we don't support generator in pfl, so we use array instead.
            "items":
            inspect.Signature([], return_annotation=list[tuple[str, vt]]),  # items
            "keys":
            inspect.Signature([], return_annotation=list[str]),  # keys
            "values":
            inspect.Signature([], return_annotation=list[vt]),  # values
        }
    }


@dataclasses.dataclass
class PFLStaticVar(PFLExprInfo):
    name: Optional[str] = None

    def __repr__(self):
        return super().__repr__()

    def to_dict(self):
        d = super().to_dict()
        d["name"] = self.name
        return d


class PFLASTType(enum.IntEnum):
    FUNC = 0
    EXPR = 1
    ARG = 2
    MODULE = 3
    CLASS = 4
    STMT_MASK = 0x10
    ASSIGN = 0x11
    IF = 0x12
    EXPR_STMT = 0x13
    AUG_ASSIGN = 0x14
    FOR = 0x15
    WHILE = 0x16
    ANN_ASSIGN = 0x17
    RETURN = 0x18
    BREAK = 0x19
    CONTINUE = 0x1A

    EXPR_MASK = 0x20

    BOOL_OP = 0x21
    BIN_OP = 0x22
    UNARY_OP = 0x23
    COMPARISON = 0x24
    ARRAY = 0x25
    CALL = 0x26
    NAME = 0x27
    CONSTANT = 0x28
    SUBSCRIPT = 0x29
    DICT = 0x2A
    ATTR = 0x2B
    IF_EXP = 0x2C
    SLICE = 0x2D
    TUPLE = 0x2E

    def __repr__(self):
        if self in _PFLAST_TYPE_TO_STR:
            return _PFLAST_TYPE_TO_STR[self]
        return super().__repr__()


_PFLAST_TYPE_TO_STR = {
    PFLASTType.FUNC: "funcdef",
    PFLASTType.ASSIGN: "assign",
    PFLASTType.IF: "if",
    PFLASTType.AUG_ASSIGN: "aug_assign",
    PFLASTType.ANN_ASSIGN: "ann_assign",
    PFLASTType.BOOL_OP: "bool_op",
    PFLASTType.BIN_OP: "bin_op",
    PFLASTType.UNARY_OP: "unary_op",
    PFLASTType.COMPARISON: "comparison",
    PFLASTType.CALL: "call",
    PFLASTType.NAME: "name",
    PFLASTType.CONSTANT: "constant",
    PFLASTType.SUBSCRIPT: "subscript",
    PFLASTType.ARRAY: "array",
    PFLASTType.DICT: "dict",
    PFLASTType.ATTR: "attr",
    PFLASTType.SLICE: "slice",

}

_PFL_UNARY_TYPE_TO_METHOD_NAME = {
    UnaryOpType.INVERT: "__invert__",
    UnaryOpType.NOT: "__not__",
    UnaryOpType.UADD: "__pos__",
    UnaryOpType.USUB: "__neg__",
}

_PFL_COMPARE_TYPE_TO_METHOD_NAME = {
    CompareType.EQUAL: "__eq__",
    CompareType.NOT_EQUAL: "__ne__",
    CompareType.LESS: "__lt__",
    CompareType.LESS_EQUAL: "__le__",
    CompareType.GREATER: "__gt__",
    CompareType.GREATER_EQUAL: "__ge__",
    CompareType.IN: "__contains__",
    CompareType.NOT_IN: "__contains__",
}

_PFL_BINARY_TYPE_TO_METHOD_NAME = {
    BinOpType.ADD: "__add__",
    BinOpType.SUB: "__sub__",
    BinOpType.MULT: "__mul__",
    BinOpType.DIV: "__truediv__",
    BinOpType.FLOOR_DIV: "__floordiv__",
    BinOpType.MOD: "__mod__",
    BinOpType.POW: "__pow__",
    BinOpType.LSHIFT: "__lshift__",
    BinOpType.RSHIFT: "__rshift__",
    BinOpType.BIT_OR: "__or__",
    BinOpType.BIT_XOR: "__xor__",
    BinOpType.BIT_AND: "__and__",
    BinOpType.MATMUL: "__matmul__",
}

_PFL_AUG_ASSIGN_METHOD_NAME = {
    BinOpType.ADD: "__iadd__",
    BinOpType.SUB: "__isub__",
    BinOpType.MULT: "__imul__",
    BinOpType.DIV: "__itruediv__",
    BinOpType.FLOOR_DIV: "__ifloordiv__",
    BinOpType.MOD: "__imod__",
    BinOpType.POW: "__ipow__",
    BinOpType.LSHIFT: "__ilshift__",
    BinOpType.RSHIFT: "__irshift__",
    BinOpType.BIT_OR: "__ior__",
    BinOpType.BIT_XOR: "__ixor__",
    BinOpType.BIT_AND: "__iand__",
    BinOpType.MATMUL: "__imatmul__",

}

_PFL_BINARY_TYPE_TO_REVERSE_METHOD_NAME = {
    BinOpType.ADD: "__radd__",
    BinOpType.SUB: "__rsub__",
    BinOpType.MULT: "__rmul__",
    BinOpType.DIV: "__rtruediv__",
    BinOpType.FLOOR_DIV: "__rfloordiv__",
    BinOpType.MOD: "__rmod__",
    BinOpType.POW: "__rpow__",
    BinOpType.LSHIFT: "__rlshift__",
    BinOpType.RSHIFT: "__rrshift__",
    BinOpType.BIT_OR: "__ror__",
    BinOpType.BIT_XOR: "__rxor__",
    BinOpType.BIT_AND: "__rand__",
    BinOpType.MATMUL: "__rmatmul__",

}

_PFL_COMPARE_TYPE_TO_REVERSE_METHOD_NAME = {
    CompareType.EQUAL: "__ne__",
    CompareType.NOT_EQUAL: "__eq__",
    CompareType.LESS: "__ge__",
    CompareType.LESS_EQUAL: "__gt__",
    CompareType.GREATER: "__le__",
    CompareType.GREATER_EQUAL: "__lt__",
    CompareType.IN: "__contains__",
    CompareType.NOT_IN: "__contains__",
}

SourceLocType: TypeAlias = tuple[int, int, Optional[int], Optional[int]]

def _is_unknown_or_any(st: PFLExprInfo) -> bool:
    return st.type == PFLExprType.UNKNOWN or st.type == PFLExprType.ANY

@dataclasses.dataclass
class PFLAstNodeBase:
    type: PFLASTType
    # record lineno/col_offset from ast node for debug
    source_loc: SourceLocType

    def get_source_loc_checked(self):
        end_l = self.source_loc[2]
        end_c = self.source_loc[3]
        assert end_l is not None and end_c is not None 
        return (self.source_loc[0], self.source_loc[1], end_l, end_c)

    def get_range_start(self):
        return (self.source_loc[0], self.source_loc[1])

    def get_range_end(self):
        end_l = self.source_loc[2]
        end_c = self.source_loc[3]
        if end_l is None or end_c is None:
            return None 
        return (end_l, end_c)

    def in_range(self, lineno: int, column: int):
        lc = (lineno, column)
        end_l = self.source_loc[2]
        end_c = self.source_loc[3]
        if end_l is None or end_c is None:
            return lc >= self.get_range_start()
        end_lc = (end_l, end_c)
        return lc >= self.get_range_start() and lc <= end_lc

    def in_range_lineno(self, lineno: int):
        end_l = self.source_loc[2]
        if end_l is None:
            return lineno >= self.get_range_start()[0]
        return lineno >= self.get_range_start()[0] and lineno <= end_l

@dataclasses.dataclass
class PFLAstStmt(PFLAstNodeBase):
    pass


@dataclasses.dataclass(kw_only=True)
class PFLExpr(PFLAstNodeBase):
    st: PFLExprInfo = dataclasses.field(
        default_factory=lambda: PFLExprInfo(PFLExprType.UNKNOWN))
    # is_const: Union[bool, Undefined] = undefined

    @property 
    def is_const(self) -> bool:
        return not isinstance(self.st._constexpr_data, Undefined)

    @staticmethod
    def all_constexpr(*args: Optional["PFLExpr"]):
        for arg in args:
            if arg is not None:
                if arg.is_const != True:
                    return False
        return True

    @staticmethod
    def any_constexpr(*args: Optional["PFLExpr"]):
        for arg in args:
            if arg is not None:
                if arg.is_const == True:
                    return True
        return False

    def consteval(self) -> bool:
        """run const evaluation. result is stored in `self.st.metadata`
        user should return True when a consteval is succeed.
        """
        return True

    def metaeval(self) -> bool:
        """run meta-data const evaluation. used when static type define "meta_infer" function.
        """
        return self.consteval()

    def _update_std_func_meta(self, fn: Callable):
        pfl_meta = getattr(fn, PFL_STDLIB_FUNC_META_ATTR, None)
        if pfl_meta is not None:
            assert isinstance(pfl_meta, PFLStdlibFuncMeta)
            self.st._meta_infer = pfl_meta.meta_infer
            self.st._static_type_infer = pfl_meta.static_type_infer
            self.st._force_meta_infer = pfl_meta.force_meta_infer

    def _get_consteval_operands(
            self, *exprs: "PFLExpr") -> Optional[list[Any]]:
        res: list[Any] = []
        for i, expr in enumerate(exprs):
            if not isinstance(expr.st.metadata, Undefined):
                assert not isinstance(expr.st.metadata, PFLExprInfo)
                res.append(expr.st.metadata)
            else:
                eval_cfg = get_eval_cfg_in_parse_ctx()
                if eval_cfg is not None and not eval_cfg.allow_partial:
                    cur_expr_str = unparse_pfl_expr(expr)
                    self_str = unparse_pfl_expr(self)
                    raise PFLEvalError(f"Arg-{i}({cur_expr_str}) of Expr {self_str}"
                                       f" consteval failed. check missing deps.", self)
                return None
        return res

    def _get_consteval_operands_st(
            self, *exprs: "PFLExpr") -> Optional[list[PFLExprInfo]]:
        res: list[PFLExprInfo] = []
        if not exprs:
            return res
        found = False
        eval_cfg = get_eval_cfg_in_parse_ctx()
        for i, expr in enumerate(exprs):
            if not isinstance(expr.st.metadata, Undefined):
                assert not isinstance(expr.st.metadata, PFLExprInfo)
                res.append(expr.st)
                found = True
            else:
                if eval_cfg is not None:
                    if not eval_cfg.allow_partial:
                        cur_expr_str = unparse_pfl_expr(expr)
                        self_str = unparse_pfl_expr(self)
                        raise PFLEvalError(f"Arg-{i}({cur_expr_str}) of Expr {self_str}"
                                        f" consteval failed. check missing deps.", self)
                    else:
                        if eval_cfg.prefer_meta_eval:
                            res.append(expr.st)
                        else:
                            return None 
                else:
                    return None
        if not found:
            return None
        return res

@dataclasses.dataclass(kw_only=True)
class PFLArg(PFLAstNodeBase):
    arg: str 
    annotation: Optional[str] = None
    default: Optional[PFLExpr] = None
    st: PFLExprInfo = dataclasses.field(
        default_factory=lambda: PFLExprInfo(PFLExprType.UNKNOWN))

@dataclasses.dataclass(kw_only=True)
class PFLAssign(PFLAstStmt):
    target: PFLExpr
    value: PFLExpr

    def check_and_infer_type(self):
        assert self.value.st.is_convertable(
            self.target.st
        ), f"{self.value.st} not convertable to {self.target.st}"


@dataclasses.dataclass(kw_only=True)
class PFLAugAssign(PFLAstStmt):
    target: PFLExpr
    op: BinOpType
    value: PFLExpr
    compilable_uid: Union[Undefined, str] = undefined

    def check_and_infer_type(self):
        resolved_custom_expr = None
        if self.target.st.type == PFLExprType.DATACLASS_OBJECT:
            resolved_custom_expr = self.target
        op_func = None
        is_custom_type = False
        if resolved_custom_expr is not None:
            dcls_type = resolved_custom_expr.st.get_origin_type_checked()
            # use custom operator in left st if found
            op_name = _PFL_AUG_ASSIGN_METHOD_NAME[self.op]
            op_func = inspect.getattr_static(dcls_type, op_name, None)
            assert op_func is not None, f"can't find {op_name} in custom type {get_qualname_of_type(dcls_type)}"
            op_func_st = get_parse_cache_checked().cached_parse_func(
                op_func, self_type=self.target.st.annotype)
            finfo = op_func_st.get_func_info_checked()
            assert len(
                finfo.args
            ) == 2, f"custom operator {op_name} must have one non-self arg, but got {len(finfo.args)}"
            assert self.value.st.is_convertable(
                finfo.args[1].type
            ), f"aug assign value {self.value.st} not convertable to {finfo.args[1].type}"
            is_custom_type = True
            custom_res_type = finfo.return_type
            assert custom_res_type is not None, f"custom operator {op_name} must have return type"
            if not resolved_custom_expr.st.is_stdlib:
                assert not finfo.is_template() and not finfo.is_always_inline(), "custom operator don't support template or inline."
                ctx = get_parse_context_checked()
                creq = ctx.enqueue_func_compile(op_func, finfo, is_method_def=True, self_type=self.target.st)
                self.compilable_uid = creq.get_func_compile_uid()

        if not is_custom_type:
            assert self.target.st.support_aug_assign()
            assert self.value.st.is_convertable(
                self.target.st
            ), f"{self.value.st} not convertable to {self.target.st}"

    def run(self, lfs: Any, rfs: Any) -> Any:
        if self.op == BinOpType.ADD:
            lfs += rfs
        elif self.op == BinOpType.SUB:
            lfs -= rfs
        elif self.op == BinOpType.MULT:
            lfs *= rfs
        elif self.op == BinOpType.DIV:
            lfs /= rfs
        elif self.op == BinOpType.FLOOR_DIV:
            lfs //= rfs
        elif self.op == BinOpType.POW:
            lfs **= rfs
        elif self.op == BinOpType.MOD:
            lfs %= rfs
        elif self.op == BinOpType.LSHIFT:
            lfs <<= rfs
        elif self.op == BinOpType.RSHIFT:
            lfs >>= rfs
        elif self.op == BinOpType.BIT_OR:
            lfs |= rfs
        elif self.op == BinOpType.BIT_XOR:
            lfs ^= rfs
        elif self.op == BinOpType.BIT_AND:
            lfs &= rfs
        else:
            raise NotImplementedError
        return lfs


@dataclasses.dataclass(kw_only=True)
class PFLAnnAssign(PFLAstStmt):
    target: PFLExpr
    annotation: str
    value: Optional[PFLExpr]

    def check_and_infer_type(self):
        assert isinstance(self.target, PFLName)
        if self.value is not None:
            assert self.value.st.is_convertable(
                self.target.st
            ), f"{self.value.st} not convertable to {self.target.st}"


@dataclasses.dataclass(kw_only=True)
class PFLFor(PFLAstStmt):
    target: PFLExpr
    iter: PFLExpr
    body: list[PFLAstStmt]

    def check_and_infer_type(self):
        if self.iter.st.type == PFLExprType.ARRAY:
            self.target.st = self.iter.st.childs[0]
        elif self.iter.st.type == PFLExprType.RANGE:
            self.target.st = PFLExprInfo(PFLExprType.NUMBER, annotype=parse_type_may_optional_undefined(int))
        else:
            raise NotImplementedError(
                "for loop iter type must be array or range object")
        return self


@dataclasses.dataclass(kw_only=True)
class PFLWhile(PFLAstStmt):
    test: PFLExpr
    body: list[PFLAstStmt]

    def check_and_infer_type(self):
        test_dtype = self.test.st
        assert test_dtype.can_cast_to_bool(
        ), f"test must be convertable to bool, but got {test_dtype}"
        if not isinstance(self.test, PFLExpr):
            raise ValueError("test must be a PFLExpr")
        return self


@dataclasses.dataclass(kw_only=True)
class PFLIf(PFLAstStmt):
    test: PFLExpr
    body: list[PFLAstStmt]
    orelse: list[PFLAstStmt] = dataclasses.field(default_factory=list)
    # indicate new variables after this if block.
    _new: Union[dict[str, PFLExprInfo], Undefined] = undefined
    def check_and_infer_type(self):
        test_dtype = self.test.st
        assert test_dtype.can_cast_to_bool(
        ), f"test must be convertable to bool, but got {test_dtype}"
        if not isinstance(self.test, PFLExpr):
            raise ValueError("test must be a PFLExpr")
        return self

    def get_flatten_test_body(self):
        stmt = self
        testAndBodyArr: list[tuple[Optional[PFLExpr], list[PFLAstStmt]]] = [(stmt.test, stmt.body)]
        while (len(stmt.orelse) == 1 and stmt.orelse[0].type == PFLASTType.IF):
            nextIfStmt = cast(PFLIf, stmt.orelse[0])
            testAndBodyArr.append((nextIfStmt.test, nextIfStmt.body))
            stmt = nextIfStmt
        # append last
        testAndBodyArr.append((None, stmt.orelse))
        return testAndBodyArr


@dataclasses.dataclass(kw_only=True)
class PFLExprStmt(PFLAstStmt):
    value: PFLExpr

@dataclasses.dataclass
class PFLReturn(PFLAstStmt):
    value: Optional[PFLExpr] = None

@dataclasses.dataclass
class PFLBreak(PFLAstStmt):
    pass

@dataclasses.dataclass
class PFLContinue(PFLAstStmt):
    pass

@dataclasses.dataclass(kw_only=True)
class PFLBoolOp(PFLExpr):
    op: BoolOpType
    values: list[PFLExpr]

    def check_and_infer_type(self):
        allow_partial = get_parse_context_checked().cfg.allow_partial_type_infer
        value_has_unk = False
        for v in self.values:
            if _is_unknown_or_any(v.st):
                value_has_unk = True
                break
        for v in self.values:
            if not allow_partial or not value_has_unk:
                assert v.st.support_bool_op()
        self.st = PFLExprInfo(PFLExprType.BOOL)
        # handle exprs such as `a or True`
        if self.op == BoolOpType.OR:
            # if any of the values is True, the result is True
            has_true = False 
            for v in self.values:
                if v.st.type == PFLExprType.BOOL and v.st._constexpr_data is True:
                    has_true = True
                    break
            if has_true:
                self.st._constexpr_data = True 
                return self 
        # handle exprs such as `a and False`
        if self.op == BoolOpType.AND:
            # if any of the values is False, the result is False
            has_false = False 
            for v in self.values:
                if v.st.type == PFLExprType.BOOL and v.st._constexpr_data is False:
                    has_false = True
                    break
            if has_false:
                self.st._constexpr_data = False 
                return self
        is_const = PFLExpr.all_constexpr(*self.values)
        if is_const:
            self.st._constexpr_data = self.run(*[v.st._constexpr_data for v in self.values])
        return self

    def consteval(self):
        operands = self._get_consteval_operands(*self.values)
        if operands is not None:
            if self.op == BoolOpType.AND:
                self.st.metadata = all(operands)
            else:
                self.st.metadata = any(operands)
            return True
        return False

    def run(self, *values: Any) -> Any:
        if self.op == BoolOpType.AND:
            return all(values)
        else:
            return any(values)

@dataclasses.dataclass(kw_only=True)
class PFLUnaryOp(PFLExpr):
    op: UnaryOpType
    operand: PFLExpr

    def check_and_infer_type(self):
        allow_partial = get_parse_context_checked().cfg.allow_partial_type_infer
        operand_has_unknown = _is_unknown_or_any(self.operand.st)
        if not allow_partial:
            assert not operand_has_unknown, f"unary operand type is unknown: {self.operand.st}"

        # overrideable operators
        left = self.operand
        is_custom_type: bool = False
        custom_res_type = None
        resolved_custom_expr = None
        if left.st.type == PFLExprType.DATACLASS_OBJECT:
            resolved_custom_expr = left
        op_func = None
        compiled_uid = None
        if resolved_custom_expr is not None:
            dcls_type = resolved_custom_expr.st.get_origin_type_checked()
            # use custom operator in left st if found
            op_name = _PFL_UNARY_TYPE_TO_METHOD_NAME[self.op]
            op_func = inspect.getattr_static(dcls_type, op_name, None)
            assert op_func is not None, f"can't find {op_name} in custom type {get_qualname_of_type(dcls_type)}"
            op_func_st = get_parse_cache_checked().cached_parse_func(
                op_func, self_type=self.operand.st.annotype)
            finfo = op_func_st.get_func_info_checked()
            assert len(
                finfo.args
            ) == 1, f"custom operator {op_name} must have no non-self arg, but got {len(finfo.args)}"
            is_custom_type = True
            custom_res_type = finfo.return_type
            assert custom_res_type is not None, f"custom operator {op_name} must have return type"
            if not resolved_custom_expr.st.is_stdlib:
                assert not finfo.is_template() and not finfo.is_always_inline(), "custom operator don't support template or inline."
                ctx = get_parse_context_checked()
                creq = ctx.enqueue_func_compile(op_func, finfo, is_method_def=True, self_type=self.operand.st)
                compiled_uid = creq.get_func_compile_uid()
        if not is_custom_type:
            if allow_partial and operand_has_unknown:
                # if any operand is unknown, and no custom type, result is unknown
                self.st = PFLExprInfo(PFLExprType.UNKNOWN)
                return self
            self.operand.st.check_support_unary_op("left")
            if self.op == UnaryOpType.NOT:
                self.st = dataclasses.replace(self.operand.st, type=PFLExprType.BOOL, annotype=parse_type_may_optional_undefined(bool))
            elif (self.op == UnaryOpType.UADD or self.op == UnaryOpType.USUB) and self.operand.st.type == PFLExprType.BOOL:
                self.st = dataclasses.replace(self.operand.st, type=PFLExprType.NUMBER, annotype=parse_type_may_optional_undefined(int))
            else:
                self.st = dataclasses.replace(self.operand.st)
        else:
            assert custom_res_type is not None
            self.st = custom_res_type
        if compiled_uid is not None:
            self.st.compiled_uid = compiled_uid
        if self.operand.is_const:
            self.st._constexpr_data = self.run(self.operand.st._constexpr_data)
        if op_func is not None:
            self._update_std_func_meta(op_func)
        return self

    def run(self, x: Any) -> Any:
        if self.op == UnaryOpType.INVERT:
            return ~x
        elif self.op == UnaryOpType.NOT:
            return not x
        elif self.op == UnaryOpType.UADD:
            return +x
        else:
            return -x

    def consteval(self):
        operands = self._get_consteval_operands(self.operand)
        if operands is not None:
            self.st.metadata = self.run(operands[0])
            return True
        return False

    def metaeval(self):
        if self.st.meta_infer is not None:
            operands = self._get_consteval_operands_st(self.operand)
            if operands is not None:
                infer_res = self.st.meta_infer(*operands)
                if infer_res is not None:
                    assert isinstance(infer_res, PFLMetaInferResult), "meta infer function must return `pfl.PFLMetaInferResult`"
                    self.st.metadata = infer_res.data
                    return True
                return False
        return self.consteval()


@dataclasses.dataclass(kw_only=True)
class PFLIfExp(PFLExpr):
    test: PFLExpr
    body: PFLExpr
    orelse: PFLExpr

    def check_and_infer_type(self):
        allow_partial = get_parse_context_checked().cfg.allow_partial_type_infer
        operands_has_unk = _is_unknown_or_any(self.body.st) or _is_unknown_or_any(self.orelse.st)
        
        if not allow_partial or not _is_unknown_or_any(self.test.st):
            assert self.test.st.can_cast_to_bool(
            ), f"test must be convertable to bool, but got {self.test.st}"
        res_st = self.body.st
        if not allow_partial or not operands_has_unk:
            msg = f"body and orelse must be same type, but got {self.body.st} and {self.orelse.st}"
            res_st = self.body.st._check_equal_type_with_unk_any_type_promption(self.orelse.st, msg) 
        if operands_has_unk:
            self.st = PFLExprInfo(PFLExprType.UNKNOWN)
        else:
            self.st = dataclasses.replace(res_st)
        return self

    def consteval(self):
        operands = self._get_consteval_operands(self.test, self.body,
                                                       self.orelse)
        if operands is not None:
            self.st.metadata = operands[1] if operands[0] else operands[2]
            return True
        return False


@dataclasses.dataclass(kw_only=True)
class PFLBinOpBase(PFLExpr):
    left: PFLExpr
    right: PFLExpr
    # when we use custom operator, indicate which operand is used
    # for custom operator resolution.
    is_right: Union[bool, Undefined] = undefined

    def get_is_right_val(self) -> bool:
        if isinstance(self.is_right, Undefined):
            return False
        return self.is_right

    def resolve_custom_type(self, op: Union[BinOpType, CompareType], is_compare: bool):
        # TODO if left and right are both custom type
        # overrideable operators
        left = self.left
        right = self.right
        is_custom_type: bool = False
        custom_res_type = None
        resolved_custom_expr = None
        resolved_op_func = None
        if is_compare:
            assert isinstance(op, CompareType), f"op must be CompareType, but got {type(op)}"
            op_name = _PFL_COMPARE_TYPE_TO_METHOD_NAME[op]
            rop_name = _PFL_COMPARE_TYPE_TO_REVERSE_METHOD_NAME[op]
        else:
            assert isinstance(op, BinOpType), f"op must be BinOpType, but got {type(op)}"
            op_name = _PFL_BINARY_TYPE_TO_METHOD_NAME[op]
            rop_name = _PFL_BINARY_TYPE_TO_REVERSE_METHOD_NAME[op]

        if left.st.type == PFLExprType.DATACLASS_OBJECT:
            dcls_type = left.st.get_origin_type_checked()
            op_func = inspect.getattr_static(dcls_type, op_name, None)
            if op_func is not None:
                resolved_custom_expr = left
                resolved_op_func = op_func
        if resolved_custom_expr is None:
            if right.st.type == PFLExprType.DATACLASS_OBJECT:
                dcls_type = right.st.get_origin_type_checked()
                op_func = inspect.getattr_static(dcls_type, rop_name, None)
                if op_func is not None:
                    resolved_custom_expr = right
                    resolved_op_func = op_func
                    self.is_right = True
        if resolved_custom_expr is not None:
            assert resolved_op_func is not None 
            op_func_st = get_parse_cache_checked().cached_parse_func(
                resolved_op_func, self_type=resolved_custom_expr.st.annotype)
            finfo = op_func_st.get_func_info_checked()
            if not resolved_custom_expr.st.is_stdlib:
                assert finfo.overloads is None, "custom operator don't support overloads."
                assert not finfo.is_template() and not finfo.is_always_inline(), "custom operator don't support template or inline."
                ctx = get_parse_context_checked()
                creq = ctx.enqueue_func_compile(resolved_op_func, finfo, is_method_def=True, self_type=resolved_custom_expr.st)
                is_custom_type = True
                custom_res_type = finfo.return_type
                assert custom_res_type is not None, f"custom operator {op_name} must have return type"
                compiled_uid = creq.get_func_compile_uid()
                custom_res_type.compiled_uid = compiled_uid
            else:
                overload_infos = [finfo]
                if finfo.overloads is not None:
                    overload_infos.extend(finfo.overloads)
                overload_scores: list[tuple[int, int, PFLExprInfo]] = []
                errors: list[str] = []
                assert len(
                    finfo.args
                ) == 2, f"custom operator {op_name} must have one non-self arg, but got {len(finfo.args)}"

                for i, overload in enumerate(overload_infos):
                    try:
                        if self.is_right == True:
                            st_to_check = left.st
                        else:
                            st_to_check = right.st
                        st_to_check.check_convertable(overload.args[1].type,
                                            f"custom operator {op_name} overload {overload} arg")
                        if st_to_check.is_equal_type(overload.args[1].type):
                            score = 2
                        else:
                            score = 1
                        assert overload.return_type is not None, f"func {op_func_st} overload {overload} must have return type"
                        overload_scores.append((score, i, overload.return_type))
                    except BaseException as e:
                        # traceback.print_exc()
                        errors.append(str(e))
                if not overload_scores:
                    error_msg = f"func {op_func_st} overloads not match args {[self.left.st, self.right.st]} error:\n"
                    for e in errors:
                        error_msg += f"  - {e}\n"
                    print(error_msg)
                    raise ValueError(error_msg)
                # find best overload
                overload_scores.sort(key=lambda x: x[0], reverse=True)
                _, best_idx, best_return_type = overload_scores[0]

                is_custom_type = True
                custom_res_type = best_return_type
                assert custom_res_type is not None, f"custom operator {op_name} must have return type"
        return is_custom_type, custom_res_type, resolved_op_func

    def metaeval(self):
        operands = self._get_consteval_operands_st(
            self.left, self.right)
        if operands is not None:
            if self.st.meta_infer is not None:
                if not is_undefined(self.is_right) and self.is_right:
                    infer_res = self.st.meta_infer(*operands[::-1])
                else:
                    infer_res = self.st.meta_infer(*operands)
                if infer_res is not None:
                    assert isinstance(infer_res, PFLMetaInferResult), "meta infer function must return `pfl.PFLMetaInferResult`"
                    self.st.metadata = infer_res.data
                    return True
                return False
        return self.consteval()


@dataclasses.dataclass(kw_only=True)
class PFLBinOp(PFLBinOpBase):
    op: BinOpType

    def check_and_infer_type(self):
        is_custom_type, custom_res_type, op_func = self.resolve_custom_type(
            self.op, is_compare=False)
        allow_partial = get_parse_context_checked().cfg.allow_partial_type_infer
        operand_has_unknown = _is_unknown_or_any(self.left.st) or _is_unknown_or_any(self.right.st)
        if not allow_partial:
            assert not operand_has_unknown, f"left or right type is unknown: {self.left.st}, {self.right.st}"

        if not is_custom_type:
            if allow_partial and operand_has_unknown:
                # if any operand is unknown, and no custom type, result is unknown
                self.st = PFLExprInfo(PFLExprType.UNKNOWN)
            else:
                if self.op == BinOpType.DIV:
                    self.st = PFLExprInfo(PFLExprType.NUMBER, annotype=parse_type_may_optional_undefined(float))
                else:
                    promotion_type = self.left.st.check_support_binary_op_and_promotion(self.op, self.right.st)
                    if self.left.st.type == PFLExprType.STRING:
                        self.st = PFLExprInfo(PFLExprType.STRING, annotype=parse_type_may_optional_undefined(str))
                    else:
                        self.st = PFLExprInfo(PFLExprType.NUMBER, annotype=promotion_type)
        else:
            assert custom_res_type is not None
            self.st = custom_res_type
        if PFLExpr.all_constexpr(self.left, self.right):
            self.st._constexpr_data = self.run(self.left.st._constexpr_data, self.right.st._constexpr_data)
        if op_func is not None:
            self._update_std_func_meta(op_func)
        return self

    def run(self, lfs: Any, rfs: Any) -> Any:
        operands = [lfs, rfs]
        if self.op == BinOpType.ADD:
            return operands[0] + operands[1]
        elif self.op == BinOpType.SUB:
            return operands[0] - operands[1]
        elif self.op == BinOpType.MULT:
            return operands[0] * operands[1]
        elif self.op == BinOpType.DIV:
            return operands[0] / operands[1]
        elif self.op == BinOpType.FLOOR_DIV:
            return operands[0] // operands[1]
        elif self.op == BinOpType.MOD:
            return operands[0] % operands[1]
        elif self.op == BinOpType.POW:
            return operands[0]**operands[1]
        elif self.op == BinOpType.LSHIFT:
            return operands[0] << operands[1]
        elif self.op == BinOpType.RSHIFT:
            return operands[0] >> operands[1]
        elif self.op == BinOpType.BIT_OR:
            return operands[0] | operands[1]
        elif self.op == BinOpType.BIT_XOR:
            return operands[0] ^ operands[1]
        elif self.op == BinOpType.BIT_AND:
            return operands[0] & operands[1]
        elif self.op == BinOpType.MATMUL:
            return operands[0] @ operands[1]
        else:
            raise NotImplementedError

    def consteval(self):
        operands = self._get_consteval_operands(self.left, self.right)
        if operands is not None:
            self.st.metadata = self.run(operands[0], operands[1])
            return True
        return False


@dataclasses.dataclass(kw_only=True)
class PFLCompare(PFLBinOpBase):
    op: CompareType

    def check_and_infer_type(self):
        allow_partial = get_parse_context_checked().cfg.allow_partial_type_infer
        operand_has_unknown = _is_unknown_or_any(self.left.st) or _is_unknown_or_any(self.right.st)
        if not allow_partial:
            assert not operand_has_unknown, f"left or right type is unknown: {self.left.st}, {self.right.st}"

        is_custom_type = False
        if self.op == CompareType.IN or self.op == CompareType.NOT_IN:
            # contains operator don't support custom datatype
            assert self.left.st.type != PFLExprType.DATACLASS_OBJECT
            assert self.right.st.type != PFLExprType.DATACLASS_OBJECT
        op_func = None
        custom_type_res = None
        if not (self.op == CompareType.IS or self.op == CompareType.IS_NOT):
            # overrideable operators
            is_custom_type, custom_type_res, op_func = self.resolve_custom_type(self.op, is_compare=True)
        if not is_custom_type:
            # all operands are base type
            if not (self.op == CompareType.EQUAL or self.op
                    == CompareType.NOT_EQUAL or self.op == CompareType.IS
                    or self.op == CompareType.IS_NOT):
                # handle string-type compares
                if not allow_partial or not operand_has_unknown:
                    if self.op == CompareType.IN or self.op == CompareType.NOT_IN:
                        assert self.left.st.type == PFLExprType.STRING and self.right.st.type == PFLExprType.OBJECT, f"left must be string and right must be object, but got {self.left.st.type} and {self.right.st.type}"
                    else:
                        self.left.st.check_support_compare_op(self.op, self.right.st, "left")
            self.st = PFLExprInfo(PFLExprType.BOOL, annotype=parse_type_may_optional_undefined(bool))
        else:
            assert custom_type_res is not None 
            self.st = custom_type_res
        if PFLExpr.all_constexpr(self.left, self.right):
            self.st._constexpr_data = self.run(self.left.st._constexpr_data, self.right.st._constexpr_data)
        if op_func is not None:
            self._update_std_func_meta(op_func)
        return self

    def run(self, lfs: Any, rfs: Any) -> Any:
        operands = [lfs, rfs]
        if self.op == CompareType.EQUAL:
            return operands[0] == operands[1]
        elif self.op == CompareType.NOT_EQUAL:
            return operands[0] != operands[1]
        elif self.op == CompareType.LESS:
            return operands[0] < operands[1]
        elif self.op == CompareType.LESS_EQUAL:
            return operands[0] <= operands[1]
        elif self.op == CompareType.GREATER:
            return operands[0] > operands[1]
        elif self.op == CompareType.GREATER_EQUAL:
            return operands[0] >= operands[1]
        elif self.op == CompareType.IS:
            return operands[0] is operands[1]
        elif self.op == CompareType.IS_NOT:
            return operands[0] is not operands[1]
        elif self.op == CompareType.IN:
            return operands[0] in operands[1]
        elif self.op == CompareType.NOT_IN:
            return operands[0] not in operands[1]
        else:
            raise NotImplementedError

    def consteval(self):
        operands = self._get_consteval_operands(self.left, self.right)
        if operands is not None:
            self.st.metadata = self.run(operands[0], operands[1])
            return True
        return False


@dataclasses.dataclass(kw_only=True)
class PFLCall(PFLExpr):
    func: PFLExpr
    args: list[PFLExpr]
    # keywords
    keys: Union[list[str], Undefined] = undefined
    vals: Union[list[PFLExpr], Undefined] = undefined


    def check_and_infer_type_with_overload(self) -> tuple[FuncMatchResult[PFLExpr], PFLExprFuncInfo]:
        # validate args
        allow_partial = get_parse_context_checked().cfg.allow_partial_type_infer
        assert self.func.st.type == PFLExprType.FUNCTION or self.func.st.type == PFLExprType.DATACLASS_TYPE, f"func must be function/dcls, but got {self.func.st.type}"
        overloads: list[tuple[FuncMatchResult[PFLExpr], PFLExprFuncInfo]] = []
        is_const = False
        constexpr_infer: Optional[Callable[..., Any]] = None
        constexpr_first_arg: Optional[Callable] = None
        if self.func.st.type == PFLExprType.FUNCTION:
            finfo = self.func.st.get_func_info_checked()
            raw_fn = finfo.raw_func
            if raw_fn is not None:
                std_item = STD_REGISTRY.get_item_by_key(raw_fn)
                if std_item is not None:
                    constexpr_infer = std_item.constexpr_infer
                    constexpr_first_arg = raw_fn
                if constexpr_infer is None:
                    std_meta: Optional[PFLStdlibFuncMeta] = getattr(raw_fn, PFL_STDLIB_FUNC_META_ATTR, None)
                    if std_meta is not None:
                        constexpr_infer = std_meta.constexpr_infer
            overload_infos = [self.func.st.get_func_info_checked()]
            if finfo.overloads is not None:
                overload_infos.extend(finfo.overloads)
            # overload match may fail. 
            match_errors: list[str] = []
            for overload_info in overload_infos:
                try:
                    match_res = self._get_matched_args(overload_info)
                except BaseException as e:
                    # traceback.print_exc()
                    match_errors.append(str(e))
                    continue
                typevar_map = self._get_typevar_map(match_res, allow_partial=allow_partial)
                if typevar_map:
                    overload_info = overload_info.typevar_substitution(typevar_map)
                    # generic type of overload_info is resolved, so we create match res again.
                    match_res = self._get_matched_args(overload_info)
                assert overload_info.return_type is not None, f"func {self.func} overload {overload_info} must have return type"
                overloads.append((match_res, overload_info))
            if not overloads:
                error_msg = f"func {self.func.st} overloads not match args {[a.st for a in self.args]} kws {self.keys}. match errors:\n"
                for e in match_errors:
                    error_msg += f"  - {e}\n"
                PFL_LOGGER.error(error_msg)
                raise ValueError(error_msg)

        elif self.func.st.type == PFLExprType.DATACLASS_TYPE:
            is_const = PFLExpr.all_constexpr(*self.args)
            if not is_undefined(self.vals):
                is_const &= PFLExpr.all_constexpr(*self.vals)
            if self.func.st.proxy_dcls is not None:
                fn = inspect.getattr_static(self.func.st.proxy_dcls, PFL_BUILTIN_PROXY_INIT_FN)
                assert isinstance(fn, staticmethod)
                func_st = get_parse_cache_checked().cached_parse_func(fn.__func__)
                assert func_st.func_info is not None 
                assert func_st.func_info.return_type is not None, "func return type must be annotated"
                match_res = self._get_matched_args(func_st.func_info)
                func_info = func_st.func_info
                # func_args, return_type = func_st.func_info.args, func_st.func_info.return_type
            else:
                if self.func.st.is_stdlib:
                    # stdlib dcls is lazy parsed.
                    assert self.func.st.annotype is not None 
                    std_item = STD_REGISTRY.get_item_by_key(self.func.st.annotype.origin_type)
                    if std_item is not None:
                        constexpr_infer = std_item.constexpr_infer
                        constexpr_first_arg = self.func.st.annotype.origin_type
                    dcls_st = get_parse_cache_checked().cached_parse_dcls(self.func.st.annotype.origin_type)
                    func_info = dcls_st.get_func_info_checked()
                else:
                    # dcls st func_info store init info if exists 
                    assert self.func.st.func_info is not None
                    func_info = self.func.st.func_info
                assert func_info.return_type is not None
                if not func_info.is_dcls:
                    # __init__, need to bound self
                    func_info = func_info.get_bounded_type(func_info.return_type)
            match_res = self._get_matched_args(func_info)
            overloads.append((match_res, func_info))
        else:
            raise NotImplementedError
        errors: list[str] = []
        overload_scores: list[tuple[int, int, PFLExprInfo]] = []
        # print("overloads", len(overloads))
        for i, overload in enumerate(overloads):
            try:
                score = self._check_single_overload(overload[0], overload[1], allow_partial)
                overload_scores.append((score, i, overload[1].return_type))
            except BaseException as e:
                # traceback.print_exc()
                errors.append(str(e))
        if not overload_scores:
            error_msg = f"func {self.func.st} overloads not match args {[a.st for a in self.args]} kws {self.keys}. error:\n"
            for e in errors:
                error_msg += f"  - {e}\n"
            PFL_LOGGER.error(error_msg)
            raise ValueError(error_msg)
        else:
            # find best overload
            overload_scores.sort(key=lambda x: x[0], reverse=True)
            _, best_idx, best_return_type = overload_scores[0]
            # if user define a static type infer function, we use this instead of static type check.
            args_st = []
            kwargs_st = {}
            if self.func.st._static_type_infer is not None:
                for a in self.args:
                    args_st.append(a.st)
                if not is_undefined(self.keys):
                    assert not is_undefined(self.vals)
                    for k, v in zip(self.keys, self.vals):
                        kwargs_st[k] = v.st

            if self.func.st._static_type_infer is not None:
                ret_type = self.func.st._static_type_infer(*args_st, **kwargs_st)
                ret_type_st = PFLExprInfo.from_annotype(parse_type_may_optional_undefined(ret_type), is_type=False, allow_union=False)
                self.st = ret_type_st
            else:
                self.st = dataclasses.replace(best_return_type)
            if self.func.st.type != PFLExprType.FUNCTION:
                # TODO better constexpr infer
                args = []
                kwargs = {}

                if is_const or constexpr_infer is not None:
                    for a in self.args:
                        assert a.is_const, "when you define static type infer, all arguments must be constexpr."
                        args.append(a.st._constexpr_data)
                    if not is_undefined(self.keys):
                        assert not is_undefined(self.vals)
                        for k, v in zip(self.keys, self.vals):
                            assert v.is_const, "when you define static type infer, all arguments must be constexpr."
                            kwargs[k] = v.st._constexpr_data
                if constexpr_infer is not None:
                    assert constexpr_first_arg is not None 
                    self.st._constexpr_data = constexpr_infer(constexpr_first_arg, *args, **kwargs)
                elif is_const:
                    if self.func.st.proxy_dcls is not None:
                        self.st._constexpr_data = inspect.getattr_static(self.func.st.proxy_dcls, PFL_BUILTIN_PROXY_INIT_FN)(*args, **kwargs)
                    else:
                        self.st._constexpr_data = self.func.st.get_origin_type_checked()(*args, **kwargs)
            return overloads[best_idx]

    def _get_matched_args(self, info: PFLExprFuncInfo):
        args = self.args 
        kwargs = {}
        if not isinstance(self.keys, Undefined):
            assert not isinstance(self.vals, Undefined)
            for name, a in zip(self.keys, self.vals):
                kwargs[name] = a
        return info.match_args_to_sig((args, kwargs))

    def _check_typevar_substitution(self, arg_value: PFLExprInfo, func_arg: PFLExprInfo, typevar_map: dict[TypeVar, PFLExprInfo]):
        annotype = func_arg.annotype
        assert annotype is not None 
        tv = cast(TypeVar, annotype.origin_type)

        if tv in typevar_map:
            assert arg_value.is_equal_type(typevar_map[tv])
        else:
            if tv.__bound__ is not None:
                bound_type = PFLExprInfo.from_annotype(
                    parse_type_may_optional_undefined(tv.__bound__), is_type=False, allow_union=True)
                arg_value.check_convertable(bound_type, f"func {self.func.st}")
            
            if tv.__constraints__:
                found = False
                for constraint in tv.__constraints__:
                    constraint_type = PFLExprInfo.from_annotype(
                        parse_type_may_optional_undefined(constraint), is_type=False, allow_union=True)
                    if arg_value.is_equal_type(constraint_type):
                        found = True
                        break
                assert found, f"func {self.func.st} arg {tv}({arg_value}) not match constraints {tv.__constraints__}"
            typevar_map[tv] = dataclasses.replace(arg_value)

    def _get_typevar_map(self, match_res: FuncMatchResult[PFLExpr], allow_partial: bool = False) -> dict[TypeVar, PFLExprInfo]:
        res: dict[TypeVar, PFLExprInfo] = {}
        for arg_info, arg_expr in match_res.args:
            if not is_undefined(arg_expr):
                if arg_info.type.type == PFLExprType.GENERIC_TYPE:
                    assert not allow_partial, "typevar substitution don't support partial infer."
                    self._check_typevar_substitution(arg_expr.st, arg_info.type, res)
        if match_res.vararg is not None:
            vararg_info = match_res.vararg[0]
            if vararg_info.type.type == PFLExprType.GENERIC_TYPE:
                assert not allow_partial, "typevar substitution don't support partial infer."
                for arg_expr in match_res.vararg[1]:
                    self._check_typevar_substitution(arg_expr.st, vararg_info.type, res)
        if match_res.var_kwarg is not None:
            kwarg_info = match_res.var_kwarg[0]
            if kwarg_info.type.type == PFLExprType.GENERIC_TYPE:
                assert not allow_partial, "typevar substitution don't support partial infer."
                for _, arg_expr in match_res.var_kwarg[1].items():
                    self._check_typevar_substitution(arg_expr.st, kwarg_info.type, res)
        return res


    def _check_single_overload(self, match_res: FuncMatchResult[PFLExpr], func_info: PFLExprFuncInfo, allow_partial: bool = False) -> int:
        match_score = 0
        for arg_info, arg_expr in match_res.args:
            log_prefix = f"func {func_info.name} arg {arg_info.name}"
            if not is_undefined(arg_expr):
                if not allow_partial or (not _is_unknown_or_any(arg_expr.st)):
                    arg_expr.st.check_convertable(arg_info.type, log_prefix)
                    if arg_expr.st.is_equal_type(arg_info.type):
                        match_score += 2
                    else:
                        match_score += 1
                else:
                    match_score += 1
            else:
                assert not is_undefined(arg_info.default_type), f"{log_prefix} don't have both arg value and default value"
        if match_res.vararg is not None:
            vararg_info = match_res.vararg[0]
            log_prefix = f"func {func_info.name} vararg {vararg_info.name}"
            for arg_expr in match_res.vararg[1]:
                if not allow_partial or (not _is_unknown_or_any(arg_expr.st)):
                    arg_expr.st.check_convertable(vararg_info.type, log_prefix)
                    if arg_expr.st.is_equal_type(vararg_info.type):
                        match_score += 2
                    else:
                        match_score += 1
                else:
                    match_score += 1
        if match_res.var_kwarg is not None:
            kwarg_info = match_res.var_kwarg[0]
            log_prefix = f"func {func_info.name} varkwarg {kwarg_info.name}"
            for _, arg_expr in match_res.var_kwarg[1].items():
                if not allow_partial or (not _is_unknown_or_any(arg_expr.st)):
                    arg_expr.st.check_convertable(kwarg_info.type, log_prefix)
                    if arg_expr.st.is_equal_type(kwarg_info.type):
                        match_score += 2
                    else:
                        match_score += 1
                else:
                    match_score += 1
        return match_score

    def consteval(self):
        args_check = [*self.args]
        check_attr_source: bool = False
        if isinstance(self.func, PFLAttribute):
            args_check.insert(0, self.func.value)
            check_attr_source = True
        operands = self._get_consteval_operands(*args_check)
        if operands is None:
            return False
        if check_attr_source:
            # remove attr source here.
            operands = operands[1:]
        kw_operands = None
        if not is_undefined(self.keys):
            assert not is_undefined(self.vals)
            kw_operands = {}
            for k, v in zip(self.keys, self.vals):
                if not v.st.has_metadata():
                    return False 
                kw_operands[k] = v.st.metadata
        if operands is None:
            operands = []
        if kw_operands is None:
            kw_operands = {}
        if isinstance(self.func, PFLName):
            if self.func.st.proxy_dcls is not None:
                fn = inspect.getattr_static(self.func.st.proxy_dcls, PFL_BUILTIN_PROXY_INIT_FN)
            else:
                fn = self.func.st.metadata
            if not isinstance(fn, Undefined):
                self.st.metadata = fn(*operands, **kw_operands)
                return True
            return False
        elif isinstance(self.func, PFLAttribute):
            obj = self.func.value.st.metadata
            if not isinstance(obj, Undefined):
                self.st.metadata = getattr(obj, self.func.attr)(*operands, **kw_operands)
                return True
        return False

    def metaeval(self):
        args_check = [*self.args]
        check_attr_source: bool = False
        if isinstance(self.func, PFLAttribute):
            args_check.insert(0, self.func.value)
            check_attr_source = True
        operands = self._get_consteval_operands_st(*args_check)
        kw_operands = None
        kw_has_defined_metadata = False
        if not is_undefined(self.keys):
            assert not is_undefined(self.vals)
            kw_operands = {}
            for k, v in zip(self.keys, self.vals):
                kw_operands[k] = v.st
                if v.st.has_metadata():
                    kw_has_defined_metadata = True 
        if operands is not None or kw_has_defined_metadata or self.func.st._force_meta_infer:
            if operands is None:
                operands = []
            else:
                if check_attr_source:
                    # remove attr source here.
                    operands = operands[1:]
            if kw_operands is None:
                kw_operands = {}
            if self.func.st.meta_infer is not None:
                finfo = self.func.st.get_func_info_checked()
                if isinstance(
                        self.func,
                        PFLAttribute):
                    if finfo.is_method:
                        obj = self.func.value.st.metadata
                        if not isinstance(obj, Undefined):
                            infer_res = self.func.st.meta_infer(
                                self.func.value.st, *operands, **kw_operands)
                            if infer_res is not None:
                                assert isinstance(infer_res, PFLMetaInferResult), "meta infer function must return `pfl.PFLMetaInferResult`"
                                self.st.metadata = infer_res.data
                                return True
                    else:
                        infer_res = self.func.st.meta_infer(*operands, **kw_operands)
                        if infer_res is not None:
                            assert isinstance(infer_res, PFLMetaInferResult), "meta infer function must return `pfl.PFLMetaInferResult`"
                            self.st.metadata = infer_res.data
                            return True
                elif isinstance(self.func, PFLName):
                    infer_res = self.func.st.meta_infer(*operands, **kw_operands)
                    if infer_res is not None:
                        assert isinstance(infer_res, PFLMetaInferResult), "meta infer function must return `pfl.PFLMetaInferResult`"
                        self.st.metadata = infer_res.data
                        return True
                return False

        return self.consteval()


@dataclasses.dataclass(kw_only=True)
class PFLName(PFLExpr):
    id: str
    is_store: Union[Undefined, bool] = undefined
    is_new: Union[Undefined, bool] = undefined

    def check_and_infer_type(self):
        if self.st.type == PFLExprType.DATACLASS_TYPE or self.st.type == PFLExprType.DATACLASS_OBJECT:
            assert self.st.annotype is not None, "dataclass must have annotype"

        elif self.st.type == PFLExprType.FUNCTION:
            if self.st.func_info is not None:
                finfo = self.st.get_func_info_checked()
                if finfo.raw_func is not None:
                    self._update_std_func_meta(finfo.raw_func)
            else:
                assert self.st.delayed_compile_req is not None 
                # compile func don't need stdlib meta update.

    def get_is_store(self) -> bool:
        if isinstance(self.is_store, Undefined):
            return False
        else:
            return self.is_store

@dataclasses.dataclass
class _AttrCompileInfo:
    # store property function PFLExprInfo.
    property_st: Optional[PFLExprInfo] = None

@dataclasses.dataclass(kw_only=True)
class PFLAttribute(PFLExpr):
    value: PFLExpr
    attr: str
    is_store: Union[Undefined, bool] = undefined
    compile_info: _AttrCompileInfo = dataclasses.field(default_factory=_AttrCompileInfo)

    def check_and_infer_type(self):
        if _is_unknown_or_any(self.value.st):
            ctx = get_parse_context_checked()
            assert ctx.cfg.allow_partial_type_infer
            self.st = PFLExprInfo(PFLExprType.UNKNOWN)
            return 
        if self.value.st.type == PFLExprType.DATACLASS_TYPE or self.value.st.type == PFLExprType.DATACLASS_OBJECT:
            if self.value.st.proxy_dcls is not None:
                assert self.value.st.type == PFLExprType.DATACLASS_TYPE # when proxy cls available, it must be dataclass type
                field_types = AnnotatedType.get_dataclass_fields_and_annotated_types_static(self.value.st.proxy_dcls)
                dcls_type = self.value.st.proxy_dcls
                field_keys = field_types.keys()
            else:
                assert self.value.st.annotype is not None, "dataclass must have annotype"
                assert self.value.st.annotype.is_dataclass_type()
                assert self.value.st.dcls_info is not None, f"dataclass {self.value.st.annotype} must have dcls_info"
                # field_types = self.value.st.annotype.get_dataclass_fields_and_annotated_types(
                # )
                field_types = {}
                field_keys = self.value.st.dcls_info._arg_name_to_idx.keys()

                dcls_type = self.value.st.annotype.origin_type
            if self.attr in field_keys:
                if self.value.st.proxy_dcls is not None:
                    field_annotype, field = field_types[self.attr]
                    field_default = field.default
                    new_st = PFLExprInfo.from_annotype(field_annotype,
                                                    is_type=False)
                else:
                    assert self.value.st.dcls_info is not None 
                    field_arg = self.value.st.dcls_info.get_parsed_field(self.attr)
                    field_default = field_arg.default
                    new_st = field_arg.type.shallow_copy()
                if self.value.st.type == PFLExprType.DATACLASS_TYPE:
                    # access constant
                    default = field_default
                    assert default is not dataclasses.MISSING, f"access field {self.attr} by type must have default value, we treat it as constant"
                    new_st._constexpr_data = default
                else:
                    if not is_undefined(self.value.st._constexpr_data):
                        # check partial constexpr fields, if defined,
                        # only forward these fields.
                        item = STD_REGISTRY.get_item_by_key(dcls_type)
                        should_fwd_constexpr = True
                        if item is not None and item.partial_constexpr_fields is not None:
                            should_fwd_constexpr = self.attr in item.partial_constexpr_fields
                        if should_fwd_constexpr:
                            new_st._constexpr_data = getattr(self.value.st._constexpr_data, self.attr, undefined)
                # print(self.attr, self.value.st._constexpr_data, new_st._constexpr_data)
                self.st = new_st
            else:
                # check attr is ClassVar (namespace alias)
                item = get_parse_cache_checked().cached_get_std_item(dcls_type)
                if item is not None and self.attr in item.namespace_aliases:
                    new_st = PFLExprInfo.from_annotype(parse_type_may_optional_undefined(item.namespace_aliases[self.attr]),
                                                   is_type=True)
                    self.st = new_st
                else:
                    unbound_func = getattr(dcls_type, self.attr)
                    prep_res = get_parse_cache_checked()._var_preproc(unbound_func)
                    is_prop = prep_res.is_property
                    unbound_func = prep_res.value
                    if is_prop:
                        assert self.value.st.is_stdlib, "don't support @property in user defined dataclass"
                    is_method_def = False
                    if self.value.st.type == PFLExprType.DATACLASS_OBJECT:
                        # TODO handle classmethod
                        is_method_def = not isinstance(unbound_func, staticmethod)
                        self_type = self.value.st.annotype
                        self_st_type = self.value.st
                    else:
                        self_type = None
                        self_st_type = None
                        assert inspecttools.isstaticmethod(
                            dcls_type, self.attr
                        ), f"{self.attr} of {dcls_type} must be staticmethod"
                    local_ids = []
                    if self.value.st.dcls_info is not None:
                        local_ids = self.value.st.dcls_info._locals_ids

                    new_st = get_parse_cache_checked().cached_parse_func(
                        unbound_func, self_type=self_type,
                        ext_preproc_res=prep_res,
                        external_local_ids=local_ids)
                    if self_st_type is not None:
                        # bound method
                        new_st = new_st.get_bounded_type(self_st_type)
                    new_finfo = new_st.get_func_info_checked()
                    new_finfo.is_property = is_prop
                    if is_prop:
                        assert new_finfo.return_type is not None, f"property {self.attr} of {dcls_type} must have return type"
                        self.st = new_finfo.return_type
                        self.compile_info.property_st = new_st
                    else:
                        self.st = new_st
                    self.st.is_stdlib = self.value.st.is_stdlib
                    if not self.value.st.is_stdlib:
                        # template/inline function need to be compiled in Call.
                        ctx = get_parse_context_checked()
                        if not new_finfo.need_delayed_processing():
                            # this is required when this attribute/name is treated as a function pointer.
                            # template/inline functions can't be used as function pointer, so no need 
                            # to set compiled_uid.
                            creq = ctx.enqueue_func_compile(unbound_func, new_finfo,
                                self_type=self_st_type, is_method_def=is_method_def,
                                is_prop=is_prop)

                            self.st.compiled_uid = creq.get_func_compile_uid()
                        else:
                            creq = ctx.get_compile_req(
                                unbound_func,
                                new_finfo,
                                new_finfo.compilable_meta,
                                is_method_def=is_method_def,
                                self_type=self_st_type,
                                is_prop=is_prop)

                            self.st.delayed_compile_req = creq
                    else:
                        self._update_std_func_meta(unbound_func)
        else:
            if self.value.st.type == PFLExprType.OBJECT or self.value.st.type == PFLExprType.ARRAY:
                annotype = self.value.st.childs[0].annotype
                assert annotype is not None and annotype.raw_type is not None 
                methods = _dftype_with_gen_to_supported_methods(
                    annotype.raw_type)[self.value.st.type]
            elif self.value.st.type == PFLExprType.STRING:
                methods = _PFLTYPE_TO_SUPPORTED_METHODS[self.value.st.type]
            else:
                raise ValueError(
                    f"attr `{self.attr}` is not supported (defined) in type {self.value.st}")
            assert self.attr in methods, f"not supported attr {self.attr} for type {self.value.st}"
            new_st = PFLExprInfo.from_signature(self.attr, methods[self.attr])
            self.st = new_st

    def consteval(self):
        if self.st.type == PFLExprType.FUNCTION:
            finfo = self.st.get_func_info_checked()
            if not finfo.is_property:
                return False 
        operands = self._get_consteval_operands(self.value)
        if operands is not None:
            if hasattr(operands[0], self.attr):
                self.st.metadata = getattr(operands[0], self.attr)
                return True
            else:
                eval_cfg = get_eval_cfg_in_parse_ctx()
                if eval_cfg is not None and not eval_cfg.allow_partial:
                    self_str = unparse_pfl_expr(self)
                    raise PFLEvalError(f"Expr {self_str} value type {type(operands[0]).__name__} don't contain attr {self.attr}", self)
        return False

    def metaeval(self):
        if self.st.func_info is not None and self.st.func_info.is_property:
            assert self.compile_info.property_st is not None 
            operands = self._get_consteval_operands_st(self.value)
            if operands is not None:
                if self.compile_info.property_st.meta_infer is not None:
                    infer_res = self.compile_info.property_st.meta_infer(*operands)
                    if infer_res is not None:
                        assert isinstance(infer_res, PFLMetaInferResult), "meta infer function must return `pfl.PFLMetaInferResult`"
                        self.st.metadata = infer_res.data
                        return True
                    return False
        return self.consteval()
        

@dataclasses.dataclass(kw_only=True)
class PFLConstant(PFLExpr):
    value: Any

    def check_and_infer_type(self):
        self.st = PFLExprInfo.from_value(self.value)
        if not isinstance(self.value, tuple):
            self.st.annotype = parse_type_may_optional_undefined(type(self.value))
        self.st._constexpr_data = self.value
        return self

    def consteval(self):
        self.st.metadata = self.value
        return True


@dataclasses.dataclass
class PFLSlice(PFLExpr):
    lo: Union[Undefined, PFLExpr] = undefined 
    hi: Union[Undefined, PFLExpr] = undefined 
    step: Union[Undefined, PFLExpr] = undefined 
    def check_and_infer_type(self):
        ctx = get_parse_context_checked()
        if not ctx.cfg.allow_partial_type_infer:
            if not is_undefined(self.lo):
                # TODO ellipsis?
                assert self.lo.st.type == PFLExprType.NUMBER, f"{self.lo.st.type}"
            if not is_undefined(self.hi):
                assert self.hi.st.type == PFLExprType.NUMBER, f"{self.hi.st.type}"
            if not is_undefined(self.step):
                assert self.step.st.type == PFLExprType.NUMBER, f"{self.step.st.type}"
        self.st = PFLExprInfo(PFLExprType.SLICE, [])

    def consteval(self):
        lo = None if is_undefined(self.lo) else self.lo.st.metadata
        hi = None if is_undefined(self.hi) else self.hi.st.metadata
        step = None if is_undefined(self.step) else self.step.st.metadata
        if not is_undefined(lo) and not is_undefined(hi) and not is_undefined(step):
            self.st.metadata = slice(lo, hi, step)
            return True
        return False

@dataclasses.dataclass(kw_only=True)
class PFLSubscript(PFLExpr):
    value: PFLExpr
    slice: Union[PFLExpr, Sequence[PFLExpr]]
    is_store: Union[Undefined, bool] = undefined

    def check_and_infer_type(self):
        parse_cfg = get_parse_context_checked().cfg
        allow_partial = parse_cfg.allow_partial_type_infer
        if _is_unknown_or_any(self.value.st):
            assert allow_partial, f"{self.value}"
            self.st = PFLExprInfo(PFLExprType.UNKNOWN)
            return
        slice_has_unk = False
        if isinstance(self.slice, PFLExpr):
            if _is_unknown_or_any(self.slice.st):
                slice_has_unk = True
        else:
            for s in self.slice:
                if _is_unknown_or_any(s.st):
                    slice_has_unk = True
        if slice_has_unk and not parse_cfg.allow_partial_in_slice:
            assert allow_partial, f"{self.slice}"
        assert not self.value.st.is_optional()
        if self.value.st.type == PFLExprType.ARRAY:
            assert not isinstance(self.slice, Sequence)
            if not allow_partial or not slice_has_unk:
                assert self.slice.st.type == PFLExprType.NUMBER, f"slice must be number, but got {self.slice.st.type}"
            self.st = self.value.st.childs[0]
        elif self.value.st.type == PFLExprType.OBJECT:
            assert not isinstance(self.slice, Sequence)
            if not allow_partial or not slice_has_unk:
                assert self.slice.st.type == PFLExprType.STRING, f"slice must be string, but got {self.slice.st.type}"
            self.st = self.value.st.childs[0]
        elif self.value.st.type == PFLExprType.TUPLE:
            assert not isinstance(self.slice, Sequence)
            if not allow_partial or not slice_has_unk:
                if not self.slice.is_const:
                    assert self.value.st.is_all_child_same(), F"only support subscript tuple when all tuple element has same type, {self.value.st}"
                    self.st = self.value.st.childs[0]
                else:
                    assert self.slice.st.type == PFLExprType.NUMBER, f"slice must be number, but got {self.slice.st.type}"
                    success = self.slice.consteval()
                    assert success, f"slice {self.slice} must be consteval"
                    self.st = self.value.st.childs[self.slice.st.metadata_checked]
            else:
                self.st = PFLExprInfo(PFLExprType.UNKNOWN)
        elif self.value.st.type == PFLExprType.DATACLASS_OBJECT:
            if allow_partial and slice_has_unk:
                self.st = PFLExprInfo(PFLExprType.UNKNOWN)
                return 
            resolved_custom_expr = self.value
            dcls_type = resolved_custom_expr.st.get_origin_type_checked()
            # use custom operator in left st if found
            setitem_op_name = "__setitem__"
            getitem_op_name = "__getitem__"
            # when you defined setitem, you must define getItem too.

            getitem_op_func = inspect.getattr_static(dcls_type, getitem_op_name, None)
            assert getitem_op_func is not None, f"can't find {getitem_op_name} in custom type {get_qualname_of_type(dcls_type)}. you must define __getitem__ to use subscript"
            ctx = get_parse_context_checked()
            getitem_op_func_st = get_parse_cache_checked().cached_parse_func(
                getitem_op_func, self_type=self.value.st.annotype)
            finfo = getitem_op_func_st.get_func_info_checked()
            assert len(
                finfo.args
            ) == 2, f"custom operator {getitem_op_name} must have 1 non-self arg, but got {len(finfo.args)}"
            if not isinstance(self.slice, Sequence):
                self.slice.st.check_convertable(
                    finfo.args[1].type,
                    f"custom operator {getitem_op_name}|{getitem_op_func_st} arg")
            assert finfo.return_type is not None, f"custom operator {getitem_op_name}|{getitem_op_func_st} must have return type"
            assert not finfo.is_template() and not finfo.is_always_inline(), "custom operator don't support template or inline."
            self.st = finfo.return_type
            if not resolved_custom_expr.st.is_stdlib:
                # enqueue compile
                creq = ctx.enqueue_func_compile(getitem_op_func, finfo, is_method_def=True, self_type=self.value.st)
                self.st.compiled_uid = creq.get_func_compile_uid()

            if self.is_store == True:
                # validate setitem type
                setitem_op_func = inspect.getattr_static(dcls_type, setitem_op_name, None)
                assert setitem_op_func is not None, f"can't find {setitem_op_name} in custom type {get_qualname_of_type(dcls_type)}."
                setitem_op_func_st = get_parse_cache_checked().cached_parse_func(
                    setitem_op_func, self_type=self.value.st.annotype)
                finfo = setitem_op_func_st.get_func_info_checked()
                assert not finfo.is_template() and not finfo.is_always_inline(), "custom operator don't support template or inline."
                assert len(finfo.args) == 3, f"custom operator {setitem_op_name} must have 2 non-self args, but got {len(finfo.args)}"
                if not resolved_custom_expr.st.is_stdlib:
                    # enqueue compile
                    creq = ctx.enqueue_func_compile(setitem_op_func, finfo, is_method_def=True, self_type=self.value.st)
                    self.st.additional_compiled_uid = creq.get_func_compile_uid()

                if not isinstance(self.slice, Sequence):
                    self.slice.st.check_convertable(
                        finfo.args[1].type,
                        f"custom operator {setitem_op_name}|{setitem_op_func_st} arg")
            else:
                self._update_std_func_meta(getitem_op_func)

        else:
            raise ValueError(f"not support subscript for {self.value.st}")
        if isinstance(self.slice, PFLExpr):
            if PFLExpr.all_constexpr(self.value, self.slice):
                assert not isinstance(self.value.st._constexpr_data, Undefined)
                self.st._constexpr_data = self.value.st._constexpr_data[self.slice.st._constexpr_data]
        return self

    def consteval(self):
        if isinstance(self.slice, PFLExpr):
            operands = self._get_consteval_operands(self.value, self.slice)
            if operands is not None:
                self.st.metadata = operands[0][operands[1]]
                return True
            return False
        else:
            operands = self._get_consteval_operands(self.value, *(self.slice))
            if operands is not None:
                self.st.metadata = operands[0][tuple(operands[1:])]
                return True
            return False

    def metaeval(self):
        if isinstance(self.slice, PFLExpr):

            operands = self._get_consteval_operands_st(
                self.value, self.slice)
            if operands is not None and self.st.meta_infer is not None:
                infer_res = self.st.meta_infer(operands[0], operands[1])
                if infer_res is not None:
                    assert isinstance(infer_res, PFLMetaInferResult), "meta infer function must return `pfl.PFLMetaInferResult`"
                    self.st.metadata = infer_res.data
                    return True
                return False
            return self.consteval()
        else:
            operands = self._get_consteval_operands_st(
                self.value, *self.slice)
            if operands is not None and self.st.meta_infer is not None:
                infer_res = self.st.meta_infer(operands[0], tuple(operands[1:]))
                if infer_res is not None:
                    assert isinstance(infer_res, PFLMetaInferResult), "meta infer function must return `pfl.PFLMetaInferResult`"
                    self.st.metadata = infer_res.data
                    return True
                return False

            return self.consteval()


@dataclasses.dataclass(kw_only=True)
class PFLArray(PFLExpr):
    elts: list[PFLExpr]

    def check_and_infer_type(self):
        if not self.elts:
            self.st = PFLExprInfo(PFLExprType.ARRAY,
                                  [PFLExprInfo(PFLExprType.UNKNOWN)])
            self.st._constexpr_data = []
            return self
        # all elts must be same type
        first_elt = self.elts[0]
        final_st = first_elt.st
        ctx = get_parse_context_checked()
        for elt in self.elts:
            if _is_unknown_or_any(elt.st):
                assert ctx.cfg.allow_partial_type_infer
                self.st = PFLExprInfo(PFLExprType.ARRAY,
                                  [PFLExprInfo(PFLExprType.UNKNOWN)])
                return
            if ctx.cfg.allow_dynamic_container_literal:
                if not first_elt.st.is_equal_type(elt.st):
                    final_st = PFLExprInfo(PFLExprType.ANY)
            else:
                assert first_elt.st.is_equal_type(elt.st), f"all elts must be same type, but got {first_elt.st} and {elt.st}"
        self.st = PFLExprInfo(PFLExprType.ARRAY,
                              [dataclasses.replace(final_st)])
        if PFLExpr.all_constexpr(*self.elts):
            self.st._constexpr_data = list(
                e.st._constexpr_data for e in self.elts)

    def consteval(self):
        operands = self._get_consteval_operands(*self.elts)
        if operands is not None:
            self.st.metadata = operands
            return True
        return False

    def metaeval(self):
        # we need to keep length info of constant array, so metaeval result can be array of undefined.
        self.st.metadata = [e.st.metadata for e in self.elts]
        return True

@dataclasses.dataclass(kw_only=True)
class PFLTuple(PFLExpr):
    elts: list[PFLExpr]
    def check_and_infer_type(self):
        if not self.elts:
            self.st = PFLExprInfo(PFLExprType.TUPLE, [])
            self.st._constexpr_data = ()
            return self
        self.st = PFLExprInfo(PFLExprType.TUPLE,
                              [dataclasses.replace(e.st) for e in self.elts])
        if PFLExpr.all_constexpr(*self.elts):
            self.st._constexpr_data = tuple(
                e.st._constexpr_data for e in self.elts)

    def consteval(self):
        # for tuple, we always store a tuple of metadata
        # even if all meta of elts are undefined.
        # TODO review this
        self.st.metadata = tuple([e.st.metadata for e in self.elts])
        return True

    def metaeval(self):
        # we need to keep length info of constant array, so metaeval result can be tuple of undefined.
        self.st.metadata = tuple(e.st.metadata for e in self.elts)
        return True

@dataclasses.dataclass(kw_only=True)
class PFLDict(PFLExpr):
    keys: list[Optional[PFLExpr]]
    values: list[PFLExpr]

    def check_and_infer_type(self):
        if not self.keys:
            self.st = PFLExprInfo(PFLExprType.OBJECT,
                                  [PFLExprInfo(PFLExprType.UNKNOWN)])
            self.st._constexpr_data = {}

            return self
        value_st: Optional[PFLExprInfo] = None
        value_is_unknown = False
        cfg = get_parse_context_checked().cfg
        allow_partial = cfg.allow_partial_type_infer
        allow_dynamic_container_literal = cfg.allow_dynamic_container_literal
        for key, value in zip(self.keys, self.values):
            if key is not None:
                value_st = value.st
                if _is_unknown_or_any(value_st):
                    value_is_unknown = True
                break
        if value_st is None:
            for key, value in zip(self.keys, self.values):
                if key is None:
                    assert value.st.type == PFLExprType.OBJECT
                    value_st = value.st.childs[0]
                    break
        assert value_st is not None, "shouldn't happen"
        # all keys and values must be same type
        for key, value in zip(self.keys, self.values):
            if key is not None:
                assert key.st.type == PFLExprType.STRING, "object key must be string"
                if not value_is_unknown or not allow_partial:
                    is_eq_type = value_st.is_equal_type(value.st)
                    if allow_dynamic_container_literal:
                        if not is_eq_type:
                            value_st = PFLExprInfo(PFLExprType.ANY)
                    else:
                        assert value_st.is_equal_type(value.st), f"all values must be same type, but got {value_st} and {value.st}"
            else:
                assert value.st.type == PFLExprType.OBJECT
                if not value_is_unknown or not allow_partial:
                    is_eq_type = value_st.is_equal_type(value.st)
                    if allow_dynamic_container_literal:
                        if not is_eq_type:
                            value_st = PFLExprInfo(PFLExprType.ANY)
                    else:
                        assert value_st.is_equal_type(value.st.childs[
                            0]), f"all values must be same type, but got {value_st} and {value.st.childs[0]}"
        self.st = PFLExprInfo(PFLExprType.OBJECT,
                              [dataclasses.replace(value_st)])
        
        if PFLExpr.all_constexpr(*self.keys, *self.values):
            constexpr_data = {}
            for key, value in zip(self.keys, self.values):
                v_cv = value.st._constexpr_data
                if key is None:
                    assert isinstance(v_cv, dict)
                    constexpr_data.update(v_cv)
                else:
                    k_cv = key.st._constexpr_data
                    constexpr_data[k_cv] = v_cv
            self.st._constexpr_data = constexpr_data

    def consteval(self):
        res = {}
        for key, value in zip(self.keys, self.values):
            if key is None:
                if not isinstance(value.st.metadata, Undefined):
                    res.update(value.st.metadata)
                else:
                    return False
            else:
                kv = self._get_consteval_operands(key, value)
                if kv is not None:
                    res[kv[0]] = kv[1]
                else:
                    return False
        self.st.metadata = res
        return True

@dataclasses.dataclass
class _FuncCompileInfo:
    code: str = ""
    path: str = "<string>"
    first_lineno: int = 0
    original: Optional[Any] = None
    meta: Optional[PFLCompileFuncMeta] = None

@dataclasses.dataclass(kw_only=True)
class PFLFunc(PFLAstStmt):
    name: str
    args: list["PFLArg"]
    st: PFLExprInfo
    body: list[PFLAstStmt] = dataclasses.field(default_factory=list)
    ret_st: Optional[PFLExprInfo] = None
    end_scope: Optional[dict[str, PFLExprInfo]] = None
    decorator_list: Optional[list[PFLExpr]] = None
    # for user, compiler don't need this.
    uid: str = ""
    backend: str = ""
    deps: list[str] = dataclasses.field(default_factory=list)

    compile_info: _FuncCompileInfo = dataclasses.field(default_factory=_FuncCompileInfo)
    # user can use compilable meta to store extra info.
    userdata: Union[Undefined, dict[str, Any]] = undefined
    def get_module_import_path(self):
        return self.get_func_uid_no_spec().split("::")[0]

    def get_func_uid_no_spec(self):
        assert self.uid != ""
        uid_parts = UniqueTreeId(self.uid).parts
        return uid_parts[0]

    def get_define_path(self):
        return self.compile_info.path

@dataclasses.dataclass
class _ClassCompileInfo:
    code: str = ""
    first_lineno: int = 0
    original: Optional[Any] = None
    meta: Optional[PFLCompileFuncMeta] = None
    path: str = "<string>"

@dataclasses.dataclass(kw_only=True)
class PFLClass(PFLAstStmt):
    """PFL Class.  
    this ast node only store metadata of a class, all methods
    are stored separately. we can find them by qualname.
    """
    name: str
    st: PFLExprInfo
    # for user, compiler don't need this.
    uid: str = ""
    parent_uid: str = ""
    init_uid: str = ""
    post_init_uid: str = ""
    compile_info: _ClassCompileInfo = dataclasses.field(default_factory=_ClassCompileInfo)

    def get_module_import_path(self):
        assert self.uid != ""
        uid_parts = UniqueTreeId(self.uid).parts
        return uid_parts[0].split("::")[0]

    def get_func_uid_no_spec(self):
        assert self.uid != ""
        uid_parts = UniqueTreeId(self.uid).parts
        return uid_parts[0]

    def get_define_path(self):
        return self.compile_info.path

@dataclasses.dataclass
class _ModCompileInfo:
    code: str = ""
    path: str = "<string>"

@dataclasses.dataclass(kw_only=True)
class PFLModule(PFLAstNodeBase):
    uid: str
    body: list[PFLAstStmt] = dataclasses.field(default_factory=list)
    compile_info: _ModCompileInfo = dataclasses.field(default_factory=_ModCompileInfo)

    def get_all_compiled(self):
        res: dict[str, Union[PFLFunc, PFLClass]] = {}
        for stmt in self.body:
            if isinstance(stmt, (PFLFunc, PFLClass)):
                res[stmt.uid] = stmt 
        return res 

def iter_fields(node):
    """
    Yield a tuple of ``(fieldname, value)`` for each field in ``node._fields``
    that is present on *node*.
    """
    for field in dataclasses.fields(node):
        try:
            yield field.name, getattr(node, field.name)
        except AttributeError:
            pass


def iter_child_nodes(node: PFLAstNodeBase):
    """
    Yield all direct child nodes of *node*, that is, all fields that are nodes
    and all items of fields that are lists of nodes.
    """
    for field in dataclasses.fields(node):
        field_value = getattr(node, field.name)
        if isinstance(field_value, PFLAstNodeBase):
            yield field_value
        elif isinstance(field_value, list):
            for item in field_value:
                if isinstance(item, PFLAstNodeBase):
                    yield item

def walk(node):
    """
    Recursively yield all descendant nodes in the tree starting at *node*
    (including *node* itself), in no specified order.  This is useful if you
    only want to modify nodes in place and don't care about the context.
    """
    from collections import deque
    todo = deque([node])
    while todo:
        node = todo.popleft()
        todo.extend(iter_child_nodes(node))
        yield node

class NodeVisitor(object):
    def visit(self, node):
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        for field, value in iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, PFLAstNodeBase):
                        self.visit(item)
            elif isinstance(value, PFLAstNodeBase):
                self.visit(value)

class NodeTransformer(NodeVisitor):
    def generic_visit(self, node):
        for field, old_value in iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, PFLAstNodeBase):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, PFLAstNodeBase):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, PFLAstNodeBase):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

_PFL_UNPARSE_BIN_TYPE_TO_OP = {
    BinOpType.ADD: "+",
    BinOpType.SUB: "-",
    BinOpType.MULT: "*",
    BinOpType.DIV: "/",
    BinOpType.FLOOR_DIV: "//",
    BinOpType.MOD: "%",
    BinOpType.POW: "**",
    BinOpType.LSHIFT: "<<",
    BinOpType.RSHIFT: ">>",
    BinOpType.BIT_OR: "|",
    BinOpType.BIT_XOR: "^",
    BinOpType.BIT_AND: "&",
}

_PFL_UNPARSE_UNARY_TYPE_TO_OP = {
    UnaryOpType.INVERT: "~",
    UnaryOpType.NOT: "not",
    UnaryOpType.UADD: "+",
    UnaryOpType.USUB: "-",
}

_PFL_UNPARSE_COMPARE_TYPE_TO_OP = {
    CompareType.EQUAL: "==",
    CompareType.NOT_EQUAL: "!=",
    CompareType.LESS: "<",
    CompareType.LESS_EQUAL: "<=",
    CompareType.GREATER: ">",
    CompareType.GREATER_EQUAL: ">=",
    CompareType.IN: "in",
    CompareType.NOT_IN: "not in",
}

def unparse_pfl_expr(expr: PFLExpr) -> str:
    """
    Unparse a PFLExpr to a string representation.
    """
    if isinstance(expr, PFLName):
        return expr.id
    elif isinstance(expr, PFLAttribute):
        return f"{unparse_pfl_expr(expr.value)}.{expr.attr}"
    elif isinstance(expr, PFLConstant):
        return repr(expr.value)
    elif isinstance(expr, PFLSlice):
        lo_exist = not is_undefined(expr.lo)
        hi_exist = not is_undefined(expr.hi)
        step_exist = not is_undefined(expr.step)
        lo_str = "" if is_undefined(expr.lo) else unparse_pfl_expr(expr.lo)
        hi_str = "" if is_undefined(expr.hi) else unparse_pfl_expr(expr.hi)
        step_str = "" if is_undefined(expr.step) else unparse_pfl_expr(
            expr.step)

        defined_cnt = int(lo_exist) + int(hi_exist) + int(step_exist)
        if defined_cnt == 0:
            return ":"
        elif step_exist:
            return f"{lo_str}:{hi_str}:{step_str}"
        else:
            return f"{lo_str}:{hi_str}"
    elif isinstance(expr, PFLSubscript):
        if isinstance(expr.slice, Sequence):
            slice_strs = [unparse_pfl_expr(s) for s in expr.slice]
            slice_str = ", ".join(slice_strs)
        else:
            slice_str = unparse_pfl_expr(expr.slice)
        if isinstance(expr.value, PFLName):
            value_str = expr.value.id
        else:
            value_str = unparse_pfl_expr(expr.value)
        return f"{value_str}[{slice_str}]"
    elif isinstance(expr, PFLArray):
        return "[" + ", ".join(unparse_pfl_expr(elt)
                               for elt in expr.elts) + "]"
    elif isinstance(expr, PFLTuple):
        return "(" + ", ".join(unparse_pfl_expr(elt)
                               for elt in expr.elts) + ")"
    elif isinstance(expr, PFLDict):
        strs = []
        for k, v in zip(expr.keys, expr.values):
            if k is None:
                strs.append(f"**{unparse_pfl_expr(v)}")
            else:
                strs.append(f"{unparse_pfl_expr(k)}: {unparse_pfl_expr(v)}")
        return "{" + ", ".join(strs) + "}"
    elif isinstance(expr, PFLBoolOp):
        if expr.op == BoolOpType.AND:
            op = "and"
        else:
            op = "or"
        return "(" + f" {op} ".join(unparse_pfl_expr(value) for value in expr.values) + ")"
    elif isinstance(expr, PFLBinOp):
        return f"({unparse_pfl_expr(expr.left)} {_PFL_UNPARSE_BIN_TYPE_TO_OP[expr.op]} {unparse_pfl_expr(expr.right)})"
    elif isinstance(expr, PFLUnaryOp):
        return f"{_PFL_UNPARSE_UNARY_TYPE_TO_OP[expr.op]}{unparse_pfl_expr(expr.operand)}"
    elif isinstance(expr, PFLCompare):
        return f"({unparse_pfl_expr(expr.left)} {_PFL_UNPARSE_COMPARE_TYPE_TO_OP[expr.op]} {unparse_pfl_expr(expr.right)})"
    elif isinstance(expr, PFLCall):
        args_strs = [unparse_pfl_expr(arg) for arg in expr.args]
        if not is_undefined(expr.keys) and not is_undefined(expr.vals):
            args_strs += [
                f"{n}={unparse_pfl_expr(arg)}" for n, arg in zip(expr.keys, expr.vals)
            ]
        args_str = ", ".join(args_strs)
        return f"{unparse_pfl_expr(expr.func)}({args_str})"
    elif isinstance(expr, PFLIfExp):
        return f"({unparse_pfl_expr(expr.body)} if {unparse_pfl_expr(expr.test)} else {unparse_pfl_expr(expr.orelse)})"
    else:
        raise NotImplementedError(f"Unrecognized PFLExpr type: {type(expr)}")


def unparse_pfl_ast_to_lines(stmt: PFLAstNodeBase, depth: int = 0) -> list[str]:
    """
    Unparse a PFLAstNodeBase to a list of string lines.
    This function is used to convert the PFL AST back to a human-readable format.
    """
    res: list[str] = []
    if isinstance(stmt, PFLExpr):
        res.append(unparse_pfl_expr(stmt))
    elif isinstance(stmt, PFLArg):
        msg = f"{stmt.arg}"
        if stmt.annotation is not None:
            msg = f"{msg}: {stmt.annotation}"
        if stmt.default is not None:
            msg = f"{msg} = {unparse_pfl_expr(stmt.default)}"
        res.append(msg)

    elif isinstance(stmt, (PFLAssign, PFLAnnAssign)):
        if stmt.value is not None:
            target_str = unparse_pfl_expr(stmt.target)
            value_str = unparse_pfl_expr(stmt.value)
            if isinstance(stmt, PFLAnnAssign):
                res.append(f"{target_str}: {stmt.target.st.annotype} = {value_str}")
            else:
                res.append(f"{target_str} = {value_str}") 
        
    elif isinstance(stmt, (PFLIf)):
        testAndBodyArr = stmt.get_flatten_test_body()
        for i in range(len(testAndBodyArr)):
            test, body = testAndBodyArr[i]
            if test is not None:
                if (i == 0):
                    res.append(f"if {unparse_pfl_expr(test)}:")
                else:
                    res.append(f"elif {unparse_pfl_expr(test)}:")
                body_lines = sum([unparse_pfl_ast_to_lines(x, 1) for x in body], [])
                res.extend(body_lines)
            else:
                # else case
                if len(body) > 0:
                    res.append("else:")
                    body_lines = sum([unparse_pfl_ast_to_lines(x, 1) for x in body], [])
                    res.extend(body_lines)
    elif isinstance(stmt, PFLAugAssign):
        target_str = unparse_pfl_expr(stmt.target)
        value_str = unparse_pfl_expr(stmt.value)
        res.append(f"{target_str} {_PFL_UNPARSE_BIN_TYPE_TO_OP[stmt.op]}= {value_str}")

    elif isinstance(stmt, PFLFor):
        target_str = unparse_pfl_expr(stmt.target)
        iter_str = unparse_pfl_expr(stmt.iter)
        res.append(f"for {target_str} in {iter_str}:")
        body_lines = sum([unparse_pfl_ast_to_lines(x, 1) for x in stmt.body], [])
        res.extend(body_lines)
    elif isinstance(stmt, PFLWhile):
        test_str = unparse_pfl_expr(stmt.test)
        res.append(f"while {test_str}:")
        body_lines = sum([unparse_pfl_ast_to_lines(x, 1) for x in stmt.body], [])
        res.extend(body_lines)
    elif isinstance(stmt, PFLExprStmt):
        res.append(unparse_pfl_expr(stmt.value))
    elif isinstance(stmt, PFLReturn):
        if stmt.value is not None:
            res.append(f"return {unparse_pfl_expr(stmt.value)}")
        else:
            res.append("return")
    elif isinstance(stmt, PFLBreak):
        res.append("break")
    elif isinstance(stmt, PFLContinue):
        res.append("continue")
    elif isinstance(stmt, PFLFunc):
        args_strs: list[str] = []
        for arg in stmt.args:
            msg = arg.arg
            if arg.annotation is not None:
                msg = f"{msg}: {arg.annotation}"
            if arg.default is not None:
                default_str = unparse_pfl_expr(arg.default)
                msg = f"{msg} = {default_str}"
            args_strs.append(msg)
        
        args_str = ", ".join(args_strs)
        res.append(f"def {stmt.name}({args_str}):")
        body_lines = sum([unparse_pfl_ast_to_lines(x, 1) for x in stmt.body], [])
        res.extend(body_lines)
    else:
        raise NotImplementedError(f"Unrecognized PFLAstNodeBase type: {type(stmt)}")
    return [f"{' ' * (depth * 4)}{line}" for line in res]

def unparse_pfl_ast(node: PFLAstNodeBase) -> str:
    """
    Unparse a PFLAstNodeBase to a string representation.
    """
    assert isinstance(node, PFLAstNodeBase)
    lines = unparse_pfl_ast_to_lines(node)
    return "\n".join(lines)

class PFLAstParseError(Exception):

    def __init__(self, msg: str, node: ast.AST):
        super().__init__(msg)
        self.node = node

class PFLEvalError(Exception):

    def __init__(self, msg: str, node: Optional[PFLAstNodeBase], traceback_set: bool = False):
        super().__init__(msg)
        self.node = node
        self.traceback_set = traceback_set


class PFLTreeNodeFinder:
    """find pfl ast node by lineno and col offset.
    """
    def __init__(self, node: PFLAstNodeBase, node_cls_tuple: tuple[Type[PFLAstNodeBase], ...]):
        all_nodes: list[PFLAstNodeBase] = []
        for child_node in walk(node):
            if isinstance(child_node, node_cls_tuple):
                all_nodes.append(child_node)

        # sort by lineno and col offset
        all_nodes.sort(key=self._sort_key)
        self._all_nodes = all_nodes
        self._hi = (self._all_nodes[-1].source_loc[0], self._all_nodes[-1].source_loc[1])

    def _sort_key(self, node: PFLAstNodeBase):
        end_l = node.source_loc[2]
        end_c = node.source_loc[3]
        if end_l is None:
            end_l = -1
        if end_c is None:
            end_c = -1
        return (node.source_loc[0], node.source_loc[1], end_l, end_c)

    def find_nearest_node_by_line_col(self, lineno: int, col_offset: int):
        cur_lc = (lineno, col_offset)
        idx = bisect.bisect_left(self._all_nodes, cur_lc, key=lambda n: (n.source_loc[0], n.source_loc[1]))
        # print(idx, len(self._all_nodes), self._all_nodes[-1].source_loc)
        if idx >= len(self._all_nodes):
            last_node = self._all_nodes[-1]
            end_l = last_node.source_loc[2]
            end_c = last_node.source_loc[3]
            if end_l is None or end_c is None:
                return None  
            if cur_lc >= (last_node.source_loc[0], last_node.source_loc[1]) and cur_lc <= (end_l, end_c):
                return last_node 
            return None 
        cur_node = self._all_nodes[idx]
        if cur_node.get_range_start() <= cur_lc and (end_lc := cur_node.get_range_end()) is not None and cur_lc <= end_lc:
            return cur_node
        if idx < 1:
            return None 
        # look backward to find suitable node
        node_to_ret: Optional[PFLAstNodeBase] = None
        for j in range(idx - 1, -1, -1):
            node = self._all_nodes[j]
            end_l = node.source_loc[2]
            end_c = node.source_loc[3]
            if end_l is None or end_c is None:
                continue 
            if node.in_range(cur_lc[0], cur_lc[1]):
                node_to_ret = node 
                continue 
            else:
                break
        return node_to_ret

    def find_nearest_node_by_line(self, lineno: int):
        idx = bisect.bisect_left(self._all_nodes, lineno, key=lambda n: n.source_loc[0])
        if idx >= len(self._all_nodes):
            last_node = self._all_nodes[-1]
            end_l = last_node.source_loc[2]
            if end_l is None :
                return None  
            if lineno >= last_node.source_loc[0] and lineno <= end_l:
                return last_node 
            return None 
        cur_node = self._all_nodes[idx]
        if cur_node.get_range_start()[0] <= lineno and (end_lc := cur_node.get_range_end()) is not None and lineno <= end_lc[0]:
            return cur_node
        if idx < 1:
            return None 
        # look backward to find suitable node
        node_to_ret: Optional[PFLAstNodeBase] = None
        for j in range(idx - 1, -1, -1):
            node = self._all_nodes[j]
            end_l = node.source_loc[2]
            if end_l is None:
                continue 
            if node.in_range_lineno(lineno):
                node_to_ret = node 
                continue 
            else:
                break
        return node_to_ret

class PFLTreeExprFinder:
    """find pfl ast node by lineno/end_lineno and col_offset/end_col_offset.

    use a simple hash map to implement.
    """
    def __init__(self, node: PFLAstNodeBase):
        all_nodes: dict[SourceLocType, PFLExpr] = {}
        for child_node in walk(node):
            if isinstance(child_node, PFLExpr):
                all_nodes[child_node.source_loc] = child_node
        self._all_nodes = all_nodes
        
    def find_expr_by_source_loc(self, source_loc: SourceLocType) -> Optional[PFLExpr]:
        """find expr by source loc, return None if not found."""
        return self._all_nodes.get(source_loc, None)