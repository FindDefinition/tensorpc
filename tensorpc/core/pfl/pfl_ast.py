import ast
import bisect
from collections.abc import Sequence
import enum
import inspect
import traceback
from typing import Any, Callable, ClassVar, Generic, Optional, Type, TypeVar, Union, cast
from typing_extensions import TypeVarTuple, Unpack
import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core import inspecttools
from tensorpc.core.annolib import (AnnotatedType, Undefined, is_undefined,
                                   parse_type_may_optional_undefined,
                                   undefined)
from tensorpc.core.pfl.constants import PFL_BUILTIN_PROXY_INIT_FN, PFL_STDLIB_FUNC_META_ATTR
from tensorpc.core.moduleid import get_qualname_of_type

from .core import (PFLCompileFuncMeta, PFLExprInfo, PFLExprType, PFLStdlibFuncMeta, PFLMetaInferResult, get_eval_cfg_in_parse_ctx, get_parse_cache_checked,
                   get_parse_context_checked, param_fn, varparam_fn)


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
            "extend":
            inspect.Signature([param_fn("iterable", dict[str, vt])],
                              return_annotation=None),  # extend
            "remove":
            inspect.Signature([param_fn("key", str)],
                              return_annotation=None),  # remove
            "pop":
            inspect.Signature([param_fn("key", str)],
                              return_annotation=vt),  # pop
            # "clear": inspect.Signature([], return_annotation=None), # clear
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

    STMT_MASK = 0x100
    ASSIGN = 0x101
    IF = 0x102
    EXPR_STMT = 0x103
    AUG_ASSIGN = 0x104
    FOR = 0x105
    WHILE = 0x106
    ANN_ASSIGN = 0x107
    RETURN = 0x108
    BREAK = 0x109
    CONTINUE = 0x10A

    EXPR_MASK = 0x200

    BOOL_OP = 0x201
    BIN_OP = 0x202
    UNARY_OP = 0x203
    COMPARISON = 0x204
    ARRAY = 0x205
    CALL = 0x206
    NAME = 0x207
    CONSTANT = 0x208
    SUBSCRIPT = 0x209
    DICT = 0x20A
    ATTR = 0x20B
    IF_EXP = 0x20C
    SLICE = 0x20D
    TUPLE = 0x20E

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


class BoolOpType(enum.IntEnum):
    AND = 0
    OR = 1


class BinOpType(enum.IntEnum):
    ADD = 0
    SUB = 1
    MULT = 2
    DIV = 3
    MOD = 4
    POW = 5
    LSHIFT = 6
    RSHIFT = 7
    BIT_OR = 8
    BIT_XOR = 9
    BIT_AND = 10
    FLOOR_DIV = 11
    MATMUL = 12


class UnaryOpType(enum.IntEnum):
    INVERT = 0
    NOT = 1
    UADD = 2
    USUB = 3


class CompareType(enum.IntEnum):
    EQUAL = 0
    NOT_EQUAL = 1
    LESS = 2
    LESS_EQUAL = 3
    GREATER = 4
    GREATER_EQUAL = 5
    IS = 6
    IS_NOT = 7
    IN = 8
    NOT_IN = 9

_PFL_UNARY_TYPE_TO_METHOD_NAME = {
    UnaryOpType.INVERT: "__invert__",
    UnaryOpType.NOT: "__not__",
    UnaryOpType.UADD: "__pos__",
    UnaryOpType.USUB: "__neg__",
}

_PFL_BINARY_TYPE_TO_METHOD_NAME = {
    CompareType.EQUAL: "__eq__",
    CompareType.NOT_EQUAL: "__ne__",
    CompareType.LESS: "__lt__",
    CompareType.LESS_EQUAL: "__le__",
    CompareType.GREATER: "__gt__",
    CompareType.GREATER_EQUAL: "__ge__",
    CompareType.IN: "__contains__",
    CompareType.NOT_IN: "__contains__",
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
    CompareType.EQUAL: "__ne__",
    CompareType.NOT_EQUAL: "__eq__",
    CompareType.LESS: "__ge__",
    CompareType.LESS_EQUAL: "__gt__",
    CompareType.GREATER: "__le__",
    CompareType.GREATER_EQUAL: "__lt__",
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

@dataclasses.dataclass
class PFLAstNodeBase:
    type: PFLASTType
    # record lineno/col_offset from ast node for debug
    source_loc: tuple[int, int, Optional[int], Optional[int]]

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
    is_const: Union[bool, Undefined] = undefined

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

    def _update_func_meta(self, fn: Callable):
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

    def check_and_infer_type(self):
        resolved_custom_expr = None
        if self.target.st.type == PFLExprType.DATACLASS_OBJECT:
            resolved_custom_expr = self.target
        op_func = None
        is_custom_type = False
        if resolved_custom_expr is not None:
            assert not resolved_custom_expr.st.is_temp, f"temp dataclass {resolved_custom_expr.st} (not stdlib) can't be used in custom operator"
            dcls_type = resolved_custom_expr.st.get_origin_type_checked()
            # use custom operator in left st if found
            op_name = _PFL_AUG_ASSIGN_METHOD_NAME[self.op]
            op_func = inspect.getattr_static(dcls_type, op_name, None)
            assert op_func is not None, f"can't find {op_name} in custom type {get_qualname_of_type(dcls_type)}"
            op_func_st = get_parse_cache_checked().cached_parse_func(
                op_func, True, self_type=self.target.st.annotype)
            assert len(
                op_func_st.childs
            ) == 1, f"custom operator {op_name} must have one arg, but got {len(op_func_st.childs)}"
            assert self.value.st.is_convertable(
                op_func_st.childs[0]
            ), f"aug assign value {self.value.st} not convertable to {op_func_st.childs[0]}"
            is_custom_type = True
            custom_res_type = op_func_st.return_type
            assert custom_res_type is not None, f"custom operator {op_name} must have return type"
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
    _new: Union[list[str], Undefined] = undefined
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
    left: PFLExpr
    right: PFLExpr

    def check_and_infer_type(self):
        assert self.left.st.support_bool_op()
        assert self.right.st.support_bool_op()
        self.st = PFLExprInfo(PFLExprType.BOOL)
        self.is_const = PFLExpr.all_constexpr(self.left, self.right)
        if self.is_const == True:
            self.st._constexpr_data = self.run(self.left.st._constexpr_data, self.right.st._constexpr_data)
        return self

    def consteval(self):
        operands = self._get_consteval_operands(self.left, self.right)
        if operands is not None:
            if self.op == BoolOpType.AND:
                self.st.metadata = operands[0] and operands[1]
            else:
                self.st.metadata = operands[0] or operands[1]
            return True
        return False

    def run(self, left: Any, right: Any) -> Any:
        if self.op == BoolOpType.AND:
            return left and right
        else:
            return left or right

@dataclasses.dataclass(kw_only=True)
class PFLUnaryOp(PFLExpr):
    op: UnaryOpType
    operand: PFLExpr

    def check_and_infer_type(self):
        # overrideable operators
        left = self.operand
        is_custom_type: bool = False
        custom_res_type = None
        resolved_custom_expr = None
        if left.st.type == PFLExprType.DATACLASS_OBJECT:
            resolved_custom_expr = left
        op_func = None
        if resolved_custom_expr is not None:
            assert not resolved_custom_expr.st.is_temp, f"temp dataclass {resolved_custom_expr.st} (not stdlib) can't be used in custom operator"
            dcls_type = resolved_custom_expr.st.get_origin_type_checked()
            # use custom operator in left st if found
            op_name = _PFL_UNARY_TYPE_TO_METHOD_NAME[self.op]
            op_func = inspect.getattr_static(dcls_type, op_name, None)
            assert op_func is not None, f"can't find {op_name} in custom type {get_qualname_of_type(dcls_type)}"
            op_func_st = get_parse_cache_checked().cached_parse_func(
                op_func, True, self_type=self.operand.st.annotype)
            assert len(
                op_func_st.childs
            ) == 1, f"custom operator {op_name} must have one arg, but got {len(op_func_st.childs)}"
            is_custom_type = True
            custom_res_type = op_func_st.return_type
            assert custom_res_type is not None, f"custom operator {op_name} must have return type"
        if not is_custom_type:
            self.operand.st.check_support_binary_op("left")
            if self.op == UnaryOpType.NOT:
                self.st = dataclasses.replace(self.operand.st, type=PFLExprType.BOOL, annotype=parse_type_may_optional_undefined(bool))
            elif (self.op == UnaryOpType.UADD or self.op == UnaryOpType.USUB) and self.operand.st.type == PFLExprType.BOOL:
                self.st = dataclasses.replace(self.operand.st, type=PFLExprType.NUMBER, annotype=parse_type_may_optional_undefined(int))
            else:
                self.st = dataclasses.replace(self.operand.st)
        else:
            assert custom_res_type is not None
            self.st = custom_res_type

        self.is_const = self.operand.is_const
        if self.is_const == True:
            self.st._constexpr_data = self.run(self.operand.st._constexpr_data)
        if op_func is not None:
            self._update_func_meta(op_func)
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
        assert self.test.st.can_cast_to_bool(
        ), f"test must be convertable to bool, but got {self.test.st}"
        assert self.body.st.is_equal_type(self.orelse.st), f"body and orelse must be same type, but got {self.body.st} and {self.orelse.st}"
        self.st = dataclasses.replace(self.body.st)
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

    def resolve_custom_type(self, op: Union[BinOpType, CompareType]):
        # TODO if left and right are both custom type
        # overrideable operators
        left = self.left
        right = self.right
        is_custom_type: bool = False
        custom_res_type = None
        resolved_custom_expr = None
        resolved_op_func = None
        op_name = _PFL_BINARY_TYPE_TO_METHOD_NAME[op]
        if left.st.type == PFLExprType.DATACLASS_OBJECT:
            dcls_type = left.st.get_origin_type_checked()
            op_func = inspect.getattr_static(dcls_type, op_name, None)
            if op_func is not None:
                resolved_custom_expr = left
                resolved_op_func = op_func
        if resolved_custom_expr is None:
            if right.st.type == PFLExprType.DATACLASS_OBJECT:
                rop_name = _PFL_BINARY_TYPE_TO_REVERSE_METHOD_NAME[op]
                dcls_type = right.st.get_origin_type_checked()
                op_func = inspect.getattr_static(dcls_type, rop_name, None)
                if op_func is not None:
                    resolved_custom_expr = right
                    resolved_op_func = op_func
                    self.is_right = True
        if resolved_custom_expr is not None:
            assert resolved_op_func is not None 
            assert not resolved_custom_expr.st.is_temp, f"temp dataclass {resolved_custom_expr.st} (not stdlib) can't be used in custom operator"
            op_func_st = get_parse_cache_checked().cached_parse_func(
                resolved_op_func, True, self_type=resolved_custom_expr.st.annotype)
            overload_infos = [op_func_st]
            if op_func_st.overloads is not None:
                overload_infos.extend(op_func_st.overloads)
            overload_scores: list[tuple[int, int, PFLExprInfo]] = []
            errors: list[str] = []
            assert len(
                op_func_st.childs
            ) == 1, f"custom operator {op_name} must have one arg, but got {len(op_func_st.childs)}"

            for i, overload in enumerate(overload_infos):
                try:
                    if self.is_right == True:
                        st_to_check = left.st
                    else:
                        st_to_check = right.st
                    st_to_check.check_convertable(overload.childs[0],
                                        f"custom operator {op_name} overload {overload} arg")
                    if st_to_check.is_equal_type(overload.childs[0]):
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
            self.op)
        if not is_custom_type:
            promotion_type = self.left.st.check_support_binary_op_and_promotion(self.right.st)
            self.st = PFLExprInfo(PFLExprType.NUMBER, annotype=promotion_type)
        else:
            assert custom_res_type is not None
            self.st = custom_res_type
        self.is_const = PFLExpr.all_constexpr(self.left, self.right)
        if self.is_const == True:
            self.st._constexpr_data = self.run(self.left.st._constexpr_data, self.right.st._constexpr_data)
        if op_func is not None:
            self._update_func_meta(op_func)
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
        is_custom_type = False
        if self.op == CompareType.IN or self.op == CompareType.NOT_IN:
            # contains operator don't support custom datatype
            assert self.left.st.type != PFLExprType.DATACLASS_OBJECT
            assert self.right.st.type != PFLExprType.DATACLASS_OBJECT
        op_func = None
        custom_type_res = None
        if not (self.op == CompareType.IS or self.op == CompareType.IS_NOT):
            # overrideable operators
            is_custom_type, custom_type_res, op_func = self.resolve_custom_type(self.op)

        if not is_custom_type:
            # all operands are base type
            if not (self.op == CompareType.EQUAL or self.op
                    == CompareType.NOT_EQUAL or self.op == CompareType.IS
                    or self.op == CompareType.IS_NOT):
                # handle string-type compares
                if self.op == CompareType.IN or self.op == CompareType.NOT_IN:
                    assert self.left.st.type == PFLExprType.STRING and self.right.st.type == PFLExprType.OBJECT, f"left must be string and right must be object, but got {self.left.st.type} and {self.right.st.type}"
                else:
                    self.left.st.check_support_binary_op("left")
                    self.right.st.check_support_binary_op("right")
            self.st = PFLExprInfo(PFLExprType.BOOL, annotype=parse_type_may_optional_undefined(bool))
        else:
            assert custom_type_res is not None 
            self.st = custom_type_res
        self.is_const = PFLExpr.all_constexpr(self.left, self.right)
        if self.is_const == True:
            self.st._constexpr_data = self.run(self.left.st._constexpr_data, self.right.st._constexpr_data)
        if op_func is not None:
            self._update_func_meta(op_func)
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

    is_ctor: Union[Undefined, bool] = undefined

    def check_and_infer_type(self):
        # validate args
        is_ctor: bool = False
        if self.func.st.type == PFLExprType.DATACLASS_TYPE:
            # create std objects
            is_ctor = True
        self.is_ctor = is_ctor
        assert self.func.st.type == PFLExprType.FUNCTION or self.func.st.type == PFLExprType.DATACLASS_TYPE, f"func must be function/dcls, but got {self.func.st.type}"
        overloads: list[tuple[list[PFLExprInfo], bool, PFLExprInfo]] = []
        if self.func.st.type == PFLExprType.FUNCTION:
            overload_infos = [self.func.st]
            if self.func.st.overloads is not None:
                overload_infos.extend(self.func.st.overloads)
            for overload_info in overload_infos:
                last_is_vaarg = False
                if overload_info.childs:
                    last_is_vaarg = overload_info.childs[-1].is_vaargs
                typevar_map = self._get_typevar_map(overload_info, last_is_vaarg)
                if typevar_map:
                    overload_info = overload_info.typevar_substitution(typevar_map)
                assert overload_info.return_type is not None, f"func {self.func} overload {overload_info} must have return type"
                func_arg_types = overload_info.childs
                overloads.append((func_arg_types, last_is_vaarg, overload_info.return_type))
        elif self.func.st.type == PFLExprType.DATACLASS_TYPE:
            # TODO: currently we only support dataclasses init as constexpr function 
            self.is_const = PFLExpr.all_constexpr(*self.args)
            if not is_undefined(self.vals):
                self.is_const &= PFLExpr.all_constexpr(*self.vals)
            func_arg_types, return_type = self.func.st._parse_dataclass_ctor()
            overloads.append((func_arg_types, False, return_type))
        else:
            raise NotImplementedError
        errors: list[str] = []
        overload_scores: list[tuple[int, int, PFLExprInfo]] = []
        for i, overload in enumerate(overloads):
            try:
                score = self._check_single_overload(overload[0], overload[1])
                overload_scores.append((score, i, overload[2]))
            except BaseException as e:
                # traceback.print_exc()
                errors.append(str(e))
        if not overload_scores:
            error_msg = f"func {self.func.st} overloads not match args {[a.st for a in self.args]} kws {self.keys}. error:\n"
            for e in errors:
                error_msg += f"  - {e}\n"
            print(error_msg)
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
                if self.is_const == True:
                    args = []
                    kwargs = {}
                    for a in self.args:
                        assert a.is_const, "when you define static type infer, all arguments must be constexpr."
                        args.append(a.st._constexpr_data)
                    if not is_undefined(self.keys):
                        assert not is_undefined(self.vals)
                        for k, v in zip(self.keys, self.vals):
                            assert v.is_const, "when you define static type infer, all arguments must be constexpr."
                            kwargs[k] = v.st._constexpr_data
                    if self.func.st.proxy_dcls is not None:
                        self.st._constexpr_data = inspect.getattr_static(self.func.st.proxy_dcls, PFL_BUILTIN_PROXY_INIT_FN)(*args, **kwargs)
                    else:
                        self.st._constexpr_data = self.func.st.get_origin_type_checked()(*args, **kwargs)

    @staticmethod
    def _get_dataclass_ctor_sig(st: PFLExprInfo):
        annotype = st.annotype
        assert annotype is not None
        func_arg_types: list[PFLExprInfo] = []
        for k, f in annotype.get_dataclass_field_annotated_types().items():
            finfo = PFLExprInfo.from_annotype(f, is_type=False)
            finfo.arg_name = k
            func_arg_types.append(finfo)
        ret_type = dataclasses.replace(st,
                                        type=PFLExprType.DATACLASS_OBJECT,
                                        childs=[],
                                        return_type=None)
        return func_arg_types, ret_type

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

    def _get_typevar_map(self, sig_info: PFLExprInfo, last_is_vaarg: bool) -> dict[TypeVar, PFLExprInfo]:
        assert sig_info.type == PFLExprType.FUNCTION or sig_info.type == PFLExprType.DATACLASS_TYPE
        func_arg_types = sig_info.childs
        res: dict[TypeVar, PFLExprInfo] = {}
        for i, a in enumerate(self.args):
            if i >= len(func_arg_types):
                # TODO should we return or raise error when overloads isn't suitable?
                return {} 
            if not last_is_vaarg:
                func_arg = func_arg_types[i]
                if func_arg.type == PFLExprType.GENERIC_TYPE:
                    self._check_typevar_substitution(a.st, func_arg, res)
            else:
                if i < len(func_arg_types) - 1:
                    func_arg = func_arg_types[i]
                    right = func_arg
                else:
                    right = func_arg_types[-1]
                if right.type == PFLExprType.GENERIC_TYPE:
                    self._check_typevar_substitution(a.st, right, res)

        if not isinstance(self.keys, Undefined):
            assert not isinstance(self.vals, Undefined)
            for name, a in zip(self.keys, self.vals):
                for arg in func_arg_types:
                    assert arg.arg_name is not None
                    if name == arg.arg_name:
                        if arg.type == PFLExprType.GENERIC_TYPE:
                            self._check_typevar_substitution(a.st, arg, res)
                        break
        return res

    def _match_arg_sts_to_sig(self, func_arg_types: list[PFLExprInfo], last_is_vaarg: bool):
        if last_is_vaarg:
            assert isinstance(self.keys, Undefined), "don't support use kwargs with *args"
        if not last_is_vaarg:
            assert len(self.args) <= len(
                func_arg_types
            ), f"func {self.func.st} expect {len(func_arg_types)} args, but got {len(self.args)}"
        res: list[tuple[PFLExprInfo, list[PFLExpr]]] = []
        for func_arg_st in func_arg_types:
            res.append((func_arg_st, []))
        for i, a in enumerate(self.args):
            if not last_is_vaarg:
                res[i][1].append(a)
            else:
                if i < len(func_arg_types) - 1:
                    res[i][1].append(a)
                else:
                    res[-1][1].append(a)
        if not isinstance(self.keys, Undefined):
            assert not isinstance(self.vals, Undefined)
            for name, a in zip(self.keys, self.vals):
                found = False
                for i, arg in enumerate(func_arg_types):
                    assert arg.arg_name is not None
                    if name == arg.arg_name:
                        found = True 
                        res[i][1].append(a)
                        break
                if not found:
                    raise ValueError(f"can't find arg {name} in function {self.func.st}")
        return res

    def _check_single_overload(self, func_arg_types: list[PFLExprInfo], last_is_vaarg: bool):
        if last_is_vaarg:
            assert isinstance(self.keys, Undefined), "don't support use kwargs with *args"
        if not last_is_vaarg:
            assert len(self.args) <= len(
                func_arg_types
            ), f"func {self.func.st} expect {len(func_arg_types)} args, but got {len(self.args)}"
        
        func_arg_st_param = self._match_arg_sts_to_sig(func_arg_types, last_is_vaarg)
        match_score = 0
        cnt = 0
        for func_arg_st, args in func_arg_st_param:
            if not args:
                if not func_arg_st.is_vaargs:
                    assert not is_undefined(func_arg_st.default), f"func overload {func_arg_types} arg {func_arg_st.arg_name}({cnt}) has no matched value and default value"
            for a in args:
                a.st.check_convertable(func_arg_st, f"func {func_arg_types} arg {func_arg_st.arg_name}({cnt})")
                if a.st.is_equal_type(func_arg_st):
                    match_score += 2
                else:
                    match_score += 1
            cnt += 1
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
                if isinstance(
                        self.func,
                        PFLAttribute):
                    if self.func.st.is_method:
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
        elif self.st.type == PFLExprType.FUNCTION and self.st.raw_func is not None:
            self._update_func_meta(self.st.raw_func)

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
        if self.value.st.type == PFLExprType.DATACLASS_TYPE or self.value.st.type == PFLExprType.DATACLASS_OBJECT:
            if self.value.st.proxy_dcls is not None:
                assert self.value.st.type == PFLExprType.DATACLASS_TYPE # when proxy cls available, it must be dataclass type
                field_types = AnnotatedType.get_dataclass_fields_and_annotated_types_static(self.value.st.proxy_dcls)
                dcls_type = self.value.st.proxy_dcls
            else:
                assert self.value.st.annotype is not None, "dataclass must have annotype"
                assert self.value.st.annotype.is_dataclass_type()
                field_types = self.value.st.annotype.get_dataclass_fields_and_annotated_types(
                )
                dcls_type = self.value.st.annotype.origin_type

            if self.attr in field_types:
                field_annotype, field = field_types[self.attr]
                new_st = PFLExprInfo.from_annotype(field_annotype,
                                                   is_type=False)
                if self.value.st.type == PFLExprType.DATACLASS_TYPE:
                    # access constant
                    default = field.default
                    assert default is not dataclasses.MISSING, f"access field {self.attr} by type must have default value, we treat it as constant"
                    self.is_const = True
                    new_st._constexpr_data = default
                self.st = new_st
            else:
                # check attr is ClassVar (namespace alias)
                assert not self.value.st.is_temp, f"function in temp dataclass {self.value.st} (not stdlib) can't be used"
                item = get_parse_cache_checked().cached_get_std_item(dcls_type)
                if self.attr in item.namespace_aliases:
                    new_st = PFLExprInfo.from_annotype(parse_type_may_optional_undefined(item.namespace_aliases[self.attr]),
                                                   is_type=True)
                    self.st = new_st
                else:
                    unbound_func = getattr(dcls_type, self.attr)
                    is_prop = False
                    if isinstance(unbound_func, property):
                        assert unbound_func.fget is not None 
                        unbound_func = unbound_func.fget
                        is_prop = True
                    if self.value.st.type == PFLExprType.DATACLASS_OBJECT:
                        ignore_self = True
                        self_type = self.value.st.annotype
                    else:
                        ignore_self = False
                        self_type = None
                        assert inspecttools.isstaticmethod(
                            dcls_type, self.attr
                        ), f"{self.attr} of {dcls_type} must be staticmethod"
                    new_st = get_parse_cache_checked().cached_parse_func(
                        unbound_func, ignore_self=ignore_self, self_type=self_type)
                    new_st.is_property = is_prop
                    if is_prop:
                        assert new_st.return_type is not None, f"property {self.attr} of {dcls_type} must have return type"
                        self.st = new_st.return_type
                        self.compile_info.property_st = new_st
                    else:
                        self.st = new_st
                    self._update_func_meta(unbound_func)
        else:
            if self.value.st.type == PFLExprType.OBJECT or self.value.st.type == PFLExprType.ARRAY:
                annotype = self.value.st.childs[0].annotype
                assert annotype is not None
                methods = _dftype_with_gen_to_supported_methods(
                    annotype.origin_type)[self.value.st.type]
            elif self.value.st.type == PFLExprType.STRING:
                methods = _PFLTYPE_TO_SUPPORTED_METHODS[self.value.st.type]
            else:
                raise ValueError(
                    f"attr `{self.attr}` is not supported (defined) in type {self.value.st}")
            assert self.attr in methods, f"not supported attr {self.attr} for type {self.value.st}"
            new_st = PFLExprInfo.from_signature(methods[self.attr])
            self.st = new_st

    def consteval(self):
        if self.st.type == PFLExprType.FUNCTION and not self.st.is_property:
            return False 
        else:
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
        if self.st.is_property:
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
        annotype = parse_type_may_optional_undefined(type(self.value))

        if isinstance(self.value, bool):
            self.st = PFLExprInfo(PFLExprType.BOOL)
        elif isinstance(self.value, int):
            self.st = PFLExprInfo(PFLExprType.NUMBER)
        elif isinstance(self.value, float):
            self.st = PFLExprInfo(PFLExprType.NUMBER)
        elif isinstance(self.value, str):
            self.st = PFLExprInfo(PFLExprType.STRING)
        elif self.value is None:
            self.st = PFLExprInfo(PFLExprType.NONE_TYPE)
        elif self.value is ...:
            self.st = PFLExprInfo(PFLExprType.ELLIPSIS)
        elif isinstance(self.value, Undefined):
            self.st = PFLExprInfo(PFLExprType.UNDEFINED_TYPE)
        else:
            self.st = PFLExprInfo.from_annotype(annotype)
        self.st.annotype = annotype
        self.is_const = True
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
        assert not self.value.st.is_optional()
        if self.value.st.type == PFLExprType.ARRAY:
            assert not isinstance(self.slice, Sequence)
            assert self.slice.st.type == PFLExprType.NUMBER, f"slice must be number, but got {self.slice.st.type}"
            self.st = self.value.st.childs[0]
        elif self.value.st.type == PFLExprType.OBJECT:
            assert not isinstance(self.slice, Sequence)
            assert self.slice.st.type == PFLExprType.STRING, f"slice must be string, but got {self.slice.st.type}"
            self.st = self.value.st.childs[0]
        elif self.value.st.type == PFLExprType.TUPLE:
            assert not isinstance(self.slice, Sequence)
            if not self.slice.is_const:
                assert self.value.st.is_all_child_same(), F"only support subscript tuple when all tuple element has same type, {self.value.st}"
                self.st = self.value.st.childs[0]
            else:
                assert self.slice.st.type == PFLExprType.NUMBER, f"slice must be number, but got {self.slice.st.type}"
                success = self.slice.consteval()
                assert success, f"slice {self.slice} must be consteval"
                self.st = self.value.st.childs[self.slice.st.metadata_checked]
        elif self.value.st.type == PFLExprType.DATACLASS_OBJECT:
            resolved_custom_expr = self.value
            assert not resolved_custom_expr.st.is_temp, f"temp dataclass {resolved_custom_expr.st} (not stdlib) can't be used in custom operator"
            dcls_type = resolved_custom_expr.st.get_origin_type_checked()
            # use custom operator in left st if found
            if self.is_store == True:
                op_name = "__setitem__"
            else:
                op_name = "__getitem__"
            op_func = inspect.getattr_static(dcls_type, op_name, None)
            assert op_func is not None, f"can't find {op_name} in custom type {get_qualname_of_type(dcls_type)}"
            op_func_st = get_parse_cache_checked().cached_parse_func(
                op_func, ignore_self=True, self_type=self.value.st.annotype)
            assert len(
                op_func_st.childs
            ) == 1, f"custom operator {op_name} must have one arg, but got {len(op_func_st.childs)}"
            if not isinstance(self.slice, Sequence):
                self.slice.st.check_convertable(
                    op_func_st.childs[0],
                    f"custom operator {op_name}|{op_func_st} arg")
            assert op_func_st.return_type is not None, f"custom operator {op_name}|{op_func_st} must have return type"
            self.st = op_func_st.return_type
            self._update_func_meta(op_func)
        else:
            raise ValueError(f"not support subscript for {self.value.st}")
        if isinstance(self.slice, PFLExpr):
            self.is_const = PFLExpr.all_constexpr(self.value, self.slice)
            if self.is_const == True:
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
            self.is_const = True
            self.st._constexpr_data = []
            return self
        # all elts must be same type
        first_elt = self.elts[0]
        for elt in self.elts:
            assert first_elt.st.is_equal_type(elt.st), f"all elts must be same type, but got {first_elt.st} and {elt.st}"
        self.st = PFLExprInfo(PFLExprType.ARRAY,
                              [dataclasses.replace(first_elt.st)])
        self.is_const = PFLExpr.all_constexpr(*self.elts)
        if self.is_const == True:
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
            self.is_const = True
            self.st._constexpr_data = ()
            return self
        self.st = PFLExprInfo(PFLExprType.TUPLE,
                              [dataclasses.replace(e.st) for e in self.elts])
        self.is_const = PFLExpr.all_constexpr(*self.elts)
        if self.is_const == True:
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
            self.is_const = True
            if self.is_const == True:
                self.st._constexpr_data = {}

            return self
        value_st: Optional[PFLExprInfo] = None
        for key, value in zip(self.keys, self.values):
            if key is not None:
                value_st = value.st
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
                assert value_st.is_equal_type(value.st), f"all values must be same type, but got {value_st} and {value.st}"
            else:
                assert value.st.type == PFLExprType.OBJECT
                assert value_st.is_equal_type(value.st.childs[
                    0]), f"all values must be same type, but got {value_st} and {value.st.childs[0]}"
        self.st = PFLExprInfo(PFLExprType.OBJECT,
                              [dataclasses.replace(value_st)])
        
        self.is_const = PFLExpr.all_constexpr(*self.keys, *self.values)
        if self.is_const == True:
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

    def get_module_import_path(self):
        assert self.uid != ""
        return self.uid.split("::")[0]

@dataclasses.dataclass
class _ModCompileInfo:
    code: str = ""

@dataclasses.dataclass(kw_only=True)
class PFLModule(PFLAstNodeBase):
    uid: str
    body: list[PFLAstStmt] = dataclasses.field(default_factory=list)
    compile_info: _ModCompileInfo = dataclasses.field(default_factory=_ModCompileInfo)

    def get_all_compiled(self):
        res: dict[str, PFLFunc] = {}
        for stmt in self.body:
            if isinstance(stmt, PFLFunc):
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
        return f"({unparse_pfl_expr(expr.left)} {op} {unparse_pfl_expr(expr.right)})"
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
    lines = unparse_pfl_ast_to_lines(node)
    return "\n".join(lines)

class PFLAstParseError(Exception):

    def __init__(self, msg: str, node: ast.AST):
        super().__init__(msg)
        self.node = node

class PFLEvalError(Exception):

    def __init__(self, msg: str, node: PFLAstNodeBase):
        super().__init__(msg)
        self.node = node


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
