import ast
from collections.abc import Sequence
import enum
import inspect
from typing import Any, Callable, Optional, Union

import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core import inspecttools
from tensorpc.core.annolib import (Undefined, is_undefined,
                                   parse_type_may_optional_undefined,
                                   undefined)
from tensorpc.core.pfl.constants import PFL_FUNC_META_ATTR
from tensorpc.core.moduleid import get_qualname_of_type

from .core import (PFLExprInfo, PFLExprType, PFLFuncMeta, PFLMetaInferResult,
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
    BLOCK = 0
    EXPR = 1
    STMT_MASK = 0x100
    ASSIGN = 0x101
    IF = 0x102
    EXPR_STMT = 0x103
    AUG_ASSIGN = 0x104
    FOR = 0x105
    WHILE = 0x106
    ANN_ASSIGN = 0x107

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

    def __repr__(self):
        if self in _PFLAST_TYPE_TO_STR:
            return _PFLAST_TYPE_TO_STR[self]
        return super().__repr__()


_PFLAST_TYPE_TO_STR = {
    PFLASTType.BLOCK: "block",
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


_AST_COMPARE_TO_PFL_COMPARE = {
    ast.Eq: CompareType.EQUAL,
    ast.NotEq: CompareType.NOT_EQUAL,
    ast.Lt: CompareType.LESS,
    ast.LtE: CompareType.LESS_EQUAL,
    ast.Gt: CompareType.GREATER,
    ast.GtE: CompareType.GREATER_EQUAL,
    ast.Is: CompareType.IS,
    ast.IsNot: CompareType.IS_NOT,
    ast.In: CompareType.IN,
    ast.NotIn: CompareType.NOT_IN,
}

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
}


@dataclasses.dataclass
class PFLAstNodeBase:
    type: PFLASTType
    # record lineno/col_offset from ast node for debug
    lineno: int 
    col_offset: int


@dataclasses.dataclass
class PFLAstStmt(PFLAstNodeBase):
    pass


@dataclasses.dataclass
class PFLFunc(PFLAstNodeBase):
    args: list[PFLStaticVar]
    body: list[PFLAstStmt] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(kw_only=True)
class PFLExpr(PFLAstNodeBase):
    st: PFLExprInfo = dataclasses.field(
        default_factory=lambda: PFLExprInfo(PFLExprType.UNKNOWN))
    is_const: Union[bool, Undefined] = undefined

    @staticmethod
    def all_constexpr(*args: Optional["PFLExpr"]):
        for arg in args:
            if arg is not None:
                if not arg.is_const:
                    return False
        return True

    def consteval(self) -> bool:
        """run const evaluation. result is stored in `self.st.metadata`
        user should return True when a consteval is succeed.
        """
        return False

    def metaeval(self) -> bool:
        """run meta-data const evaluation. used when static type define "meta_infer" function.
        """
        return self.consteval()

    def _update_func_meta(self, fn: Callable):
        pfl_meta = getattr(fn, PFL_FUNC_META_ATTR, None)
        if pfl_meta is not None:
            assert isinstance(pfl_meta, PFLFuncMeta)
            self.st._meta_infer = pfl_meta.meta_infer

    def _get_consteval_operands_if_all(
            self, *exprs: "PFLExpr") -> Optional[list[Any]]:
        res: list[Any] = []
        for expr in exprs:
            if not isinstance(expr.st.metadata, Undefined):
                assert not isinstance(expr.st.metadata, PFLExprInfo)
                res.append(expr.st.metadata)
            else:
                return None
        return res

    def _get_consteval_operands_st_if_all(
            self, *exprs: "PFLExpr") -> Optional[list[PFLExprInfo]]:
        res: list[PFLExprInfo] = []
        for expr in exprs:
            if not isinstance(expr.st.metadata, Undefined):
                assert not isinstance(expr.st.metadata, PFLExprInfo)
                res.append(expr.st)
            else:
                return None
        return res

    def _get_consteval_operands_st_if_any(
            self, *exprs: "PFLExpr") -> Optional[list[PFLExprInfo]]:
        res: list[PFLExprInfo] = []
        found = False
        for expr in exprs:
            if not isinstance(expr.st.metadata, Undefined):
                assert not isinstance(expr.st.metadata, PFLExprInfo)
                res.append(expr.st)
                found = True
            else:
                res.append(expr.st)
        if not found:
            return None
        return res


@dataclasses.dataclass
class PFLAssign(PFLAstStmt):
    target: PFLExpr
    value: PFLExpr

    def __post_init__(self):
        assert self.value.st.is_convertable(
            self.target.st
        ), f"{self.value.st} not convertable to {self.target.st}"


@dataclasses.dataclass
class PFLAugAssign(PFLAstStmt):
    target: PFLExpr
    op: BinOpType
    value: PFLExpr

    def __post_init__(self):
        assert self.target.st.support_aug_assign()
        assert self.value.st.is_convertable(
            self.target.st
        ), f"{self.value.st} not convertable to {self.target.st}"


@dataclasses.dataclass
class PFLAnnAssign(PFLAstStmt):
    target: PFLExpr
    annotation: str
    value: Optional[PFLExpr]

    def __post_init__(self):
        assert isinstance(self.target, PFLName)
        if self.value is not None:
            assert self.value.st.is_convertable(
                self.target.st
            ), f"{self.value.st} not convertable to {self.target.st}"


@dataclasses.dataclass
class PFLFor(PFLAstStmt):
    target: PFLExpr
    iter: PFLExpr
    body: list[PFLAstStmt]

    def __post_init__(self):
        if self.iter.st.type == PFLExprType.ARRAY:
            self.target.st = self.iter.st.childs[0]
        elif self.iter.st.type == PFLExprType.RANGE:
            self.target.st = PFLExprInfo(PFLExprType.NUMBER)
        else:
            raise NotImplementedError(
                "for loop iter type must be array or range object")
        return self


@dataclasses.dataclass
class PFLWhile(PFLAstStmt):
    test: PFLExpr
    body: list[PFLAstStmt]

    def __post_init__(self):
        test_dtype = self.test.st
        assert test_dtype.can_cast_to_bool(
        ), f"test must be convertable to bool, but got {test_dtype}"
        if not isinstance(self.test, PFLExpr):
            raise ValueError("test must be a PFLExpr")
        return self


@dataclasses.dataclass
class PFLIf(PFLAstStmt):
    test: PFLExpr
    body: list[PFLAstStmt]
    orelse: list[PFLAstStmt] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        test_dtype = self.test.st
        assert test_dtype.can_cast_to_bool(
        ), f"test must be convertable to bool, but got {test_dtype}"
        if not isinstance(self.test, PFLExpr):
            raise ValueError("test must be a PFLExpr")
        return self


@dataclasses.dataclass
class PFLExprStmt(PFLAstStmt):
    value: PFLExpr


@dataclasses.dataclass(kw_only=True)
class PFLBoolOp(PFLExpr):
    op: BoolOpType
    left: PFLExpr
    right: PFLExpr

    def __post_init__(self):
        assert self.left.st.support_bool_op()
        assert self.right.st.support_bool_op()
        self.st = PFLExprInfo(PFLExprType.BOOL)
        self.is_const = PFLExpr.all_constexpr(self.left, self.right)
        return self

    def consteval(self):
        operands = self._get_consteval_operands_if_all(self.left, self.right)
        if operands is not None:
            if self.op == BoolOpType.AND:
                self.st.metadata = operands[0] and operands[1]
            else:
                self.st.metadata = operands[0] or operands[1]
            return True
        return False


@dataclasses.dataclass(kw_only=True)
class PFLUnaryOp(PFLExpr):
    op: UnaryOpType
    operand: PFLExpr

    def __post_init__(self):
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
            op_func_st = get_parse_context_checked().cached_parse_func(
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
        if op_func is not None:
            self._update_func_meta(op_func)
        return self

    def consteval(self):
        operands = self._get_consteval_operands_if_all(self.operand)
        if operands is not None:
            if self.op == UnaryOpType.INVERT:
                self.st.metadata = ~operands[0]
            elif self.op == UnaryOpType.NOT:
                self.st.metadata = not operands[0]
            elif self.op == UnaryOpType.UADD:
                self.st.metadata = +operands[0]
            else:
                self.st.metadata = -operands[0]
            return True
        return False

    def metaeval(self):
        if self.st.meta_infer is not None:
            operands = self._get_consteval_operands_st_if_any(self.operand)
            if operands is not None:
                infer_res = self.st.meta_infer(*operands)
                if infer_res is not None:
                    assert isinstance(infer_res, PFLMetaInferResult), "meta infer function must return `pfl.PFLMetaInferResult`"
                    self.st.metadata = infer_res.data

                return True
        return self.consteval()


@dataclasses.dataclass(kw_only=True)
class PFLIfExp(PFLExpr):
    test: PFLExpr
    body: PFLExpr
    orelse: PFLExpr

    def __post_init__(self):
        assert self.test.st.can_cast_to_bool(
        ), f"test must be convertable to bool, but got {self.test.st}"
        assert self.body.st == self.orelse.st, f"body and orelse must be same type, but got {self.body.st} and {self.orelse.st}"
        self.st = dataclasses.replace(self.body.st)
        return self

    def consteval(self):
        operands = self._get_consteval_operands_if_all(self.test, self.body,
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
        # overrideable operators
        left = self.left
        right = self.right
        is_custom_type: bool = False
        custom_res_type = None
        resolved_custom_expr = None
        if left.st.type == PFLExprType.DATACLASS_OBJECT:
            resolved_custom_expr = left
        elif right.st.type == PFLExprType.DATACLASS_OBJECT:
            resolved_custom_expr = right
            self.is_right = True
        op_func = None
        if resolved_custom_expr is not None:
            assert not resolved_custom_expr.st.is_temp, f"temp dataclass {resolved_custom_expr.st} (not stdlib) can't be used in custom operator"
            dcls_type = resolved_custom_expr.st.get_origin_type_checked()
            # use custom operator in left st if found
            op_name = _PFL_BINARY_TYPE_TO_METHOD_NAME[op]
            op_func = inspect.getattr_static(dcls_type, op_name, None)
            assert op_func is not None, f"can't find {op_name} in custom type {get_qualname_of_type(dcls_type)}"
            op_func_st = get_parse_context_checked().cached_parse_func(
                op_func, True, self_type=resolved_custom_expr.st.annotype)
            assert len(
                op_func_st.childs
            ) == 1, f"custom operator {op_name} must have one arg, but got {len(op_func_st.childs)}"
            if left.st.type == PFLExprType.DATACLASS_OBJECT:
                right.st.check_convertable(op_func_st.childs[0],
                                           f"custom operator {op_name} arg")
            elif right.st.type == PFLExprType.DATACLASS_OBJECT:
                left.st.check_convertable(op_func_st.childs[0],
                                          f"custom operator {op_name} arg")
            is_custom_type = True
            custom_res_type = op_func_st.return_type
            assert custom_res_type is not None, f"custom operator {op_name} must have return type"
        return is_custom_type, custom_res_type, op_func

    def metaeval(self):
        operands = self._get_consteval_operands_st_if_any(
            self.left, self.right)
        if operands is not None:
            if self.st.meta_infer is not None:
                infer_res = self.st.meta_infer(*operands)
                if infer_res is not None:
                    assert isinstance(infer_res, PFLMetaInferResult), "meta infer function must return `pfl.PFLMetaInferResult`"
                    self.st.metadata = infer_res.data
                    return True
        return self.consteval()


@dataclasses.dataclass(kw_only=True)
class PFLBinOp(PFLBinOpBase):
    op: BinOpType

    def __post_init__(self):
        is_custom_type, custom_res_type, op_func = self.resolve_custom_type(
            self.op)
        if not is_custom_type:
            promotion_type = self.left.st.check_support_binary_op_and_promotion(self.right.st)
            self.st = PFLExprInfo(PFLExprType.NUMBER, annotype=promotion_type)
        else:
            assert custom_res_type is not None
            self.st = custom_res_type
        self.is_const = PFLExpr.all_constexpr(self.left, self.right)
        if op_func is not None:
            self._update_func_meta(op_func)
        return self

    def consteval(self):
        operands = self._get_consteval_operands_if_all(self.left, self.right)
        if operands is not None:
            if self.op == BinOpType.ADD:
                self.st.metadata = operands[0] + operands[1]
            elif self.op == BinOpType.SUB:
                self.st.metadata = operands[0] - operands[1]
            elif self.op == BinOpType.MULT:
                self.st.metadata = operands[0] * operands[1]
            elif self.op == BinOpType.DIV:
                self.st.metadata = operands[0] / operands[1]
            elif self.op == BinOpType.FLOOR_DIV:
                self.st.metadata = operands[0] // operands[1]
            elif self.op == BinOpType.MOD:
                self.st.metadata = operands[0] % operands[1]
            elif self.op == BinOpType.POW:
                self.st.metadata = operands[0]**operands[1]
            elif self.op == BinOpType.LSHIFT:
                self.st.metadata = operands[0] << operands[1]
            elif self.op == BinOpType.RSHIFT:
                self.st.metadata = operands[0] >> operands[1]
            elif self.op == BinOpType.BIT_OR:
                self.st.metadata = operands[0] | operands[1]
            elif self.op == BinOpType.BIT_XOR:
                self.st.metadata = operands[0] ^ operands[1]
            elif self.op == BinOpType.BIT_AND:
                self.st.metadata = operands[0] & operands[1]
            else:
                raise NotImplementedError
            return True
        return False


@dataclasses.dataclass(kw_only=True)
class PFLCompare(PFLBinOpBase):
    op: CompareType

    def __post_init__(self):
        is_custom_type = False
        if self.op == CompareType.IN or self.op == CompareType.NOT_IN:
            # contains operator don't support custom datatype
            assert self.left.st.type != PFLExprType.DATACLASS_OBJECT
            assert self.right.st.type != PFLExprType.DATACLASS_OBJECT
        op_func = None
        if not (self.op == CompareType.IS or self.op == CompareType.IS_NOT):
            # overrideable operators
            is_custom_type, _, op_func = self.resolve_custom_type(self.op)

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
        self.is_const = PFLExpr.all_constexpr(self.left, self.right)
        if op_func is not None:
            self._update_func_meta(op_func)
        return self

    def consteval(self):
        operands = self._get_consteval_operands_if_all(self.left, self.right)
        if operands is not None:
            if self.op == CompareType.EQUAL:
                self.st.metadata = operands[0] == operands[1]
            elif self.op == CompareType.NOT_EQUAL:
                self.st.metadata = operands[0] != operands[1]
            elif self.op == CompareType.LESS:
                self.st.metadata = operands[0] < operands[1]
            elif self.op == CompareType.LESS_EQUAL:
                self.st.metadata = operands[0] <= operands[1]
            elif self.op == CompareType.GREATER:
                self.st.metadata = operands[0] > operands[1]
            elif self.op == CompareType.GREATER_EQUAL:
                self.st.metadata = operands[0] >= operands[1]
            elif self.op == CompareType.IS:
                self.st.metadata = operands[0] is operands[1]
            elif self.op == CompareType.IS_NOT:
                self.st.metadata = operands[0] is not operands[1]
            elif self.op == CompareType.IN:
                self.st.metadata = operands[0] in operands[1]
            elif self.op == CompareType.NOT_IN:
                self.st.metadata = operands[0] not in operands[1]
            else:
                raise NotImplementedError
            return True
        return False


@dataclasses.dataclass(kw_only=True)
class PFLCall(PFLExpr):
    # don't support kwargs due to js
    func: PFLExpr
    args: list[PFLExpr]
    kws: Union[list[tuple[str, PFLExpr]], Undefined] = undefined
    is_ctor: Union[Undefined, bool] = undefined

    def __post_init__(self):
        # validate args
        is_ctor: bool = False
        if self.func.st.type == PFLExprType.DATACLASS_TYPE:
            # create std objects
            is_ctor = True
        self.is_ctor = is_ctor
        assert self.func.st.type == PFLExprType.FUNCTION or self.func.st.type == PFLExprType.DATACLASS_TYPE, f"func must be function/dcls, but got {self.func.st.type}"
        last_is_vaarg = False
        if self.func.st.type == PFLExprType.FUNCTION:
            func_arg_types = self.func.st.childs
            if func_arg_types:
                last_is_vaarg = func_arg_types[-1].is_vaargs
        else:
            annotype = self.func.st.annotype
            assert annotype is not None
            field_types = list(
                annotype.get_dataclass_field_annotated_types().values())
            func_arg_types = [
                PFLExprInfo.from_annotype(f, is_type=False)
                for f in field_types
            ]
        if last_is_vaarg:
            assert isinstance(self.kws, Undefined), "don't support use kwargs with *args"
        if not last_is_vaarg:
            assert len(self.args) <= len(
                func_arg_types
            ), f"func {self.func} expect {len(func_arg_types)} args, but got {len(self.args)}"
        for i, a in enumerate(self.args):
            if not last_is_vaarg:
                func_arg = func_arg_types[i]
                a.st.check_convertable(func_arg, f"func {self.func} arg {i}")
            else:
                if i < len(func_arg_types) - 1:
                    func_arg = func_arg_types[i]
                    a.st.check_convertable(func_arg,
                                           f"func {self.func} arg {i}")
                else:
                    a.st.check_convertable(func_arg_types[-1],
                                           f"func {self.func} arg {i}")
        if not isinstance(self.kws, Undefined):
            for name, a in self.kws:
                found = False
                for arg in func_arg_types:
                    assert arg.arg_name is not None
                    if name == arg.arg_name:
                        found = True 
                        a.st.check_convertable(arg,
                                           f"func {self.func} kwarg {name}")
                        break
                if not found:
                    raise ValueError(f"can't find arg {name} in function {self.func.st}")
        if self.func.st.type != PFLExprType.FUNCTION:
            self.st = dataclasses.replace(self.func.st,
                                          type=PFLExprType.DATACLASS_OBJECT,
                                          childs=[],
                                          return_type=None)
        else:
            assert self.func.st.return_type is not None, f"func {self.func} must have return type"
            self.st = dataclasses.replace(self.func.st.return_type)

    def consteval(self):
        operands = self._get_consteval_operands_if_all(*self.args)
        if operands is not None:
            if isinstance(self.func, PFLName):
                fn = self.func.st.metadata
                if not isinstance(fn, Undefined):
                    self.st.metadata = fn(*operands)
                    return True
                return False
            elif isinstance(self.func, PFLAttribute):
                obj = self.func.value.st.metadata
                if not isinstance(obj, Undefined):
                    self.st.metadata = getattr(obj, self.func.attr)(*operands)
                    return True
        return False

    def metaeval(self):
        operands = self._get_consteval_operands_st_if_any(*self.args)
        if operands is not None:
            if isinstance(
                    self.func,
                    PFLAttribute) and self.func.st.meta_infer is not None:
                if self.func.st.is_method:
                    obj = self.func.value.st.metadata
                    if not isinstance(obj, Undefined):
                        infer_res = self.func.st.meta_infer(
                            self.func.value.st, *operands)
                        if infer_res is not None:
                            assert isinstance(infer_res, PFLMetaInferResult), "meta infer function must return `pfl.PFLMetaInferResult`"
                            self.st.metadata = infer_res.data
                            return True
                else:
                    infer_res = self.func.st.meta_infer(*operands)
                    if infer_res is not None:
                        assert isinstance(infer_res, PFLMetaInferResult), "meta infer function must return `pfl.PFLMetaInferResult`"
                        self.st.metadata = infer_res.data
                        return True

        return self.consteval()


@dataclasses.dataclass(kw_only=True)
class PFLName(PFLExpr):
    id: str
    is_store: Union[Undefined, bool] = undefined
    is_new: Union[Undefined, bool] = undefined

    def __post_init__(self):
        if self.st.type == PFLExprType.DATACLASS_TYPE or self.st.type == PFLExprType.DATACLASS_OBJECT:
            assert self.st.annotype is not None, "dataclass must have annotype"


@dataclasses.dataclass(kw_only=True)
class PFLAttribute(PFLExpr):
    value: PFLExpr
    attr: str
    is_store: Union[Undefined, bool] = undefined

    def __post_init__(self):
        if self.value.st.type == PFLExprType.DATACLASS_TYPE or self.value.st.type == PFLExprType.DATACLASS_OBJECT:
            assert self.st.annotype is not None, "dataclass must have annotype"
            assert self.st.annotype.is_dataclass_type()
            field_types = self.st.annotype.get_dataclass_fields_and_annotated_types(
            )
            if self.attr in field_types:
                field_annotype, field = field_types[self.attr]
                if self.value.st.type == PFLExprType.DATACLASS_TYPE:
                    # access constant
                    default = field.default
                    assert default is not dataclasses.MISSING, f"access field {self.attr} by type must have default value"
                new_st = PFLExprInfo.from_annotype(field_annotype,
                                                   is_type=False)
                self.st = new_st
            else:
                # TODO add support for property
                assert not self.value.st.is_temp, f"function in temp dataclass {self.value.st} (not stdlib) can't be used"
                dcls_type = self.st.annotype.origin_type
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
                new_st = get_parse_context_checked().cached_parse_func(
                    unbound_func, ignore_self=ignore_self, self_type=self_type)
                new_st.is_property = is_prop
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
                    f"not support {self.value.st.type} for {self.attr}")
            assert self.attr in methods, f"not support {self.attr} for {self.value.st.type}"
            new_st = PFLExprInfo.from_signature(methods[self.attr])
            self.st = new_st

    def consteval(self):
        if self.st.type == PFLExprType.FUNCTION and not self.st.is_property:
            return False 
        else:
            operands = self._get_consteval_operands_if_all(self.value)
            if operands is not None:
                self.st.metadata = getattr(operands[0], self.attr)
                return True
            return False

    def metaeval(self):
        if self.st.type == PFLExprType.FUNCTION and self.st.is_property:
            operands = self._get_consteval_operands_st_if_any(self.value)
            if operands is not None:
                if self.st.meta_infer is not None:
                    infer_res = self.st.meta_infer(*operands)
                    if infer_res is not None:
                        assert isinstance(infer_res, PFLMetaInferResult), "meta infer function must return `pfl.PFLMetaInferResult`"
                        self.st.metadata = infer_res.data
                        return True
                else:
                    # for property without a meta_infer fn, we need to stop eval because
                    # metadata may not contains the attr.
                    return False
        return self.consteval()
        

@dataclasses.dataclass(kw_only=True)
class PFLConstant(PFLExpr):
    value: Any

    def __post_init__(self):
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
        elif isinstance(self.value, Undefined):
            self.st = PFLExprInfo(PFLExprType.UNDEFINED_TYPE)
        else:
            annotype = parse_type_may_optional_undefined(type(self.value))
            self.st = PFLExprInfo.from_annotype(annotype)
        self.is_const = True
        return self

    def consteval(self):
        self.st.metadata = self.value
        return True


@dataclasses.dataclass
class PFLSlice(PFLExpr):
    lo: Union[Undefined, PFLExpr] = undefined 
    hi: Union[Undefined, PFLExpr] = undefined 
    step: Union[Undefined, PFLExpr] = undefined 
    def __post_init__(self):
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

    def __post_init__(self):
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
            assert self.value.st.is_all_child_same(), "only support subscript tuple when all tuple element has same type"
            self.st = self.value.st.childs[0]
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
            op_func_st = get_parse_context_checked().cached_parse_func(
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
        return self

    def consteval(self):
        if isinstance(self.slice, PFLExpr):
            operands = self._get_consteval_operands_if_all(self.value, self.slice)
            if operands is not None:
                self.st.metadata = operands[0][operands[1]]
                return True
            return False
        else:
            operands = self._get_consteval_operands_if_all(self.value, *(self.slice))
            if operands is not None:
                self.st.metadata = operands[0][tuple(operands[1:])]
                return True
            return False

    def metaeval(self):
        if isinstance(self.slice, PFLExpr):

            operands = self._get_consteval_operands_st_if_all(
                self.value, self.slice)
            if operands is not None and self.st.meta_infer is not None:
                infer_res = self.st.meta_infer(operands[0], operands[1])
                if infer_res is not None:
                    assert isinstance(infer_res, PFLMetaInferResult), "meta infer function must return `pfl.PFLMetaInferResult`"
                    self.st.metadata = infer_res.data
                    return True
            return self.consteval()
        else:
            operands = self._get_consteval_operands_st_if_all(
                self.value, *self.slice)
            if operands is not None and self.st.meta_infer is not None:
                infer_res = self.st.meta_infer(operands[0], tuple(operands[1:]))
                if infer_res is not None:
                    assert isinstance(infer_res, PFLMetaInferResult), "meta infer function must return `pfl.PFLMetaInferResult`"
                    self.st.metadata = infer_res.data
                    return True

            return self.consteval()


@dataclasses.dataclass(kw_only=True)
class PFLArray(PFLExpr):
    elts: list[PFLExpr]

    def __post_init__(self):
        if not self.elts:
            self.st = PFLExprInfo(PFLExprType.ARRAY,
                                  [PFLExprInfo(PFLExprType.UNKNOWN)])
            self.is_const = True
            return self
        # all elts must be same type
        first_elt = self.elts[0]
        for elt in self.elts:
            assert first_elt.st == elt.st, f"all elts must be same type, but got {first_elt.st} and {elt.st}"
        self.st = PFLExprInfo(PFLExprType.ARRAY,
                              [dataclasses.replace(first_elt.st)])
        self.is_const = PFLExpr.all_constexpr(*self.elts)

    def consteval(self):
        operands = self._get_consteval_operands_if_all(*self.elts)
        print("???", operands, [m.st.metadata for m in self.elts])
        if operands is not None:
            self.st.metadata = operands
            return True
        return False


@dataclasses.dataclass(kw_only=True)
class PFLDict(PFLExpr):
    keys: list[Optional[PFLExpr]]
    values: list[PFLExpr]

    def __post_init__(self):
        if not self.keys:
            self.st = PFLExprInfo(PFLExprType.OBJECT,
                                  [PFLExprInfo(PFLExprType.UNKNOWN)])
            self.is_const = True
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
                assert value_st == value.st, f"all values must be same type, but got {value_st} and {value.st}"
            else:
                assert value.st.type == PFLExprType.OBJECT
                assert value_st == value.st.childs[
                    0], f"all values must be same type, but got {value_st} and {value.st.childs[0]}"
        self.st = PFLExprInfo(PFLExprType.OBJECT,
                              [dataclasses.replace(value_st)])
        self.is_const = PFLExpr.all_constexpr(*self.keys, *self.values)

    def consteval(self):
        res = {}
        for key, value in zip(self.keys, self.values):
            if key is None:
                if not isinstance(value.st.metadata, Undefined):
                    res.update(value.st.metadata)
                else:
                    return False
            else:
                kv = self._get_consteval_operands_if_all(key, value)
                if kv is not None:
                    res[kv[0]] = kv[1]
                else:
                    return False
        self.st.metadata = res
        return True
