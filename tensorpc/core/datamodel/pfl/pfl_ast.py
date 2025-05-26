"""Python Frontend Language (Domain Specific Language of python)

We extend python ast node (expr) to a simple DSL with if/assign/math support.

usually used to write frontend code in a more readable way.

WARNING: PFL is static typed to get better performance, union isn't supported except None (optional).

"""
import contextlib
from typing import Any, Callable, Optional, Type, Union

from tensorpc.core import inspecttools
from tensorpc.core.annolib import AnnotatedType, T_dataclass, Undefined, child_dataclass_type_generator, child_type_generator_with_dataclass, parse_annotated_function, parse_type_may_optional_undefined, undefined
import tensorpc.core.dataclass_dispatch as dataclasses
from typing_extensions import Self
import enum 
import ast 
import inspect 
import contextvars

from tensorpc.core.datamodel.pfl.pfl_std import Math, MathUtil
from tensorpc.core.funcid import clean_source_code, remove_common_indent_from_code
from tensorpc.core.moduleid import get_qualname_of_type
from tensorpc.core.tree_id import UniqueTreeId
from .pfl_reg import STD_REGISTRY, StdRegistryItem

class PFLStaticTypeType(enum.IntEnum):
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

_BASE_TYPE_TO_STRING = {
    PFLStaticTypeType.UNKNOWN: "unknown",
    PFLStaticTypeType.NUMBER: "number",
    PFLStaticTypeType.BOOL: "bool",
    PFLStaticTypeType.STRING: "string",
    PFLStaticTypeType.NONE_TYPE: "null",
    PFLStaticTypeType.UNDEFINED_TYPE: "undefined",
    PFLStaticTypeType.ANY: "any",
}

_TYPE_CAN_CAST_TO_BOOL = {
    PFLStaticTypeType.NUMBER,
    PFLStaticTypeType.BOOL,
    PFLStaticTypeType.STRING,
    PFLStaticTypeType.ARRAY,
    PFLStaticTypeType.OBJECT,
    PFLStaticTypeType.DATACLASS_OBJECT,
}

_TYPE_SUPPORT_BINARY_OP = {
    PFLStaticTypeType.NUMBER,
    PFLStaticTypeType.BOOL,
}

_TYPE_SUPPORT_UNARY_OP = {
    PFLStaticTypeType.NUMBER,
    PFLStaticTypeType.BOOL,
}

_TYPE_SUPPORT_COMPARE_OP = {
    PFLStaticTypeType.NUMBER,
    PFLStaticTypeType.BOOL,
    PFLStaticTypeType.STRING,
    PFLStaticTypeType.ARRAY,
    PFLStaticTypeType.OBJECT,
}

_BASE_ANNO_TYPE_TO_PFLSTATIC_TYPE = {
    int: PFLStaticTypeType.NUMBER,
    float: PFLStaticTypeType.NUMBER,
    bool: PFLStaticTypeType.BOOL,
    str: PFLStaticTypeType.STRING,
    type(None): PFLStaticTypeType.NONE_TYPE,
    Undefined: PFLStaticTypeType.UNDEFINED_TYPE, 
}

class PFLParseContext:
    def __init__(self, lines: list[str], backend: str = "js", temp_std_items: Optional[dict[Type, StdRegistryItem]] = None):
        self.lines = lines

        self.error_node: Optional[ast.AST] = None
        self._backend = backend

        self._std_item_cache: dict[Type, StdRegistryItem] = {}
        if temp_std_items is not None:
            self._std_item_cache.update(temp_std_items)
        self._func_parse_result_cache: dict[Callable, PFLStaticType] = {}
        self._mapped_type_cache: dict[Type, StdRegistryItem] = {}

    def format_error_from_lines_node(self, node: ast.AST):
        if hasattr(node, "lineno") and hasattr(node, "col_offset"):
            lineno = node.lineno # type: ignore
            col_offset = node.col_offset # type: ignore
            if lineno > 0 and lineno <= len(self.lines):
                line = self.lines[lineno - 1]
                return f"{line}\n{' ' * col_offset}^"
        return ""

    def cached_parse_func(self, func: Callable, ignore_self: bool = False) -> "PFLStaticType":
        if func in self._func_parse_result_cache:
            return self._func_parse_result_cache[func]
        sig = inspect.signature(func)
        return PFLStaticType.from_signature(sig, ignore_self=ignore_self)

    def cached_get_std_item(self, dcls: Type[T_dataclass]):
        if dcls in self._std_item_cache:
            return self._std_item_cache[dcls]
        item = STD_REGISTRY.get_item_by_dcls(dcls, self._backend)
        if item is None:
            raise ValueError(f"can't find your type {get_qualname_of_type(dcls)} from std registry.")
        self._std_item_cache[dcls] = item 
        return item 

    def cached_get_dcls_by_mapped_type(self, usercls: Any):
        if usercls in self._mapped_type_cache:
            return self._mapped_type_cache[usercls]
        item = STD_REGISTRY.get_dcls_item_by_mapped_type(usercls, self._backend)
        if item is None:
            raise ValueError(f"can't find your mapped type {get_qualname_of_type(usercls)} from std registry.")
        self._mapped_type_cache[usercls] = item 
        return item 

_PFLPARSE_CONTEXT: contextvars.ContextVar[
    Optional[PFLParseContext]] = contextvars.ContextVar(
        "PFLParseContext", default=None)

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

def param_fn(name: str, anno: Any, default: Any = inspect.Parameter.empty):
    # since js don't support keyword, we don't need to care about kw.
    return inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=anno, default=default)

def varparam_fn(name: str, anno: Any):
    # since js don't support keyword, we don't need to care about kw.
    return inspect.Parameter(name, inspect.Parameter.VAR_POSITIONAL, annotation=anno)

_PFLTYPE_TO_SUPPORTED_METHODS = {
    PFLStaticTypeType.STRING: {
        "startswith": inspect.Signature([param_fn("prefix", str)], return_annotation=bool), # startsWith
        "endswith": inspect.Signature([param_fn("suffix", str)], return_annotation=bool), # endsWith
        "find": inspect.Signature([param_fn("sub", str), param_fn("start", int, 0)], return_annotation=int), # indexOf
        "rfind": inspect.Signature([param_fn("sub", str), param_fn("start", int, 0)], return_annotation=int), # indexOf
        "replace": inspect.Signature([param_fn("old", str), param_fn("new", str)], return_annotation=str), # replace
        "split": inspect.Signature([param_fn("sep", str, None), param_fn("maxsplit", int, -1)], return_annotation=list[str]), # split
        "join": inspect.Signature([param_fn("iterable", list[str])], return_annotation=str), # join
    },
}

def _dftype_with_gen_to_supported_methods(vt: Any):
    return {
        PFLStaticTypeType.ARRAY: {
            "append": inspect.Signature([param_fn("value", vt)], return_annotation=None), # push
            "extend": inspect.Signature([param_fn("iterable", list[vt])], return_annotation=None), # extend
            "insert": inspect.Signature([param_fn("index", int), param_fn("value", vt)], return_annotation=None), # insert
            "remove": inspect.Signature([param_fn("value", vt)], return_annotation=None), # remove
            "pop": inspect.Signature([param_fn("index", int, -1)], return_annotation=vt), # pop
            "clear": inspect.Signature([], return_annotation=None), # clear
        },
        PFLStaticTypeType.OBJECT: {
            "extend": inspect.Signature([param_fn("iterable", dict[str, vt])], return_annotation=None), # extend
            "remove": inspect.Signature([param_fn("key", str)], return_annotation=None), # remove
            "pop": inspect.Signature([param_fn("key", str)], return_annotation=vt), # pop
            # "clear": inspect.Signature([], return_annotation=None), # clear
        }
    } 


@dataclasses.dataclass
class PFLStaticType:
    type: PFLStaticTypeType
    childs: list['PFLStaticType'] = dataclasses.field(default_factory=list)
    has_optional: bool = False
    has_undefined: bool = False
    # for custom dataclass
    mapped: str = ""
    # for container and dataclass
    annotype: Optional[AnnotatedType] = None
    # for function
    return_type: Optional["PFLStaticType"] = None
    default: Union[Undefined, Any] = undefined
    is_vaargs: bool = False
    # for dataclass in function arg
    is_temp: bool = False

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

        return res

    def __repr__(self) -> str:
        if self.type == PFLStaticTypeType.ARRAY:
            child_repr = str(self.childs[0])
            res = f"{child_repr}[]"
        elif self.type == PFLStaticTypeType.OBJECT:
            child_repr = str(self.childs[0])
            res = f"Record<string, {child_repr}>"
        elif self.type == PFLStaticTypeType.DATACLASS_TYPE:
            assert self.annotype is not None 
            res = str(self.annotype.origin_type.__name__)
        elif self.type == PFLStaticTypeType.DATACLASS_OBJECT:
            assert self.annotype is not None 
            res = str(self.annotype.origin_type.__name__) + "()"
        elif self.type == PFLStaticTypeType.FUNCTION:
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
            res += f"({self.mapped})"

        if self.has_optional:
            res += " | null"
        if self.has_undefined:
            res += " | undefined"
        return res

    def get_origin_type_checked(self):
        assert self.annotype is not None 
        return self.annotype.origin_type

    @classmethod 
    def from_annotype(cls, annotype: AnnotatedType, is_type: bool = True) -> Self:
        if annotype.origin_type in _BASE_ANNO_TYPE_TO_PFLSTATIC_TYPE:
            res = cls(_BASE_ANNO_TYPE_TO_PFLSTATIC_TYPE[annotype.origin_type])
        elif annotype.is_number_type():
            res = cls(PFLStaticTypeType.NUMBER)
        elif annotype.is_list_type():
            value_anno_type = annotype.get_list_value_anno_type()
            res = cls(PFLStaticTypeType.ARRAY, [PFLStaticType.from_annotype(value_anno_type, is_type)])
        elif annotype.is_dict_type():
            value_anno_type = annotype.get_dict_value_anno_type()
            res = cls(PFLStaticTypeType.OBJECT, [PFLStaticType.from_annotype(value_anno_type, is_type)])
        elif annotype.is_dataclass_type():
            res = cls(PFLStaticTypeType.DATACLASS_TYPE if is_type else PFLStaticTypeType.DATACLASS_OBJECT)
            item = get_parse_context_checked().cached_get_std_item(annotype.origin_type)
            res.mapped = item.mapped_name
            res.is_temp = item.is_temp
        elif annotype.is_tuple_type():
            # TODO add real tuple support
            first_child_type = annotype.child_types[0]
            assert all(c == first_child_type for c in annotype.child_types), "tuple must be same type"
            res = cls(PFLStaticTypeType.ARRAY, [PFLStaticType.from_annotype(parse_type_may_optional_undefined(first_child_type), is_type)])
        elif annotype.is_any_type():
            res = cls(PFLStaticTypeType.ANY, [])
        else:
            mapped_item = get_parse_context_checked().cached_get_dcls_by_mapped_type(annotype.origin_type)
            if mapped_item is None:
                raise ValueError(f"not support annotype {annotype}")
            res = cls(PFLStaticTypeType.DATACLASS_TYPE if is_type else PFLStaticTypeType.DATACLASS_OBJECT)
            res.mapped = mapped_item.mapped_name
            annotype = parse_type_may_optional_undefined(mapped_item.dcls)

        res.annotype = annotype
        res.has_optional = annotype.is_optional
        res.has_undefined = annotype.is_undefined 
        return res

    @classmethod 
    def from_signature(cls, sig: inspect.Signature, ignore_self: bool = False) -> Self:
        res = cls(PFLStaticTypeType.FUNCTION)
        cnt = 0
        for param in sig.parameters.values():
            if ignore_self and cnt == 0 and param.name == "self":
                continue 
            assert param.annotation is not inspect.Parameter.empty, f"param {param.name} must have annotation"
            annotype = parse_type_may_optional_undefined(param.annotation)
            arg = PFLStaticType.from_annotype(annotype, is_type=False)
            res.childs.append(arg)
            if param.default is not inspect.Parameter.empty:
                arg.default = param.default
            arg.is_vaargs = param.kind == inspect.Parameter.VAR_POSITIONAL
            cnt += 1
        if sig.return_annotation is not inspect.Parameter.empty:
            if sig.return_annotation is not None:
                annotype = parse_type_may_optional_undefined(sig.return_annotation)
                res.return_type = PFLStaticType.from_annotype(annotype, is_type=False)
            else:
                res.return_type = PFLStaticType(PFLStaticTypeType.NONE_TYPE)
        return res 

    def __eq__(self, other):
        if not isinstance(other, PFLStaticType):
            return False
        if other.type == PFLStaticTypeType.ANY or self.type == PFLStaticTypeType.ANY:
            return True
        if self.type != other.type:
            return False
        if self.type == PFLStaticTypeType.DATACLASS_OBJECT:
            assert self.annotype is not None 
            return self.get_origin_type_checked() is other.get_origin_type_checked()
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
        assert self.support_binary_op(), f"not support binary op for {self.type}, {msg}"

    def check_support_compare_op(self, msg: str = ""):
        assert self.support_binary_op(), f"not support binary op for {self.type}, {msg}"

    def support_aug_assign(self):
        return self.type in _TYPE_SUPPORT_BINARY_OP

    def is_convertable(self, tgt: "PFLStaticType"):
        # TODO
        if tgt.type == PFLStaticTypeType.NUMBER or tgt.type == PFLStaticTypeType.BOOL:
            return self.type == PFLStaticTypeType.NUMBER or self.type == PFLStaticTypeType.BOOL
        elif tgt.type == PFLStaticTypeType.ARRAY or tgt.type == PFLStaticTypeType.OBJECT:
            if tgt.type == self.type:
                return self.childs[0].is_convertable(tgt.childs[0])
            return False
        return self == tgt

    def check_convertable(self, tgt: "PFLStaticType", desc: str):
        assert self.is_convertable(tgt), f"{desc} is not convertable from {self} to {tgt}"

@dataclasses.dataclass
class PFLStaticVar(PFLStaticType):
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

    def __repr__(self):
        if self in _PFLAST_TYPE_TO_STR:
            return _PFLAST_TYPE_TO_STR[self]
        return super().__repr__()

_PFLAST_TYPE_TO_STR = {
    PFLASTType.BLOCK: "block",
    PFLASTType.ASSIGN: "assign",
    PFLASTType.IF: "if",
    PFLASTType.AUG_ASSIGN: "aug_assign",
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

_AST_BINOP_TO_PFL_BINOP = {
    ast.Add: BinOpType.ADD,
    ast.Sub: BinOpType.SUB,
    ast.Mult: BinOpType.MULT,
    ast.Div: BinOpType.DIV,
    ast.Mod: BinOpType.MOD,
    ast.Pow: BinOpType.POW,
    ast.LShift: BinOpType.LSHIFT,
    ast.RShift: BinOpType.RSHIFT,
    ast.BitOr: BinOpType.BIT_OR,
    ast.BitXor: BinOpType.BIT_XOR,
    ast.BitAnd: BinOpType.BIT_AND,
}

class UnaryOpType(enum.IntEnum):
    INVERT = 0
    NOT = 1
    UADD = 2
    USUB = 3

_AST_UNARYOP_TO_PFL_UNARYOP = {
    ast.Invert: UnaryOpType.INVERT,
    ast.Not: UnaryOpType.NOT,
    ast.UAdd: UnaryOpType.UADD,
    ast.USub: UnaryOpType.USUB,
}

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
    
@dataclasses.dataclass
class PFLAstStmt(PFLAstNodeBase):
    pass

@dataclasses.dataclass
class PFLFunc(PFLAstNodeBase):
    args: list[PFLStaticVar]
    body: list[PFLAstStmt] = dataclasses.field(default_factory=list)

@dataclasses.dataclass(kw_only=True)
class PFLExpr(PFLAstNodeBase):
    st: PFLStaticType = dataclasses.field(default_factory=lambda: PFLStaticType(PFLStaticTypeType.UNKNOWN))
    is_const: Union[bool, Undefined] = undefined

    @staticmethod 
    def all_constexpr(*args: Optional["PFLExpr"]):
        for arg in args:
            if arg is not None:
                if not arg.is_const:
                    return False
        return True


@dataclasses.dataclass
class PFLAssign(PFLAstStmt):
    target: PFLExpr
    value: PFLExpr
    def __post_init__(self):
        assert self.value.st.is_convertable(self.target.st), f"{self.value.st} not convertable to {self.target.st}"

@dataclasses.dataclass
class PFLAugAssign(PFLAstStmt):
    target: PFLExpr
    op: BinOpType
    value: PFLExpr
    def __post_init__(self):
        assert self.target.st.support_aug_assign()
        assert self.value.st.is_convertable(self.target.st), f"{self.value.st} not convertable to {self.target.st}"

@dataclasses.dataclass
class PFLIf(PFLAstStmt):
    test: PFLExpr 
    body: list[PFLAstStmt]
    orelse: list[PFLAstStmt] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        test_dtype = self.test.st
        assert test_dtype.can_cast_to_bool(), f"test must be convertable to bool, but got {test_dtype}"
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
        self.st = PFLStaticType(PFLStaticTypeType.BOOL)
        self.is_const = PFLExpr.all_constexpr(self.left, self.right)
        return self 

@dataclasses.dataclass(kw_only=True)
class PFLUnaryOp(PFLExpr):
    op: UnaryOpType
    operand: PFLExpr

    def __post_init__(self):
        self.operand.st.check_support_binary_op("left")
        self.st = dataclasses.replace(self.operand.st)
        self.is_const = self.operand.is_const
        return self 

@dataclasses.dataclass(kw_only=True)
class PFLIfExp(PFLExpr):
    test: PFLExpr
    body: PFLExpr
    orelse: PFLExpr

    def __post_init__(self):
        assert self.test.st.can_cast_to_bool(), f"test must be convertable to bool, but got {self.test.st}"
        assert self.body.st == self.orelse.st, f"body and orelse must be same type, but got {self.body.st} and {self.orelse.st}"
        self.st = dataclasses.replace(self.body.st)
        return self 

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
        if left.st.type == PFLStaticTypeType.DATACLASS_OBJECT:
            resolved_custom_expr = left 
        elif right.st.type == PFLStaticTypeType.DATACLASS_OBJECT:
            resolved_custom_expr = right 
            self.is_right = True
        if resolved_custom_expr is not None:
            assert not resolved_custom_expr.st.is_temp, f"temp dataclass {resolved_custom_expr.st} (not stdlib) can't be used in custom operator"
            dcls_type = resolved_custom_expr.st.get_origin_type_checked()
            # use custom operator in left st if found
            op_name = _PFL_BINARY_TYPE_TO_METHOD_NAME[op]
            op_func = inspect.getattr_static(dcls_type, op_name, None)
            assert op_func is not None, f"can't find {op_name} in custom type {get_qualname_of_type(dcls_type)}"
            op_func_st = get_parse_context_checked().cached_parse_func(op_func, True)
            assert len(op_func_st.childs) == 1, f"custom operator {op_name} must have one arg, but got {len(op_func_st.childs)}"
            if left.st.type == PFLStaticTypeType.DATACLASS_OBJECT:
                right.st.check_convertable(op_func_st.childs[0], f"custom operator {op_name} arg")
            elif right.st.type == PFLStaticTypeType.DATACLASS_OBJECT:
                left.st.check_convertable(op_func_st.childs[0], f"custom operator {op_name} arg")
            is_custom_type = True
            custom_res_type = op_func_st.return_type
            assert custom_res_type is not None, f"custom operator {op_name} must have return type"
        return is_custom_type, custom_res_type

@dataclasses.dataclass(kw_only=True)
class PFLBinOp(PFLBinOpBase):
    op: BinOpType
    def __post_init__(self):
        is_custom_type, custom_res_type = self.resolve_custom_type(self.op)
        if not is_custom_type:
            self.left.st.check_support_binary_op("left")
            self.right.st.check_support_binary_op("right")
            self.st = PFLStaticType(PFLStaticTypeType.NUMBER)
        else:
            assert custom_res_type is not None 
            self.st = custom_res_type
        self.is_const = PFLExpr.all_constexpr(self.left, self.right)
        return self 

@dataclasses.dataclass(kw_only=True)
class PFLCompare(PFLBinOpBase):
    op: CompareType
    def __post_init__(self):
        is_custom_type = False
        if self.op == CompareType.IN or self.op == CompareType.NOT_IN:
            # contains operator don't support custom datatype
            assert self.left.st.type != PFLStaticTypeType.DATACLASS_OBJECT
            assert self.right.st.type != PFLStaticTypeType.DATACLASS_OBJECT

        if not (self.op == CompareType.IS or self.op == CompareType.IS_NOT):
            # overrideable operators
            is_custom_type, _ = self.resolve_custom_type(self.op)
        
        if not is_custom_type:
            # all operands are base type
            if not (self.op == CompareType.EQUAL or self.op == CompareType.NOT_EQUAL or self.op == CompareType.IS or self.op == CompareType.IS_NOT):
                # handle string-type compares
                if self.op == CompareType.IN or self.op == CompareType.NOT_IN:
                    assert self.left.st.type == PFLStaticTypeType.STRING and self.right.st.type == PFLStaticTypeType.OBJECT, f"left must be string and right must be object, but got {self.left.st.type} and {self.right.st.type}"
                else:
                    self.left.st.check_support_binary_op("left")
                    self.right.st.check_support_binary_op("right")
        self.st = PFLStaticType(PFLStaticTypeType.BOOL)
        self.is_const = PFLExpr.all_constexpr(self.left, self.right)
        return self 

@dataclasses.dataclass(kw_only=True)
class PFLCall(PFLExpr):
    # don't support kwargs due to js
    func: PFLExpr
    args: list[PFLExpr]
    is_ctor: Union[Undefined, bool] = undefined
    def __post_init__(self):
        # validate args
        is_ctor: bool = False
        if self.func.st.type == PFLStaticTypeType.DATACLASS_TYPE:
            # create std objects 
            is_ctor = True 
        self.is_ctor = is_ctor
        assert self.func.st.type == PFLStaticTypeType.FUNCTION or self.func.st.type == PFLStaticTypeType.DATACLASS_TYPE, f"func must be function/dcls, but got {self.func.st.type}"
        last_is_vaarg = False 
        if self.func.st.type == PFLStaticTypeType.FUNCTION:
            func_arg_types = self.func.st.childs
            if func_arg_types:
                last_is_vaarg = func_arg_types[-1].is_vaargs
        else:
            annotype = self.func.st.annotype
            assert annotype is not None 
            field_types = list(annotype.get_dataclass_field_annotated_types().values())
            func_arg_types = [PFLStaticType.from_annotype(f, is_type=False) for f in field_types]
        if not last_is_vaarg:
            assert len(self.args) <= len(func_arg_types), f"func {self.func} expect {len(func_arg_types)} args, but got {len(self.args)}"
        for i, a in enumerate(self.args):
            if not last_is_vaarg:
                func_arg = func_arg_types[i]
                a.st.check_convertable(func_arg, f"func {self.func} arg {i}")
            else:
                if i < len(func_arg_types) - 1:
                    func_arg = func_arg_types[i]
                    a.st.check_convertable(func_arg, f"func {self.func} arg {i}")
                else:
                    a.st.check_convertable(func_arg_types[-1], f"func {self.func} arg {i}")
        if self.func.st.type != PFLStaticTypeType.FUNCTION:
            self.st = dataclasses.replace(self.func.st, type=PFLStaticTypeType.DATACLASS_OBJECT, childs=[], return_type=None)
        else:
            assert self.func.st.return_type is not None, f"func {self.func} must have return type"
            self.st = dataclasses.replace(self.func.st.return_type)


@dataclasses.dataclass(kw_only=True)
class PFLName(PFLExpr):
    id: str
    is_store: Union[Undefined, bool] = undefined
    is_new: Union[Undefined, bool] = undefined
    def __post_init__(self):
        if self.st.type == PFLStaticTypeType.DATACLASS_TYPE or self.st.type == PFLStaticTypeType.DATACLASS_OBJECT:
            assert self.st.annotype is not None, "dataclass must have annotype"


@dataclasses.dataclass(kw_only=True)
class PFLAttribute(PFLExpr):
    value: PFLExpr
    attr: str
    is_store: Union[Undefined, bool] = undefined
    def __post_init__(self):
        if self.value.st.type == PFLStaticTypeType.DATACLASS_TYPE or self.value.st.type == PFLStaticTypeType.DATACLASS_OBJECT:
            assert self.st.annotype is not None, "dataclass must have annotype"
            assert self.st.annotype.is_dataclass_type()
            field_types = self.st.annotype.get_dataclass_fields_and_annotated_types()
            if self.attr in field_types:
                field_annotype, field = field_types[self.attr]
                if self.value.st.type == PFLStaticTypeType.DATACLASS_TYPE:
                    # access constant
                    default = field.default
                    assert default is not dataclasses.MISSING, f"access field {self.attr} by type must have default value"
                new_st = PFLStaticType.from_annotype(field_annotype, is_type=False)
                self.st = new_st
            else:
                assert not self.value.st.is_temp, f"function in temp dataclass {self.value.st} (not stdlib) can't be used"
                dcls_type = self.st.annotype.origin_type
                unbound_func = getattr(dcls_type, self.attr)
                if self.value.st.type == PFLStaticTypeType.DATACLASS_OBJECT:
                    ignore_self = True
                else:
                    ignore_self = False
                    assert inspecttools.isstaticmethod(dcls_type, self.attr), f"{self.attr} of {dcls_type} must be staticmethod"
                new_st = PFLStaticType.from_signature(inspect.signature(unbound_func), ignore_self=ignore_self)
                self.st = new_st
        else:
            if self.value.st.type == PFLStaticTypeType.OBJECT or self.value.st.type == PFLStaticTypeType.ARRAY:
                annotype = self.value.st.childs[0].annotype
                assert annotype is not None     
                methods = _dftype_with_gen_to_supported_methods(annotype.origin_type)[self.value.st.type]
            elif self.value.st.type == PFLStaticTypeType.STRING:
                methods = _PFLTYPE_TO_SUPPORTED_METHODS[self.value.st.type]
            else:
                raise ValueError(f"not support {self.value.st.type} for {self.attr}")
            assert self.attr in methods, f"not support {self.attr} for {self.value.st.type}"
            new_st = PFLStaticType.from_signature(methods[self.attr])
            self.st = new_st


@dataclasses.dataclass(kw_only=True)
class PFLConstant(PFLExpr):
    value: Any
    def __post_init__(self):
        if isinstance(self.value, bool):
            self.st = PFLStaticType(PFLStaticTypeType.BOOL)
        elif isinstance(self.value, int):
            self.st = PFLStaticType(PFLStaticTypeType.NUMBER)
        elif isinstance(self.value, float):
            self.st = PFLStaticType(PFLStaticTypeType.NUMBER)
        elif isinstance(self.value, str):
            self.st = PFLStaticType(PFLStaticTypeType.STRING)
        elif self.value is None:
            self.st = PFLStaticType(PFLStaticTypeType.NONE_TYPE)
        elif isinstance(self.value, Undefined):
            self.st = PFLStaticType(PFLStaticTypeType.UNDEFINED_TYPE)
        else:
            self.st = PFLStaticType(PFLStaticTypeType.UNKNOWN)
        self.is_const = True
        return self

@dataclasses.dataclass(kw_only=True)
class PFLSubscript(PFLExpr):
    value: PFLExpr
    slice: PFLExpr
    is_store: Union[Undefined, bool] = undefined
    def __post_init__(self):
        assert not self.value.st.is_optional() and not self.slice.st.is_optional()
        if self.value.st.type == PFLStaticTypeType.ARRAY:
            assert self.slice.st.type == PFLStaticTypeType.NUMBER, f"slice must be number, but got {self.slice.st.type}"
            self.st = self.value.st.childs[0]
        elif self.value.st.type == PFLStaticTypeType.OBJECT:
            assert self.slice.st.type == PFLStaticTypeType.STRING, f"slice must be string, but got {self.slice.st.type}"
            self.st = self.value.st.childs[0]
        elif self.value.st.type == PFLStaticTypeType.DATACLASS_OBJECT:
            resolved_custom_expr = self.value
            assert not resolved_custom_expr.st.is_temp, f"temp dataclass {resolved_custom_expr.st} (not stdlib) can't be used in custom operator"
            assert self.slice.st.type == PFLStaticTypeType.NUMBER
            dcls_type = resolved_custom_expr.st.get_origin_type_checked()
            # use custom operator in left st if found
            if self.is_store == True:
                op_name = "__setitem__"
            else:
                op_name = "__getitem__"
            op_func = inspect.getattr_static(dcls_type, op_name, None)
            assert op_func is not None, f"can't find {op_name} in custom type {get_qualname_of_type(dcls_type)}"
            op_func_st = get_parse_context_checked().cached_parse_func(op_func, ignore_self=True)
            assert len(op_func_st.childs) == 1, f"custom operator {op_name} must have one arg, but got {len(op_func_st.childs)}"
            self.slice.st.check_convertable(op_func_st.childs[0], f"custom operator {op_name}|{op_func_st} arg")
            assert op_func_st.return_type is not None, f"custom operator {op_name}|{op_func_st} must have return type"
            self.st = op_func_st.return_type
        else:
            raise ValueError(f"not support subscript for {self.value.st}")
        self.is_const = PFLExpr.all_constexpr(self.value, self.slice)
        return self 

@dataclasses.dataclass(kw_only=True)
class PFLArray(PFLExpr):
    elts: list[PFLExpr]
    def __post_init__(self):
        if not self.elts:
            self.st = PFLStaticType(PFLStaticTypeType.ARRAY, [PFLStaticType(PFLStaticTypeType.UNKNOWN)])
            self.is_const = True
            return self
        # all elts must be same type
        first_elt = self.elts[0]
        for elt in self.elts:
            assert first_elt.st == elt.st, f"all elts must be same type, but got {first_elt.st} and {elt.st}"
        self.st = PFLStaticType(PFLStaticTypeType.ARRAY, [dataclasses.replace(first_elt.st)])
        self.is_const = PFLExpr.all_constexpr(*self.elts)


@dataclasses.dataclass(kw_only=True)
class PFLDict(PFLExpr):
    keys: list[Optional[PFLExpr]]
    values: list[PFLExpr]
    def __post_init__(self):
        if not self.keys:
            self.st = PFLStaticType(PFLStaticTypeType.OBJECT, [PFLStaticType(PFLStaticTypeType.UNKNOWN)])
            self.is_const = True
            return self
        value_st: Optional[PFLStaticType] = None
        for key, value in zip(self.keys, self.values):
            if key is not None:
                value_st = value.st
        if value_st is None:
            for key, value in zip(self.keys, self.values):
                if key is None:
                    assert value.st.type == PFLStaticTypeType.OBJECT
                    value_st = value.st.childs[0]
                    break
        assert value_st is not None, "shouldn't happen"
        # all keys and values must be same type
        for key, value in zip(self.keys, self.values):
            if key is not None:
                assert key.st.type == PFLStaticTypeType.STRING, "object key must be string"
                assert value_st == value.st, f"all values must be same type, but got {value_st} and {value.st}"
            else:
                assert value.st.type == PFLStaticTypeType.OBJECT
                assert value_st == value.st.childs[0], f"all values must be same type, but got {value_st} and {value.st.childs[0]}"
        self.st = PFLStaticType(PFLStaticTypeType.OBJECT, [dataclasses.replace(value_st)])
        self.is_const = PFLExpr.all_constexpr(*self.keys, *self.values)

_ALL_SUPPORTED_AST_TYPES = {
    ast.BoolOp, 
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Subscript,
    ast.Attribute,
    ast.List,
    ast.Dict,
    ast.Assign,
    ast.AugAssign,
    ast.If,    
    ast.Expr,
    ast.IfExp,
}

class PFLAstParseError(Exception):
    def __init__(self, msg: str, node: ast.AST):
        super().__init__(msg)
        self.node = node

def _parse_expr_to_df_ast(expr: ast.expr, scope: dict[str, PFLStaticType]) -> PFLExpr:
    try:
        if isinstance(expr, ast.Name):
            if expr.id not in scope:
                raise PFLAstParseError(f"undefined name {expr.id}", expr)
            st = scope[expr.id]
            res = PFLName(PFLASTType.NAME, id=expr.id, st=st)
        elif isinstance(expr, ast.Attribute):
            value = _parse_expr_to_df_ast(expr.value, scope)
            attr = expr.attr
            st = value.st
            res = PFLAttribute(PFLASTType.ATTR, value=value, attr=attr, st=st)
        elif isinstance(expr, ast.Constant):
            res = PFLConstant(PFLASTType.CONSTANT, value=expr.value)
        elif isinstance(expr, ast.Subscript):
            value = _parse_expr_to_df_ast(expr.value, scope)
            slice = _parse_expr_to_df_ast(expr.slice, scope)
            is_store = undefined
            if isinstance(expr.ctx, ast.Store):
                is_store = True
            res = PFLSubscript(PFLASTType.SUBSCRIPT, value=value, slice=slice, is_store=is_store)
        elif isinstance(expr, ast.List):
            elts = [_parse_expr_to_df_ast(elt, scope) for elt in expr.elts]
            res = PFLArray(PFLASTType.ARRAY, elts=elts)
        elif isinstance(expr, ast.Dict):
            keys = [_parse_expr_to_df_ast(key, scope) if key is not None else None for key in expr.keys]
            values = [_parse_expr_to_df_ast(value, scope) for value in expr.values]
            res = PFLDict(PFLASTType.DICT, keys=keys, values=values)
        elif isinstance(expr, ast.BoolOp):
            op = BoolOpType.AND if isinstance(expr.op, ast.And) else BoolOpType.OR
            values = [_parse_expr_to_df_ast(value, scope) for value in expr.values]
            res = PFLBoolOp(PFLASTType.BOOL_OP, op=op, left=values[0], right=values[1])
        elif isinstance(expr, ast.BinOp):
            op = _AST_BINOP_TO_PFL_BINOP[type(expr.op)]
            left = _parse_expr_to_df_ast(expr.left, scope)
            right = _parse_expr_to_df_ast(expr.right, scope)
            res = PFLBinOp(PFLASTType.BIN_OP, op=op, left=left, right=right)
        elif isinstance(expr, ast.UnaryOp):
            op = _AST_UNARYOP_TO_PFL_UNARYOP[type(expr.op)]
            operand = _parse_expr_to_df_ast(expr.operand, scope)
            res = PFLUnaryOp(PFLASTType.UNARY_OP, op=op, operand=operand)
        elif isinstance(expr, ast.Compare):
            left = _parse_expr_to_df_ast(expr.left, scope)
            assert len(expr.ops) == 1
            op = _AST_COMPARE_TO_PFL_COMPARE[type(expr.ops[0])]
            assert len(expr.comparators) == 1
            right = _parse_expr_to_df_ast(expr.comparators[0], scope)
            res = PFLCompare(PFLASTType.COMPARISON, op=op, left=left, right=right)
        elif isinstance(expr, ast.Call):
            func = _parse_expr_to_df_ast(expr.func, scope)
            args = [_parse_expr_to_df_ast(arg, scope) for arg in expr.args]
            res = PFLCall(PFLASTType.CALL, func=func, args=args)
        elif isinstance(expr, ast.IfExp):
            res = PFLIfExp(PFLASTType.IF_EXP, test=_parse_expr_to_df_ast(expr.test, scope), 
                body=_parse_expr_to_df_ast(expr.body, scope), 
                orelse=_parse_expr_to_df_ast(expr.orelse, scope))
        else:
            raise NotImplementedError(f"not support {type(expr)}", expr)
        if isinstance(res, (PFLName, PFLAttribute, PFLSubscript)):
            assert isinstance(expr, (ast.Name, ast.Attribute, ast.Subscript))
            if isinstance(expr.ctx, ast.Store):
                res.is_store = True
    except:
        ctx = get_parse_context_checked()
        if ctx.error_node is None:
            ctx.error_node = expr
        raise 
    return res 

def _parse_block_to_df_ast(body: list[ast.stmt], scope: dict[str, PFLStaticType]) -> PFLFunc:
    block = PFLFunc(PFLASTType.BLOCK, [], [])
    for stmt in body:
        try:
            if not isinstance(stmt, tuple(_ALL_SUPPORTED_AST_TYPES)):
                raise PFLAstParseError(f"not support {type(stmt)}", stmt)
            if isinstance(stmt, ast.Assign):
                if len(stmt.targets) != 1:
                    raise PFLAstParseError("only support single assign", stmt)
                value = _parse_expr_to_df_ast(stmt.value, scope)
                tgt = stmt.targets[0]
                is_new_var = False
                if isinstance(tgt, ast.Name):
                    if tgt.id not in scope:
                        is_new_var = True
                    else:
                        value.st.check_convertable(scope[tgt.id], "assign value")
                    scope[tgt.id] = value.st
                target = _parse_expr_to_df_ast(stmt.targets[0], scope)
                if isinstance(target, PFLName):
                    target.is_new = is_new_var
                block.body.append(PFLAssign(PFLASTType.ASSIGN, target=target, value=value))
            elif isinstance(stmt, ast.AugAssign):
                target = _parse_expr_to_df_ast(stmt.target, scope)
                op = _AST_BINOP_TO_PFL_BINOP[type(stmt.op)]
                value = _parse_expr_to_df_ast(stmt.value, scope)
                block.body.append(PFLAugAssign(PFLASTType.AUG_ASSIGN, target=target, op=op, value=value))
                if isinstance(target, PFLName):
                    scope[target.id] = target.st
            elif isinstance(stmt, ast.If):
                scope = scope.copy()
                test = _parse_expr_to_df_ast(stmt.test, scope)
                ifbody = _parse_block_to_df_ast(stmt.body, scope)
                orelse = _parse_block_to_df_ast(stmt.orelse, scope)
                block.body.append(PFLIf(PFLASTType.IF, test=test, body=ifbody.body, orelse=orelse.body))
            elif isinstance(stmt, ast.Expr):
                block.body.append(PFLExprStmt(PFLASTType.EXPR_STMT, value=_parse_expr_to_df_ast(stmt.value, scope)))
            else:
                raise PFLAstParseError(f"not support {type(stmt)}", stmt)
        except:
            ctx = get_parse_context_checked()
            if ctx.error_node is None:
                ctx.error_node = stmt
            raise 
    return block

class RewriteSTLName(ast.NodeTransformer):
    def __init__(self, func_globals: dict[str, Any], backend: str = "js"):
        super().__init__()
        self.func_globals = func_globals
        self.backend = backend

    def _visit_Attribute_or_name(self, node: Union[ast.Attribute, ast.Name]):
        parts: list[str] = []
        cur_node = node 
        name_found = False
        while isinstance(cur_node, (ast.Attribute, ast.Name)):
            if isinstance(cur_node, ast.Attribute):
                parts.append(cur_node.attr)
                cur_node = cur_node.value
            else:
                parts.append(cur_node.id)
                name_found = True 
                break
        if not name_found:
            return self.generic_visit(node) 
        parts = parts[::-1]
        cur_obj = self.func_globals
        for part in parts:
            if isinstance(cur_obj, dict):
                if part in cur_obj:
                    cur_obj = cur_obj[part]
                else:
                    return self.generic_visit(node)
            else:
                if hasattr(cur_obj, part):
                    cur_obj = getattr(cur_obj, part)
                else:
                    return node
        if isinstance(cur_obj, type) or inspect.ismodule(cur_obj):
            item = STD_REGISTRY.get_item_by_dcls(cur_obj, self.backend)
            if item is not None:
                mapped_name_parts = item.mapped_name.split(".")
                # create new name and attributes
                cur_node = ast.Name(id=mapped_name_parts[-1], ctx=node.ctx, lineno=node.lineno, col_offset=node.col_offset)
                for part in mapped_name_parts[::-1][1:]:
                    cur_node = ast.Attribute(value=cur_node, attr=part, ctx=node.ctx, lineno=node.lineno, col_offset=node.col_offset)
                return cur_node
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        return self._visit_Attribute_or_name(node)

    def visit_Attribute(self, node: ast.Attribute):
        return self._visit_Attribute_or_name(node)

def parse_func_to_df_ast(func: Callable, scope: Optional[dict[str, PFLStaticType]] = None, backend: str = "js") -> tuple[PFLFunc, str]:
    if isinstance(func, staticmethod):
        func = func.__func__
    # print(func.__globals__)
    # return
    # get func code
    func_code_lines, _ = inspect.getsourcelines(func)
    func_code_lines = [line.rstrip() for line in func_code_lines]
    func_code_lines = clean_source_code(func_code_lines)
    code = "\n".join(func_code_lines)
    code = remove_common_indent_from_code(code)
    tree = ast.parse(code)
    tree = ast.fix_missing_locations(RewriteSTLName(func.__globals__).visit(tree))

    # find funcdef
    body = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            body = node.body
    assert body is not None 
    args, _ = parse_annotated_function(func)
    with enter_parse_context(PFLParseContext(func_code_lines)) as ctx:
        init_scope: dict[str, PFLStaticType] = {
            "len": PFLStaticType.from_signature(inspect.Signature([param_fn("x", list[Any])], return_annotation=int)),
            "print": PFLStaticType.from_signature(inspect.Signature([varparam_fn("x", Any)], return_annotation=None)),
            "int": PFLStaticType.from_signature(inspect.Signature([param_fn("x", Any)], return_annotation=int)),
            "float": PFLStaticType.from_signature(inspect.Signature([param_fn("x", Any)], return_annotation=float)),
            "str": PFLStaticType.from_signature(inspect.Signature([param_fn("x", Any)], return_annotation=str)),
            "bool": PFLStaticType.from_signature(inspect.Signature([param_fn("x", Any)], return_annotation=bool)),
        }
        for k, v in STD_REGISTRY.global_dict.items():
            init_scope[v.mapped_name] = PFLStaticType.from_annotype(parse_type_may_optional_undefined(v.dcls))
        if scope is not None:
            scope = init_scope
        dffunc_args: list[PFLStaticVar] = []
        for arg in args:
            param = arg.param 
            assert param is not None 
            # only support positional args
            assert param.kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY]
            arg_type = arg.type
            if arg_type not in _BASE_ANNO_TYPE_TO_PFLSTATIC_TYPE:
                if not dataclasses.is_dataclass(arg_type):
                    item = STD_REGISTRY.get_item_by_dcls(arg_type, backend)
                    if item is None:
                        raise NotImplementedError(f"can't find your type {arg_type} in std library. you must implement it.")
                    arg_type = item.dcls
                else:
                    assert inspect.isclass(arg_type)

                    for dcls_child in child_type_generator_with_dataclass(arg_type):
                        if dataclasses.is_dataclass(dcls_child):
                            item = STD_REGISTRY.get_item_by_dcls(dcls_child, backend)
                            if item is None:
                                # add a temp item for this dataclass type
                                temp_mapped_name = UniqueTreeId.from_parts(["temp", get_qualname_of_type(dcls_child)]).uid_encoded
                                STD_REGISTRY.global_dict[temp_mapped_name] = StdRegistryItem(dcls_child, temp_mapped_name, backend=backend, is_temp=True)

            anno_type = parse_type_may_optional_undefined(arg_type)
            st = PFLStaticVar.from_annotype(anno_type, is_type=False)
            st.name = param.name
            init_scope[param.name] = st
            dffunc_args.append(st)
        try:
            block = _parse_block_to_df_ast(body, init_scope)
            block.args = dffunc_args
        except:
            if ctx.error_node is not None:
                error_line = get_parse_context_checked().format_error_from_lines_node(ctx.error_node)
                print(error_line)
            raise 
    return block, code

def _ast_as_dict(obj):
    if isinstance(obj, PFLAstNodeBase):
        result = []
        for f in dataclasses.fields(obj):
            value = _ast_as_dict(getattr(obj, f.name))
            if not isinstance(value, Undefined):
                result.append((f.name, value))
        return dict(result)
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        return type(obj)(
            *[_ast_as_dict(v) for v in obj])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(_ast_as_dict(v)
                         for v in obj)
    elif isinstance(obj, dict):
        return type(obj)((_ast_as_dict(k),
                          _ast_as_dict(v))
                         for k, v in obj.items())
    else:
        if isinstance(obj, PFLStaticType):
            return obj.to_dict()
        return obj

def _ast_as_dict_for_dump(obj):
    if isinstance(obj, PFLAstNodeBase):
        result = []
        for f in dataclasses.fields(obj):
            value = _ast_as_dict_for_dump(getattr(obj, f.name))
            if not isinstance(value, Undefined):
                result.append((f.name, value))
        return dict(result)
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        return type(obj)(
            *[_ast_as_dict_for_dump(v) for v in obj])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(_ast_as_dict_for_dump(v)
                         for v in obj)
    elif isinstance(obj, dict):
        return type(obj)((_ast_as_dict_for_dump(k),
                          _ast_as_dict_for_dump(v))
                         for k, v in obj.items())
    else:
        if isinstance(obj, PFLStaticType):
            return str(obj)
        return obj


def pfl_ast_to_dict(node: PFLAstNodeBase):
    return _ast_as_dict(node)

def pfl_ast_dump(node: PFLAstNodeBase):
    return _ast_as_dict_for_dump(node)
