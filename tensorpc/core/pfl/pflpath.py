from collections.abc import Mapping, Sequence
import inspect
from typing import Any, Union

from tensorpc.core.tree_id import UniqueTreeId
from .parser import parse_expr_string_to_pfl_ast
from .backends import js 
from .pfl_ast import PFLExpr, PFLName, PFLAttribute, PFLConstant, PFLSlice, PFLSubscript, PFLArray, PFLTuple, PFLDict, PFLBoolOp, BoolOpType, PFLBinOp, PFLCompare, PFLUnaryOp, PFLCall, PFLIfExp, PFLFunc, PFLClass, PFLExprType, is_undefined, PFL_BUILTIN_PROXY_INIT_FN
from .pfl_reg import STD_REGISTRY, register_pfl_std
from tensorpc.utils.perfetto_colors import create_slice_name, perfetto_string_to_color
import json 
import numpy as np 

@register_pfl_std(mapped_name="getRoot", backend="js")
def getRoot() -> Any:
    raise NotImplementedError("getRoot isn't supported in backend.")

@register_pfl_std(mapped_name="getItem", backend="js")
def getItem(obj: Any, idx: Union[str, int]) -> Any:
    if isinstance(obj, Sequence) and isinstance(idx, int):
        return obj[idx]
    elif isinstance(obj, Mapping) and isinstance(idx, str):
        return obj[idx]
    else:
        return None

@register_pfl_std(mapped_name="getAttr", backend="js")
def getAttr(obj: Any, attr: str) -> Any:
    if hasattr(obj, attr):
        return getattr(obj, attr)
    return None

@register_pfl_std(mapped_name="cformat", backend="js")
def cformat(obj: Any, *attrs: Any) -> Any:
    return obj % attrs

@register_pfl_std(mapped_name="getItemPath", backend="js")
def getItemPath(obj: Any, attrs: list[Any]) -> Any:
    for attr in attrs:
        obj = obj[attr]
    return obj

@register_pfl_std(mapped_name="concat", backend="js")
def concat(*arrs: Any) -> Any:
    return sum(arrs, [])

@register_pfl_std(mapped_name="matchCase", backend="js")
def matchCase(cond: Any, items: Any) -> Any:
    if not isinstance(items, list):
        return None
    for pair in items:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None
        if pair[0] == cond:
            return pair[1]
    return None 

@register_pfl_std(mapped_name="matchCaseVarg", backend="js")
def matchCaseVarg(cond: Any, *items: Any) -> Any:
    if len(items) == 0 or len(items) % 2 != 0:
        return None
    for i in range(0, len(items), 2):
        if items[i] == cond:
            return items[i + 1]
    return None 

@register_pfl_std(mapped_name="npToList", backend="js")
def npToList(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return None 

@register_pfl_std(mapped_name="npGetSubArray", backend="js")
def npGetSubArray(obj: Any, index: int) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return None
        return obj[index]
    return None 

@register_pfl_std(mapped_name="npSliceFirstAxis", backend="js")
def npSliceFirstAxis(obj: Any, start: int, end: int) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return None
        return obj[start:end]
    return None 

@register_pfl_std(mapped_name="ndarrayGetItem", backend="js")
def ndarrayGetItem(obj: Any, *index: int) -> Any:
    if isinstance(obj, np.ndarray):
        return obj[index]
    return None 

@register_pfl_std(mapped_name="maximum", backend="js")
def maximum(x: Any, y: Any) -> Any:
    return max(x, y)

@register_pfl_std(mapped_name="array", backend="js")
def array(*x: Any) -> list[Any]:
    return list(x)

@register_pfl_std(mapped_name="minimum", backend="js")
def minimum(x: Any, y: Any) -> Any:
    return min(x, y)

@register_pfl_std(mapped_name="clamp", backend="js")
def clamp(x: Any, a: Any, b: Any) -> Any:
    return max(a, min(x, b))

@register_pfl_std(mapped_name="printForward", backend="js")
def printForward(*x: Any) -> Any:
    print(*x)
    return x[0]

@register_pfl_std(mapped_name="not_null", backend="js")
def not_null(*x: Any) -> Any:
    for v in x:
        if v is not None:
            return v
    return None

@register_pfl_std(mapped_name="join", backend="js")
def join(split: str, *arr: Any) -> Any:
    return split.join([str(v) for v in arr])


@register_pfl_std(mapped_name="where", backend="js")
def where(cond: bool, x: Any, y: Any) -> Any:
    return x if cond else y

@register_pfl_std(mapped_name="to_string", backend="js")
def to_string(x: Any) -> str:
    return str(x)


def _get_default_js_constants():
    return {
        "MathUtil": js.MathUtil,
        "Math": js.Math,
        "ColorUtil": js.ColorUtil,
        "PerfUtil": js.PerfUtil,
        "Numpy": js.Numpy,
        "getRoot": getRoot,
        "print": js.print_func,
        "int": js.int_func,
        "float": js.float_func,
        "str": js.str_func,
        "len": js.len_func,
        "bool": js.bool_func,
        "getItem": getItem,
        "getAttr": getAttr,
        "cformat": cformat,
        "getItemPath": getItemPath,
        "concat": concat,
        "matchCase": matchCase,
        "matchCaseVarg": matchCaseVarg,
        "npToList": npToList,
        "npGetSubArray": npGetSubArray,
        "npSliceFirstAxis": npSliceFirstAxis,
        "ndarrayGetItem": ndarrayGetItem,
        "maximum": maximum,
        "minimum": minimum,
        "clamp": clamp,
        "printForward": printForward,
        "not_null": not_null,
        "where": where,
        "array": array,
        "join": join,
        "to_string": to_string,
    }

class PFLPathEvaluator:
    def __init__(self, node: PFLExpr, backend: str):
        self._node = node 
        std_scope: dict[str, Any] = {}
        for k, v in STD_REGISTRY.global_dict.items():
            if v.backend is None or v.backend == backend:
                std_scope[v.mapped_name] = v.dcls
        std_scope.update(_get_default_js_constants())
        self._std_scope = std_scope

    def _get_subscript_target_slice(self, node: PFLSubscript, scope: dict[str, Any]):
        tgt = self._run_expr(node.value, scope)
        if isinstance(node.slice, Sequence):
            slice_strs = [self._run_expr(s, scope) for s in node.slice]
            slice_str = tuple(slice_strs)
        else:
            slice_str = self._run_expr(node.slice, scope)
        return tgt, slice_str

    def _run_expr(self, expr: PFLExpr, scope: dict[str, Any]) -> Any:
        if isinstance(expr, PFLName):
            if expr.id in self._std_scope:
                res = self._std_scope[expr.id]
            else:
                res = scope[expr.id]
            # return scope[expr.id]
        elif isinstance(expr, PFLAttribute):
            tgt = self._run_expr(expr.value, scope)
            if isinstance(tgt, dict) and isinstance(expr.attr, str):
                res = tgt.get(expr.attr, None)
            else:
                res = getattr(tgt, expr.attr)
        elif isinstance(expr, PFLConstant):
            res = expr.value
        elif isinstance(expr, PFLSlice):
            lo_str = None if is_undefined(expr.lo) else self._run_expr(expr.lo, scope)
            hi_str = None if is_undefined(expr.hi) else self._run_expr(expr.hi, scope)
            step_str = None if is_undefined(expr.step) else self._run_expr(
                expr.step, scope)
            res = slice(lo_str, hi_str, step_str)
        elif isinstance(expr, PFLSubscript):
            tgt, slice_str = self._get_subscript_target_slice(expr, scope)
            res = tgt[slice_str]
        elif isinstance(expr, PFLArray):
            res = [self._run_expr(elt, scope)
                                for elt in expr.elts]
        elif isinstance(expr, PFLTuple):
            res = tuple([self._run_expr(elt, scope)
                                for elt in expr.elts])
        elif isinstance(expr, PFLDict):
            res = {}
            for k, v in zip(expr.keys, expr.values):
                vv: Any = self._run_expr(v, scope)
                if k is None:
                    res.update(vv)
                else:
                    kk = self._run_expr(v, scope)
                    res[kk] = vv
        elif isinstance(expr, PFLBoolOp):
            if expr.op == BoolOpType.AND:
                early_exit = False
                res_arr: list[bool] = []
                for v in expr.values:
                    val = self._run_expr(v, scope)
                    if not val:
                        early_exit = True
                        break
                    res_arr.append(val)
                if early_exit:
                    res = False
                else:
                    res = all(res_arr)
            else:
                early_exit = False
                res_arr: list[bool] = []
                for v in expr.values:
                    val = self._run_expr(v, scope)
                    if val:
                        early_exit = True
                        break
                    res_arr.append(val)
                if early_exit:
                    res = True
                else:
                    res = any(res_arr)
        elif isinstance(expr, (PFLBinOp, PFLCompare)):
            left = self._run_expr(expr.left, scope)
            right = self._run_expr(expr.right, scope)
            res = expr.run(left, right)
        elif isinstance(expr, PFLUnaryOp):
            left = self._run_expr(expr.operand, scope)
            res = expr.run(left)
        elif isinstance(expr, PFLCall):
            func_val = self._run_expr(expr.func, scope)
            func_node = expr.func
            if isinstance(func_node, PFLName):
                # getRoot must be handled specially
                if func_node.id == "getRoot":
                    return scope
            if expr.func.st.proxy_dcls is not None:
                func_val = inspect.getattr_static(expr.func.st.proxy_dcls, PFL_BUILTIN_PROXY_INIT_FN)
            args = []
            kwargs = {}
            for arg_expr in expr.args:
                arg_expr_val = self._run_expr(arg_expr, scope)
                args.append(arg_expr_val)
            if not is_undefined(expr.keys):
                assert not is_undefined(expr.vals)
                for key_expr, value_expr in zip(expr.keys, expr.vals):
                    value_value = self._run_expr(value_expr, scope)
                    kwargs[key_expr] = value_value
            res = func_val(*args, **kwargs) 
        elif isinstance(expr, PFLIfExp):
            test = self._run_expr(expr.test, scope)
            if test:
                res = self._run_expr(expr.body, scope)
            else:
                res = self._run_expr(expr.orelse, scope)
        else:
            raise NotImplementedError(f"Unrecognized PFLExpr type: {type(expr)}")
        return res

    def run_expr(self, scope: dict[str, Any]):
        return self._run_expr(self._node, scope)

def compile_pflpath(pflpath: str):
    node = parse_expr_string_to_pfl_ast(pflpath, _get_default_js_constants(), {}, partial_type_infer=True)
    return node

def dump_pflpath(node: PFLExpr):
    from .parser import _AstAsDict
    return _AstAsDict(ignore_dcls_info=True, ignore_func_info=True)._ast_as_dict(node)

def compile_pflpath_to_compact_str(pflpath: str):
    node = parse_expr_string_to_pfl_ast(pflpath, _get_default_js_constants(), {}, partial_type_infer=True)
    node_dict = dump_pflpath(node)
    node_dict_str = json.dumps(node_dict, separators=(',', ':'))
    return UniqueTreeId.from_parts([pflpath, node_dict_str]).uid_encoded

def search(expression: Union[str, PFLExpr], data: dict) -> Any:
    if isinstance(expression, str):
        node = compile_pflpath(expression)
    else:
        node = expression
    evaluator = PFLPathEvaluator(node, backend="js")
    return evaluator.run_expr(data)