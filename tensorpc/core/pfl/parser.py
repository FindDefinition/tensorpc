import ast
from collections.abc import Sequence
import inspect
import sys
import traceback
from typing import Any, Callable, ForwardRef, Optional, Type, Union, cast

import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.annolib import (AnnotatedArg, AnnotatedType, T_dataclass, Undefined,
                                   child_dataclass_type_generator,
                                   child_type_generator_with_dataclass,
                                   is_undefined, parse_annotated_function,
                                   parse_type_may_optional_undefined,
                                   undefined)
from tensorpc.core.pfl.constants import PFL_COMPILE_META_ATTR, PFL_STDLIB_FUNC_META_ATTR
from tensorpc.core.funcid import (clean_source_code,
                                  remove_common_indent_from_code)
from tensorpc.core.moduleid import get_module_id_of_type, get_qualname_of_type
from tensorpc.core.tree_id import UniqueTreeId

from .core import (BACKEND_CONFIG_REGISTRY, BASE_ANNO_TYPE_TO_PFLSTATIC_TYPE, PFL_LOGGER, StaticEvalConfig, PFLCompilable, PFLCompileFuncMeta, PFLErrorFormatContext, PFLMetaInferResult, PFLParseConfig,
                   PFLParseContext, PFLExprInfo, PFLExprType,
                   enter_parse_context, get_compilable_meta, get_eval_cfg_in_parse_ctx, get_parse_context, get_parse_context_checked, param_fn,
                   varparam_fn, PFLStdlibFuncMeta)
from .pfl_ast import (BinOpType, BoolOpType, CompareType, PFLAnnAssign, PFLArg,
                      PFLArray, PFLAssign, PFLAstNodeBase, PFLAstStmt,
                      PFLASTType, PFLAttribute, PFLAugAssign, PFLBinOp,
                      PFLBoolOp, PFLBreak, PFLCall, PFLCompare, PFLConstant, PFLContinue, PFLDict,
                      PFLExpr, PFLExprStmt, PFLFor, PFLFunc, PFLIf, PFLIfExp, PFLModule,
                      PFLName, PFLReturn, PFLSlice, PFLStaticVar, PFLSubscript, PFLTreeNodeFinder,
                      PFLUnaryOp, PFLWhile, UnaryOpType, iter_child_nodes, unparse_pfl_expr, walk,
                      PFLAstParseError, PFLEvalError)

from .pfl_reg import ALL_COMPILE_TIME_FUNCS, STD_REGISTRY, StdRegistryItem, compiler_print_type, compiler_print_metadata

_ALL_SUPPORTED_AST_TYPES = {
    ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Name,
    ast.Constant, ast.Subscript, ast.Attribute, ast.List, ast.Dict, ast.Assign,
    ast.AugAssign, ast.If, ast.Expr, ast.IfExp, ast.For, ast.While,
    ast.AnnAssign, ast.Return, ast.FunctionDef
}

_AST_BINOP_TO_PFL_BINOP = {
    ast.Add: BinOpType.ADD,
    ast.Sub: BinOpType.SUB,
    ast.Mult: BinOpType.MULT,
    ast.Div: BinOpType.DIV,
    ast.FloorDiv: BinOpType.FLOOR_DIV,
    ast.Mod: BinOpType.MOD,
    ast.Pow: BinOpType.POW,
    ast.LShift: BinOpType.LSHIFT,
    ast.RShift: BinOpType.RSHIFT,
    ast.BitOr: BinOpType.BIT_OR,
    ast.BitXor: BinOpType.BIT_XOR,
    ast.BitAnd: BinOpType.BIT_AND,
}

_AST_UNARYOP_TO_PFL_UNARYOP = {
    ast.Invert: UnaryOpType.INVERT,
    ast.Not: UnaryOpType.NOT,
    ast.UAdd: UnaryOpType.UADD,
    ast.USub: UnaryOpType.USUB,
}

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

class PFLLibrary:
    def __init__(self, modules: dict[str, PFLModule]):
        all_compiled_units: dict[str, PFLFunc] = {}
        module_id_to_finder: dict[str, PFLTreeNodeFinder] = {}
        backend: str = ""
        for k, v in modules.items():
            all_units = v.get_all_compiled()
            for k1, v1 in all_units.items():
                if backend == "":
                    backend = v1.backend 
                else:
                    assert v1.backend == backend, "all compiled units must have the same backend"
                all_compiled_units[k1] = v1
            module_id_to_finder[k] = PFLTreeNodeFinder(v, (PFLAstStmt,))
        assert modules and all_compiled_units

        self._stmt_finder = module_id_to_finder
        self._modules = modules
        self._compiled_units = all_compiled_units
        self._backend = backend

    def get_compiled_unit(self, key: Union[str, Callable]) -> PFLFunc:
        if not isinstance(key, str):
            key = get_module_id_of_type(key)
        return self._compiled_units[key]

    def get_module_by_func(self, key: Union[str, Callable]) -> PFLModule:
        if not isinstance(key, str):
            key = get_module_id_of_type(key)
        module_key = key.split("::")[0]
        return self._modules[module_key]

    def find_stmt_by_path_lineno(self, module_id: str, lineno: int):
        finder = self._stmt_finder[module_id]
        return finder.find_nearest_node_by_line(lineno)

    @property 
    def modules(self) -> dict[str, PFLModule]:
        return self._modules

    @property
    def all_compiled(self) -> dict[str, PFLFunc]:
        return self._compiled_units

    @property
    def backend(self) -> str:
        return self._backend

def _parse_expr_to_df_ast(expr: ast.expr, scope: dict[str,
                                                      PFLExprInfo]) -> PFLExpr:
    source_loc = (expr.lineno, expr.col_offset, expr.end_lineno, expr.end_col_offset)
    try:
        if isinstance(expr, ast.Name):
            ctx = get_parse_context_checked()
            if expr in ctx.node_to_compilable:
                compilable = ctx.node_to_compilable[expr]
                if compilable.uid not in ctx._all_compiled:
                    assert compilable.uid not in ctx._current_compiling
                    ctx._current_compiling.add(compilable.uid)
                    func_node = parse_func_to_pfl_ast(compilable.func)
                    ctx._current_compiling.remove(compilable.uid)
                    ctx._all_compiled[compilable.uid] = func_node
                ctx.depend_compilables.append(compilable.uid)
                st = ctx._all_compiled[compilable.uid].st
            else:
                if expr.id not in scope:
                    raise PFLAstParseError(f"undefined name {expr.id}", expr)
                st = scope[expr.id]
            res = PFLName(PFLASTType.NAME,
                          source_loc,
                          id=expr.id,
                          st=dataclasses.replace(st))
            res.check_and_infer_type()
        elif isinstance(expr, ast.Attribute):
            ctx = get_parse_context_checked()
            if expr in ctx.node_to_std_item:
                item = ctx.node_to_std_item[expr]
                new_name = item.mapped_name
                st = ctx.cache.cached_parse_std_item(item)
                res = PFLName(PFLASTType.NAME,
                                source_loc,
                                id=new_name,
                                st=st)
            elif expr in ctx.node_to_compilable:
                compilable = ctx.node_to_compilable[expr]
                if compilable.uid not in ctx._all_compiled:
                    assert compilable.uid not in ctx._current_compiling
                    ctx._current_compiling.add(compilable.uid)
                    func_node = parse_func_to_pfl_ast(compilable.func)
                    ctx._current_compiling.remove(compilable.uid)
                    ctx._all_compiled[compilable.uid] = func_node
                ctx.depend_compilables.append(compilable.uid)
                st = ctx._all_compiled[compilable.uid].st
                # TODO better new name
                res = PFLName(PFLASTType.NAME,
                                source_loc,
                                id=compilable.uid,
                                st=st)
            else:
                value = _parse_expr_to_df_ast(expr.value, scope)
                attr = expr.attr
                st = value.st
                res = PFLAttribute(PFLASTType.ATTR,
                                source_loc,
                                value=value,
                                attr=attr,
                                st=st)
            res.check_and_infer_type()
        elif isinstance(expr, ast.Constant):
            res = PFLConstant(PFLASTType.CONSTANT,
                              source_loc,
                              value=expr.value)
            res.check_and_infer_type()
        elif isinstance(expr, ast.Slice):
            assert get_parse_context_checked(
            ).cfg.allow_slice, "slice is disabled in config"
            lo = _parse_expr_to_df_ast(
                expr.lower, scope) if expr.lower is not None else undefined
            hi = _parse_expr_to_df_ast(
                expr.upper, scope) if expr.upper is not None else undefined
            step = _parse_expr_to_df_ast(
                expr.step, scope) if expr.step is not None else undefined
            res = PFLSlice(PFLASTType.SLICE,
                           source_loc,
                           lo=lo,
                           hi=hi,
                           step=step)
            res.check_and_infer_type()
        elif isinstance(expr, ast.Subscript):
            value = _parse_expr_to_df_ast(expr.value, scope)
            slice: Union[Sequence[PFLExpr], PFLExpr]
            if isinstance(expr.slice, ast.Tuple):
                assert get_parse_context_checked(
                ).cfg.allow_nd_slice, "nd slice is disabled in config"
                slice = []
                for item in expr.slice.elts:
                    slice.append(_parse_expr_to_df_ast(item, scope))
            else:
                slice = _parse_expr_to_df_ast(expr.slice, scope)
            is_store = undefined
            if isinstance(expr.ctx, ast.Store):
                is_store = True
            res = PFLSubscript(PFLASTType.SUBSCRIPT,
                               source_loc,
                               value=value,
                               slice=slice,
                               is_store=is_store)
            res.check_and_infer_type()
        elif isinstance(expr, ast.List):
            elts = [_parse_expr_to_df_ast(elt, scope) for elt in expr.elts]
            res = PFLArray(PFLASTType.ARRAY,
                           source_loc,
                           elts=elts)
            res.check_and_infer_type()
        elif isinstance(expr, ast.Dict):
            keys = [
                _parse_expr_to_df_ast(key, scope) if key is not None else None
                for key in expr.keys
            ]
            values = [
                _parse_expr_to_df_ast(value, scope) for value in expr.values
            ]
            res = PFLDict(PFLASTType.DICT,
                          source_loc,
                          keys=keys,
                          values=values)
            res.check_and_infer_type()
        elif isinstance(expr, ast.BoolOp):
            op = BoolOpType.AND if isinstance(expr.op,
                                              ast.And) else BoolOpType.OR
            values = [
                _parse_expr_to_df_ast(value, scope) for value in expr.values
            ]
            res = PFLBoolOp(PFLASTType.BOOL_OP,
                            source_loc,
                            op=op,
                            left=values[0],
                            right=values[1])
            res.check_and_infer_type()
        elif isinstance(expr, ast.BinOp):
            op = _AST_BINOP_TO_PFL_BINOP[type(expr.op)]
            left = _parse_expr_to_df_ast(expr.left, scope)
            right = _parse_expr_to_df_ast(expr.right, scope)
            res = PFLBinOp(PFLASTType.BIN_OP,
                           source_loc,
                           op=op,
                           left=left,
                           right=right)
            res.check_and_infer_type()
        elif isinstance(expr, ast.UnaryOp):
            op = _AST_UNARYOP_TO_PFL_UNARYOP[type(expr.op)]
            operand = _parse_expr_to_df_ast(expr.operand, scope)
            res = PFLUnaryOp(PFLASTType.UNARY_OP,
                             source_loc,
                             op=op,
                             operand=operand)
            res.check_and_infer_type()
        elif isinstance(expr, ast.Compare):
            left = _parse_expr_to_df_ast(expr.left, scope)
            assert len(expr.ops) == 1
            op = _AST_COMPARE_TO_PFL_COMPARE[type(expr.ops[0])]
            assert len(expr.comparators) == 1
            right = _parse_expr_to_df_ast(expr.comparators[0], scope)
            res = PFLCompare(PFLASTType.COMPARISON,
                             source_loc,
                             op=op,
                             left=left,
                             right=right)
            res.check_and_infer_type()
        elif isinstance(expr, ast.Call):
            func = _parse_expr_to_df_ast(expr.func, scope)
            args = [_parse_expr_to_df_ast(arg, scope) for arg in expr.args]
            kw_keys: list[str] = []
            vals: list[PFLExpr] = []
            for arg in expr.args:
                assert not isinstance(
                    arg, ast.Starred), "don't support *arg for now"
            for kw in expr.keywords:
                assert kw.arg is not None, "don't support **kw"
                kw_keys.append(kw.arg)
                vals.append(_parse_expr_to_df_ast(kw.value, scope))
            # check is compile-time function
            if func.st.raw_func in ALL_COMPILE_TIME_FUNCS:
                assert len(args) == 1 and len(kw_keys) == 0, "compile-time functions only support one argument"
                if func.st.raw_func is compiler_print_type:
                    args_str = ", ".join(str(a.st) for a in args)
                    PFL_LOGGER.warning(args_str)
                res = args[0]
                expr = expr.args[0]
            else:
                # TODO support pfl function (PFLFunc)
                parse_cfg = get_parse_context_checked().cfg
                if not parse_cfg.allow_kw:
                    assert not expr.keywords, f"kwargs is disabled, you need to enable it in parse config."
                res = PFLCall(PFLASTType.CALL,
                            source_loc,
                            func=func,
                            args=args,
                            keys=kw_keys if kw_keys else undefined,
                            vals=vals if vals else undefined)
                res.check_and_infer_type()

        elif isinstance(expr, ast.IfExp):
            res = PFLIfExp(PFLASTType.IF_EXP,
                           source_loc,
                           test=_parse_expr_to_df_ast(expr.test, scope),
                           body=_parse_expr_to_df_ast(expr.body, scope),
                           orelse=_parse_expr_to_df_ast(expr.orelse, scope))
            res.check_and_infer_type()
        else:
            raise PFLAstParseError(f"not support {type(expr)}", expr)
        if isinstance(res, (PFLName, PFLAttribute, PFLSubscript)):
            assert isinstance(expr, (ast.Name, ast.Attribute, ast.Subscript)), f"expr must be Name, Attribute or Subscript, got {type(expr)}"
            if isinstance(expr.ctx, ast.Store):
                res.is_store = True
    except PFLAstParseError:
        raise 
    except BaseException as e:
        raise PFLAstParseError(f"Unknown error {e}", expr) from e
    return res

@dataclasses.dataclass
class ReturnInfo:
    complete: bool 
    all_return_stmts: list[PFLReturn]

def _evaluate_annotation_expr(annotation: ast.expr):
    ann_str = ast.unparse(annotation)
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

def _parse_block_to_pfl_ast(body: list[ast.stmt],
                            scope: dict[str, PFLExprInfo]) -> tuple[list[PFLAstStmt], ReturnInfo]:
    # TODO add return type support
    block: list[PFLAstStmt] = []
    # block = PFLFunc(PFLASTType.BLOCK, -1, -1, "", [], [])
    return_info = ReturnInfo(complete=False, all_return_stmts=[])
    for stmt in body:
        source_loc = (stmt.lineno, stmt.col_offset, stmt.end_lineno, stmt.end_col_offset)

        try:
            if not isinstance(stmt, tuple(_ALL_SUPPORTED_AST_TYPES)):
                raise PFLAstParseError(f"not support {type(stmt)}", stmt)
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                if isinstance(stmt, ast.AnnAssign):
                    assert stmt.simple == 1, "only support simple ann assign"
                if isinstance(stmt, ast.Assign):
                    if len(stmt.targets) != 1:
                        raise PFLAstParseError("only support single assign",
                                               stmt)
                if stmt.value is not None:
                    value = _parse_expr_to_df_ast(stmt.value, scope)
                else:
                    value = None
                if isinstance(stmt, ast.Assign):
                    tgt = stmt.targets[0]
                else:
                    tgt = stmt.target
                is_new_var = False
                if isinstance(tgt, ast.Name) and value is not None:
                    if tgt.id not in scope:
                        is_new_var = True
                    else:
                        value.st.check_convertable(scope[tgt.id],
                                                   "assign value")
                    scope[tgt.id] = value.st
                target = _parse_expr_to_df_ast(tgt, scope)
                if isinstance(target, PFLName) and value is not None:
                    target.is_new = is_new_var
                if isinstance(stmt, ast.Assign):
                    assert value is not None
                    target.st = dataclasses.replace(value.st)
                    node = PFLAssign(PFLASTType.ASSIGN,
                                  source_loc,
                                  target=target,
                                  value=value)
                    node.check_and_infer_type()
                    block.append(node)
                else:
                    assert value is not None
                    ann_res = _evaluate_annotation_expr(stmt.annotation)
                    target.st = PFLExprInfo.from_annotype(
                        parse_type_may_optional_undefined(ann_res),
                        is_type=False)
                    node = PFLAnnAssign(PFLASTType.ANN_ASSIGN,
                                     source_loc,
                                     target=target,
                                     annotation=ast.unparse(stmt.annotation),
                                     value=value)
                    node.check_and_infer_type()
                    block.append(node)
            elif isinstance(stmt, ast.AugAssign):
                target = _parse_expr_to_df_ast(stmt.target, scope)
                op = _AST_BINOP_TO_PFL_BINOP[type(stmt.op)]
                value = _parse_expr_to_df_ast(stmt.value, scope)
                node = PFLAugAssign(PFLASTType.AUG_ASSIGN,
                                 source_loc,
                                 target=target,
                                 op=op,
                                 value=value)
                node.check_and_infer_type()
                block.append(node)
                if isinstance(target, PFLName):
                    scope[target.id] = target.st
            elif isinstance(stmt, ast.If):
                test = _parse_expr_to_df_ast(stmt.test, scope)
                private_scope_if = scope.copy()
                ifbody, if_rinfo = _parse_block_to_pfl_ast(stmt.body, private_scope_if)
                private_scope_else = scope.copy()
                orelse, orelse_rinfo = _parse_block_to_pfl_ast(stmt.orelse, private_scope_else)
                if if_rinfo.complete and orelse_rinfo.complete:
                    return_info.complete = True
                return_info.all_return_stmts.extend(if_rinfo.all_return_stmts)
                return_info.all_return_stmts.extend(orelse_rinfo.all_return_stmts)
                common_vars = undefined
                if get_parse_context_checked().cfg.allow_new_var_after_if:
                    # compare and merge scopes
                    # 1. get new variables in each scope
                    new_vars_if = set(private_scope_if.keys()) - set(scope.keys())
                    new_vars_else = set(private_scope_else.keys()) - set(scope.keys())
                    # 2. get common variables in both scopes, common vars must have same type.
                    common_vars = list(new_vars_if & new_vars_else)
                    for common_var in common_vars:
                        var_in_if = private_scope_if[common_var]
                        var_in_else = private_scope_else[common_var]
                        merged = var_in_if.try_merge_two_info(var_in_else)
                        scope[common_var] = merged
                    
                node = PFLIf(PFLASTType.IF,
                          source_loc,
                          test=test,
                          body=ifbody,
                          orelse=orelse,
                          _new=common_vars)
                node.check_and_infer_type()
                block.append(node)
            elif isinstance(stmt, ast.For):
                # variable created in for/while scope won't leaked to parent scope.
                private_scope = scope.copy()
                value = _parse_expr_to_df_ast(stmt.iter, private_scope)
                tgt = stmt.target
                assert isinstance(tgt, ast.Name)
                if value.st.type == PFLExprType.ARRAY:
                    target_st = value.st.childs[0]
                elif value.st.type == PFLExprType.RANGE:
                    target_st = PFLExprInfo(PFLExprType.NUMBER, annotype=parse_type_may_optional_undefined(int))
                else:
                    raise NotImplementedError(
                        "for loop iter type must be array or range object")
                is_new_var = False
                if isinstance(tgt, ast.Name):
                    if tgt.id not in private_scope:
                        is_new_var = True
                    else:
                        target_st.check_convertable(private_scope[tgt.id],
                                                   "assign value")
                    private_scope[tgt.id] = target_st
                target = _parse_expr_to_df_ast(stmt.target, private_scope)
                forbody, rinfo = _parse_block_to_pfl_ast(stmt.body, private_scope)
                return_info.all_return_stmts.extend(rinfo.all_return_stmts)
                res_node = PFLFor(PFLASTType.FOR,
                           source_loc,
                           target=target,
                           iter=value,
                           body=forbody)
                res_node.check_and_infer_type()
                block.append(res_node)
            elif isinstance(stmt, ast.While):
                private_scope = scope.copy()
                test = _parse_expr_to_df_ast(stmt.test, private_scope)
                forbody, rinfo = _parse_block_to_pfl_ast(stmt.body, private_scope)
                return_info.all_return_stmts.extend(rinfo.all_return_stmts)
                node = PFLWhile(PFLASTType.WHILE,
                             source_loc,
                             test=test,
                             body=forbody)
                node.check_and_infer_type()

                block.append(node)
            elif isinstance(stmt, ast.Expr):
                node = PFLExprStmt(PFLASTType.EXPR_STMT,
                                source_loc,
                                value=_parse_expr_to_df_ast(stmt.value,
                                                            scope))
                block.append(node)
            elif isinstance(stmt, ast.Return):
                value = None 
                if stmt.value is not None:
                    value = _parse_expr_to_df_ast(stmt.value, scope)
                ret_stmt = PFLReturn(PFLASTType.RETURN, source_loc, value=value)
                
                block.append(ret_stmt)
                return_info.all_return_stmts.append(ret_stmt)
                return_info.complete = True
                # for return/break/continue, ignore all following statements
                break
            elif isinstance(stmt, ast.Break):
                block.append(PFLBreak(PFLASTType.BREAK, source_loc))
                break
            elif isinstance(stmt, ast.Continue):
                block.append(PFLContinue(PFLASTType.CONTINUE, source_loc))
                break
            elif isinstance(stmt, ast.FunctionDef):
                assert not stmt.args.posonlyargs, "posonlyargs is not supported in PFL"
                assert not stmt.args.vararg, "vararg is not supported in PFL"
                assert not stmt.args.kwonlyargs, "kwonlyargs is not supported in PFL"
                assert not stmt.args.kwarg, "kwarg is not supported in PFL"
                args: list[PFLArg] = []
                private_scope = scope.copy()

                num_arg_no_default = len(stmt.args.args) - len(
                    stmt.args.defaults)
                for arg in stmt.args.args:
                    arg_loc = (arg.lineno, arg.col_offset, arg.end_lineno, arg.end_col_offset)
                    assert arg.annotation is not None, "arg annotation must be provided in PFL"
                    ann_res = _evaluate_annotation_expr(arg.annotation)
                    st = PFLExprInfo.from_annotype(
                            parse_type_may_optional_undefined(ann_res))
                    st.arg_name = arg.arg
                    arg_obj = PFLArg(PFLASTType.ARG, arg_loc, arg=arg.arg, st=st, annotation=ast.unparse(arg.annotation))
                    args.append(arg_obj)
                    private_scope[arg_obj.arg] = arg_obj.st
                for pfl_arg, default in zip(args[num_arg_no_default:], stmt.args.defaults):
                    default_pfl = _parse_expr_to_df_ast(default, scope)
                    pfl_arg.default = default_pfl
                
                funbody, rinfo = _parse_block_to_pfl_ast(stmt.body, private_scope)
                if not isinstance(funbody[-1], PFLReturn):
                    rinfo.all_return_stmts.append(
                        PFLReturn(PFLASTType.RETURN, (-1, -1, None, None), value=PFLConstant(PFLASTType.CONSTANT, (-1, -1, None, None), value=None)))

                ret_sts: list[PFLExprInfo] = []
                if rinfo.all_return_stmts:
                    first_rstmt = rinfo.all_return_stmts[0]
                    first_rstmt_st = PFLExprInfo(PFLExprType.NONE_TYPE) if first_rstmt.value is None else first_rstmt.value.st
                    ret_sts.append(first_rstmt_st)
                    for rstmt in rinfo.all_return_stmts[1:]:
                        rstmt_st = PFLExprInfo(PFLExprType.NONE_TYPE) if rstmt.value is None else rstmt.value.st
                        assert rstmt_st.is_equal_type(first_rstmt_st), \
                            f"all return stmts must have same type, but got {rstmt_st} and {first_rstmt_st}"
                        ret_sts.append(rstmt_st)
                func_node_st = PFLExprInfo(
                    type=PFLExprType.FUNCTION, 
                    childs=[a.st for a in args],
                    return_type=ret_sts[0]
                )
                func_node = PFLFunc(PFLASTType.FUNC, source_loc, name=stmt.name, args=args, st=func_node_st, body=funbody)
                if ret_sts:
                    func_node.ret_st = ret_sts[0]
                if stmt.decorator_list:
                    # ctx = get_parse_context_checked()
                    # ctx._disable_type_check = True
                    func_node.decorator_list = [_parse_expr_to_df_ast(e, scope) for e in stmt.decorator_list]
                    # ctx._disable_type_check = False
                if stmt.returns is not None:
                    ann_res = _evaluate_annotation_expr(stmt.returns)
                    st = PFLExprInfo.from_annotype(
                            parse_type_may_optional_undefined(ann_res))
                    if ret_sts:
                        assert st.is_equal_type(ret_sts[0])
                    func_node.ret_st = st
                # TODO disable nested func support
                # always clear return info when func is end
                # TODO add function to scope
                return_info.all_return_stmts.clear()
                # func_node.end_scope = private_scope.copy()
                block.append(func_node)
            else:
                raise PFLAstParseError(f"not support {type(stmt)}", stmt)
        except PFLAstParseError:
            raise 
        except BaseException as e:
            raise PFLAstParseError(f"Unknown error {e}", stmt) from e
    return block, return_info


class RewriteSTLName(ast.NodeTransformer):

    def __init__(self, func_globals: dict[str, Any], error_ctx: PFLErrorFormatContext, backend: str = "js"):
        super().__init__()
        self.func_globals = func_globals
        self.backend = backend

        self._node_to_std_item: dict[ast.AST, StdRegistryItem] = {}
        self._node_to_compilable: dict[ast.AST, PFLCompilable] = {}

        self.error_ctx = error_ctx

    def _visit_Attribute_or_name(self, node: Union[ast.Attribute, ast.Name]):
        parts: list[str] = []
        parts_node: list[ast.AST] = []
        cur_node = node
        name_found = False
        while isinstance(cur_node, (ast.Attribute, ast.Name)):
            if isinstance(cur_node, ast.Attribute):
                parts.append(cur_node.attr)
                parts_node.append(cur_node)
                cur_node = cur_node.value
            else:
                parts.append(cur_node.id)
                parts_node.append(cur_node)
                name_found = True
                break
        if not name_found:
            return self.generic_visit(node)
        parts = parts[::-1]
        parts_node = parts_node[::-1]
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
        if isinstance(cur_obj, type) or inspect.ismodule(cur_obj) or inspect.isfunction(cur_obj):
            item = STD_REGISTRY.get_item_by_dcls(cur_obj, self.backend)
            if item is not None:
                # print(ast.unparse(cur_node), item.mapped_name)
                self._node_to_std_item[node] = item
                # don't access child nodes
                return node
            else:
                # check is marked as pfl compilable
                # TODO support compile class
                if inspect.isfunction(cur_obj):
                    meta = get_compilable_meta(cur_obj)
                    if meta is not None:
                        if meta.backends is None or self.backend in meta.backends:
                            func_uid = get_module_id_of_type(cur_obj)
                            self._node_to_compilable[node] = PFLCompilable(cur_obj, func_uid, meta) 
                            return node 
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        return self._visit_Attribute_or_name(node)

    def visit_Attribute(self, node: ast.Attribute):
        return self._visit_Attribute_or_name(node)

def _register_temp_dcls_to_std(args: list[AnnotatedArg], backend: str, global_dict: dict):
    for arg in args:
        param = arg.param
        assert param is not None
        # only support positional args
        assert param.kind in [
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY
        ]
        arg_type = arg.type
        if arg_type not in BASE_ANNO_TYPE_TO_PFLSTATIC_TYPE:
            if not dataclasses.is_dataclass(arg_type):
                raise NotImplementedError(
                    f"can't find your type {arg_type} in std library. you must implement it."
                )
            else:
                assert inspect.isclass(arg_type)

                for dcls_child in child_type_generator_with_dataclass(
                        arg_type):
                    if dataclasses.is_dataclass(dcls_child):
                        item = STD_REGISTRY.get_item_by_dcls(
                            dcls_child, backend)
                        if item is None:
                            # add a temp item for this dataclass type
                            temp_mapped_name = UniqueTreeId.from_parts([
                                backend, "temp",
                                get_qualname_of_type(dcls_child)
                            ]).uid_encoded
                            global_dict[
                                (temp_mapped_name, backend)] = StdRegistryItem(
                                    dcls_child,
                                    temp_mapped_name,
                                    backend=backend,
                                    is_temp=True)

def _get_module_code_by_fn(func: Callable):
    mod = inspect.getmodule(func)
    assert mod is not None, "module_code_getter must be provided if func isn't a module function"
    module_code = inspect.getsource(mod)
    return module_code

def parse_func_to_pfl_ast(
        func: Callable,
        scope: Optional[dict[str, PFLExprInfo]] = None,
        backend: str = "js",
        parse_cfg: Optional[PFLParseConfig] = None,
        func_code_getter: Callable[[Any], tuple[list[str], int]] = inspect.getsourcelines,
        module_code_getter: Callable[[Any], str] = _get_module_code_by_fn,
        all_compiled: Optional[dict[str, PFLFunc]] = None) -> PFLFunc:
    if parse_cfg is None:
        assert backend in BACKEND_CONFIG_REGISTRY, "you must register backend config first if parse_cfg isn't provided."
        parse_cfg = BACKEND_CONFIG_REGISTRY[backend]
    if isinstance(func, staticmethod):
        func = func.__func__
    func_uid = get_module_id_of_type(func)
    outer_ctx = get_parse_context()
    if outer_ctx is not None:
        func_code_getter = outer_ctx.func_code_getter
        module_code_getter = outer_ctx.module_code_getter
    # TODO should we include the decorators?
    func_code_lines, first_lineno = func_code_getter(func)
    func_code_lines = [l.rstrip() for l in func_code_lines]
    module_code = module_code_getter(func)
    code = "\n".join(func_code_lines)
    tree = ast.parse(code)
    transformer = RewriteSTLName(func.__globals__, PFLErrorFormatContext(func_code_lines), backend=backend)
    tree = ast.fix_missing_locations(transformer.visit(tree))
    # find funcdef
    body = None
    func_node: Optional[ast.FunctionDef] = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
            body = node.body
            func_node = node
    assert body is not None and func_node is not None 
    for child_node in ast.walk(func_node):
        # add first_lineno to all nodes
        if hasattr(child_node, 'lineno'):
            child_node.lineno += first_lineno - 1 # type: ignore
        if hasattr(child_node, 'end_lineno'):
            child_node.end_lineno += first_lineno - 1 # type: ignore

    args, _ = parse_annotated_function(func)
    anno_eval_globals = func.__globals__.copy()
    outer_ctx = get_parse_context()
    is_root = True
    if outer_ctx is not None:
        assert all_compiled is None
        parse_ctx = PFLParseContext.from_outer_ctx(outer_ctx, module_code.split("\n"), anno_eval_globals, 
            node_to_std_item=transformer._node_to_std_item,
            node_to_compilable=transformer._node_to_compilable)
        is_root = False
    else:
        parse_ctx = PFLParseContext(module_code.split("\n"), anno_eval_globals,
                                cfg=parse_cfg, backend=backend,
                                node_to_std_item=transformer._node_to_std_item,
                                node_to_compilable=transformer._node_to_compilable,
                                func_code_getter=func_code_getter,
                                module_code_getter=module_code_getter)
        if all_compiled is not None:
            parse_ctx._all_compiled = all_compiled
    _register_temp_dcls_to_std(args, backend, parse_ctx.cache._temp_dcls_dict)
    with enter_parse_context(parse_ctx) as ctx:
        # if is_root:
        init_scope: dict[str, PFLExprInfo] = {}
        for k, v in STD_REGISTRY.global_dict.items():
            if v.backend is None or v.backend == backend:
                init_scope[v.mapped_name] = ctx.cache.cached_parse_std_item(v)
                if not v.is_func:
                    anno_eval_globals[v.mapped_name] = v.dcls
        if scope is None:
            scope = init_scope.copy()
        else:
            scope = {**init_scope, **scope}
        try:
            body, return_info = _parse_block_to_pfl_ast([func_node], scope)
            assert isinstance(body[0], PFLFunc), \
                "the first node of block must be PFLFunc"
            if not isinstance(body[-1], PFLReturn):
                return_info.all_return_stmts.append(
                    PFLReturn(PFLASTType.RETURN, (-1, -1, None, None), value=PFLConstant(PFLASTType.CONSTANT, (-1, -1, None, None), value=None)))
            block = body[0]
            block.uid = func_uid
            block.deps = ctx.depend_compilables
            block.backend = backend
            block.st.compiled_uid = func_uid

            block.compile_info.code = code
            block.compile_info.first_lineno = first_lineno
            block.compile_info.original = func
            if is_root:
                ctx._all_compiled[func_uid] = block

        except PFLAstParseError as e:
            error_line = get_parse_context_checked(
            ).format_error_from_lines_node(e.node)
            print(error_line)
            raise e
    return block

def parse_func_to_pfl_library(
        func: Callable,
        scope: Optional[dict[str, PFLExprInfo]] = None,
        backend: str = "js",
        parse_cfg: Optional[PFLParseConfig] = None,
        func_code_getter: Callable[[Any], tuple[list[str], int]] = inspect.getsourcelines,
        module_code_getter: Callable[[Any], str] = _get_module_code_by_fn) -> PFLLibrary:
    """Parse func and its dependencies to a PFL library.
    this function will parse whole file instead of func code only to ast. 
    if your func is dynamic generated, you need to use `tempfile_in_linecache` to add your dynamic code to linecache.
    """
    all_compiled: dict[str, PFLFunc] = {}
    parse_func_to_pfl_ast(func, scope, backend, parse_cfg, func_code_getter, module_code_getter, all_compiled)
    # TODO: better modules 
    all_modules: dict[str, PFLModule] = {}
    for k, v in all_compiled.items():
        module_id = v.get_module_import_path()
        if module_id not in all_modules:
            module_code = module_code_getter(v.compile_info.original)
            mod = PFLModule(PFLASTType.MODULE, (-1, -1, None, None), uid=module_id)
            mod.compile_info.code = module_code
            all_modules[module_id] = mod
        all_modules[module_id].body.append(v)
    return PFLLibrary(all_modules)

def parse_expr_to_df_ast(
    expr_str: str,
    var_scope: Optional[dict[str, Any]] = None,
    backend: str = "js",
    parse_cfg: Optional[PFLParseConfig] = None
) -> tuple[PFLExpr, dict[str, Any]]:
    if parse_cfg is None:
        assert backend in BACKEND_CONFIG_REGISTRY, "you must register backend config first if parse_cfg isn't provided."
        parse_cfg = BACKEND_CONFIG_REGISTRY[backend]
    expr_str_lines = expr_str.split("\n")
    tree = ast.parse(expr_str, mode="eval")
    node_to_std_item = {}
    if var_scope is not None:
        transformer = RewriteSTLName(var_scope, PFLErrorFormatContext(expr_str_lines), backend=backend)
        tree = ast.fix_missing_locations(transformer.visit(tree))
        node_to_std_item = transformer._node_to_std_item
        # tree = ast.fix_missing_locations(
        #     RewriteSTLName(var_scope, PFLErrorFormatContext(expr_str_lines), backend=backend).visit(tree))
    assert isinstance(tree, ast.Expression)
    tree_expr = tree.body
    # find funcdef
    with enter_parse_context(PFLParseContext(expr_str_lines, {},
                                             cfg=parse_cfg, backend=backend,
                                             node_to_std_item=node_to_std_item)) as ctx:
        init_scope: dict[str, PFLExprInfo] = {}
        for k, v in STD_REGISTRY.global_dict.items():
            init_scope[v.mapped_name] = ctx.cache.cached_parse_std_item(v)
        if var_scope is None:
            scope = init_scope.copy()
        else:
            scope = init_scope.copy()
            for k, v in var_scope.items():
                scope[k] = PFLExprInfo.from_annotype(
                    parse_type_may_optional_undefined(type(v)), is_type=False)
        try:
            res = _parse_expr_to_df_ast(tree_expr, scope)
        except PFLAstParseError as e:
            error_line = get_parse_context_checked(
            ).format_error_from_lines_node(e.node)
            print(error_line)
            raise e
    return res, scope


def _ast_as_dict(obj):
    if isinstance(obj, PFLAstNodeBase):
        result = []
        for f in dataclasses.fields(obj):
            value = _ast_as_dict(getattr(obj, f.name))
            if not isinstance(value, Undefined):
                result.append((f.name, value))
        return dict(result)
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        return type(obj)(*[_ast_as_dict(v) for v in obj])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(_ast_as_dict(v) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)(
            (_ast_as_dict(k), _ast_as_dict(v)) for k, v in obj.items())
    else:
        if isinstance(obj, PFLExprInfo):
            return obj.to_dict()
        return obj


def _ast_as_dict_for_dump(obj):
    if isinstance(obj, PFLAstNodeBase):
        result = []
        for f in dataclasses.fields(obj):
            # FIXME: better way to remove code field in PFLFunc
            if f.name == "compile_info" or f.name == "source_loc":
                continue 
            value = _ast_as_dict_for_dump(getattr(obj, f.name))
            if not isinstance(value, Undefined):
                result.append((f.name, value))
        return dict(result)
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        return type(obj)(*[_ast_as_dict_for_dump(v) for v in obj])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(_ast_as_dict_for_dump(v) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)((_ast_as_dict_for_dump(k), _ast_as_dict_for_dump(v))
                         for k, v in obj.items())
    else:
        if isinstance(obj, PFLExprInfo):
            return str(obj)
        return obj


def pfl_ast_to_dict(node: PFLAstNodeBase):
    return _ast_as_dict(node)


def ast_dump(node: PFLAstNodeBase):
    return _ast_as_dict_for_dump(node)
