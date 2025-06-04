import ast
import inspect
import sys
from typing import Any, Callable, ForwardRef, Optional, Type, Union

import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.annolib import (AnnotatedType, T_dataclass, Undefined,
                                   child_dataclass_type_generator,
                                   child_type_generator_with_dataclass,
                                   parse_annotated_function,
                                   parse_type_may_optional_undefined,
                                   undefined)
from tensorpc.core.pfl.constants import PFL_FUNC_META_ATTR
from tensorpc.core.funcid import (clean_source_code,
                                  remove_common_indent_from_code)
from tensorpc.core.moduleid import get_qualname_of_type
from tensorpc.core.tree_id import UniqueTreeId

from .core import (BASE_ANNO_TYPE_TO_PFLSTATIC_TYPE, PFLParseContext,
                   PFLExprInfo, PFLExprType, enter_parse_context,
                   get_parse_context_checked, param_fn, varparam_fn,
                   PFLFuncMeta)
from .pfl_ast import (BinOpType, BoolOpType, CompareType, PFLAnnAssign,
                      PFLArray, PFLAssign, PFLAstNodeBase, PFLAstStmt,
                      PFLASTType, PFLAttribute, PFLAugAssign, PFLBinOp,
                      PFLBoolOp, PFLCall, PFLCompare, PFLConstant, PFLDict,
                      PFLExpr, PFLExprStmt, PFLFor, PFLFunc, PFLIf, PFLIfExp,
                      PFLName, PFLStaticVar, PFLSubscript, PFLUnaryOp,
                      PFLWhile, UnaryOpType)
from .pfl_reg import STD_REGISTRY, StdRegistryItem

_ALL_SUPPORTED_AST_TYPES = {
    ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Name,
    ast.Constant, ast.Subscript, ast.Attribute, ast.List, ast.Dict, ast.Assign,
    ast.AugAssign, ast.If, ast.Expr, ast.IfExp, ast.For, ast.While,
    ast.AnnAssign
}

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


class PFLAstParseError(Exception):

    def __init__(self, msg: str, node: ast.AST):
        super().__init__(msg)
        self.node = node


class PFLMetaEvalError(Exception):

    def __init__(self, msg: str, node: PFLAstNodeBase):
        super().__init__(msg)
        self.node = node


def _parse_expr_to_df_ast(expr: ast.expr, scope: dict[str,
                                                      PFLExprInfo]) -> PFLExpr:
    try:
        if isinstance(expr, ast.Name):
            if expr.id not in scope:
                raise PFLAstParseError(f"undefined name {expr.id}", expr)
            st = scope[expr.id]
            res = PFLName(PFLASTType.NAME,
                          id=expr.id,
                          st=dataclasses.replace(st))
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
            res = PFLSubscript(PFLASTType.SUBSCRIPT,
                               value=value,
                               slice=slice,
                               is_store=is_store)
        elif isinstance(expr, ast.List):
            elts = [_parse_expr_to_df_ast(elt, scope) for elt in expr.elts]
            res = PFLArray(PFLASTType.ARRAY, elts=elts)
        elif isinstance(expr, ast.Dict):
            keys = [
                _parse_expr_to_df_ast(key, scope) if key is not None else None
                for key in expr.keys
            ]
            values = [
                _parse_expr_to_df_ast(value, scope) for value in expr.values
            ]
            res = PFLDict(PFLASTType.DICT, keys=keys, values=values)
        elif isinstance(expr, ast.BoolOp):
            op = BoolOpType.AND if isinstance(expr.op,
                                              ast.And) else BoolOpType.OR
            values = [
                _parse_expr_to_df_ast(value, scope) for value in expr.values
            ]
            res = PFLBoolOp(PFLASTType.BOOL_OP,
                            op=op,
                            left=values[0],
                            right=values[1])
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
            res = PFLCompare(PFLASTType.COMPARISON,
                             op=op,
                             left=left,
                             right=right)
        elif isinstance(expr, ast.Call):
            func = _parse_expr_to_df_ast(expr.func, scope)
            args = [_parse_expr_to_df_ast(arg, scope) for arg in expr.args]
            res = PFLCall(PFLASTType.CALL, func=func, args=args)
        elif isinstance(expr, ast.IfExp):
            res = PFLIfExp(PFLASTType.IF_EXP,
                           test=_parse_expr_to_df_ast(expr.test, scope),
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


def _parse_block_to_df_ast(body: list[ast.stmt],
                           scope: dict[str, PFLExprInfo]) -> PFLFunc:
    block = PFLFunc(PFLASTType.BLOCK, [], [])
    for stmt in body:
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
                    block.body.append(
                        PFLAssign(PFLASTType.ASSIGN,
                                  target=target,
                                  value=value))
                else:
                    assert value is not None
                    ann_str = ast.unparse(stmt.annotation)
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
                    target.st = PFLExprInfo.from_annotype(
                        parse_type_may_optional_undefined(ann_res),
                        is_type=False)
                    block.body.append(
                        PFLAnnAssign(PFLASTType.ANN_ASSIGN,
                                     target=target,
                                     annotation=ast.unparse(stmt.annotation),
                                     value=value))
            elif isinstance(stmt, ast.AugAssign):
                target = _parse_expr_to_df_ast(stmt.target, scope)
                op = _AST_BINOP_TO_PFL_BINOP[type(stmt.op)]
                value = _parse_expr_to_df_ast(stmt.value, scope)
                block.body.append(
                    PFLAugAssign(PFLASTType.AUG_ASSIGN,
                                 target=target,
                                 op=op,
                                 value=value))
                if isinstance(target, PFLName):
                    scope[target.id] = target.st
            elif isinstance(stmt, ast.If):
                # TODO currently variable created in if scope won't leaked to parent scope.
                private_scope = scope.copy()
                test = _parse_expr_to_df_ast(stmt.test, private_scope)
                ifbody = _parse_block_to_df_ast(stmt.body, private_scope)
                orelse = _parse_block_to_df_ast(stmt.orelse, private_scope)
                block.body.append(
                    PFLIf(PFLASTType.IF,
                          test=test,
                          body=ifbody.body,
                          orelse=orelse.body))
            elif isinstance(stmt, ast.For):
                # TODO currently variable created in if scope won't leaked to parent scope.
                private_scope = scope.copy()
                value = _parse_expr_to_df_ast(stmt.iter, private_scope)
                tgt = stmt.target
                assert isinstance(tgt, ast.Name)
                is_new_var = False
                if isinstance(tgt, ast.Name):
                    if tgt.id not in private_scope:
                        is_new_var = True
                    else:
                        value.st.check_convertable(private_scope[tgt.id],
                                                   "assign value")
                    private_scope[tgt.id] = value.st
                target = _parse_expr_to_df_ast(stmt.target, private_scope)
                forbody = _parse_block_to_df_ast(stmt.body, private_scope)
                block.body.append(
                    PFLFor(PFLASTType.FOR,
                           target=target,
                           iter=value,
                           body=forbody.body))
            elif isinstance(stmt, ast.While):
                private_scope = scope.copy()
                test = _parse_expr_to_df_ast(stmt.test, private_scope)
                forbody = _parse_block_to_df_ast(stmt.body, private_scope)
                block.body.append(
                    PFLWhile(PFLASTType.WHILE, test=test, body=forbody.body))
            elif isinstance(stmt, ast.Expr):
                block.body.append(
                    PFLExprStmt(PFLASTType.EXPR_STMT,
                                value=_parse_expr_to_df_ast(stmt.value,
                                                            scope)))
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
                cur_node = ast.Name(id=mapped_name_parts[-1],
                                    ctx=node.ctx,
                                    lineno=node.lineno,
                                    col_offset=node.col_offset)
                for part in mapped_name_parts[::-1][1:]:
                    cur_node = ast.Attribute(value=cur_node,
                                             attr=part,
                                             ctx=node.ctx,
                                             lineno=node.lineno,
                                             col_offset=node.col_offset)
                return cur_node
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        return self._visit_Attribute_or_name(node)

    def visit_Attribute(self, node: ast.Attribute):
        return self._visit_Attribute_or_name(node)


def _get_init_scope():
    init_scope: dict[str, PFLExprInfo] = {
        "len":
        PFLExprInfo.from_signature(
            inspect.Signature([param_fn("x", list[Any])],
                              return_annotation=int)),
        "print":
        PFLExprInfo.from_signature(
            inspect.Signature([varparam_fn("x", Any)],
                              return_annotation=None)),
        "int":
        PFLExprInfo.from_signature(
            inspect.Signature([param_fn("x", Any)], return_annotation=int)),
        "float":
        PFLExprInfo.from_signature(
            inspect.Signature([param_fn("x", Any)], return_annotation=float)),
        "str":
        PFLExprInfo.from_signature(
            inspect.Signature([param_fn("x", Any)], return_annotation=str)),
        "bool":
        PFLExprInfo.from_signature(
            inspect.Signature([param_fn("x", Any)], return_annotation=bool)),
        "range":
        PFLExprInfo.from_signature(
            inspect.Signature([
                param_fn("start", int),
                param_fn("stop", int, None),
                param_fn("step", int, None)
            ],
                              return_annotation=range)),
    }
    return init_scope


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


def parse_func_to_df_ast(func: Callable,
                         scope: Optional[dict[str, PFLExprInfo]] = None,
                         backend: str = "js") -> tuple[PFLFunc, str]:
    if isinstance(func, staticmethod):
        func = func.__func__
    func_code_lines, _ = inspect.getsourcelines(func)
    func_code_lines = [line.rstrip() for line in func_code_lines]
    func_code_lines = clean_source_code(func_code_lines)
    code = "\n".join(func_code_lines)
    code = remove_common_indent_from_code(code)
    tree = ast.parse(code)
    tree = ast.fix_missing_locations(
        RewriteSTLName(func.__globals__, backend=backend).visit(tree))
    # find funcdef
    body = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            body = node.body
    assert body is not None
    args, _ = parse_annotated_function(func)
    anno_eval_globals = func.__globals__.copy()
    with enter_parse_context(
            PFLParseContext(func_code_lines, anno_eval_globals)) as ctx:
        init_scope = _get_init_scope()
        for k, v in STD_REGISTRY.global_dict.items():
            if v.backend == backend:
                init_scope[v.mapped_name] = PFLExprInfo.from_annotype(
                    parse_type_may_optional_undefined(v.dcls), is_type=True)
                anno_eval_globals[v.mapped_name] = v.dcls
        if scope is None:
            scope = init_scope.copy()
        else:
            scope = {**init_scope, **scope}
        dffunc_args: list[PFLStaticVar] = []
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
                    item = STD_REGISTRY.get_item_by_dcls(arg_type, backend)
                    if item is None:
                        raise NotImplementedError(
                            f"can't find your type {arg_type} in std library. you must implement it."
                        )
                    arg_type = item.dcls
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
                                    "temp",
                                    get_qualname_of_type(dcls_child)
                                ]).uid_encoded
                                STD_REGISTRY.global_dict[
                                    temp_mapped_name] = StdRegistryItem(
                                        dcls_child,
                                        temp_mapped_name,
                                        backend=backend,
                                        is_temp=True)

            anno_type = parse_type_may_optional_undefined(arg_type)
            st = PFLStaticVar.from_annotype(anno_type, is_type=False)
            st.name = param.name
            scope[param.name] = st
            dffunc_args.append(st)
        try:
            block = _parse_block_to_df_ast(body, scope)
            block.args = dffunc_args
        except:
            if ctx.error_node is not None:
                error_line = get_parse_context_checked(
                ).format_error_from_lines_node(ctx.error_node)
                print(error_line)
            raise
    return block, code


def parse_expr_to_df_ast(
        expr_str: str,
        var_scope: Optional[dict[str, Any]] = None,
        backend: str = "js") -> tuple[PFLExpr, dict[str, Any]]:
    expr_str_lines = expr_str.split("\n")
    tree = ast.parse(expr_str, mode="eval")
    if var_scope is not None:
        tree = ast.fix_missing_locations(
            RewriteSTLName(var_scope, backend=backend).visit(tree))
    assert isinstance(tree, ast.Expression)
    tree_expr = tree.body
    # find funcdef
    with enter_parse_context(PFLParseContext(expr_str_lines, {})) as ctx:
        init_scope = _get_init_scope()
        for k, v in STD_REGISTRY.global_dict.items():
            init_scope[v.mapped_name] = PFLExprInfo.from_annotype(
                parse_type_may_optional_undefined(v.dcls), is_type=True)
        if var_scope is None:
            scope = init_scope.copy()
        else:
            scope = init_scope.copy()
            for k, v in var_scope.items():
                scope[k] = PFLExprInfo.from_annotype(
                    parse_type_may_optional_undefined(type(v)), is_type=False)
        try:
            res = _parse_expr_to_df_ast(tree_expr, scope)
        except:
            if ctx.error_node is not None:
                error_line = get_parse_context_checked(
                ).format_error_from_lines_node(ctx.error_node)
                print(error_line)
            raise
    return res, scope


def _clear_consteval_result(node: PFLAstNodeBase):
    for n in walk(node):
        if isinstance(n, PFLExpr):
            n.st.metadata = undefined


def _consteval_expr(expr_node: PFLExpr, scope: dict[str, Any]):
    # perform const fold and meta inference, result is stored in metadata in each static type.
    # WARNING: inplace operation
    if isinstance(expr_node, PFLName):
        assert expr_node.id in scope, f"undefined name {expr_node.id}"
        value = scope[expr_node.id]
        expr_node.st.metadata = value
        return True
    else:
        child_nodes = iter_child_nodes(expr_node)
        all_success: list[bool] = []
        for n in child_nodes:
            assert isinstance(n, PFLExpr), f"expect PFLExpr, but got {type(n)}"
            success = _consteval_expr(n, scope)
            all_success.append(success)
        if not all(all_success):
            return False
        return expr_node.consteval()


def consteval_expr(expr_node: PFLExpr,
                   scope: Optional[dict[str, Any]] = None,
                   backend: str = "js"):
    _clear_consteval_result(expr_node)
    init_scope = scope
    if init_scope is None:
        init_scope = {}
        for k, v in STD_REGISTRY.global_dict.items():
            if v.backend == backend:
                init_scope[v.mapped_name] = v.dcls

    _consteval_expr(expr_node, init_scope)
    return expr_node.st.metadata


def _metaeval_expr(expr_node: PFLExpr, scope: dict[str, Any]):
    # perform const fold and meta inference, result is stored in metadata in each static type.
    # WARNING: inplace operation
    if isinstance(expr_node, PFLName):
        if expr_node.id not in scope:
            return False
        value = scope[expr_node.id]
        expr_node.st.metadata = value
        return True
    else:
        child_nodes = iter_child_nodes(expr_node)
        all_success: list[bool] = []
        for n in child_nodes:
            if isinstance(n, PFLExpr):
                success = _metaeval_expr(n, scope)
                all_success.append(success)
        if all_success and not any(all_success):
            return False
        return expr_node.metaeval()


def _metaeval_total_tree(body: list[PFLAstStmt], scope: dict[str, Any]):
    for stmt in body:
        if isinstance(stmt, (PFLAssign, PFLAnnAssign)):
            # TODO add dataclass level type meta eval support
            target_metadata = undefined
            scope_metadata = undefined
            if isinstance(stmt, PFLAnnAssign):
                metadata_from_anno = stmt.target.st.get_metadata_from_anno()
                if metadata_from_anno is not None:
                    target_metadata = metadata_from_anno
                    stmt.target.st.metadata = target_metadata
            if isinstance(target_metadata,
                          Undefined) and stmt.value is not None:
                _metaeval_expr(stmt.value, scope)
                target_metadata = stmt.value.st.metadata_checked
                stmt.target.st.metadata = target_metadata
            if isinstance(stmt.target, PFLName) and stmt.target.id in scope:
                scope_metadata = scope[stmt.target.id]

            if not isinstance(scope_metadata, Undefined) and not isinstance(
                    target_metadata, Undefined):
                # convertable check is already done in parse.
                # perform meta assign check if exists
                assert isinstance(stmt.target, PFLName)
                if stmt.target.st.type == PFLExprType.DATACLASS_OBJECT:
                    dcls = stmt.target.st.get_origin_type_checked()
                    if hasattr(dcls, PFL_FUNC_META_ATTR):
                        func_meta: PFLFuncMeta = getattr(
                            dcls, PFL_FUNC_META_ATTR)
                        if func_meta.meta_assign_check is not None:
                            new_meta_val = func_meta.meta_assign_check(
                                scope_metadata, target_metadata)
                            if not isinstance(new_meta_val, Undefined):
                                new_meta = new_meta_val
                                stmt.target.st.metadata = new_meta
            if isinstance(stmt.target, PFLName):
                scope[stmt.target.id] = target_metadata
        elif isinstance(stmt, PFLAugAssign):
            _metaeval_expr(stmt.value, scope)
        elif isinstance(stmt, PFLIf):
            private_scope = scope.copy()
            _metaeval_total_tree(stmt.body, private_scope)
            _metaeval_total_tree(stmt.orelse, private_scope)
        elif isinstance(stmt, PFLFor):
            private_scope = scope.copy()
            _metaeval_expr(stmt.iter, private_scope)
            iter_st = stmt.iter.st
            tgt = stmt.target
            assert isinstance(tgt, PFLName)
            if iter_st.type == PFLExprType.ARRAY:
                stmt.target.st.metadata = iter_st.childs[0].metadata
                private_scope[tgt.id] = iter_st.childs[0].metadata
            # Range iter is always number (never constexpr), so no metadata here.
            _metaeval_total_tree(stmt.body, private_scope)
        elif isinstance(stmt, PFLWhile):
            private_scope = scope.copy()
            _metaeval_expr(stmt.test, private_scope)
            _metaeval_total_tree(stmt.body, private_scope)
        elif isinstance(stmt, PFLExprStmt):
            _metaeval_expr(stmt.value, scope)
        else:
            raise PFLMetaEvalError(f"not support {type(stmt)}", stmt)


def metaeval_total_tree(func_node: PFLFunc,
                        scope: dict[str, Any],
                        backend: str = "js"):
    # perform const fold and meta inference, result is stored in metadata in each static type.
    # WARNING: inplace operation
    name_to_args = {n.name: n for n in func_node.args}
    for k, v in scope.items():
        assert k in name_to_args
        # TODO add type check?
    _clear_consteval_result(func_node)
    init_scope = scope
    if init_scope is None:
        init_scope = {}
    for k, v in STD_REGISTRY.global_dict.items():
        if v.backend == backend:
            init_scope[v.mapped_name] = v.dcls
    _metaeval_total_tree(func_node.body, scope)


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


def pfl_ast_dump(node: PFLAstNodeBase):
    return _ast_as_dict_for_dump(node)


_PFL_UNPARSE_UNARY_TYPE_TO_OP = {}

_PFL_UNPARSE_OP_TYPE_TO_OP = {
    UnaryOpType.INVERT: "~",
    UnaryOpType.NOT: "not",
    UnaryOpType.UADD: "+",
    UnaryOpType.USUB: "-",
    CompareType.EQUAL: "==",
    CompareType.NOT_EQUAL: "!=",
    CompareType.LESS: "<",
    CompareType.LESS_EQUAL: "<=",
    CompareType.GREATER: ">",
    CompareType.GREATER_EQUAL: ">=",
    CompareType.IN: "in",
    CompareType.NOT_IN: "not in",
    BinOpType.ADD: "+",
    BinOpType.SUB: "-",
    BinOpType.MULT: "*",
    BinOpType.DIV: "/",
    BinOpType.MOD: "%",
    BinOpType.POW: "**",
    BinOpType.LSHIFT: "<<",
    BinOpType.RSHIFT: ">>",
    BinOpType.BIT_OR: "|",
    BinOpType.BIT_XOR: "&",
    BinOpType.BIT_AND: "&",
}


def unparse_expr(expr: PFLExpr) -> str:
    """
    Unparse a PFLExpr to a string representation.
    """
    if isinstance(expr, PFLName):
        return expr.id
    elif isinstance(expr, PFLAttribute):
        return f"{unparse_expr(expr.value)}.{expr.attr}"
    elif isinstance(expr, PFLConstant):
        return repr(expr.value)
    elif isinstance(expr, PFLSubscript):
        slice_str = unparse_expr(expr.slice)
        if isinstance(expr.value, PFLName):
            value_str = expr.value.id
        else:
            value_str = unparse_expr(expr.value)
        return f"{value_str}[{slice_str}]"
    elif isinstance(expr, PFLArray):
        return "[" + ", ".join(unparse_expr(elt) for elt in expr.elts) + "]"
    elif isinstance(expr, PFLDict):
        strs = []
        for k, v in zip(expr.keys, expr.values):
            if k is None:
                strs.append(f"**{unparse_expr(v)}")
            else:
                strs.append(f"{unparse_expr(k)}: {unparse_expr(v)}")
        return "{" + ", ".join(strs) + "}"
    elif isinstance(expr, PFLBoolOp):
        if expr.op == BoolOpType.AND:
            op = "and"
        else:
            op = "or"
        return f"({unparse_expr(expr.left)} {op} {unparse_expr(expr.right)})"
    elif isinstance(expr, PFLBinOp):
        return f"({unparse_expr(expr.left)} {_PFL_UNPARSE_OP_TYPE_TO_OP[expr.op]} {unparse_expr(expr.right)})"
    elif isinstance(expr, PFLUnaryOp):
        return f"{_PFL_UNPARSE_OP_TYPE_TO_OP[expr.op]}{unparse_expr(expr.operand)}"
    elif isinstance(expr, PFLCompare):
        return f"({unparse_expr(expr.left)} {_PFL_UNPARSE_OP_TYPE_TO_OP[expr.op]} {unparse_expr(expr.right)})"
    elif isinstance(expr, PFLCall):
        args_str = ", ".join(unparse_expr(arg) for arg in expr.args)
        return f"{unparse_expr(expr.func)}({args_str})"
    elif isinstance(expr, PFLIfExp):
        return f"({unparse_expr(expr.body)} if {unparse_expr(expr.test)} else {unparse_expr(expr.orelse)})"
    else:
        raise NotImplementedError(f"Unrecognized PFLExpr type: {type(expr)}")
