import asyncio
from collections.abc import Sequence
from contextlib import AbstractContextManager, ExitStack
import contextlib
import enum
import inspect
import traceback
from typing import Any, Callable, Coroutine, ForwardRef, Optional, Type, Union, cast
from typing_extensions import TypeAlias
import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.annolib import (Undefined, is_undefined, undefined)
from tensorpc.core.event_emitter.single import SingleAsyncEventEmitter
from tensorpc.core.moduleid import get_module_id_of_type

from .core import (BACKEND_CONFIG_REGISTRY, PFLErrorFormatContext, PFLParseCache, StaticEvalConfig, PFLMetaInferResult, PFLParseConfig,
                   PFLParseContext, PFLExprInfo, PFLExprType,
                   enter_parse_context, get_parse_context, get_parse_context_checked)
from .pfl_ast import (BinOpType, BoolOpType, CompareType, PFLAnnAssign, PFLArg,
                      PFLArray, PFLAssign, PFLAstNodeBase, PFLAstStmt,
                      PFLASTType, PFLAttribute, PFLAugAssign, PFLBinOp,
                      PFLBoolOp, PFLBreak, PFLCall, PFLCompare, PFLConstant, PFLContinue, PFLDict,
                      PFLExpr, PFLExprStmt, PFLFor, PFLFunc, PFLIf, PFLIfExp, PFLModule,
                      PFLName, PFLReturn, PFLSlice, PFLStaticVar, PFLSubscript, PFLTreeNodeFinder, PFLTuple,
                      PFLUnaryOp, PFLWhile, UnaryOpType, iter_child_nodes, unparse_pfl_ast, unparse_pfl_expr, walk,
                      PFLAstParseError, PFLEvalError)

from .pfl_reg import  STD_REGISTRY, StdRegistryItem
from .parser import PFLLibrary

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
        try:
            return expr_node.consteval()
        except BaseException as e:
            traceback.print_exc()
            raise PFLEvalError(f"Eval node error {e}", expr_node) from e


def consteval_expr(expr_node: PFLExpr,
                   scope: Optional[dict[str, Any]] = None,
                   backend: str = "js"):
    _clear_consteval_result(expr_node)
    init_scope = scope
    if init_scope is None:
        init_scope = {}
        for k, v in STD_REGISTRY.global_dict.items():
            if v.backend is None or v.backend == backend:
                init_scope[v.mapped_name] = v.dcls

    _consteval_expr(expr_node, init_scope)
    return expr_node.st.metadata


class PFLStaticEvaluator:
    def __init__(self, library: PFLLibrary, cfg: StaticEvalConfig, assign_check: Optional[Callable[[PFLExprInfo, PFLExprInfo],
                                         Optional[PFLMetaInferResult]]] = None):
        self.cfg = cfg
        self._library = library
        self._assign_check = assign_check


    @classmethod 
    def meta_evaulator(cls, library: PFLLibrary, prefer_meta_eval: bool = True, assign_check: Optional[Callable[[PFLExprInfo, PFLExprInfo],
                                         Optional[PFLMetaInferResult]]] = None):
        cfg = StaticEvalConfig(prefer_meta_eval=prefer_meta_eval, allow_partial=True)
        return cls(library, cfg, assign_check=assign_check)

    def _eval_expr(self, expr_node: PFLExpr, scope: dict[str, PFLExprInfo]):
        # perform const fold and meta inference, result is stored in metadata in each static type.
        # WARNING: inplace operation
        if isinstance(expr_node, PFLName):
            if expr_node.id not in scope:
                if self.cfg.allow_partial:
                    return False
                else:
                    raise PFLEvalError(f"{expr_node.id} not found in current scope.", expr_node)
            value = scope[expr_node.id]
            expr_node.st.metadata = value.metadata
            return True            
        else:
            child_nodes = iter_child_nodes(expr_node)
            all_success: list[bool] = []
            for n in child_nodes:
                if isinstance(n, PFLExpr):
                    success = self._eval_expr(n, scope)
                    all_success.append(success)
            if all_success and not any(all_success):
                if self.cfg.allow_partial:
                    return False
                else:
                    expr_node_str = unparse_pfl_expr(expr_node)
                    raise PFLEvalError(f"Some child of Expr {expr_node_str} eval failed.", expr_node)
            try:
                if isinstance(expr_node, PFLCall):
                    func_st = expr_node.st
                    if func_st.compiled_uid is not None:
                        all_compiled = self._library.all_compiled
                        assert func_st.compiled_uid in all_compiled
                        compiled_node = all_compiled[func_st.compiled_uid]
                        func_arg_sts = compiled_node.st.childs
                        last_is_vaargs = func_arg_sts[-1].is_vaargs
                        matched = expr_node._match_arg_sts_to_sig(func_arg_sts, last_is_vaargs)
                        func_scope = {}
                        for func_arg_st, args in matched:
                            # TODO validate vaargs type
                            if args:
                                # may use default here.
                                func_scope[func_arg_st.arg_name] = args[0]
                        self._eval_total_tree_node(compiled_node, func_scope)
                        if compiled_node.ret_st is not None:
                            expr_node.st.metadata = compiled_node.ret_st.metadata
                        return True
                if self.cfg.prefer_meta_eval:
                    res = expr_node.metaeval()
                else:
                    res = expr_node.consteval()
            except BaseException as e:
                traceback.print_exc()
                raise PFLEvalError(f"eval error {e}", expr_node) from e
            return res 

    def _get_init_scope(self, func_node: PFLFunc, scope: dict[str, Any]):
        ctx = get_parse_context_checked()
        name_to_args = {n.arg: n for n in func_node.args}
        init_scope: dict[str, PFLExprInfo] = {}
        for k, v in scope.items():
            if isinstance(v, PFLExprInfo):
                v = v.metadata
            assert k in name_to_args
            info = dataclasses.replace(name_to_args[k].st)
            info.metadata = v
            arg_st = name_to_args[k].st
            if arg_st.has_metadata() and self._assign_check is not None:
                # perform assign check
                res = self._assign_check(arg_st, info)
                if res is not None:
                    info.metadata = res.data
                    name_to_args[k].st.metadata = res.data
            else:
                name_to_args[k].st.metadata = v
            init_scope[k] = info
        for k, v in STD_REGISTRY.global_dict.items():
            if v.backend is None or v.backend == ctx._backend:
                init_var = ctx.cache.cached_parse_std_item(v)
                init_var.metadata = v.dcls
                init_scope[v.mapped_name] = init_var
        return init_scope

    def _eval_total_tree_node(self, func_node: PFLFunc,
                            scope: dict[str, Any],
                            parse_cfg: Optional[PFLParseConfig] = None):
        # perform const fold and meta inference, result is stored in metadata in each static type.
        # WARNING: inplace operation
        backend = self._library.backend
        if parse_cfg is None:
            assert backend in BACKEND_CONFIG_REGISTRY, "you must register backend config first if parse_cfg isn't provided."
            parse_cfg = BACKEND_CONFIG_REGISTRY[backend]
        _clear_consteval_result(func_node)
        code_for_error = self._library.get_module_by_func(func_node.uid).compile_info.code
        lines = []
        if code_for_error is not None:
            lines = code_for_error.split("\n")
        outer_ctx = get_parse_context() 
        all_compiled = self._library.all_compiled
        if outer_ctx is not None:
            assert all_compiled is None
            parse_ctx = PFLParseContext.from_outer_ctx(outer_ctx, lines, {})
        else:
            parse_ctx = PFLParseContext(lines, {}, backend, cfg=parse_cfg, eval_cfg=self.cfg)
        with enter_parse_context(parse_ctx) as ctx:
            init_scope = self._get_init_scope(func_node, scope)
            try:
                self._eval_total_tree(func_node.body, init_scope)
            except PFLEvalError as e:
                error_line = ctx.format_error_from_lines_node(e.node)
                if error_line:
                    print(error_line)
                raise e

    def eval_total_tree(self, func: Union[str, Callable],
                            scope: dict[str, Any],
                            parse_cfg: Optional[PFLParseConfig] = None):
        func_node = self._library.get_compiled_unit(func)
        # perform const fold and meta inference, result is stored in metadata in each static type.
        # WARNING: inplace operation
        backend = self._library.backend
        if parse_cfg is None:
            assert backend in BACKEND_CONFIG_REGISTRY, "you must register backend config first if parse_cfg isn't provided."
            parse_cfg = BACKEND_CONFIG_REGISTRY[backend]
        _clear_consteval_result(func_node)
        code_for_error = self._library.get_module_by_func(func).compile_info.code
        lines = []
        if code_for_error is not None:
            lines = code_for_error.split("\n")
        outer_ctx = get_parse_context() 
        all_compiled = self._library.all_compiled
        if outer_ctx is not None:
            assert all_compiled is None
            parse_ctx = PFLParseContext.from_outer_ctx(outer_ctx, lines, {})
        else:
            parse_ctx = PFLParseContext(lines, {}, backend, cfg=parse_cfg, eval_cfg=self.cfg)
        with enter_parse_context(parse_ctx) as ctx:
            init_scope = self._get_init_scope(func_node, scope)
            try:
                self._eval_total_tree(func_node.body, init_scope)
            except PFLEvalError as e:
                error_line = ctx.format_error_from_lines_node(e.node)
                if error_line:
                    print(error_line)
                raise e

    def _eval_total_tree(self, body: list[PFLAstStmt], scope: dict[str, PFLExprInfo]):
        for stmt in body:
            try:
                if isinstance(stmt, (PFLAssign, PFLAnnAssign)):
                    # TODO add dataclass level type meta eval support
                    target_metadata = undefined
                    if isinstance(stmt, PFLAnnAssign):
                        metadata_from_anno = stmt.target.st.get_eval_metadata_from_anno()
                        if metadata_from_anno is not None:
                            target_metadata = metadata_from_anno
                            stmt.target.st.metadata = target_metadata
                    if isinstance(target_metadata,
                                Undefined) and stmt.value is not None:
                        # print(stmt.value)
                        self._eval_expr(stmt.value, scope)
                        if stmt.value.st.has_metadata():
                            if isinstance(stmt.target, PFLTuple):
                                assert isinstance(stmt.value.st.metadata, tuple)
                                for i, elt in enumerate(stmt.target.elts):
                                    elt.st.metadata = stmt.value.st.metadata[i]
                                stmt.target.st.metadata = stmt.value.st.metadata
                            else:
                                target_metadata = stmt.value.st.metadata_checked
                                stmt.target.st.metadata = target_metadata
                    target_names: list[PFLName] = []
                    target_metadatas: list[Any] = []
                    if isinstance(stmt.target, PFLName):
                        target_names = [stmt.target]
                        target_metadatas = [stmt.target.st.metadata]
                    elif isinstance(stmt.target, PFLTuple):
                        # assert isinstance(target_metadata)
                        for elt in stmt.target.elts:
                            assert isinstance(elt, PFLName)
                            target_names.append(elt)
                            target_metadatas.append(elt.st.metadata)
                    for target_name, target_metadata in zip(target_names, target_metadatas):
                        scope_metadata = undefined
                        if target_name.id in scope:
                            scope_metadata = scope[target_name.id].metadata
                        # convertable check is already done in parse.
                        # perform meta assign check if exists
                        if not isinstance(scope_metadata, Undefined):
                            if self._assign_check is not None:
                                new_meta_val = self._assign_check(
                                    scope_metadata, target_metadata)
                                if new_meta_val is not None:
                                    new_meta = new_meta_val.data
                                    stmt.target.st.metadata = new_meta
                        if not isinstance(target_metadata, Undefined):
                            scope[target_name.id] = dataclasses.replace(target_name.st)
                elif isinstance(stmt, PFLAugAssign):
                    self._eval_expr(stmt.value, scope)
                    if isinstance(stmt.target, PFLName):
                        assert stmt.target.id in scope, \
                            f"undefined name {stmt.target.id} in scope {scope.keys()}"
                        target_st = scope[stmt.target.id]
                        stmt.target.st.metadata = target_st.metadata
                elif isinstance(stmt, PFLIf):
                    private_scope_if = scope.copy()
                    private_scope_else = scope.copy()

                    self._eval_total_tree(stmt.body, private_scope_if)
                    self._eval_total_tree(stmt.orelse, private_scope_else)
                    if get_parse_context_checked().cfg.allow_new_var_after_if:
                        # compare and merge scopes
                        # 1. get new variables in each scope
                        new_vars_if = set(private_scope_if.keys()) - set(scope.keys())
                        new_vars_else = set(private_scope_else.keys()) - set(scope.keys())
                        # 2. get common variables in both scopes, common vars must have same type.
                        common_vars = new_vars_if & new_vars_else
                        for common_var in common_vars:
                            var_in_if = private_scope_if[common_var]
                            var_in_else = private_scope_else[common_var]
                            # type is compared in parse, so we only need to check metadata.
                            assign_check = self._assign_check
                            new_info = var_in_if.try_merge_two_info(var_in_else)
                            if assign_check is not None:
                                new_meta_val = assign_check(var_in_if,
                                                                var_in_else)
                                new_info.metadata = new_meta_val.data if new_meta_val is not None else undefined
                            scope[common_var] = new_info

                elif isinstance(stmt, PFLFor):
                    private_scope = scope.copy()
                    self._eval_expr(stmt.iter, private_scope)
                    iter_st = stmt.iter.st
                    tgt = stmt.target
                    assert isinstance(tgt, PFLName)
                    if iter_st.type == PFLExprType.ARRAY:
                        stmt.target.st.metadata = iter_st.childs[0].metadata
                        private_scope[tgt.id] = dataclasses.replace(iter_st.childs[0])
                    # Range iter is always number (never constexpr), so no metadata here.
                    self._eval_total_tree(stmt.body, private_scope)
                elif isinstance(stmt, PFLWhile):
                    private_scope = scope.copy()
                    self._eval_expr(stmt.test, private_scope)
                    self._eval_total_tree(stmt.body, private_scope)
                elif isinstance(stmt, PFLExprStmt):
                    self._eval_expr(stmt.value, scope)
                elif isinstance(stmt, PFLReturn):
                    if stmt.value is not None:
                        self._eval_expr(stmt.value, scope)
                else:
                    raise PFLEvalError(f"not support {type(stmt)}", stmt)
            except PFLEvalError:
                raise
            except BaseException as e:
                raise PFLEvalError(f"Unknown error {e}", stmt) from e

class PFLRunnerResultType(enum.IntEnum):
    BREAK = 0
    CONTINUE = 1
    RETURN = 2

@dataclasses.dataclass
class PFLRunnerResult:
    type: PFLRunnerResultType
    data: Optional[Any] = None

@dataclasses.dataclass
class PFLCtrlBase:
    node: PFLAstNodeBase
    should_pause: bool = False
    enabled: bool = True

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class PFLCtrlFor(PFLCtrlBase):
    # which step should stop
    step: int 
    range: range
    stop_in_start: bool 

    def __post_init__(self):
        assert isinstance(self.node, PFLFor)
        assert self.node.iter.st.type == PFLExprType.RANGE, "only support base for"

class PFLAsyncRunnerStateType(enum.IntEnum):
    IDLE = 0
    RUNNING_ON_THE_FLY = 1
    DURING_RUNNING_TO = 2
    PAUSE = 3
    NEED_STOP = 4

@dataclasses.dataclass
class PFLBreakpointDesc:
    lineno: int 
    enabled: bool = True
    one_shot: bool = False

@dataclasses.dataclass
class PFLBreakpoint:
    node: PFLAstNodeBase
    scope: dict[str, Any]

@dataclasses.dataclass
class PFLAsyncRunnerState:
    type: PFLAsyncRunnerStateType
    cur_bkpt: Optional[PFLBreakpoint] = None
    cur_ctrl_points: dict[int, PFLCtrlBase] = dataclasses.field(default_factory=dict)
    

class PFLEvalStop(Exception):

    def __init__(self, msg: str, node: PFLAstNodeBase):
        super().__init__(msg)
        self.node = node

_CORO_NONE: TypeAlias = Union[Coroutine[None, None, None], None]

class PFLAsyncRunner:
    """A PFL runner that support breakpoints (via asyncio).
    other option is write a VM which is too complex.

    TODO: currently we only support all function defines in same file
    """
    def __init__(self, library: PFLLibrary):
        self._library = library
        # TODO temp data class support?
        self._parse_cache = PFLParseCache(library.backend)
        std_scope: dict[str, Any] = {}
        for k, v in STD_REGISTRY.global_dict.items():
            if v.backend is None or v.backend == library.backend:
                std_scope[v.mapped_name] = v.dcls
        self._std_scope = std_scope

        self._state = PFLAsyncRunnerState(PFLAsyncRunnerStateType.IDLE)
        self._event = asyncio.Event()

        self._breakpoints: dict[int, PFLBreakpointDesc] = {}

        self.event_enter_bkpt: SingleAsyncEventEmitter[PFLBreakpoint] = SingleAsyncEventEmitter()
        self.event_leave_bkpt: SingleAsyncEventEmitter[PFLBreakpoint] = SingleAsyncEventEmitter()
        self.event_new_ctrl_point: SingleAsyncEventEmitter[PFLCtrlBase] = SingleAsyncEventEmitter()
        self.event_delete_ctrl_point: SingleAsyncEventEmitter[PFLCtrlBase] = SingleAsyncEventEmitter()
        self.event_ctrl_point_change: SingleAsyncEventEmitter[PFLCtrlBase] = SingleAsyncEventEmitter()
        self.event_eval_stop: SingleAsyncEventEmitter[()] = SingleAsyncEventEmitter()


    async def _run_coro_none(self, fn: Callable[..., _CORO_NONE], *args) -> None:
        """Run a coroutine that returns None."""
        res = fn(*args)
        if inspect.iscoroutine(res):
            await res

    def clear_runtime_state(self):
        self._state = PFLAsyncRunnerState(PFLAsyncRunnerStateType.IDLE)
        self._event.clear()
        self._breakpoints.clear()

    def release_breakpoint(self, stop: bool = False):
        assert self._state.type == PFLAsyncRunnerStateType.PAUSE, \
            f"release_breakpoint called in state {self._state.type}, expected PAUSE."
        self._event.set()
        if stop:
            self._state.type = PFLAsyncRunnerStateType.NEED_STOP

    def add_breakpoint(self, lineno: int, enabled: bool = True, one_shot: bool = False):
        self._breakpoints[lineno] = PFLBreakpointDesc(lineno, enabled, one_shot)

    def remove_breakpoint(self, lineno: int):
        if lineno in self._breakpoints:
            self._breakpoints.pop(lineno)

    async def _enter_breakpoint(self, node: PFLAstNodeBase, scope: dict[str, Any]):
        self._state.cur_bkpt = PFLBreakpoint(node, scope)
        self._state.type = PFLAsyncRunnerStateType.PAUSE
        self._event.clear()
        # call user callback after event set to let user release this bkpt in callback.
        if not self.event_enter_bkpt.is_empty():
            await self.event_enter_bkpt.emit_async(self._state.cur_bkpt)
        await self._event.wait() 
        if not self.event_leave_bkpt.is_empty():
            await self.event_leave_bkpt.emit_async(self._state.cur_bkpt)
        self._state.cur_bkpt = None 
        if self._state.type == PFLAsyncRunnerStateType.NEED_STOP:
            self._state.type = PFLAsyncRunnerStateType.IDLE
            for cp in self._state.cur_ctrl_points.values():
                await self.event_delete_ctrl_point.emit_async(cp)
            self._state.cur_ctrl_points.clear()
            self._breakpoints.clear()
            raise PFLEvalStop("Eval Stop by user.", node)
        self._state.type = PFLAsyncRunnerStateType.RUNNING_ON_THE_FLY

    async def _may_pause_by_ctrl_points(self, node: PFLAstNodeBase, scope: dict[str, Any]):
        if not self._state.cur_ctrl_points:
            return False
        if all(cp.should_pause for cp in self._state.cur_ctrl_points.values()):
            await self._enter_breakpoint(node, scope)
            return True 
        return False

    async def _check_enter_breakpoint(self, node: PFLAstNodeBase, scope: dict[str, Any]):
        if node.source_loc[0] in self._breakpoints:
            bkpt_desc = self._breakpoints[node.source_loc[0]]
            if self._state.cur_ctrl_points:
                should_pause = all(cp.should_pause for cp in self._state.cur_ctrl_points.values())
            else:
                should_pause = True
            if should_pause:
                try:
                    return await self._enter_breakpoint(node, scope)
                finally:
                    if bkpt_desc.one_shot:
                        self._breakpoints.pop(node.source_loc[0])

    async def _get_subscript_target_slice(self, node: PFLSubscript, scope: dict[str, Any]):
        tgt = await self._run_expr(node.value, scope)
        if isinstance(node.slice, Sequence):
            slice_strs = [await self._run_expr(s, scope) for s in node.slice]
            slice_str = tuple(slice_strs)
        else:
            slice_str = await self._run_expr(node.slice, scope)
        return tgt, slice_str

    async def _run_expr(self, expr: PFLExpr, scope: dict[str, Any]) -> Any:
        if isinstance(expr, PFLName):
            if expr.st.compiled_uid is not None:
                return self._library.all_compiled[expr.st.compiled_uid]
            return scope[expr.id]
        elif isinstance(expr, PFLAttribute):
            if expr.st.compiled_uid is not None:
                return self._library.all_compiled[expr.st.compiled_uid]
            return getattr(await self._run_expr(expr.value, scope), expr.attr)
        elif isinstance(expr, PFLConstant):
            return expr.value
        elif isinstance(expr, PFLSlice):
            lo_str = None if is_undefined(expr.lo) else await self._run_expr(expr.lo, scope)
            hi_str = None if is_undefined(expr.hi) else await self._run_expr(expr.hi, scope)
            step_str = None if is_undefined(expr.step) else await self._run_expr(
                expr.step, scope)
            return slice(lo_str, hi_str, step_str)
        elif isinstance(expr, PFLSubscript):
            tgt, slice_str = await self._get_subscript_target_slice(expr, scope)
            return tgt[slice_str]
        elif isinstance(expr, PFLArray):
            return [await self._run_expr(elt, scope)
                                for elt in expr.elts]
        elif isinstance(expr, PFLTuple):
            return tuple([await self._run_expr(elt, scope)
                                for elt in expr.elts])
        elif isinstance(expr, PFLDict):
            res = {}
            for k, v in zip(expr.keys, expr.values):
                vv: Any = await self._run_expr(v, scope)
                if k is None:
                    res.update(vv)
                else:
                    kk = await self._run_expr(v, scope)
                    res[kk] = vv
            return res
        elif isinstance(expr, PFLBoolOp):
            left = await self._run_expr(expr.left, scope)
            right = await self._run_expr(expr.right, scope)
            if expr.op == BoolOpType.AND:
                res = left and right
            else:
                res = left or right
            return res
        elif isinstance(expr, (PFLBinOp, PFLCompare)):
            left = await self._run_expr(expr.left, scope)
            right = await self._run_expr(expr.right, scope)
            return expr.run(left, right)
        elif isinstance(expr, PFLUnaryOp):
            left = await self._run_expr(expr.operand, scope)
            return expr.run(left)
        elif isinstance(expr, PFLCall):
            func_st = expr.func.st
            if func_st.compiled_uid is not None:
                kwargs = {}
                # compiled pfl function don't have vaargs
                matched = expr._match_arg_sts_to_sig(func_st.childs, False)
                for arg_st, exprs in matched:
                    assert arg_st.arg_name is not None 
                    assert len(exprs) <= 1
                    for arg_expr in exprs:
                        arg_value = await self._run_expr(arg_expr, scope)
                        kwargs[arg_st.arg_name] = (arg_value)
                res = await self._run_func(func_st.compiled_uid, kwargs)
            else:
                func_val = await self._run_expr(expr.func, scope)
                assert inspect.isfunction(func_val) or inspect.ismethod(func_val), f"expect function, but got {type(func_val)}"
                arg_values = []
                # compiled pfl function don't have vaargs
                matched = expr._match_arg_sts_to_sig(func_st.childs, func_st.childs[-1].is_vaargs)
                for arg_st, exprs in matched:
                    for arg_expr in exprs:
                        arg_value = await self._run_expr(arg_expr, scope)
                        arg_values.append(arg_value)
                res = func_val(*arg_values) 
            return res 
        elif isinstance(expr, PFLIfExp):
            body = await self._run_expr(expr.body, scope)
            test = await self._run_expr(expr.test, scope)
            orelse = await self._run_expr(expr.orelse, scope)
            return body if test else orelse
        else:
            raise NotImplementedError(f"Unrecognized PFLExpr type: {type(expr)}")

    async def run_body(self, block_body: list[PFLAstStmt], scope: dict[str, Any]) -> Union[Any, PFLRunnerResult]:
        for stmt in block_body:
            if self._breakpoints:
                await self._check_enter_breakpoint(stmt, scope)
            try:
                if isinstance(stmt, PFLExpr):
                    await self._run_expr(stmt, scope)
                elif isinstance(stmt, (PFLAssign, PFLAnnAssign)):
                    if stmt.value is not None:
                        value = await self._run_expr(stmt.value, scope)
                        # when stmt.target is attr or subscript, we need to evaluate more deeper thing.
                        if isinstance(stmt.target, (PFLAttribute, PFLSubscript)):
                            assert not is_undefined(stmt.target.is_store) and stmt.target.is_store == True
                            deep_val = await self._run_expr(stmt.target.value, scope)
                            if isinstance(stmt.target, (PFLAttribute)):
                                setattr(deep_val, stmt.target.attr, value)
                            else:
                                tgt, slice_str = await self._get_subscript_target_slice(stmt.target, scope)
                                tgt[slice_str] = value
                        elif isinstance(stmt.target, PFLTuple):
                            for i, elt in enumerate(stmt.target.elts):
                                assert isinstance(elt, PFLName)
                                scope[elt.id] = value[i]
                        else:
                            assert isinstance(stmt.target, PFLName)
                            scope[stmt.target.id] = value
                elif isinstance(stmt, (PFLIf)):
                    testAndBodyArr = stmt.get_flatten_test_body()
                    for i in range(len(testAndBodyArr)):
                        test, body = testAndBodyArr[i]
                        if test is not None:
                            test_val = await self._run_expr(test, scope)
                        else:
                            test_val = True 
                        if test_val:
                            private_scope = scope.copy()
                            await self.run_body(body, private_scope)
                            if not is_undefined(stmt._new):
                                for v in stmt._new:
                                    scope[v] = private_scope[v]
                            for k in scope.keys():
                                scope[k] = private_scope[k]
                            break
                elif isinstance(stmt, PFLAugAssign):
                    value = await self._run_expr(stmt.value, scope)
                    if isinstance(stmt.target, (PFLAttribute, PFLSubscript)):
                        assert not is_undefined(stmt.target.is_store) and stmt.target.is_store == True
                        deep_val = await self._run_expr(stmt.target.value, scope)
                        if isinstance(stmt.target, (PFLAttribute)):
                            val = getattr(deep_val, stmt.target.attr)
                            new_val = stmt.run(val, value)
                            setattr(deep_val, stmt.target.attr, new_val)
                        else:
                            tgt, slice_str = await self._get_subscript_target_slice(stmt.target, scope)
                            new_val = stmt.run(tgt[slice_str], value)
                            tgt[slice_str] = new_val
                    else:
                        assert isinstance(stmt.target, PFLName)
                        new_val = stmt.run(scope[stmt.target.id], value)
                        scope[stmt.target.id] = new_val
                elif isinstance(stmt, PFLFor):
                    iter_obj = await self._run_expr(stmt.iter, scope)
                    tgt = stmt.target
                    assert isinstance(tgt, PFLName)
                    if isinstance(iter_obj, range):
                        stmt_id = id(stmt)
                        private_scope = scope.copy()
                        ctrl = PFLCtrlFor(stmt, enabled=True, step=iter_obj.start, range=iter_obj, stop_in_start=False)
                        self._state.cur_ctrl_points[stmt_id] = ctrl
                        if not self.event_new_ctrl_point.is_empty():
                            await self.event_new_ctrl_point.emit_async(ctrl)
                        for i in iter_obj:
                            if ctrl.enabled:
                                # print(i, ctrl.step, iter_obj)
                                if ctrl.step < i:
                                    ctrl.step = i
                                    await self.event_ctrl_point_change.emit_async(ctrl)
                                # print("AFTER", i, ctrl.step)
                                ctrl.should_pause = ctrl.step == i
                                private_scope[tgt.id] = i
                                result = await self.run_body(stmt.body, private_scope)
                                    # ctrl.should_pause = False
                            else:
                                private_scope[tgt.id] = i
                                result = await self.run_body(stmt.body, private_scope)
                            if isinstance(result, PFLRunnerResult):
                                if result.type == PFLRunnerResultType.BREAK:
                                    break 
                                elif result.type == PFLRunnerResultType.RETURN:
                                    return result
                                # dont need to handle continue here.
                        for k in scope.keys():
                            scope[k] = private_scope[k]
                        self._state.cur_ctrl_points.pop(stmt_id)
                        if not self.event_delete_ctrl_point.is_empty():
                            await self.event_delete_ctrl_point.emit_async(ctrl)
                        
                    elif isinstance(iter_obj, list):
                        private_scope = scope.copy()
                        for obj in iter_obj:
                            private_scope[tgt.id] = obj
                            result = await self.run_body(stmt.body, private_scope)
                            if isinstance(result, PFLRunnerResult):
                                if result.type == PFLRunnerResultType.BREAK:
                                    break 
                                elif result.type == PFLRunnerResultType.RETURN:
                                    return result
                                # dont need to handle continue here.
                        for k in scope.keys():
                            scope[k] = private_scope[k]
                    else:
                        raise NotImplementedError
                elif isinstance(stmt, PFLWhile):
                    private_scope = scope.copy()
                    while True:
                        test_obj = await self._run_expr(stmt.test, private_scope)
                        if not test_obj:
                            break 
                        result = await self.run_body(stmt.body, private_scope)
                        if isinstance(result, PFLRunnerResult):
                            if result.type == PFLRunnerResultType.BREAK:
                                break 
                            elif result.type == PFLRunnerResultType.RETURN:
                                return result
                            # dont need to handle continue here.
                    for k in scope.keys():
                        scope[k] = private_scope[k]

                elif isinstance(stmt, PFLExprStmt):
                    await self._run_expr(stmt.value, scope)
                elif isinstance(stmt, PFLReturn):
                    if stmt.value is not None:
                        value = await self._run_expr(stmt.value, scope)
                        return PFLRunnerResult(PFLRunnerResultType.RETURN, value)
                    else:
                        return
                elif isinstance(stmt, PFLBreak):
                    return PFLRunnerResult(PFLRunnerResultType.BREAK)
                elif isinstance(stmt, PFLContinue):
                    return PFLRunnerResult(PFLRunnerResultType.CONTINUE)
                elif isinstance(stmt, PFLFunc):
                    # no-op here because we don't support direct func def here.
                    return 
                else:
                    raise NotImplementedError(f"Unrecognized PFLAstNodeBase type: {type(stmt)}")
            except PFLEvalStop:
                raise
            except PFLEvalError:
                raise
            except BaseException as e:
                raise PFLEvalError(f"stmt eval error.", stmt) from e

    async def _run_func(self, func_uid: str, scope: dict[str, Any]):
        func_node = self._library.all_compiled[func_uid]
        module = self._library.get_module_by_func(func_uid)
        error_ctx = PFLErrorFormatContext(module.compile_info.code.split("\n"))
        try:
            res = await self.run_body(func_node.body, {**scope, **self._std_scope})
        except PFLEvalStop:
            raise
        except PFLEvalError as e:
            error_line = error_ctx.format_error_from_lines_node(e.node)
            if error_line:
                print(error_line)
            raise e

        if isinstance(res, PFLRunnerResult):
            return res.data 
        return res 

    async def run_func(self, func_uid: str, scope: Optional[dict[str, Any]] = None):
        func_node = self._library.all_compiled[func_uid]
        if scope is not None:
            return await self._run_func(func_uid, scope)
        else:
            assert func_node.compile_info.meta is not None 
            fn_meta = func_node.compile_info.meta
            assert fn_meta.inline_run_env_fn is not None 
            inline_run_env = fn_meta.inline_run_env_fn()
            scope = inline_run_env.kwargs
            ctxes = inline_run_env.contexts
            with contextlib.ExitStack() as stack:
                for ctx in ctxes:
                    stack.enter_context(ctx)
                return await self._run_func(func_uid, scope)
        return await self._run_func(func_uid, scope)

    async def run_until(self, lineno: int, func_uid: str, scope: Optional[dict[str, Any]] = None, exit_event: Optional[asyncio.Event] = None):
        func_node = self._library.all_compiled[func_uid]
        assert self._state.type == PFLAsyncRunnerStateType.IDLE, \
            "You must call clear_runtime_state() before run_until()"
        stmt_should_pause = self._library.find_stmt_by_path_lineno(func_node.get_module_import_path(), lineno)
        assert stmt_should_pause is not None, \
            f"Cannot find statement at {func_node.get_module_import_path()}:{lineno}"
        stmt_start_lineno = stmt_should_pause.source_loc[0]
        self.add_breakpoint(stmt_start_lineno, one_shot=False)
        try:
            return await self.run_func(func_uid, scope)
        except PFLEvalStop:
            self.clear_runtime_state()
        except:
            traceback.print_exc()
            raise
        finally:
            self.clear_runtime_state()
            if exit_event is not None:
                exit_event.set()
            await self.event_eval_stop.emit_async()

    def get_state(self) -> PFLAsyncRunnerState:
        return self._state

    def is_paused(self) -> bool:
        return self._state.type == PFLAsyncRunnerStateType.PAUSE

    async def continue_until(self, lineno: int):
        assert self._state.type == PFLAsyncRunnerStateType.PAUSE
        self.release_breakpoint()
        self._breakpoints.clear()
        self.add_breakpoint(lineno, one_shot=False)
