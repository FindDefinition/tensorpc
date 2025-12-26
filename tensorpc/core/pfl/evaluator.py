import asyncio
from collections.abc import Sequence
from contextlib import AbstractContextManager, ExitStack
import contextlib
import contextvars
import enum
import inspect
from pathlib import Path
import traceback
import dataclasses as dataclasses_plain
from typing import Any, Callable, Coroutine, ForwardRef, Optional, Type, Union, cast
from typing_extensions import TypeAlias, Unpack
from tensorpc.core.asynctools import cancel_task
import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.annolib import (Undefined, is_undefined, undefined)
from tensorpc.core.event_emitter.single import SingleAsyncEventEmitter
from tensorpc.core.inspecttools import unwrap_fn_static_cls_property
from tensorpc.core.moduleid import get_module_id_of_type
from tensorpc.core.pfl.constants import PFL_BUILTIN_PROXY_INIT_FN
from tensorpc.utils.uniquename import UniqueNamePool

from .core import (BACKEND_CONFIG_REGISTRY, PFL_LOGGER, PFLErrorFormatContext, PFLInlineRunEnv, PFLParseCache, StaticEvalConfig, PFLMetaInferResult, PFLParseConfig,
                   PFLParseContext, PFLExprInfo, PFLExprType,
                   enter_parse_context, get_parse_context, get_parse_context_checked)
from .pfl_ast import (BinOpType, BoolOpType, CompareType, PFLAnnAssign, PFLArg,
                      PFLArray, PFLAssign, PFLAstNodeBase, PFLAstStmt,
                      PFLASTType, PFLAttribute, PFLAugAssign, PFLBinOp,
                      PFLBoolOp, PFLBreak, PFLCall, PFLClass, PFLCompare, PFLConstant, PFLContinue, PFLDict,
                      PFLExpr, PFLExprStmt, PFLFor, PFLFunc, PFLIf, PFLIfExp, PFLModule,
                      PFLName, PFLReturn, PFLSlice, PFLStaticVar, PFLSubscript, PFLTreeNodeFinder, PFLTuple,
                      PFLUnaryOp, PFLWhile, UnaryOpType, iter_child_nodes, unparse_pfl_ast, unparse_pfl_expr, walk,
                      PFLAstParseError, PFLEvalError)

from .pfl_reg import  STD_REGISTRY, StdRegistryItem
from .parser import PFLLibrary, default_pfl_var_proc

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
    def __init__(self, library: PFLLibrary, cfg: StaticEvalConfig, assign_check: Optional[Callable[[Any, Any],
                                         Optional[PFLMetaInferResult]]] = None):
        self.cfg = cfg
        self._library = library
        self._assign_check = assign_check


    @classmethod 
    def meta_evaulator(cls, library: PFLLibrary, prefer_meta_eval: bool = True, assign_check: Optional[Callable[[Any, Any],
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
            if isinstance(expr_node, PFLCall) and isinstance(expr_node.func, PFLAttribute):
                child_nodes = list(iter_child_nodes(expr_node)) + [expr_node.func.value]
            else:
                child_nodes = list(iter_child_nodes(expr_node))
            # child_nodes_types = [type(x) for x in child_nodes]
            all_success: list[bool] = []
            for n in child_nodes:
                if isinstance(n, PFLExpr):
                    success = self._eval_expr(n, scope)
                    all_success.append(success)
            # print("EVAL START", unparse_pfl_ast(expr_node), all_success, child_nodes_types)
            node_allow_all_child_fail = False
            if isinstance(expr_node, (PFLArray, PFLTuple)):
                # list/tuple have length info, so we allow all fail.
                node_allow_all_child_fail = True
            is_compiled_func: bool = False 
            if isinstance(expr_node, (PFLCall)):
                is_compiled_func = expr_node.func.st.compiled_uid is not None
            elif isinstance(expr_node, ( PFLUnaryOp, PFLBinOp, PFLCompare, PFLSubscript)):
                is_compiled_func = expr_node.st.compiled_uid is not None
            if all_success and not any(all_success) and not node_allow_all_child_fail and not is_compiled_func:
                if self.cfg.allow_partial:
                    return False
                else:
                    expr_node_str = unparse_pfl_expr(expr_node)
                    raise PFLEvalError(f"Some child of Expr {expr_node_str} eval failed.", expr_node)
            try:
                if isinstance(expr_node, PFLCall):
                    func_st = expr_node.func.st
                    all_compiled = self._library.all_compiled
                    if func_st.compiled_uid is not None:
                        if func_st.type == PFLExprType.FUNCTION:
                            assert func_st.compiled_uid in all_compiled
                            compiled_node = all_compiled[func_st.compiled_uid]
                            assert isinstance(compiled_node, PFLFunc)
                            matched = expr_node._get_matched_args(compiled_node.st.get_func_info_checked())
                            assert matched.var_kwarg is None and matched.vararg is None, "compiled pfl function don't have vaargs"
                            func_scope = {}
                            for arg_info, arg_expr in matched.args:
                                if not is_undefined(arg_expr):
                                    func_scope[arg_info.name] = arg_expr.st.metadata
                                else:
                                    assert not is_undefined(arg_info.default)
                                    func_scope[arg_info.name] = arg_info.default
                            self._eval_total_tree_node(compiled_node, func_scope)
                            if compiled_node.ret_st is not None:
                                expr_node.st.metadata = compiled_node.ret_st.metadata
                            return True
                        elif func_st.type == PFLExprType.DATACLASS_TYPE:
                            assert func_st.compiled_uid in all_compiled
                            cls_node = all_compiled[func_st.compiled_uid]
                            assert isinstance(cls_node, PFLClass)
                            cls_eval_uids: list[str] = []
                            if cls_node.init_uid != "":
                                cls_eval_uids.append(cls_node.init_uid)
                            if cls_node.post_init_uid != "":
                                cls_eval_uids.append(cls_node.post_init_uid)
                            for init_uid in cls_eval_uids:
                                init_node = all_compiled[init_uid]
                                assert isinstance(init_node, PFLFunc)
                                matched = expr_node._get_matched_args(init_node.st.get_func_info_checked())
                                assert matched.var_kwarg is None and matched.vararg is None, "compiled pfl function don't have vaargs"
                                func_scope = {}
                                for arg_info, arg_expr in matched.args:
                                    if not is_undefined(arg_expr):
                                        func_scope[arg_info.name] = arg_expr.st.metadata
                                    else:
                                        assert not is_undefined(arg_info.default)
                                        func_scope[arg_info.name] = arg_info.default
                                self._eval_total_tree_node(init_node, func_scope)
                            # if class have no user defined init/post init, no need to do meta eval.
                            return True

                elif isinstance(expr_node, (PFLUnaryOp, PFLBinOp, PFLCompare, PFLSubscript)):
                    if expr_node.st.compiled_uid is not None:
                        # compiled unary op.
                        all_compiled = self._library.all_compiled
                        assert expr_node.st.compiled_uid in all_compiled
                        compiled_node = all_compiled[expr_node.st.compiled_uid]
                        assert isinstance(compiled_node, PFLFunc)
                        func_args = compiled_node.st.get_func_info_checked().args
                        func_scope = {}
                        if isinstance(expr_node, PFLUnaryOp):
                            val_args = [expr_node.operand.st.metadata]
                        elif isinstance(expr_node, (PFLBinOp, PFLCompare)):
                            val_args = [expr_node.left.st.metadata, expr_node.right.st.metadata]
                            if expr_node.get_is_right_val():
                                val_args = [val_args[1], val_args[0]] 
                        elif isinstance(expr_node, (PFLSubscript)):
                            val_args = [expr_node.value.st.metadata]
                            if isinstance(expr_node.slice, PFLExpr):
                                val_args.append(expr_node.slice.st.metadata)
                            else:
                                val_args.append(tuple(s.st.metadata for s in expr_node.slice))
                        else:
                            raise NotImplementedError   
                        for func_arg, val in zip(func_args, val_args):
                            func_scope[func_arg.name] = val
                        self._eval_total_tree_node(compiled_node, func_scope)
                        if compiled_node.ret_st is not None:
                            expr_node.st.metadata = compiled_node.ret_st.metadata
                        return True

                if self.cfg.prefer_meta_eval:
                    res = expr_node.metaeval()
                else:
                    res = expr_node.consteval()
                # print("EVAL", unparse_pfl_ast(expr_node), expr_node.st.metadata)
            except PFLEvalError as e:
                raise e
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
            if k not in name_to_args:
                # we don't raise error here because we may need to reuse the test data generation code.
                continue
            info = dataclasses.replace(name_to_args[k].st)
            info.metadata = v
            arg_st = name_to_args[k].st
            if arg_st.has_metadata() and self._assign_check is not None:
                # perform assign check
                res = self._assign_check(arg_st.metadata, info.metadata)
                if res is not None:
                    info.metadata = res.data
            # TODO how to enable assign check for function argument here?
            #         name_to_args[k].st.metadata = res.data
            # else:
            #     name_to_args[k].st.metadata = v
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
        code_for_error = self._library.get_module_by_func_uid(func_node.uid).compile_info.code
        lines = []
        if code_for_error is not None:
            lines = code_for_error.split("\n")
        outer_ctx = get_parse_context() 
        if outer_ctx is not None:
            parse_ctx = PFLParseContext.from_outer_ctx(outer_ctx, lines, {})
        else:
            parse_ctx = PFLParseContext(lines, {}, default_pfl_var_proc, backend, cfg=parse_cfg, eval_cfg=self.cfg)
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
        func_nodes = self._library.get_compiled_unit_specs(func)
        assert len(func_nodes) == 1, "only support evaluate non-template function."
        func_node = func_nodes[0]
        # perform const fold and meta inference, result is stored in metadata in each static type.
        # WARNING: inplace operation
        backend = self._library.backend
        if parse_cfg is None:
            assert backend in BACKEND_CONFIG_REGISTRY, "you must register backend config first if parse_cfg isn't provided."
            parse_cfg = BACKEND_CONFIG_REGISTRY[backend]
        _clear_consteval_result(func_node)
        if isinstance(func, str):
            code_for_error = self._library.get_module_by_func_uid(func_node.uid).compile_info.code
        else:
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
            parse_ctx = PFLParseContext(lines, {}, default_pfl_var_proc, backend, cfg=parse_cfg, eval_cfg=self.cfg)
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
                        if stmt.target.id in scope:
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
                            if assign_check is not None and var_in_if.has_metadata() and var_in_else.has_metadata():
                                new_meta_val = assign_check(var_in_if.metadata,
                                                                var_in_else.metadata)
                                if new_meta_val is not None:
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
class PFLRunnerExprHit:
    expr: PFLExpr
    for_stack: list[tuple[PFLFor, int, int, int]]
    data: Any

@dataclasses.dataclass
class PFLRunnerResult:
    type: PFLRunnerResultType
    data: Optional[Any] = None

@dataclasses.dataclass
class PFLRunnerCtrlBase:
    node: PFLAstNodeBase
    should_pause: bool = False
    enabled: bool = True

@dataclasses.dataclass(kw_only=True, config=dataclasses.PyDanticConfigForAnyObject)
class PFLRunnerCtrlFor(PFLRunnerCtrlBase):
    # which step should stop
    step: int 
    range: range
    stop_in_start: bool 

    def __post_init__(self):
        assert isinstance(self.node, PFLFor)
        assert self.node.iter.st.type == PFLExprType.RANGE, "only support base for"

class PFLRunnerStateType(enum.IntEnum):
    IDLE = 0
    RUNNING = 1
    DURING_RUNNING_TO = 2
    PAUSE = 3
    NEED_STOP = 4

@dataclasses.dataclass
class PFLBreakpointDesc:
    lineno: int 
    enabled: bool = True
    one_shot: bool = False

@dataclasses.dataclass
class PFLRunnerFrame:
    node: PFLFunc
    func_uid_no_spec: str
    path: str
    call_node: Optional[PFLAstNodeBase]
    scope: dict[str, Any]
    module_code_lines: list[str]
    cur_stmt: Optional[PFLAstStmt] = None

@dataclasses.dataclass
class PFLRunnerBreakpoint:
    node: PFLAstNodeBase
    scope: dict[str, Any]
    stack: list[PFLRunnerFrame]
    is_main_thread: bool

@dataclasses.dataclass
class _AsyncThreadRequest:
    func_uid: str 
    scope: dict[str, Any]
    call_node: Optional[PFLAstNodeBase] = None
    name_suffix: Optional[str] = None

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class PFLRunnerState:
    type: PFLRunnerStateType
    cur_bkpt: Optional[PFLRunnerBreakpoint] = None
    cur_ctrl_points: dict[int, PFLRunnerCtrlBase] = dataclasses.field(default_factory=dict)
    pause_next_line: bool = False
    stack: list[PFLRunnerFrame] = dataclasses.field(default_factory=list)
    cur_expr: Optional[PFLExpr] = None
    cur_call_std_func: Optional[Callable] = None
    temp_bkpt_loc: tuple[str, int] = ("", -1) # (path, lineno)
    thread_queue: Optional[asyncio.Queue[_AsyncThreadRequest]] = None


@dataclasses_plain.dataclass
class _AsyncProgramSharedState:
    library: PFLLibrary
    observed_exprs: dict[tuple[int, int, Optional[int], Optional[int]], str]
    breakpoints: dict[tuple[str, int], PFLBreakpointDesc]
    std_scope: dict[str, Any] 
    event_enter_bkpt: SingleAsyncEventEmitter[str, PFLRunnerBreakpoint] = dataclasses_plain.field(default_factory=SingleAsyncEventEmitter)
    event_leave_bkpt: SingleAsyncEventEmitter[str, PFLRunnerBreakpoint] = dataclasses_plain.field(default_factory=SingleAsyncEventEmitter)
    event_new_ctrl_point: SingleAsyncEventEmitter[str, PFLRunnerCtrlBase] = dataclasses_plain.field(default_factory=SingleAsyncEventEmitter)
    event_delete_ctrl_point: SingleAsyncEventEmitter[str, PFLRunnerCtrlBase] = dataclasses_plain.field(default_factory=SingleAsyncEventEmitter)
    event_ctrl_point_change: SingleAsyncEventEmitter[str, PFLRunnerCtrlBase] = dataclasses_plain.field(default_factory=SingleAsyncEventEmitter)
    event_run_stop: SingleAsyncEventEmitter[str] = dataclasses_plain.field(default_factory=SingleAsyncEventEmitter)
    event_run_start: SingleAsyncEventEmitter[str] = dataclasses_plain.field(default_factory=SingleAsyncEventEmitter)
    event_expr_hit: SingleAsyncEventEmitter[str, PFLRunnerExprHit] = dataclasses_plain.field(default_factory=SingleAsyncEventEmitter)


_PFL_STATE_CONTEXT: contextvars.ContextVar[Optional[PFLRunnerState]] = contextvars.ContextVar(
    "PFLRunnerState", default=None)

def get_pfl_runner_state() -> Optional[PFLRunnerState]:
    """Get the current PFL async runner state."""
    return _PFL_STATE_CONTEXT.get(None)

@contextlib.contextmanager
def enter_pfl_runner_state(state: PFLRunnerState):
    """Context manager to enter a PFL async runner state."""
    token = _PFL_STATE_CONTEXT.set(state)
    try:
        yield state
    finally:
        _PFL_STATE_CONTEXT.reset(token)

class PFLEvalStop(Exception):

    def __init__(self, msg: str, node: PFLAstNodeBase):
        super().__init__(msg)
        self.node = node

_CORO_NONE: TypeAlias = Union[Coroutine[None, None, None], None]

class PFLAsyncThread:
    """A PFL runner that support breakpoints (via asyncio).
    other option is write a VM which is too complex.

    TODO: currently we only support all function defines in same file
    """
    def __init__(self, thread_id: str, shared_state: _AsyncProgramSharedState, 
                 thread_queue: asyncio.Queue[_AsyncThreadRequest], 
                 is_main_thread: bool = False):
        self.thread_id = thread_id
        # TODO temp data class support?
        self._shared_state = shared_state
        self._library = shared_state.library
        self._state = PFLRunnerState(PFLRunnerStateType.IDLE, thread_queue=thread_queue)
        self._bkpt_event = asyncio.Event()
        self._exit_event = asyncio.Event()
        self._thread_queue = thread_queue
        self._is_main_thread = is_main_thread

    def close(self):
        self._bkpt_event.clear()

    @property 
    def is_main_thread(self):
        return self._is_main_thread

    @property 
    def event_enter_bkpt(self):
        return self._shared_state.event_enter_bkpt

    @property 
    def event_leave_bkpt(self):
        return self._shared_state.event_leave_bkpt

    @property 
    def event_new_ctrl_point(self):
        return self._shared_state.event_new_ctrl_point

    @property 
    def event_delete_ctrl_point(self):
        return self._shared_state.event_delete_ctrl_point

    @property 
    def event_ctrl_point_change(self):
        return self._shared_state.event_ctrl_point_change

    @property 
    def event_run_stop(self):
        return self._shared_state.event_run_stop

    @property 
    def event_run_start(self):
        return self._shared_state.event_run_start

    @property 
    def event_expr_hit(self):
        return self._shared_state.event_expr_hit

    async def _run_coro_none(self, fn: Callable[..., _CORO_NONE], *args) -> None:
        """Run a coroutine that returns None."""
        res = fn(*args)
        if inspect.iscoroutine(res):
            await res

    def release_breakpoint(self, stop: bool = False):
        assert self._state.type == PFLRunnerStateType.PAUSE, \
            f"release_breakpoint called in state {self._state.type}, expected PAUSE."
        self._bkpt_event.set()
        if stop:
            self._state.type = PFLRunnerStateType.NEED_STOP

    def add_temp_breakpoint(self, path: str, lineno: int):
        if not path.startswith("<"):
            path_unified = str(Path(path).resolve())
        else:
            path_unified = path
        self._state.temp_bkpt_loc = (path_unified, lineno)

    def add_breakpoint(self, path: str, lineno: int, enabled: bool = True, one_shot: bool = False):
        if not path.startswith("<"):
            path_unified = str(Path(path).resolve())
        else:
            path_unified = path
        self._shared_state.breakpoints[(path_unified, lineno)] = PFLBreakpointDesc(lineno, enabled, one_shot)

    def remove_breakpoint(self, lineno: int):
        if lineno in self._shared_state.breakpoints:
            self._shared_state.breakpoints.pop(lineno)

    async def _enter_breakpoint(self, node: PFLAstNodeBase, scope: dict[str, Any]):
        self._state.cur_bkpt = PFLRunnerBreakpoint(node, scope, self._state.stack, self._is_main_thread)
        self._state.type = PFLRunnerStateType.PAUSE
        self._bkpt_event.clear()
        # call user callback after event set to let user release this bkpt in callback.
        if not self.event_enter_bkpt.is_empty():
            await self.event_enter_bkpt.emit_async(self.thread_id, self._state.cur_bkpt)
        await self._bkpt_event.wait() 
        if self._state.type == PFLRunnerStateType.NEED_STOP:
            self._state.type = PFLRunnerStateType.IDLE
            for cp in self._state.cur_ctrl_points.values():
                await self.event_delete_ctrl_point.emit_async(self.thread_id, cp)
            self._state.cur_ctrl_points.clear()
            # self._breakpoints.clear()
            raise PFLEvalStop("Eval Stop by user.", node)
        self._state.type = PFLRunnerStateType.RUNNING
        if not self.event_leave_bkpt.is_empty():
            await self.event_leave_bkpt.emit_async(self.thread_id, self._state.cur_bkpt)
        self._state.cur_bkpt = None 

    async def _may_pause_by_ctrl_points(self, node: PFLAstNodeBase, scope: dict[str, Any]):
        if not self._state.cur_ctrl_points:
            return False
        if all(cp.should_pause for cp in self._state.cur_ctrl_points.values()):
            await self._enter_breakpoint(node, scope)
            return True 
        return False

    async def _check_enter_breakpoint(self, node: PFLAstNodeBase, scope: dict[str, Any]):
        cur_loc = (self._state.stack[-1].path, node.source_loc[0])
        if self._state.pause_next_line:
            # if we are in temp breakpoint, we should pause.
            self._state.pause_next_line = False
        elif self._state.temp_bkpt_loc == cur_loc:
            # if we are in temp breakpoint, we should pause.
            self._state.temp_bkpt_loc = ("", -1)
        elif cur_loc in self._shared_state.breakpoints:
            pass
        else:
            return
        if self._state.cur_ctrl_points:
            should_pause = all(cp.should_pause for cp in self._state.cur_ctrl_points.values())
        else:
            should_pause = True
        if should_pause:
            return await self._enter_breakpoint(node, scope)

    async def _get_subscript_target_slice(self, node: PFLSubscript, scope: dict[str, Any]):
        tgt = await self._run_expr(node.value, scope)
        if isinstance(node.slice, Sequence):
            slice_strs = [await self._run_expr(s, scope) for s in node.slice]
            slice_str = tuple(slice_strs)
        else:
            slice_str = await self._run_expr(node.slice, scope)
        return tgt, slice_str


    async def _run_expr(self, expr: PFLExpr, scope: dict[str, Any]) -> Any:
        prev = self._state.cur_expr
        self._state.cur_expr = expr
        try:
            if isinstance(expr, PFLName):
                if expr.st.compiled_uid is not None:
                    res = self._library.all_compiled[expr.st.compiled_uid]
                else:
                    res = scope[expr.id]
                # return scope[expr.id]
            elif isinstance(expr, PFLAttribute):
                if expr.st.compiled_uid is not None:
                    return self._library.all_compiled[expr.st.compiled_uid]
                res = getattr(await self._run_expr(expr.value, scope), expr.attr)
            elif isinstance(expr, PFLConstant):
                res = expr.value
            elif isinstance(expr, PFLSlice):
                lo_str = None if is_undefined(expr.lo) else await self._run_expr(expr.lo, scope)
                hi_str = None if is_undefined(expr.hi) else await self._run_expr(expr.hi, scope)
                step_str = None if is_undefined(expr.step) else await self._run_expr(
                    expr.step, scope)
                res = slice(lo_str, hi_str, step_str)
            elif isinstance(expr, PFLSubscript):
                tgt, slice_str = await self._get_subscript_target_slice(expr, scope)
                if expr.st.compiled_uid is not None:
                    custom_fn = self._library.get_compiled_func_by_uid(expr.st.compiled_uid)
                    assert isinstance(custom_fn, PFLFunc)
                    res = await self._run_func(expr.st.compiled_uid, {
                        custom_fn.args[0].arg: tgt,
                        custom_fn.args[1].arg: slice_str
                    }, call_node=expr)
                else:
                    res = tgt[slice_str]
            elif isinstance(expr, PFLArray):
                res = [await self._run_expr(elt, scope)
                                    for elt in expr.elts]
            elif isinstance(expr, PFLTuple):
                res = tuple([await self._run_expr(elt, scope)
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
            elif isinstance(expr, PFLBoolOp):
                if expr.op == BoolOpType.AND:
                    early_exit = False
                    res_arr: list[bool] = []
                    for v in expr.values:
                        val = await self._run_expr(v, scope)
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
                        val = await self._run_expr(v, scope)
                        if val:
                            early_exit = True
                            break
                        res_arr.append(val)
                    if early_exit:
                        res = True
                    else:
                        res = any(res_arr)
            elif isinstance(expr, (PFLBinOp, PFLCompare)):
                left = await self._run_expr(expr.left, scope)
                right = await self._run_expr(expr.right, scope)
                if expr.st.compiled_uid is not None:
                    custom_fn = self._library.get_compiled_func_by_uid(expr.st.compiled_uid)
                    assert isinstance(custom_fn, PFLFunc)
                    if expr.get_is_right_val():
                        left, right = right, left
                    res = await self._run_func(expr.st.compiled_uid, {
                        custom_fn.args[0].arg: left,
                        custom_fn.args[1].arg: right
                    }, call_node=expr)
                else:
                    res = expr.run(left, right)
            elif isinstance(expr, PFLUnaryOp):
                left = await self._run_expr(expr.operand, scope)
                if expr.st.compiled_uid is not None:

                    custom_fn = self._library.get_compiled_func_by_uid(expr.st.compiled_uid)
                    assert isinstance(custom_fn, PFLFunc)
                    res = await self._run_func(expr.st.compiled_uid, {
                        custom_fn.args[0].arg: left,
                    }, call_node=expr)
                else:
                    res = expr.run(left)
            elif isinstance(expr, PFLCall):
                func_st = expr.func.st
                if func_st.compiled_uid is not None:
                    kwargs = {}
                    finfo = func_st.get_func_info_checked()
                    # compiled pfl function don't have vaargs
                    matched = expr._get_matched_args(finfo)
                    assert matched.var_kwarg is None and matched.vararg is None, "compiled pfl function don't have vaargs"
                    for arg_info, arg_expr in matched.args:
                        if not is_undefined(arg_expr):
                            arg_value = await self._run_expr(arg_expr, scope)
                            kwargs[arg_info.name] = arg_value
                        else:
                            assert not is_undefined(arg_info.default)
                            kwargs[arg_info.name] = arg_info.default

                    if func_st.type == PFLExprType.FUNCTION:
                        res = await self._run_func(func_st.compiled_uid, kwargs, call_node=expr)
                    else:
                        cls_node = self._library.all_compiled[func_st.compiled_uid]
                        assert isinstance(cls_node, PFLClass)
                        assert func_st.type == PFLExprType.DATACLASS_TYPE
                        # WARNING: for dataclasses, we will call python init/post_init firstly
                        # then call user-defined init and post_init to support breakpoint in init/post_init.
                        # this requires your init/post_init must have no side effect.
                        dcls_info = func_st.get_dcls_info_checked()
                        dcls = dcls_info.raw_func
                        assert dcls is not None
                        res = dcls(**kwargs)
                        # run user-defined init and post_init if exists.
                        if cls_node.init_uid != "":
                            kwargs["self"] = res
                            await self._run_func(cls_node.init_uid, kwargs, call_node=expr)
                        if cls_node.post_init_uid != "":
                            await self._run_func(cls_node.post_init_uid, {
                                "self": res
                            }, call_node=expr)
                else:
                    func_val = await self._run_expr(expr.func, scope)
                    if expr.func.st.proxy_dcls is not None:
                        func_val = inspect.getattr_static(expr.func.st.proxy_dcls, PFL_BUILTIN_PROXY_INIT_FN)
                    args = []
                    kwargs = {}
                    for arg_expr in expr.args:
                        arg_expr_val = await self._run_expr(arg_expr, scope)
                        args.append(arg_expr_val)
                    if not is_undefined(expr.keys):
                        assert not is_undefined(expr.vals)
                        for key_expr, value_expr in zip(expr.keys, expr.vals):
                            value_value = await self._run_expr(value_expr, scope)
                            kwargs[key_expr] = value_value
                    try:
                        self._state.cur_call_std_func = func_val
                        res = func_val(*args, **kwargs) 
                    finally:
                        self._state.cur_call_std_func = None
            elif isinstance(expr, PFLIfExp):
                test = await self._run_expr(expr.test, scope)
                if test:
                    res = await self._run_expr(expr.body, scope)
                else:
                    res = await self._run_expr(expr.orelse, scope)
            else:
                raise NotImplementedError(f"Unrecognized PFLExpr type: {type(expr)}")
            if self._shared_state.observed_exprs:
                if expr.source_loc in self._shared_state.observed_exprs:
                    await self.event_expr_hit.emit_async(self.thread_id, 
                        PFLRunnerExprHit(expr, self._get_for_stack(), res))
            return res
        finally:
            self._state.cur_expr = prev

    def _get_for_stack(self) -> list[tuple[PFLFor, int, int, int]]:
        """Get the current for stack, each item is a tuple of (PFLFor, step, range_start, range_stop)."""
        res: list[tuple[PFLFor, int, int, int]] = []
        for cp in self._state.cur_ctrl_points.values():
            if isinstance(cp, PFLRunnerCtrlFor):
                res.append((cast(PFLFor, cp.node), cp.step, cp.range.start, cp.range.stop))
        return res

    async def run_body(self, block_body: list[PFLAstStmt], scope: dict[str, Any]) -> Union[Any, PFLRunnerResult]:
        frame = self._state.stack[-1]
        prev_scope = frame.scope 

        try:
            frame.scope = scope
            for stmt in block_body:
                frame.cur_stmt = stmt
                if self._shared_state.breakpoints or self._state.pause_next_line or self._state.temp_bkpt_loc[1] != -1:
                    await self._check_enter_breakpoint(stmt, scope)
                # print("RUN STMT", unparse_pfl_ast(stmt))
                try:
                    if isinstance(stmt, PFLExpr):
                        await self._run_expr(stmt, scope)
                    elif isinstance(stmt, (PFLAssign, PFLAnnAssign)):
                        if stmt.value is not None:
                            value = await self._run_expr(stmt.value, scope)
                            if self._shared_state.observed_exprs:
                                if stmt.target.source_loc in self._shared_state.observed_exprs:
                                    await self.event_expr_hit.emit_async(self.thread_id, 
                                        PFLRunnerExprHit(stmt.target, self._get_for_stack(), value))
                            # when stmt.target is attr or subscript, we need to evaluate more deeper thing.
                            if isinstance(stmt.target, (PFLAttribute, PFLSubscript)):
                                assert not is_undefined(stmt.target.is_store) and stmt.target.is_store == True
                                deep_val = await self._run_expr(stmt.target.value, scope)
                                if isinstance(stmt.target, (PFLAttribute)):
                                    setattr(deep_val, stmt.target.attr, value)
                                else:
                                    tgt, slice_str = await self._get_subscript_target_slice(stmt.target, scope)
                                    if stmt.target.st.additional_compiled_uid is not None:
                                        custom_fn = self._library.get_compiled_func_by_uid(stmt.target.st.additional_compiled_uid)
                                        await self._run_func(stmt.target.st.additional_compiled_uid, {
                                            custom_fn.args[0].arg: tgt,
                                            custom_fn.args[1].arg: slice_str,
                                            custom_fn.args[2].arg: value,
                                        }, call_node=stmt)
                                    else:
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
                                    for v in stmt._new.keys():
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
                                if stmt.target.st.additional_compiled_uid is not None:
                                    assert stmt.target.st.compiled_uid is not None 
                                    custom_fn = self._library.get_compiled_func_by_uid(stmt.target.st.compiled_uid)
                                    custom_set_fn = self._library.get_compiled_func_by_uid(stmt.target.st.additional_compiled_uid)
                                    new_val = await self._run_func(stmt.target.st.compiled_uid, {
                                        custom_fn.args[0].arg: tgt,
                                        custom_fn.args[1].arg: slice_str,
                                    }, call_node=stmt)
                                    await self._run_func(stmt.target.st.additional_compiled_uid, {
                                        custom_set_fn.args[0].arg: tgt,
                                        custom_set_fn.args[1].arg: slice_str,
                                        custom_set_fn.args[2].arg: new_val,
                                    }, call_node=stmt)
                                else:
                                    new_val = stmt.run(tgt[slice_str], value)
                                    tgt[slice_str] = new_val
                        else:
                            assert isinstance(stmt.target, PFLName)
                            new_val = stmt.run(scope[stmt.target.id], value)
                            scope[stmt.target.id] = new_val
                    elif isinstance(stmt, PFLFor):
                        iter_obj = await self._run_expr(stmt.iter, scope)
                        tgt = stmt.target
                        # TODO support tuple
                        assert isinstance(tgt, PFLName)
                        if isinstance(iter_obj, range):
                            stmt_id = id(stmt)
                            private_scope = scope.copy()
                            ctrl = PFLRunnerCtrlFor(stmt, enabled=True, step=iter_obj.start, range=iter_obj, stop_in_start=False)
                            self._state.cur_ctrl_points[stmt_id] = ctrl
                            if not self.event_new_ctrl_point.is_empty():
                                await self.event_new_ctrl_point.emit_async(self.thread_id, ctrl)
                            for i in iter_obj:
                                if ctrl.enabled:
                                    # print(i, ctrl.step, iter_obj)
                                    if ctrl.step < i:
                                        ctrl.step = i
                                        await self.event_ctrl_point_change.emit_async(self.thread_id, ctrl)
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
                                await self.event_delete_ctrl_point.emit_async(self.thread_id, ctrl)
                            
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
                except PFLEvalError as e:
                    if not e.traceback_set:
                        exc_str = "".join(traceback.format_exception_only(e))
                        assert e.node is not None
                        tb_str = self._format_current_traceback(e.node)
                        e.args = (f"{tb_str}\nOriginal Error: {exc_str}",)
                        e.node = None
                        e.traceback_set = True
                    raise
                except BaseException as e:
                    exc_str = "".join(traceback.format_exception_only(e))
                    tb_str = self._format_current_traceback(stmt)
                    raise PFLEvalError(f"{tb_str}\nOriginal Error: {exc_str}", None, traceback_set=True)
        finally:
            self._state.stack[-1].scope = prev_scope

    def _format_current_traceback(self, node: PFLAstNodeBase) -> str:
        msgs: list[str] = ["PFLAsyncRunner Traceback (most recent call last):"]
        stacks = self._state.stack.copy()
        stacks.append(dataclasses.replace(stacks[-1], call_node=node))
        for i, stack in enumerate(stacks):
            prev_stack = stack if i == 0 else stacks[i - 1]
            path = stack.node.compile_info.path
            if stack.call_node is not None:
                msgs.append(f"  File \"{path}\", line {stack.call_node.source_loc[0]}, in {prev_stack.node.name}")

                lineno = stack.call_node.source_loc[0]
                col = stack.call_node.source_loc[1]
                end_lineno = stack.call_node.source_loc[2]
                end_col = stack.call_node.source_loc[3]
                if end_lineno is None:
                    end_line = lineno
                else:
                    end_line = min(end_lineno, len(stack.module_code_lines))
                lines = stack.module_code_lines[lineno - 1:end_line]
                error_lines = lines.copy()
                if end_col is None:
                    end_col = max(len(line) for line in lines)
                min_length = max(1, end_col - col)

                if error_lines:
                    indicate_line = f"{' ' * col}{'^' * min_length}"
                    error_lines.insert(end_line - lineno + 1, indicate_line)
                for line in error_lines:
                    msgs.append(f"    {line}")
            else:
                msgs.append(f"  File \"{path}\", line {stack.node.compile_info.first_lineno}, in {prev_stack.node.name}")
        return "\n".join(msgs)

    async def _run_func(self, func_uid: str, scope: dict[str, Any], call_node: Optional[PFLAstNodeBase] = None) -> Any:
        func_node = self._library.get_compiled_func_by_uid(func_uid)
        module = self._library.get_module_by_func_uid(func_node.uid)
        module_lines = module.compile_info.code.split("\n")
        # error_ctx = PFLErrorFormatContext(module.compile_info.code.split("\n"))
        try:
            fn_scope = {**scope, **self._shared_state.std_scope}
            self._state.stack.append(PFLRunnerFrame(func_node, func_node.get_func_uid_no_spec(), func_node.compile_info.path, call_node, fn_scope, module_lines))
            res = await self.run_body(func_node.body, fn_scope)
        except PFLEvalStop:
            raise
        except PFLEvalError as e:
            # error_line = error_ctx.format_error_from_lines_node(e.node)
            # if error_line:
            #     print(error_line)
            raise e
        finally:
            self._state.stack.pop()

        if isinstance(res, PFLRunnerResult):
            return res.data 
        return res 

    async def entry_point(self, func_uid: str, scope: Optional[dict[str, Any]] = None,
            external_inline_env: Optional[PFLInlineRunEnv] = None,
            pause_loc: Optional[tuple[str, int]] = None):
        func_node = self._library.get_compiled_func_by_uid(func_uid)
        if pause_loc is not None:
            stmt_should_pause = self._library.find_stmt_by_path_lineno(func_node.get_define_path(), pause_loc[1])
            assert stmt_should_pause is not None, \
                f"Cannot find statement at {func_node.get_define_path()}:{pause_loc[1]}"
            stmt_start_lineno = stmt_should_pause.source_loc[0]
            self.add_temp_breakpoint(pause_loc[0], stmt_start_lineno)
        try:
            await self.event_run_start.emit_async(self.thread_id)
            with enter_pfl_runner_state(self._state):
                if scope is not None:
                    return await self._run_func(func_uid, scope)
                else:
                    if external_inline_env is None:
                        assert func_node.compile_info.meta is not None 
                        fn_meta = func_node.compile_info.meta
                        assert fn_meta.inline_run_env_fn is not None 
                        inline_run_env = fn_meta.inline_run_env_fn()
                    else:
                        inline_run_env = external_inline_env
                    scope = inline_run_env.kwargs
                    ctxes = inline_run_env.contexts
                    with contextlib.ExitStack() as stack:
                        for ctx in ctxes:
                            stack.enter_context(ctx)
                        return await self._run_func(func_uid, scope)
                return await self._run_func(func_uid, scope)
        except PFLEvalStop:
            PFL_LOGGER.warning("Eval stopped by user.")
        finally:
            self._exit_event.set()
            await self.event_run_stop.emit_async(self.thread_id)

    def get_state(self) -> PFLRunnerState:
        return self._state

    def get_cur_bkpt_checked(self) -> PFLRunnerBreakpoint:
        """Get the current breakpoint, raise if not in breakpoint state."""
        if self._state.type != PFLRunnerStateType.PAUSE:
            raise RuntimeError(f"Cannot get current breakpoint in state {self._state.type}, expected PAUSE.")
        assert self._state.cur_bkpt is not None, "Current breakpoint should not be None in PAUSE state."
        return self._state.cur_bkpt

    def is_paused(self) -> bool:
        return self._state.type == PFLRunnerStateType.PAUSE

    def continue_until(self, path: str, lineno: int):
        assert self._state.type == PFLRunnerStateType.PAUSE
        self.release_breakpoint()
        self.add_temp_breakpoint(path, lineno)

    def continue_next_line(self):
        assert self._state.type == PFLRunnerStateType.PAUSE
        self.release_breakpoint()
        # self._breakpoints.clear()
        self._state.pause_next_line = True


class PFLAsyncRunner:
    def __init__(self, library: PFLLibrary, observed_exprs: Optional[dict[str, PFLExpr]] = None):
        self._library = library
        # TODO temp data class support?
        std_scope: dict[str, Any] = {}
        for k, v in STD_REGISTRY.global_dict.items():
            if v.backend is None or v.backend == library.backend:
                std_scope[v.mapped_name] = v.dcls
        self._std_scope = std_scope
        self._observed_exprs: dict[tuple[int, int, Optional[int], Optional[int]], str] = {}
        if observed_exprs is not None:
            self.set_observed_exprs(observed_exprs)

        shared_state = _AsyncProgramSharedState(library, self._observed_exprs,
            {}, std_scope, )

        self._shared_state = shared_state

        self._running_threads: dict[str, PFLAsyncThread] = {}
        self._internal_exit_event: asyncio.Event = asyncio.Event()

        self._shutdown_event: asyncio.Event = asyncio.Event()
        self.event_run_stop: SingleAsyncEventEmitter[()] = SingleAsyncEventEmitter()
        self.event_run_start: SingleAsyncEventEmitter[()] = SingleAsyncEventEmitter()
        self.event_thread_changed: SingleAsyncEventEmitter[dict[str, PFLAsyncThread]] = SingleAsyncEventEmitter()

    @property 
    def event_thread_enter_bkpt(self):
        return self._shared_state.event_enter_bkpt

    @property 
    def event_thread_leave_bkpt(self):
        return self._shared_state.event_leave_bkpt

    @property 
    def event_thread_new_ctrl_point(self):
        return self._shared_state.event_new_ctrl_point

    @property 
    def event_thread_delete_ctrl_point(self):
        return self._shared_state.event_delete_ctrl_point

    @property 
    def event_thread_ctrl_point_change(self):
        return self._shared_state.event_ctrl_point_change

    @property 
    def event_thread_run_stop(self):
        return self._shared_state.event_run_stop

    @property 
    def event_thread_run_start(self):
        return self._shared_state.event_run_start

    @property 
    def event_thread_expr_hit(self):
        return self._shared_state.event_expr_hit


    def set_observed_exprs(self, observed_exprs: dict[str, PFLExpr]):
        """Set the observed expressions for the runner."""
        self._observed_exprs.clear()
        for k, v in observed_exprs.items():
            assert v.source_loc not in self._observed_exprs, \
                f"Observed expr {v.source_loc} already exists, please use different source_loc."
            self._observed_exprs[v.source_loc] = k

    def set_observed_source_locs(self, observed_slocs: dict[str, tuple[int, int, Optional[int], Optional[int]]]):
        """Set the observed expressions for the runner."""
        self._observed_exprs.clear()
        for k, v in observed_slocs.items():
            assert v not in self._observed_exprs, \
                f"Observed expr {v} already exists, please use different source_loc."
            self._observed_exprs[v] = k

    async def run_func(self, func_uid: str, 
            scope: Optional[dict[str, Any]] = None,
            external_inline_env: Optional[PFLInlineRunEnv] = None,
            exit_event: Optional[asyncio.Event] = None,
            pause_loc_map: Optional[dict[str, tuple[str, int]]] = None):
        assert not self._running_threads
        self._shutdown_event.clear()
        self._internal_exit_event.clear()
        new_thread_q: asyncio.Queue[_AsyncThreadRequest] = asyncio.Queue()
        main_thread = PFLAsyncThread(self._library.extract_fn_name(func_uid), self._shared_state, new_thread_q, is_main_thread=True)
        func_uid_nospec = self._library.extract_fn_uid_nospec(func_uid)
        new_thread_task = asyncio.create_task(new_thread_q.get())
        if pause_loc_map is None:
            pause_loc_map = {}
        pause_loc = pause_loc_map.get(func_uid_nospec, None)
        main_task = asyncio.create_task(main_thread.entry_point(func_uid, scope, external_inline_env, pause_loc), name=main_thread.thread_id)
        sd_task = asyncio.create_task(self._shutdown_event.wait())
        wait_tasks: list[asyncio.Task] = [new_thread_task, sd_task, main_task]
        task_to_thread: dict[asyncio.Task, PFLAsyncThread] = {main_task: main_thread}
        self._running_threads[main_thread.thread_id] = main_thread
        uniq_pool = UniqueNamePool()
        try:
            await self.event_run_start.emit_async()
            await self.event_thread_changed.emit_async(self._running_threads)
            while True:
                done, pending = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)
                if sd_task in done:
                    # shutdown event triggered, cancel all tasks.
                    for task in pending:
                        await cancel_task(task)
                    break
                for task in done:
                    if new_thread_task is task:
                        call_req = new_thread_task.result()
                        new_thread_id = self._library.extract_fn_name(call_req.func_uid)
                        if call_req.name_suffix is not None:
                            new_thread_id = f"{new_thread_id}_{call_req.name_suffix}"
                        new_thread_id_uniq = uniq_pool(new_thread_id)
                        thread = PFLAsyncThread(new_thread_id_uniq, self._shared_state, new_thread_q)
                        func_uid_nospec = self._library.extract_fn_uid_nospec(call_req.func_uid)
                        # currently pause loc is only enabled for main thread. 
                        # TODO if user can create unique and stable thread name_suffix, we can support pause loc for other threads.
                        # pause_loc = pause_loc_map.get(func_uid_nospec, None)
                        new_program_run_task = asyncio.create_task(thread.entry_point(call_req.func_uid, call_req.scope), name=new_thread_id)
                        self._running_threads[thread.thread_id] = thread
                        new_thread_task = asyncio.create_task(new_thread_q.get())
                        # wait_tasks = [new_thread_task, sd_task, *pending, new_program_run_task]
                        task_to_thread[new_program_run_task] = thread
                        await self.event_thread_changed.emit_async(self._running_threads)
                    else:
                        thread = task_to_thread.pop(task)
                        self._running_threads.pop(thread.thread_id)
                        exc = task.exception()
                        if exc is not None:
                            # shutdown all thread if some thread error.
                            PFL_LOGGER.error(f"Thread {thread.thread_id} Error. shutdown all.\n{str(task.exception())}", exc_info=exc)
                            for task in pending:
                                await cancel_task(task)
                            raise exc
                        if task_to_thread:
                            await self.event_thread_changed.emit_async(self._running_threads)
                if len(task_to_thread) == 0:
                    for task in pending:
                        await cancel_task(task)
                    break
                wait_tasks = [new_thread_task, sd_task, *task_to_thread.keys()]
        except asyncio.CancelledError:
            raise
        finally:
            self._running_threads.clear()
            if exit_event is not None:
                exit_event.set()
            await self.event_run_stop.emit_async()
            await self.event_thread_changed.emit_async({})
            self._shutdown_event.clear()
            self._internal_exit_event.set()

    async def run_until(self, lineno: int, func_uid: str, scope: Optional[dict[str, Any]] = None, exit_event: Optional[asyncio.Event] = None,
            external_inline_env: Optional[PFLInlineRunEnv] = None):
        fn_uid_no_spec = self._library.extract_fn_uid_nospec(func_uid)
        func_node = self._library.get_compiled_func_by_uid(func_uid)
        return await self.run_func(func_uid, scope, external_inline_env, exit_event,
            pause_loc_map={fn_uid_no_spec: (func_node.compile_info.path, lineno)})
    
    def get_current_pause_loc_map(self) -> dict[str, tuple[str, int]]:
        res: dict[str, tuple[str, int]] = {} # fn_uid_no_spec -> (path, lineno)
        for thread in self._running_threads.values():
            if thread._state.type == PFLRunnerStateType.PAUSE:
                cur_bkpt = thread.get_cur_bkpt_checked()
                frame = thread._state.stack[-1]
                fn_uid_nospec = PFLLibrary.extract_fn_uid_nospec(frame.func_uid_no_spec)
                res[fn_uid_nospec] = (frame.path, cur_bkpt.node.source_loc[0])
        return res

    def get_current_bkpts(self) -> dict[str, PFLRunnerBreakpoint]:
        res: dict[str, PFLRunnerBreakpoint] = {} # thread_id -> bkpt
        for thread in self._running_threads.values():
            if thread._state.type == PFLRunnerStateType.PAUSE:
                cur_bkpt = thread.get_cur_bkpt_checked()
                res[thread.thread_id] = cur_bkpt
        return res

    def continue_until(self, thread_id: str, path: str, lineno: int):
        # TODO should we suppress all bkpts here?
        thread = self._running_threads.get(thread_id)
        assert thread is not None, f"Thread {thread_id} not found."
        assert thread._state.type == PFLRunnerStateType.PAUSE
        thread.release_breakpoint()
        thread.continue_until(path, lineno)

    def continue_next_line(self, thread_id: str):
        # TODO should we suppress all bkpts here?
        thread = self._running_threads.get(thread_id)
        assert thread is not None, f"Thread {thread_id} not found."
        assert thread._state.type == PFLRunnerStateType.PAUSE
        thread.release_breakpoint()
        thread.continue_next_line()

    async def shutdown(self):
        """Shutdown the current running program."""
        if self._running_threads:
            self._shutdown_event.set()
            await self._internal_exit_event.wait()

    def is_paused(self):
        return any(t._state.type == PFLRunnerStateType.PAUSE for t in self._running_threads.values())
        
    def is_thread_paused(self, thread_id: str):
        thread = self._running_threads.get(thread_id)
        assert thread is not None, f"Thread {thread_id} not found."
        return thread._state.type == PFLRunnerStateType.PAUSE

    def is_running(self):
        return bool(self._running_threads) and self._internal_exit_event.is_set() == False

    def has_paused_thread(self):
        for t in self._running_threads.values():
            if t._state.type == PFLRunnerStateType.PAUSE:
                return True 
        return False

    def release_breakpoint(self, thread_id: str, stop: bool = False):
        thread = self._running_threads.get(thread_id)
        assert thread is not None, f"Thread {thread_id} not found."
        thread.release_breakpoint(stop)

    def release_all_breakpoint(self, stop: bool = False):
        for t in self._running_threads.values():
            t.release_breakpoint(stop)

    def sync_breakpoints(self, breakpoints: dict[tuple[str, int], PFLBreakpointDesc]):
        """Sync the breakpoints for the runner.
        threads will check shared state breakpoints in each stmt.
        """
        self._shared_state.breakpoints = breakpoints.copy()

    def get_thread(self, thread_id: str) -> PFLAsyncThread:
        thread = self._running_threads.get(thread_id)
        assert thread is not None, f"Thread {thread_id} not found."
        return thread 

    def get_all_threads(self) -> dict[str, PFLAsyncThread]:
        return self._running_threads.copy()

    def get_main_thread(self) -> Optional[PFLAsyncThread]:
        for t in self._running_threads.values():
            if t.is_main_thread:
                return t 
        return None

