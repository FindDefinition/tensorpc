"""tracer that used for cursor selection
"""

import ast
from dataclasses import dataclass
import enum
import inspect
from pathlib import Path
import sys
import threading
import traceback
from types import FrameType
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Type, Union

from tensorpc.core.astex.astcache import AstCache, AstCacheItem
from tensorpc.core.moduleid import get_module_id_of_type
from .calltracer import CallTracerContext
from .core import TraceEventType, FrameEventBase


class TraceType(enum.IntEnum):
    FuncArgs = 0
    LocalExprs = 1


@dataclass
class CursorFrameLineResult(FrameEventBase):
    eval_result: Any


@dataclass
class CodeFragTracerResult:
    frame_results: List[FrameEventBase]
    line_result: CursorFrameLineResult


@dataclass
class TraceInfo:
    type: TraceType
    path: Path
    func_name: str
    func_lineno: int
    expr: Union[str, ast.AST]
    lineno: int
    is_attr: bool
    is_store: bool
    assign_lineno_range: Optional[Tuple[int, int]] = None


def get_trace_infos_from_coderange_item(
        item: AstCacheItem, code_range: Tuple[int, int, int,
                                              int]) -> Optional[TraceInfo]:
    func_nodes = item.query_func_def_nodes_by_lineno_range(
        code_range[0], code_range[2])
    if not func_nodes:
        return None
    func_node = func_nodes[-1]
    func_name = func_node.name
    func_lineno = func_node.lineno
    deco_list = func_node.decorator_list
    # we fix lineno to match inspect
    if len(deco_list) > 0:
        func_lineno = min([d.lineno for d in deco_list])

    coderange_nodes = item.query_code_range_nodes(*code_range)
    expr_or_attr_nodes: List[Union[ast.expr, ast.Attribute, ast.Name]] = []
    has_attr = False
    code_range_length = code_range[3] - code_range[1]
    for n in coderange_nodes:
        if isinstance(n, (ast.expr, ast.Attribute, ast.Name)):
            expr_or_attr_nodes.append(n)
            if isinstance(n, (ast.Attribute, ast.Name)):
                has_attr = True
    if not expr_or_attr_nodes:
        # check func arg
        node_first_lineno_col = item.query_first_lineno_col_nodes(
            code_range[0], code_range[1])
        for n in node_first_lineno_col:
            if isinstance(n, ast.arg) and len(n.arg) == code_range_length:
                return TraceInfo(type=TraceType.FuncArgs,
                                 path=item.path,
                                 func_name=func_name,
                                 func_lineno=func_lineno,
                                 expr=n.arg,
                                 lineno=n.lineno,
                                 is_attr=False,
                                 is_store=False)
    else:
        is_store = False
        res = TraceInfo(type=TraceType.LocalExprs,
                        path=item.path,
                        func_name=func_name,
                        func_lineno=func_lineno,
                        expr=expr_or_attr_nodes[0],
                        lineno=expr_or_attr_nodes[0].lineno,
                        is_attr=has_attr,
                        is_store=is_store)
        if isinstance(expr_or_attr_nodes[0],
                      (ast.Name, ast.Attribute, ast.expr)):
            n = expr_or_attr_nodes[0]
            if hasattr(n, "ctx"):
                ctx = getattr(n, "ctx")
                if isinstance(ctx, ast.Store):
                    is_store = True
                    # find parent assign node
                    found = False
                    # we assume no nested assign exists in python language.
                    # TODO NamedExpr?
                    res.is_store = True
                    for node in item.assign_ctx_nodes:
                        ass_lineno = node.lineno
                        ass_end_lineno = node.end_lineno
                        if (n.lineno >= ass_lineno and n.end_lineno is not None
                                and ass_end_lineno is not None
                                and n.lineno <= ass_end_lineno):
                            res.assign_lineno_range = (ass_lineno,
                                                       ass_end_lineno)
                            found = True
                            break
                    if found:
                        return res
                    else:
                        # if we can't find assign ctx, drop this case.
                        return None

        return res


class TracerContext(object):
    def __init__(self,
                 trace_info: TraceInfo,
                 *,
                 _frame_cnt: int = 1):
        self.target_frames: Set[FrameType] = set()
        self.thread_local = threading.local()
        # code type -> (should trace, filter_res)
        self._frame_cnt = _frame_cnt
        self._inner_frame_fnames: Set[str] = set(
            [TracerContext.__enter__.__code__.co_filename])
        self._trace_info = trace_info
        self._expr_found = False
        self._trace_cur_assign_range: Optional[Tuple[int, int]] = None

        self.result_call_stack: List[FrameEventBase] = []
        self.result_line: Optional[CursorFrameLineResult] = None

    def _filter_frame(self, frame: FrameType):
        frame_path = Path(frame.f_code.co_filename).resolve()
        if self._expr_found:
            return False
        else:
            if frame_path != self._trace_info.path:
                return False
            if frame.f_code.co_firstlineno != self._trace_info.func_lineno:
                return False
        return True

    def __enter__(self):
        cur_frame = inspect.currentframe()
        self._expr_found = False
        self._trace_cur_assign_range = None
        assert cur_frame is not None
        frame = cur_frame
        _frame_cnt = self._frame_cnt
        while _frame_cnt > 0:
            self._inner_frame_fnames.add(cur_frame.f_code.co_filename)
            frame = cur_frame.f_back
            assert frame is not None
            cur_frame = frame
            _frame_cnt -= 1
        calling_frame = cur_frame
        if self._trace_info.type == TraceType.FuncArgs:
            trace_fn = self.trace_call_func
        else:
            trace_fn = self.trace_line_func
        if not self._is_internal_frame(calling_frame):
            calling_frame.f_trace = trace_fn
            self.target_frames.add(calling_frame)

        stack = self.thread_local.__dict__.setdefault(
            'original_trace_functions', [])
        stack.append(sys.gettrace())
        sys.settrace(trace_fn)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # print("EXIT", self._frame_cnt, self._inner_frame_fnames, self.target_frames)
        stack = self.thread_local.original_trace_functions
        sys.settrace(stack.pop())
        cur_frame = inspect.currentframe()
        assert cur_frame is not None
        frame = cur_frame
        _frame_cnt = self._frame_cnt
        while _frame_cnt > 0:
            frame = cur_frame.f_back
            assert frame is not None
            cur_frame = frame
            _frame_cnt -= 1
        calling_frame = cur_frame
        assert calling_frame is not None
        self.target_frames.discard(calling_frame)

    def _is_internal_frame(self, frame: FrameType):
        return frame.f_code.co_filename in self._inner_frame_fnames

    def trace_call_func(self, frame: FrameType, event, arg):
        if not (frame in self.target_frames):
            if self._is_internal_frame(frame):
                return None
        # we only handle methods and global functions.
        self.result_call_stack.append(
            FrameEventBase(
                type=TraceEventType.Call,
                qualname=frame.f_code.co_name,
                filename=frame.f_code.co_filename,
                lineno=frame.f_lineno,
            ))
        if not self._filter_frame(frame):
            return None
        if event == "call":
            # stack = self.thread_local.original_trace_functions
            # sys.settrace(stack[-1])

            cur_locals = frame.f_locals
            target_arg_name = self._trace_info.expr
            assert isinstance(target_arg_name, str)
            self._expr_found = True

            if target_arg_name in cur_locals:
                self.result_line = CursorFrameLineResult(
                    type=TraceEventType.Line,
                    qualname=frame.f_code.co_name,
                    filename=frame.f_code.co_filename,
                    lineno=frame.f_lineno,
                    eval_result=cur_locals[target_arg_name])
                return None
            else:
                return None
        return None

    def trace_line_func(self, frame: FrameType, event, arg):
        if not (frame in self.target_frames):
            if self._is_internal_frame(frame):
                return None
        # we only handle methods and global functions.
        if event == "call":
            self.result_call_stack.append(
                FrameEventBase(
                    type=TraceEventType.Call,
                    qualname=frame.f_code.co_name,
                    filename=frame.f_code.co_filename,
                    lineno=frame.f_lineno,
                ))
        if not self._filter_frame(frame):
            return None
        time_to_eval = False
        if event == "line" and frame.f_lineno == self._trace_info.lineno:
            # delay eval to next event because we want to eval assign target.
            if self._trace_info.assign_lineno_range is not None:
                self._trace_cur_assign_range = self._trace_info.assign_lineno_range
            else:
                time_to_eval = True
        elif self._trace_cur_assign_range is not None and (
                frame.f_lineno > self._trace_cur_assign_range[1]
                or event == "return"):
            time_to_eval = True
            self._trace_cur_assign_range = None
        # print("?", event, frame.f_lineno, self._trace_info.lineno, time_to_eval)
        if (event == "line" or event == "return") and time_to_eval:
            cur_locals = frame.f_locals
            expr = self._trace_info.expr
            assert isinstance(expr, ast.AST)
            expr_str = ast.unparse(expr)
            self._expr_found = True
            # stack = self.thread_local.original_trace_functions
            # sys.settrace(stack[-1])
            try:
                cobj = compile(expr_str, "<ast>", "eval")
                eval_result = eval(cobj, frame.f_globals, cur_locals)
                self.result_line = CursorFrameLineResult(
                    type=TraceEventType.Line,
                    qualname=frame.f_code.co_name,
                    filename=frame.f_code.co_filename,
                    lineno=frame.f_lineno,
                    eval_result=eval_result)
                return None
            except:
                traceback.print_exc()
                return None
        else:
            return self.trace_line_func


class CursorFuncTracer:
    def __init__(self, ) -> None:
        self._ast_cache = AstCache()
        self._cached_trace_file: Dict[str, Set[Path]] = {}

    def _get_func_id(self, func: Callable):
        if inspect.ismethod(func):
            func = func.__func__
        func = inspect.unwrap(func)
        return get_module_id_of_type(type(func))

    def prepare_func_trace(
            self,
            func: Callable,
            args: Tuple,
            kwargs: Dict[str, Any]) -> List[FrameEventBase]:
        func_id = self._get_func_id(func)
        with CallTracerContext() as ctx:
            func(*args, **kwargs)
        self._cached_trace_file[func_id] = {
            Path(o.filename).resolve()
            for o in ctx.result_call_stack
        }
        return ctx.result_call_stack

    def run_trace_from_code_range(
            self,
            func: Callable,
            args: Tuple,
            kwargs: Dict[str, Any],
            path: str,
            code_range: Tuple[int, int, int, int]) -> Optional[CodeFragTracerResult]:
        func_id = self._get_func_id(func)
        if func_id in self._cached_trace_file:
            if Path(path).resolve() not in self._cached_trace_file[func_id]:
                return None

        trace_info = get_trace_infos_from_coderange_item(
            self._ast_cache.query_path(Path(path)), code_range)
        print(trace_info)
        # breakpoint()
        if trace_info is not None:
            with TracerContext(trace_info) as ctx:
                func(*args, **kwargs)
            self._cached_trace_file[func_id] = {
                Path(o.filename).resolve()
                for o in ctx.result_call_stack
            }
            if ctx.result_line is not None:
                return CodeFragTracerResult(frame_results=ctx.result_call_stack,
                                          line_result=ctx.result_line)
        return None


# if __name__ == "__main__":
#     import tensorpc.core.serviceunit
#     from tensorpc.core.serviceunit import ServiceUnit
#     su = ServiceUnit("tensorpc.services.collection::FileOps", {})
#     tracer = CursorFuncTracer()
#     path = Path(tensorpc.core.serviceunit.__file__).resolve()
#     code_range = (1340, 43, 1340, 46)
#     res = tracer.run_trace_from_code_range(su.get_service_unit_ids, [], {}, str(path), code_range)
#     print(res)
#     pass
