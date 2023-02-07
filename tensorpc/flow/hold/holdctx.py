import ast
import asyncio
from concurrent.futures import thread
import importlib
from pathlib import Path
import inspect
from runpy import run_path
from types import ModuleType
from typing import AsyncIterator, Generator, Iterator, Optional, List, Dict, TypeVar, Union, Any, Callable
import contextlib
import time
import typing
from . import funcid
import traceback
import watchdog.events
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

_WATCHDOG_MODIFY_EVENT_TYPES = Union[watchdog.events.DirModifiedEvent,
                                     watchdog.events.FileModifiedEvent]
T = TypeVar("T")

class _WatchDogForHoldFile(FileSystemEventHandler):

    def __init__(self,
                 key: str,
                 locals_hold: Dict[str, Any],
                 globals_hold: Dict[str, Any],
                 module: Optional[ModuleType],
                 prev_source: str,
                 prev_hold_source: str,
                 prev_hold_except_source: str,
                 ob: Observer,
                 local_func_id: str,
                 shutdown_ev: Optional[asyncio.Event] = None,
                 loop: Optional[asyncio.AbstractEventLoop] = None,
                 reload_file: bool = True) -> None:
        super().__init__()
        self.local_func_id = local_func_id
        self.loop = loop
        self.shutdown_ev = shutdown_ev
        self.ob = ob
        self.key = key
        self.prev_source = prev_source
        self.prev_hold_source = prev_hold_source
        self.prev_hold_except_source = prev_hold_except_source
        self.reload_file = reload_file
        self.module = module
        self.locals_hold = locals_hold
        self.globals_hold = globals_hold

    def _exit(self):
        self.ob.stop()
        if self.shutdown_ev is not None:
            self.shutdown_ev.clear()

    @staticmethod
    def _extract_hold_body_lines(func_node: Union[ast.FunctionDef,
                                                  ast.AsyncFunctionDef],
                                 hold_node: ast.With, source_lines: List[str]):
        func_first_lineno = func_node.lineno
        deco_list = func_node.decorator_list
        # fix lineno to match inspect
        if len(deco_list) > 0:
            func_first_lineno = min([d.lineno for d in deco_list])
        func_end_lineno = func_node.body[-1].lineno

        with_body_first_lineno = hold_node.body[0].lineno
        with_body_first_col_offset = hold_node.body[0].col_offset
        with_end_lineno = hold_node.body[-1].lineno
        for n in ast.walk(hold_node.body[-1]):
            if hasattr(n, "lineno"):
                with_end_lineno = max(with_end_lineno, n.lineno)
        # extract body code
        lines = source_lines[with_body_first_lineno - 1:with_end_lineno].copy()
        lines = [l[with_body_first_col_offset:].rstrip() for l in lines]
        lines = funcid.clean_source_code(lines)
        hold_source = "\n".join(lines)
        # TODO clean comments
        lines_except_hold = source_lines[:func_first_lineno - 1].copy()
        lines_except_hold += source_lines[func_end_lineno:].copy()
        lines_except_hold = funcid.clean_source_code(lines_except_hold)
        source_except_hold = "\n".join(lines_except_hold)
        return hold_source, source_except_hold

    def _run_hold_ctx(self, path: str, first_time: bool):
        with open(str(path), "r") as f:
            file_source = f.read()
        if file_source == self.prev_source and not first_time:
            return
        try:
            tree = ast.parse(file_source)
        except SyntaxError:
            traceback.print_exc()
            return
        func_node: Optional[Union[ast.FunctionDef,
                                  ast.AsyncFunctionDef]] = None
        # find previous local fid
        toplevels = funcid.get_toplevel_func_node(tree)
        for func_node_can, ns in toplevels:
            local_fid_can = ".".join([n.name
                                      for n in ns] + [func_node_can.name])
            if local_fid_can == self.local_func_id:
                func_node = func_node_can
                break
        if func_node is None:
            self._exit()
            return
        hold_context_node, hold_key = _find_hold_ctx_node(func_node, self.key)
        if hold_context_node is None:
            self._exit()
            return
        self.prev_source = file_source
        source_lines = file_source.split("\n")
        hold_source, source_except_hold = self._extract_hold_body_lines(
            func_node, hold_context_node, source_lines)
        code_block_equal = hold_source == self.prev_hold_source
        code_except_hold_equal = self.prev_hold_except_source == source_except_hold
        if not code_except_hold_equal and self.reload_file:
            if self.module is None:
                mod_dict = run_path(path)
            else:
                self.module = importlib.reload(self.module)
                mod_dict = self.module.__dict__
            self.globals_hold = mod_dict
        if code_block_equal:
            return
        self.prev_hold_except_source = source_except_hold
        self.prev_hold_source = hold_source
        try:
            exec(hold_source, self.globals_hold, self.locals_hold)
        except Exception as e:
            traceback.print_exc()

    def on_modified(self, ev: _WATCHDOG_MODIFY_EVENT_TYPES):
        if not isinstance(ev, watchdog.events.FileModifiedEvent):
            # this shouldn't happen
            return
        path = ev.src_path
        return self._run_hold_ctx(path, False)


def _find_hold_ctx_node(func_node: Union[ast.FunctionDef,
                                         ast.AsyncFunctionDef],
                        hold_name: Optional[str] = None):
    hold_context_node: Optional[ast.With] = None
    hold_name_res: Optional[str] = hold_name
    for node in ast.walk(func_node):
        if isinstance(node, ast.With):
            for with_item in node.items:
                # if isinstance(with_item.context_expr, )
                # if isinstance(with_item, ast)
                if isinstance(with_item.context_expr, ast.Call):
                    func_name_parts = funcid.get_attribute_name_parts(
                        with_item.context_expr.func)
                    if func_name_parts[-1] == hold_ctx.__name__:
                        if hold_name is not None:
                            first_argument = funcid.from_constant(
                                with_item.context_expr.args[0])
                            if first_argument == hold_name:
                                hold_context_node = node
                                break
                        else:
                            hold_context_node = node
                            first_argument = funcid.from_constant(
                                with_item.context_expr.args[0])
                            assert isinstance(first_argument, str)
                            hold_name_res = first_argument
                            break
        if hold_context_node is not None:
            break
    return hold_context_node, hold_name_res


@contextlib.contextmanager
def hold_ctx(key: str, reload_file=True, yield_obj: Optional[T] = None, *, _frame_cnt=2, ) -> Iterator[Optional[T]]:
    # capture prev locals
    cur_frame = inspect.currentframe()
    assert cur_frame is not None
    frame = cur_frame
    while _frame_cnt > 0:
        frame = cur_frame.f_back
        assert frame is not None
        cur_frame = frame
        _frame_cnt -= 1
    # get unique function id
    file_name = Path(frame.f_code.co_filename)
    if not file_name.exists():
        print("your file not exist", file_name)
        yield yield_obj
        return
    first_lineno = frame.f_code.co_firstlineno
    _locals = frame.f_locals
    _globals = frame.f_globals
    mod = inspect.getmodule(frame)
    del frame
    with open(str(file_name), "r") as f:
        file_source = f.read()
    try:
        tree = ast.parse(file_source)
    except SyntaxError:
        traceback.print_exc()
        yield yield_obj
        return
    func_node_ns = funcid.find_toplevel_func_node_by_lineno(tree, first_lineno)
    # find with xxx.hold()
    if func_node_ns is None:
        # this shouldn't happen
        yield yield_obj
        return
    func_node, namespaces = func_node_ns
    local_fid = ".".join([n.name for n in namespaces] + [func_node.name])
    hold_context_node, hold_key = _find_hold_ctx_node(func_node)
    if hold_context_node is None:
        raise ValueError(
            "can't find hold context, may be you use var instead of function in with."
        )
    assert hold_key is not None and hold_key == key, f"{key} {hold_key}"
    observer = Observer()
    source_lines = file_source.split("\n")
    hold_source, hold_except_source = _WatchDogForHoldFile._extract_hold_body_lines(
        func_node, hold_context_node, source_lines)
    handler = _WatchDogForHoldFile(hold_key,
                                   _locals,
                                   _globals,
                                   mod,
                                   file_source,
                                   hold_source,
                                   hold_except_source,
                                   observer,
                                   local_fid,
                                   reload_file=reload_file)
    
    yield yield_obj
    observer.schedule(handler, file_name, recursive=False)
    observer.start()
    observer.join()
    return

@contextlib.asynccontextmanager
async def hold_ctx_async(key: str, reload_file=True, yield_obj: Optional[T] = None, *, _frame_cnt=2) -> AsyncIterator[Optional[T]]:
    # capture prev locals
    cur_frame = inspect.currentframe()
    assert cur_frame is not None
    frame = cur_frame
    while _frame_cnt > 0:
        frame = cur_frame.f_back
        assert frame is not None
        cur_frame = frame
        _frame_cnt -= 1
    # get unique function id
    file_name = Path(frame.f_code.co_filename)
    if not file_name.exists():
        print("your file not exist", file_name)
        yield yield_obj
        return
    first_lineno = frame.f_code.co_firstlineno
    _locals = frame.f_locals
    _globals = frame.f_globals
    mod = inspect.getmodule(frame)
    del frame
    with open(str(file_name), "r") as f:
        file_source = f.read()
    try:
        tree = ast.parse(file_source)
    except SyntaxError:
        traceback.print_exc()
        yield yield_obj
        return
    func_node_ns = funcid.find_toplevel_func_node_by_lineno(tree, first_lineno)
    # find with xxx.hold()
    if func_node_ns is None:
        # this shouldn't happen
        yield yield_obj
        return
    func_node, namespaces = func_node_ns
    local_fid = ".".join([n.name for n in namespaces] + [func_node.name])
    hold_context_node, hold_key = _find_hold_ctx_node(func_node)
    if hold_context_node is None:
        raise ValueError(
            "can't find hold context, may be you use var instead of function in with."
        )
    assert hold_key is not None and hold_key == key, f"{key} {hold_key}"
    observer = Observer()
    source_lines = file_source.split("\n")

    hold_source, hold_except_source = _WatchDogForHoldFile._extract_hold_body_lines(
        func_node, hold_context_node, source_lines)
    ev = asyncio.Event()
    yield yield_obj
    handler = _WatchDogForHoldFile(hold_key,
                                   _locals,
                                   _globals,
                                   mod,
                                   file_source,
                                   hold_source,
                                   hold_except_source,
                                   observer,
                                   local_fid,
                                   shutdown_ev=ev,
                                   reload_file=reload_file)
    observer.schedule(handler, file_name, recursive=False)
    observer.start()
    await ev.wait()
    return
