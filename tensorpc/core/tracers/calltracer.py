"""tracer that used for cursor selection
"""

import ast
from dataclasses import dataclass
import enum
import inspect
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from types import FrameType
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Type, Union

from tensorpc.core.astex.astcache import AstCache, AstCacheItem
from tensorpc.core.inspecttools import get_co_qualname_from_frame
from tensorpc.core.moduleid import get_module_id_of_type
from .core import TraceEventType, FrameEventBase, FrameEventCall

THREAD_GLOBALS = threading.local()


class CallTracerContext(object):
    def __init__(self,
                 trace_call_only: bool = True,
                 max_depth: int = 10000,
                 traced_folders: Optional[Set[Union[str, Path]]] = None,
                 frame_isvalid_func: Optional[Callable[[FrameType, int],
                                                       bool]] = None,
                 *,
                 _frame_cnt: int = 1):
        self.target_frames: Set[FrameType] = set()
        self.thread_local = threading.local()
        # code type -> (should trace, filter_res)
        self._frame_cnt = _frame_cnt
        self._inner_frame_fnames: Set[str] = set(
            [CallTracerContext.__enter__.__code__.co_filename])
        self.result_call_stack: List[FrameEventCall] = []
        self._frame_isvalid_func = frame_isvalid_func
        self._trace_call_only = trace_call_only
        self._max_depth = max_depth

        traced_folders_absolute: List[str] = []
        if traced_folders is not None:
            for folder in traced_folders:
                if isinstance(folder, str):
                    traced_folders_absolute.append(os.path.abspath(folder))
                else:
                    traced_folders_absolute.append(str(folder.absolute()))
        self._traced_folders_tuple = tuple(traced_folders_absolute)

    def __enter__(self):
        THREAD_GLOBALS.__dict__.setdefault('depth', 0)
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
        if self._trace_call_only:
            trace_fn = self.trace_call_func
        else:
            trace_fn = self.trace_call_ret_func
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
        res = frame.f_code.co_filename in self._inner_frame_fnames or frame.f_code.co_name.startswith("<")
        if self._traced_folders_tuple:
            res |= not frame.f_code.co_filename.startswith(
                self._traced_folders_tuple)
        return res 

    def trace_call_func(self, frame: FrameType, event, arg):
        if not (frame in self.target_frames):
            if self._is_internal_frame(frame):
                return None
        if self._frame_isvalid_func is not None and not self._frame_isvalid_func(
                frame, -1):
            return None
        self.result_call_stack.append(
            FrameEventCall(
                type=TraceEventType.Call,
                qualname=frame.f_code.co_name,
                filename=frame.f_code.co_filename,
                lineno=frame.f_lineno,
            ))
        return None

    def _trace_ret_only_func(self, frame: FrameType, event, arg):
        if event == "return":
            THREAD_GLOBALS.depth -= 1
            self.result_call_stack.append(
                FrameEventCall(
                    type=TraceEventType.Return,
                    qualname=get_co_qualname_from_frame(frame),
                    filename=frame.f_code.co_filename,
                    lineno=frame.f_lineno,
                    timestamp=time.time_ns(),
                    depth=THREAD_GLOBALS.depth,
                ))

    def trace_call_ret_func(self, frame: FrameType, event, arg):
        if not (frame in self.target_frames):
            if self._is_internal_frame(frame):
                return None
        if event == "call":
            THREAD_GLOBALS.depth += 1
            if THREAD_GLOBALS.depth >= self._max_depth:
                return None
            if self._frame_isvalid_func is not None and not self._frame_isvalid_func(
                    frame, THREAD_GLOBALS.depth):
                return None
            self.result_call_stack.append(
                FrameEventCall(
                    type=TraceEventType.Call,
                    qualname=get_co_qualname_from_frame(frame),
                    filename=frame.f_code.co_filename,
                    lineno=frame.f_lineno,
                    depth=THREAD_GLOBALS.depth,
                    timestamp=time.time_ns(),
                    caller_lineno=-1 if frame.f_back is None else frame.f_back.f_lineno
                ))
            frame.f_trace_lines = False
        return self._trace_ret_only_func
