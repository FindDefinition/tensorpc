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
from .core import TraceEventType, FrameEventBase


class CallTracerContext(object):
    def __init__(self,
                 trace_call_only: bool = True,
                 frame_isvalid_func: Optional[Callable[[FrameType],
                                                       bool]] = None,
                 *,
                 _frame_cnt: int = 1):
        self.target_frames: Set[FrameType] = set()
        self.thread_local = threading.local()
        # code type -> (should trace, filter_res)
        self._frame_cnt = _frame_cnt
        self._inner_frame_fnames: Set[str] = set(
            [CallTracerContext.__enter__.__code__.co_filename])
        self.result_call_stack: List[FrameEventBase] = []
        self._frame_isvalid_func = frame_isvalid_func
        self._trace_call_only = trace_call_only

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
        return frame.f_code.co_filename in self._inner_frame_fnames

    def trace_call_func(self, frame: FrameType, event, arg):
        if not (frame in self.target_frames):
            if self._is_internal_frame(frame):
                return None
        if self._frame_isvalid_func is not None and not self._frame_isvalid_func(
                frame):
            return None
        self.result_call_stack.append(
            FrameEventBase(
                type=TraceEventType.Call,
                qualname=frame.f_code.co_name,
                filename=frame.f_code.co_filename,
                lineno=frame.f_lineno,
            ))
        return None

    def _trace_ret_only_func(self, frame: FrameType, event, arg):
        if event == "return":
            self.result_call_stack.append(
                FrameEventBase(
                    type=TraceEventType.Return,
                    qualname=frame.f_code.co_name,
                    filename=frame.f_code.co_filename,
                    lineno=frame.f_lineno,
                ))

    def trace_call_ret_func(self, frame: FrameType, event, arg):
        if not (frame in self.target_frames):
            if self._is_internal_frame(frame):
                return None
        if event == "call":
            if self._frame_isvalid_func is not None and not self._frame_isvalid_func(
                    frame):
                return None
            self.result_call_stack.append(
                FrameEventBase(
                    type=TraceEventType.Call,
                    qualname=frame.f_code.co_name,
                    filename=frame.f_code.co_filename,
                    lineno=frame.f_lineno,
                ))
            frame.f_trace_lines = False
        return self._trace_ret_only_func
