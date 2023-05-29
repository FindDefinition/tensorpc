from dataclasses import dataclass
import enum
from pathlib import Path
import sys
import inspect
import threading
from types import FrameType
from typing import Any, Callable, Dict, Mapping, Optional, Set, Tuple, Type
from tensorpc import compat 

THREAD_GLOBALS = threading.local()

class TraceType(enum.Enum):
    Call = 0
    Return = 1


@dataclass
class FrameResult:
    type: TraceType
    qualname: str
    filename: str
    lineno: int
    local_vars: Mapping[str, Any]
    # depth only available when trace return (slower mode).
    depth: int = -1


class Tracer(object):
    """A simple tracer for python functions.
    Reference: https://github.com/cool-RR/PySnooper/blob/1.1.1/pysnooper/tracer.py
    filters: 
    1. class base types 
    2. method/function names
    3. include folders
    """
    def __init__(self,
                 callback: Callable[[FrameResult], Any],
                 depth: int = -1,
                 traced_types: Optional[Tuple[Type]] = None,
                 traced_names: Optional[Set[str]] = None,
                 traced_folders: Optional[Set[str]] = None,
                 trace_return: bool = True):
        self.target_frames: Set[FrameType] = set()
        self.thread_local = threading.local()
        self.depth = depth
        self.traced_types = traced_types
        self.traced_names = traced_names
        self.traced_folders: Optional[Set[Path]] = None
        self.trace_return = trace_return
        self.callback = callback
        if traced_folders is not None:
            self.traced_folders = set(
                Path(folder) for folder in traced_folders)


    def _filter_frame(self, frame: FrameType):
        is_traced_types = True
        is_traced_names = True
        is_traced_folders = True
        if self.traced_types is not None and "self" in frame.f_locals:
            is_traced_types = isinstance(frame.f_locals["self"],self.traced_types)
        if self.traced_names is not None:
            is_traced_names = frame.f_code.co_name in self.traced_names
        if self.traced_folders is not None:
            code_path = Path(frame.f_code.co_filename)
            found = False
            for candidate in self.traced_folders:
                if compat.is_relative_to(code_path, candidate):
                    found = True
                    break
            is_traced_folders = found
        return is_traced_types and is_traced_names and is_traced_folders

    def __enter__(self):
        THREAD_GLOBALS.__dict__.setdefault('depth', 0)

        current_frame = inspect.currentframe()
        assert current_frame is not None
        calling_frame = current_frame.f_back
        assert calling_frame is not None
        trace_fn = self.trace_return_func
        if not self.trace_return:
            trace_fn = self.trace_call_func
        if not self._is_internal_frame(calling_frame):
            calling_frame.f_trace = trace_fn
            self.target_frames.add(calling_frame)

        stack = self.thread_local.__dict__.setdefault(
            'original_trace_functions', [])
        stack.append(sys.gettrace())
        sys.settrace(trace_fn)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        stack = self.thread_local.original_trace_functions
        sys.settrace(stack.pop())
        current_frame = inspect.currentframe()
        assert current_frame is not None
        calling_frame = current_frame.f_back
        assert calling_frame is not None
        self.target_frames.discard(calling_frame)

    def _is_internal_frame(self, frame: FrameType):
        return frame.f_code.co_filename == Tracer.__enter__.__code__.co_filename

    def trace_lite(self, frame: FrameType, event, arg):
        if event == 'return':
            THREAD_GLOBALS.depth -= 1
            self.callback(self._get_frame_result(TraceType.Return, frame, THREAD_GLOBALS.depth))

    def _get_frame_result(self, trace_type: TraceType, frame: FrameType, depth: int=-1):
        qname = frame.f_code.co_name
        if "self" in frame.f_locals:
            qname = type(frame.f_locals["self"]).__qualname__ + "." + qname
        return FrameResult(
            type=trace_type,
            qualname=qname,
            filename=frame.f_code.co_filename,
            lineno=frame.f_lineno,
            local_vars=frame.f_locals.copy(),
            depth=depth,
        )

    def trace_call_func(self, frame: FrameType, event, arg):
        if not (frame in self.target_frames):
            if self.depth == 1:
                # We did the most common and quickest check above, because the
                # trace function runs so incredibly often, therefore it's
                # crucial to hyper-optimize it for the common case.
                return None
            elif self._is_internal_frame(frame):
                return None
            else:
                if self.depth > 0:
                    _frame_candidate = frame
                    for i in range(1, self.depth):
                        _frame_candidate = _frame_candidate.f_back
                        if _frame_candidate is None:
                            return None
                        elif _frame_candidate in self.target_frames:
                            break
                    else:
                        return None
        # we only handle methods and global functions.
        if not self._filter_frame(frame):
            return None
        if event == "call":
            self.callback(self._get_frame_result(TraceType.Call, frame))
        return None

    def trace_return_func(self, frame: FrameType, event, arg):
        if not (frame in self.target_frames):
            if self.depth == 1:
                # We did the most common and quickest check above, because the
                # trace function runs so incredibly often, therefore it's
                # crucial to hyper-optimize it for the common case.
                return None
            elif self._is_internal_frame(frame):
                return None
            else:
                if self.depth > 0:
                    _frame_candidate = frame
                    for i in range(1, self.depth):
                        _frame_candidate = _frame_candidate.f_back
                        if _frame_candidate is None:
                            return None
                        elif _frame_candidate in self.target_frames:
                            break
                    else:
                        return None
        # we only handle methods and global functions.
        if not self._filter_frame(frame):
            return None
        if event == "call":
            self.callback(self._get_frame_result(TraceType.Call, frame, THREAD_GLOBALS.depth))
            THREAD_GLOBALS.depth += 1

        return self.trace_lite
