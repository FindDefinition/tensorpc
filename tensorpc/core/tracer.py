from dataclasses import dataclass
import enum
from pathlib import Path
import pickle
import sys
import inspect
import threading
import time
from types import CodeType, FrameType
from typing import Any, Callable, Dict, Mapping, Optional, Set, Tuple, Type
from tensorpc import compat
from tensorpc.constants import TENSORPC_FILE_NAME_PREFIX 

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
    module_qname: str = ""
    timestamp: int = -1

    def get_unique_id(self):
        return f"{self.filename}@:{self.lineno}{self.qualname}"


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
                 traced_types: Optional[Tuple[Type]] = None,
                 traced_names: Optional[Set[str]] = None,
                 traced_folders: Optional[Set[str]] = None,
                 trace_return: bool = True,
                 depth: int = 5,
                 ignored_names: Optional[Set[str]] = None,
                 *,
                 _frame_cnt: int = 1):
        self.target_frames: Set[FrameType] = set()
        self.thread_local = threading.local()
        self.depth = depth
        self.traced_types = traced_types
        self.traced_names = traced_names
        self.ignored_names = ignored_names
        self._tensorpc_prefix = f"<{TENSORPC_FILE_NAME_PREFIX}"
        self.keep_one_frame_for_ignored: bool = True
        # code type -> (should trace, filter_res)
        self.code_res: Dict[CodeType, Tuple[bool, bool]] = {}
        self.traced_folders: Optional[Set[Path]] = None
        self.trace_return = trace_return
        self.callback = callback
        if traced_folders is not None and len(traced_folders) > 0:
            self.traced_folders = set(
                Path(folder) for folder in traced_folders)
        self._frame_cnt = _frame_cnt

        self._inner_frame_fnames: Set[str] = set([Tracer.__enter__.__code__.co_filename])

    def _filter_frame(self, frame: FrameType):
        if frame.f_code in self.code_res:
            return self.code_res[frame.f_code][0]
        is_traced_types = True
        is_traced_names = True
        is_traced_folders = True
        co_name = frame.f_code.co_name
        # TODO better handle tensorpc scripts
        if frame.f_code.co_filename.startswith(self._tensorpc_prefix):
            self.code_res[frame.f_code] = (True, True)
            return True 
        if co_name.startswith("<") and co_name.endswith(">"):
            # ignore all comp frame such as <listcomp>
            # listcomp frame will be removed in python 3.12
            self.code_res[frame.f_code] = (False, False)
            return False 
        # TODO better check
        keep_one_frame = False
        if co_name == "__getattr__" or co_name == "__setattr__" :
            self.code_res[frame.f_code] = (False, False)
            return False 
        if self.traced_types is not None and "self" in frame.f_locals:
            is_traced_types = isinstance(frame.f_locals["self"],self.traced_types)
        if self.traced_names is not None:
            is_traced_names = frame.f_code.co_name in self.traced_names
        if self.ignored_names is not None:
            if frame.f_code.co_name in self.ignored_names:
                self.code_res[frame.f_code] = (False, False)
                return False 
        if self.traced_folders is not None:
            code_path = Path(frame.f_code.co_filename)
            found = False
            for candidate in self.traced_folders:
                if compat.is_relative_to(code_path, candidate):
                    found = True
                    break
            is_traced_folders = found
            if not found:
                # keep external trace for one depth
                back = frame.f_back
                if self.keep_one_frame_for_ignored and back is not None and back.f_code in self.code_res and self.code_res[back.f_code][1]:
                    keep_one_frame = True
        res = is_traced_types and is_traced_names and is_traced_folders
        filter_res = res 
        if keep_one_frame:
            filter_res = keep_one_frame
        self.code_res[frame.f_code] = (filter_res, res)
        return res

    def __enter__(self):
        THREAD_GLOBALS.__dict__.setdefault('depth', 0)

        cur_frame = inspect.currentframe()
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

    def trace_lite(self, frame: FrameType, event, arg):
        # print(frame.f_code.co_name, event, frame.f_lineno)
        if event == 'return':
            THREAD_GLOBALS.depth -= 1
            # print(event, frame.f_code.co_name, "THREAD_GLOBALS.depth", THREAD_GLOBALS.depth)
            self.callback(self.get_frame_result(TraceType.Return, frame, THREAD_GLOBALS.depth))

    @staticmethod
    def get_frame_result(trace_type: TraceType, frame: FrameType, depth: int=-1):
        qname = frame.f_code.co_name
        if sys.version_info[:2] >= (3, 11):
            qname = frame.f_code.co_qualname # type: ignore
        else:
            if "self" in frame.f_locals:
                qname = type(frame.f_locals["self"]).__qualname__ + "." + qname                
        module = inspect.getmodule(frame)
        module_qname = ""
        if module is not None:
            module_qname = module.__name__
        return FrameResult(
            type=trace_type,
            qualname=qname,
            filename=frame.f_code.co_filename,
            lineno=frame.f_lineno,
            local_vars=frame.f_locals.copy(),
            depth=depth,
            module_qname=module_qname,
            timestamp=time.time_ns(),
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
            self.callback(self.get_frame_result(TraceType.Call, frame))
        return None

    def trace_return_func(self, frame: FrameType, event, arg):
        # print(event, frame.f_code.co_name, self.target_frames, frame.f_code.co_filename, self._is_internal_frame(frame))

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
        # print(event, frame.f_code.co_name)
        if not self._filter_frame(frame):
            return None
        if event == "call":
            self.callback(self.get_frame_result(TraceType.Call, frame, THREAD_GLOBALS.depth))
            # print("THREAD_GLOBALS.depth CALL", THREAD_GLOBALS.depth)

            THREAD_GLOBALS.depth += 1

        return self.trace_lite

