"""A real-time tracer that store events to storage immediately.

Often used to solve hang problem in distributed program (SPMD).
"""

import contextlib
import contextvars
import ast
from dataclasses import dataclass
import enum
from functools import partial
import gzip
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
from tensorpc.core.bgserver import BACKGROUND_SERVER
from tensorpc.core.client import simple_chunk_call, simple_remote_call
from tensorpc.core.inspecttools import get_co_qualname_from_frame
from tensorpc.core.moduleid import get_module_id_of_type
from tensorpc.dbg.serv_names import serv_names


THREAD_GLOBALS = threading.local()

RTTRACER_CONTEXT: contextvars.ContextVar[Optional["RTTracerContext"]] = contextvars.ContextVar('RTTRACER_CONTEXT', default=None)

class ChromeTraceStorage:
    def __init__(self, proc_name: str, enable_lock: bool = True, pid: int = 0, tid: int = 0):
        self._lock = threading.Lock() if enable_lock else None
        self._chrome_trace_events: List[dict] = [
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "tid": tid,
                "args": {
                    "name": f"Proc {proc_name}",
                }
            },
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": tid,
                "args": {
                    "name": "MainThread",
                }
            },
        ]

        self._pid = pid
        self._tid = tid

    def log_begin(self, name: str):
        ts = time.time_ns() / 1000.0
        ev = {
            "name": name,
            "ph": "B",
            "ts": ts,
            "pid": self._pid,
            "tid": self._tid,
        }
        if self._lock is not None:
            with self._lock:
                self._chrome_trace_events.append(ev)
        else:
            self._chrome_trace_events.append(ev)

    def log_end(self, name: str):
        ts = time.time_ns() / 1000.0
        ev = ({
            "name": name,
            "ph": "E",
            "ts": ts,
            "pid": self._pid,
            "tid": self._tid,
        })
        if self._lock is not None:
            with self._lock:
                self._chrome_trace_events.append(ev)
        else:
            self._chrome_trace_events.append(ev)

    def log_instant(self, name: str, args: Any = None):
        ts = time.time_ns() / 1000.0
        ev = ({
            "name": name,
            "ph": "I",
            "ts": ts,
            "pid": self._pid,
            "tid": self._tid,
            "args": args,
        })
        if self._lock is not None:
            with self._lock:
                self._chrome_trace_events.append(ev)
        else:
            self._chrome_trace_events.append(ev)

    def get_trace_result(self):
        if self._lock is not None:
            with self._lock:
                res = self._chrome_trace_events.copy()
        else:
            res = self._chrome_trace_events.copy()
        # parse B/E to X (complete duration event)
        new_results = []
        stack = []
        for ev in res:
            if ev["ph"] == "B":
                stack.append(ev)
            elif ev["ph"] == "E":
                if stack:
                    begin_ev = stack.pop()
                    new_results.append({
                        "name": ev["name"],
                        "ph": "X",
                        "ts": begin_ev["ts"],
                        "dur": ev["ts"] - begin_ev["ts"],
                        "pid": ev["pid"],
                        "tid": ev["tid"],
                    })
            else:
                new_results.append(ev)
        new_results.extend(stack)
        return {
            "traceEvents": new_results
        }
            

class RTTracerContext(object):
    def __init__(self,
                storage: ChromeTraceStorage,
                 *,
                 _frame_cnt: int = 1):
        self.target_frames: Set[FrameType] = set()
        self.thread_local = threading.local()
        # code type -> (should trace, filter_res)
        self._frame_cnt = _frame_cnt
        self._inner_frame_fnames: Set[str] = set(
            [RTTracerContext.__enter__.__code__.co_filename])
        self._chrome_trace_events: List[dict] = []
        self._storage = storage
        self._ctx_token = None

    def __enter__(self):
        assert self._ctx_token is None, "Context already entered"
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
        trace_fn = self.trace_call_ret_func
        if not self._is_internal_frame(calling_frame):
            calling_frame.f_trace = trace_fn
            self.target_frames.add(calling_frame)

        stack = self.thread_local.__dict__.setdefault(
            'original_trace_functions', [])
        stack.append(sys.gettrace())
        sys.settrace(trace_fn)
        self._ctx_token = RTTRACER_CONTEXT.set(self)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self._ctx_token is not None:
            RTTRACER_CONTEXT.reset(self._ctx_token)
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
        res |= frame.f_code.co_filename.endswith("pytree.py")
        return res 

    def _trace_ret_only_func(self, frame: FrameType, event, arg):
        if event == "return":
            name = f"{frame.f_code.co_name}({frame.f_code.co_filename}:{frame.f_lineno})"

            self._storage.log_end(name)

    def trace_call_ret_func(self, frame: FrameType, event, arg):
        if not (frame in self.target_frames):
            if self._is_internal_frame(frame):
                return None
        if event == "call":
            name = f"{frame.f_code.co_name}({frame.f_code.co_filename}:{frame.f_lineno})"
            self._storage.log_begin(name)
            frame.f_trace_lines = False
        return self._trace_ret_only_func

def record_instant_event(name: str, args: Any = None):
    ctx = RTTRACER_CONTEXT.get()
    if ctx is None:
        return
    ctx._storage.log_instant(name, args)

@contextlib.contextmanager
def enter_tracer(key: str, *, process_name: Optional[str] = None, frame_cnt: int = 1, pid: Optional[int] = None):
    if pid is None:
        pid = os.getpid()
    storage = ChromeTraceStorage(key if process_name is None else process_name, pid=pid, tid=threading.get_ident())
    assert BACKGROUND_SERVER.is_started, "you must call breakpoint or init before use this function."
    BACKGROUND_SERVER.execute_service(serv_names.RT_TRACE_SET_STORAGE, key, storage)
    ctx = RTTracerContext(storage, _frame_cnt=frame_cnt)
    with ctx:
        yield

def fetch_trace_result(url: str, key: str, rpc_timeout: int = 10) -> bytes:
    return simple_chunk_call(url, serv_names.RT_TRACE_GET_TRACE_RESULT, key, rpc_timeout=rpc_timeout)
