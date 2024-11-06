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

from tensorpc.core.inspecttools import get_co_qualname_from_frame
from tensorpc.compat import Python3_11AndLater

THREAD_GLOBALS = threading.local()

class TargetTracer(object):
    def __init__(self,
                 target_fname: str,
                 target_co_qualname: str,
                 callback: Callable[[FrameType], Any],
                 max_depth: int = 10000):
        assert Python3_11AndLater, "only support python >= 3.11 due to requirement of co_qualname"
        self._max_depth = max_depth

        self._target_fname = target_fname
        self._target_co_qualname = target_co_qualname
        self._callback = callback
        self._original_trace_func = None

    def start(self):
        self._original_trace_func = sys.gettrace()
        THREAD_GLOBALS.__dict__.setdefault('depth', 0)

        sys.settrace(self._trace_func)

    def stop(self):
        sys.settrace(self._original_trace_func)

    def _trace_func(self, frame: FrameType, event, arg):
        if event == "call":
            THREAD_GLOBALS.depth += 1
            if THREAD_GLOBALS.depth > self._max_depth:
                return None
            if frame.f_code.co_filename == self._target_fname and get_co_qualname_from_frame(frame) == self._target_co_qualname:
                self._callback(frame)
                sys.settrace(None)
            else:
                return None
        elif event == "return":
            THREAD_GLOBALS.depth -= 1
        
        return self._trace_func
