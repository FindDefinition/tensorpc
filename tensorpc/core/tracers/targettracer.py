"""tracer that used for cursor selection
"""

from pathlib import Path
import sys
import threading
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
                 stop_callback: Optional[Callable[[], Any]] = None,
                 max_depth: int = 10000,
                 max_num_variable: int = 1):
        assert Python3_11AndLater, "only support python >= 3.11 due to requirement of co_qualname"
        assert max_num_variable > 0
        self._max_depth = max_depth

        self._target_fname = target_fname
        self._target_co_qualname = target_co_qualname
        self._callback = callback
        self._stop_callback = stop_callback
        self._original_trace_func = None
        self._max_num_variable = max_num_variable
        self._max_num_variable_state = max_num_variable

    def start(self):
        self._original_trace_func = sys.gettrace()
        THREAD_GLOBALS.__dict__.setdefault('depth', 0)
        self._max_num_variable_state = self._max_num_variable
        sys.settrace(self._trace_func)

    def stop(self):
        sys.settrace(self._original_trace_func)
        if self._max_num_variable_state > 0:
            if self._stop_callback is not None:
                self._stop_callback()
        self._original_trace_func = None
        self._max_num_variable_state = 1

    def _trace_func(self, frame: FrameType, event, arg):
        if event == "call":
            # THREAD_GLOBALS.depth += 1
            # if THREAD_GLOBALS.depth > self._max_depth:
            #     THREAD_GLOBALS.depth -= 1
            #     return None
            if frame.f_code.co_filename != self._target_fname or get_co_qualname_from_frame(frame) != self._target_co_qualname:
                return None 
        elif event == "return":
            # THREAD_GLOBALS.depth -= 1
            if frame.f_code.co_filename == self._target_fname and get_co_qualname_from_frame(frame) == self._target_co_qualname:
                
                if self._max_num_variable_state > 0:
                    self._callback(frame)
                    self._max_num_variable_state -= 1
                if self._max_num_variable_state == 0:
                    self.stop()
                    if self._stop_callback is not None:
                        self._stop_callback()
            else:
                return None
        return self._trace_func
