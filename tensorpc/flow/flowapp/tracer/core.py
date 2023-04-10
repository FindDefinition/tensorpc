"""tracer modified from pysnooper
"""

import ast
import collections
import copy
import datetime
import functools
import inspect
import itertools
import logging
import logging.handlers
import os
import re
import sys
import threading
import traceback
import types
from collections import OrderedDict
from pathlib import Path

import opcode

import codeai.utils.codeai_logging as codeai_logging
from codeai import astex
from codeai.core.diff import compare
from codeai.core.prettystr import prettystr

ipython_filename_pattern = re.compile('^<ipython-input-([0-9]+)-.*>$')

logger = codeai_logging.get_logger("codeai.trace")
logger.setLevel(codeai_logging.INFO)
"""
def get_code_from_frame(frame):
    file_name = frame.f_code.co_filename
    first_lineno = frame.f_code.co_firstlineno
    if not Path(file_name).exists():
        return None
    code_objs = astex.core.cached_get_func_class_code_by_ast(file_name)
    for code_obj in code_objs:
        if code_obj.first_lineno == first_lineno:
            return code_obj
    return None
"""


def get_source_from_frame(frame, cache: dict):
    f_globals = frame.f_globals or {}
    module_name = f_globals.get('__name__')
    file_name = frame.f_code.co_filename
    loader = f_globals.get('__loader__')
    cache_key = (module_name, file_name)
    if cache_key in cache:
        return cache[cache_key]
    source = None
    if hasattr(loader, 'get_source'):
        try:
            source = loader.get_source(module_name)
        except ImportError:
            pass
        if source is not None:
            source = source.splitlines()
    if source is None:
        ipython_filename_match = ipython_filename_pattern.match(file_name)
        if ipython_filename_match:
            entry_number = int(ipython_filename_match.group(1))
            try:
                import IPython
                ipython_shell = IPython.get_ipython()
                ((_, _, source_chunk),) = ipython_shell.history_manager. \
                                  get_range(0, entry_number, entry_number + 1)
                source = source_chunk.splitlines()
            except Exception:
                pass
        else:
            path = Path(file_name)
            if path.exists():
                with path.open("r") as f:
                    source = f.read().splitlines()

    res = (file_name, source)
    cache[cache_key] = res
    return res


def get_local_vars(frame):
    code = frame.f_code
    vars_order = code.co_varnames + code.co_cellvars + code.co_freevars + tuple(
        frame.f_locals.keys())
    result_items = [(key, value) for key, value in frame.f_locals.items()]
    result_items.sort(key=lambda key_value: vars_order.index(key_value[0]))
    result_items = filter(lambda x: not isinstance(x[1], types.ModuleType),
                          result_items)
    result = OrderedDict(result_items)
    return result


def get_vars_reprs(local_vars, length_limit=40):
    reprs = {}
    for k, v in local_vars.items():
        string = prettystr(v)
        if len(string) >= length_limit:
            string = string[:length_limit] + "..."
        reprs[k] = string
    return reprs


def print_reprs(reprs):
    res = []
    for k, v in reprs.items():
        res.append("{}={}".format(k, v))
    return ",".join(res)


class Tracer(object):
    """
    TODO:
    """
    def __init__(self, depth=1, use_deepcopy=True, log_file=None):
        self.traced_codes = set()
        self.traced_frames = set()
        self.thread_local = threading.local()
        self.frame_to_local_vars = {}
        self.last_source_path = None
        self.source_cache = {}
        self.frame_to_longest_line = {}
        self.depth = depth
        self.last_line = ""
        self.last_lineno = -1
        self.last_var_msg = ""
        self.use_deepcopy = use_deepcopy
        self.log_file = log_file
        self.log_handler = None

    def __call__(self, func):
        self.traced_codes.add(func.__code__)

        @functools.wraps(func)
        def simple_wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return simple_wrapper

    def _is_tracer_frame(self, frame):
        # tracer's frame, when you use tracer as function decorator, return True.
        return frame.f_code.co_filename == Tracer.__enter__.__code__.co_filename

    def __enter__(self):
        if self.log_file is not None:
            self.log_handler = logging.handlers.WatchedFileHandler(
                self.log_file)
            logger.addHandler(self.log_handler)
        logger.info("Enter trace")
        caller_frame = inspect.currentframe().f_back
        if not self._is_tracer_frame(caller_frame):
            caller_frame.f_trace = self.trace
            self.traced_frames.add(caller_frame)

        stack = self.thread_local.__dict__.setdefault(
            'original_trace_functions', [])
        stack.append(sys.gettrace())
        sys.settrace(self.trace)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        stack = self.thread_local.original_trace_functions
        sys.settrace(stack.pop())
        caller_frame = inspect.currentframe().f_back
        self.traced_frames.discard(caller_frame)
        self.frame_to_local_vars.pop(caller_frame, None)
        logger.info("Exit trace")
        if self.log_file is not None:
            logger.removeHandler(self.log_handler)
            self.log_handler = None
        # if self.last_line != "":
        #     print(self.last_line)

    def will_exit(self, frame):
        res = frame.f_code.co_filename == Tracer.__exit__.__code__.co_filename
        res &= frame.f_lineno == Tracer.__exit__.__code__.co_firstlineno
        return res

    def _check_is_traced_frame(self, frame):
        if frame.f_code not in self.traced_codes and frame not in self.traced_frames:
            if self.depth == 1:
                # We did the most common and quickest check above, because the
                # trace function runs so incredibly often, therefore it's
                # crucial to hyper-optimize it for the common case.
                return False
            elif self._is_tracer_frame(frame):
                return False
            else:
                _frame_candidate = frame
                for i in range(1, self.depth):
                    _frame_candidate = _frame_candidate.f_back
                    if _frame_candidate is None:
                        return False
                    elif _frame_candidate.f_code in self.traced_codes or _frame_candidate in self.traced_frames:
                        break
                else:
                    return False
        return True

    def trace(self, frame, event, arg):
        """
        Args:
            arg: when event is return, arg will be returned value
                or None if exception raised.
                when event is exception, arg will be (exception, value, traceback)
        """
        # print("??", frame.f_lineno, frame.f_code.co_name)
        if not self._check_is_traced_frame(frame):
            return None
        lineno = frame.f_lineno
        print(lineno)
        source_path, source = get_source_from_frame(frame, self.source_cache)
        if source is None:
            source_line = "NOT FOUND"
        else:
            source_line = source[lineno - 1]
        if self.last_source_path != source_path:
            logger.info("Enter {}({}):".format(source_path, lineno))
            self.last_source_path = source_path

        # print("next line", source_line)
        old_vars = {}
        if frame in self.frame_to_local_vars:
            old_vars = self.frame_to_local_vars[frame]
        local_vars = get_local_vars(frame)
        # print(frame.f_code.co_firstlineno, lineno, prettystr(local_vars))
        if self.use_deepcopy:
            # TODO: handle unpickleable object
            local_vars = copy.deepcopy(local_vars)
        self.frame_to_local_vars[frame] = local_vars
        var_msg = ""
        new_vars = {k: v for k, v in local_vars.items() if k not in old_vars}
        # TODO: support repr-based compare
        # TODO: support display structed difference in modified result
        if self.use_deepcopy:
            modified_vars = {
                k: local_vars[k]
                for k, v in old_vars.items() if not compare(local_vars[k], v)
            }
        else:
            # use 'is' operator is enough. impossible to detect inplace change if don't use deepcopy
            # modified_vars = {}
            modified_vars = {
                k: local_vars[k]
                for k, v in old_vars.items() if local_vars[k] is not v
            }
        if len(new_vars) > 0:
            var_msg += "|New: " + print_reprs(get_vars_reprs(new_vars))
        if len(modified_vars) > 0:
            var_msg += "|Modified: " + print_reprs(
                get_vars_reprs(modified_vars))
        self.last_var_msg = var_msg
        # print(lineno, line_to_be_printed)
        if source is not None and event == 'call' and source_line.lstrip(
        ).startswith('@'):
            # If a function decorator is found, skip lines until an actual
            # function definition is found.
            for candidate_line_no in itertools.count(lineno):
                try:
                    candidate_source_line = source[candidate_line_no - 1]
                except IndexError:
                    # End of source file reached without finding a function
                    # definition. Fall back to original source line.
                    break

                if candidate_source_line.lstrip().startswith('def'):
                    # Found the def line!
                    lineno = candidate_line_no
                    source_line = candidate_source_line
                    break
        line_to_be_printed = self.last_line
        self.last_line = source_line
        lineno_to_be_printed = self.last_lineno
        self.last_lineno = lineno
        # print(lineno)
        code_byte = frame.f_code.co_code[frame.f_lasti]
        # print(opcode.opname[code_byte])
        if not isinstance(code_byte, int):
            code_byte = ord(code_byte)
        ended_by_exception = (event == 'return' and arg is None
                              and (opcode.opname[code_byte]
                                   not in ('RETURN_VALUE', 'YIELD_VALUE')))
        if ended_by_exception:
            logger.info("Call ended by exception")
        else:
            if self.will_exit(frame):
                logger.info("({}){}   {}".format(lineno_to_be_printed,
                                                 line_to_be_printed,
                                                 "|last line can't be traced"))
            else:
                logger.info("({}){}   {}".format(lineno_to_be_printed,
                                                 line_to_be_printed, var_msg))
        if event == 'return':
            del self.frame_to_local_vars[frame]
            if not ended_by_exception:
                logger.info("Return {}".format(prettystr(arg)))
        if event == 'exception':
            exception, value, tb = arg
            logger.info("raise {}: {}".format(exception, value))

        return self.trace


def is_exception(frame, event, arg):
    # https://stackoverflow.com/a/12800909/2482744
    code_byte = frame.f_code.co_code[frame.f_lasti]
    if not isinstance(code_byte, int):
        code_byte = ord(code_byte)
    ended_by_exception = (event == 'return' and arg is None
                          and (opcode.opname[code_byte]
                               not in ('RETURN_VALUE', 'YIELD_VALUE')))
    return ended_by_exception


class TracerV3(object):
    def __init__(self, depth=1, use_deepcopy=True, log_file=None):
        self.traced_codes = set()
        self.traced_frames = set()
        self.thread_local = threading.local()
        self.frame_to_vars = {}
        self.frame_to_values = {}
        self.last_source_path = None
        self.source_cache = {}
        self.frame_to_longest_line = {}
        self.depth = depth
        self.last_line = ""
        self.last_lineno = -1
        self.last_var_msg = ""
        self.use_deepcopy = use_deepcopy
        self.log_file = log_file
        self.log_handler = None

    def __call__(self, func):
        self.traced_codes.add(func.__code__)

        @functools.wraps(func)
        def simple_wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return simple_wrapper

    def _is_tracer_frame(self, frame):
        # tracer's frame, when you use tracer as function decorator, return True.
        return frame.f_code.co_filename == Tracer.__enter__.__code__.co_filename

    def __enter__(self):
        caller_frame = inspect.currentframe().f_back
        if not self._is_tracer_frame(caller_frame):
            caller_frame.f_trace = self.trace
            self.traced_frames.add(caller_frame)

        stack = self.thread_local.__dict__.setdefault(
            'original_trace_functions', [])
        stack.append(sys.gettrace())
        sys.settrace(self.trace)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        stack = self.thread_local.original_trace_functions
        sys.settrace(stack.pop())
        caller_frame = inspect.currentframe().f_back
        self.traced_frames.discard(caller_frame)
        self.frame_to_vars.pop(caller_frame, None)
        self.frame_to_values.pop(caller_frame, None)

        logger.info("Exit trace")
        if self.log_file is not None:
            logger.removeHandler(self.log_handler)
            self.log_handler = None
        # if self.last_line != "":
        #     print(self.last_line)

    def will_exit(self, frame):
        res = frame.f_code.co_filename == Tracer.__exit__.__code__.co_filename
        res &= frame.f_lineno == Tracer.__exit__.__code__.co_firstlineno
        return res

    def _check_is_traced_frame(self, frame):
        if frame.f_code not in self.traced_codes and frame not in self.traced_frames:
            if self.depth == 1:
                # We did the most common and quickest check above, because the
                # trace function runs so incredibly often, therefore it's
                # crucial to hyper-optimize it for the common case.
                return False
            elif self._is_tracer_frame(frame):
                return False
            else:
                _frame_candidate = frame
                for i in range(1, self.depth):
                    _frame_candidate = _frame_candidate.f_back
                    if _frame_candidate is None:
                        return False
                    elif _frame_candidate.f_code in self.traced_codes or _frame_candidate in self.traced_frames:
                        break
                else:
                    return False
        return True

    def update_vars_and_values(self, frame):
        local_vars = get_local_vars(frame)
        local_values = {k: prettystr(v) for k, v in local_vars.items()}
        old_vars = {}
        old_values = {}
        if frame in self.frame_to_vars:
            old_vars = self.frame_to_vars[frame]
            old_values = self.frame_to_values[frame]
        new_vars = {k: v for k, v in local_vars.items() if k not in old_vars}
        new_values = {
            k: v
            for k, v in local_values.items() if k not in old_values
        }
        modified_vars = {
            k: local_vars[k]
            for k, v in old_vars.items() if local_vars[k] is not v
        }
        modified_values = {
            k: local_values[k]
            for k, v in old_values.items() if local_values[k] != v
        }
        self.frame_to_vars[frame] = local_vars
        self.frame_to_values[frame] = local_values

        return {
            "new_vars": new_vars,
            "new_values": new_values,
            "modified_vars": modified_vars,
            "modified_values": modified_values,
        }

    def trace(self, frame, event, arg):
        """
        Args:
            arg: when event is return, arg will be returned value
                or None if exception raised.
                when event is exception, arg will be (exception, value, traceback)
        """
        # ignore all line events, we don't need it.
        if event == "line":
            return 
        if event == "call":
            pass 
        elif event == "return":
            pass 
        elif event == "exception":
            pass
        return
        # print("??", frame.f_lineno, frame.f_code.co_name)
        if not self._check_is_traced_frame(frame):
            return None
        next_lineno = frame.f_lineno
        extend_tree, first_lineno = astex.rt.get_extend_tree_by_frame(frame)
        if extend_tree is None:
            return None
        nodeex = extend_tree.get_nodeexes_by_lineno(first_lineno)[0]
        lines = extend_tree.get_node_source_lines(nodeex.node)
        # code_obj = get_code_from_frame(frame)
        total_vars = frame.f_locals
        changes = self.update_vars_and_values(frame)
        new_vars = changes["new_vars"]
        new_values = changes["new_values"]
        modified_vars = changes["modified_vars"]
        modified_values = changes["modified_values"]
        # get inplace change
        inplace_modified_values = {}
        regular_modified_values = {}
        lineno_to_be_printed = self.last_lineno
        self.last_lineno = next_lineno

        for k, v in modified_values.items():
            if k not in modified_vars:
                # inplace change
                inplace_modified_values[k] = v
            else:
                regular_modified_values[k] = v
        if nodeex.first_lineno == next_lineno:
            assert event == "call"
            # enter a new frame
            print("Enter", prettystr(new_values))
        else:
            is_exc = is_exception(frame, event, arg)
            if is_exc:
                print("end by exception")
            else:
                if event == "return":
                    lineno = next_lineno
                    self.frame_to_vars.pop(frame)
                    self.frame_to_values.pop(frame)
                else:
                    lineno = lineno_to_be_printed
                codes = lines[lineno - nodeex.first_lineno]
                line_msg = "({}){}".format(lineno, codes)
                if event != "return":
                    if len(new_values) > 0 or len(modified_values) > 0:
                        line_msg += "   "
                    if len(new_values) > 0:
                        line_msg += "+{} ".format(prettystr(new_values))
                    if len(inplace_modified_values) > 0:
                        line_msg += "I{}".format(
                            prettystr(inplace_modified_values))
                    if len(regular_modified_values) > 0:
                        line_msg += "{}".format(
                            prettystr(regular_modified_values))
                else:
                    line_msg += "   |Return {}".format(prettystr(arg))

                print(line_msg)
        # print(event, next_lineno)
        return self.trace


def _identity_filter(obj):
    return True


@TracerV3()
def func_test(a, b=3):
    c = a + b
    d = b
    e = c * d
    a = e * b
    a += 1
    f = (1, 2)
    if f == (1, 2):
        d += 2
    ("asd" "asd")
    for i in range(5):
        e += 1
    c = \
        """
    asd
    """"""asd
    """
    print(c)
    return e


if __name__ == "__main__":
    import numpy as np
    func_test(1, 2)
    # func_test(np.array([1]), np.array([2]))
