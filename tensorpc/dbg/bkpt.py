import contextlib
import dataclasses
from functools import partial
import importlib.util
import inspect
import io
import json
import os
import threading
from pathlib import Path
import traceback
from types import FrameType
from typing import Any, List, Optional

from tensorpc.compat import InWindows
from tensorpc.constants import TENSORPC_MAIN_PID
from tensorpc.core.bgserver import BACKGROUND_SERVER
from tensorpc.core.tracers.targettracer import TargetTracer
from tensorpc.dbg.constants import (TENSORPC_DBG_FRAME_INSPECTOR_KEY,
                                    TENSORPC_DBG_TRACE_VIEW_KEY,
                                    TENSORPC_ENV_DBG_ENABLE, BreakpointEvent,
                                    BreakpointType, TraceLaunchType,
                                    TracerConfig, TraceResult, TracerType,
                                    RecordFilterConfig, DebugDistributedInfo)
from tensorpc.dbg.tracer import DebugTracerWrapper, VizTracerAndPytorchTracer
from tensorpc.utils.rich_logging import get_logger
import sys
from .serv_names import serv_names

LOGGER = get_logger("tensorpc.dbg")

RECORDING = False

_TRACER_WRAPPER = DebugTracerWrapper()


def _try_get_distributed_meta():
    # try find torch dist (only support torchrun since it set enought env)
    if "torch" in sys.modules:
        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            res = DebugDistributedInfo(rank=rank,
                                       world_size=world_size,
                                       backend="pytorch")
            if os.getenv("TORCHELASTIC_RUN_ID", None) is not None:
                run_id = os.getenv("TORCHELASTIC_RUN_ID", None)
                local_world_size = os.getenv("LOCAL_WORLD_SIZE", None)
                res.run_id = run_id
                if local_world_size is not None:
                    res.local_world_size = int(local_world_size)
            return res
    elif "OMPI_COMM_WORLD_SIZE" in os.environ:
        mpi_world_size = os.getenv("OMPI_COMM_WORLD_SIZE", None)
        mpi_rank = os.getenv("OMPI_COMM_WORLD_RANK", None)
        mpi_local_world_size = os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE", None)
        if mpi_world_size is not None and mpi_rank is not None:
            res = DebugDistributedInfo(rank=int(mpi_rank),
                                       world_size=int(mpi_world_size),
                                       backend="openmpi")
            if mpi_local_world_size is not None:
                res.local_world_size = int(mpi_local_world_size)
            return res
    return DebugDistributedInfo()


def _extract_module_path(module: str):
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        return []
    if spec.submodule_search_locations is not None:
        return spec.submodule_search_locations
    else:
        return [spec.origin]


def _parse_record_filter(cfg: RecordFilterConfig):
    include_files: List[str] = []
    exclude_files: List[str] = []
    if cfg.include_files is not None:
        for f in cfg.include_files:
            include_files.append(f)
    if cfg.exclude_files is not None:
        for f in cfg.exclude_files:
            exclude_files.append(f)
    if cfg.include_modules is not None:
        for mod in cfg.include_modules:
            include_files.extend(_extract_module_path(mod))
    if cfg.exclude_modules is not None:
        for mod in cfg.exclude_modules:
            exclude_files.extend(_extract_module_path(mod))
    return include_files, exclude_files


class _TargetTraceHandler:
    def __init__(self, target_expr: str, target_qname: str, is_dist: bool):
        self.target_expr = target_expr
        self.target_qname = target_qname
        self.is_dist = is_dist
        self.var_uid = f"{self.target_qname}::{self.target_expr}"

        self._var_collected = []


    def _handler_collect(self, frame: FrameType):
        try:
            res = eval(self.target_expr, frame.f_globals, frame.f_locals)
            self._var_collected.append(res)
        except Exception as e:
            LOGGER.error(f"Eval target expr {self.target_expr} error: {e}")

    def _handler_stop(self):
        try:
            res = self._var_collected
            if self.is_dist:
                import torch.distributed as dist 
                if dist.is_initialized():
                    res_list = [None] * dist.get_world_size()
                    dist.all_gather_object(res_list, self._var_collected)
                    res = res_list
            BACKGROUND_SERVER.execute_service(
                serv_names.DBG_TRACEVIEW_SET_VARIABLE_INSPECT, self.var_uid, res)
        except Exception as e:
            LOGGER.error(f"Set Target {self.target_expr} error: {e}")

    def _handler_single(self, frame: FrameType):
        # eval expr in frame
        try:
            res = eval(self.target_expr, frame.f_globals, frame.f_locals)
            if self.is_dist:
                import torch.distributed as dist 
                if dist.is_initialized():
                    res_list = [None] * dist.get_world_size()
                    dist.all_gather_object(res_list, res)
                    res = res_list
            BACKGROUND_SERVER.execute_service(
                serv_names.DBG_TRACEVIEW_SET_VARIABLE_INSPECT, self.var_uid, res)
        except Exception as e:
            LOGGER.error(f"Eval target expr {self.target_expr} error: {e}")


def _get_viztracer(cfg: Optional[TracerConfig], name: Optional[str] = None):
    if cfg is not None and cfg.launch_type != TraceLaunchType.DEFAULT:
        # handle special trace.
        if (cfg.launch_type == TraceLaunchType.TARGET_VARIABLE
                and cfg.target_trace_cfg is not None):
            tcfg = cfg.target_trace_cfg
            handler = _TargetTraceHandler(tcfg.target_expr,
                                            tcfg.target_func_qname,
                                            tcfg.is_distributed)
            if tcfg.max_num_variable == 1:
                tracer = TargetTracer(tcfg.target_filename,
                                    tcfg.target_func_qname,
                                    handler._handler_single,
                                    max_depth=cfg.max_stack_depth)
            else:
                tracer = TargetTracer(tcfg.target_filename,
                                    tcfg.target_func_qname,
                                    handler._handler_collect,
                                    handler._handler_stop,
                                    max_depth=cfg.max_stack_depth,
                                    max_num_variable=tcfg.max_num_variable)
            return tracer, TracerType.TARGET_TRACER
        return None, TracerType.VIZTRACER
    try:
        from viztracer import VizTracer

        # file_info=False to reduce the size of trace data
        # TODO let user customize this
        if cfg is not None:
            inc_files, exc_files = _parse_record_filter(cfg.record_filter)
            if not inc_files:
                inc_files = None
            if not exc_files:
                exc_files = None
            tracer_type = cfg.tracer
            LOGGER.warning("%s %s", inc_files, exc_files)
            if tracer_type == TracerType.VIZTRACER:
                tracer = VizTracer(process_name=name,
                                   file_info=False,
                                   max_stack_depth=cfg.max_stack_depth,
                                   include_files=inc_files,
                                   exclude_files=exc_files,
                                   min_duration=cfg.min_duration,
                                   ignore_c_function=cfg.ignore_c_function)
                return tracer, TracerType.VIZTRACER
            elif tracer_type == TracerType.PYTORCH:
                import torch.profiler as profiler
                # pytorch tracer can't control ignored files and max_stack_depth, so
                # never use with_stack.
                tracer = profiler.profile(activities=[
                    profiler.ProfilerActivity.CPU,
                    profiler.ProfilerActivity.CUDA
                ],
                                          with_stack=cfg.pytorch_with_stask,
                                          profile_memory=cfg.profile_memory)
                return tracer, TracerType.PYTORCH
            elif tracer_type == TracerType.VIZTRACER_PYTORCH:
                import torch.profiler as profiler
                viz_tracer = VizTracer(process_name=name,
                                       file_info=False,
                                       max_stack_depth=cfg.max_stack_depth,
                                       include_files=inc_files,
                                       exclude_files=exc_files,
                                       min_duration=cfg.min_duration,
                                       ignore_c_function=cfg.ignore_c_function)
                pytorch_tracer = profiler.profile(
                    activities=[
                        profiler.ProfilerActivity.CPU,
                        profiler.ProfilerActivity.CUDA
                    ],
                    with_stack=False,
                    profile_memory=cfg.profile_memory)
                return VizTracerAndPytorchTracer(
                    viz_tracer, pytorch_tracer), TracerType.VIZTRACER_PYTORCH
            else:
                # TODO raise here? may break user code
                return None, TracerType.VIZTRACER
        else:
            return VizTracer(process_name=name,
                             file_info=False,
                             max_stack_depth=8), TracerType.VIZTRACER
    except ImportError:
        return None, TracerType.VIZTRACER


def should_enable_debug() -> bool:
    """Check if the debug environment is enabled"""
    from tensorpc.flow.client import is_inside_app_session
    enable = is_inside_app_session()
    enable |= TENSORPC_ENV_DBG_ENABLE
    return enable


def init(proc_name: Optional[str] = None, port: int = -1):
    """Initialize the background server with the given process name
    if already started, this function does nothing.
    TODO setup pytorch extra
    """
    if not should_enable_debug():
        return False
    if not BACKGROUND_SERVER.is_started:
        # put app import here to reduce import time
        from tensorpc.flow.components.plus.dbg.bkptpanel import BreakpointDebugPanel
        from tensorpc.flow.components.plus.dbg.traceview import TraceView
        assert not InWindows, "init is not supported in Windows due to setproctitle."
        cur_pid = os.getpid()
        if proc_name is None:
            proc_name = Path(__file__).stem
        if cur_pid != TENSORPC_MAIN_PID:
            proc_name += f"_fork"
        userdata = _try_get_distributed_meta()
        if userdata.backend is not None:
            proc_name += f"_{userdata.get_backend_short()}_{userdata.rank}"

        BACKGROUND_SERVER.start_async(id=proc_name,
                                      port=port,
                                      userdata=userdata)
        panel = BreakpointDebugPanel().prop(flex=1)
        trace_view = TraceView(userdata).prop(flex=1)
        set_background_layout(TENSORPC_DBG_FRAME_INSPECTOR_KEY, panel)
        set_background_layout(TENSORPC_DBG_TRACE_VIEW_KEY, trace_view)

        BACKGROUND_SERVER.execute_service(serv_names.DBG_INIT_BKPT_DEBUG_PANEL,
                                          panel)
        BACKGROUND_SERVER.execute_service(
            serv_names.DBG_TRY_FETCH_VSCODE_BREAKPOINTS)
        if userdata.backend is not None:
            BACKGROUND_SERVER.execute_service(
                serv_names.DBG_SET_DISTRIBUTED_META, userdata)

    return True


def _patch_events_for_pytorch_dist(events: List[Any]):
    import torch.distributed as dist
    rank = dist.get_rank()
    for ev in events:
        if ev["ph"] == "X":
            # set thread id to rank
            ev["tid"] = rank


def _patch_events_pid_for_pytorch_dist(events: List[Any]):
    import torch.distributed as dist
    rank = dist.get_rank()
    pid = os.getpid()
    for ev in events:
        if ev["ph"] == "X":
            # set thread id to rank
            ev["pid"] = pid


def breakpoint(name: Optional[str] = None,
               timeout: Optional[float] = None,
               init_port: int = -1,
               init_proc_name: Optional[str] = None,
               type: BreakpointType = BreakpointType.Normal,
               *,
               _frame_cnt: int = 1,
               pytorch_dist_extra: bool = False,
               external_frame: Optional[FrameType] = None):
    """Enter a breakpoint in the background server.
    you must use specific UI or command tool to exit breakpoint.
    WARNING: currently don't support multi-thread

    Args:
        name: the name of the breakpoint, currently only used during record (instant event).
        timeout: the timeout of the breakpoint
        init_port: the port of the background server
        init_proc_name: the process name of the background server
        type: the type of the breakpoint
        _frame_cnt: the frame count to skip
        pytorch_dist_extra: whether to enable pytorch distributed extra.
            if user enable this, user must ensure all breakpoint call is synchronized,
            hang may happen if some rank is diverged.
    """
    global RECORDING
    if not should_enable_debug():
        return False
    bev = BreakpointEvent(threading.Event())
    if external_frame is not None:
        frame = external_frame
    else:
        frame = inspect.currentframe()
        if frame is None:
            return False
        while _frame_cnt > 0:
            if frame is not None:
                frame = frame.f_back
            _frame_cnt -= 1
        if frame is None:
            return False
    if init_proc_name is None:
        init_proc_name = frame.f_code.co_name

    init(init_proc_name, init_port)
    if name is not None:
        record_instant_event(name,
                             args={
                                 "path": frame.f_code.co_filename,
                                 "lineno": frame.f_lineno
                             })
    trace_res = BACKGROUND_SERVER.execute_service(
        serv_names.DBG_ENTER_BREAKPOINT, frame, bev, type, name)
    if trace_res is not None:
        tracer_to_stop, tracer_cfg = trace_res
        RECORDING = False
        _TRACER_WRAPPER.stop()
        res = _TRACER_WRAPPER.save(BACKGROUND_SERVER.cur_proc_title)
        trace_res_obj = None
        if res is not None:
            if pytorch_dist_extra:
                # broadcast external events to all rank.
                # usually used for perfetto distributed visualization in single rank.
                # you can insert custom duration event to show distributed events
                # such as pipeline parallel.
                for single_res in res:
                    if single_res.external_events:
                        import torch.distributed as dist
                        ws = dist.get_world_size()
                        output = [None for _ in range(ws)]
                        _patch_events_for_pytorch_dist(
                            single_res.external_events)
                        dist.all_gather_object(output,
                                               single_res.external_events)
                        events_all_rank = []
                        for events in output:
                            assert events is not None
                            events_all_rank.extend(events)
                        _patch_events_pid_for_pytorch_dist(events_all_rank)
                        single_res.external_events = events_all_rank
            trace_res_obj = TraceResult(res)
        _TRACER_WRAPPER.reset_tracer()
        # tracer_to_stop.stop()
        LOGGER.warning(f"Record Stop.")
        if trace_res_obj is not None:
            BACKGROUND_SERVER.execute_service(serv_names.DBG_SET_TRACE_DATA,
                                              trace_res_obj, tracer_cfg)

    bev.event.wait(timeout)
    if bev.enable_trace_in_main_thread:
        # tracer must be create/start/stop in main thread (or same thread)
        meta = BACKGROUND_SERVER.get_userdata_typed(DebugDistributedInfo)
        tracer_name: Optional[str] = None
        if meta.backend is not None:
            tracer_name = f"{meta.backend}|{meta.rank}/{meta.world_size}"
        else:
            tracer_name = f"Process"
        tracer, tracer_type = _get_viztracer(bev.trace_cfg, name=tracer_name)
        if tracer is not None:
            LOGGER.warning(f"Record Start. Config:")
            LOGGER.warning(bev.trace_cfg)
            RECORDING = True
            _TRACER_WRAPPER.set_tracer(bev.trace_cfg, tracer, tracer_type,
                                       tracer_name, meta)
            _TRACER_WRAPPER.start()
            BACKGROUND_SERVER.execute_service(serv_names.DBG_SET_TRACER,
                                              tracer)
        else:
            LOGGER.error(
                "viztracer is not installed, can't record trace data. use `pip install viztracer` to install."
            )
    return True

def breakpoint_dist_pth(name: Optional[str] = None,
                        timeout: Optional[float] = None,
                        init_port: int = -1,
                        init_proc_name: Optional[str] = None):
    """Enter a breakpoint in the background server.
    pytorch distributed sync may called in breakpoint (init and record),
    so user must ensure all breakpoint call is synchronized,
    """
    return breakpoint(name,
                      timeout,
                      init_port,
                      init_proc_name,
                      BreakpointType.Normal,
                      _frame_cnt=2,
                      pytorch_dist_extra=True)


def vscode_breakpoint(name: Optional[str] = None,
                      timeout: Optional[float] = None,
                      init_port: int = -1,
                      init_proc_name: Optional[str] = None):
    """Enter a breakpoint in the background server.
    only triggered if a vscode breakpoint is set on the same line.
    you can use specific UI or command tool or just remove breakpoint
    in vscode to exit breakpoint.
    WARNING: currently don't support multi-thread
    """
    return breakpoint(name,
                      timeout,
                      init_port,
                      init_proc_name,
                      BreakpointType.Vscode,
                      _frame_cnt=2)


def vscode_breakpoint_dist_pth(name: Optional[str] = None,
                               timeout: Optional[float] = None,
                               init_port: int = -1,
                               init_proc_name: Optional[str] = None):
    """Enter a vscode breakpoint in the background server.
    pytorch distributed sync may called in breakpoint (init and record),
    so user must ensure all breakpoint call is synchronized,
    """
    return breakpoint(name,
                      timeout,
                      init_port,
                      init_proc_name,
                      BreakpointType.Vscode,
                      _frame_cnt=2,
                      pytorch_dist_extra=True)


def set_background_layout(key: str, layout: Any):
    if not should_enable_debug():
        return
    from tensorpc.flow.serv_names import serv_names as app_serv_names
    BACKGROUND_SERVER.execute_service(
        app_serv_names.REMOTE_COMP_SET_LAYOUT_OBJECT, key, layout)


class Debugger:

    def __init__(self, proc_name: str, port: int = -1):
        """
        Args:
            proc_name: the process name of the background server, only valid before init
            port: the port of the background server, only valid before init
        """
        self._proc_name = proc_name
        self._port = port

    def breakpoint(self,
                   name: Optional[str] = None,
                   timeout: Optional[float] = None):
        breakpoint(name, timeout, self._port, self._proc_name)


def record_instant_event(name: str, args: Any = None, *, _frame_cnt: int = 1):
    if RECORDING:
        if args is None:
            frame = inspect.currentframe()
            if frame is None:
                return
            if _frame_cnt == 1:
                # fast path
                frame = frame.f_back
            else:
                while _frame_cnt > 0:
                    if frame is not None:
                        frame = frame.f_back
                    _frame_cnt -= 1
            if frame is None:
                return
            args = {"path": frame.f_code.co_filename, "lineno": frame.f_lineno}
        _TRACER_WRAPPER.log_instant(name, args)


def record_print(*args, _frame_cnt: int = 1, name: str = "print"):
    if RECORDING:
        ss = io.StringIO()
        print(*args, file=ss)
        frame = inspect.currentframe()
        if frame is None:
            return
        if _frame_cnt == 1:
            # fast path
            frame = frame.f_back
        else:
            while _frame_cnt > 0:
                if frame is not None:
                    frame = frame.f_back
                _frame_cnt -= 1
        if frame is None:
            return
        ev_args = {
            "path": frame.f_code.co_filename,
            "lineno": frame.f_lineno,
            "msg": ss.getvalue()
        }
        _TRACER_WRAPPER.log_instant(name, ev_args)


@contextlib.contextmanager
def record_duration(name: str,
                    args: Any = None,
                    *,
                    _frame_cnt: int = 1,
                    thread_id: int = 0,
                    cat: Optional[str] = None):
    if RECORDING:
        frame = inspect.currentframe()
        if frame is None:
            yield
            return
        if _frame_cnt == 1:
            # fast path
            frame = frame.f_back
        else:
            while _frame_cnt > 0:
                if frame is not None:
                    frame = frame.f_back
                _frame_cnt -= 1
        if frame is None:
            yield
            return
        if args is None:
            args = {"path": frame.f_code.co_filename, "lineno": frame.f_lineno}
        with _TRACER_WRAPPER.log_duration(name, args, thread_id, cat):
            yield
    else:
        yield

@contextlib.contextmanager
def exception_breakpoint():
    """Enter a breakpoint when exception is captured by this context manager. 
    do nothing otherwise.
    """
    try:
        yield
    except KeyboardInterrupt:
        raise 
    except Exception as e:
        _, _, exc_traceback = sys.exc_info()
        if exc_traceback is None:
            raise e
        frame: Optional[FrameType] = None
        # walk to the innermost frame
        for frame, _ in traceback.walk_tb(exc_traceback):
            pass
        if frame is None:
            raise e
        traceback.print_exc()
        breakpoint(external_frame=frame)
        raise e
