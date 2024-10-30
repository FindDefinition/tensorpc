import dataclasses
import importlib.util
import inspect
import io
import json
import os
import threading
from pathlib import Path
from typing import Any, List, Optional

from tensorpc.compat import InWindows
from tensorpc.constants import TENSORPC_MAIN_PID
from tensorpc.core.bgserver import BACKGROUND_SERVER
from tensorpc.dbg.constants import (TENSORPC_DBG_FRAME_INSPECTOR_KEY,
                                    TENSORPC_DBG_TRACER_KEY,
                                    TENSORPC_ENV_DBG_ENABLE, BreakpointEvent,
                                    BreakpointType, TracerConfig, TraceResult,
                                    TracerType, RecordFilterConfig)
from tensorpc.dbg.tracer import DebugTracerWrapper, VizTracerAndPytorchTracer
from tensorpc.flow.client import is_inside_app_session
from tensorpc.flow.components.plus.dbg.bkptpanel import BreakpointDebugPanel
from tensorpc.flow.serv_names import serv_names as app_serv_names
from tensorpc.utils.rich_logging import get_logger

from .serv_names import serv_names

LOGGER = get_logger("tensorpc.dbg")

RECORDING = False

_TRACER_WRAPPER = DebugTracerWrapper()


@dataclasses.dataclass
class _DebugBkptMeta:
    rank: int = 0
    world_size: int = 1
    backend: Optional[str] = None


def _extrace_module_path(module: str):
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
            include_files.extend(_extrace_module_path(mod))
    if cfg.exclude_modules is not None:
        for mod in cfg.exclude_modules:
            exclude_files.extend(_extrace_module_path(mod))
    return include_files, exclude_files


def _get_viztracer(cfg: Optional[TracerConfig], name: Optional[str] = None):
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
            if tracer_type == TracerType.VIZTRACER:
                tracer = VizTracer(process_name=name,
                                   file_info=False,
                                   max_stack_depth=cfg.max_stack_depth,
                                   include_files=inc_files,
                                   exclude_files=exc_files)
                return tracer, TracerType.VIZTRACER
            elif tracer_type == TracerType.PYTORCH:
                import torch.profiler as profiler
                # pytorch tracer can't control ignored files and max_stack_depth, so
                # never use with_stack.
                tracer = profiler.profile(activities=[
                    profiler.ProfilerActivity.CPU,
                    profiler.ProfilerActivity.CUDA
                ],
                                          with_stack=False,
                                          profile_memory=False)
                return tracer, TracerType.PYTORCH
            elif tracer_type == TracerType.VIZTRACER_PYTORCH:
                import torch.profiler as profiler
                viz_tracer = VizTracer(process_name=name,
                                       file_info=False,
                                       max_stack_depth=cfg.max_stack_depth,
                                       include_files=inc_files,
                                       exclude_files=exc_files)
                pytorch_tracer = profiler.profile(activities=[
                    profiler.ProfilerActivity.CPU,
                    profiler.ProfilerActivity.CUDA
                ],
                                                  with_stack=False,
                                                  profile_memory=False)
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
    enable = is_inside_app_session()
    enable |= TENSORPC_ENV_DBG_ENABLE
    return enable


def init(proc_name: Optional[str] = None, port: int = -1):
    """Initialize the background server with the given process name
    if already started, this function does nothing.
    """
    if not should_enable_debug():
        return False
    if not BACKGROUND_SERVER.is_started:
        assert not InWindows, "init is not supported in Windows due to setproctitle."
        cur_pid = os.getpid()
        if proc_name is None:
            proc_name = Path(__file__).stem
        # pytorch distributed environment variables
        world_size = os.getenv("WORLD_SIZE", None)
        rank = os.getenv("RANK", None)
        mpi_world_size = os.getenv("OMPI_COMM_WORLD_SIZE", None)
        mpi_rank = os.getenv("OMPI_COMM_WORLD_RANK", None)
        # TODO we can only detect distributed workers inside same machine.
        # so we only support single-machine distributed debugging.
        rank_int = 0
        world_size_int = 1
        backend: Optional[str] = None
        if rank is not None:
            rank_int = int(rank)
        if world_size is not None:
            world_size_int = int(world_size)
        if world_size is not None and rank is not None:
            # assume pytorch distributed
            proc_name += f"_pth_rank{rank}"
            backend = "Pytorch"
        elif mpi_world_size is not None and mpi_rank is not None:
            # assume mpi
            proc_name += f"_mpi_rank{mpi_rank}"
            backend = "Mpi"
        if cur_pid != TENSORPC_MAIN_PID:
            proc_name += f"_fork"
        userdata = _DebugBkptMeta(rank=rank_int,
                                  world_size=world_size_int,
                                  backend=backend)
        BACKGROUND_SERVER.start_async(id=proc_name,
                                      port=port,
                                      userdata=userdata)
        panel = BreakpointDebugPanel().prop(flex=1)
        set_background_layout(TENSORPC_DBG_FRAME_INSPECTOR_KEY, panel)
        BACKGROUND_SERVER.execute_service(serv_names.DBG_INIT_BKPT_DEBUG_PANEL,
                                          panel)
        BACKGROUND_SERVER.execute_service(
            serv_names.DBG_TRY_FETCH_VSCODE_BREAKPOINTS)

    return True


def breakpoint(name: Optional[str] = None,
               timeout: Optional[float] = None,
               init_port: int = -1,
               init_proc_name: Optional[str] = None,
               type: BreakpointType = BreakpointType.Normal,
               *,
               _frame_cnt: int = 1):
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
    """
    global RECORDING
    if not should_enable_debug():
        return
    bev = BreakpointEvent(threading.Event())
    frame = inspect.currentframe()
    if frame is None:
        return
    while _frame_cnt > 0:
        if frame is not None:
            frame = frame.f_back
        _frame_cnt -= 1
    if frame is None:
        return
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
            trace_res_obj = TraceResult(
                res, _TRACER_WRAPPER._trace_instant_events_for_pth)
        _TRACER_WRAPPER.reset_tracer()
        # tracer_to_stop.stop()
        LOGGER.warning(f"Record Stop.")
        BACKGROUND_SERVER.execute_service(serv_names.DBG_SET_TRACE_DATA,
                                          trace_res_obj, tracer_cfg)

    bev.event.wait(timeout)
    if bev.enable_trace_in_main_thread:
        # tracer must be create/start/stop in main thread (or same thread)
        meta = BACKGROUND_SERVER.get_userdata_typed(_DebugBkptMeta)
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
            _TRACER_WRAPPER.set_tracer(tracer, tracer_type, tracer_name)
            _TRACER_WRAPPER.start()
            BACKGROUND_SERVER.execute_service(serv_names.DBG_SET_TRACER,
                                              tracer)
        else:
            LOGGER.error(
                "viztracer is not installed, can't record trace data. use `pip install viztracer` to install."
            )


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


def set_background_layout(key: str, layout: Any):
    if not should_enable_debug():
        return
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
            while _frame_cnt > 0:
                if frame is not None:
                    frame = frame.f_back
                _frame_cnt -= 1
            if frame is None:
                return
            args = {"path": frame.f_code.co_filename, "lineno": frame.f_lineno}
        _TRACER_WRAPPER.log_instant(name, args)
