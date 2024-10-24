import inspect
import os
from pathlib import Path
import threading
from typing import Any, Optional
from tensorpc.constants import TENSORPC_MAIN_PID
from tensorpc.core.bgserver import BACKGROUND_SERVER
from tensorpc.dbg.constants import TENSORPC_DBG_FRAME_INSPECTOR_KEY, TENSORPC_ENV_DBG_ENABLE, BreakpointType
from tensorpc.flow.client import is_inside_app_session
from tensorpc.flow.components.plus.dbg.bkptpanel import BreakpointDebugPanel
from .serv_names import serv_names
from tensorpc.flow.serv_names import serv_names as app_serv_names
from tensorpc.compat import InWindows


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
        if world_size is not None and rank is not None:
            # assume pytorch distributed
            proc_name += f"_pth_rank{rank}"
        elif mpi_world_size is not None and mpi_rank is not None:
            # assume mpi
            proc_name += f"_mpi_rank{mpi_rank}"
        if cur_pid != TENSORPC_MAIN_PID:
            proc_name += f"_fork"
        BACKGROUND_SERVER.start_async(id=proc_name, port=port)
        panel = BreakpointDebugPanel().prop(flex=1)
        set_background_layout(TENSORPC_DBG_FRAME_INSPECTOR_KEY, panel)
        BACKGROUND_SERVER.execute_service(serv_names.DBG_INIT_BKPT_DEBUG_PANEL,
                                          panel)
        BACKGROUND_SERVER.execute_service(serv_names.DBG_TRY_FETCH_VSCODE_BREAKPOINTS)

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
    """
    if not should_enable_debug():
        return
    init(init_proc_name, init_port)
    ev = threading.Event()
    frame = inspect.currentframe()
    if frame is None:
        return
    while _frame_cnt > 0:
        if frame is not None:
            frame = frame.f_back
        _frame_cnt -= 1
    if frame is None:
        return
    BACKGROUND_SERVER.execute_service(serv_names.DBG_ENTER_BREAKPOINT, frame,
                                      ev, type, name)
    ev.wait(timeout)

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
    return breakpoint(name, timeout, init_port, init_proc_name, BreakpointType.Vscode, _frame_cnt=2)

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

    def breakpoint(self, name: Optional[str] = None, timeout: Optional[float] = None):
        breakpoint(name, timeout, self._port, self._proc_name)