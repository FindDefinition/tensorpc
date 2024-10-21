import inspect
import os
from pathlib import Path
import threading
from typing import Any, Optional
from tensorpc.core.bgserver import BACKGROUND_SERVER
from tensorpc.dbg.constants import TENSORPC_DBG_FRAME_INSPECTOR_KEY
from tensorpc.flow.components.plus.dbg.bkptpanel import BreakpointDebugPanel
from .serv_names import serv_names
from tensorpc.flow.serv_names import serv_names as app_serv_names
import rich
from tensorpc.flow import mui, plus
from tensorpc.compat import InWindows


def init(name: Optional[str] = None, port: int = -1):
    """Initialize the background server with the given name
    if already started, this function does nothing.
    """
    assert not InWindows, "init is not supported in Windows due to setproctitle."
    if name is None:
        name = Path(__file__).stem
    # pytorch distributed environment variables
    world_size = os.getenv("WORLD_SIZE", None)
    rank = os.getenv("RANK", None)
    mpi_world_size = os.getenv("OMPI_COMM_WORLD_SIZE", None)
    mpi_rank = os.getenv("OMPI_COMM_WORLD_RANK", None)
    # TODO we can only detect distributed workers inside same machine.
    # so we only support single-machine distributed debugging.
    if world_size is not None and rank is not None:
        # assume pytorch distributed
        name += f"_pth_rank{rank}"
    elif mpi_world_size is not None and mpi_rank is not None:
        # assume mpi
        name += f"_mpi_rank{mpi_rank}"
    if not BACKGROUND_SERVER.is_started:
        BACKGROUND_SERVER.start_async(id=name, port=port)
        panel = BreakpointDebugPanel().prop(flex=1)
        set_background_layout(TENSORPC_DBG_FRAME_INSPECTOR_KEY, panel)
        BACKGROUND_SERVER.execute_service(serv_names.DBG_INIT_BKPT_DEBUG_PANEL,
                                          panel)


def breakpoint(name: Optional[str] = None,
               timeout: Optional[float] = None,
               init_port: int = -1):
    """Enter a breakpoint in the background server.
    you must use specific UI or command tool to exit breakpoint.
    """
    init(name, init_port)
    ev = threading.Event()
    frame = inspect.currentframe()
    if frame is None:
        return
    frame = frame.f_back
    if frame is None:
        return
    rich.print(
        f"[bold red]Entering breakpoint... port={BACKGROUND_SERVER.port}[/bold red]"
    )
    BACKGROUND_SERVER.execute_service(serv_names.DBG_ENTER_BREAKPOINT, frame,
                                      ev)
    ev.wait(timeout)

def set_background_layout(key: str, layout: Any):
    BACKGROUND_SERVER.execute_service(
        app_serv_names.REMOTE_COMP_SET_LAYOUT_OBJECT, key, layout)
