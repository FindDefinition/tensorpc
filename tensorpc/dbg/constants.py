import enum 
import dataclasses
import os
import threading
from types import FrameType
from typing import Any, Optional
from tensorpc.core import typemetas
from typing_extensions import Annotated


class DebugServerStatus(enum.IntEnum):
    Idle = 0
    InsideBreakpoint = 1

@dataclasses.dataclass
class DebugFrameMeta:
    name: str
    qualname: str
    
    path: str 
    lineno: int

@dataclasses.dataclass
class BreakpointEvent:
    event: threading.Event
    # props below are set in background server
    enable_trace_in_main_thread: bool = False

    def set(self):
        self.event.set()


class RecordMode(enum.IntEnum):
    NEXT_BREAKPOINT = 0
    SAME_BREAKPOINT = 1


@dataclasses.dataclass
class BackgroundDebugToolsConfig:
    skip_breakpoint: bool = False

@dataclasses.dataclass
class DebugFrameState:
    frame: Optional[FrameType]

@dataclasses.dataclass
class TracerConfig:
    enable: bool
    # trace until this number of breakpoints is reached
    breakpoint_count: int = 1 
    trace_name: Optional[str] = None
    trace_timestamp: Optional[int] = None
    mode: RecordMode = RecordMode.NEXT_BREAKPOINT

@dataclasses.dataclass
class TracerUIConfig:
    breakpoint_count: Annotated[int, typemetas.CommonObject(alias="Breakpoint Count")] = 1
    trace_name: Annotated[str, typemetas.CommonObject(alias="Trace Name")] = "trace"
    mode: RecordMode = RecordMode.NEXT_BREAKPOINT


class BreakpointType(enum.IntEnum):
    Normal = 0
    # breakpoint that only enable if a vscode breakpoint 
    # is set on the same line
    Vscode = 1


TENSORPC_ENV_DBG_ENABLE = os.getenv("TENSORPC_DBG_ENABLE", "1") != "0"
TENSORPC_ENV_DBG_DEFAULT_BREAKPOINT_ENABLE = os.getenv("TENSORPC_DBG_DEFAULT_BREAKPOINT_ENABLE", "1") != "0"

TENSORPC_DBG_FRAME_INSPECTOR_KEY = "__tensorpc_debug_frame_inspector"
TENSORPC_DBG_FRAMESCRIPT_STORAGE_PREFIX = "__tensorpc_dbg_frame_scripts"

TENSORPC_DBG_SPLIT = "::"

TENSORPC_DBG_FRAME_STORAGE_PREFIX = "__tensorpc_dbg_frame"

TENSORPC_DBG_TRACER_KEY = "__tensorpc_dbg_tracer"