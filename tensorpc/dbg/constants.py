import enum 
import dataclasses
import os
import threading
from types import FrameType
from typing import Any, List, Optional
from tensorpc.core import typemetas
from typing_extensions import Annotated, Literal
from tensorpc.core import dataclass_dispatch as pydantic_dataclasses

class DebugServerStatus(enum.IntEnum):
    Idle = 0
    InsideBreakpoint = 1

@dataclasses.dataclass
class DebugFrameInfo:
    name: str
    qualname: str
    
    path: str 
    lineno: int

@dataclasses.dataclass
class BreakpointEvent:
    event: threading.Event
    # props below are set in background server
    enable_trace_in_main_thread: bool = False
    trace_cfg: Optional["TracerConfig"] = None
    def set(self):
        self.event.set()


class RecordMode(enum.IntEnum):
    NEXT_BREAKPOINT = 0
    SAME_BREAKPOINT = 1
    INFINITE = 2

class TracerType(enum.IntEnum):
    VIZTRACER = 0
    PYTORCH = 1
    # use viztracer for python code and pytorch profiler for pytorch+cuda code
    # `with_stack` in pytorch profiler must be disabled.
    VIZTRACER_PYTORCH = 2


@pydantic_dataclasses.dataclass
class RecordFilterConfig:
    include_modules: Optional[List[str]] = None
    exclude_modules: Optional[List[str]] = None
    include_files: Optional[List[str]] = None
    exclude_files: Optional[List[str]] = None


@dataclasses.dataclass
class BackgroundDebugToolsConfig:
    skip_breakpoint: bool = False

@dataclasses.dataclass
class DebugFrameState:
    frame: Optional[FrameType]

@dataclasses.dataclass
class TracerUIConfig:
    breakpoint_count: Annotated[int, typemetas.CommonObject(alias="Breakpoint Count")] = 1
    trace_name: Annotated[str, typemetas.CommonObject(alias="Trace Name")] = "trace"
    mode: RecordMode = RecordMode.NEXT_BREAKPOINT
    max_stack_depth: Annotated[int, typemetas.CommonObject(alias="Max Stack Depth")] = 10
    tracer: TracerType = TracerType.VIZTRACER

@dataclasses.dataclass
class TracerConfig(TracerUIConfig):
    enable: bool = True
    # trace until this number of breakpoints is reached
    trace_timestamp: Optional[int] = None
    record_filter: RecordFilterConfig = dataclasses.field(default_factory=RecordFilterConfig)

@dataclasses.dataclass
class TraceMetrics:
    breakpoint_count: int

@dataclasses.dataclass
class TraceResult:
    data: List[bytes] 
    external_events: List[Any] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class DebugMetric:
    total_skipped_bkpt: int

@dataclasses.dataclass
class ExternalTrace:
    backend: Literal["pytorch"]
    data: Any

@dataclasses.dataclass
class DebugInfo:
    metric: DebugMetric
    frame_meta: Optional[DebugFrameInfo]
    trace_cfg: Optional[TracerConfig]


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