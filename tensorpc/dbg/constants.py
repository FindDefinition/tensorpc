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
    exclude_name_prefixes: Optional[List[str]] = None

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
    tracer: TracerType = TracerType.VIZTRACER
    trace_name: Annotated[str, typemetas.CommonObject(alias="Trace Name")] = "trace"
    mode: RecordMode = RecordMode.NEXT_BREAKPOINT
    breakpoint_count: Annotated[int, typemetas.RangedInt(1, 100, alias="Breakpoint Count")] = 1
    max_stack_depth: Annotated[int, typemetas.RangedInt(1, 50, alias="Max Stack Depth")] = 10
    ignore_c_function: Annotated[bool, typemetas.CommonObject(alias="Ignore C Function")] = True
    min_duration: Annotated[float, typemetas.RangedInt(0, 5000, alias="Min Duration (us, VizTracer)")] = 0
    profile_memory: Annotated[bool, typemetas.CommonObject(alias="Profile Memory (PyTorch)")] = False
    pytorch_with_stask: Annotated[bool, typemetas.CommonObject(alias="PyTorch Record Python")] = False
    replace_sitepkg_prefix: Annotated[bool, typemetas.CommonObject(alias="Remove site-packages Prefix")] = True

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

@dataclasses.dataclass
class DebugDistributedMeta:
    rank: int = 0
    world_size: int = 1
    backend: Optional[str] = None

TENSORPC_ENV_DBG_ENABLE = os.getenv("TENSORPC_DBG_ENABLE", "1") != "0"
TENSORPC_ENV_DBG_DEFAULT_BREAKPOINT_ENABLE = os.getenv("TENSORPC_DBG_DEFAULT_BREAKPOINT_ENABLE", "1") != "0"

TENSORPC_DBG_FRAME_INSPECTOR_KEY = "__tensorpc_debug_frame_inspector"
TENSORPC_DBG_FRAMESCRIPT_STORAGE_PREFIX = "__tensorpc_dbg_frame_scripts"

TENSORPC_DBG_SPLIT = "::"

TENSORPC_DBG_FRAME_STORAGE_PREFIX = "__tensorpc_dbg_frame"

TENSORPC_DBG_TRACER_KEY = "__tensorpc_dbg_tracer"