import enum 
import dataclasses
import os
import threading
from types import FrameType
from typing import Any, List, Optional
from tensorpc.core.datamodel import typemetas
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
    # special tracer types
    TARGET_TRACER = 3

class TraceLaunchType(enum.IntEnum):
    DEFAULT = 0
    # launch simple tracer, when encounter target file and function, run callback in return
    # and disable trace.
    # won't save trace data to perfetto and other trace-based tool.
    # used to inspect a variable in a loop with `tensorpc.dbg.breakpoint``.
    TARGET_VARIABLE = 1

@pydantic_dataclasses.dataclass
class RecordFilterConfig:
    exclude_name_prefixes: Optional[List[str]] = None
    exclude_file_names: Optional[List[str]] = None

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
    tracer: Annotated[TracerType, typemetas.Enum(excludes=[TracerType.TARGET_TRACER])] = TracerType.VIZTRACER
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
class TargetTraceConfig:
    target_filename: str 
    target_func_qname: str
    target_expr: str
    is_distributed: bool = False
    max_num_variable: int = 1

@dataclasses.dataclass
class TracerConfig(TracerUIConfig):
    enable: bool = True
    # trace until this number of breakpoints is reached
    trace_timestamp: Optional[int] = None
    record_filter: RecordFilterConfig = dataclasses.field(default_factory=RecordFilterConfig)
    launch_type: TraceLaunchType = TraceLaunchType.DEFAULT

    target_trace_cfg: Optional[TargetTraceConfig] = None

@dataclasses.dataclass
class TraceMetrics:
    breakpoint_count: int

@dataclasses.dataclass
class TracerSingleResult:
    data: bytes 
    tracer_type: TracerType
    trace_events: Optional[List[Any]] = None
    site_packages_prefix: Optional[str] = None
    external_events: Optional[List[Any]] = None

@dataclasses.dataclass
class TraceResult:
    single_results: List[TracerSingleResult] 

@dataclasses.dataclass
class DebugMetric:
    total_skipped_bkpt: int

@dataclasses.dataclass
class ExternalTrace:
    backend: Literal["pytorch"]
    data: Any

@dataclasses.dataclass
class DebugDistributedInfo:
    rank: int = 0
    world_size: int = 1
    backend: Optional[Literal["pytorch", "openmpi"]] = None
    run_id: Optional[str] = None
    local_world_size: Optional[int] = None

    def get_backend_short(self):
        if self.backend == "pytorch":
            return "pth"
        elif self.backend == "openmpi":
            return "mpi"
        else:
            return self.backend

@dataclasses.dataclass
class DebugInfo:
    metric: DebugMetric
    frame_meta: Optional[DebugFrameInfo]
    trace_cfg: Optional[TracerConfig]
    dist_info: Optional[DebugDistributedInfo] = None

class BreakpointType(enum.IntEnum):
    Normal = 0
    # breakpoint that only enable if a vscode breakpoint 
    # is set on the same line
    Vscode = 1

class RemoteDebugEventType(enum.Enum):
    DIST_TARGET_VARIABLE_TRACE = "dist_target_variable_trace"

@dataclasses.dataclass
class RemoteDebugEvent:
    type: RemoteDebugEventType


@dataclasses.dataclass
class RemoteDebugTargetTrace(RemoteDebugEvent):
    dist_info: DebugDistributedInfo
    target_filename: str 
    target_func_qname: str
    target_expr: str
    is_distributed: bool = False
    max_num_variable: int = 1

TENSORPC_ENV_DBG_ENABLE = os.getenv("TENSORPC_DBG_ENABLE", "1") != "0"
TENSORPC_ENV_DBG_DEFAULT_BREAKPOINT_ENABLE = os.getenv("TENSORPC_DBG_DEFAULT_BREAKPOINT_ENABLE", "1") != "0"

TENSORPC_DBG_FRAME_INSPECTOR_KEY = "__tensorpc_debug_frame_inspector"
TENSORPC_DBG_TRACE_VIEW_KEY = "__tensorpc_debug_trace_view"

TENSORPC_DBG_FRAMESCRIPT_STORAGE_PREFIX = "__tensorpc_dbg_frame_scripts"

TENSORPC_DBG_SPLIT = "::"

TENSORPC_DBG_FRAME_STORAGE_PREFIX = "__tensorpc_dbg_frame"

TENSORPC_DBG_TRACER_KEY = "__tensorpc_dbg_tracer"

TENSORPC_DBG_USER_DURATION_EVENT_KEY = "__tensorpc_dbg_E_dur"


TENSORPC_DBG_REMOTE_EVENT_TARGET_TRACE = "__tensorpc_remote_ev_target_trace"