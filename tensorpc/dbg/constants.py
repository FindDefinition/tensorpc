import enum 
import dataclasses
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
class BackgroundDebugToolsConfig:
    skip_breakpoint: bool = False

TENSORPC_DBG_FRAME_INSPECTOR_KEY = "__tensorpc_debug_frame_inspector"