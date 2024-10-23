import enum 
import dataclasses
import os
from types import FrameType
from typing import Optional
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

@dataclasses.dataclass
class DebugFrameState:
    frame: Optional[FrameType]

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