import dataclasses
from typing import Any

from tensorpc.apps.dbg.constants import TracerConfig

@dataclasses.dataclass
class BreakpointEvent:
    pass 

@dataclasses.dataclass
class BkptLeaveEvent(BreakpointEvent):
    data: Any = None 
    should_raise: bool = False


@dataclasses.dataclass
class BkptLaunchTraceEvent(BreakpointEvent):
    trace_cfg: TracerConfig

@dataclasses.dataclass
class BkptRunScriptEvent(BreakpointEvent):
    code: str