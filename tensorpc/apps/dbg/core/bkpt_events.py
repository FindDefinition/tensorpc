import dataclasses

from tensorpc.apps.dbg.constants import TracerConfig

@dataclasses.dataclass
class BreakpointEvent:
    pass 

@dataclasses.dataclass
class BkptLeaveEvent(BreakpointEvent):
    pass 


@dataclasses.dataclass
class BkptLaunchTraceEvent(BreakpointEvent):
    trace_cfg: TracerConfig