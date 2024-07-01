import enum 
import dataclasses

class TraceEventType(enum.IntEnum):
    Call = 0
    Return = 1
    Line = 2


@dataclasses.dataclass
class FrameEventBase:
    type: TraceEventType
    qualname: str
    filename: str
    lineno: int

    def get_unique_id(self):
        return f"{self.filename}@:{self.lineno}{self.qualname}"

