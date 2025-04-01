from types import FrameType
from tensorpc.core import dataclass_dispatch as dataclasses 
from tensorpc.core import inspecttools

@dataclasses.dataclass
class DebugFrameInfo:
    name: str
    qualname: str
    
    path: str 
    lineno: int

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class TracerState:
    pass 


@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class PyDbgModel:
    # backend props
    frame: FrameType
    # frontend props 
    info: DebugFrameInfo

    @staticmethod
    def get_frame_info_from_frame(frame: FrameType) -> DebugFrameInfo:
        qname = inspecttools.get_co_qualname_from_frame(frame)
        return DebugFrameInfo(frame.f_code.co_name, qname,
                              frame.f_code.co_filename,
                              frame.f_lineno)
