"""Methods in this file should only be runned inside UI callbacks in background Remote Component Server."""
from types import FrameType
from typing import Optional
from tensorpc import prim 
from .serv_names import serv_names as dbg_serv_names

def get_current_breakpoint_frame() -> Optional[FrameType]:
    frame = prim.get_service(
        dbg_serv_names.DBG_BKGD_GET_CURRENT_FRAME)()
    return frame