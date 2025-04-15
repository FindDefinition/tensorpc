import traceback
import torch.distributed as dist
from tensorpc.apps.distssh.constants import TENSORPC_ENV_DISTSSH_URL_WITH_PORT
from tensorpc.core import BuiltinServiceKeys
from tensorpc.core.client import simple_remote_call
from tensorpc.apps.dbg.bkpt import breakpoint

import os 

def pth_control_point(*, _frame_cnt: int = 2):
    url_with_port = os.environ.get(TENSORPC_ENV_DISTSSH_URL_WITH_PORT)
    if url_with_port is None:
        raise ValueError("You must use pth_control_point inside distssh.")
    if not dist.is_initialized():
        raise RuntimeError(
            "You must use pth_control_point inside a pytorch distributed process group."
        )
    global_rank = dist.get_rank()
    should_enter_breakpoint = False 
    if global_rank == 0:
        try:
            should_enter_breakpoint = simple_remote_call(
                BuiltinServiceKeys.FaultToleranceSSHServer.value + ".is_user_control_enabled", url_with_port
            )
        except:
            # server may not prepared yet, ignore this control.
            traceback.print_exc()
            should_enter_breakpoint = False
    # broadcast should_enter_breakpoint to all rank
    world_size = dist.get_world_size()
    obj_list = [should_enter_breakpoint] * world_size
    dist.broadcast_object_list(obj_list, src=0)
    should_enter_breakpoint = obj_list[global_rank]
    if not should_enter_breakpoint:
        return 

    return breakpoint(_frame_cnt=_frame_cnt)