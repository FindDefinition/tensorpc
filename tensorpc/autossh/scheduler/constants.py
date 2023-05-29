""" tensorpc tmux program name format:

prefix::port::uuid

"""

import os
from typing import Optional 
TMUX_SESSION_NAME_SPLIT = "-"
TMUX_SESSION_PREFIX = "__tensorpc_ssh_scheduler"

TMUX_SESSION_TASK_PREFIX = "__tensorpc_ssh_scheduled_task"

TENSORPC_TMUX_TASK_SCHEDULER_PORT = "TENSORPC_TMUX_TASK_SCHEDULER_PORT"

TENSORPC_TMUX_TASK_UID = "TENSORPC_TMUX_TASK_UID"

TENSORPC_TMUX_SCHEDULER_UUID = "TENSORPC_TMUX_SCHEDULER_UUID"

class TmuxSchedulerEnvVariables:
    port: Optional[int]
    def __init__(self) -> None:
        port = os.environ.get(TENSORPC_TMUX_TASK_SCHEDULER_PORT)
        if port is not None:
            self.port = int(port)
        else:
            self.port = None 
        self.uid = os.environ.get(TENSORPC_TMUX_TASK_UID, "")


        