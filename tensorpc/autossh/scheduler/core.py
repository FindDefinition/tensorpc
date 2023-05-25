import dataclasses 

import enum
from typing import Dict, Any, List, Optional, Tuple
from tensorpc.autossh.coretypes import SSHTarget
from .constants import TMUX_SESSION_TASK_PREFIX, TMUX_SESSION_NAME_SPLIT
class TaskStatus(enum.Enum):
    Unknown = 0
    Pending = 1 
    Running = 2 
    Finished = 3 
    Failed = 4 
    # this means task send finish message to scheduler, but the process is still running
    AlmostFinished = 5
    # user send cancel to task, task need to check this manually and do graceful exit
    NeedToCancel = 6
    AlmostCanceled = 7
    Canceled = 8

    Booting = 9


ALL_RUNNING_STATUS = set([TaskStatus.Running, TaskStatus.AlmostFinished, TaskStatus.AlmostCanceled, TaskStatus.Booting, TaskStatus.NeedToCancel])

ALL_CTRL_C_CANCELABLE_STATUS = set([TaskStatus.Running, TaskStatus.Booting, TaskStatus.NeedToCancel])

ALL_KILLABLE_STATUS = ALL_RUNNING_STATUS

class TaskType(enum.Enum):
    # shell command
    Command = 0
    # function id and args
    FunctionId = 1

class ResourceType(enum.Enum):
    GPU = 0
    CPU = 1

@dataclasses.dataclass 
class TaskOutput:
    userdata: Dict[str, Any]
    paths: List[str]
    timestamp: int 
    progress: float


@dataclasses.dataclass 
class TaskState:
    status: TaskStatus
    progress: float 
    outputs: List[TaskOutput]
    pid: int 
    exception_str: str
    # timestamp when updated
    timestamp: int = -1
    # resources used by this task such as GPUs.
    resources: List[Tuple[ResourceType, int]] = dataclasses.field(default_factory=list)
    tmux_pane_last_lines: str = ""

@dataclasses.dataclass 
class Task:
    type: TaskType
    # if type is Command, command is shell command
    # otherwise, command is function id
    command: str  
    # when provide params as list, this task
    # will be executed multiple times with different params
    params: Optional[List[Dict[str, Any]]] = None
    # set by scheduler
    # if user provide a id, scheduler use this id, otherwise scheduler generate a uuid
    # if user submit a id that already exists, scheduler will run iff task is not running 
    # (status is pending, failed, cancelled or finished)
    id: str = ""
    # timestamp when submitted
    create_timestamp: int = -1
    
    num_gpu_used: int = 0
    # if user provide allowed targets, scheduler will only run this task on these targets
    allowed_target_ips: Optional[List[str]] = None 
    state: TaskState = dataclasses.field(default_factory=lambda: TaskState(TaskStatus.Pending, 0.0, [], -1, ""))
    # if false, the tmux session will be closed after task finished or failed,
    # user can't check the stdout or error message.
    keep_tmux_session: bool = True

    name: Optional[str] = None 

    desp: str = ""

    cwd: str = ""

    tags: List[str] = dataclasses.field(default_factory=list)

    def empty_state(self):
        self.state = TaskState(TaskStatus.Pending, 0.0, [], -1, "")
        return self 

    def push_params(self, **kwargs):
        if self.params is None:
            self.params = []
        self.params.append(kwargs)
        return self 
    
    def get_tmux_session_name(self):
        return f"{TMUX_SESSION_TASK_PREFIX}{TMUX_SESSION_NAME_SPLIT}{self.id}"
    
