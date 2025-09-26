import enum
from typing_extensions import Annotated 
import tensorpc.core.dataclass_dispatch as dataclasses
from typing import Any, Awaitable, Callable, Optional, Union

from tensorpc.core.datamodel.draft import DraftFieldMeta
from tensorpc.core.distributed.ftgroup import FTStatus, FTStateBase

class CmdStatus(enum.IntEnum):
    IDLE = 0
    RUNNING = 1
    # when some rank is restarted during cmd running,
    # master will enter this state and try to restart all workers with 
    # same cmd.
    DURING_RESTART = 2

class SSHStatus(enum.IntEnum):
    IDLE = 0
    DISCONNECTED = 1
    RUNNING = 2
    ERROR = 3

class CheckpointType(enum.IntEnum):
    # same as standard checkpoint
    TRAIN_MAJOR = 0
    # fast ckpt cache
    TRAIN_MINOR = 1
    # infer only ckpt cache
    FIXED = 2

@dataclasses.dataclass
class CheckpointMetadata:
    type: CheckpointType
    # key to identify the different model
    key: str
    # train step, for fixed checkpoint cache, this is None.
    step: Optional[int] = None
    rank: int = 0

@dataclasses.dataclass
class FTSSHServerArgs:
    rank: int
    world_size: int
    # used to save ip of each worker to a folder to ensure
    # failed worker can discover the master
    # assume your cluster has a NAS.
    # also save state.
    workdir: str
    cmd: str
    password: str
    username: str = "root"
    # max_retries: int = 1
    # log_path: Optional[str] = None
    # distributed arguments
    master_discovery_fn: Optional[str] = None
    heartbeat_interval: int = 5
    # 5 min
    # when some worker or master disconnected, we assume
    # your cluster manager will restart it. so we 
    # wait for 5 min to check if the worker is really.
    disconnect_total_retry: int = 120
    disconnect_rpc_check_timeout: int = 2
    # cmd shutdown configs
    cmd_shutdown_timeout: int = 10
    cmd_ctrl_c_retry: int = 3

    nproc_per_node: int = -1

    logdir: str = ""

    cmd_retry_when_reconnect: bool = True
    env_fwd_re: str = ""
    local_ssh_port: int = 22
    log_to_stdout: bool = False

@dataclasses.dataclass
class FTState(FTStateBase):
    label: str = ""
    cur_cmd: Annotated[Optional[str], DraftFieldMeta(is_external=True)] = None
    ssh_status: SSHStatus = SSHStatus.IDLE
    # when enabled, your distributed problem will enter breakpoint
    is_user_control_enabled: Annotated[bool, DraftFieldMeta(is_external=True)] = False
    num_bkpt_proc: int = 0
    title_msg: str = ""

    def is_state_equal_non_external(self, other: "FTState") -> bool:
        return (
            self.label == other.label
            and self.rank == other.rank
            and self.ip == other.ip
            and self.port == other.port
            and self.is_master == other.is_master
            and self.status == other.status
            and self.ssh_status == other.ssh_status
            and self.num_bkpt_proc == other.num_bkpt_proc
        )

@dataclasses.dataclass
class MasterUIState:
    cmd_status: CmdStatus
    client_states: Annotated[list[FTState], DraftFieldMeta(is_store_external=True)] = dataclasses.field(default_factory=list)
    selected_client_state: Optional[dict[str, Any]] = None
    cmd: str = "echo $HOME"
    cmd_history: list[str] = dataclasses.field(default_factory=list)
    pending_ctrl: list[Any] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class FTStatusBoxState:
    id: str
    rank: int 
    ip: str
    status: FTStatus
    ssh_status: SSHStatus
    color: str
    selected: bool
    num_bkpt_proc: int = 0
    @staticmethod 
    def from_ft_state(ft_state: FTState, selected: bool):
        if ft_state.status == FTStatus.WORKER_DISCONNECTED:
            color = "orange"
        elif ft_state.status == FTStatus.UNKNOWN:
            color = "gray"
        else:
            if ft_state.ssh_status == SSHStatus.IDLE:
                color = "blue"
            elif ft_state.ssh_status == SSHStatus.DISCONNECTED:
                color = "red"
            elif ft_state.ssh_status == SSHStatus.RUNNING:
                color = "lime"
            elif ft_state.ssh_status == SSHStatus.ERROR:
                color = "red"
            else:
                color = "gray"
        return FTStatusBoxState(
            id=str(ft_state.rank),
            rank=ft_state.rank,
            ip=ft_state.ip,
            status=ft_state.status,
            ssh_status=ft_state.ssh_status,
            color=color,
            selected=selected,
            num_bkpt_proc=ft_state.num_bkpt_proc,
        )


class MasterActions(enum.Enum):
    RECONNECT_ALL_CLIENT = "Reconnect All Client"
    CLEAR_ALL_CKPT = "Clear All Checkpoint"
    CLEAR_ALL_TERMINALS = "Clear All Terminals"

    SHUTDOWN_ALL = "Shutdown All"
    KILL_ALL = "KILL ALL"
    START_OR_CANCEL = "Start/Cancel"


class UILocalActions(enum.Enum):
    PYTORCH_SPY = "_local_PYTORCH_SPY"
    INTERNAL_DEBUG = "_local_INTERNAL_DEBUG"


class CheckpointActions(enum.Enum):
    LOAD_ITEM = "LOAD_ITEM"
    # DELETE_ITEM = "Delete Item"
    SAVE = "SAVE"


class PyspyTraceMode(enum.IntEnum):
    PYTORCH_DISTRIBUTED = 0
    ALL_SUBPROCESS = 1
    LOCAL_AIO_TASKS = 2
    SERVER_PROCESS = 3
    PYTORCH_LOCAL = 4
