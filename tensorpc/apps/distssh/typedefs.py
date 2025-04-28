import enum
from typing_extensions import Annotated 
import tensorpc.core.dataclass_dispatch as dataclasses
from typing import Any, Awaitable, Callable, Optional, Union

from tensorpc.core.datamodel.draft import DraftFieldMeta


class CmdStatus(enum.IntEnum):
    IDLE = 0
    RUNNING = 1
    # when some rank is restarted during cmd running,
    # master will enter this state and try to restart all workers with 
    # same cmd.
    DURING_RESTART = 2

class FTStatus(enum.IntEnum):
    OK = 0
    MASTER_DISCONNECTED = 1
    WORKER_DISCONNECTED = 2
    UNKNOWN = 3

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
    disconnect_total_retry: int = 60
    disconnect_rpc_check_timeout: int = 2
    # cmd shutdown configs
    cmd_shutdown_timeout: int = 10
    cmd_ctrl_c_retry: int = 3

    logdir: str = ""

    cmd_retry_when_reconnect: bool = True
    env_fwd_re: str = ""

@dataclasses.dataclass
class FTState:
    label: str
    rank: int
    ip: str
    port: int
    is_master: bool 
    cur_cmd: Annotated[Optional[str], DraftFieldMeta(is_external=True)] = None
    status: FTStatus = FTStatus.OK
    ssh_status: SSHStatus = SSHStatus.IDLE
    # backend only states don't need to send to frontend
    uuid: Annotated[str, DraftFieldMeta(is_external=True)] = ""
    master_uuid: Annotated[str, DraftFieldMeta(is_external=True)] = ""
    master_ip: Annotated[str, DraftFieldMeta(is_external=True)] = ""
    # when enabled, your distributed problem will enter breakpoint
    is_user_control_enabled: Annotated[bool, DraftFieldMeta(is_external=True)] = False
    num_bkpt_proc: int = 0

@dataclasses.dataclass
class MasterUIState:
    cmd_status: CmdStatus
    client_states: list[FTState]
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