from typing import Any, Literal, Optional, Self
from typing_extensions import TypedDict
from tensorpc.autossh.core import SSHConnDesc
from tensorpc.core import dataclass_dispatch as dataclasses

import enum

from tensorpc.core.distributed.comm.grpcimpl import AsyncGRPCCommConfig
from tensorpc.utils import rich_logging 
from tensorpc.core.distributed.raft import LeaderQueryResultBase, PeerInfo, RaftConfig
from tensorpc.dock.components import mui

CM_LOGGER = rich_logging.get_logger("tensorpc.apps.cm")

class UserCmdType(enum.IntEnum):
    SHELL_CMD = 0
    TRY_CTRL_C = 1

    RECONNECT_SSH = 2
    REGISTER_WORKER = 3
    UNREGISTER_WORKER = 4
    KILL_TO_IDLE = 5

class RuntimeCmdType(enum.IntEnum):
    RELEASE_BREAKPOINT = 0
    LAUNCH_CUSTOM_RECORD = 1
    DISABLE_ALL_BREAKPOINTS = 2
    ENABLE_ALL_BREAKPOINTS = 3

@dataclasses.dataclass
class ResourceInfo:
    num_cpu: int 
    num_mem_gb: int
    num_gpu: int
    gpu_type: Optional[str] = None

    def is_sufficient_for(self, other: "ResourceInfo") -> bool:
        if self.num_cpu < other.num_cpu:
            return False 
        if self.num_mem_gb < other.num_mem_gb:
            return False 
        if self.num_gpu < other.num_gpu:
            return False 
        # we don't compare gpu type for now.
        return True


@dataclasses.dataclass
class WorkerInfo:
    peer_info: PeerInfo
    rank: int
    resource: Optional[ResourceInfo] = None

    @property 
    def uid(self) -> str:
        return self.peer_info.uid

    def replace_uid(self, new_uid: str) -> Self:
        return dataclasses.replace(self, peer_info=self.peer_info.replace_uid(new_uid))

@dataclasses.dataclass
class UserCmd:
    type: UserCmdType
    content: str = ""
    # for register/unregister worker cmd
    worker_info: Optional[WorkerInfo] = None

@dataclasses.dataclass
class UserCmdQueryResult:
    cmd: Optional[UserCmd]
    leader_id: Optional[str]

@dataclasses.dataclass
class SSHWorkerConfig:
    local_ssh_desc: SSHConnDesc
    comm_cfg: AsyncGRPCCommConfig
    raft_cfg: RaftConfig
    # we will look for last user raft log to determine what should we do
    # next.  
    leader_check_interval: int = 5
    worker_check_interval: int = 5

    worker_disconnect_timeout: int = 30
    worker_update_debounce: float = 0.5
    log_to_stdout: bool = False 
    env_fwd_re: str = ""
    workdir: str = ""
    enable_debug_panel: bool = True


@dataclasses.dataclass
class CMNodeManagerArgs:
    # used to save ip of each worker to a folder to ensure
    # failed worker can discover the master
    # assume your cluster has a NAS.
    # also save state.
    password: str
    username: str = "root"
    workdir: str = ""

    logdir: str = ""

    env_fwd_re: str = ""
    local_ssh_port: int = 22
    log_to_stdout: bool = False
    # -1 means use all resources.
    num_cpu: int = -1
    num_gpu: int = -1
    num_mem_gb: int = -1

    # default worker cfgs
    leader_check_interval: int = 5
    worker_check_interval: int = 5

    worker_disconnect_timeout: int = 30

    # default comm config
    master_comm_cfg: AsyncGRPCCommConfig = dataclasses.field(default_factory=lambda: AsyncGRPCCommConfig(max_fail_before_mark_down=-1, common_timeout=8))
    worker_comm_cfg: AsyncGRPCCommConfig = dataclasses.field(default_factory=lambda: AsyncGRPCCommConfig(max_fail_before_mark_down=5, common_timeout=5))
    raft_cfg: RaftConfig = dataclasses.field(default_factory=lambda: RaftConfig(apply_sync=True))

    def get_worker_cfg(self) -> SSHWorkerConfig:
        return SSHWorkerConfig(
            local_ssh_desc=SSHConnDesc(
                url_with_port="localhost:{}".format(self.local_ssh_port),
                username=self.username,
                password=self.password,
            ),
            comm_cfg=self.worker_comm_cfg,
            raft_cfg=self.raft_cfg,
            workdir=self.workdir,
            env_fwd_re=self.env_fwd_re,
            log_to_stdout=self.log_to_stdout,
            leader_check_interval=self.leader_check_interval,
            worker_check_interval=self.worker_check_interval,
            worker_disconnect_timeout=self.worker_disconnect_timeout,
        )

@dataclasses.dataclass
class ClusterBaseInfo:
    id: str 
    name: str 
    provider: str

class NodeFlags(enum.IntFlag):
    IS_RAFT_NODE = enum.auto()
    IS_COMPUTE_NODE = enum.auto()
    IS_RAFT_LEADER = enum.auto()

@dataclasses.dataclass
class GroupNodeSpec:
    peer_info: PeerInfo 
    flags: NodeFlags
    resource: Optional[ResourceInfo] = None
    # store url from provider in cluster panel.
    # cluster panel can use this to build a map
    # between node id in server and node id from provider.
    url_from_provider: Optional[str] = None

@dataclasses.dataclass
class TaskGroupInfo:
    id: str
    cluster_id: str 
    cluster_name: str 

    group_name: str
    num_nodes: int 
    num_cpu: int 
    num_gpu: int 

    # plain shell based group, or compute flow managed group
    # type: str
    status: str
    # only for display.
    cur_cmd: str = ""

    dragData: dict[str, Any] = dataclasses.field(default_factory=dict)
    color: mui.StdColorNoDefault = "inherit"
    tags: list[mui.ChipGroupItem] = dataclasses.field(default_factory=list)
    worker_last_activity: str = ""

class WorkerUIType(enum.Enum):
    TERMINAL = "terminal"
    RAFT_PANEL = "raft_panel"

class RaftMgrActions(enum.Enum):
    RECONNECT_ALL_CLIENT = "Reconnect All Client"
    CLEAR_ALL_CKPT = "Clear All Checkpoint"
    CLEAR_ALL_TERMINALS = "Clear All Terminals"

    SHUTDOWN_ALL = "Shutdown All"
    KILL_ALL = "KILL ALL"
    START_OR_CANCEL = "Start/Cancel"

    PYTORCH_SPY = "PYTORCH_SPY"
    INTERNAL_DEBUG = "INTERNAL_DEBUG"


class GroupSSHStatus(enum.IntEnum):
    HAS_DISCONNECTED = 0
    HAS_RUNNING = 1
    HAS_PARTIAL_RUNNING = 2

    # indicate some child hit tensorpc.dbg.breakpoint and paused.
    HAS_PAUSED_PROCESS = 3
    ALL_IDLE_WITH_LAST_ERROR = 4
    ALL_IDLE_WITHOUT_ERROR = 5
    UNKNOWN = 6

@dataclasses.dataclass
class WorkerSSHStatus:
    status: Literal["running", "idle", "error", "disconnected"]
    last_ts: int 
    exit_code: Optional[int] = None
    is_paused: bool = False
    last_ssh_out_ts: int = -1

@dataclasses.dataclass
class WorkerSelectItem:
    id: str
    label: str 
    # disconnected, error, running, idle
    ssh_status: str
    url: str
    rank: int

@dataclasses.dataclass
class WorkerUISSHState:
    id: str
    ssh_status: WorkerSSHStatus
    world_size: int
    num_connected: int
    cur_worker: Optional[WorkerSelectItem] = None
    can_workers_run_cmd: bool = False
    workers: list[WorkerSelectItem] = dataclasses.field(default_factory=list)
    cur_leader_id: Optional[str] = None
    cur_leader_url: Optional[str] = None

    group_ssh_status: int = int(GroupSSHStatus.HAS_DISCONNECTED)

    user_cmd: str = "echo $HOME"
    num_paused: int = 0
    is_raft_node: bool = False

    is_user_control_enabled: bool = False

    worker_last_activity: str = ""


    @mui.DataModel.mark_pfl_query_func
    def get_common_query(self):
        is_leader = self.id == self.cur_leader_id
        connect_info = f"{self.num_connected}/{self.world_size}"
        if self.is_raft_node:
            if is_leader:
                header = "Raft Leader"
            else:
                header = "Raft Worker"
            if self.cur_leader_url is not None and not is_leader:
                header += f" (leader: {self.cur_leader_url})"
            header += " - " + connect_info
        else:
            header = "Worker"
        start_or_cancel_btn_icon = mui.IconType.Stop
        is_stop_btn_disabled = False
        if self.can_workers_run_cmd:
            start_or_cancel_btn_icon = mui.IconType.PlayArrow
            is_stop_btn_disabled = True 
        terminal_is_local = True
        cur_worker = self.cur_worker
        if cur_worker is not None:
            if cur_worker.id != self.id:
                terminal_is_local = False 
        worker_select_label = "Workers"
        if self.num_paused > 0:
            worker_select_label += f" ({self.num_paused} paused) "
        elif self.group_ssh_status == GroupSSHStatus.HAS_DISCONNECTED:
            worker_select_label += " (disconnected)"
        elif self.group_ssh_status == GroupSSHStatus.ALL_IDLE_WITHOUT_ERROR or self.group_ssh_status == GroupSSHStatus.ALL_IDLE_WITH_LAST_ERROR:
            worker_select_label += " (idle)"
        elif self.group_ssh_status == GroupSSHStatus.HAS_PARTIAL_RUNNING:
            if self.worker_last_activity != "":
                worker_select_label += f" (running partial|{self.worker_last_activity})"
            else:
                worker_select_label += " (running partial)"
        else:
            if self.worker_last_activity != "":
                worker_select_label += f" (running|{self.worker_last_activity})"
            else:
                worker_select_label += " (running)"
        return {
            "header": header,
            "connect_info": connect_info,
            "is_leader": is_leader,
            "start_or_cancel_icon": start_or_cancel_btn_icon,
            "not_leader_disabled": not is_leader,
            "stop_btn_disabled": is_stop_btn_disabled or not is_leader,
            "terminal_is_local": terminal_is_local,
            "worker_select_label": worker_select_label
        } 

@dataclasses.dataclass
class GroupCoarseStatus:
    id: str
    success: bool
    # tell client whether this node is raft node.
    is_raft_node: bool
    group_ssh_status: int = int(GroupSSHStatus.HAS_DISCONNECTED)
    last_cmd: Optional[UserCmd] = None
    leader_info: Optional[PeerInfo] = None
    worker_last_activity: str = ""

@dataclasses.dataclass
class LeaderUIStateResult(LeaderQueryResultBase):
    is_user_control_enabled: bool = False
