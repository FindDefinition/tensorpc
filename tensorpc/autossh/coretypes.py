import enum
from typing import Dict, Any, List, Optional, Tuple
from typing_extensions import Literal
import tensorpc.core.dataclass_dispatch as dataclasses


@dataclasses.dataclass
class SSHTarget:
    hostname: str
    port: int
    username: str
    password: str
    known_hosts: Optional[str] = None
    client_keys: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    uid: str = ""
    forward_port_pairs: List[Tuple[int, int]] = dataclasses.field(
        default_factory=list)
    remote_forward_port_pairs: List[Tuple[int, int]] = dataclasses.field(
        default_factory=list)
    init_commands: str = ""

    @property
    def url(self):
        return f"{self.hostname}:{self.port}"

    @staticmethod
    def create_fake_target():
        return SSHTarget("localhost", 22, "root", "root")

    def is_localhost(self):
        return self.hostname == "localhost"

@dataclasses.dataclass
class TaskWrapperArgs:
    cmd: str
    password: str
    username: str = "root"
    max_retries: int = 1
    log_path: Optional[str] = None
    # rate limit for sending message to slack or other notification service
    msg_throttle: int = 30
    # distributed arguments
    master_url: Optional[str] = None
    num_workers: int = 1
    init_timeout: int = 60
    # message parser arguments
    # called every time a new line event is received
    # sig: def func(events: list[Event], current_cmd: str)
    msg_handler: Optional[str] = None
    # called once after init, can be used to log environment info
    # to file.
    init_info_getter: Optional[str] = None
    # called when error happens.
    # sig: def func(title: str, msg: str)
    error_handle_throttle: int = 15
    error_handler: Optional[str] = None
    # if set, log traceback of all child process to file (main-thread only)
    # only enabled if log_path is set and pyspy_period is set, 
    # won't be logged to stdout.
    pyspy_period: Optional[int] = None
    request_pty: bool = False

    @staticmethod 
    def empty():
        return TaskWrapperArgs("", "")

@dataclasses.dataclass
class TaskWrapperWorkerState:
    status: Literal["idle", "running", "error", "done"]
    addr: str
    last_timestamp: int = -1
    is_master: bool = False
    init_timestamp: int = -1
    pid: Optional[int] = None
    userinfo: Any = None
