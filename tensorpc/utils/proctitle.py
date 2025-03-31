import psutil

from tensorpc.constants import TENSORPC_SERVER_PROCESS_NAME_PREFIX
from tensorpc.core.tree_id import UniqueTreeId 
import dataclasses

@dataclasses.dataclass
class TensorpcServerProcessMeta:
    pid: int
    args: list[str]

def list_all_tensorpc_server_in_machine():
    # format: __tensorpc_server-unique_id
    res: list[TensorpcServerProcessMeta] = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        proc_name: str = proc.info["name"]
        proc_cmdline = proc.info["cmdline"]

        if proc_name.startswith(TENSORPC_SERVER_PROCESS_NAME_PREFIX):
            first_split = proc_name.find("-")
            uid_encoded = proc_name[first_split + 1:]
            uid_obj = UniqueTreeId(uid_encoded)
            res.append(TensorpcServerProcessMeta(proc.info["pid"], uid_obj.parts))
            continue 
        if proc_cmdline and proc_cmdline[0].startswith(TENSORPC_SERVER_PROCESS_NAME_PREFIX):
            proc_name = proc_cmdline[0]
            first_split = proc_name.find("-")
            uid_encoded = proc_name[first_split + 1:]
            uid_obj = UniqueTreeId(uid_encoded)
            res.append(TensorpcServerProcessMeta(proc.info["pid"], uid_obj.parts))
    return res

def set_tensorpc_server_process_title(*args: str):
    uid_encoded = UniqueTreeId.from_parts(list(args)).uid_encoded
    title = f"{TENSORPC_SERVER_PROCESS_NAME_PREFIX}-{uid_encoded}"
    import setproctitle  # type: ignore
    setproctitle.setproctitle(title)
