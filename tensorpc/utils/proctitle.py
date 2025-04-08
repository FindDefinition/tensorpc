import traceback
from typing import Optional, Union
import psutil

from tensorpc.constants import TENSORPC_SERVER_PROCESS_NAME_PREFIX
from tensorpc.core import BuiltinServiceProcType
from tensorpc.core.tree_id import UniqueTreeId 
import dataclasses

@dataclasses.dataclass
class TensorpcServerProcessMeta:
    pid: int
    name: str
    type: BuiltinServiceProcType
    args: list[str]

def list_all_tensorpc_server_in_machine(target_proc_type: Optional[Union[set[BuiltinServiceProcType], BuiltinServiceProcType]] = None, parent_pid: Optional[int] = None):
    # format: __tensorpc_server-unique_id
    if target_proc_type is not None:
        if isinstance(target_proc_type, BuiltinServiceProcType):
            target_proc_type = {target_proc_type}
    res: list[TensorpcServerProcessMeta] = []
    if parent_pid is None:
        proc_iter = psutil.process_iter(['pid', 'name', 'cmdline'])
    else:
        proc_main = psutil.Process(parent_pid)
        proc_iter = proc_main.children(recursive=True)
    for proc in proc_iter:
        proc_name: str = proc.info["name"]
        proc_cmdline = proc.info["cmdline"]
        try:

            if proc_name.startswith(TENSORPC_SERVER_PROCESS_NAME_PREFIX):
                first_split = proc_name.find("-")
                uid_encoded = proc_name[first_split + 1:]
                uid_obj = UniqueTreeId(uid_encoded)
                proc_type = BuiltinServiceProcType(int(uid_obj.parts[0]))
                res.append(TensorpcServerProcessMeta(proc.info["pid"], proc_name, proc_type, uid_obj.parts[1:]))
                continue 
            if proc_cmdline and proc_cmdline[0].startswith(TENSORPC_SERVER_PROCESS_NAME_PREFIX):
                proc_name = proc_cmdline[0]
                first_split = proc_name.find("-")
                uid_encoded = proc_name[first_split + 1:]
                uid_obj = UniqueTreeId(uid_encoded)
                proc_type = BuiltinServiceProcType(int(uid_obj.parts[0]))
                res.append(TensorpcServerProcessMeta(proc.info["pid"], proc_name, proc_type, uid_obj.parts[1:]))
        
        except:
            traceback.print_exc()
            continue
    if target_proc_type is not None:
        res = [r for r in res if r.type in target_proc_type]
    return res

def get_tensorpc_server_process_title(type: BuiltinServiceProcType, *args: str) -> str:
    """Get the process title of the tensorpc server."""
    uid_encoded = UniqueTreeId.from_parts([str(type.value)] + list(args)).uid_encoded
    title = f"{TENSORPC_SERVER_PROCESS_NAME_PREFIX}-{uid_encoded}"
    return title

def set_tensorpc_server_process_title(type: BuiltinServiceProcType, *args: str):
    import setproctitle  # type: ignore
    title = get_tensorpc_server_process_title(type, *args)
    setproctitle.setproctitle(title)
    return title
