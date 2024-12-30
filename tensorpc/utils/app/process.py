import psutil 
import dataclasses 
from typing import List 

TENSORPC_FLOW_PROCESS_NAME_PREFIX = "__tensorpc_flow_app"

@dataclasses.dataclass
class AppProcessMeta:
    name: str
    pid: int
    grpc_port: int
    port: int
    app_grpc_port: int
    app_port: int


def list_all_app_in_machine():
    res: List[AppProcessMeta] = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        proc_name = proc.info["name"]
        proc_cmdline = proc.info["cmdline"]

        if proc_name.startswith(TENSORPC_FLOW_PROCESS_NAME_PREFIX):
            parts = proc_name.split("-")
            ports = list(map(int, parts[1:-1]))
            meta = AppProcessMeta(parts[-1], proc.info["pid"], ports[0],
                                  ports[1], ports[2], ports[3])
            res.append(meta)
            continue 
        if proc_cmdline and proc_cmdline[0].startswith(TENSORPC_FLOW_PROCESS_NAME_PREFIX):
            parts = proc_cmdline[0].split("-")
            ports = list(map(int, parts[1:-1]))
            meta = AppProcessMeta(parts[-1], proc.info["pid"], ports[0],
                                  ports[1], ports[2], ports[3])
            res.append(meta)
    return res
