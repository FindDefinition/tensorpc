import psutil
import dataclasses
def list_all_dbg_server_in_machine():
    res: List[AppProcessMeta] = []
    for proc in psutil.process_iter(['pid', 'name']):
        proc_name = proc.info["name"]
        if proc_name.startswith(constants.TENSORPC_FLOW_PROCESS_NAME_PREFIX):
            ports = list(map(int, proc_name.split("-")[1:]))
            meta = AppProcessMeta(proc_name, proc.info["pid"], ports[0],
                                  ports[1], ports[2], ports[3])
            res.append(meta)
    return res
