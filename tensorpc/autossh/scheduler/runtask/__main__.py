import base64
import io 

import os
import traceback
from typing import Optional 
import fire
import json
from tensorpc import simple_remote_call, RemoteManager
from tensorpc.autossh.scheduler.core import TaskType 
from tensorpc.autossh.scheduler.constants import TmuxSchedulerEnvVariables
from tensorpc.autossh.scheduler.task_client import enter_task_conetxt
from tensorpc.autossh.serv_names import serv_names
import subprocess
import importlib
import inspect 
import psutil

def run_func_in_module(module_func_id: str, *args, **kwargs):
    # module_func_id: tensorpc.xxx.yyy::zzz
    parts = module_func_id.split("::")
    module_import_path = parts[0]
    local_parts = parts[1:]
    mod = importlib.import_module(module_import_path)
    module_dict = mod.__dict__
    func_obj = module_dict[local_parts[0]]
    for part in local_parts[1:]:
        func_obj = getattr(func_obj, part)
    assert inspect.isfunction(func_obj) or inspect.isbuiltin(func_obj)
    print(func_obj)
    return func_obj(*args, **kwargs)

def main(type_int: int):
    type = TaskType(type_int)
    # send pid to scheduler
    pid = os.getpid()
    env_vars = TmuxSchedulerEnvVariables()
    assert env_vars.port is not None 
    scheduler_url = f"localhost:{env_vars.port}"
    # tell scheduler we are ready, fetch params if task is func id task.
    # here we must tell scheduler our pid to ensure
    # scheduler can access our status.
    res = simple_remote_call(scheduler_url, serv_names.SCHED_TASK_INIT, env_vars.uid, pid)
    assert res is not None 
    command, func_id_params = res
    if type == TaskType.Command:
        try:
            # print(command)
            subprocess.run(command, shell=True, check=True)
            simple_remote_call(scheduler_url, serv_names.SCHED_TASK_SET_FINISHED, env_vars.uid)

        except:
            ss = io.StringIO()
            traceback.print_exc(file=ss)
            simple_remote_call(scheduler_url, serv_names.SCHED_TASK_SET_EXCEPTION, env_vars.uid, ss.getvalue())
    else:
        assert func_id_params is not None
        assert len(func_id_params) == 1, "currently only support one param"
        kwargs = func_id_params[0]
        with RemoteManager(scheduler_url) as robj:
            with enter_task_conetxt(robj):
                try:
                    # print(command)
                    run_func_in_module(command, **kwargs)
                    robj.remote_call(serv_names.SCHED_TASK_SET_FINISHED, env_vars.uid)
                except:
                    traceback.print_exc()
                    ss = io.StringIO()
                    traceback.print_exc(file=ss)
                    robj.remote_call(serv_names.SCHED_TASK_SET_EXCEPTION, env_vars.uid, ss.getvalue())
    # print("EXIT!!!")

if __name__ == "__main__":
    fire.Fire(main)