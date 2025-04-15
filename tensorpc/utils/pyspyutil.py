import asyncio
from pathlib import Path
import psutil 
import json 
from typing import Any
import traceback

async def get_process_traceback_by_pyspy(pid: int, with_locals: bool = False) -> Any:
    cmd = [
        "py-spy", "dump", f"--pid={pid}", "-j", # "--nonblocking"
    ]
    if with_locals:
        cmd.append("--locals")
    p = await asyncio.create_subprocess_exec(
        *cmd, 
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )   
    stdout_data, stderr_data = await p.communicate()

    if p.returncode == 0:
        return json.loads(stdout_data.decode("utf-8"))
    else:
        raise ValueError(f"Failed to get traceback for pid {pid}: {stderr_data.decode('utf-8')}")

async def get_all_process_traceback_with_prefix_by_pyspy(prefix: str, main_thread_only: bool = True, with_locals: bool = False) -> list[Any]:
    """don't work in docker.
    """
    try:
        pids: list[int] = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            proc_name = proc.info["name"]
            proc_cmdline = proc.info["cmdline"]
            if proc_name.startswith(prefix):
                pids.append(proc.pid)
                continue 
            if proc_cmdline and proc_cmdline[0].startswith(prefix):
                pids.append(proc.pid)
        res = []
        pids.sort()
        for pid in pids:
            single_res = await get_process_traceback_by_pyspy(int(pid), with_locals)
            if main_thread_only:
                single_res_to_filter = []
                for item in single_res:
                    if item["thread_name"] == "MainThread":
                        single_res_to_filter.append(item)
                        break 
                single_res = single_res_to_filter
            res.append(single_res)
        return res
    except:
        traceback.print_exc()
        return []

def _determine_proc_name(info: dict):
    # assume cmd without whitespace is setted by user code.
    proc_name = info["name"]
    proc_cmdline = info["cmdline"]
    candidates: list[str] = [proc_name]
    if proc_cmdline:
        if len(proc_cmdline) > 1:
            return proc_name
        candidates.append(proc_cmdline[0])
    for candidate in candidates:
        if " " not in candidate:
            return candidate
    return proc_name

async def get_all_subprocess_traceback_by_pyspy(pid: int):
    current_process = psutil.Process(pid)
    children = current_process.children(recursive=True)
    name_to_pid_to_tb: dict[str, dict[int, Any]] = {}
    for child in children:
        try:
            info = psutil.Process(child.pid).as_dict(attrs=["name", "cmdline"])
            tb_res = await get_process_traceback_by_pyspy(child.pid)
        except psutil.NoSuchProcess:
            continue
        name = _determine_proc_name(info)
        if name not in name_to_pid_to_tb:
            name_to_pid_to_tb[name] = {}
        name_to_pid_to_tb[name][child.pid] = tb_res
    return name_to_pid_to_tb

async def _get_torchrun_traceback_by_pyspy(main_thread_only: bool = True, is_data_worker: bool = False):
    import torch 
    torch_path = Path(torch.__file__).parent
    # 1. locate torchrun process named `pt_elastic``
    main_pid: int = -1
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        proc_name = proc.info["name"]
        if proc_name == "pt_elastic":
            main_pid = proc.pid
            break 
    if main_pid == -1:
        raise ValueError("pt_elastic process not found.")
    # 2. get all subprocess except data worker named `pt_data_worker`
    current_process = psutil.Process(main_pid)
    children = current_process.children(recursive=True)
    name_to_pid_to_tb: dict[str, dict[int, Any]] = {}
    for child in children:
        try:
            info = psutil.Process(child.pid).as_dict(attrs=["name", "cmdline"])
            ignore_proc_found = False
            for item in info["cmdline"]:
                if "compile_worker" in item:
                    # ignore torch inductor compile worker
                    ignore_proc_found = True 
                    break 
            if ignore_proc_found:
                continue
            # if child.status() != psutil.STATUS_RUNNING:
            #     continue
            if is_data_worker:
                if info["name"] != "pt_data_worker":
                    continue 
            else:
                if info["name"] == "pt_data_worker":
                    continue
            tb_res = await get_process_traceback_by_pyspy(child.pid)
        except psutil.NoSuchProcess:
            continue
        if main_thread_only:
            single_res_to_filter = []
            for item in tb_res:
                if item["thread_name"] == "MainThread":
                    single_res_to_filter.append(item)
                    break 
            tb_res = single_res_to_filter
        name = _determine_proc_name(info)
        if name not in name_to_pid_to_tb:
            name_to_pid_to_tb[name] = {}
        name_to_pid_to_tb[name][child.pid] = tb_res
    return name_to_pid_to_tb

async def get_torchrun_traceback_by_pyspy(main_thread_only: bool = True):
    return await _get_torchrun_traceback_by_pyspy(main_thread_only, is_data_worker=False)

async def get_torchrun_dataworker_traceback_by_pyspy(main_thread_only: bool = True):
    return await _get_torchrun_traceback_by_pyspy(main_thread_only, is_data_worker=True)

def _main():
    import rich 
    # res = asyncio.run(get_torchrun_traceback_by_pyspy(main_thread_only=True))
    res = asyncio.run(get_process_traceback_by_pyspy(185137))

    rich.print(res)
    
if __name__ == "__main__":
    _main()