import asyncio
from pathlib import Path
import psutil 
import json 
from typing import Any, Optional
import traceback

from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.core.inspecttools import get_co_qualname_from_frame

@dataclasses.dataclass
class PyspyFrame:
    name: str 
    filename: str 
    module: Optional[str]
    short_filename: str
    line: int 
    locals: Optional[Any] = None
    # for mui AutoComplete
    label: str = ""

@dataclasses.dataclass
class PyspyTrace:
    pid: int 
    thread_id: int 
    thread_name: str
    frames: list[PyspyFrame]


async def get_process_traceback_by_pyspy(pid: int, with_locals: bool = False) -> Any:
    cmd = [
        "py-spy", "dump", f"--pid={pid}", "-j", "--nonblocking"
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

async def get_all_subprocess_traceback_by_pyspy(pid: int, ignore_error: bool = True):
    current_process = psutil.Process(pid)
    children = current_process.children(recursive=True)
    name_to_pid_to_tb: dict[str, dict[int, Any]] = {}
    for child in children:
        try:
            info = psutil.Process(child.pid).as_dict(attrs=["name", "cmdline"])
        except psutil.NoSuchProcess:
            continue
        name = _determine_proc_name(info)
        if name not in name_to_pid_to_tb:
            name_to_pid_to_tb[name] = {}
        try:
            tb_res = await get_process_traceback_by_pyspy(child.pid)
        except ValueError as e:
            if ignore_error:
                print(f"Failed to get traceback for pid {child.pid}: {e}")
                name_to_pid_to_tb[name][child.pid] = []
                continue
            else:
                raise e
        name_to_pid_to_tb[name][child.pid] = tb_res
    return name_to_pid_to_tb

def _proc_info_iter(pid: int):
    root_process = psutil.Process(pid)
    for child in root_process.children(recursive=True):
        try:
            info = psutil.Process(child.pid).as_dict(attrs=["pid", "name", "cmdline"])
            yield info
        except psutil.NoSuchProcess:
            continue

def _proc_info_iter_all():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        yield proc.info

async def _get_torchrun_traceback_by_pyspy(main_thread_only: bool = True, is_data_worker: bool = False, root_pid: int = -1, ignore_error: bool = False):
    # 1. locate torchrun process named `pt_elastic``
    if root_pid != -1:
        proc_iter = _proc_info_iter(root_pid)
    else:
        proc_iter = _proc_info_iter_all()
    main_pid: int = -1
    for info in proc_iter:
        proc_name = info["name"]
        if proc_name == "pt_elastic":
            main_pid = info["pid"]
            break 
    name_to_pid_to_tb: dict[str, dict[int, Any]] = {}

    if main_pid == -1:
        return name_to_pid_to_tb
    # 2. get all subprocess except data worker named `pt_data_worker`
    current_process = psutil.Process(main_pid)
    children = current_process.children(recursive=is_data_worker)
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
        except psutil.NoSuchProcess:
            continue
        name = _determine_proc_name(info)
        if name not in name_to_pid_to_tb:
            name_to_pid_to_tb[name] = {}
        try:
            tb_res = await get_process_traceback_by_pyspy(child.pid)
        except ValueError as e:
            if ignore_error:
                print(f"Failed to get traceback for pid {child.pid}: {e}")
                name_to_pid_to_tb[name][child.pid] = []
                continue
            else:
                raise e
        if main_thread_only:
            single_res_to_filter = []
            for item in tb_res:
                if item["thread_name"] == "MainThread":
                    single_res_to_filter.append(item)
                    break 
            tb_res = single_res_to_filter
        name_to_pid_to_tb[name][child.pid] = tb_res
    return name_to_pid_to_tb

async def get_torchrun_traceback_by_pyspy(main_thread_only: bool = True, root_pid: int = -1, ignore_error: bool = False):
    return await _get_torchrun_traceback_by_pyspy(main_thread_only, is_data_worker=False, root_pid=root_pid, ignore_error=ignore_error)

async def get_torchrun_dataworker_traceback_by_pyspy(main_thread_only: bool = True, root_pid: int = -1):
    return await _get_torchrun_traceback_by_pyspy(main_thread_only, is_data_worker=True)

def get_pyspy_style_asyncio_task_traceback():
    all_tasks = asyncio.all_tasks()
    res_traces: dict[int, list[dict]] = {}
    for i, task in enumerate(all_tasks):
        task_id = f"{task.get_name()}-{i}"
        stack = task.get_stack()
        if not stack:
            continue 
        # stack to PyspyFrame
        frames = []
        for cur_frame in stack:
            fname = cur_frame.f_code.co_filename
            if cur_frame.f_lineno is None:
                continue
            qname = get_co_qualname_from_frame(cur_frame)
            module = ".".join(qname.split(".")[:-1])
            if module == "":
                module = None
            frame_info = PyspyFrame(
                    name=cur_frame.f_code.co_name, 
                    filename=cur_frame.f_code.co_filename, 
                    line=cur_frame.f_lineno, module=module, 
                    short_filename=Path(fname).name)
            frames.append(frame_info)
        trace = PyspyTrace(
            pid=i,
            thread_id=i,
            thread_name=task_id,
            frames=frames,
        )
        res_traces[i] = [dataclasses.asdict(trace)]
    return res_traces

def _main():
    import rich 
    # res = asyncio.run(get_torchrun_traceback_by_pyspy(main_thread_only=True))
    res = asyncio.run(get_process_traceback_by_pyspy(185137))

    rich.print(res)
    
if __name__ == "__main__":
    _main()