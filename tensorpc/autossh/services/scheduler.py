import asyncio
import enum
from typing import Dict, List, Optional, Set, Tuple
from tensorpc.autossh.scheduler.core import ALL_CTRL_C_CANCELABLE_STATUS, ALL_KILLABLE_STATUS, ALL_RUNNING_STATUS, Task, TaskOutput, TaskStatus, TaskType, ResourceType
from tensorpc.autossh.scheduler import tmux
from tensorpc.core import marker, prim
import uuid
import time 
import psutil
import subprocess 
import dataclasses
import io 
import csv 
from tensorpc.utils.gpuusage import get_nvidia_gpu_measures

_SUPPORTED_SET_STATUS = set([TaskStatus.NeedToCancel])

class ResourceManager:
    def __init__(self, num_cpu: int, num_gpu: int) -> None:
        self.idle_resources: Dict[ResourceType, Set[Tuple[ResourceType, int]]] = {}
        self.occupied_resources: Dict[ResourceType, Set[Tuple[ResourceType, int]]] = {}
        self.num_gpu = num_gpu
        for item in ResourceType:
            self.idle_resources[item] = set()
            self.occupied_resources[item] = set()

        for i in range(num_cpu):
            self.idle_resources[ResourceType.CPU].add((ResourceType.CPU, i))
        for i in range(num_gpu):
            self.idle_resources[ResourceType.GPU].add((ResourceType.GPU, i))

    def __repr__(self):
        num_gpu_idle = len(self.idle_resources[ResourceType.GPU])
        return f"ResourceManager(GPU={num_gpu_idle}/{self.num_gpu})"

    def request_idle_cpus(self, num_cpu: int) -> List[Tuple[ResourceType, int]]:
        return self._request_idle_resources(ResourceType.CPU, num_cpu)
    
    def request_idle_gpus(self, num_gpu: int) -> List[Tuple[ResourceType, int]]:
        return self._request_idle_resources(ResourceType.GPU, num_gpu)

    def _request_idle_resources(self, resource_type: ResourceType, num: int) -> List[Tuple[ResourceType, int]]:
        idle_resources = self.idle_resources[resource_type]
        if len(idle_resources) < num:
            return []
        else:
            resources = list(idle_resources)[:num]
            for r in resources:
                idle_resources.remove(r)
                self.occupied_resources[resource_type].add(r)
            return resources

    def release_resources(self, resources: List[Tuple[ResourceType, int]]):
        for r in resources:
            for item in ResourceType:
                if r in self.occupied_resources[item]:
                    self.occupied_resources[item].remove(r)
                    self.idle_resources[item].add(r)

class Scheduler:
    def __init__(self, uid: str = "scheduler", max_number_of_task = 32) -> None:
        self.tasks: Dict[str, Task] = {}
        self.uid = uid
        self.grpc_port = -1
        self.period_check_duration = 1.0
        max_number_of_task = min(psutil.cpu_count(False), max_number_of_task)
        self.resource_manager = ResourceManager(max_number_of_task, len(get_nvidia_gpu_measures()))


    @marker.mark_async_init
    async def init_scheduler(self):
        self.lock = asyncio.Lock()
        self.grpc_port = prim.get_server_grpc_port()
        self._period_task = asyncio.create_task(self._period_check_task_status())

    def init_task(self, task_id: str, pid: int):
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.state.status = TaskStatus.Running
            task.state.pid = pid 
            self._update_task_timestamp(task)
            return task.command, task.params
        # task may be deleted before init and after tmux process launch.
        return None 

    def _release_task_resources(self, task: Task):
        self.resource_manager.release_resources(task.state.resources)
        task.state.resources = []


    async def _period_check_task_status(self):
        await asyncio.sleep(self.period_check_duration)
        # task_changed = False
        for task in self.tasks.values():
            if task.state.status == TaskStatus.Running:
                pid_exists = psutil.pid_exists(task.state.pid)
                # print(task.id, task.state.pid, pid_exists, "Running")
                if not pid_exists:
                    # the process is dead.
                    task.state.status = TaskStatus.Failed
                    # print("RELEASE TASK", task.id)
                    self._release_task_resources(task)
                    self._update_task_timestamp(task)
                    # task_changed = True
            elif task.state.status == TaskStatus.AlmostFinished or task.state.status == TaskStatus.AlmostCanceled:
                pid_exists = psutil.pid_exists(task.state.pid)
                # print(task.id, task.state.pid, pid_exists)

                is_almost_finish = task.state.status == TaskStatus.AlmostFinished
                if not pid_exists:
                    # ensure the process is end instead of hang.
                    task.state.status = TaskStatus.Finished if is_almost_finish else TaskStatus.Canceled
                    self._update_task_timestamp(task)
                    # print("RELEASE TASK", task.id)

                    self._release_task_resources(task)
                    # task_changed = True 
        # if task_changed:
        # print("PERIOD SCHEDULE")

        self._do_schedule()
        # print("num idle", len(self.resource_manager.idle_resources[ResourceType.GPU]), "num occ", len(self.resource_manager.occupied_resources[ResourceType.GPU]))

        self._period_task = asyncio.create_task(self._period_check_task_status())

    def get_all_task_state(self):
        return list(self.tasks.values()) 
    
    def get_resource_usage(self):
        return self.resource_manager.idle_resources, self.resource_manager.occupied_resources

    def query_task_updates(self, ts_uids: List[Tuple[int, str]], tmux_pane_lines: int = 0):
        """compare query timestamp, return updated + new and deleted tasks
        """
        deleted_uids: List[str] = []
        update_tasks: List[Task] = []
        all_query_uids = set(x[1] for x in ts_uids)
        for ts, uid in ts_uids:
            if uid in self.tasks:
                task = self.tasks[uid]
                if task.state.timestamp > ts:
                    update_tasks.append(task)
            else:
                deleted_uids.append(uid)
        for k in self.tasks.keys():
            if k not in all_query_uids:
                update_tasks.append(self.tasks[k])
        # for each update task, get their tmux pane last lines
        if tmux_pane_lines > 0:
            for task in update_tasks:
                res = tmux.capture_pane_last_lines(task.id, tmux_pane_lines)
                if isinstance(res, list):
                    res = "\n".join(res)
                task.state.tmux_pane_last_lines = res
        return update_tasks, deleted_uids

    def query_task_tmux_lines(self, task_uids: List[str], tmux_pane_lines: int = 0):
        returns: Dict[str, str] = {}
        for task_id in task_uids:
            res = tmux.capture_pane_last_lines(task_id, tmux_pane_lines)
            if isinstance(res, list):
                res = "\n".join(res)
            returns[task_id] = res 
        return returns

    def submit_task(self, task: Task):
        # print("submit_task START", task.id)

        if task.id == "":
            task.id = str(uuid.uuid4())
        if task.type == TaskType.FunctionId and task.params is None:
            # set init params with empty dict
            task.params = [{}]
        if task.id in self.tasks:
            prev_task = self.tasks[task.id]
            # print("submit_task PREV", prev_task.id, prev_task.state.status)

            if prev_task.state.status in ALL_RUNNING_STATUS:
                raise RuntimeError(f"task {task.id} is already running or pending")
            else:
                # replace old task with new one
                self.tasks[task.id] = task
        else:
            self.tasks[task.id] = task
        task.empty_state()
        task.state.timestamp = time.time_ns()
        task.create_timestamp = task.state.timestamp
        task.state.status = TaskStatus.Pending
        self._do_schedule()
        # print("submit_task END", task.id)

    def check_task_status(self, task_id: str):
        if task_id in self.tasks:
            task = self.tasks[task_id]
            return task.state.status
        else:
            return TaskStatus.Unknown
        
    def set_task_status(self, task_id: str, status: TaskStatus):
        assert status in _SUPPORTED_SET_STATUS, f"only support set to {list(_SUPPORTED_SET_STATUS)}"
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if status == TaskStatus.NeedToCancel:
                if task.state.status not in ALL_RUNNING_STATUS:
                    return False 
            task.state.status = status 
            self._update_task_timestamp(task)
            return True
        return False 

    def run_task(self, task_id: str):
        task = self.tasks[task_id]
        if task.state.status not in ALL_RUNNING_STATUS:
            # print("RUNTASK", task_id)
            cmd = f"python -m tensorpc.autossh.scheduler.runtask {task.type.value}"
            task.state.status = TaskStatus.Booting
            tmux.launch_tmux_task(task_id, cmd, not task.keep_tmux_session, self.grpc_port, task.state.resources, task.cwd)
            self._update_task_timestamp(task)
            return True 
        return False 
    
    def cancel_task(self, task_id: str):
        task = self.tasks[task_id]
        if task.state.status in ALL_CTRL_C_CANCELABLE_STATUS:
            tmux.cancel_task(task_id)
            return True 
        elif task.state.status == TaskStatus.Pending:
            task.state.status = TaskStatus.Canceled
            self._update_task_timestamp(task)
            return True
        return False 
    
    def kill_task(self, task_id: str):
        task = self.tasks[task_id]
        if task.state.status in ALL_KILLABLE_STATUS:
            tmux.kill_task(task_id, task.state.pid)
            return True 
        elif task.state.status == TaskStatus.Pending:
            task.state.status = TaskStatus.Canceled
            self._update_task_timestamp(task)
            return True
        return False 
    
    def delete_task(self, task_id: str):
        if task_id in self.tasks:
            task = self.tasks[task_id]
            assert task.state.status not in ALL_RUNNING_STATUS, "you can't delete a running task"
            self.tasks.pop(task_id)
            tmux.delete_task(task_id)
            return True 
        return False 
    
    def _update_task_timestamp(self, task: Task):
        task.state.timestamp = time.time_ns()

    def _do_schedule(self):
        pending_tasks: List[Task] = []
        for task in self.tasks.values():
            if task.state.status == TaskStatus.Pending:
                pending_tasks.append(task)
        if not pending_tasks:
            return 
        pending_tasks.sort(key=lambda x: x.create_timestamp)
        for task in pending_tasks:
            num_cpu_used = 1 
            num_gpu_used = task.num_gpu_used
            resources = self.resource_manager.request_idle_cpus(num_cpu_used)
            if len(resources) == 0:
                break 
            if num_gpu_used > 0:
                # print("BEFORE REQUEST GPU", self.resource_manager)

                gpu_resources = self.resource_manager.request_idle_gpus(num_gpu_used)
                # print(task.id, num_gpu_used, gpu_resources, self.resource_manager)
                if len(gpu_resources) == 0:
                    self.resource_manager.release_resources(resources)
                    continue 
                resources.extend(gpu_resources)
            task.state.resources = resources
            self.run_task(task.id)

    def set_task_exception(self, task_id: str, exception_str: str):
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.state.status = TaskStatus.Failed
            task.state.exception_str = exception_str
            self._update_task_timestamp(task)
            self._release_task_resources(task)
            self._do_schedule()

    def update_task(self, task_id: str, progress: float, output: Optional[TaskOutput] = None):
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.state.progress = max(min(progress, 1.0), 0.0)
            if output is not None:
                task.state.outputs.append(output)
            self._update_task_timestamp(task)

    def set_task_finished(self, task_id: str):
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.state.status == TaskStatus.NeedToCancel:
                task.state.status = TaskStatus.AlmostCanceled
            else:
                task.state.status = TaskStatus.AlmostFinished
                task.state.progress = 1.0
            self._update_task_timestamp(task)

