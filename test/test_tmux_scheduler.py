import time
from typing import Dict, List
import asyncio
from tensorpc.autossh.scheduler.core import SSHTarget, Task, TaskStatus, TaskType
from tensorpc.autossh.scheduler.client import SchedulerClient
from tensorpc.constants import PACKAGE_ROOT


async def test_basic():
    local_ssh_target = SSHTarget("localhost", 22, "root", "root")
    client = SchedulerClient(local_ssh_target)
    await client.async_init()
    task = Task(TaskType.FunctionId, "tensorpc.autossh.scheduler.test_data::simple_task", [{}])
    task.id = "test"
    # task.keep_tmux_session = False
    await client.submit_task(task)
    for i in range(10):
        await asyncio.sleep(1)
        if i == 2:
            await client.cancel_task(task.id)
        await client.update_tasks()
        print("------")
        for task in client.tasks.values():
            print(task.id, task.state.status, task.state.progress)
    await client.shutdown_scheduler()

async def test_basic_with_client():
    local_ssh_target = SSHTarget("localhost", 22, "root", "root")
    client = SchedulerClient(local_ssh_target)
    await client.async_init()
    task = Task(TaskType.FunctionId, "tensorpc.autossh.scheduler.test_data::simple_task_with_client", [{}])
    task.id = "test"
    # task.keep_tmux_session = False
    await client.submit_task(task)
    for i in range(10):
        await asyncio.sleep(1)
        await client.update_tasks()
        print("------")
        for task in client.tasks.values():
            print(task.id, task.state.status, task.state.progress)
    await client.shutdown_scheduler()

async def test_basic_with_client_with_cancel():
    local_ssh_target = SSHTarget("localhost", 22, "root", "root")
    client = SchedulerClient(local_ssh_target)
    await client.async_init()
    task = Task(TaskType.FunctionId, "tensorpc.autossh.scheduler.test_data::simple_task_with_client", [{}])
    task.id = "test"
    task.keep_tmux_session = False
    await client.submit_task(task)
    for i in range(10):
        await asyncio.sleep(1)
        if i == 2:
            await client.set_task_status(task.id, TaskStatus.NeedToCancel)
        await client.update_tasks()
        print("------")
        for task in client.tasks.values():
            print(task.id, task.state.status, task.state.progress)
    await client.shutdown_scheduler()

async def test_parallel_task():
    local_ssh_target = SSHTarget("localhost", 22, "root", "root")
    client = SchedulerClient(local_ssh_target)
    await client.async_init()
    task1 = Task(TaskType.FunctionId, "tensorpc.autossh.scheduler.test_data::simple_task_with_client", [{}])
    task1.id = "test1"
    task1.keep_tmux_session = False

    task2 = Task(TaskType.FunctionId, "tensorpc.autossh.scheduler.test_data::simple_task_with_client", [{}])
    task2.id = "test2"
    task2.keep_tmux_session = False
    await client.submit_task(task1)
    await client.submit_task(task2)

    for i in range(10):
        await asyncio.sleep(1)
        await client.update_tasks()
        print("------")
        for task in client.tasks.values():
            print(task.id, task.state.status, task.state.progress, task.state.pid)
    await client.shutdown_scheduler()

async def test_gpu_resources():
    local_ssh_target = SSHTarget("localhost", 22, "root", "root")
    client = SchedulerClient(local_ssh_target)
    await client.async_init()
    task1 = Task(TaskType.FunctionId, "tensorpc.autossh.scheduler.test_data::simple_task_with_client", [{}])
    task1.id = "test1"
    task1.num_gpu_used = 1
    task1.keep_tmux_session = False

    task2 = Task(TaskType.FunctionId, "tensorpc.autossh.scheduler.test_data::simple_task_with_client", [{}])
    task2.id = "test2"
    task2.num_gpu_used = 1
    task2.keep_tmux_session = False

    # task.keep_tmux_session = False
    await client.submit_task(task1)
    await client.submit_task(task2)

    for i in range(15):
        await asyncio.sleep(1)
        await client.update_tasks()
        print("------")
        for task in client.tasks.values():
            print(task.id, task.state.status, task.state.progress, task.state.pid)
    await client.shutdown_scheduler()

async def test_shell_with_client():
    local_ssh_target = SSHTarget("localhost", 22, "root", "root")
    client = SchedulerClient(local_ssh_target)
    await client.async_init()
    path = PACKAGE_ROOT / "autossh" / "scheduler" / "test_data.py"
    task = Task(TaskType.Command, f"python {path}", [{}])
    task.id = "test"
    # task.keep_tmux_session = False
    await client.submit_task(task)
    for i in range(10):
        await asyncio.sleep(1)
        await client.update_tasks()
        print("------")
        for task in client.tasks.values():
            print(task.id, task.state.status, task.state.progress)
    await client.shutdown_scheduler()

def main():
    asyncio.run(test_parallel_task()) 

if __name__ == "__main__":
    main()

