import dataclasses
import time
from typing import Dict, List, Set, Tuple 
import libtmux
import asyncio
from tensorpc.autossh.serv_names import serv_names 
from tensorpc.autossh.scheduler.constants import TMUX_SESSION_PREFIX, TMUX_SESSION_NAME_SPLIT
from tensorpc.autossh.scheduler.core import ResourceType, SSHTarget, Task, TaskStatus, TaskType
from tensorpc.core.asyncclient import shutdown_server_async, simple_remote_call_async
from tensorpc.utils.wait_tools import get_free_ports
from tensorpc.autossh.core import SSHClient
from tensorpc.constants import TENSORPC_SPLIT
import base64, json
import uuid
from tensorpc import prim
from tensorpc.autossh.scheduler.tmux import get_tmux_scheduler_info_may_create
from tensorpc import AsyncRemoteManager, simple_chunk_call_async

class SchedulerClient:
    def __init__(self, ssh_target: SSHTarget) -> None:
        assert isinstance(ssh_target, SSHTarget)
        self.ssh_target = dataclasses.replace(ssh_target)
        self.tunnel_tasks: List[asyncio.Task] = []
        self.port = -1

    async def async_init(self):
        self.shutdown_ev = asyncio.Event()

        if prim.is_in_server_context():
            self.shutdown_ev = prim.get_async_shutdown_event()
            self.shutdown_task = asyncio.create_task(prim.get_async_shutdown_event().wait())
        else:
            self.shutdown_task = asyncio.create_task(self.shutdown_ev.wait())

        # fetch scheduler session in all ssh targets
        target = self.ssh_target
        hostname = target.hostname.strip()
        is_local = hostname == "localhost" or hostname == "127.0.0.1"
        client = SSHClient.from_ssh_target(target)
        # fetch scheduler port
        if is_local:
            port, schr_uid = get_tmux_scheduler_info_may_create()
            target.forward_port_pairs.append((port, port))
            self.port = port
        else:
            # print(target)
            async with client.simple_connect(False) as conn:
                # try:
                if target.init_commands:
                    result = await conn.run(f"bash -i -c \"{target.init_commands} && python -m tensorpc.autossh.scheduler.init_scheduler\"", check=True)
                else:
                    result = await conn.run("bash -i -c \"python -m tensorpc.autossh.scheduler.init_scheduler\"", check=True)
                # except Exception as e:
                #     print(e)
                #     raise e 
                # print(result.stdout, result.stderr)
                stdout = result.stdout
                assert stdout is not None 
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8")
                port_str, schr_uid = stdout.strip().split(",")
                port = int(port_str)
            local_free_port = get_free_ports(1)[0]
            self.tunnel_tasks.append(asyncio.create_task(client.create_local_tunnel([(local_free_port, port)], self.shutdown_task)))
            target.forward_port_pairs.append((local_free_port, port))
            self.port = local_free_port
        self.local_url = f"localhost:{self.port}"
        self.schr_uid = schr_uid
        self.schr_session_name = f"{TMUX_SESSION_PREFIX}{TMUX_SESSION_NAME_SPLIT}{port}{TMUX_SESSION_NAME_SPLIT}{schr_uid}"

        # fetch init task state 
        async with AsyncRemoteManager(self.local_url) as robj:
            await robj.wait_for_channel_ready()
            all_tasks: List[Task] = await robj.chunked_remote_call(serv_names.SCHED_TASK_GET_ALL_TASK)
        self.tasks = {t.id: t for t in all_tasks}

    async def update_tasks(self, tmux_pane_lines: int = 0):
        ts_uids = [(t.state.timestamp, t.id) for t in self.tasks.values()]
        updated, deleted_uids = await simple_chunk_call_async(self.local_url, serv_names.SCHED_TASK_QUERY_UPDATES, ts_uids, tmux_pane_lines)
        for t in updated:
            self.tasks[t.id] = t
        for uid in deleted_uids:
            self.tasks.pop(uid, None)
        return updated, deleted_uids
    
    async def query_tmux_panes(self, task_ids: List[str], tmux_pane_lines: int):
        return await simple_chunk_call_async(self.local_url, serv_names.SCHED_TASK_QUERY_TMUX_PANES, task_ids, tmux_pane_lines)

    async def get_resource_usage(self):
        idle_resources, occupied_resources = await simple_chunk_call_async(self.local_url, serv_names.SCHED_TASK_RESOURCE_USAGE)
        idle_resources: Dict[ResourceType, Set[Tuple[ResourceType, int]]]
        occupied_resources: Dict[ResourceType, Set[Tuple[ResourceType, int]]]
        return idle_resources, occupied_resources

    async def submit_task(self, task: Task):
        return await simple_remote_call_async(self.local_url, serv_names.SCHED_TASK_SUBMIT_TASK, task)
    
    async def cancel_task(self, task_id: str):
        """use Ctrl-C to cancel task
        """
        return await simple_remote_call_async(self.local_url, serv_names.SCHED_TASK_CANCEL_TASK, task_id)
    
    async def delete_task(self, task_id: str):
        """delete task from task list
        """
        return await simple_remote_call_async(self.local_url, serv_names.SCHED_TASK_DELETE, task_id)

    async def kill_task(self, task_id: str):
        return await simple_remote_call_async(self.local_url, serv_names.SCHED_TASK_KILL_TASK, task_id)

    async def shutdown_scheduler(self):
        return await shutdown_server_async(self.local_url)

    async def set_task_status(self, task_id: str, status: TaskStatus):
        """set task to some specific status (i.e. tell task to do something)
        """
        return await simple_remote_call_async(self.local_url, serv_names.SCHED_TASK_SET_STATUS, task_id, status)

    async def soft_cancel_task(self, task_id: str):
        """use status to cancel task, task need to check status by
        TaskClient and cancel itself
        """
        return await self.set_task_status(task_id, TaskStatus.NeedToCancel)

def main():
    s = libtmux.Server()
    sessions = s.sessions
    sess_names = [sess.name for sess in sessions]
    print(sess_names, s.socket_path)
    scheduler_sess_names = [sess_name for sess_name in sess_names if sess_name.startswith(TMUX_SESSION_PREFIX)]
    if len(scheduler_sess_names) == 0:
        uuid_str = uuid.uuid4().hex
        uuid_str = "0"
        serv_name = f"tensorpc.autossh.services.scheduler{TENSORPC_SPLIT}Scheduler"
        cfg = {
            serv_name: {
                "uid": uuid_str,
            }
        }
        cfg_encoded = base64.b64encode(
            json.dumps(cfg).encode("utf-8")).decode("utf-8")
        port = get_free_ports(1)[0]
        window_command = f"python -m tensorpc.serve --port {port} --serv_config_b64 {cfg_encoded}"
        scheduler_sess_name = f"{TMUX_SESSION_PREFIX}{TMUX_SESSION_NAME_SPLIT}{port}{TMUX_SESSION_NAME_SPLIT}{uuid_str}"
        print(window_command)
        sess = s.new_session(scheduler_sess_name, window_command=window_command)
        window = sess.windows[0]
        print("?", s, window)

    else:
        assert len(scheduler_sess_names) == 1
        scheduler_sess_name = scheduler_sess_names[0]
        sess_parts = scheduler_sess_name.split(TMUX_SESSION_NAME_SPLIT)
        port = int(sess_parts[1])
        uuid_str = sess_parts[2]
        sess = s.sessions.get(session_name=scheduler_sess_name)
        assert isinstance(sess, libtmux.Session)
        
        print(scheduler_sess_name, sess, sess.windows[0].panes[0])

    pass 

