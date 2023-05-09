import time
from typing import List, Tuple
import libtmux
from tensorpc.autossh.scheduler import constants
from tensorpc.autossh.scheduler.core import ResourceType
from tensorpc.utils.wait_tools import get_free_ports
from tensorpc.constants import TENSORPC_SPLIT
import base64
import json
import uuid
import psutil

_SPLIT = constants.TMUX_SESSION_NAME_SPLIT

def get_tmux_scheduler_info_may_create():
    s = libtmux.Server()
    sessions = s.sessions
    sess_names = [sess.name for sess in sessions]
    scheduler_sess_names = [
        sess_name for sess_name in sess_names if sess_name.startswith(constants.TMUX_SESSION_PREFIX)]
    if len(scheduler_sess_names) == 0:
        uuid_str = uuid.uuid4().hex
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
        scheduler_sess_name = f"{constants.TMUX_SESSION_PREFIX}{_SPLIT}{port}{_SPLIT}{uuid_str}"
        sess = s.new_session(scheduler_sess_name,
                             window_command=window_command)
    else:
        assert len(scheduler_sess_names) == 1
        scheduler_sess_name = scheduler_sess_names[0]
        sess_parts = scheduler_sess_name.split(_SPLIT)
        port = int(sess_parts[1])
        uuid_str = sess_parts[2]
        sess = s.sessions.get(session_name=scheduler_sess_name)
        assert isinstance(sess, libtmux.Session)
    return port, uuid_str


def launch_tmux_task(uuid_str: str, window_command: str, one_shot: bool, sched_port: int, resources: List[Tuple[ResourceType, int]]):
    s = libtmux.Server()
    sess_name = f"{constants.TMUX_SESSION_TASK_PREFIX}{_SPLIT}{uuid_str}"
    envs = {
        constants.TENSORPC_TMUX_TASK_SCHEDULER_PORT: sched_port,
        constants.TENSORPC_TMUX_TASK_UID: uuid_str,
    }
    all_gpu_ids: List[int] = []
    for res_type, res_index in resources:
        if res_type == ResourceType.GPU:
            all_gpu_ids.append(res_index)
    if len(all_gpu_ids) > 0:
        envs["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(gpu_id) for gpu_id in all_gpu_ids])
    env_export_str = " && ".join([f"export {k}={v}" for k, v in envs.items()])
    if s.has_session(sess_name):
        assert not one_shot, f"one shot task already exists, {sess_name}"
        sess = s.sessions.get(session_name=sess_name)
        assert isinstance(sess, libtmux.Session)
        pane: libtmux.Pane = sess.windows[0].panes[0]
        pane.send_keys(env_export_str)
        pane.send_keys(window_command)
    else:
        if one_shot:
            sess = s.new_session(
                f"{constants.TMUX_SESSION_TASK_PREFIX}{_SPLIT}{uuid_str}", window_command=f"{env_export_str} && {window_command}")
        else:
            sess = s.new_session(
                f"{constants.TMUX_SESSION_TASK_PREFIX}{_SPLIT}{uuid_str}")
            pane: libtmux.Pane = sess.windows[0].panes[0]
            pane.send_keys(env_export_str)
            pane.send_keys(window_command)

def kill_task(uuid_str: str, pid: int):
    s = libtmux.Server()
    sess_name = f"{constants.TMUX_SESSION_TASK_PREFIX}{_SPLIT}{uuid_str}"
    if pid > 0:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True): 
            child.kill()
        parent.kill()
    else:
        assert s.has_session(sess_name)
        sess = s.sessions.get(session_name=sess_name)
        assert isinstance(sess, libtmux.Session)
        sess.kill_session()

def cancel_task(uuid_str: str):
    s = libtmux.Server()
    sess_name = f"{constants.TMUX_SESSION_TASK_PREFIX}{_SPLIT}{uuid_str}"
    assert s.has_session(sess_name)
    # if s.has_session(sess_name):
    sess = s.sessions.get(session_name=sess_name)
    pane: libtmux.Pane = sess.windows[0].panes[0]
    pane.send_keys("\x03")
