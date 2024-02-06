
import subprocess
import time
from typing import List, Tuple
import fire
import libtmux
from tensorpc.utils.wait_tools import get_free_ports
from tensorpc.constants import TENSORPC_SPLIT
from tensorpc.flow.constants import TENSORPC_FLOW_LANG_SERVER_NAME_SPLIT, TENSORPC_FLOW_LANG_SERVER_PREFIX

_SPLIT = TENSORPC_FLOW_LANG_SERVER_NAME_SPLIT

def get_tmux_lang_server_info_may_create(ls_type: str, uid: str, port: int):
    # TODO jedi support
    assert ls_type in ["pyright", "pylsp", "clangd"]
    if ls_type == "pyright":
        window_command_fmt = "python -m tensorpc.cli.pyls --port={}"
        try:
            import pyright
        except ImportError:
            raise Exception("pyright not installed, you can install by pip install pyright")
    elif ls_type == "pylsp":
        window_command_fmt = "pylsp --ws --port {}"
        try:
            subprocess.check_call(["pylsp", "--version"])
        except Exception:
            raise Exception("pylsp not installed, you can install by pip install python-lsp-server[websockets] yapf")
    else:
        window_command_fmt = "python -m tensorpc.cli.cppls --port={}"
        try:
            subprocess.check_call(["clangd", "--version"])
        except Exception:
            raise Exception("clangd not installed, you can install by sudo apt install clangd")

    s = libtmux.Server()
    sessions = s.sessions
    sess_names = [sess.name for sess in sessions]
    scheduler_sess_names = [
        sess_name for sess_name in sess_names if sess_name.startswith(TENSORPC_FLOW_LANG_SERVER_PREFIX)]
    found = False
    for sess_name in scheduler_sess_names:
        sess_parts = sess_name.split(_SPLIT)
        port_candidate = int(sess_parts[1])
        uid_candidate = sess_parts[2]
        if uid != uid_candidate:
            continue 
        found = True
        # print(port, port_candidate, port_candidate != port)
        if port_candidate != port:
            close_tmux_lang_server(uid)
            window_command = window_command_fmt.format(port)
            scheduler_sess_name = f"{TENSORPC_FLOW_LANG_SERVER_PREFIX}{_SPLIT}{port}{_SPLIT}{uid}"
            sess = s.new_session(scheduler_sess_name, window_command=window_command)
            # pane: libtmux.Pane = sess.windows[0].panes[0]
            # pane.send_keys(window_command)

            return port
        else:
            assert port_candidate == port
            scheduler_sess_name = scheduler_sess_names[0]
            sess = s.sessions.get(session_name=scheduler_sess_name)
            assert isinstance(sess, libtmux.Session)
            break
    # if port == -1:
    #     port = get_free_ports(1)[0]
    if not found:
        window_command = window_command_fmt.format(port)
        print(window_command)

        scheduler_sess_name = f"{TENSORPC_FLOW_LANG_SERVER_PREFIX}{_SPLIT}{port}{_SPLIT}{uid}"
        sess = s.new_session(scheduler_sess_name, window_command=window_command)

        # pane: libtmux.Pane = sess.windows[0].panes[0]
        # pane.send_keys(window_command)

    return port

def close_tmux_lang_server(uid: str):
    _prefix = TENSORPC_FLOW_LANG_SERVER_PREFIX
    # print("CLOSE CLANG", uid)
    # raise NotImplementedError

    # TODO pyright support
    s = libtmux.Server()
    sessions = s.sessions
    sess_names = [sess.name for sess in sessions]
    scheduler_sess_names = [
        sess_name for sess_name in sess_names if sess_name.startswith(_prefix) and sess_name.endswith(uid)]
    if len(scheduler_sess_names) != 0:
        scheduler_sess_name = scheduler_sess_names[0]
        sess_parts = scheduler_sess_name.split(_SPLIT)
        port = int(sess_parts[1])
        sess = s.sessions.get(session_name=scheduler_sess_name)
        assert isinstance(sess, libtmux.Session)
        pane: libtmux.Pane = sess.windows[0].panes[0]
        pane.send_keys("\x03")
