
import subprocess
import time
from typing import List, Tuple
import fire
import libtmux
from tensorpc.utils.wait_tools import get_free_ports
from tensorpc.constants import TENSORPC_SPLIT
from tensorpc.flow.constants import TENSORPC_FLOW_LANG_SERVER_NAME_SPLIT, TENSORPC_FLOW_LANG_SERVER_PREFIX

_SPLIT = TENSORPC_FLOW_LANG_SERVER_NAME_SPLIT

def get_tmux_lang_server_info_may_create(ls_type: str):
    # TODO pyright support
    assert ls_type in ["jedi"]
    if ls_type == "pyright":
        window_command_fmt = ""
        try:
            subprocess.check_call(["pyright", "--version"])
        except Exception:
            raise Exception("pyright not installed, you can install by pip install pyright")
    else:
        window_command_fmt = "jedi-language-server --ws --port {}"

        try:
            subprocess.check_call(["jedi-language-server", "--version"])
        except Exception:
            raise Exception("jedi-language-server not installed, you can install by pip install jedi-language-server pygls[ws]")
    
    
    s = libtmux.Server()
    sessions = s.sessions
    sess_names = [sess.name for sess in sessions]
    scheduler_sess_names = [
        sess_name for sess_name in sess_names if sess_name.startswith(TENSORPC_FLOW_LANG_SERVER_PREFIX)]
    if len(scheduler_sess_names) == 0:
        port = get_free_ports(1)[0]
        window_command = window_command_fmt.format(port)
        scheduler_sess_name = f"{TENSORPC_FLOW_LANG_SERVER_PREFIX}{_SPLIT}{port}"
        sess = s.new_session(scheduler_sess_name,
                             window_command=window_command)
    else:
        assert len(scheduler_sess_names) == 1
        scheduler_sess_name = scheduler_sess_names[0]
        sess_parts = scheduler_sess_name.split(_SPLIT)
        port = int(sess_parts[1])
        sess = s.sessions.get(session_name=scheduler_sess_name)
        assert isinstance(sess, libtmux.Session)
    return port

def main(ls_type: str):
    port = get_tmux_lang_server_info_may_create(ls_type)
    print(f"{port}")

if __name__ == "__main__":
    fire.Fire(main)