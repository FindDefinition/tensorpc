import fire
from tensorpc.apps.cm.constants import TENSORPC_CM_NODEMGR_DEFAULT_PORT
from tensorpc.serve.__main__ import serve_in_terminal
from tensorpc.core import BuiltinServiceKeys


def main(
    *,
    uid: str,
    password: str,
    workdir: str = "",
    port: int = TENSORPC_CM_NODEMGR_DEFAULT_PORT,
    username: str = "root",
    logdir: str = "",
    env_fwd_re: str = "",
    local_ssh_port: int = 22,
    log_to_stdout: bool = False,
    leader_check_interval: float = 5,
    worker_check_interval: float = 5,
    worker_disconnect_timeout: int = 30,

):
    if isinstance(password, int):
        password = str(password)
    service_config = {
        BuiltinServiceKeys.ClusterNodeManager.value: {
            "uid": uid,
            "config_dict": {
                "password": password,
                "username": username,
                "workdir": workdir,
                "logdir": logdir,
                "env_fwd_re": env_fwd_re,
                "local_ssh_port": local_ssh_port,
                "log_to_stdout": log_to_stdout,
                "leader_check_interval": leader_check_interval,
                "worker_check_interval": worker_check_interval,
                "worker_disconnect_timeout": worker_disconnect_timeout,
            }
        }
    }
    serve_in_terminal(BuiltinServiceKeys.ClusterNodeManager.value,
                      port=port,
                      serv_config_json=service_config,
                      max_port_retry=1)


if __name__ == "__main__":
    # python -m tensorpc.apps.cm.cli --workdir=./build --password= 
    fire.Fire(main)
