import fire
from tensorpc.constants import TENSORPC_APPS_DISTSSH_DEFAULT_PORT
from tensorpc.serve.__main__ import serve_in_terminal
from tensorpc.core import BuiltinServiceKeys


def main(
    *cmd: str,
    rank: int,
    world_size: int,
    password: str,
    workdir: str,
    port: int = TENSORPC_APPS_DISTSSH_DEFAULT_PORT,
    username: str = "root",
):
    service_config = {
        BuiltinServiceKeys.FaultToleranceSSHServer.value: {
            "config_dict": {
                "cmd": " ".join(cmd),
                "password": password,
                "username": username,
                "workdir": workdir,
                "rank": rank,
                "world_size": world_size,
            }
        }
    }
    serve_in_terminal(BuiltinServiceKeys.FaultToleranceSSHServer.value,
                      port=port,
                      serv_config_json=service_config,
                      max_port_retry=1)


if __name__ == "__main__":
    fire.Fire(main)
