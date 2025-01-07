import shutil
from typing import Optional
import fire
import setproctitle
from tensorpc.constants import TENSORPC_SSH_TASK_DEFAULT_PORT
from tensorpc.serve.__main__ import serve_in_terminal
from tensorpc.core import BuiltinServiceKeys


def main(
    *cmd: str,
    password: str,
    max_retries: int = 1,
    username: str = "root",
    log_path: Optional[str] = None,
    msg_throttle: int = 30,
    master_url: Optional[str] = None,
    num_workers: int = 1,
    init_timeout: int = 60,
    msg_handler: Optional[str] = None,
    init_info_getter: Optional[str] = None,
    error_handler: Optional[str] = None,
    pyspy_period: Optional[int] = None,
    wait_time: int = -1,
    port: int = TENSORPC_SSH_TASK_DEFAULT_PORT,
    http_port=None,
    length=-1,
    max_threads=10,
    ssl_key_path: str = "",
    ssl_crt_path: str = "",
):
    if pyspy_period is not None:
        assert log_path is not None, "log_path must be set if pyspy_period is set."
        # check pyspy is installed
        if not shutil.which("py-spy"):
            raise ValueError("py-spy is not installed, please install it first.")
    service_config = {
        BuiltinServiceKeys.TaskWrapper.value: {
            "config": {
                "cmd": " ".join(cmd),
                "max_retries": max_retries,
                "password": password,
                "username": username,
                "log_path": log_path,
                "msg_throttle": msg_throttle,
                "master_url": master_url,
                "num_workers": num_workers,
                "init_timeout": init_timeout,
                "msg_handler": msg_handler,
                "init_info_getter": init_info_getter,
                "error_handler": error_handler,
                "pyspy_period": pyspy_period,
                
            }
        }
    }
    if master_url is not None:
        assert ":" in master_url
        # if master_url exists (MPI-like distributed task), use this port instead of port in argument.
        parts = master_url.split(":")
        assert len(parts) == 2
        port = int(parts[1])
    # max_port_retry: we won't retry random port here.
    serve_in_terminal(wait_time=wait_time,
                      port=port,
                      http_port=http_port,
                      length=length,
                      max_threads=max_threads,
                      ssl_key_path=ssl_key_path,
                      ssl_crt_path=ssl_crt_path,
                      serv_config_json=service_config,
                      max_port_retry=1)


if __name__ == "__main__":
    fire.Fire(main)
