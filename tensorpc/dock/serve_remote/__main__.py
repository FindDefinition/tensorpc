from typing import Dict, Optional, Union
from typing_extensions import Literal
import fire
from tensorpc.constants import TENSORPC_PORT_MAX_TRY
from tensorpc.core.funcid import split_func_id
from tensorpc.serve.__main__ import serve_in_terminal
from tensorpc.utils.uniquename import UniqueNamePool

# TODO service prompt
def serve_remote_comp(*app_ids: str,
                      wait_time=-1,
                      port: Union[str, int, tuple]=50051,
                      http_port=None,
                      length=-1,
                      serv_def_file: Optional[str] = None,
                      max_threads=10,
                      ssl_key_path: str = "",
                      ssl_crt_path: str = "",
                      default_app_keys: str = "",
                      max_port_retry: int = TENSORPC_PORT_MAX_TRY,
                      distributed_mode: Literal['torch_gloo', 'torch_nccl', 'torch_gloo_nccl', 'none'] = "none"):
    if default_app_keys != "":
        default_app_key_list = default_app_keys.split(",")
        assert len(default_app_key_list) == len(app_ids)
    else:
        uq = UniqueNamePool()
        default_app_key_list: list[str] = []
        for app_id in app_ids:
            app_name = app_id.split("::")[-1]
            default_app_key_list.append(uq(app_name))
    serv_config_json = {
        "tensorpc.dock.serv.remote_comp::RemoteComponentService": {
            "init_apps": {
                default_app_key_list[i]: app_id
                for i, app_id in enumerate(app_ids)
            }
        }
    }
    return serve_in_terminal(
        wait_time=wait_time,
        port=port,
        http_port=http_port,
        length=length,
        serv_def_file=serv_def_file,
        max_threads=max_threads,
        ssl_key_path=ssl_key_path,
        ssl_crt_path=ssl_crt_path,
        serv_config_json=serv_config_json,
        max_port_retry=max_port_retry,
        distributed_mode=distributed_mode,
    )


if __name__ == "__main__":
    fire.Fire(serve_remote_comp)
