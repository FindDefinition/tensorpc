from pathlib import Path
from typing import Optional

import fire
import time
from tensorpc.core.client import RemoteManager
from tensorpc.core.server import serve, serve_with_http
from tensorpc.core.defs import Service, ServiceDef, from_yaml_path
from tensorpc.core import BUILTIN_SERVICES


# TODO service prompt
def serve_in_terminal(*modules: str,
                      wait_time=-1,
                      port=50051,
                      http_port=None,
                      length=-1,
                      serv_def_file: Optional[str] = None,
                      max_threads=10,
                      ssl_key_path: str = "",
                      ssl_crt_path: str = ""):
    if serv_def_file is not None:
        service_def = from_yaml_path(serv_def_file)
    else:
        servs = [Service(m, {}) for m in modules]
        service_def = ServiceDef(servs)
    service_def.services.extend(BUILTIN_SERVICES)
    if http_port is not None:
        return serve_with_http(wait_time=wait_time,
                               http_port=http_port,
                               port=port,
                               length=length,
                               max_threads=max_threads,
                               service_def=service_def)
    return serve(wait_time=wait_time,
                 port=port,
                 length=length,
                 max_threads=max_threads,
                 service_def=service_def)


def serve_in_terminal_main():
    fire.Fire(serve_in_terminal)


def ping(ip: str):
    with RemoteManager(ip) as robj:
        t = time.time()
        robj.health_check()
        print("[distflow.ping]{} response time: {:.4f}ms".format(
            ip, 1000 * (time.time() - t)))


def ping_main():
    fire.Fire(ping)


if __name__ == "__main__":
    fire.Fire(serve_in_terminal)
