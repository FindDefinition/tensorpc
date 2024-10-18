import fire
import time
from tensorpc.core.client import RemoteManager
import rich

def ping(ip: str):
    with RemoteManager(ip) as robj:
        t = time.time()
        res = robj.query_server_meta()
        print("[tensorpc.ping]{} response time: {:.4f}ms".format(
            ip, 1000 * (time.time() - t)))
        rich.print(res)


def ping_main():
    fire.Fire(ping)


if __name__ == "__main__":
    ping_main()
