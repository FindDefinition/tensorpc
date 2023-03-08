import fire
import time
from tensorpc.core.client import RemoteManager


def ping(ip: str):
    with RemoteManager(ip) as robj:
        t = time.time()
        robj.health_check()
        print("[distflow.ping]{} response time: {:.4f}ms".format(
            ip, 1000 * (time.time() - t)))


def ping_main():
    fire.Fire(ping)


if __name__ == "__main__":
    ping_main()
