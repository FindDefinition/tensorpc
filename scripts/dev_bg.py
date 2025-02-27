import multiprocessing
import time 
from typing import Tuple, List
import random
def process_func(q, ev):
    from tensorpc.core.bgserver import BACKGROUND_SERVER
    port = BACKGROUND_SERVER.start_async(id="wtf-1")
    q.put(port)
    for i in range(10):
        if ev.is_set():
            break
        # print(1)
        time.sleep(random.uniform(0.1, 0.5))
        # print(2)
        time.sleep(random.uniform(0.1, 0.5))
        # print(3)
        time.sleep(random.uniform(0.1, 0.5))
        # print(4)
        # time.sleep(random.uniform(0.1, 0.5))
        # print(5)
        # time.sleep(random.uniform(0.1, 0.5))
        # print(6)
        # time.sleep(random.uniform(0.1, 0.5))
        # print(7)
        # time.sleep(random.uniform(0.1, 0.5))
    print("EXIT")
def process_func2(q, ev):
    from tensorpc.core.bgserver import BACKGROUND_SERVER
    port = BACKGROUND_SERVER.start_async(id="wtf-2")
    q.put(port)
    for i in range(10):
        if ev.is_set():
            break
        # print(1)
        time.sleep(random.uniform(0.1, 0.5))
        # print(2)
        time.sleep(random.uniform(0.1, 0.5))
        # print(3)
        time.sleep(random.uniform(0.1, 0.5))
        # print(4)
        # time.sleep(random.uniform(0.1, 0.5))
        # print(5)
        # time.sleep(random.uniform(0.1, 0.5))
        # print(6)
        # time.sleep(random.uniform(0.1, 0.5))
        # print(7)
        # time.sleep(random.uniform(0.1, 0.5))
    print("EXIT")

def multiprocess_main():
    ctx = multiprocessing.get_context("fork")
    procs: List[Tuple[multiprocessing.Process, multiprocessing.Event]] = []
    robjs: List["RemoteManager"] = []
    for i in range(2):
        q = ctx.Queue()
        ev = ctx.Event()
        p = ctx.Process(target=process_func if i == 0 else process_func2, args=(q, ev))
        p.start()
        port = q.get()
        print(port)
        procs.append((p, ev, port))
    from tensorpc import RemoteManager
    for i in range(2):
        robj = RemoteManager(f"localhost:{procs[i][2]}")
        robjs.append(robj)

    for i in range(3):
        time.sleep(0.5)
        print("-----")
        for robj in robjs:
            print(robj.url, robj.remote_call("tensorpc.services.collection::ProcessObserver.get_threads_current_status"))
    for i in range(len(procs)):
        procs[i][1].set()
        procs[i][0].join()
        print(f"Process {i} stopped")

    # BACKGROUND_SERVER.stop()
    pass 

def main():
    from tensorpc.core.bgserver import BACKGROUND_SERVER
    BACKGROUND_SERVER.start_async()
    print(BACKGROUND_SERVER.port)
    from tensorpc import RemoteManager
    robj = RemoteManager(f"localhost:{BACKGROUND_SERVER.port}")
    fts = []
    for i in range(1):
        future = robj.remote_call_future("tensorpc.services.collection::Simple.sleep", 2)
        print("GET FUTURE")
        fts.append(future)
    for ft in fts:
        print(ft.result())
    # time.sleep(10)
    # BACKGROUND_SERVER.stop()

if __name__ == "__main__":
    multiprocess_main()