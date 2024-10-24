import concurrent
import setproctitle
setproctitle.getproctitle()
import multiprocessing
import os
import tensorpc 
from tensorpc.flow import mui
import time 
import numpy as np
import concurrent.futures
import torch 
def mp_func(rank):
    a = 5
    b = 3
    complex_obj = mui.Button("Hello")
    arr = np.random.uniform(-1, 1, size=[1000, 3])
    tensorpc.dbg.breakpoint(name=f"WTF-{rank}")
    time.sleep(2)
    tensorpc.dbg.breakpoint(name=f"WTF-{rank}")
    print("Finish!")

def mp_func_for_fork_debug(rank):
    print("FORK START", os.getpid())
    a = 5
    b = 3
    c = torch.rand(100, 3)
    complex_obj = mui.Button("Hello")
    arr = np.random.uniform(-1, 1, size=[1000, 3])
    tensorpc.dbg.vscode_breakpoint(name=f"WTF-{rank}")
    print("Finish!")

def main(c = 5):
    a = 5
    b = 3
    complex_obj = mui.Button("Hello")
    arr = np.random.uniform(-1, 1, size=[1000, 3])
    tensorpc.dbg.breakpoint(name="WTF", init_port=54321)
    time.sleep(2)
    tensorpc.dbg.vscode_breakpoint(name="WTF", init_port=54321)

    print("Finish!")

def main_mp():
    ctx = multiprocessing.get_context("spawn")
    num_proc = 16
    procs = []
    for j in range(num_proc):
        p = ctx.Process(target=mp_func, args=(j,))
        p.daemon = True
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

def main_mp_fork_debug():
    ctx = multiprocessing.get_context("fork")
    num_proc = 2
    procs = []
    tensorpc.dbg.breakpoint(name="WTF", init_port=54322)
    for j in range(num_proc):
        p = ctx.Process(target=mp_func_for_fork_debug, args=(j,))
        p.daemon = True
        p.start()
        procs.append(p)
    # time.sleep(1)
    # tensorpc.dbg.breakpoint(name="WTF2", init_port=54322)

    for p in procs:
        p.join()


if __name__ == "__main__":
    main_mp_fork_debug()