import concurrent
import multiprocessing
import tensorpc 
from tensorpc.flow import mui
import time 
import numpy as np
import concurrent.futures
def mp_func(rank):
    a = 5
    b = 3
    complex_obj = mui.Button("Hello")
    arr = np.random.uniform(-1, 1, size=[1000, 3])
    tensorpc.dbg.breakpoint(name=f"WTF-{rank}")
    time.sleep(2)
    tensorpc.dbg.breakpoint(name=f"WTF-{rank}")
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

if __name__ == "__main__":
    main()