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

def func_with_exc():
    a = 5
    b = 3
    complex_obj = mui.Button("Hello")
    arr = np.random.uniform(-1, 1, size=[1000, 3])
    raise ValueError("WTF")
    print("Finish!")

def main():
    # tensorpc.dbg.breakpoint()
    wtf = 3
    with tensorpc.dbg.exception_breakpoint():
        func_with_exc()


if __name__ == "__main__":
    main()