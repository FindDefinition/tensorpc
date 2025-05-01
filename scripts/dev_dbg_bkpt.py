import concurrent
import setproctitle
setproctitle.getproctitle()
import multiprocessing
import os
import tensorpc 
from tensorpc.dock import mui
import time 
import numpy as np
import concurrent.futures
import torch 

import torch

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

import torch.profiler as profiler


class MyModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, input, mask):
        with profiler.record_function("LINEAR PASS"):
            out = self.linear(input)

        with profiler.record_function("MASK INDICES"):
            threshold = out.sum(axis=1).mean().item()
            hi_idx = np.argwhere(mask.cpu().numpy() > threshold)
            hi_idx = torch.from_numpy(hi_idx).to(input.device)

        return out, hi_idx

class MyModule2(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, input):
        # if input.shape[1] >= 64:
        return F.relu(self.linear(input))
        # else:
        #     return F.relu(self.linear(input) + input)

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
    # tensorpc.dbg.vscode_breakpoint(name=f"WTF-{rank}")
    tensorpc.dbg.breakpoint(name="WTF")
    complex_obj = mui.Button("Hello")
    arr = np.random.uniform(-1, 1, size=[1000, 3])

    _trace_func()
    tensorpc.dbg.breakpoint(name="WTF")
    

    print("Finish!")

def mp_func_inf_record(rank):
    print("FORK START", os.getpid())
    tensorpc.dbg.init()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    for j in range(1000):
        a = 5
        b = 3
        # tensorpc.dbg.record_instant_event(f"WTf-{j}")
        c = torch.rand(100, 3)
        complex_obj = mui.Button("Hello")
        arr = np.random.uniform(-1, 1, size=[1000, 3])
        ten = torch.rand(1).to(torch.bfloat16)
        ten2 = torch.rand(1).to(torch.float16)

        # tensorpc.dbg.vscode_breakpoint(name=f"WTF-{rank}")
        # tensorpc.dbg.breakpoint(name="WTF")
        model = MyModule(50, 10).to(device)
        model2 = MyModule2(50, 10).to(device)
        model3 = torch.nn.Transformer()
        input = torch.rand(128, 50).to(device)
        mask = torch.rand((50, 50, 50), dtype=torch.float32).to(device)
        # with tensorpc.dbg.rttracer.enter_tracer("debug"):
        model(input, mask)
        complex_obj = mui.Button("Hello")
        arr = np.random.uniform(-1, 1, size=[1000, 3])
        _trace_func()

        tensorpc.dbg.breakpoint(name="WTF")
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

def _trace_func2():
    img = np.random.randint(0, 255, size=[1080, 1920, 3], dtype=np.uint8)
    img += 5
    conv = torch.nn.Conv2d(3, 3, 3)

def _trace_func():
    print("TRACE START")
    img = np.random.randint(0, 255, size=[1080, 1920, 3], dtype=np.uint8)
    img += 5
    time.sleep(0.5)

    print("TRACE STOP")

def main_mp_fork_debug():
    ctx = multiprocessing.get_context("spawn")
    num_proc = 2
    procs = []
    img = np.random.randint(0, 255, size=[1080, 1920, 3], dtype=np.uint8)
    imgs = np.random.randint(0, 255, size=[10, 480, 640, 3], dtype=np.uint8)

    # tensorpc.dbg.breakpoint(name="WTF", init_port=54322)
    # _trace_func()
    # tensorpc.dbg.breakpoint(name="WTF", init_port=54322)
    
    for j in range(num_proc):
        p = ctx.Process(target=mp_func_for_fork_debug, args=(j,))
        p.daemon = True
        p.start()
        procs.append(p)
    # # time.sleep(1)
    # # tensorpc.dbg.breakpoint(name="WTF2", init_port=54322)

    for p in procs:
        p.join()

def main_debug_inf_record():
    ctx = multiprocessing.get_context("spawn")
    num_proc = 2
    procs = []
    for j in range(num_proc):
        p = ctx.Process(target=mp_func_inf_record, args=(j,))
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == "__main__":
    main_debug_inf_record()