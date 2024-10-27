import concurrent
import io
import json
import multiprocessing
import os
import numpy as np
import concurrent.futures
import rich
import torch
def _trace_func2():
    img = np.random.randint(0, 255, size=[1080, 1920, 3], dtype=np.uint8)
    img += 5
    conv = torch.nn.Conv2d(3, 3, 3)

def _trace_func():
    print("TRACE START")
    img = np.random.randint(0, 255, size=[1080, 1920, 3], dtype=np.uint8)
    img += 5
    # _trace_func2()
    # conv = torch.nn.Conv2d(3, 3, 3)

    print("TRACE STOP")

def trace_func(j, tracer):
    if j == 0:
        tracer.start()
    a = np.random.uniform(-1, 1)
    b = 5
    c = a * b
    if j == 2:
        tracer.stop()

def dev_trace():
    ctx = multiprocessing.get_context("fork")
    num_proc = 2
    procs = []
    img = np.random.randint(0, 255, size=[1080, 1920, 3], dtype=np.uint8)
    imgs = np.random.randint(0, 255, size=[10, 480, 640, 3], dtype=np.uint8)
    from viztracer import VizTracer
    tracer = VizTracer()
    tracer.start()
    # for j in range(4):
    #     trace_func(j, tracer)
    _trace_func()
    tracer.stop()
    ss = io.StringIO()
    tracer.save(ss)
    rich.print(json.loads(ss.getvalue()))


if __name__ == "__main__":
    dev_trace()