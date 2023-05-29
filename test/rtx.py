from pathlib import Path
import traceback
import sys
import inspect 
import threading
from types import FrameType
import re
from typing import Optional, Set, Type
from tensorpc.core.tracer import Tracer
THREAD_GLOBAL = threading.local()

def trace_Dev2(a, b):
    c = a + b 
    d = c * 2
    return c + d

def trace_dev(a, b):
    c = a + b 
    d = c * 2
    e = trace_Dev2(c, d)
    return c + d + e
import random 
def main():
    lst = []
    for i in range(10):
        lst.append(random.randrange(1, 1000))

    frame = inspect.currentframe()
    tracer = Tracer(lambda x: print(x), traced_names={"trace_dev", "trace_Dev2"})
    with tracer:
        trace_dev(1, 2)
    pass 

if __name__ == '__main__':
    main()