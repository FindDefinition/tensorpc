from pathlib import Path
import traceback
import sys
import inspect 
import threading
from types import FrameType
import re
from typing import List, Optional, Set, Type
from tensorpc.core.tracer import FrameResult, Tracer
from tensorpc.flow.flowapp.components.plus.objinspect.treeitems import parse_frame_result_to_trace_item
import numpy as np 
THREAD_GLOBAL = threading.local()

def trace_Dev2(a, b):
    c = a + b 
    d = c * 2
    x = np.zeros([2, 3])
    xadd = np.unique(x)
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
    trace_res: List[FrameResult] = []
    tracer = Tracer(lambda x: trace_res.append(x), depth=3)
    with tracer:
        try:
            trace_dev(1, 2)
        except:
            pass 
    item = parse_frame_result_to_trace_item(trace_res)
    # print(item[0].child_trace_res)
    print([x.qualname for x in trace_res])
    pass 

if __name__ == '__main__':
    main()