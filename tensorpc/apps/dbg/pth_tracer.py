import contextlib 
import os
import tempfile
from .bkpt import _TRACER_WRAPPER, RECORDING


@contextlib.contextmanager
def pytorch_profiler():
    import torch.profiler as profiler

    with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA], 
                        with_stack=False, profile_memory=False) as p:
        yield p
    # if _TRACER_WRAPPER.
    fp = tempfile.NamedTemporaryFile("w+t", suffix=".json", delete=False)
    fp.close()
    p.export_chrome_trace(fp.name)
    with open(fp.name, "rb") as f:
        data = f.read()
    # remove temp file
    os.remove(fp.name)
