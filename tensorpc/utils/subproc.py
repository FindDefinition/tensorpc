"""use python 3.8 shared memory manager to pass arguments to subprocess in sync/async
"""
# require python 3.8
import asyncio
import subprocess
from tensorpc.constants import TENSORPC_SUBPROCESS_SMEM
from multiprocessing import shared_memory
import contextlib
from tensorpc.utils.typeutils import take_annotation_from
from tensorpc.core.core_io import dumps


@take_annotation_from(subprocess.check_call)
def check_call(*args, **kwargs):
    if "custom_data" not in kwargs:
        return subprocess.check_call(*args, **kwargs)
    try:
        data = dumps(kwargs["custom_data"])

        shm_a = shared_memory.SharedMemory(create=True, size=len(data))
    except:
        pass

    pass


breakpoint()
