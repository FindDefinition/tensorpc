import asyncio
from typing import Annotated, Any, Optional, Union

import rich
import numpy as np
import triton 
from tensorpc.apps.mls import tsim
from tensorpc.core import pfl
from tensorpc.core.pfl import compiler_print_type, compiler_print_metadata
from tensorpc.apps.mls.backends import tritonstd
import triton.language as tl
import dataclasses

def _debug_test_fn() -> pfl.PFLInlineRunEnv:

    test_kwargs: dict[str, Any] = {
        "x_ptr": np.zeros(8, dtype=np.int32),
        "BLOCK_SIZE": 128,
        "num_stages": 3,
    }
    return pfl.PFLInlineRunEnv(test_kwargs)


@triton.jit
@tritonstd.mark_triton_compilable(inline_run_env_fn=_debug_test_fn)
def debug_kernel(x_ptr, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    x = tl.load(x_ptr).to(tl.pointer_type(tl.float16))
    pfl.compiler_print_type(x_ptr)

def _block_tensor_test_fn() -> pfl.PFLInlineRunEnv:
    M = 128
    N = 125
    test_kwargs: dict[str, Any] = {
        "in_out_ptr": np.random.uniform(-1, 1, size=[M, N]).astype(np.float32),
        "M": M,
        "N": N,
        "M_BLOCK": 32,
        "N_BLOCK": 32,
    }
    return pfl.PFLInlineRunEnv(test_kwargs, userdata=tritonstd.TritonSimInfo((4, 4, 1), {
        "in_out_ptr": np.abs(test_kwargs["in_out_ptr"]),
    }))


@triton.jit
@tritonstd.mark_triton_compilable(inline_run_env_fn=_block_tensor_test_fn)
def inplace_abs(in_out_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    desc = tl.make_tensor_descriptor(
        in_out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )
    if isinstance(desc, tl._experimental_tensor_descriptor):
        print(12345)
    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK
    value = desc.load([moffset, noffset])
    desc.store([moffset, noffset], tl.abs(value))


def test_base_ops():
    runner = tritonstd.parse_triton_compilable_to_runner(inplace_abs)
    asyncio.run(runner.validate_kernel_by_test_data(inplace_abs.fn))

if __name__ == "__main__":
    test_base_ops()
