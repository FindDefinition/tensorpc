import asyncio
from typing import Annotated, Any, Optional, Union

import numpy as np
import triton 
from tensorpc.core import pfl
from tensorpc.apps.ppcl.backends import tritonstd
import triton.language as tl

def _softmax_kernel_test_fn() -> pfl.PFLInlineRunEnv:
    import torch
    x = torch.randn(1823, 781)
    x_np = x.numpy()
    y_np = np.empty_like(x)
    ref = torch.softmax(x, dim=1)


    BLOCK_SIZE = triton.next_power_of_2(x.shape[1])
    num_programs = x.shape[0]

    test_kwargs: dict[str, Any] = {
        "output_ptr": y_np,
        "input_ptr": x_np,
        "input_row_stride": x_np.strides[0] // x_np.itemsize,
        "output_row_stride": y_np.strides[0] // y_np.itemsize,
        "n_rows": x_np.shape[0],
        "n_cols": x_np.shape[1],
        "BLOCK_SIZE": BLOCK_SIZE,
        "num_stages": 3,
    }
    ref_kwargs = {
        "output_ptr": ref.numpy()
    }
    return pfl.PFLInlineRunEnv(test_kwargs, userdata=tritonstd.TritonSimInfo((num_programs, 1, 1), ref_kwargs))

@triton.jit
@tritonstd.mark_triton_compilable(inline_run_env_fn=_softmax_kernel_test_fn)
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def test_softmax():
    runner = tritonstd.parse_triton_compilable_to_runner(softmax_kernel)
    asyncio.run(runner.validate_kernel_by_test_data(softmax_kernel.fn))

if __name__ == "__main__":
    test_softmax()
