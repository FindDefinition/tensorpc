from typing import Annotated, Any, Optional, Union

import rich
import numpy as np 
from tensorpc.core import pfl
from tensorpc.core.pfl import compiler_print_type, compiler_print_metadata
from tensorpc.apps.ppcl.std import Tensor, PointerTensor, ppcl
from tensorpc.apps.ppcl.core import TensorMeta, DTypeEnum, ConstExprMeta
from tensorpc.apps.ppcl.backends.triton_gen import pfl_ast_to_triton


def matmul_kernel(
        # Pointers to matrices
        a_ptr: PointerTensor, b_ptr: PointerTensor, c_ptr: PointerTensor,
        # Matrix dimensions
        M: int, N: int, K: int,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am: Annotated[int, ConstExprMeta()], stride_ak: Annotated[int, ConstExprMeta()],  #
        stride_bk: Annotated[int, ConstExprMeta()], stride_bn: Annotated[int, ConstExprMeta()],  #
        stride_cm: Annotated[int, ConstExprMeta()], stride_cn: Annotated[int, ConstExprMeta()],
        # Meta-parameters
        BLOCK_SIZE_M: Annotated[int, ConstExprMeta()], BLOCK_SIZE_N: Annotated[int, ConstExprMeta()], BLOCK_SIZE_K: Annotated[int, ConstExprMeta()],  #
        GROUP_SIZE_M: Annotated[int, ConstExprMeta()],  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = ppcl.program_id(0)
    num_pid_m = ppcl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = ppcl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    # wtf1 = ppcl.arange(0, BLOCK_SIZE_M)
    # wtf = wtf1 + pid_m
    # return
    offs_am = (pid_m * BLOCK_SIZE_M + ppcl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + ppcl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = ppcl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    pfl.compiler_print_type(a_ptrs)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = ppcl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=ppcl.float32)
    for k in range(0, ppcl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = ppcl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = ppcl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = ppcl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = ppcl.abs(accumulator.to(ppcl.float16))
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + ppcl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + ppcl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    # # return c_mask
    ppcl.store(c_ptrs, c, mask=c_mask)

def _layer_norm_fwd_fused(
    X: PointerTensor,  # pointer to the input
    Y: PointerTensor,  # pointer to the output
    W: PointerTensor,  # pointer to the weights
    B: PointerTensor,  # pointer to the biases
    Mean: PointerTensor,  # pointer to the mean
    Rstd: PointerTensor,  # pointer to the 1/std
    stride: int,  # how much to increase the pointer when moving by 1 row
    N: int,  # number of columns in X
    eps: float,  # epsilon to avoid division by zero
    BLOCK_SIZE: Annotated[int, ConstExprMeta()],
):
    # Map the program id to the row of X and Y it should compute.
    row = ppcl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    _mean = ppcl.zeros([BLOCK_SIZE], dtype=ppcl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + ppcl.arange(0, BLOCK_SIZE)
        a = ppcl.load(X + cols, mask=cols < N, other=0.).to(ppcl.float32)
        _mean += a
    mean = ppcl.sum(_mean, axis=0) / N
    # Compute variance
    _var = ppcl.zeros([BLOCK_SIZE], dtype=ppcl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + ppcl.arange(0, BLOCK_SIZE)
        x = ppcl.load(X + cols, mask=cols < N, other=0.).to(ppcl.float32)
        x = ppcl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = ppcl.sum(_var, axis=0) / N
    rstd = 1 / ppcl.sqrt(var + eps)
    # Write mean / rstd
    ppcl.store(Mean + row, mean)
    ppcl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + ppcl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = ppcl.load(W + cols, mask=mask)
        b = ppcl.load(B + cols, mask=mask)
        x = ppcl.load(X + cols, mask=mask, other=0.).to(ppcl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        ppcl.store(Y + cols, y, mask=mask)

def _main():
    ast = pfl.parse_func_to_pfl_ast(_layer_norm_fwd_fused, backend="ppcl")

    ast = pfl.parse_func_to_pfl_ast(matmul_kernel, backend="ppcl")
    # print(ast.ret_st)
    pfl.metaeval_total_tree(ast, {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "a_ptr": TensorMeta([], DTypeEnum.float16, is_pointer=True),
        "b_ptr": TensorMeta([], DTypeEnum.float16, is_pointer=True),
        "c_ptr": TensorMeta([], DTypeEnum.float16, is_pointer=True),
    }, backend="ppcl", code_for_error=ast.compile_info.code)
    for n in pfl.walk(ast):
        if isinstance(n, pfl.PFLName):
            print(f"\"{pfl.unparse_pfl_expr(n)}\"", n.st.metadata)
    # print(pfl.unparse_pfl_ast(ast))
    print(pfl_ast_to_triton(ast))
if __name__ == "__main__":
    _main()