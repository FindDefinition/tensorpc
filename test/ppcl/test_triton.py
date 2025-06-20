import asyncio
from typing import Annotated, Any, Optional, Union

import rich
import numpy as np
import triton 
from tensorpc.apps.ppcl import tsim
from tensorpc.core import pfl
from tensorpc.core.pfl import compiler_print_type, compiler_print_metadata
from tensorpc.apps.ppcl.std import Tensor, PointerTensor, ppcl
from tensorpc.apps.ppcl.core import TensorMeta, DTypeEnum, ConstExprMeta
from tensorpc.apps.ppcl.backends import tritonstd
import triton.language as tl
import dataclasses

def _matmul_kernel_test_fn() -> pfl.PFLInlineRunEnv:
    M = 240
    N = 256
    K = 64
    a = np.random.uniform(-1, 1, size=[M, K]).astype(np.float32)
    b = np.random.uniform(-1, 1, size=[K, N]).astype(a.dtype)
    c = np.empty([M, N], dtype=a.dtype)
    c_ref = a @ b
    test_kwargs: dict[str, Any] = {
        "a_ptr": a,
        "b_ptr": b,
        "c_ptr": c,
        "M": M,
        "N": N,
        "K": K,
        "stride_am": a.strides[0] // a.itemsize,
        "stride_ak": a.strides[1] // a.itemsize,
        "stride_bk": b.strides[0] // b.itemsize,
        "stride_bn": b.strides[1] // b.itemsize,
        "stride_cm": c.strides[0] // c.itemsize,
        "stride_cn": c.strides[1] // c.itemsize,
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    ref_kwargs = {
        "c_ptr": c_ref
    }
    num_grid = triton.cdiv(M, test_kwargs["BLOCK_SIZE_M"]) * triton.cdiv(N, test_kwargs["BLOCK_SIZE_N"])
    return pfl.PFLInlineRunEnv(test_kwargs, userdata=tritonstd.TritonSimInfo((num_grid, 1, 1), ref_kwargs))

@triton.jit
@tritonstd.mark_triton_compilable(inline_run_env_fn=_matmul_kernel_test_fn)
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M: int, N: int, K: int,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am: tl.constexpr, stride_ak: tl.constexpr,  #
        stride_bk: tl.constexpr, stride_bn: tl.constexpr,  #
        stride_cm: tl.constexpr, stride_cn: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
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
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator.to(tl.float32)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    # # return c_mask
    tl.store(c_ptrs, c, mask=c_mask)

def _grouped_matmul_kernel_test_fn() -> pfl.PFLInlineRunEnv:
    np.random.seed(1212)
    num_gemm = 3
    all_mnks: list[tuple[int, int, int]] = []
    g_lds = []
    for j in range(num_gemm):
        M = np.random.randint(220, 260)
        N = np.random.randint(220, 260)
        K = np.random.randint(40, 70)
        all_mnks.append((M, N, K))
        g_lds.append((K, N, N))
    a_mem = np.random.uniform(-1, 1, size=[sum(x[0] * x[2] for x in all_mnks)]).astype(np.float32)
    b_mem = np.random.uniform(-1, 1, size=[sum(x[1] * x[2] for x in all_mnks)]).astype(np.float32)
    c_mem = np.empty([sum(x[0] * x[1] for x in all_mnks)], dtype=np.float32)
    c_mem_ref = np.empty_like(c_mem)
    mk_cumsum = 0
    kn_cumsum = 0
    mn_cumsum = 0
    a_offsets = np.zeros([num_gemm], np.int64)
    b_offsets = np.zeros([num_gemm], np.int64)
    c_offsets = np.zeros([num_gemm], np.int64)
    for i, (M, N, K) in enumerate(all_mnks):
        a = a_mem[mk_cumsum:mk_cumsum + M * K].reshape(M, K)
        a_offsets[i] = (mk_cumsum + M * K)
        mk_cumsum += M * K
        b = b_mem[kn_cumsum:kn_cumsum + K * N].reshape(K, N)
        b_offsets[i] = (kn_cumsum + K * N)
        kn_cumsum += N * K
        c_ref = c_mem_ref[mn_cumsum:mn_cumsum + M * N].reshape(M, N)
        c_offsets[i] = (mn_cumsum + M * N)
        mn_cumsum += M * N
        c_ref[:] = a @ b
    a_memstorage = tsim.create_sim_memory("a", a_mem)
    b_memstorage = tsim.create_sim_memory("b", b_mem)
    c_memstorage = tsim.create_sim_memory("c", c_mem)
    mnks = np.array(all_mnks, dtype=np.int32)
    a_pointer = tritonstd.PointerTensor(tsim.create_pointer_tensor(DTypeEnum.float32, a_offsets, a_memstorage))
    b_pointer = tritonstd.PointerTensor(tsim.create_pointer_tensor(DTypeEnum.float32, b_offsets, b_memstorage))
    c_pointer = tritonstd.PointerTensor(tsim.create_pointer_tensor(DTypeEnum.float32, c_offsets, c_memstorage))
    NUM_SM = 15
    
    test_kwargs: dict[str, Any] = {
        "group_a_ptrs": a_pointer,
        "group_b_ptrs": b_pointer,
        "group_c_ptrs": c_pointer,
        "group_gemm_sizes": mnks,
        "g_lds": np.array(g_lds, dtype=np.int32),
        "group_size": num_gemm,
        "NUM_SM": NUM_SM,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "BLOCK_SIZE_M": 32,
    }
    ref_kwargs = {
        "group_c_ptrs": c_mem_ref
    }
    return pfl.PFLInlineRunEnv(test_kwargs, userdata=tritonstd.TritonSimInfo((NUM_SM, 1, 1), ref_kwargs))


@triton.jit
@tritonstd.mark_triton_compilable(inline_run_env_fn=_grouped_matmul_kernel_test_fn)
def grouped_matmul_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float32))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float32))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float32))
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                # assume full tile for now
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(tl.float32)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]

            # assumes full tile for now
            tl.store(c_ptrs, c)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


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

def _main():
    # ast = pfl.parse_func_to_pfl_ast(_layer_norm_fwd_fused, backend="ppcl")
    runner = tritonstd.parse_triton_compilable_to_runner(matmul_kernel)
    lib = runner._library
    inline_env = lib.get_compiled_unit_inline_env(matmul_kernel.fn)
    sim_info = inline_env.get_userdata_typed(tritonstd.TritonSimInfo)
    print(sim_info.grid_size[0])
    for j in range(sim_info.grid_size[0]):
        with tsim.enter_tensorsim_context([j, 0, 0], sim_info.grid_size):
            asyncio.run(runner.run_func(lib.get_compiled_unit(matmul_kernel.fn).uid, inline_env.kwargs))
    
    c_ref = sim_info.ref_results["c_ptr"]
    print(np.linalg.norm(inline_env.kwargs["c_ptr"]._wrapped.memory_storage.data - c_ref))

def _main_softmax():
    # ast = pfl.parse_func_to_pfl_ast(_layer_norm_fwd_fused, backend="ppcl")
    runner = tritonstd.parse_triton_compilable_to_runner(softmax_kernel)
    lib = runner._library
    inline_env = lib.get_compiled_unit_inline_env(softmax_kernel.fn)
    sim_info = inline_env.get_userdata_typed(tritonstd.TritonSimInfo)
    for j in range(sim_info.grid_size[0]):
        with tsim.enter_tensorsim_context([j, 0, 0], sim_info.grid_size):
            asyncio.run(runner.run_func(lib.get_compiled_unit(softmax_kernel.fn).uid, inline_env.kwargs))
    
    c_ref = sim_info.ref_results["output_ptr"]
    print(np.linalg.norm(inline_env.kwargs["output_ptr"]._wrapped.memory_storage.data - c_ref))

def _main_grouped_gemm():
    # ast = pfl.parse_func_to_pfl_ast(_layer_norm_fwd_fused, backend="ppcl")
    runner = tritonstd.parse_triton_compilable_to_runner(grouped_matmul_kernel)
    lib = runner._library
    inline_env = lib.get_compiled_unit_inline_env(grouped_matmul_kernel.fn)
    sim_info = inline_env.get_userdata_typed(tritonstd.TritonSimInfo)
    for j in range(sim_info.grid_size[0]):
        with tsim.enter_tensorsim_context([j, 0, 0], sim_info.grid_size):
            asyncio.run(runner.run_func(lib.get_compiled_unit(grouped_matmul_kernel.fn).uid, inline_env.kwargs))
    
    c_ref = sim_info.ref_results["group_c_ptrs"]
    print(np.linalg.norm(inline_env.kwargs["group_c_ptrs"]._wrapped.memory_storage.data - c_ref))


if __name__ == "__main__":
    # _main()
    # _main_softmax()
    _main_grouped_gemm()