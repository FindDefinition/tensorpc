import asyncio
from typing import Annotated, Any, Optional, Union

import numpy as np
import triton 
from tensorpc.apps.ppcl import tsim
from tensorpc.core import pfl
from tensorpc.apps.ppcl.backends import tritonstd
import triton.language as tl

def _grouped_matmul_kernel_test_fn() -> pfl.PFLInlineRunEnv:
    np.random.seed(1212)
    num_gemm = 3
    all_mnks: list[tuple[int, int, int]] = []
    g_lds = []
    for j in range(num_gemm):
        M = np.random.randint(220, 260)
        N = np.random.randint(220, 260)
        K = np.random.randint(40, 70)
        M = 256
        N = 256
        K = 64
        all_mnks.append((M, N, K))
        g_lds.append((K, N, N))
    g_lds_mem = np.array(g_lds, dtype=np.int32)
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
        a_offsets[i] = (mk_cumsum)
        mk_cumsum += M * K
        b = b_mem[kn_cumsum:kn_cumsum + K * N].reshape(K, N)
        b_offsets[i] = (kn_cumsum)
        kn_cumsum += N * K
        c_ref = c_mem_ref[mn_cumsum:mn_cumsum + M * N].reshape(M, N)
        c_offsets[i] = (mn_cumsum)
        mn_cumsum += M * N
        c_ref[:] = a @ b
    mnks = np.array(all_mnks, dtype=np.int32)
    # we have to create a single memory here to support pointer of pointer.
    global_mem = tsim.create_sim_memory({
        "group_a_ptrs": a_offsets,
        "group_b_ptrs": b_offsets,
        "group_c_ptrs": c_offsets,
        "as": a_mem,
        "bs": b_mem,
        "cs": c_mem,
        "g_lds": g_lds_mem,
        "group_gemm_sizes": mnks,
    })
    a_offsets = global_mem.memory_blocks["group_a_ptrs"].get_data_view_checked()
    b_offsets = global_mem.memory_blocks["group_b_ptrs"].get_data_view_checked()
    c_offsets = global_mem.memory_blocks["group_c_ptrs"].get_data_view_checked()

    a_offsets += global_mem.memory_blocks["as"].offset_with_hole
    b_offsets += global_mem.memory_blocks["bs"].offset_with_hole
    c_offsets += global_mem.memory_blocks["cs"].offset_with_hole
    NUM_SM = 15
    test_kwargs: dict[str, Any] = {
        "group_a_ptrs": a_offsets,
        "group_b_ptrs": b_offsets,
        "group_c_ptrs": c_offsets,
        "group_gemm_sizes": mnks,
        "g_lds": g_lds_mem,
        "group_size": num_gemm,
        "NUM_SM": NUM_SM,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "BLOCK_SIZE_M": 32,
    }
    ref_kwargs = {
        "cs": c_mem_ref
    }
    return pfl.PFLInlineRunEnv(test_kwargs, userdata=tritonstd.TritonSimInfo((NUM_SM, 1, 1), ref_kwargs, global_mem))


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
            a_ptr = tl.cast(tl.load(group_a_ptrs + g), tl.pointer_type(tl.float32))
            b_ptr = tl.cast(tl.load(group_b_ptrs + g), tl.pointer_type(tl.float32))
            c_ptr = tl.cast(tl.load(group_c_ptrs + g), tl.pointer_type(tl.float32))
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
            accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
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
        
def _comparer(x, y):
    np.testing.assert_allclose(x, y, rtol=1e-3, atol=1e-3)

def test_grouped_mm():
    runner = tritonstd.parse_triton_compilable_to_runner(grouped_matmul_kernel)
    asyncio.run(runner.validate_kernel_by_test_data(grouped_matmul_kernel.fn, _comparer))

if __name__ == "__main__":
    test_grouped_mm()
