import asyncio
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Optional, Union

import numpy as np
import triton 
from tensorpc.core import pfl
from tensorpc.apps.mls.backends import tritonstd
import triton.language as tl

# np.seterr(all='raise')

def _attn_fwd_kernel_test_fn(is_fwd: bool = True) -> pfl.PFLInlineRunEnv:
    import torch 
    # TODO triton code don't support BLOCK_M < BLOCK_N
    BATCH, H, N_CTX, HEAD_DIM = 1, 1, 256, 64
    BLOCK_M = 32
    BLOCK_N = 32
    is_causal = True 
    stage = 3 if is_causal else 1
    sm_scale = 0.5
    torch.manual_seed(20)
    
    # q = torch.empty((BATCH, H, N_CTX, HEAD_DIM), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5).float().cpu().requires_grad_()
    # k = torch.empty((BATCH, H, N_CTX, HEAD_DIM), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5).float().cpu().requires_grad_()
    # v = torch.empty((BATCH, H, N_CTX, HEAD_DIM), dtype=torch.float16, device="cuda").normal_(mean=0.0, std=0.5).float().cpu().requires_grad_()
    # dref = torch.randn_like(q, dtype=torch.float16, device="cuda").float().cpu()

    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=torch.float32).normal_(mean=0.0, std=0.5).requires_grad_()
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=torch.float32).normal_(mean=0.0, std=0.5).requires_grad_()
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=torch.float32).normal_(mean=0.0, std=0.5).requires_grad_()
    dref = torch.randn_like(q)

    M = torch.empty((BATCH, H, N_CTX), dtype=torch.float32)
    M_np = M.detach().numpy()
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=is_causal, scale=sm_scale
    )
    ref.backward(dref)
    dref_np = dref.detach().numpy()
    dq_np = q.grad.numpy()
    dk_np = k.grad.numpy()
    dv_np = v.grad.numpy()
    q_np = q.detach().numpy()
    k_np = k.detach().numpy()
    v_np = v.detach().numpy()
    o_np = np.empty_like(ref.detach().numpy())
    ref_np = ref.detach().numpy()
    y_dim = q.shape[0] * q.shape[1] * q.shape[2]
    fwd_grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    fwd_kwargs: dict[str, Any] = {
        "sm_scale": sm_scale,
        "M": M_np,
        "Z": q.shape[0], 
        "H": q.shape[1],
        "Q": q_np,
        "K": k_np,
        "V": v_np,
        "Out": o_np,
        "stride_qz": q.stride(0),
        "stride_qh": q.stride(1),
        "stride_qm": q.stride(2),
        "stride_qk": q.stride(3),
        "stride_kz": k.stride(0),
        "stride_kh": k.stride(1),
        "stride_kn": k.stride(2),
        "stride_kk": k.stride(3),
        "stride_vz": v.stride(0),
        "stride_vh": v.stride(1),
        "stride_vk": v.stride(2),
        "stride_vn": v.stride(3),
        "stride_oz": ref.stride(0),
        "stride_oh": ref.stride(1),
        "stride_om": ref.stride(2),
        "stride_on": ref.stride(3),

        "STAGE": stage,
        "N_CTX": N_CTX,
        "HEAD_DIM": HEAD_DIM,
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "warp_specialize": False,
    }
    if is_fwd:
        ref_kwargs = {
            "Out": ref_np.reshape(y_dim, -1)
        }
        vis_layout=[
            [None, "Q.T", None],
            ["K", "Out.T", "V"]
        ]
        return pfl.PFLInlineRunEnv(fwd_kwargs, userdata=tritonstd.TritonSimInfo(fwd_grid, ref_kwargs, vis_layout=vis_layout))
    else:
        delta = torch.empty_like(M)
        delta = (ref * dref).sum(dim=-1)

        delta_np = delta.detach().numpy()
        # we have to run fwd kernel here to get correct M.
        runner = tritonstd.parse_triton_compilable_to_runner(_attn_fwd, module_code_getter=lambda x: Path(__file__).read_text())
        global_mem_fwd = tritonstd.create_global_mem_from_kwargs(fwd_kwargs)
        runner.run_kernel_in_executor(
            _attn_fwd.fn, grid_size=fwd_grid, global_mem=global_mem_fwd, **fwd_kwargs)
        M_np = global_mem_fwd.memory_blocks["M"].get_data_view_checked().copy()
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * H)
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        grid = (N_CTX // BLOCK_N1, 1, BATCH * H)
        test_kwargs: dict[str, Any] = {
            "sm_scale": sm_scale,
            "Q": q_np,
            "K": arg_k.detach().numpy(),
            "V": v_np,
            "DO": dref_np,
            "DQ": np.empty_like(dq_np),
            "DK": np.empty_like(dk_np),
            "DV": np.empty_like(dv_np),
            "M": M_np,
            "D": delta_np,
            "stride_z": q.stride(0),
            "stride_h": q.stride(1),
            "stride_tok": q.stride(2),
            "stride_d": q.stride(3),
            "H": H,

            "N_CTX": N_CTX,
            "HEAD_DIM": HEAD_DIM,
            "BLOCK_M1": BLOCK_M1,
            "BLOCK_N1": BLOCK_N1,
            "BLOCK_M2": BLOCK_M2,
            "BLOCK_N2": BLOCK_N2,
            "BLK_SLICE_FACTOR": BLK_SLICE_FACTOR,
        }
        ref_kwargs = {
            "DV": dv_np,
            "DK": dk_np,
            "DQ": dq_np,
        }
        return pfl.PFLInlineRunEnv(test_kwargs, userdata=tritonstd.TritonSimInfo(grid, ref_kwargs))

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0.0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


@triton.jit
@tritonstd.mark_triton_compilable(inline_run_env_fn=_attn_fwd_kernel_test_fn)
def _attn_fwd(Q, K, V, sm_scale, M, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):
    dtype = tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = tl.cast(off_z, tl.int64) * stride_qz + tl.cast(off_h, tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX, False  #
                                        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX, False  #
                                        )
    # epilogue
    m_i += tl.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(dtype))

# The main inner-loop logic for computing dK and dV.
@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_bwd_dkdv(dk, dv,  #
                   Q, k, v, sm_scale,  #
                   DO,  #
                   M, D,  #
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   H, N_CTX, BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                   MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_bwd_dq(dq, q, K, V,  #
                 do, m, D,
                 # shared by Q/K/V/DO.
                 stride_tok, stride_d,  #
                 H, N_CTX,  #
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps,  #
                 MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


@triton.jit
@tritonstd.mark_triton_compilable(inline_run_env_fn=partial(_attn_fwd_kernel_test_fn, is_fwd=False))
def _attn_bwd(Q, K, V, sm_scale,  #
              DO,  #
              DQ, DK, DV,  #
              M, D,
              # shared by Q/K/V/DO.
              stride_z, stride_h, stride_tok, stride_d,  #
              H, N_CTX,  #
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              HEAD_DIM: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = tl.cast(bhid * N_CTX, tl.int64)
    adj = tl.cast(stride_h * (bhid % H) + stride_z * (bhid // H), tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = start_n

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    num_steps = BLOCK_N1 // MASK_BLOCK_M1

    dk, dv = _attn_bwd_dkdv(dk, dv,  #
                            Q, k, v, sm_scale,  #
                            DO,  #
                            M, D,  #
                            stride_tok, stride_d,  #
                            H, N_CTX,  #
                            MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
                            start_n, start_m, num_steps,  #
                            MASK=True  #
                            )

    start_m += num_steps * MASK_BLOCK_M1
    num_steps = (N_CTX - start_m) // BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dkdv(  #
        dk, dv,  #
        Q, k, v, sm_scale,  #
        DO,  #
        M, D,  #
        stride_tok, stride_d,  #
        H, N_CTX,  #
        BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
        start_n, start_m, num_steps,  #
        MASK=False  #
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    num_steps = BLOCK_M2 // MASK_BLOCK_N2
    dq = _attn_bwd_dq(dq, q, K, V,  #
                      do, m, D,  #
                      stride_tok, stride_d,  #
                      H, N_CTX,  #
                      BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,  #
                      start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,  #
                      MASK=True  #
                      )
    end_n -= num_steps * MASK_BLOCK_N2
    # stage 2
    num_steps = end_n // BLOCK_N2
    dq = _attn_bwd_dq(dq, q, K, V,  #
                      do, m, D,  #
                      stride_tok, stride_d,  #
                      H, N_CTX,  #
                      BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
                      start_m, end_n - num_steps * BLOCK_N2, num_steps,  #
                      MASK=False  #
                      )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)



def test_attn_fwd():
    runner = tritonstd.parse_triton_compilable_to_runner(_attn_fwd)
    asyncio.run(runner.validate_kernel_by_test_data(_attn_fwd.fn))

def test_attn_bwd():
    runner = tritonstd.parse_triton_compilable_to_runner(_attn_bwd)
    asyncio.run(runner.validate_kernel_by_test_data(_attn_bwd.fn))

def _main():
    test_attn_fwd()
    test_attn_bwd()

if __name__ == "__main__":
    _main()
