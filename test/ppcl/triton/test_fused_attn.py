import asyncio
from functools import partial
from pathlib import Path
import traceback
from typing import Annotated, Any, Optional, Union

import numpy as np
import triton 
from tensorpc.core import pfl
from tensorpc.apps.mls.backends import tritonstd
import triton.language as tl
import torch 

# np.seterr(all='raise')
def _attn_fwd_grid(META, q_shape):
    return (triton.cdiv(q_shape[2], META["BLOCK_M"]), q_shape[0] * q_shape[1], 1)

def _attn_fwd_kernel_test_fn(N_CTX: int = 256, HEAD_DIM: int = 64, H = 1, dtype = torch.float32, 
        is_fwd: bool = True, head_first: bool = True, DKV_VARIANT: int = 0, DQ_VARIANT: int = 0,
        is_tma: bool = False) -> pfl.PFLInlineRunEnv:
    # TODO triton code don't support BLOCK_M < BLOCK_N
    BATCH = 1
    BLOCK_M = 32
    BLOCK_N = 32
    is_causal = False 
    stage = 3 if is_causal else 1
    sm_scale = 0.5
    torch.manual_seed(20)
    M = torch.empty((BATCH, H, N_CTX), dtype=torch.float32)
    M_np = M.detach().numpy()
    shape = (BATCH, H, N_CTX, HEAD_DIM) if head_first else (BATCH, N_CTX, H, HEAD_DIM)
    if N_CTX > 1000:
        # random on cpu is very slow.
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_()
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_()
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_()
        dref = torch.randn_like(q)

        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=is_causal, scale=sm_scale
        )
        dref = dref.cpu()
        ref = ref.cpu() 
    else:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype).normal_(mean=0.0, std=0.5).requires_grad_()
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype).normal_(mean=0.0, std=0.5).requires_grad_()
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype).normal_(mean=0.0, std=0.5).requires_grad_()
        dref = torch.randn_like(q)

        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=is_causal, scale=sm_scale
        )
    ref.backward(dref)
    dref_np = dref.detach().numpy()
    if q.is_cuda:
        dq_np = q.grad.cpu().numpy()
        dk_np = k.grad.cpu().numpy()
        dv_np = v.grad.cpu().numpy()
        q = q.cpu()
        k = k.cpu()
        v = v.cpu()

    else:
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
    if is_tma:
        desc_q = tritonstd.HostTensorDescriptor(q_np.reshape(y_dim, -1), block_shape=[BLOCK_M, HEAD_DIM])
        desc_v = tritonstd.HostTensorDescriptor(v_np.reshape(y_dim, -1), block_shape=[BLOCK_N, HEAD_DIM])
        desc_k = tritonstd.HostTensorDescriptor(k_np.reshape(y_dim, -1), block_shape=[BLOCK_N, HEAD_DIM])
        desc_o = tritonstd.HostTensorDescriptor(o_np.reshape(y_dim, -1), block_shape=[BLOCK_M, HEAD_DIM])

        fwd_kwargs: dict[str, Any] = {
            "sm_scale": sm_scale,
            "M": M_np,
            "Z": q.shape[0], 
            "H": q.shape[1],

            "desc_q": desc_q,
            "desc_v": desc_v,
            "desc_k": desc_k,
            "desc_o": desc_o,
            "STAGE": stage,
            "N_CTX": N_CTX,
            "HEAD_DIM": HEAD_DIM,
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "warp_specialize": False,
            "FP8_OUTPUT": False,
        }
    else:
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
        }
    if is_fwd:
        if is_tma:  
            ref_kwargs = {
                "desc_o": ref_np.reshape(y_dim, -1)
            }
            vis_layout=[
                [None, "desc_q.T", None],
                ["desc_k", "desc_o.T", "desc_v"]
            ]
        else:
            ref_kwargs = {
                "Out": ref_np # .reshape(y_dim, -1)
            }
            vis_layout=[
                [None, "Q.T", None],
                ["K", "Out.T", "V"]
            ]
        sim_info = tritonstd.TritonSimInfo(fwd_grid, ref_kwargs, vis_layout=vis_layout, grid_size_for_triton=partial(_attn_fwd_grid, q_shape=q.shape))
        return pfl.PFLInlineRunEnv(fwd_kwargs, userdata=sim_info)
    else:
        delta = torch.empty_like(M)
        delta = (ref * dref).sum(dim=-1)

        delta_np = delta.detach().numpy()
        # we have to run fwd kernel here to get correct M.
        # when we run real triton kernel, we shouldn't run slow fwd kernel in simulation.
        if N_CTX <= 1000:
            runner = tritonstd.parse_triton_compilable_to_runner(_attn_fwd, module_code_getter=lambda x: Path(__file__).read_text())
            global_mem_fwd = tritonstd.create_global_mem_from_kwargs(fwd_kwargs)
            runner.run_kernel_in_executor(
                _attn_fwd.fn, grid_size=fwd_grid, global_mem=global_mem_fwd, **fwd_kwargs)
            M_np = global_mem_fwd.memory_blocks["M"].get_data_view_checked().copy()
        BLK_SLICE_FACTOR = 1
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * H)
        DQ_ATOMIC = True # faster in Hopper, slower in Ampere
        if DQ_ATOMIC:
            BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 64, 128, 128, 64
        else:
            if HEAD_DIM == 64:
                BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 64, 64, 32
            else:
                BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 64, 128, 128, 64
        grid = (N_CTX // BLOCK_N1, BATCH, H)
        if not head_first:
            grid = (H, N_CTX // BLOCK_N1, BATCH)
        if not head_first:
            q_np = np.ascontiguousarray(np.moveaxis(q_np, 1, 2))
            k_np_scaled = np.ascontiguousarray(np.moveaxis(arg_k.detach().numpy(), 1, 2))
            v_np = np.ascontiguousarray(np.moveaxis(v_np, 1, 2))
            dref_np = np.ascontiguousarray(np.moveaxis(dref_np, 1, 2))
            dq_np = np.ascontiguousarray(np.moveaxis(dq_np, 1, 2))
            dk_np = np.ascontiguousarray(np.moveaxis(dk_np, 1, 2))

            dv_np = np.ascontiguousarray(np.moveaxis(dv_np, 1, 2))
            q = q.transpose(1, 2).contiguous()
        else:
            k_np_scaled = arg_k.detach().numpy()
        if DQ_ATOMIC:
            dq_np_out = np.zeros_like(dq_np).astype(np.float32)
        else:
            dq_np_out = np.empty_like(dq_np)
        dk_np_out = np.empty_like(dk_np)
        dv_np_out = np.empty_like(dv_np)
        test_kwargs: dict[str, Any] = {
            "sm_scale": sm_scale,
            "Q": q_np,
            "K": k_np_scaled,
            "V": v_np,
            "DO": dref_np,
            "DQ": dq_np_out,
            "DK": dk_np_out,
            "DV": dv_np_out,
            "M": M_np,
            "D": delta_np,
            "stride_z": q.stride(0),
            "stride_h": q.stride(1 if head_first else 2),
            "stride_tok": q.stride(2 if head_first else 1),
            "stride_d": q.stride(3),
            "H": H,
            "N_CTX": N_CTX,
            "HEAD_DIM": HEAD_DIM,
            "BLOCK_M1": BLOCK_M1,
            "BLOCK_N1": BLOCK_N1,
            "BLOCK_M2": BLOCK_M2,
            "BLOCK_N2": BLOCK_N2,
            "BLK_SLICE_FACTOR": BLK_SLICE_FACTOR,
            "num_warps": 8 if DQ_ATOMIC else 8,
            "num_stages": 3 if DQ_ATOMIC else 3,
            "IS_CAUSAL": is_causal,
            "is_head_first": head_first,
            "maxnreg": 256,
            "DQ_ATOMIC": DQ_ATOMIC,
            "DKV_VARIANT": DKV_VARIANT,
            "DQ_VARIANT": DQ_VARIANT,
        }
        ref_kwargs = {
            "DV": dv_np,
            "DK": dk_np,
            "DQ": dq_np,
        }
        return pfl.PFLInlineRunEnv(test_kwargs, userdata=tritonstd.TritonSimInfo(grid, ref_kwargs, 
            grid_size_for_triton=lambda META: grid))

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
            qk = qk * qk_scale + tl.where(mask, 0.0, -float("inf"))
            # stable softmax: calc max of each row
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
            p = p.to(q.dtype)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64]\
    for s in ([3, 4, 5, 6, 7])\
    for w in [4, 8]\
]

def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True

# @triton.jit
# @tritonstd.mark_triton_compilable(is_template=True)
# def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
#     if isinstance(desc_or_ptr, tl._experimental_tensor_descriptor):
#         return desc_or_ptr
#     else:
#         return tl._experimental_make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)

TMA_NUM_STAGES_OPTIONS = [2, 3, 4]


def _host_descriptor_pre_hook(nargs):
    from triton.tools.tensor_descriptor import TensorDescriptor

    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]



tma_configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w, pre_hook=_host_descriptor_pre_hook) \
    for BM in [64, 128]\
    for BN in [32, 64, 128]\
    for s in TMA_NUM_STAGES_OPTIONS \
    for w in [4, 8]\
]

def tma_keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (torch.cuda.get_device_capability()[0] == 9 and BLOCK_M * BLOCK_N < 128 * 128
                and conf.num_warps == 8)
    
def tma_prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]

    # Filter out configs where BLOCK_M > N_CTX
    return [conf for conf in configs if conf.kwargs.get("BLOCK_M", 0) <= N_CTX]


def _torch_bench_fn_fwd(kwargs):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    if "desc_q" in kwargs:
        # host desc is reshaped to 2d, we need to recover 4d.
        N_CTX = kwargs["N_CTX"]
        Z = kwargs["Z"]
        H = kwargs["H"]

        q = kwargs["desc_q"].view(Z, H, N_CTX, -1)
        k = kwargs["desc_k"].view(Z, H, N_CTX, -1)
        v = kwargs["desc_v"].view(Z, H, N_CTX, -1)
    else:
        q = kwargs["Q"]
        k = kwargs["K"]
        v = kwargs["V"]
    with tritonstd.measure_duration_torch() as dur:
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):

            res = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=kwargs["STAGE"] == 3, scale=kwargs["sm_scale"]
            )
    return dur.val

def _xformers_bench_fn_fwd(kwargs):
    from xformers.ops import fmha
    if "desc_q" in kwargs:
        # host desc is reshaped to 2d, we need to recover 4d.
        N_CTX = kwargs["N_CTX"]
        Z = kwargs["Z"]
        H = kwargs["H"]

        q = kwargs["desc_q"].view(Z, H, N_CTX, -1)
        k = kwargs["desc_k"].view(Z, H, N_CTX, -1)
        v = kwargs["desc_v"].view(Z, H, N_CTX, -1)
    else:
        q = kwargs["Q"]
        k = kwargs["K"]
        v = kwargs["V"]
    with tritonstd.measure_duration_torch() as dur:
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        res = fmha.memory_efficient_attention(q, k, v)

    return dur.val

def _torch_bench_fn_bwd(kwargs):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q = kwargs["Q"].detach().clone()
    k = kwargs["K"].detach().clone()
    v = kwargs["V"].detach().clone()
    if not kwargs["is_head_first"]:
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):

        res = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=kwargs["IS_CAUSAL"], scale=kwargs["sm_scale"]
        )
        dres = torch.empty_like(res)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            with tritonstd.measure_duration_torch() as dur:

                res.backward(dres)
    prof.export_chrome_trace("/root/bwd_trace.json")
    return dur.val

def _torch_bench_fn_bwd_xformer(kwargs):
    from xformers.ops import fmha
    q = kwargs["Q"].detach().clone()
    k = kwargs["K"].detach().clone()
    v = kwargs["V"].detach().clone()
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    res = fmha.memory_efficient_attention(q, k, v)
    dres = torch.empty_like(res)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        with tritonstd.measure_duration_torch() as dur:
            res.backward(dres)
    prof.export_chrome_trace("/root/bwd_trace_xformers.json")
    return dur.val

@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
@tritonstd.mark_triton_compilable(inline_run_env_fn=_attn_fwd_kernel_test_fn, 
    real_kwargs={"N_CTX": 30720, "HEAD_DIM": 128, "H": 16, "dtype": torch.float16},
    raw_fn=_torch_bench_fn_fwd,
)
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
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    tl.static_assert(BLOCK_M >= BLOCK_N)

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
    dtype = q.dtype
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

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_fwd_inner_tma(acc, l_i, m_i, q,  #
                    desc_k, desc_v,  #
                    offset_y, dtype: tl.constexpr, start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, warp_specialize: tl.constexpr):
    # range of values handled by this stage
    # print("INNER", STAGE, acc.dtype)
    if STAGE == 1:
        # run full part
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        # run last casual part
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = desc_k.load([offsetk_y, 0]).T
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
        # -- compute correction factor
        alpha = tl.exp2(m_i - m_ij)
        # print("DTYPE", alpha.dtype, m_i.dtype, m_ij.dtype, qk.dtype)
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        if warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute([0, 2, 1]).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]
        # prepare p and v for the dot
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


@triton.autotune(configs=list(filter(tma_keep, tma_configs)), key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
                 prune_configs_by={'early_config_prune': tma_prune_invalid_configs})
@triton.jit
@tritonstd.mark_triton_compilable(inline_run_env_fn=partial(_attn_fwd_kernel_test_fn, is_tma=True),
    real_kwargs={"N_CTX": 30720, "HEAD_DIM": 128, "H": 16, "dtype": torch.float16},
    raw_fn=_torch_bench_fn_fwd,
)
def _attn_fwd_tma(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #
              ):
    dtype = tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                        block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    # print("!", start_m, qo_offset_y, off_hz, sm_scale)

    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner_tma(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        offset_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX,  #
                                        warp_specialize)
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner_tma(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        offset_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX,  #
                                        warp_specialize)
    # epilogue
    m_i += tl.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


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
    # qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    q_ptrs = Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d

    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        # qT = tl.load(qT_ptrs)
        q = tl.load(q_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        # qkT = tl.dot(k, qT)
        qkT = tl.dot(k, tl.trans(q))

        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        # Compute dV.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        # dk += tl.dot(dsT, tl.trans(qT))
        dk += tl.dot(dsT, q)

        # Increment pointers.
        curr_m += step_m
        # qT_ptrs += step_m * stride_tok
        q_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_bwd_dqdkdv_q_k_v(DQ, dk, dv,  #
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
                   MASK: tl.constexpr,
                   DQ_ATOMIC: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    # qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    q_ptrs = Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d

    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        # qT = tl.load(qT_ptrs)
        q = tl.load(q_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        # qkT = tl.dot(k, qT)
        qkT = tl.dot(k, tl.trans(q))

        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        # Compute dV.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        # dk += tl.dot(dsT, tl.trans(qT))
        dk += tl.dot(dsT, q)
        if DQ_ATOMIC:
            dq = tl.dot(tl.trans(dsT), k).to(tl.float32) * 0.6931471824645996
            tl.atomic_add(
                dq_ptrs,
                dq,
                sem="relaxed",
            )
        # Increment pointers.
        curr_m += step_m
        # qT_ptrs += step_m * stride_tok
        q_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
        if DQ_ATOMIC:
            dq_ptrs += step_m * stride_tok
    return dk, dv

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_bwd_dqdkdv_qT_k_v(DQ, dk, dv,  #
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
                   MASK: tl.constexpr,
                   DQ_ATOMIC: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d

    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # q = tl.load(q_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        # qkT = tl.dot(k, tl.trans(q))

        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        # Compute dV.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))
        # dk += tl.dot(dsT, q)
        if DQ_ATOMIC:
            dq = tl.dot(tl.trans(dsT), k).to(tl.float32) * 0.6931471824645996
            tl.atomic_add(
                dq_ptrs,
                dq,
                sem="relaxed",
            )
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        # q_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
        if DQ_ATOMIC:
            dq_ptrs += step_m * stride_tok
    return dk, dv

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_bwd_dqdkdv_q_kT_vT(DQ, dk, dv,  #
                   Q, kT, vT, sm_scale,  #
                   DO,  #
                   M, D,  #
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   H, N_CTX, BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                   MASK: tl.constexpr,
                   DQ_ATOMIC: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    # qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    q_ptrs = Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # dqT_ptrs = DQ + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d

    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    # kT = tl.trans(k)
    # vT = tl.trans(v)
    for blk_idx in range(num_steps):
        # qT = tl.load(qT_ptrs)
        q = tl.load(q_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        # qkT = tl.dot(k, qT)
        qk = tl.dot(q, kT)

        p = tl.math.exp2(qk - m[:, None])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            p = tl.where(mask, p, 0.0)
        do = tl.load(do_ptrs)
        # Compute dV.
        dp = tl.dot(do, vT).to(tl.float32)
        pp = p
        pp = pp.to(tl.float16)
        dv += tl.dot(tl.trans(pp), do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        # dk += tl.dot(dsT, tl.trans(qT))
        dk += tl.dot(tl.trans(ds), q)
        # # dqT = tl.dot(kT, tl.trans(ds)).to(tl.float32) * 0.6931471824645996
        if DQ_ATOMIC:
            dq = tl.dot(ds, tl.trans(kT)).to(tl.float32) * 0.6931471824645996
            tl.atomic_add(
                dq_ptrs,
                dq,
                sem="relaxed",
            )
        # Increment pointers.
        curr_m += step_m
        # qT_ptrs += step_m * stride_tok
        q_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
        if DQ_ATOMIC:
            dq_ptrs += step_m * stride_tok
    return dk, dv

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_bwd_dqdkdv_qT_kT_vT(DQ, dk, dv,  #
                   Q, kT, vT, sm_scale,  #
                   DO,  #
                   M, D,  #
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   H, N_CTX, BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                   MASK: tl.constexpr,
                   DQ_ATOMIC: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    # q_ptrs = Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dqT_ptrs = DQ + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d

    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    # kT = tl.trans(k)
    # vT = tl.trans(v)
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # q = tl.load(q_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        # qkT = tl.dot(k, qT)
        qkT = tl.dot(tl.trans(kT), qT)

        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        # Compute dV.
        dpT = tl.dot(tl.trans(vT), tl.trans(do)).to(tl.float32)
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))
        # dk += tl.dot(tl.trans(ds), q)
        if DQ_ATOMIC:
            dqT = tl.dot(kT, dsT).to(tl.float32) * 0.6931471824645996

            tl.atomic_add(
                dqT_ptrs,
                dqT,
                sem="relaxed",
            )
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        # q_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
        if DQ_ATOMIC:
            dqT_ptrs += step_m * stride_tok
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
    # kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    k_ptrs = K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    v_ptrs = V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d

    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        qk = tl.dot(q, tl.trans(k))
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq = tl.dot(ds, k, dq)
        # Increment pointers.
        curr_n += step_n
        k_ptrs += step_n * stride_tok
        v_ptrs += step_n * stride_tok
    return dq

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_bwd_dq_qT(dq, q, K, V,  #
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
    # kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    k_ptrs = K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    v_ptrs = V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d

    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    qT = tl.trans(q)
    for blk_idx in range(num_steps):
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq = tl.dot(tl.trans(dsT), k, dq)
        # Increment pointers.
        curr_n += step_n
        k_ptrs += step_n * stride_tok
        v_ptrs += step_n * stride_tok
    return dq

@triton.jit
@tritonstd.mark_triton_compilable(inline_run_env_fn=partial(_attn_fwd_kernel_test_fn, is_fwd=False),
    real_kwargs={"N_CTX": 30720, "HEAD_DIM": 128, "H": 16, "dtype": torch.float16},
    raw_fn=_torch_bench_fn_bwd,
)
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
              HEAD_DIM: tl.constexpr,
              IS_CAUSAL: tl.constexpr,
              DQ_ATOMIC: tl.constexpr,
              DKV_VARIANT: tl.constexpr,
              DQ_VARIANT: tl.constexpr):
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
    # kT = tl.load(K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d)
    # vT = tl.load(V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d)

    start_m = 0
    num_steps = N_CTX // BLOCK_M1
    if DKV_VARIANT == 0:
        kT = tl.load(K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d)
        vT = tl.load(V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d)

        dk, dv = _attn_bwd_dqdkdv_q_kT_vT(  #
            DQ, dk, dv,  #
            Q, kT, vT, sm_scale,  #
            DO,  #
            M, D,  #
            stride_tok, stride_d,  #
            H, N_CTX,  #
            BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
            start_n, start_m, num_steps,  #
            MASK=False,  #
            DQ_ATOMIC=DQ_ATOMIC,
        )
    elif DKV_VARIANT == 1:
        k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
        v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

        dk, dv = _attn_bwd_dqdkdv_qT_k_v(  #
            DQ, dk, dv,  #
            Q, k, v, sm_scale,  #
            DO,  #
            M, D,  #
            stride_tok, stride_d,  #
            H, N_CTX,  #
            BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
            start_n, start_m, num_steps,  #
            MASK=False,  #
            DQ_ATOMIC=DQ_ATOMIC,
        )
    elif DKV_VARIANT == 2:
        k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
        v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

        dk, dv = _attn_bwd_dqdkdv_q_k_v(  #
            DQ, dk, dv,  #
            Q, k, v, sm_scale,  #
            DO,  #
            M, D,  #
            stride_tok, stride_d,  #
            H, N_CTX,  #
            BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
            start_n, start_m, num_steps,  #
            MASK=False,  #
            DQ_ATOMIC=DQ_ATOMIC,
        )
    else:
        kT = tl.load(K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d)
        vT = tl.load(V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d)

        dk, dv = _attn_bwd_dqdkdv_qT_kT_vT(  #
            DQ, dk, dv,  #
            Q, kT, vT, sm_scale,  #
            DO,  #
            M, D,  #
            stride_tok, stride_d,  #
            H, N_CTX,  #
            BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
            start_n, start_m, num_steps,  #
            MASK=False,  #
            DQ_ATOMIC=DQ_ATOMIC,
        )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    if not DQ_ATOMIC:
        start_m = pid * BLOCK_M2
        end_n = start_m + BLOCK_M2

        MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
        offs_m = start_m + tl.arange(0, BLOCK_M2)

        q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
        dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
        do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

        m = tl.load(M + offs_m)

        num_steps = N_CTX // BLOCK_N2
        if DQ_VARIANT == 0:
            m = m[:, None]
            dq = _attn_bwd_dq(dq, q, K, V,  #
                            do, m, D,  #
                            stride_tok, stride_d,  #
                            H, N_CTX,  #
                            BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
                            start_m, 0, num_steps,  #
                            MASK=False  #
                            )
        else:
            m = m[None, :]
            dq = _attn_bwd_dq_qT(dq, q, K, V,  #
                            do, m, D,  #
                            stride_tok, stride_d,  #
                            H, N_CTX,  #
                            BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
                            start_m, 0, num_steps,  #
                            MASK=False  #
                            )
        # Write back dQ.
        dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
        dq *= LN2
        tl.store(dq_ptrs, dq)

@triton.jit
@tritonstd.mark_triton_compilable(inline_run_env_fn=partial(_attn_fwd_kernel_test_fn, is_fwd=False),
    real_kwargs={"N_CTX": 30720, "HEAD_DIM": 128, "H": 16, "dtype": torch.float16},
    raw_fn=_torch_bench_fn_bwd,
)
def _attn_bwd_dq_only(Q, K, V, sm_scale,  #
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
              HEAD_DIM: tl.constexpr,
              IS_CAUSAL: tl.constexpr):
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
    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[None, :]

    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    # num_steps = BLOCK_M2 // MASK_BLOCK_N2
    # dq = _attn_bwd_dq(dq, q, K, V,  #
    #                   do, m, D,  #
    #                   stride_tok, stride_d,  #
    #                   H, N_CTX,  #
    #                   BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,  #
    #                   start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,  #
    #                   MASK=True  #
    #                   )
    # end_n -= num_steps * MASK_BLOCK_N2
    # stage 2
    num_steps = N_CTX // BLOCK_N2
    dq = _attn_bwd_dq_qT(dq, q, K, V,  #
                      do, m, D,  #
                      stride_tok, stride_d,  #
                      H, N_CTX,  #
                      BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
                      start_m, 0, num_steps,  #
                      MASK=False  #
                      )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


def test_attn_fwd():
    runner = tritonstd.parse_triton_compilable_to_runner(_attn_fwd)
    # asyncio.run(runner.validate_kernel_by_test_data(_attn_fwd, run_triton=True))
    asyncio.run(runner.bench_kernel_in_triton_process(_attn_fwd, run_cnt=10))

def test_attn_fwd_tma():
    runner = tritonstd.parse_triton_compilable_to_runner(_attn_fwd_tma)
    # asyncio.run(runner.validate_kernel_by_test_data(_attn_fwd, run_triton=True))
    asyncio.run(runner.bench_kernel_in_triton_process(_attn_fwd_tma, run_cnt=10))

def test_attn_bwd():
    runner = tritonstd.parse_triton_compilable_to_runner(_attn_bwd)
    asyncio.run(runner.validate_kernel_by_test_data(_attn_bwd))
    for DKV_VARIANT in [3, 2, 1, 0]:
        for DQ_VARIANT in [0, 1]:
            print(f"Testing DKV_VARIANT={DKV_VARIANT}, DQ_VARIANT={DQ_VARIANT}")
            asyncio.run(runner.bench_kernel_in_triton_process(_attn_bwd, override_kwargs={
                "DKV_VARIANT": DKV_VARIANT,
                "DQ_VARIANT": DQ_VARIANT,
            }))

    # runner = tritonstd.parse_triton_compilable_to_runner(_attn_bwd_dq_only)
    # asyncio.run(runner.validate_kernel_by_test_data(_attn_bwd_dq_only))
    # asyncio.run(runner.bench_kernel_in_triton_process(_attn_bwd_dq_only,))

def _main():
    # test_attn_fwd()
    test_attn_fwd_tma()
    # test_attn_bwd()

if __name__ == "__main__":
    _main()
