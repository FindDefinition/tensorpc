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
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.nn.attention.flex_attention import BlockMask
flex_attention = torch.compile(flex_attention, dynamic=True)
torch._dynamo.config.recompile_limit = 10000
# np.seterr(all='raise')
def _attn_fwd_grid(META, q_shape):
    res = (triton.cdiv(q_shape[2], META["BLOCK_M"]), q_shape[0] * q_shape[1], 1)
    print("FWD GRID", res)
    return res

def _pad_seq_tensor(q: np.ndarray, pad_length: int) -> np.ndarray:
    q_np_padded = np.concatenate(
        [q, np.zeros((q.shape[0], q.shape[1], pad_length, q.shape[3]), dtype=q.dtype)],
        axis=-2 
    )
    return q_np_padded

def _prepare_causal_attn_mask(
        device: torch.device | str, total_length: int
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        def attention_mask(b, h, q_idx, kv_idx):
            return (q_idx) >= (kv_idx)
        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length,
                                       KV_LEN=total_length, device=device)
        return block_mask


def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, total_length: int, block_causal: int
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        def attention_mask(b, h, q_idx, kv_idx):
            #return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            # return (q_idx == kv_idx) | (kv_idx < ends[q_idx])
            # print((q_idx // block_causal) >= (kv_idx // block_causal))
            return (q_idx // block_causal) >= (kv_idx // block_causal)

            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length,
                                       KV_LEN=total_length, _compile=True, device=device)
        return block_mask

def _attn_fwd_kernel_test_fn(N_CTX: int = 256, HEAD_DIM: int = 64, H = 1, dtype = torch.float32, 
        is_fwd: bool = True, head_first: bool = True,
         Q_TRANSPOSED: bool = False, 
        KV_TRANSPOSED: bool = True, 
        SCORE_TRANSPOSED: bool = False, 
        DQ_TRANSPOSED: bool = True,
        PARALLEL_DQ: bool = True,
        is_tma: bool = False,
        block_causal: Optional[int] = None) -> pfl.PFLInlineRunEnv:
    # TODO triton code don't support BLOCK_M < BLOCK_N
    BATCH = 1
    BLOCK_M = 128
    BLOCK_N = 64
    is_causal = False or block_causal is not None
    stage = 3 if is_causal else 1
    sm_scale = 0.5
    M = torch.empty((BATCH, H, N_CTX), dtype=torch.float32)
    M_np = M.detach().numpy()
    block_mask = None 
    if block_causal is not None:
        block_mask = _prepare_blockwise_causal_attn_mask(
            device="cuda", total_length=N_CTX, block_causal=block_causal
        )
        # print(block_mask.)
    elif is_causal:
        block_mask = _prepare_causal_attn_mask(
            device="cuda", total_length=N_CTX
        )
    torch.manual_seed(20)

    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_()
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_()
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_()
    dref = torch.randn_like(q)
    lse_ref = None
    if block_causal is not None:
        # print(block_mask)
        ko = {
            "USE_TMA": False,
            "num_stages": 1,
        }
        # ko = None
        ref, lse_ref = flex_attention(q, k, v, block_mask=block_mask, scale=sm_scale, kernel_options=ko, return_lse=True)
        ref = ref.contiguous()

    else:
        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=is_causal, scale=sm_scale
        )
        ref = ref.contiguous()
    dref = dref.cpu()
    ref = ref.cpu() 

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
    if not head_first:
        q_np = np.ascontiguousarray(np.moveaxis(q_np, 1, 2))
        k_np = np.ascontiguousarray(np.moveaxis(k_np, 1, 2))
        v_np = np.ascontiguousarray(np.moveaxis(v_np, 1, 2))
        dq_np = np.ascontiguousarray(np.moveaxis(dq_np, 1, 2))
        dk_np = np.ascontiguousarray(np.moveaxis(dk_np, 1, 2))
        dv_np = np.ascontiguousarray(np.moveaxis(dv_np, 1, 2))
        o_np = np.ascontiguousarray(np.moveaxis(o_np, 1, 2))
        ref_np = np.ascontiguousarray(np.moveaxis(ref_np, 1, 2))
    IS_DIVISIBLE = N_CTX % 128 == 0
    N_CTX_PAD_LENGTH = triton.cdiv(N_CTX, 128) * 128 - N_CTX
    OUT_USE_TMA = True
    fwd_grid = (triton.cdiv(N_CTX, BLOCK_M), H * BATCH, 1)
    if is_tma:
        if head_first:
            y_dim = q.shape[0] * q.shape[1] * q.shape[2]

            desc_q = tritonstd.HostTensorDescriptor(q_np.reshape(y_dim, -1), block_shape=[BLOCK_M, HEAD_DIM])
            desc_v = tritonstd.HostTensorDescriptor(v_np.reshape(y_dim, -1), block_shape=[BLOCK_N, HEAD_DIM])
            desc_k = tritonstd.HostTensorDescriptor(k_np.reshape(y_dim, -1), block_shape=[BLOCK_N, HEAD_DIM])
            if OUT_USE_TMA:
                if IS_DIVISIBLE:
                    desc_o = tritonstd.HostTensorDescriptor(o_np.reshape(y_dim, -1), block_shape=[BLOCK_M, HEAD_DIM])
                else:
                    desc_o = tritonstd.HostTensorDescriptor(o_np.reshape(q.shape[0] * q.shape[1], q.shape[2], -1), block_shape=[1, BLOCK_M, HEAD_DIM])
            else:
                desc_o = o_np
        else:
            desc_q = tritonstd.HostTensorDescriptor(q_np.reshape(BATCH * N_CTX, H * HEAD_DIM), block_shape=[BLOCK_M, HEAD_DIM])
            desc_v = tritonstd.HostTensorDescriptor(v_np.reshape(BATCH * N_CTX, H * HEAD_DIM), block_shape=[BLOCK_N, HEAD_DIM])
            desc_k = tritonstd.HostTensorDescriptor(k_np.reshape(BATCH * N_CTX, H * HEAD_DIM), block_shape=[BLOCK_N, HEAD_DIM])
            if OUT_USE_TMA:
                desc_o = tritonstd.HostTensorDescriptor(o_np.reshape(BATCH * N_CTX, H * HEAD_DIM), block_shape=[BLOCK_M, HEAD_DIM])
            else:
                desc_o = o_np
        fwd_kwargs: dict[str, Any] = {
            "M": M_np,
            "Z": BATCH, 
            "H": H,
            "desc_q": desc_q,
            "desc_v": desc_v,
            "desc_k": desc_k,
            "desc_o": desc_o,
            "MASK_VARIANT": "causal" if is_causal else "full",
            "N_CTX": N_CTX,
            "HEAD_DIM": HEAD_DIM,
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "warp_specialize": False,
            "FP8_OUTPUT": False,
            "SM_SCALE": sm_scale,
            "IS_DIVISIBLE": IS_DIVISIBLE,
            "OUT_USE_TMA": OUT_USE_TMA,
        }
    else:
        assert head_first
        fwd_kwargs: dict[str, Any] = {
            "M": M_np,
            "Z": BATCH, 
            "H": H,
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
            "MASK_VARIANT": "causal" if is_causal else "full",
            "N_CTX": N_CTX,
            "HEAD_DIM": HEAD_DIM,
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "SM_SCALE": sm_scale,
            "IS_DIVISIBLE": IS_DIVISIBLE,
        }
    if block_causal is not None:
        fwd_kwargs["block_causal"] = block_causal
    if is_fwd:
        if is_tma:  
            if OUT_USE_TMA:
                if head_first:
                    if IS_DIVISIBLE:

                        ref_kwargs = {
                            "desc_o": ref_np.reshape(BATCH * N_CTX * H, -1)
                        }
                    else:
                        ref_kwargs = {
                            "desc_o": ref_np.reshape(BATCH * H, N_CTX, -1)
                        }
                else:
                    ref_kwargs = {
                        "desc_o": ref_np.reshape(BATCH * N_CTX, H * HEAD_DIM)
                    }

            else:
                ref_kwargs = {
                    "desc_o": ref_np
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
        sim_info = tritonstd.TritonSimInfo(fwd_grid, ref_kwargs, vis_layout=vis_layout, grid_size_for_triton=partial(_attn_fwd_grid, q_shape=q.shape),
            kwargs_for_raw={
                "block_mask": block_mask,
            })
        return pfl.PFLInlineRunEnv(fwd_kwargs, userdata=sim_info)
    else:
        delta = torch.empty_like(M)
        delta = (ref * dref).sum(dim=-1)

        delta_np = delta.detach().numpy()
        # we have to run fwd kernel here to get correct M.
        # when we run real triton kernel, we shouldn't run slow fwd kernel in simulation.
        
        # if N_CTX <= 1000:
        # if block_causal is not None:
        q_fl = q.cuda()
        k_fl = k.cuda()
        v_fl = v.cuda()
        _, lse = flex_attention(
            q_fl, k_fl, v_fl, scale=sm_scale, block_mask=block_mask,
            return_lse=True, kernel_options={
                "USE_TMA": False,
                "num_stages": 1,
            }
        )
        M_np = lse.detach().cpu().numpy()[..., :N_CTX]

        # print(lse)
        # else:
        #     from xformers.ops import fmha
        #     q_xf = q.transpose(1, 2).contiguous().cuda()
        #     k_xf = k.transpose(1, 2).contiguous().cuda()
        #     v_xf = v.transpose(1, 2).contiguous().cuda()
        #     _, lse_xf = fmha.memory_efficient_attention_forward_requires_grad(q_xf, k_xf, v_xf, scale=sm_scale, attn_bias=fmha.LowerTriangularMask() if is_causal else None)
        #     M_np = lse_xf.detach().cpu().numpy()[..., :N_CTX]
        # # print(lse)
        # # print(M_np)

        M_np *= np.log2(np.e)
            # if is_tma:
            #     _attn_fwd_var = _attn_fwd_tma_v2
            # else:
            #     _attn_fwd_var = _attn_fwd
            # runner = tritonstd.parse_triton_compilable_to_runner(_attn_fwd_var, module_code_getter=lambda x: Path(__file__).read_text())
            # global_mem_fwd = tritonstd.create_global_mem_from_kwargs(fwd_kwargs)
            # runner.run_kernel_in_executor(
            #     _attn_fwd_var.fn, grid_size=fwd_grid, global_mem=global_mem_fwd, **fwd_kwargs)
            # M_np = global_mem_fwd.memory_blocks["M"].get_data_view_checked().copy()
            # Out_np = global_mem_fwd.memory_blocks["Out"].get_data_view_checked().copy()

            # print("XFORMERS")
            # print(lse, )
            # print(M_np / np.log2(np.e))
        BLK_SLICE_FACTOR = 1
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        # assert N_CTX % PRE_BLOCK == 0
        # pre_grid = (N_CTX // PRE_BLOCK, BATCH * H)
        DQ_ATOMIC = False # faster in Hopper, slower in Ampere
        if DQ_ATOMIC:
            BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        else:
            if HEAD_DIM == 64:
                BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 64, 64, 32
            else:
                BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 64, 64, 32
        if DQ_ATOMIC:
            PARALLEL_DQ = False
        pdq = 1
        if PARALLEL_DQ and not DQ_ATOMIC:
            pdq = 2
        
        grid = (triton.cdiv(N_CTX, BLOCK_N1) * pdq, BATCH, H)
        if not head_first:
            grid = (H, triton.cdiv(N_CTX, BLOCK_N1) * pdq, BATCH)
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
            # always pad dq for atomic
            dq_np_out = _pad_seq_tensor(dq_np_out, N_CTX_PAD_LENGTH)

        else:
            dq_np_out = np.empty_like(dq_np)
        dk_np_out = np.empty_like(dk_np)
        dv_np_out = np.empty_like(dv_np)
        dk_np_out[:] = -1
        dv_np_out[:] = -1
        M_np_padded = np.concatenate(
            [M_np, np.zeros((M_np.shape[0], M_np.shape[1], N_CTX_PAD_LENGTH), dtype=M_np.dtype)],
            axis=-1
        )
        delta_np_padded = np.concatenate(
            [delta_np, np.zeros((delta_np.shape[0], delta_np.shape[1], N_CTX_PAD_LENGTH), dtype=delta_np.dtype)],
            axis=-1
        )
        PADDED_QKV = True 
        PADDED_OUTPUTS = False
        if PADDED_QKV and N_CTX_PAD_LENGTH > 0:
            q_np_padded = _pad_seq_tensor(q_np, N_CTX_PAD_LENGTH)
            k_np_scaled_padded = _pad_seq_tensor(k_np_scaled, N_CTX_PAD_LENGTH)
            v_np_padded = _pad_seq_tensor(v_np, N_CTX_PAD_LENGTH)
            do_padded = _pad_seq_tensor(dref_np, N_CTX_PAD_LENGTH)
        else:
            q_np_padded = q_np
            k_np_scaled_padded = k_np_scaled
            v_np_padded = v_np
            do_padded = dref_np

        if PADDED_OUTPUTS and N_CTX_PAD_LENGTH > 0:
            if not DQ_ATOMIC:
                dk_np_out = _pad_seq_tensor(dk_np_out, N_CTX_PAD_LENGTH)
            dv_np_out = _pad_seq_tensor(dv_np_out, N_CTX_PAD_LENGTH)
            # dv_np = _pad_seq_tensor(dv_np, N_CTX_PAD_LENGTH)
            # dk_np = _pad_seq_tensor(dk_np, N_CTX_PAD_LENGTH)
            # dq_np = _pad_seq_tensor(dq_np, N_CTX_PAD_LENGTH)

        stride_z_pad = q_np_padded.strides[0] // q_np_padded.itemsize 
        stride_h_pad = q_np_padded.strides[1] // q_np_padded.itemsize if head_first else q_np_padded.strides[2] // q_np_padded.itemsize
        stride_tok = q_np_padded.strides[2] // q_np_padded.itemsize if head_first else q_np_padded.strides[1] // q_np_padded.itemsize
        stride_d = q_np_padded.strides[3] // q_np_padded.itemsize
        stride_z = q_np.strides[0] // q_np.itemsize
        stride_h = q_np.strides[1] // q_np.itemsize if head_first else q_np.strides[2] // q_np.itemsize
        test_kwargs: dict[str, Any] = {
            "SM_SCALE": sm_scale,
            "Z": q_np_padded.shape[0],
            "DK": dk_np_out,
            "DV": dv_np_out,
            "M": M_np_padded,
            "D": delta_np_padded,
            "stride_z_pad": stride_z_pad,
            "stride_h_pad": stride_h_pad,

            "stride_z": stride_z,
            "stride_h": stride_h,
            "stride_tok": stride_tok,
            "stride_d": stride_d,
            "H": H,
            "N_CTX": N_CTX,
            "N_CTX_PADDED": triton.cdiv(N_CTX, 128) * 128,
            "HEAD_DIM": HEAD_DIM,
            "BLOCK_M1": BLOCK_M1,
            "BLOCK_N1": BLOCK_N1,
            "BLOCK_M2": BLOCK_M2,
            "BLOCK_N2": BLOCK_N2,
            "BLK_SLICE_FACTOR": BLK_SLICE_FACTOR,
            "num_warps": 8 if DQ_ATOMIC else 4,
            "num_stages": 3 if DQ_ATOMIC else 3,
            "MASK_VARIANT": "causal" if is_causal else "full",
            "is_head_first": head_first,
            "maxnreg": 256,
            "DQ_ATOMIC": DQ_ATOMIC,
            "Q_TRANSPOSED": Q_TRANSPOSED,
            "KV_TRANSPOSED": KV_TRANSPOSED,
            "SCORE_TRANSPOSED": SCORE_TRANSPOSED,
            "DQ_TRANSPOSED": DQ_TRANSPOSED,
            "IS_DIVISIBLE": IS_DIVISIBLE,
            "PARALLEL_DQ": PARALLEL_DQ,
            "PADDED_QKV": PADDED_QKV,
            "PADDED_OUTPUTS": PADDED_OUTPUTS,
            "IS_TMA": is_tma,
        }
        if block_causal is not None:
            test_kwargs.update({
                "block_causal": block_causal,
            })
        if is_tma: 
            test_kwargs.update({
                "Q": tritonstd.HostTensorDescriptor(q_np_padded.reshape(-1, HEAD_DIM), block_shape=[BLOCK_M1, HEAD_DIM]),
                "K": tritonstd.HostTensorDescriptor(k_np_scaled_padded.reshape(-1, HEAD_DIM), block_shape=[BLOCK_N1, HEAD_DIM]),
                "V": tritonstd.HostTensorDescriptor(v_np_padded.reshape(-1, HEAD_DIM), block_shape=[BLOCK_N1, HEAD_DIM]),
                # "K": k_np_scaled_padded,
                # "V": v_np_padded,

                "DO": tritonstd.HostTensorDescriptor(do_padded.reshape(-1, HEAD_DIM), block_shape=[BLOCK_M1, HEAD_DIM]),
                "DQ": tritonstd.HostTensorDescriptor(dq_np_out.reshape(-1, HEAD_DIM), block_shape=[BLOCK_M1, HEAD_DIM]),
            })
        else:
            test_kwargs.update({
                "Q": q_np_padded,
                "K": k_np_scaled_padded,
                "V": v_np_padded,
                "DO": do_padded,
                "DQ": dq_np_out,
            })
        ref_kwargs = {
            "DV": dv_np,
            "DK": dk_np,
            "DQ": dq_np,
        }
        def postprocess_padded(key: str, data: np.ndarray):
            if is_tma:
                return data.reshape(BATCH, H, -1, HEAD_DIM)[:, :, :N_CTX, :]
            else:
                return data[:, :, :N_CTX, :] 
        return pfl.PFLInlineRunEnv(test_kwargs, userdata=tritonstd.TritonSimInfo(grid, ref_kwargs, 
            grid_size_for_triton=lambda META: grid,
            postprocess_fn=postprocess_padded, kwargs_for_raw={
                "block_mask": block_mask,
            }))

configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64]\
    for s in ([3, 4, 5, 6])\
    for w in [4, 8]\
]
# configs = [
#     triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
#     for BM in [128]\
#     for BN in [64]\
#     for s in ([3])\
#     for w in [8]\
# ]

def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M < BLOCK_N:
        return False
    if BLOCK_M * BLOCK_N < 128 * 64 and conf.num_warps == 8:
        return False
    return True

if tritonstd.TRITON_VERSION_TUPLE <= (3, 3):
    @triton.jit
    @tritonstd.mark_triton_compilable(is_template=True)
    def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
        if isinstance(desc_or_ptr, tl._experimental_tensor_descriptor):
            return desc_or_ptr
        else:
            return tl._experimental_make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)

else:
    @triton.jit
    @tritonstd.mark_triton_compilable(is_template=True)
    def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
        if isinstance(desc_or_ptr, tl.tensor_descriptor):
            return desc_or_ptr
        else:
            return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


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
    is_causal = kwargs["MASK_VARIANT"] == "causal"

    with tritonstd.measure_duration_torch() as dur:
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):

            res = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=is_causal, scale=kwargs["SM_SCALE"]
            )
    return dur.val

def _flex_bench_fn_fwd(kwargs):
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
    is_causal = kwargs["MASK_VARIANT"] == "causal"
    block_mask = None 
    if kwargs["block_causal"] > 1:
        block_mask = _prepare_blockwise_causal_attn_mask(
            device="cuda", total_length=kwargs["N_CTX"], block_causal=kwargs["block_causal"]
        )
        # print(block_mask)
    with tritonstd.measure_duration_torch() as dur:
        res = flex_attention(
            q, k, v, block_mask=block_mask, scale=kwargs["SM_SCALE"],
            kernel_options={
                "USE_TMA": False,
                "num_stages": 2,

            }
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
    is_causal = kwargs["MASK_VARIANT"] == "causal"
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:

        with tritonstd.measure_duration_torch() as dur:
            res = fmha.memory_efficient_attention(q, k, v, fmha.LowerTriangularMask() if is_causal else None)
    prof.export_chrome_trace("/root/fwd_trace_xformers.json")

    return dur.val

def _torch_bench_fn_bwd(kwargs):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q = kwargs["Q"].detach().clone()
    k = kwargs["K"].detach().clone()
    v = kwargs["V"].detach().clone()
    if kwargs["IS_TMA"]:
        N_CTX = kwargs["N_CTX"]
        Z = kwargs["Z"]
        H = kwargs["H"]
        HEAD_DIM = kwargs["HEAD_DIM"]
        # may padded, remove pad data
        q = q.view(Z, H, -1, HEAD_DIM)[:, :, :N_CTX]
        k = k.view(Z, H, -1, HEAD_DIM)[:, :, :N_CTX]
        v = v.view(Z, H, -1, HEAD_DIM)[:, :, :N_CTX]
    if not kwargs["is_head_first"]:
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):

        res = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=kwargs["MASK_VARIANT"] == "causal", scale=kwargs["SM_SCALE"]
        )
        dres = torch.empty_like(res)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            with tritonstd.measure_duration_torch() as dur:

                res.backward(dres)
    prof.export_chrome_trace("/root/bwd_trace.json")
    return dur.val

def _torch_bench_fn_bwd_bc(kwargs):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q = kwargs["Q"].detach().clone()
    k = kwargs["K"].detach().clone()
    v = kwargs["V"].detach().clone()
    # if kwargs["IS_TMA"]:
    N_CTX = kwargs["N_CTX"]
    Z = kwargs["Z"]
    H = kwargs["H"]
    HEAD_DIM = kwargs["HEAD_DIM"]
    # may padded, remove pad data
    q = q.view(Z, H, -1, HEAD_DIM)[:, :, :N_CTX].contiguous()
    k = k.view(Z, H, -1, HEAD_DIM)[:, :, :N_CTX].contiguous()
    v = v.view(Z, H, -1, HEAD_DIM)[:, :, :N_CTX].contiguous()

    # block_mask = None 
    # if kwargs["block_causal"] > 1:
    #     block_mask = _prepare_blockwise_causal_attn_mask(
    #         device="cuda", total_length=kwargs["N_CTX"], block_causal=kwargs["block_causal"]
    #     )
        # print(block_mask)
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    # res = flex_attention(
    #     q, k, v, block_mask=kwargs["block_mask"], scale=kwargs["SM_SCALE"],
    #     kernel_options={
    #         "USE_TMA": False,
    #         "num_stages": 2,
    #     }
    # )
    # dres = torch.empty_like(res)
    with tritonstd.measure_duration_torch() as dur:
        pass
        # res.backward(dres)
    return dur.val


def _torch_bench_fn_bwd_xformer(kwargs):
    from xformers.ops import fmha
    q = kwargs["Q"].detach().clone()
    k = kwargs["K"].detach().clone()
    v = kwargs["V"].detach().clone()
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    if kwargs["IS_TMA"]:
        N_CTX = kwargs["N_CTX"]
        Z = kwargs["Z"]
        H = kwargs["H"]
        HEAD_DIM = kwargs["HEAD_DIM"]
        # may padded, remove pad data
        q = q.view(Z, H, -1, HEAD_DIM)[:, :, :N_CTX]
        k = k.view(Z, H, -1, HEAD_DIM)[:, :, :N_CTX]
        v = v.view(Z, H, -1, HEAD_DIM)[:, :, :N_CTX]

    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    is_causal = kwargs["MASK_VARIANT"] == "causal"
    res = fmha.memory_efficient_attention(q, k, v, fmha.LowerTriangularMask() if is_causal else None)
    dres = torch.empty_like(res)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        with tritonstd.measure_duration_torch() as dur:
            res.backward(dres)
    prof.export_chrome_trace("/root/bwd_trace_xformers.json")
    return dur.val

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_fwd_inner_single(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    offs_m, offs_n, 
                    Q_LEN, KV_LEN,
                    SM_SCALE: tl.constexpr,  #
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, 
                    HEAD_DIM: tl.constexpr, 
                    IS_DIVISIBLE: tl.constexpr,
                    CHECK_BLOCK_BOUNDARY: tl.constexpr,
                    MASK_VARIANT: tl.constexpr):
    # -- compute qk ----
    # offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_DIVISIBLE:
        k = tl.load(K_block_ptr)
    else:
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option = "zero")
    qk = tl.dot(q, k) * SM_SCALE
    if CHECK_BLOCK_BOUNDARY:
        # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
        qk = tl.where(offs_n < KV_LEN, qk, float("-inf"))
    if MASK_VARIANT == "causal":
        # only support causal currently
        mask = offs_m[:, None] >= (offs_n[None, :])
        qk = qk + tl.where(mask, 0.0, float("-inf"))
        # stable softmax: calc max of each row
        # print("P", MASK_VARIANT, qk)

    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    # -- update m_i and l_i
    alpha = tl.math.exp2(m_i - m_ij)

    p = tl.math.exp2(qk - m_ij[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    # -- update output accumulator --
    acc = acc * alpha[:, None]
    # update acc
    if IS_DIVISIBLE:
        v = tl.load(V_block_ptr)
    else:
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option = "zero")
    acc = tl.dot(p.to(q.dtype), v, acc)
    # update m_i and l_i
    m_i = m_ij
    return acc, l_i, m_i

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_fwd_inner_tma_single(acc, l_i, m_i, q,  #
                    k_desc, v_desc,  #
                    offs_m, offs_n, 
                    offset_y,
                    Q_LEN, KV_LEN,
                    SM_SCALE,  #
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, 
                    HEAD_DIM: tl.constexpr, 
                    IS_DIVISIBLE: tl.constexpr,
                    CHECK_BLOCK_BOUNDARY: tl.constexpr,
                    MASK_VARIANT: tl.constexpr):
    # -- compute qk ----
    # offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    kT = k_desc.load([offset_y, 0]).T
    qk = tl.dot(q, kT)
    if CHECK_BLOCK_BOUNDARY:
        # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
        qk = tl.where(offs_n < KV_LEN, qk, float("-inf"))
    if MASK_VARIANT == "causal":
        # only support causal currently
        mask = offs_m[:, None] >= (offs_n[None, :])
        qk = qk + tl.where(mask, 0.0, float("-inf"))

    m_ij = tl.maximum(m_i, tl.max(qk, 1) * SM_SCALE)
    # -- update m_i and l_i
    p = tl.math.exp2(qk * SM_SCALE - m_ij[:, None])
    alpha = tl.math.exp2(m_i - m_ij)
    l_ij = tl.sum(p, 1)


    # -- update output accumulator --
    acc = acc * alpha[:, None]
    # update acc
    v = v_desc.load([offset_y, 0])

    # if IS_DIVISIBLE:
    #     v = tl.load(V_block_ptr)
    # else:
    #     v = tl.load(V_block_ptr, boundary_check=(0,), padding_option = "zero")
    acc = tl.dot(p.to(q.dtype), v, acc)
    # update m_i and l_i
    l_i = l_i * alpha + l_ij
    m_i = m_ij
    return acc, l_i, m_i

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_fwd_inner_tma_single_blockcausal(acc, l_i, m_i, q,  #
                    k_desc, v_desc,  #
                    offs_m, offs_n, 
                    offset_y,
                    Q_LEN, KV_LEN,
                    block_causal,
                    SM_SCALE: tl.constexpr,  #
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, 
                    HEAD_DIM: tl.constexpr, 
                    IS_DIVISIBLE: tl.constexpr,
                    CHECK_BLOCK_BOUNDARY: tl.constexpr,
                    MASK_VARIANT: tl.constexpr):
    # -- compute qk ----
    # offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    kT = k_desc.load([offset_y, 0]).T
    qk = tl.dot(q, kT) * SM_SCALE
    if CHECK_BLOCK_BOUNDARY:
        # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
        qk = tl.where(offs_n < KV_LEN, qk, float("-inf"))
    if MASK_VARIANT == "causal":
        # only support causal currently
        mask = (offs_m[:, None] // block_causal) >= (offs_n[None, :] // block_causal)
        qk = qk + tl.where(mask, 0.0, float("-inf"))

    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    masked_out_rows = (m_ij == float("-inf"))
    m_ij = tl.where(masked_out_rows, 0.0, m_ij)
    # -- update m_i and l_i
    p = tl.math.exp2(qk - m_ij[:, None])
    alpha = tl.math.exp2(m_i - m_ij)
    l_ij = tl.sum(p, 1)


    # -- update output accumulator --
    acc = acc * alpha[:, None]
    # update acc
    v = v_desc.load([offset_y, 0])

    # if IS_DIVISIBLE:
    #     v = tl.load(V_block_ptr)
    # else:
    #     v = tl.load(V_block_ptr, boundary_check=(0,), padding_option = "zero")
    acc = tl.dot(p.to(q.dtype), v, acc)
    # update m_i and l_i
    l_i = l_i * alpha + l_ij
    m_i = m_ij
    return acc, l_i, m_i


@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_fwd_inner_tma_bshd_single(acc, l_i, m_i, q,  #
                    k_desc, v_desc,  #
                    offs_m, offs_n, 
                    offset_y, offset_z,
                    Q_LEN, KV_LEN,
                    SM_SCALE: tl.constexpr,  #
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, 
                    HEAD_DIM: tl.constexpr, 
                    IS_DIVISIBLE: tl.constexpr,
                    CHECK_BLOCK_BOUNDARY: tl.constexpr,
                    MASK_VARIANT: tl.constexpr):
    # -- compute qk ----
    # offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    kT = k_desc.load([offset_y, offset_z]).T
    qk = tl.dot(q, kT)
    if CHECK_BLOCK_BOUNDARY:
        # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
        qk = tl.where(offs_n < KV_LEN, qk, float("-inf"))
    if MASK_VARIANT == "causal":
        # only support causal currently
        mask = offs_m[:, None] >= (offs_n[None, :])
        qk = qk + tl.where(mask, 0.0, float("-inf"))

    m_ij = tl.maximum(m_i, tl.max(qk, 1) * SM_SCALE)
    # -- update m_i and l_i
    p = tl.math.exp2(qk * SM_SCALE - m_ij[:, None])
    alpha = tl.math.exp2(m_i - m_ij)
    l_ij = tl.sum(p, 1)


    # -- update output accumulator --
    acc = acc * alpha[:, None]
    # update acc
    v = v_desc.load([offset_y, offset_z])

    # if IS_DIVISIBLE:
    #     v = tl.load(V_block_ptr)
    # else:
    #     v = tl.load(V_block_ptr, boundary_check=(0,), padding_option = "zero")
    acc = tl.dot(p.to(q.dtype), v, acc)
    # update m_i and l_i
    l_i = l_i * alpha + l_ij
    m_i = m_ij
    return acc, l_i, m_i

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    offs_m, offs_n,  #
                    start, end,
                    Q_LEN, KV_LEN,
                    SM_SCALE: tl.constexpr,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    IS_DIVISIBLE: tl.constexpr,
                    MASK_VARIANT: tl.constexpr):
    # range of values handled by this stage
    # loop over k, v and update accumulator
    offs_n = offs_n + start * BLOCK_N
    for start_n in range(start, end):
        # start_n = tl.multiple_of(start_n, BLOCK_N)
        if IS_DIVISIBLE:
            acc, l_i, m_i = _attn_fwd_inner_single(
                acc, l_i, m_i, q, K_block_ptr, V_block_ptr, 
                offs_m, offs_n, Q_LEN, KV_LEN, 
                SM_SCALE, BLOCK_M, BLOCK_N, HEAD_DIM, 
                IS_DIVISIBLE, CHECK_BLOCK_BOUNDARY=False, MASK_VARIANT=MASK_VARIANT
            )
        else:
            acc, l_i, m_i = _attn_fwd_inner_single(
                acc, l_i, m_i, q, K_block_ptr, V_block_ptr, 
                offs_m, offs_n, Q_LEN, KV_LEN, 
                SM_SCALE, BLOCK_M, BLOCK_N, HEAD_DIM, 
                IS_DIVISIBLE, CHECK_BLOCK_BOUNDARY=True, MASK_VARIANT=MASK_VARIANT
            )

        offs_n += BLOCK_N

        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_fwd_inner_tma_v2(acc, l_i, m_i, q,  #
                    k_desc, v_desc,  #
                    offs_m, offs_n,  #
                    offset_y,
                    start, end,
                    Q_LEN, KV_LEN, H, 
                    SM_SCALE,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    IS_DIVISIBLE: tl.constexpr,
                    MASK_VARIANT: tl.constexpr):
    # range of values handled by this stage
    # loop over k, v and update accumulator
    offs_n = offs_n + start * BLOCK_N
    offset_y = offset_y + start * BLOCK_N
    for start_n in range(start, end):
        # start_n = tl.multiple_of(start_n, BLOCK_N)
        if IS_DIVISIBLE:
            acc, l_i, m_i = _attn_fwd_inner_tma_single(
                acc, l_i, m_i, q, k_desc, v_desc, 
                offs_m, offs_n, offset_y, Q_LEN, KV_LEN, 
                SM_SCALE, BLOCK_M, BLOCK_N, HEAD_DIM, 
                IS_DIVISIBLE, CHECK_BLOCK_BOUNDARY=False, MASK_VARIANT=MASK_VARIANT
            )
        else:
            acc, l_i, m_i = _attn_fwd_inner_tma_single(
                acc, l_i, m_i, q, k_desc, v_desc, 
                offs_m, offs_n, offset_y, Q_LEN, KV_LEN, 
                SM_SCALE, BLOCK_M, BLOCK_N, HEAD_DIM, 
                IS_DIVISIBLE, CHECK_BLOCK_BOUNDARY=True, MASK_VARIANT=MASK_VARIANT
            )
        if MASK_VARIANT != "full" or not IS_DIVISIBLE:
            offs_n += BLOCK_N
        offset_y += BLOCK_N

    return acc, l_i, m_i

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_fwd_inner_tma_blockcausal_v2(acc, l_i, m_i, q,  #
                    k_desc, v_desc,  #
                    offs_m, offs_n,  #
                    offset_y,
                    start, end,
                    Q_LEN, KV_LEN, H, 
                    block_causal,
                    SM_SCALE,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    IS_DIVISIBLE: tl.constexpr,
                    MASK_VARIANT: tl.constexpr):
    # range of values handled by this stage
    # loop over k, v and update accumulator
    offs_n = offs_n + start * BLOCK_N
    offset_y = offset_y + start * BLOCK_N
    for start_n in range(start, end):
        # start_n = tl.multiple_of(start_n, BLOCK_N)
        if IS_DIVISIBLE:
            acc, l_i, m_i = _attn_fwd_inner_tma_single_blockcausal(
                acc, l_i, m_i, q, k_desc, v_desc, 
                offs_m, offs_n, offset_y, Q_LEN, KV_LEN, block_causal,
                SM_SCALE, BLOCK_M, BLOCK_N, HEAD_DIM, 
                IS_DIVISIBLE, CHECK_BLOCK_BOUNDARY=False, MASK_VARIANT=MASK_VARIANT
            )
        else:
            acc, l_i, m_i = _attn_fwd_inner_tma_single_blockcausal(
                acc, l_i, m_i, q, k_desc, v_desc, 
                offs_m, offs_n, offset_y, Q_LEN, KV_LEN, block_causal,
                SM_SCALE, BLOCK_M, BLOCK_N, HEAD_DIM, 
                IS_DIVISIBLE, CHECK_BLOCK_BOUNDARY=True, MASK_VARIANT=MASK_VARIANT
            )
        if MASK_VARIANT != "full" or not IS_DIVISIBLE:
            offs_n += BLOCK_N
        offset_y += BLOCK_N

    return acc, l_i, m_i


@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_fwd_inner_tma_bshd_v2(acc, l_i, m_i, q,  #
                    k_desc, v_desc,  #
                    offs_m, offs_n,  #
                    offset_y, offset_z,
                    start, end,
                    Q_LEN, KV_LEN, H, 
                    SM_SCALE: tl.constexpr,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    IS_DIVISIBLE: tl.constexpr,
                    MASK_VARIANT: tl.constexpr):
    # range of values handled by this stage
    # loop over k, v and update accumulator
    offs_n = offs_n + start * BLOCK_N
    offset_y = offset_y + start * BLOCK_N

    for start_n in range(start, end):
        # start_n = tl.multiple_of(start_n, BLOCK_N)
        if IS_DIVISIBLE:
            acc, l_i, m_i = _attn_fwd_inner_tma_bshd_single(
                acc, l_i, m_i, q, k_desc, v_desc, 
                offs_m, offs_n, offset_y, offset_z, Q_LEN, KV_LEN, 
                SM_SCALE, BLOCK_M, BLOCK_N, HEAD_DIM, 
                IS_DIVISIBLE, CHECK_BLOCK_BOUNDARY=False, MASK_VARIANT=MASK_VARIANT
            )
        else:
            acc, l_i, m_i = _attn_fwd_inner_tma_bshd_single(
                acc, l_i, m_i, q, k_desc, v_desc, 
                offs_m, offs_n, offset_y, offset_z, Q_LEN, KV_LEN, 
                SM_SCALE, BLOCK_M, BLOCK_N, HEAD_DIM, 
                IS_DIVISIBLE, CHECK_BLOCK_BOUNDARY=True, MASK_VARIANT=MASK_VARIANT
            )
        if MASK_VARIANT != "full" or not IS_DIVISIBLE:
            offs_n += BLOCK_N
        offset_y += BLOCK_N

    return acc, l_i, m_i

@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
@tritonstd.mark_triton_compilable(inline_run_env_fn=_attn_fwd_kernel_test_fn, 
    real_kwargs={"N_CTX": 61440, "HEAD_DIM": 128, "H": 16, "dtype": torch.float16},
    # raw_fn=_xformers_bench_fn_fwd,
)
def _attn_fwd(Q, K, V, M, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, N_CTX,  #
              SM_SCALE: tl.constexpr,
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              MASK_VARIANT: tl.constexpr,
              IS_DIVISIBLE: tl.constexpr,  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    tl.static_assert(BLOCK_M >= BLOCK_N)

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = tl.cast(off_z, tl.int64) * stride_qz + tl.cast(off_h, tl.int64) * stride_qh

    # block pointers
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    QK_SCALE: tl.constexpr = SM_SCALE * 1.44269504  # 1/log(2)
    Q_USE_BLOCK_PTR: tl.constexpr = True
    if Q_USE_BLOCK_PTR:
        # load q: it will stay in SRAM throughout
        # load q: it stays in SRAM throughout the inner loop.
        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        if IS_DIVISIBLE:
            q = tl.load(Q_block_ptr)
        else:
            # boundary check is not free, so we only do it when necessary.
            q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option = "zero")
    else:
        q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + tl.arange(0, HEAD_DIM)[None, :] * stride_qk
        if IS_DIVISIBLE:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    dtype = q.dtype
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    v_order: tl.constexpr = (1, 0)
    BLOCK_M_DIV_N: tl.constexpr = BLOCK_M // BLOCK_N
    total_cnt = tl.cdiv(N_CTX, BLOCK_N)
    if MASK_VARIANT == "causal":
        lo, hi = start_m * BLOCK_M_DIV_N, min((start_m + 1) * BLOCK_M_DIV_N, total_cnt)
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vk, stride_vn),
            offsets=(start_m * (BLOCK_M_DIV_N * BLOCK_N), 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=v_order,
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset,
            shape=(HEAD_DIM, N_CTX),
            strides=(stride_kk, stride_kn),
            offsets=(0, start_m * (BLOCK_M_DIV_N * BLOCK_N)),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(0, 1),
        )
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        offs_m, offs_n, lo, hi,
                                         N_CTX, N_CTX, 
                                        QK_SCALE, BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        IS_DIVISIBLE,
                                        "causal")
    if MASK_VARIANT == "causal":
        lo, hi = 0, start_m * BLOCK_M_DIV_N
    else:
        lo, hi = 0, total_cnt
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
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                    offs_m, offs_n, lo, hi,
                                    N_CTX, N_CTX, 
                                    QK_SCALE, BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    IS_DIVISIBLE,
                                    "full")

    # [Note] Handle fully masked out rows:
    # Li will be the sum(e^(-inf)) == 0.0 for masked out rows, mi will be -inf.
    # We set Li to 1.0 which will result in lse/out = 0.0 | after the log(li) + mi(0.0) step
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    # epilogue
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    m_i += tl.log2(l_i)
    OUT_USE_BLOCK_PTR: tl.constexpr = False
    if OUT_USE_BLOCK_PTR:
        O_block_ptr = tl.make_block_ptr(
            base=Out + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        if IS_DIVISIBLE:
            tl.store(m_ptrs, m_i)
            tl.store(O_block_ptr, acc.to(dtype))
        else:
            tl.store(m_ptrs, m_i, mask=offs_m < N_CTX)
            tl.store(O_block_ptr, acc.to(dtype), boundary_check=(0,))
    else:
        out_ptrs = Out + qvk_offset + offs_m[:, None] * stride_om + tl.arange(0, HEAD_DIM)[None, :]
        if IS_DIVISIBLE:
            tl.store(m_ptrs, m_i)
            tl.store(out_ptrs, acc.to(dtype))
        else:
            tl.store(m_ptrs, m_i, mask=offs_m < N_CTX)
            tl.store(out_ptrs, acc.to(dtype), offs_m[:, None] < N_CTX)



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
    if "desc_o" in nargs:
        if hasattr(nargs["desc_o"], "block_shape"):
            if len(nargs["desc_o"].block_shape) == 3:
                nargs["desc_o"].block_shape = [1, BLOCK_M, HEAD_DIM]
            else:
                nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]


TMA_NUM_STAGES_OPTIONS = [2, 3, 4]

tma_configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w, pre_hook=_host_descriptor_pre_hook) \
    for BM in [64, 128]\
    for BN in [32, 64, 128]\
    for s in TMA_NUM_STAGES_OPTIONS \
    for w in [4, 8]\
]

tma_configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w, pre_hook=_host_descriptor_pre_hook) \
    for BM in [128]\
    for BN in [64]\
    for s in [2] \
    for w in [4]\
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



@triton.autotune(configs=list(filter(tma_keep, tma_configs)), key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
                 prune_configs_by={'early_config_prune': tma_prune_invalid_configs})
@triton.jit
@tritonstd.mark_triton_compilable(inline_run_env_fn=partial(_attn_fwd_kernel_test_fn, is_tma=True),
    real_kwargs={"N_CTX": 30720, "HEAD_DIM": 128, "H": 16, "dtype": torch.float16},
    raw_fn=_xformers_bench_fn_fwd,
)
def _attn_fwd_tma_v2(M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
              SM_SCALE,
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              MASK_VARIANT: tl.constexpr,
              IS_DIVISIBLE: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
                OUT_USE_TMA: tl.constexpr,  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    tl.static_assert(BLOCK_M >= BLOCK_N)

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M

    # block pointers
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    QK_SCALE = SM_SCALE * 1.44269504  # 1/log(2)
    q = desc_q.load([qo_offset_y, 0])
    dtype = q.dtype
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    BLOCK_M_DIV_N: tl.constexpr = BLOCK_M // BLOCK_N
    total_cnt = tl.cdiv(N_CTX, BLOCK_N)
    if MASK_VARIANT == "causal":
        lo, hi = start_m * BLOCK_M_DIV_N, min((start_m + 1) * BLOCK_M_DIV_N, total_cnt)
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner_tma_v2(acc, l_i, m_i, q, desc_k, desc_v,  #
                                        offs_m, offs_n, offset_y, lo, hi,
                                         N_CTX, N_CTX, H,
                                        QK_SCALE, BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        IS_DIVISIBLE,
                                        "causal")
    if MASK_VARIANT == "causal":
        lo, hi = 0, start_m * BLOCK_M_DIV_N
    else:
        lo, hi = 0, total_cnt
    acc, l_i, m_i = _attn_fwd_inner_tma_v2(acc, l_i, m_i, q, desc_k, desc_v,  #
                                    offs_m, offs_n, offset_y, lo, hi,
                                    N_CTX, N_CTX, H,
                                    QK_SCALE, BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    IS_DIVISIBLE,
                                    "full")

    # [Note] Handle fully masked out rows:
    # Li will be the sum(e^(-inf)) == 0.0 for masked out rows, mi will be -inf.
    # We set Li to 1.0 which will result in lse/out = 0.0 | after the log(li) + mi(0.0) step
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    # epilogue
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    m_i += tl.log2(l_i)
    if IS_DIVISIBLE:
        tl.store(m_ptrs, m_i)
        if OUT_USE_TMA:
            desc_o.store([qo_offset_y, 0], acc.to(dtype))
        else:
            qvk_offset = tl.cast(off_z, tl.int64) * H * N_CTX * HEAD_DIM + tl.cast(off_h, tl.int64) * N_CTX * HEAD_DIM
            out_ptrs = desc_o + qvk_offset + offs_m[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
            tl.store(out_ptrs, acc.to(dtype))
    else:
        tl.store(m_ptrs, m_i, mask=offs_m < N_CTX)
        if OUT_USE_TMA:
            desc_o.store([off_hz, start_m * BLOCK_M, 0], acc.to(dtype).reshape(1, BLOCK_M, HEAD_DIM))
        else:
            qvk_offset = tl.cast(off_z, tl.int64) * H * N_CTX * HEAD_DIM + tl.cast(off_h, tl.int64) * N_CTX * HEAD_DIM
            out_ptrs = desc_o + qvk_offset + offs_m[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
            tl.store(out_ptrs, acc.to(dtype), offs_m[:, None] < N_CTX)

@triton.autotune(configs=list(filter(tma_keep, tma_configs)), key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
                 prune_configs_by={'early_config_prune': tma_prune_invalid_configs})
@triton.jit
@tritonstd.mark_triton_compilable(inline_run_env_fn=partial(_attn_fwd_kernel_test_fn, is_tma=True, block_causal=192, N_CTX=384),
    real_kwargs={"N_CTX": 30000, "HEAD_DIM": 128, "H": 16, "dtype": torch.float16, "block_causal": 3000},
    raw_fn=_flex_bench_fn_fwd,
)
def _attn_fwd_tma_block_causal_v2(M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
              block_causal,
              SM_SCALE,
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              MASK_VARIANT: tl.constexpr,
              IS_DIVISIBLE: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
                OUT_USE_TMA: tl.constexpr,  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    tl.static_assert(BLOCK_M >= BLOCK_N)

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M

    # block pointers
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    QK_SCALE = SM_SCALE * 1.44269504  # 1/log(2)
    q = desc_q.load([qo_offset_y, 0])
    dtype = q.dtype
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    BLOCK_M_DIV_N: tl.constexpr = BLOCK_M // BLOCK_N
    total_cnt = tl.cdiv(N_CTX, BLOCK_N)
    bc_right_top = tl.cdiv((start_m) * BLOCK_M, block_causal) * block_causal

    bc_right_down = tl.cdiv((start_m + 1) * BLOCK_M, block_causal) * block_causal

    if MASK_VARIANT == "causal":
        lo, hi = bc_right_top // BLOCK_N, min(tl.cdiv(bc_right_down, BLOCK_N), total_cnt)
        # print(bc_left, bc_left_elem, bc_right_elem, block_causal)
        # print("RANGE", lo, hi)
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner_tma_blockcausal_v2(acc, l_i, m_i, q, desc_k, desc_v,  #
                                        offs_m, offs_n, offset_y, lo, hi,
                                         N_CTX, N_CTX, H, block_causal,
                                        QK_SCALE, BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        IS_DIVISIBLE,
                                        "causal")
    if MASK_VARIANT == "causal":
        lo, hi = 0, bc_right_top // BLOCK_N
    else:
        lo, hi = 0, total_cnt
    # print("RANGE FULL", lo, hi)

    acc, l_i, m_i = _attn_fwd_inner_tma_blockcausal_v2(acc, l_i, m_i, q, desc_k, desc_v,  #
                                    offs_m, offs_n, offset_y, lo, hi,
                                    N_CTX, N_CTX, H, block_causal,
                                    QK_SCALE, BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    IS_DIVISIBLE,
                                    "full")

    # [Note] Handle fully masked out rows:
    # Li will be the sum(e^(-inf)) == 0.0 for masked out rows, mi will be -inf.
    # We set Li to 1.0 which will result in lse/out = 0.0 | after the log(li) + mi(0.0) step
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    # epilogue
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    m_i += tl.log2(l_i)
    if IS_DIVISIBLE:
        tl.store(m_ptrs, m_i)
        if OUT_USE_TMA:
            desc_o.store([qo_offset_y, 0], acc.to(dtype))
        else:
            qvk_offset = tl.cast(off_z, tl.int64) * H * N_CTX * HEAD_DIM + tl.cast(off_h, tl.int64) * N_CTX * HEAD_DIM
            out_ptrs = desc_o + qvk_offset + offs_m[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
            tl.store(out_ptrs, acc.to(dtype))
    else:
        tl.store(m_ptrs, m_i, mask=offs_m < N_CTX)
        if OUT_USE_TMA:
            desc_o.store([off_hz, start_m * BLOCK_M, 0], acc.to(dtype).reshape(1, BLOCK_M, HEAD_DIM))
        else:
            qvk_offset = tl.cast(off_z, tl.int64) * H * N_CTX * HEAD_DIM + tl.cast(off_h, tl.int64) * N_CTX * HEAD_DIM
            out_ptrs = desc_o + qvk_offset + offs_m[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
            tl.store(out_ptrs, acc.to(dtype), offs_m[:, None] < N_CTX)

@triton.autotune(configs=list(filter(tma_keep, tma_configs)), key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
                 prune_configs_by={'early_config_prune': tma_prune_invalid_configs})
@triton.jit
@tritonstd.mark_triton_compilable(inline_run_env_fn=partial(_attn_fwd_kernel_test_fn, is_tma=True, head_first=False),
    real_kwargs={"N_CTX": 61440, "HEAD_DIM": 128, "H": 16, "dtype": torch.float16},
    raw_fn=_xformers_bench_fn_fwd,
)
def _attn_fwd_tma_bshd_v2(M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
              SM_SCALE: tl.constexpr,
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              MASK_VARIANT: tl.constexpr,
              IS_DIVISIBLE: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
                OUT_USE_TMA: tl.constexpr,  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    tl.static_assert(BLOCK_M >= BLOCK_N)

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    offset_y = off_z * (N_CTX * H)
    offset_z = off_h * HEAD_DIM
    qo_offset_y = (offset_y + start_m * BLOCK_M)
    # block pointers
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    QK_SCALE: tl.constexpr = SM_SCALE * 1.44269504  # 1/log(2)
    q = desc_q.load([qo_offset_y, offset_z])
    dtype = q.dtype
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    BLOCK_M_DIV_N: tl.constexpr = BLOCK_M // BLOCK_N
    total_cnt = tl.cdiv(N_CTX, BLOCK_N)
    if MASK_VARIANT == "causal":
        lo, hi = start_m * BLOCK_M_DIV_N, min((start_m + 1) * BLOCK_M_DIV_N, total_cnt)
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner_tma_bshd_v2(acc, l_i, m_i, q, desc_k, desc_v,  #
                                        offs_m, offs_n, offset_y, offset_z, lo, hi,
                                         N_CTX, N_CTX, H,
                                        QK_SCALE, BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        IS_DIVISIBLE,
                                        "causal")
    if MASK_VARIANT == "causal":
        lo, hi = 0, start_m * BLOCK_M_DIV_N
    else:
        lo, hi = 0, total_cnt
    acc, l_i, m_i = _attn_fwd_inner_tma_bshd_v2(acc, l_i, m_i, q, desc_k, desc_v,  #
                                    offs_m, offs_n, offset_y, offset_z, lo, hi,
                                    N_CTX, N_CTX, H,
                                    QK_SCALE, BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    IS_DIVISIBLE,
                                    "full")

    # [Note] Handle fully masked out rows:
    # Li will be the sum(e^(-inf)) == 0.0 for masked out rows, mi will be -inf.
    # We set Li to 1.0 which will result in lse/out = 0.0 | after the log(li) + mi(0.0) step
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    # epilogue
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    m_i += tl.log2(l_i)
    tl.store(m_ptrs, m_i)
    if OUT_USE_TMA:
        desc_o.store([qo_offset_y, offset_z], acc.to(dtype))
    else:
        qvk_offset = tl.cast(off_z, tl.int64) * H * N_CTX * HEAD_DIM + tl.cast(off_h, tl.int64) * HEAD_DIM
        out_ptrs = desc_o + qvk_offset + offs_m[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
        tl.store(out_ptrs, acc.to(dtype))


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
        # run last causal part
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
    raw_fn=_xformers_bench_fn_fwd,
)
def _attn_fwd_tma(M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
              SM_SCALE: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              MASK_VARIANT: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #
              OUT_USE_TMA: tl.constexpr,  #
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
    qk_scale = SM_SCALE
    # print("!", start_m, qo_offset_y, off_hz, sm_scale)

    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if MASK_VARIANT == "full":
        STAGE = 1
    else:
        STAGE = 3
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

@triton.jit
def load_checked_2d(
    ptr,
    offs_m,
    offs_n,
    stride_m,
    stride_n,
    IS_DIVISIBLE_M: tl.constexpr,
    IS_DIVISIBLE_N: tl.constexpr,
    M_LEN: tl.constexpr,
    N_DIM: tl.constexpr,
):
    # Calculate final pointer if strides are provided
    if stride_m is not None and stride_n is not None:
        ptr = ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n

    # Handle all masking cases
    if not IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN) & (offs_n[None, :] < N_DIM), other=0.0)
    elif IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_n[None, :] < N_DIM), other=0.0)
    elif not IS_DIVISIBLE_M and IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN), other=0.0)
    else:  # Both divisible
        return tl.load(ptr)

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _may_trans(
    ten,
    TRANS: tl.constexpr,
):
    # Calculate final pointer if strides are provided
    if TRANS:
        return tl.trans(ten)
    else:
        return ten 

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_bwd_dqdkdv_single(q_ptrs, dq_ptrs, do_ptrs, 
                   dk, dv,  #
                   k, v,  #
                   LSE, D,  #
                   start_m, offs_n,
                   # shared by Q/K/V/DO.
                   N_CTX, 
                   # Filled in by the wrapper.
                   BLOCK_M: tl.constexpr,
                   QK_SCALE: tl.constexpr,
                   DQ_ATOMIC: tl.constexpr,
                   Q_TRANSPOSED: tl.constexpr,
                   KV_TRANSPOSED: tl.constexpr,
                   SCORE_TRANSPOSED: tl.constexpr,
                   DQ_TRANSPOSED: tl.constexpr,
                   IS_DIVISIBLE: tl.constexpr,
                   CHECK_BLOCK_BOUNDARY: tl.constexpr,
                   MASK_VARIANT: tl.constexpr,
                   PADDED_QKV: tl.constexpr,
                   PADDED_OUTPUTS: tl.constexpr,
                   CALC_DP_PRE_LOAD_DELTA: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M)

    if IS_DIVISIBLE or PADDED_QKV:
        if Q_TRANSPOSED:
            q_or_qT = tl.load(q_ptrs)
        else:
            q_or_qT = tl.load(q_ptrs)
    else:
        if Q_TRANSPOSED:
            q_or_qT = tl.load(q_ptrs, mask=offs_m[None, :] < N_CTX, other=0.0)
        else:
            q_or_qT = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    # if IS_DIVISIBLE:
    m = tl.load(LSE + offs_m)
    # else:
    #     m = tl.load(LSE + offs_m, mask=offs_m < N_CTX)
    # m = tl.where(m == -float("inf"), 0.0, m)

    # Load m before computing qk to reduce pipeline stall.
    # qkT = tl.dot(k, qT)
    if SCORE_TRANSPOSED:
        qk_or_qkT = tl.dot(_may_trans(k, KV_TRANSPOSED), _may_trans(q_or_qT, not Q_TRANSPOSED))
    else:
        qk_or_qkT = tl.dot(_may_trans(q_or_qT, Q_TRANSPOSED), _may_trans(k, not KV_TRANSPOSED))
    # memory_fence(qk_or_qkT)
    if CHECK_BLOCK_BOUNDARY:
        # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
        if SCORE_TRANSPOSED:
            qk_or_qkT = tl.where(offs_n[:, None] < N_CTX, qk_or_qkT, float("-inf"))
        else:
            qk_or_qkT = tl.where(offs_n[None, :] < N_CTX, qk_or_qkT, float("-inf"))

    if SCORE_TRANSPOSED:
        # pT
        p_or_pT = tl.math.exp2(qk_or_qkT - m[None, :])
    else:
        # p
        p_or_pT = tl.math.exp2(qk_or_qkT - m[:, None])
    # Autoregressive masking.
    if MASK_VARIANT == "causal":
        if SCORE_TRANSPOSED:
            p_or_pT = tl.where((offs_m[None, :] >= offs_n[:, None]), p_or_pT, 0.0)
        else:
            p_or_pT = tl.where((offs_m[:, None] >= offs_n[None, :]), p_or_pT, 0.0)
    if IS_DIVISIBLE or PADDED_QKV:
        do = tl.load(do_ptrs)
    else:
        do = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    if CALC_DP_PRE_LOAD_DELTA:
        if SCORE_TRANSPOSED:
            # dpT
            dp_or_dpT = tl.dot(_may_trans(v, KV_TRANSPOSED), tl.trans(do))
        else:
            dp_or_dpT = tl.dot(do, _may_trans(v, not KV_TRANSPOSED))

    p_p_or_pT = p_or_pT
    p_p_or_pT = p_p_or_pT.to(tl.float16)
    dv += tl.dot(_may_trans(p_p_or_pT, not SCORE_TRANSPOSED), do)
    # D (= delta) is pre-divided by ds_scale.
    # if IS_DIVISIBLE:
    Di = tl.load(D + offs_m)
    # else:
    #     Di = tl.load(D + offs_m, mask=offs_m < N_CTX)
    # Compute dV.
    if not CALC_DP_PRE_LOAD_DELTA:
        if SCORE_TRANSPOSED:
            # dpT
            dp_or_dpT = tl.dot(_may_trans(v, KV_TRANSPOSED), tl.trans(do))
        else:
            dp_or_dpT = tl.dot(do, _may_trans(v, not KV_TRANSPOSED), )

    if SCORE_TRANSPOSED:
        # dsT
        ds_or_dsT = p_or_pT * (dp_or_dpT - Di[None, :])
    else:
        # ds
        ds_or_dsT = p_or_pT * (dp_or_dpT - Di[:, None])
    ds_or_dsT = ds_or_dsT.to(tl.float16)
    dk += tl.dot(_may_trans(ds_or_dsT, not SCORE_TRANSPOSED), _may_trans(q_or_qT, Q_TRANSPOSED))

    if DQ_ATOMIC:
        if DQ_TRANSPOSED:
            dq_or_dqT = tl.dot(_may_trans(k, not KV_TRANSPOSED), _may_trans(ds_or_dsT, not SCORE_TRANSPOSED)).to(tl.float32)
        else:
            dq_or_dqT = tl.dot(_may_trans(ds_or_dsT, SCORE_TRANSPOSED), _may_trans(k, KV_TRANSPOSED)).to(tl.float32)
        if IS_DIVISIBLE or PADDED_OUTPUTS:
            tl.atomic_add(
                dq_ptrs,
                dq_or_dqT,
                sem="relaxed",
            )
        else:
            if DQ_TRANSPOSED:
                mask = offs_m[None, :] < N_CTX
            else:
                mask = offs_m[:, None] < N_CTX
            tl.atomic_add(
                dq_ptrs,
                dq_or_dqT,
                sem="relaxed",
                mask=mask,
            )

    return dk, dv

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_bwd_dqdkdv_single_blockcausal(q_ptrs, dq_ptrs, do_ptrs, 
                   dk, dv,  #
                   k, v,  #
                   LSE, D,  #
                   start_m, offs_n,
                   # shared by Q/K/V/DO.
                   N_CTX, 
                   block_causal,
                   # Filled in by the wrapper.
                   BLOCK_M: tl.constexpr,
                   QK_SCALE: tl.constexpr,
                   DQ_ATOMIC: tl.constexpr,
                   Q_TRANSPOSED: tl.constexpr,
                   KV_TRANSPOSED: tl.constexpr,
                   SCORE_TRANSPOSED: tl.constexpr,
                   DQ_TRANSPOSED: tl.constexpr,
                   IS_DIVISIBLE: tl.constexpr,
                   CHECK_BLOCK_BOUNDARY: tl.constexpr,
                   MASK_VARIANT: tl.constexpr,
                   PADDED_QKV: tl.constexpr,
                   PADDED_OUTPUTS: tl.constexpr,
                   CALC_DP_PRE_LOAD_DELTA: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M)

    if IS_DIVISIBLE or PADDED_QKV:
        if Q_TRANSPOSED:
            q_or_qT = tl.load(q_ptrs)
        else:
            q_or_qT = tl.load(q_ptrs)
    else:
        if Q_TRANSPOSED:
            q_or_qT = tl.load(q_ptrs, mask=offs_m[None, :] < N_CTX, other=0.0)
        else:
            q_or_qT = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    # if IS_DIVISIBLE:
    m = tl.load(LSE + offs_m)
    # else:
    #     m = tl.load(LSE + offs_m, mask=offs_m < N_CTX)
    # m = tl.where(m == -float("inf"), 0.0, m)

    # Load m before computing qk to reduce pipeline stall.
    # qkT = tl.dot(k, qT)
    if SCORE_TRANSPOSED:
        qk_or_qkT = tl.dot(_may_trans(k, KV_TRANSPOSED), _may_trans(q_or_qT, not Q_TRANSPOSED))
    else:
        qk_or_qkT = tl.dot(_may_trans(q_or_qT, Q_TRANSPOSED), _may_trans(k, not KV_TRANSPOSED))
    if CHECK_BLOCK_BOUNDARY:
        # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
        if SCORE_TRANSPOSED:
            qk_or_qkT = tl.where(offs_n[:, None] < N_CTX, qk_or_qkT, float("-inf"))
        else:
            qk_or_qkT = tl.where(offs_n[None, :] < N_CTX, qk_or_qkT, float("-inf"))

    if SCORE_TRANSPOSED:
        # pT
        p_or_pT = tl.math.exp2(qk_or_qkT - m[None, :])
    else:
        # p
        p_or_pT = tl.math.exp2(qk_or_qkT - m[:, None])
    # Autoregressive masking.
    if MASK_VARIANT == "causal":
        if SCORE_TRANSPOSED:
            p_or_pT = tl.where(((offs_m[None, :] // block_causal) >= (offs_n[:, None] // block_causal)), p_or_pT, 0.0)
        else:
            p_or_pT = tl.where(((offs_m[:, None] // block_causal) >= (offs_n[None, :] // block_causal)), p_or_pT, 0.0)
    if IS_DIVISIBLE or PADDED_QKV:
        do = tl.load(do_ptrs)
    else:
        do = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    if CALC_DP_PRE_LOAD_DELTA:
        if SCORE_TRANSPOSED:
            # dpT
            dp_or_dpT = tl.dot(_may_trans(v, KV_TRANSPOSED), tl.trans(do))
        else:
            dp_or_dpT = tl.dot(do, _may_trans(v, not KV_TRANSPOSED))

    p_p_or_pT = p_or_pT
    p_p_or_pT = p_p_or_pT.to(tl.float16)
    dv += tl.dot(_may_trans(p_p_or_pT, not SCORE_TRANSPOSED), do)
    # D (= delta) is pre-divided by ds_scale.
    # if IS_DIVISIBLE:
    Di = tl.load(D + offs_m)
    # else:
    #     Di = tl.load(D + offs_m, mask=offs_m < N_CTX)
    # Compute dV.
    if not CALC_DP_PRE_LOAD_DELTA:
        if SCORE_TRANSPOSED:
            # dpT
            dp_or_dpT = tl.dot(_may_trans(v, KV_TRANSPOSED), tl.trans(do))
        else:
            dp_or_dpT = tl.dot(do, _may_trans(v, not KV_TRANSPOSED), )

    if SCORE_TRANSPOSED:
        # dsT
        ds_or_dsT = p_or_pT * (dp_or_dpT - Di[None, :])
    else:
        # ds
        ds_or_dsT = p_or_pT * (dp_or_dpT - Di[:, None])
    ds_or_dsT = ds_or_dsT.to(tl.float16)
    dk += tl.dot(_may_trans(ds_or_dsT, not SCORE_TRANSPOSED), _may_trans(q_or_qT, Q_TRANSPOSED))

    if DQ_ATOMIC:
        if DQ_TRANSPOSED:
            dq_or_dqT = tl.dot(_may_trans(k, not KV_TRANSPOSED), _may_trans(ds_or_dsT, not SCORE_TRANSPOSED)).to(tl.float32)
        else:
            dq_or_dqT = tl.dot(_may_trans(ds_or_dsT, SCORE_TRANSPOSED), _may_trans(k, KV_TRANSPOSED)).to(tl.float32)
        if IS_DIVISIBLE or PADDED_OUTPUTS:
            tl.atomic_add(
                dq_ptrs,
                dq_or_dqT,
                sem="relaxed",
            )
        else:
            if DQ_TRANSPOSED:
                mask = offs_m[None, :] < N_CTX
            else:
                mask = offs_m[:, None] < N_CTX
            tl.atomic_add(
                dq_ptrs,
                dq_or_dqT,
                sem="relaxed",
                mask=mask,
            )

    return dk, dv
@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_bwd_dqdkdv_single_blockcausal_dynamic(q_ptrs, dq_ptrs, do_ptrs, 
                   dk, dv,  #
                   k, v,  #
                   LSE, D,  #
                   start_m, offs_n,
                   # shared by Q/K/V/DO.
                   N_CTX, 
                   block_causal,
                   # Filled in by the wrapper.
                   BLOCK_M: tl.constexpr,
                   QK_SCALE: tl.constexpr,
                   DQ_ATOMIC: tl.constexpr,
                   Q_TRANSPOSED: tl.constexpr,
                   KV_TRANSPOSED: tl.constexpr,
                   SCORE_TRANSPOSED: tl.constexpr,
                   DQ_TRANSPOSED: tl.constexpr,
                   IS_DIVISIBLE: tl.constexpr,
                   CHECK_BLOCK_BOUNDARY: tl.constexpr,
                   MASK_VARIANT: tl.constexpr,
                   PADDED_QKV: tl.constexpr,
                   PADDED_OUTPUTS: tl.constexpr,
                   CALC_DP_PRE_LOAD_DELTA: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M)

    if IS_DIVISIBLE or PADDED_QKV:
        if Q_TRANSPOSED:
            q_or_qT = tl.load(q_ptrs)
        else:
            q_or_qT = tl.load(q_ptrs)
    else:
        if Q_TRANSPOSED:
            q_or_qT = tl.load(q_ptrs, mask=offs_m[None, :] < N_CTX, other=0.0)
        else:
            q_or_qT = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    # if IS_DIVISIBLE:
    m = tl.load(LSE + offs_m)
    # else:
    #     m = tl.load(LSE + offs_m, mask=offs_m < N_CTX)
    # m = tl.where(m == -float("inf"), 0.0, m)

    # Load m before computing qk to reduce pipeline stall.
    # qkT = tl.dot(k, qT)
    if SCORE_TRANSPOSED:
        qk_or_qkT = tl.dot(_may_trans(k, KV_TRANSPOSED), _may_trans(q_or_qT, not Q_TRANSPOSED))
    else:
        qk_or_qkT = tl.dot(_may_trans(q_or_qT, Q_TRANSPOSED), _may_trans(k, not KV_TRANSPOSED))
    if MASK_VARIANT == "causal":
        if SCORE_TRANSPOSED:
            qk_or_qkT = tl.where((offs_n[:, None] < N_CTX) & ((offs_m[None, :] // block_causal) >= (offs_n[:, None] // block_causal)), qk_or_qkT, float("-inf"))
        else:
            qk_or_qkT = tl.where((offs_n[None, :] < N_CTX) & ((offs_m[:, None] // block_causal) >= (offs_n[None, :] // block_causal)), qk_or_qkT, float("-inf"))
    else:
        if SCORE_TRANSPOSED:
            qk_or_qkT = tl.where((offs_n[:, None] < N_CTX), qk_or_qkT, float("-inf"))
        else:
            qk_or_qkT = tl.where((offs_n[None, :] < N_CTX), qk_or_qkT, float("-inf"))

    if SCORE_TRANSPOSED:
        # pT
        p_or_pT = tl.math.exp2(qk_or_qkT - m[None, :])
    else:
        # p
        p_or_pT = tl.math.exp2(qk_or_qkT - m[:, None])
    if IS_DIVISIBLE or PADDED_QKV:
        do = tl.load(do_ptrs)
    else:
        do = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    if CALC_DP_PRE_LOAD_DELTA:
        if SCORE_TRANSPOSED:
            # dpT
            dp_or_dpT = tl.dot(_may_trans(v, KV_TRANSPOSED), tl.trans(do))
        else:
            dp_or_dpT = tl.dot(do, _may_trans(v, not KV_TRANSPOSED))

    p_p_or_pT = p_or_pT
    p_p_or_pT = p_p_or_pT.to(tl.float16)
    dv += tl.dot(_may_trans(p_p_or_pT, not SCORE_TRANSPOSED), do)
    # D (= delta) is pre-divided by ds_scale.
    # if IS_DIVISIBLE:
    Di = tl.load(D + offs_m)
    # else:
    #     Di = tl.load(D + offs_m, mask=offs_m < N_CTX)
    # Compute dV.
    if not CALC_DP_PRE_LOAD_DELTA:
        if SCORE_TRANSPOSED:
            # dpT
            dp_or_dpT = tl.dot(_may_trans(v, KV_TRANSPOSED), tl.trans(do))
        else:
            dp_or_dpT = tl.dot(do, _may_trans(v, not KV_TRANSPOSED), )

    if SCORE_TRANSPOSED:
        # dsT
        ds_or_dsT = p_or_pT * (dp_or_dpT - Di[None, :])
    else:
        # ds
        ds_or_dsT = p_or_pT * (dp_or_dpT - Di[:, None])
    ds_or_dsT = ds_or_dsT.to(tl.float16)
    dk += tl.dot(_may_trans(ds_or_dsT, not SCORE_TRANSPOSED), _may_trans(q_or_qT, Q_TRANSPOSED))

    if DQ_ATOMIC:
        if DQ_TRANSPOSED:
            dq_or_dqT = tl.dot(_may_trans(k, not KV_TRANSPOSED), _may_trans(ds_or_dsT, not SCORE_TRANSPOSED)).to(tl.float32)
        else:
            dq_or_dqT = tl.dot(_may_trans(ds_or_dsT, SCORE_TRANSPOSED), _may_trans(k, KV_TRANSPOSED)).to(tl.float32)
        if IS_DIVISIBLE or PADDED_OUTPUTS:
            tl.atomic_add(
                dq_ptrs,
                dq_or_dqT,
                sem="relaxed",
            )
        else:
            if DQ_TRANSPOSED:
                mask = offs_m[None, :] < N_CTX
            else:
                mask = offs_m[:, None] < N_CTX
            tl.atomic_add(
                dq_ptrs,
                dq_or_dqT,
                sem="relaxed",
                mask=mask,
            )

    return dk, dv

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_bwd_dqdkdv_single_dynamic(q_ptrs, dq_ptrs, do_ptrs, 
                   dk, dv,  #
                   k, v,  #
                   LSE, D,  #
                   start_m, offs_n,
                   # shared by Q/K/V/DO.
                   N_CTX, 
                   # Filled in by the wrapper.
                   BLOCK_M: tl.constexpr,
                   QK_SCALE: tl.constexpr,
                   DQ_ATOMIC: tl.constexpr,
                   Q_TRANSPOSED: tl.constexpr,
                   KV_TRANSPOSED: tl.constexpr,
                   MASK_VARIANT: tl.constexpr,
                   SCORE_TRANSPOSED: tl.constexpr,
                   DQ_TRANSPOSED: tl.constexpr,
                   IS_DIVISIBLE: tl.constexpr,
                   PADDED_QKV: tl.constexpr,
                   PADDED_OUTPUTS: tl.constexpr,
                   CALC_DP_PRE_LOAD_DELTA: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M)

    if IS_DIVISIBLE or PADDED_QKV:
        if Q_TRANSPOSED:
            q_or_qT = tl.load(q_ptrs)
        else:
            q_or_qT = tl.load(q_ptrs)
    else:
        if Q_TRANSPOSED:
            q_or_qT = tl.load(q_ptrs, mask=offs_m[None, :] < N_CTX, other=0.0)
        else:
            q_or_qT = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    # if IS_DIVISIBLE:
    m = tl.load(LSE + offs_m)
    # else:
    #     m = tl.load(LSE + offs_m, mask=offs_m < N_CTX)
    # m = tl.where(m == -float("inf"), 0.0, m)

    # Load m before computing qk to reduce pipeline stall.
    # qkT = tl.dot(k, qT)
    if SCORE_TRANSPOSED:
        qk_or_qkT = tl.dot(_may_trans(k, KV_TRANSPOSED), _may_trans(q_or_qT, not Q_TRANSPOSED))
    else:
        qk_or_qkT = tl.dot(_may_trans(q_or_qT, Q_TRANSPOSED), _may_trans(k, not KV_TRANSPOSED))
    # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
    # apply seq and causal mask
    if MASK_VARIANT == "causal":
        if SCORE_TRANSPOSED:
            qk_or_qkT = tl.where((offs_n[:, None] < N_CTX) & (offs_m[None, :] >= offs_n[:, None]), qk_or_qkT, float("-inf"))
        else:
            qk_or_qkT = tl.where((offs_n[None, :] < N_CTX) & (offs_m[:, None] >= offs_n[None, :]), qk_or_qkT, float("-inf"))
    else:
        if SCORE_TRANSPOSED:
            qk_or_qkT = tl.where((offs_n[:, None] < N_CTX), qk_or_qkT, float("-inf"))
        else:
            qk_or_qkT = tl.where((offs_n[None, :] < N_CTX), qk_or_qkT, float("-inf"))
    if SCORE_TRANSPOSED:
        # pT
        p_or_pT = tl.math.exp2(qk_or_qkT - m[None, :])
    else:
        # p
        p_or_pT = tl.math.exp2(qk_or_qkT - m[:, None])
    # Autoregressive masking.
    if IS_DIVISIBLE or PADDED_QKV:
        do = tl.load(do_ptrs)
    else:
        do = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    if CALC_DP_PRE_LOAD_DELTA:
        if SCORE_TRANSPOSED:
            # dpT
            dp_or_dpT = tl.dot(_may_trans(v, KV_TRANSPOSED), tl.trans(do))
        else:
            dp_or_dpT = tl.dot(do, _may_trans(v, not KV_TRANSPOSED))

    p_p_or_pT = p_or_pT
    p_p_or_pT = p_p_or_pT.to(tl.float16)
    dv += tl.dot(_may_trans(p_p_or_pT, not SCORE_TRANSPOSED), do)

    # D (= delta) is pre-divided by ds_scale.
    # if IS_DIVISIBLE:
    Di = tl.load(D + offs_m)
    # else:
    #     Di = tl.load(D + offs_m, mask=offs_m < N_CTX)
    # Compute dV.
    if not CALC_DP_PRE_LOAD_DELTA:
        if SCORE_TRANSPOSED:
            # dpT
            dp_or_dpT = tl.dot(_may_trans(v, KV_TRANSPOSED), tl.trans(do))
        else:
            dp_or_dpT = tl.dot(do, _may_trans(v, not KV_TRANSPOSED), )


    if SCORE_TRANSPOSED:
        # dsT
        ds_or_dsT = p_or_pT * (dp_or_dpT - Di[None, :])
    else:
        # ds
        ds_or_dsT = p_or_pT * (dp_or_dpT - Di[:, None])
    ds_or_dsT = ds_or_dsT.to(tl.float16)
    dk += tl.dot(_may_trans(ds_or_dsT, not SCORE_TRANSPOSED), _may_trans(q_or_qT, Q_TRANSPOSED))
    if DQ_ATOMIC:
        if DQ_TRANSPOSED:
            dq_or_dqT = tl.dot(_may_trans(k, not KV_TRANSPOSED), _may_trans(ds_or_dsT, not SCORE_TRANSPOSED)).to(tl.float32)
        else:
            dq_or_dqT = tl.dot(_may_trans(ds_or_dsT, SCORE_TRANSPOSED), _may_trans(k, KV_TRANSPOSED)).to(tl.float32)
        if IS_DIVISIBLE or PADDED_OUTPUTS:
            tl.atomic_add(
                dq_ptrs,
                dq_or_dqT * 0.6931471824645996,
                sem="relaxed",
            )
        else:
            if DQ_TRANSPOSED:
                mask = offs_m[None, :] < N_CTX
            else:
                mask = offs_m[:, None] < N_CTX
            tl.atomic_add(
                dq_ptrs,
                dq_or_dqT * 0.6931471824645996,
                sem="relaxed",
                mask=mask,
            )

    return dk, dv


@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_bwd_dqdkdv_single_tma(desc_q, desc_dq, desc_do, 
                   dk, dv,  #
                   k, v,  #
                   LSE, D,  #
                   start_m, offs_n,
                   # shared by Q/K/V/DO.
                   offset_y,
                   N_CTX, 
                   # Filled in by the wrapper.
                   BLOCK_M: tl.constexpr,
                   QK_SCALE: tl.constexpr,
                   DQ_ATOMIC: tl.constexpr,
                   Q_TRANSPOSED: tl.constexpr,
                   KV_TRANSPOSED: tl.constexpr,
                   SCORE_TRANSPOSED: tl.constexpr,
                   DQ_TRANSPOSED: tl.constexpr,
                   IS_DIVISIBLE: tl.constexpr,
                   CHECK_BLOCK_BOUNDARY: tl.constexpr,
                   MASK_VARIANT: tl.constexpr,
                   PADDED_QKV: tl.constexpr,
                   PADDED_OUTPUTS: tl.constexpr,
                   CALC_DP_PRE_LOAD_DELTA: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M)

    if Q_TRANSPOSED:
        q_or_qT = desc_q.load([offset_y, 0]).T
    else:
        q_or_qT = desc_q.load([offset_y, 0])
    # if IS_DIVISIBLE:
    m = tl.load(LSE + offs_m)
    # else:
    #     m = tl.load(LSE + offs_m, mask=offs_m < N_CTX)
    # m = tl.where(m == -float("inf"), 0.0, m)
    # Load m before computing qk to reduce pipeline stall.
    # qkT = tl.dot(k, qT)
    if SCORE_TRANSPOSED:
        qk_or_qkT = tl.dot(_may_trans(k, KV_TRANSPOSED), _may_trans(q_or_qT, not Q_TRANSPOSED))
    else:
        qk_or_qkT = tl.dot(_may_trans(q_or_qT, Q_TRANSPOSED), _may_trans(k, not KV_TRANSPOSED))
    if CHECK_BLOCK_BOUNDARY:
        # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
        if SCORE_TRANSPOSED:
            qk_or_qkT = tl.where(offs_n[:, None] < N_CTX, qk_or_qkT, float("-inf"))
        else:
            qk_or_qkT = tl.where(offs_n[None, :] < N_CTX, qk_or_qkT, float("-inf"))
    if SCORE_TRANSPOSED:
        # pT
        p_or_pT = tl.math.exp2(qk_or_qkT - m[None, :])
    else:
        # p
        p_or_pT = tl.math.exp2(qk_or_qkT - m[:, None])
    # Autoregressive masking.
    if MASK_VARIANT == "causal":
        if SCORE_TRANSPOSED:
            p_or_pT = tl.where((offs_m[None, :] >= offs_n[:, None]), p_or_pT, 0.0)
        else:
            p_or_pT = tl.where((offs_m[:, None] >= offs_n[None, :]), p_or_pT, 0.0)
    do = desc_do.load([offset_y, 0])
    if CALC_DP_PRE_LOAD_DELTA:
        if SCORE_TRANSPOSED:
            # dpT
            dp_or_dpT = tl.dot(_may_trans(v, KV_TRANSPOSED), tl.trans(do))
        else:
            dp_or_dpT = tl.dot(do, _may_trans(v, not KV_TRANSPOSED))

    p_p_or_pT = p_or_pT
    p_p_or_pT = p_p_or_pT.to(tl.float16)
    dv += tl.dot(_may_trans(p_p_or_pT, not SCORE_TRANSPOSED), do)
    # D (= delta) is pre-divided by ds_scale.
    # if IS_DIVISIBLE:
    Di = tl.load(D + offs_m)
    # else:
    #     Di = tl.load(D + offs_m, mask=offs_m < N_CTX)
    # Compute dV.
    if not CALC_DP_PRE_LOAD_DELTA:
        if SCORE_TRANSPOSED:
            # dpT
            dp_or_dpT = tl.dot(_may_trans(v, KV_TRANSPOSED), tl.trans(do))
        else:
            dp_or_dpT = tl.dot(do, _may_trans(v, not KV_TRANSPOSED))

    if SCORE_TRANSPOSED:
        # dsT
        ds_or_dsT = p_or_pT * (dp_or_dpT - Di[None, :])
    else:
        # ds
        ds_or_dsT = p_or_pT * (dp_or_dpT - Di[:, None])
    ds_or_dsT = ds_or_dsT.to(tl.float16)
    dk += tl.dot(_may_trans(ds_or_dsT, not SCORE_TRANSPOSED), _may_trans(q_or_qT, Q_TRANSPOSED))
    if DQ_ATOMIC:
        if DQ_TRANSPOSED:
            dq_or_dqT = tl.dot(_may_trans(k, not KV_TRANSPOSED), _may_trans(ds_or_dsT, not SCORE_TRANSPOSED)).to(tl.float32)
        else:
            dq_or_dqT = tl.dot(_may_trans(ds_or_dsT, SCORE_TRANSPOSED), _may_trans(k, KV_TRANSPOSED)).to(tl.float32)
        desc_dq.atomic_add(
            [offset_y, 0],
            _may_trans(dq_or_dqT, DQ_TRANSPOSED) * 0.6931471824645996)
    return dk, dv

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_bwd_dqdkdv_single_tma_dynamic(desc_q, desc_dq, desc_do, 
                   dk, dv,  #
                   k, v,  #
                   LSE, D,  #
                   start_m, offs_n,
                   offset_y,
                   # shared by Q/K/V/DO.
                   N_CTX, 
                   # Filled in by the wrapper.
                   BLOCK_M: tl.constexpr,
                   QK_SCALE: tl.constexpr,
                   DQ_ATOMIC: tl.constexpr,
                   Q_TRANSPOSED: tl.constexpr,
                   KV_TRANSPOSED: tl.constexpr,
                   SCORE_TRANSPOSED: tl.constexpr,
                   DQ_TRANSPOSED: tl.constexpr,
                   MASK_VARIANT: tl.constexpr,
                   IS_DIVISIBLE: tl.constexpr,
                   PADDED_QKV: tl.constexpr,
                   PADDED_OUTPUTS: tl.constexpr,
                   CALC_DP_PRE_LOAD_DELTA: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M)

    if Q_TRANSPOSED:
        q_or_qT = desc_q.load([offset_y, 0]).T
    else:
        q_or_qT = desc_q.load([offset_y, 0])
    # if IS_DIVISIBLE:
    m = tl.load(LSE + offs_m)
    # else:
    #     m = tl.load(LSE + offs_m, mask=offs_m < N_CTX)
    # m = tl.where(m == -float("inf"), 0.0, m)
    # Load m before computing qk to reduce pipeline stall.
    # qkT = tl.dot(k, qT)
    if SCORE_TRANSPOSED:
        qk_or_qkT = tl.dot(_may_trans(k, KV_TRANSPOSED), _may_trans(q_or_qT, not Q_TRANSPOSED))
    else:
        qk_or_qkT = tl.dot(_may_trans(q_or_qT, Q_TRANSPOSED), _may_trans(k, not KV_TRANSPOSED))
    if MASK_VARIANT == "causal":
        if SCORE_TRANSPOSED:
            qk_or_qkT = tl.where((offs_n[:, None] < N_CTX) & (offs_m[None, :] >= offs_n[:, None]), qk_or_qkT, float("-inf"))
        else:
            qk_or_qkT = tl.where((offs_n[None, :] < N_CTX) & (offs_m[:, None] >= offs_n[None, :]), qk_or_qkT, float("-inf"))
    else:
        if SCORE_TRANSPOSED:
            qk_or_qkT = tl.where((offs_n[:, None] < N_CTX), qk_or_qkT, float("-inf"))
        else:
            qk_or_qkT = tl.where((offs_n[None, :] < N_CTX), qk_or_qkT, float("-inf"))

    if SCORE_TRANSPOSED:
        # pT
        p_or_pT = tl.math.exp2(qk_or_qkT - m[None, :])
    else:
        # p
        p_or_pT = tl.math.exp2(qk_or_qkT - m[:, None])
    # Autoregressive masking.
    do = desc_do.load([offset_y, 0])
    if CALC_DP_PRE_LOAD_DELTA:
        if SCORE_TRANSPOSED:
            # dpT
            dp_or_dpT = tl.dot(_may_trans(v, KV_TRANSPOSED), tl.trans(do))
        else:
            dp_or_dpT = tl.dot(do, _may_trans(v, not KV_TRANSPOSED))

    p_p_or_pT = p_or_pT
    p_p_or_pT = p_p_or_pT.to(tl.float16)
    dv += tl.dot(_may_trans(p_p_or_pT, not SCORE_TRANSPOSED), do)
    # D (= delta) is pre-divided by ds_scale.
    # if IS_DIVISIBLE:
    Di = tl.load(D + offs_m)
    # else:
    #     Di = tl.load(D + offs_m, mask=offs_m < N_CTX)
    # Compute dV.
    if not CALC_DP_PRE_LOAD_DELTA:
        if SCORE_TRANSPOSED:
            # dpT
            dp_or_dpT = tl.dot(_may_trans(v, KV_TRANSPOSED), tl.trans(do))
        else:
            dp_or_dpT = tl.dot(do, _may_trans(v, not KV_TRANSPOSED))


    if SCORE_TRANSPOSED:
        # dsT
        ds_or_dsT = p_or_pT * (dp_or_dpT - Di[None, :])
    else:
        # ds
        ds_or_dsT = p_or_pT * (dp_or_dpT - Di[:, None])
    ds_or_dsT = ds_or_dsT.to(tl.float16)
    dk += tl.dot(_may_trans(ds_or_dsT, not SCORE_TRANSPOSED), _may_trans(q_or_qT, Q_TRANSPOSED))
    if DQ_ATOMIC:
        if DQ_TRANSPOSED:
            dq_or_dqT = tl.dot(_may_trans(k, not KV_TRANSPOSED), _may_trans(ds_or_dsT, not SCORE_TRANSPOSED)).to(tl.float32)
        else:
            dq_or_dqT = tl.dot(_may_trans(ds_or_dsT, SCORE_TRANSPOSED), _may_trans(k, KV_TRANSPOSED)).to(tl.float32)
        desc_dq.atomic_add(
            [offset_y, 0],
            _may_trans(dq_or_dqT, DQ_TRANSPOSED) * 0.6931471824645996)
    return dk, dv

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_bwd_dqdkdv_single_tma_dynamic_blockcausal(desc_q, desc_dq, desc_do, 
                   dk, dv,  #
                   k, v,  #
                   LSE, D,  #
                   start_m, offs_n,
                   offset_y,
                   # shared by Q/K/V/DO.
                   N_CTX, 
                   block_causal,
                   # Filled in by the wrapper.
                   BLOCK_M: tl.constexpr,
                   QK_SCALE: tl.constexpr,
                   DQ_ATOMIC: tl.constexpr,
                   Q_TRANSPOSED: tl.constexpr,
                   KV_TRANSPOSED: tl.constexpr,
                   SCORE_TRANSPOSED: tl.constexpr,
                   DQ_TRANSPOSED: tl.constexpr,
                   MASK_VARIANT: tl.constexpr,
                   IS_DIVISIBLE: tl.constexpr,
                   PADDED_QKV: tl.constexpr,
                   PADDED_OUTPUTS: tl.constexpr,
                   CALC_DP_PRE_LOAD_DELTA: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M)

    if Q_TRANSPOSED:
        q_or_qT = desc_q.load([offset_y, 0]).T
    else:
        q_or_qT = desc_q.load([offset_y, 0])
    # if IS_DIVISIBLE:
    m = tl.load(LSE + offs_m)
    # else:
    #     m = tl.load(LSE + offs_m, mask=offs_m < N_CTX)
    # m = tl.where(m == -float("inf"), 0.0, m)
    # Load m before computing qk to reduce pipeline stall.
    # qkT = tl.dot(k, qT)
    if SCORE_TRANSPOSED:
        qk_or_qkT = tl.dot(_may_trans(k, KV_TRANSPOSED), _may_trans(q_or_qT, not Q_TRANSPOSED))
    else:
        qk_or_qkT = tl.dot(_may_trans(q_or_qT, Q_TRANSPOSED), _may_trans(k, not KV_TRANSPOSED))
    if MASK_VARIANT == "causal":
        if SCORE_TRANSPOSED:
            qk_or_qkT = tl.where((offs_n[:, None] < N_CTX) & ((offs_m[None, :] // block_causal) >= (offs_n[:, None] // block_causal)), qk_or_qkT, float("-inf"))
        else:
            qk_or_qkT = tl.where((offs_n[None, :] < N_CTX) & ((offs_m[:, None] // block_causal) >= (offs_n[None, :] // block_causal)), qk_or_qkT, float("-inf"))
    else:
        if SCORE_TRANSPOSED:
            qk_or_qkT = tl.where((offs_n[:, None] < N_CTX), qk_or_qkT, float("-inf"))
        else:
            qk_or_qkT = tl.where((offs_n[None, :] < N_CTX), qk_or_qkT, float("-inf"))

    if SCORE_TRANSPOSED:
        # pT
        p_or_pT = tl.math.exp2(qk_or_qkT - m[None, :])
    else:
        # p
        p_or_pT = tl.math.exp2(qk_or_qkT - m[:, None])
    # Autoregressive masking.
    do = desc_do.load([offset_y, 0])
    if CALC_DP_PRE_LOAD_DELTA:
        if SCORE_TRANSPOSED:
            # dpT
            dp_or_dpT = tl.dot(_may_trans(v, KV_TRANSPOSED), tl.trans(do))
        else:
            dp_or_dpT = tl.dot(do, _may_trans(v, not KV_TRANSPOSED))

    p_p_or_pT = p_or_pT
    p_p_or_pT = p_p_or_pT.to(tl.float16)
    dv += tl.dot(_may_trans(p_p_or_pT, not SCORE_TRANSPOSED), do)
    # D (= delta) is pre-divided by ds_scale.
    # if IS_DIVISIBLE:
    Di = tl.load(D + offs_m)
    # else:
    #     Di = tl.load(D + offs_m, mask=offs_m < N_CTX)
    # Compute dV.
    if not CALC_DP_PRE_LOAD_DELTA:
        if SCORE_TRANSPOSED:
            # dpT
            dp_or_dpT = tl.dot(_may_trans(v, KV_TRANSPOSED), tl.trans(do))
        else:
            dp_or_dpT = tl.dot(do, _may_trans(v, not KV_TRANSPOSED))


    if SCORE_TRANSPOSED:
        # dsT
        ds_or_dsT = p_or_pT * (dp_or_dpT - Di[None, :])
    else:
        # ds
        ds_or_dsT = p_or_pT * (dp_or_dpT - Di[:, None])
    ds_or_dsT = ds_or_dsT.to(tl.float16)
    dk += tl.dot(_may_trans(ds_or_dsT, not SCORE_TRANSPOSED), _may_trans(q_or_qT, Q_TRANSPOSED))
    if DQ_ATOMIC:
        if DQ_TRANSPOSED:
            dq_or_dqT = tl.dot(_may_trans(k, not KV_TRANSPOSED), _may_trans(ds_or_dsT, not SCORE_TRANSPOSED)).to(tl.float32)
        else:
            dq_or_dqT = tl.dot(_may_trans(ds_or_dsT, SCORE_TRANSPOSED), _may_trans(k, KV_TRANSPOSED)).to(tl.float32)
        desc_dq.atomic_add(
            [offset_y, 0],
            _may_trans(dq_or_dqT, DQ_TRANSPOSED) * 0.6931471824645996)
    return dk, dv

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_bwd_dqdkdv(DQ, dk, dv,  #
                   Q, k, v,  #
                   DO,  #
                   LSE, D,  #
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   N_CTX, 
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                   QK_SCALE: tl.constexpr,  #
                   BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   DQ_ATOMIC: tl.constexpr,
                   Q_TRANSPOSED: tl.constexpr,
                   KV_TRANSPOSED: tl.constexpr,
                   SCORE_TRANSPOSED: tl.constexpr,
                   DQ_TRANSPOSED: tl.constexpr,
                   IS_DIVISIBLE: tl.constexpr,
                   PADDED_QKV: tl.constexpr,
                   PADDED_OUTPUTS: tl.constexpr,
                   CALC_DP_PRE_LOAD_DELTA: tl.constexpr,
                   MASK_VARIANT: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    if Q_TRANSPOSED:
        q_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    else:
        q_ptrs = Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    if DQ_TRANSPOSED:
        dq_ptrs = DQ + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    else:
        dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d

    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    # NOTE: increment start_m and create offs_m is slightly faster than 
    # increment offs_m.
    if not IS_DIVISIBLE:
        if num_steps >= 1:

            for blk_idx in range(num_steps - 1):
                dk, dv = _attn_bwd_dqdkdv_single(
                    q_ptrs, dq_ptrs, do_ptrs, 
                    dk, dv,  #
                    k, v,  #
                    LSE, D,  #
                    start_m, offs_n,
                    # shared by Q/K/V/DO.
                    N_CTX, 
                    BLOCK_M=BLOCK_M1,
                    QK_SCALE=QK_SCALE,
                    DQ_ATOMIC=DQ_ATOMIC,
                    Q_TRANSPOSED=Q_TRANSPOSED,
                    KV_TRANSPOSED=KV_TRANSPOSED,
                    SCORE_TRANSPOSED=SCORE_TRANSPOSED,
                    DQ_TRANSPOSED=DQ_TRANSPOSED,
                    # TODO why slower if we set IS_DIVISIBLE=True here?
                    IS_DIVISIBLE=MASK_VARIANT == "full", 
                    CHECK_BLOCK_BOUNDARY=False,
                    MASK_VARIANT=MASK_VARIANT,
                    PADDED_QKV=PADDED_QKV,
                    PADDED_OUTPUTS=PADDED_OUTPUTS,
                    CALC_DP_PRE_LOAD_DELTA=CALC_DP_PRE_LOAD_DELTA,
                )
                # Increment pointers.
                start_m += BLOCK_M1
                # qT_ptrs += step_m * stride_tok
                q_ptrs += BLOCK_M1 * stride_tok
                do_ptrs += BLOCK_M1 * stride_tok
                # offs_m += BLOCK_M1

                if DQ_ATOMIC:
                    dq_ptrs += BLOCK_M1 * stride_tok
            dk, dv = _attn_bwd_dqdkdv_single(
                q_ptrs, dq_ptrs, do_ptrs, 
                dk, dv,  #
                k, v,  #
                LSE, D,  #
                start_m, offs_n,
                # shared by Q/K/V/DO.
                N_CTX, 
                BLOCK_M=BLOCK_M1,
                QK_SCALE=QK_SCALE,
                DQ_ATOMIC=DQ_ATOMIC,
                Q_TRANSPOSED=Q_TRANSPOSED,
                KV_TRANSPOSED=KV_TRANSPOSED,
                SCORE_TRANSPOSED=SCORE_TRANSPOSED,
                DQ_TRANSPOSED=DQ_TRANSPOSED,
                IS_DIVISIBLE=IS_DIVISIBLE, 
                CHECK_BLOCK_BOUNDARY=True,
                MASK_VARIANT=MASK_VARIANT,
                PADDED_QKV=PADDED_QKV,
                PADDED_OUTPUTS=PADDED_OUTPUTS,
                CALC_DP_PRE_LOAD_DELTA=CALC_DP_PRE_LOAD_DELTA,

            )
    else:
        for blk_idx in range(num_steps):
            dk, dv = _attn_bwd_dqdkdv_single(
                q_ptrs, dq_ptrs, do_ptrs, 
                dk, dv,  #
                k, v,  #
                LSE, D,  #
                start_m, offs_n,
                # shared by Q/K/V/DO.
                N_CTX, 
                BLOCK_M=BLOCK_M1,
                QK_SCALE=QK_SCALE,
                DQ_ATOMIC=DQ_ATOMIC,
                Q_TRANSPOSED=Q_TRANSPOSED,
                KV_TRANSPOSED=KV_TRANSPOSED,
                SCORE_TRANSPOSED=SCORE_TRANSPOSED,
                DQ_TRANSPOSED=DQ_TRANSPOSED,
                IS_DIVISIBLE=IS_DIVISIBLE, 
                CHECK_BLOCK_BOUNDARY=False,
                MASK_VARIANT=MASK_VARIANT,
                PADDED_QKV=PADDED_QKV,
                PADDED_OUTPUTS=PADDED_OUTPUTS,
                CALC_DP_PRE_LOAD_DELTA=CALC_DP_PRE_LOAD_DELTA,

            )
            # Increment pointers.
            start_m += BLOCK_M1
            # offs_m += BLOCK_M1

            # qT_ptrs += step_m * stride_tok
            q_ptrs += BLOCK_M1 * stride_tok
            do_ptrs += BLOCK_M1 * stride_tok
            if DQ_ATOMIC:
                dq_ptrs += BLOCK_M1 * stride_tok
    return dk, dv
@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_bwd_dqdkdv_blockcausal(DQ, dk, dv,  #
                   Q, k, v,  #
                   DO,  #
                   LSE, D,  #
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   N_CTX, 
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                   block_causal,
                   QK_SCALE: tl.constexpr,  #
                   BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   DQ_ATOMIC: tl.constexpr,
                   Q_TRANSPOSED: tl.constexpr,
                   KV_TRANSPOSED: tl.constexpr,
                   SCORE_TRANSPOSED: tl.constexpr,
                   DQ_TRANSPOSED: tl.constexpr,
                   IS_DIVISIBLE: tl.constexpr,
                   PADDED_QKV: tl.constexpr,
                   PADDED_OUTPUTS: tl.constexpr,
                   CALC_DP_PRE_LOAD_DELTA: tl.constexpr,
                   MASK_VARIANT: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    if Q_TRANSPOSED:
        q_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    else:
        q_ptrs = Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    if DQ_TRANSPOSED:
        dq_ptrs = DQ + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    else:
        dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d

    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    # NOTE: increment start_m and create offs_m is slightly faster than 
    # increment offs_m.
    if not IS_DIVISIBLE:
        if num_steps >= 1:

            for blk_idx in range(num_steps - 1):
                dk, dv = _attn_bwd_dqdkdv_single_blockcausal(
                    q_ptrs, dq_ptrs, do_ptrs, 
                    dk, dv,  #
                    k, v,  #
                    LSE, D,  #
                    start_m, offs_n,
                    # shared by Q/K/V/DO.
                    N_CTX, 
                    block_causal,
                    BLOCK_M=BLOCK_M1,
                    QK_SCALE=QK_SCALE,
                    DQ_ATOMIC=DQ_ATOMIC,
                    Q_TRANSPOSED=Q_TRANSPOSED,
                    KV_TRANSPOSED=KV_TRANSPOSED,
                    SCORE_TRANSPOSED=SCORE_TRANSPOSED,
                    DQ_TRANSPOSED=DQ_TRANSPOSED,
                    # TODO why slower if we set IS_DIVISIBLE=True here?
                    IS_DIVISIBLE=MASK_VARIANT == "full", 
                    CHECK_BLOCK_BOUNDARY=False,
                    MASK_VARIANT=MASK_VARIANT,
                    PADDED_QKV=PADDED_QKV,
                    PADDED_OUTPUTS=PADDED_OUTPUTS,
                    CALC_DP_PRE_LOAD_DELTA=CALC_DP_PRE_LOAD_DELTA,
                )
                # Increment pointers.
                start_m += BLOCK_M1
                # qT_ptrs += step_m * stride_tok
                q_ptrs += BLOCK_M1 * stride_tok
                do_ptrs += BLOCK_M1 * stride_tok
                # offs_m += BLOCK_M1

                if DQ_ATOMIC:
                    dq_ptrs += BLOCK_M1 * stride_tok
            dk, dv = _attn_bwd_dqdkdv_single_blockcausal(
                q_ptrs, dq_ptrs, do_ptrs, 
                dk, dv,  #
                k, v,  #
                LSE, D,  #
                start_m, offs_n,
                # shared by Q/K/V/DO.
                N_CTX, 
                block_causal,
                BLOCK_M=BLOCK_M1,
                QK_SCALE=QK_SCALE,
                DQ_ATOMIC=DQ_ATOMIC,
                Q_TRANSPOSED=Q_TRANSPOSED,
                KV_TRANSPOSED=KV_TRANSPOSED,
                SCORE_TRANSPOSED=SCORE_TRANSPOSED,
                DQ_TRANSPOSED=DQ_TRANSPOSED,
                IS_DIVISIBLE=IS_DIVISIBLE, 
                CHECK_BLOCK_BOUNDARY=True,
                MASK_VARIANT=MASK_VARIANT,
                PADDED_QKV=PADDED_QKV,
                PADDED_OUTPUTS=PADDED_OUTPUTS,
                CALC_DP_PRE_LOAD_DELTA=CALC_DP_PRE_LOAD_DELTA,

            )
    else:
        for blk_idx in range(num_steps):
            dk, dv = _attn_bwd_dqdkdv_single_blockcausal(
                q_ptrs, dq_ptrs, do_ptrs, 
                dk, dv,  #
                k, v,  #
                LSE, D,  #
                start_m, offs_n,
                # shared by Q/K/V/DO.
                N_CTX, 
                block_causal,
                BLOCK_M=BLOCK_M1,
                QK_SCALE=QK_SCALE,
                DQ_ATOMIC=DQ_ATOMIC,
                Q_TRANSPOSED=Q_TRANSPOSED,
                KV_TRANSPOSED=KV_TRANSPOSED,
                SCORE_TRANSPOSED=SCORE_TRANSPOSED,
                DQ_TRANSPOSED=DQ_TRANSPOSED,
                IS_DIVISIBLE=IS_DIVISIBLE, 
                CHECK_BLOCK_BOUNDARY=False,
                MASK_VARIANT=MASK_VARIANT,
                PADDED_QKV=PADDED_QKV,
                PADDED_OUTPUTS=PADDED_OUTPUTS,
                CALC_DP_PRE_LOAD_DELTA=CALC_DP_PRE_LOAD_DELTA,

            )
            # Increment pointers.
            start_m += BLOCK_M1
            # offs_m += BLOCK_M1

            # qT_ptrs += step_m * stride_tok
            q_ptrs += BLOCK_M1 * stride_tok
            do_ptrs += BLOCK_M1 * stride_tok
            if DQ_ATOMIC:
                dq_ptrs += BLOCK_M1 * stride_tok
    return dk, dv
    

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_bwd_dqdkdv_dynamic(DQ, dk, dv,  #
                   Q, k, v,  #
                   DO,  #
                   LSE, D,  #
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   N_CTX, 
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                   QK_SCALE: tl.constexpr,  #
                   BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   DQ_ATOMIC: tl.constexpr,
                   Q_TRANSPOSED: tl.constexpr,
                   KV_TRANSPOSED: tl.constexpr,
                   SCORE_TRANSPOSED: tl.constexpr,
                   DQ_TRANSPOSED: tl.constexpr,
                   IS_DIVISIBLE: tl.constexpr,
                   PADDED_QKV: tl.constexpr,
                   PADDED_OUTPUTS: tl.constexpr,
                   CALC_DP_PRE_LOAD_DELTA: tl.constexpr,
                   MASK_VARIANT: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    if Q_TRANSPOSED:
        q_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    else:
        q_ptrs = Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    if DQ_TRANSPOSED:
        dq_ptrs = DQ + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    else:
        dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d

    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    # NOTE: increment start_m and create offs_m is slightly faster than 
    # increment offs_m.
    for blk_idx in range(num_steps):
        dk, dv = _attn_bwd_dqdkdv_single_dynamic(
            q_ptrs, dq_ptrs, do_ptrs, 
            dk, dv,  #
            k, v,  #
            LSE, D,  #
            start_m, offs_n,
            # shared by Q/K/V/DO.
            N_CTX, 
            BLOCK_M=BLOCK_M1,
            QK_SCALE=QK_SCALE,
            DQ_ATOMIC=DQ_ATOMIC,
            Q_TRANSPOSED=Q_TRANSPOSED,
            KV_TRANSPOSED=KV_TRANSPOSED,
            SCORE_TRANSPOSED=SCORE_TRANSPOSED,
            DQ_TRANSPOSED=DQ_TRANSPOSED,
            MASK_VARIANT=MASK_VARIANT,
            IS_DIVISIBLE=IS_DIVISIBLE, 
            PADDED_QKV=PADDED_QKV,
            PADDED_OUTPUTS=PADDED_OUTPUTS,
            CALC_DP_PRE_LOAD_DELTA=CALC_DP_PRE_LOAD_DELTA,
        )
        # Increment pointers.
        start_m += BLOCK_M1
        # offs_m += BLOCK_M1

        # qT_ptrs += step_m * stride_tok
        q_ptrs += BLOCK_M1 * stride_tok
        do_ptrs += BLOCK_M1 * stride_tok
        if DQ_ATOMIC:
            dq_ptrs += BLOCK_M1 * stride_tok
    return dk, dv

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_bwd_dqdkdv_tma(desc_dq, dk, dv,  #
                   desc_q, k, v,  #
                   desc_do,  #
                   LSE, D,  #
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   N_CTX, 
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                   offset_y,
                   QK_SCALE: tl.constexpr,  #
                   BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   DQ_ATOMIC: tl.constexpr,
                   Q_TRANSPOSED: tl.constexpr,
                   KV_TRANSPOSED: tl.constexpr,
                   SCORE_TRANSPOSED: tl.constexpr,
                   DQ_TRANSPOSED: tl.constexpr,
                   IS_DIVISIBLE: tl.constexpr,
                   PADDED_QKV: tl.constexpr,
                   PADDED_OUTPUTS: tl.constexpr,
                   CALC_DP_PRE_LOAD_DELTA: tl.constexpr,
                   MASK_VARIANT: tl.constexpr):
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    # offset_y
    offset_y = offset_y + start_m
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    # NOTE: increment start_m and create offs_m is slightly faster than 
    # increment offs_m.
    if not IS_DIVISIBLE:
        if num_steps >= 1:

            for blk_idx in range(num_steps - 1):
                dk, dv = _attn_bwd_dqdkdv_single_tma(
                    desc_q, desc_dq, desc_do, 
                    dk, dv,  #
                    k, v,  #
                    LSE, D,  #
                    start_m, offs_n,
                    offset_y,
                    # shared by Q/K/V/DO.
                    N_CTX, 
                    BLOCK_M=BLOCK_M1,
                    QK_SCALE=QK_SCALE,
                    DQ_ATOMIC=DQ_ATOMIC,
                    Q_TRANSPOSED=Q_TRANSPOSED,
                    KV_TRANSPOSED=KV_TRANSPOSED,
                    SCORE_TRANSPOSED=SCORE_TRANSPOSED,
                    DQ_TRANSPOSED=DQ_TRANSPOSED,
                    # TODO why slower if we set IS_DIVISIBLE=True here?
                    IS_DIVISIBLE=MASK_VARIANT == "full", 
                    CHECK_BLOCK_BOUNDARY=False,
                    MASK_VARIANT=MASK_VARIANT,
                    PADDED_QKV=PADDED_QKV,
                    PADDED_OUTPUTS=PADDED_OUTPUTS,
                    CALC_DP_PRE_LOAD_DELTA=CALC_DP_PRE_LOAD_DELTA,
                )
                # Increment pointers.
                start_m += BLOCK_M1
                offset_y += BLOCK_M1
            dk, dv = _attn_bwd_dqdkdv_single_tma(
                desc_q, desc_dq, desc_do, 
                dk, dv,  #
                k, v,  #
                LSE, D,  #
                start_m, offs_n,
                offset_y,
                # shared by Q/K/V/DO.
                N_CTX, 
                BLOCK_M=BLOCK_M1,
                QK_SCALE=QK_SCALE,
                DQ_ATOMIC=DQ_ATOMIC,
                Q_TRANSPOSED=Q_TRANSPOSED,
                KV_TRANSPOSED=KV_TRANSPOSED,
                SCORE_TRANSPOSED=SCORE_TRANSPOSED,
                DQ_TRANSPOSED=DQ_TRANSPOSED,
                IS_DIVISIBLE=IS_DIVISIBLE, 
                CHECK_BLOCK_BOUNDARY=True,
                MASK_VARIANT=MASK_VARIANT,
                PADDED_QKV=PADDED_QKV,
                PADDED_OUTPUTS=PADDED_OUTPUTS,
                CALC_DP_PRE_LOAD_DELTA=CALC_DP_PRE_LOAD_DELTA,
            )
    else:
        for blk_idx in range(num_steps):
            dk, dv = _attn_bwd_dqdkdv_single_tma(
                desc_q, desc_dq, desc_do, 
                dk, dv,  #
                k, v,  #
                LSE, D,  #
                start_m, offs_n,
                offset_y,
                # shared by Q/K/V/DO.
                N_CTX, 
                BLOCK_M=BLOCK_M1,
                QK_SCALE=QK_SCALE,
                DQ_ATOMIC=DQ_ATOMIC,
                Q_TRANSPOSED=Q_TRANSPOSED,
                KV_TRANSPOSED=KV_TRANSPOSED,
                SCORE_TRANSPOSED=SCORE_TRANSPOSED,
                DQ_TRANSPOSED=DQ_TRANSPOSED,
                IS_DIVISIBLE=IS_DIVISIBLE, 
                CHECK_BLOCK_BOUNDARY=False,
                MASK_VARIANT=MASK_VARIANT,
                PADDED_QKV=PADDED_QKV,
                PADDED_OUTPUTS=PADDED_OUTPUTS,
                CALC_DP_PRE_LOAD_DELTA=CALC_DP_PRE_LOAD_DELTA,
            )
            # Increment pointers.
            start_m += BLOCK_M1
            offset_y += BLOCK_M1
            # offs_m += BLOCK_M1

            # qT_ptrs += step_m * stride_tok
            # q_ptrs += BLOCK_M1 * stride_tok
            # do_ptrs += BLOCK_M1 * stride_tok
            # if DQ_ATOMIC:
            #     dq_ptrs += BLOCK_M1 * stride_tok
    return dk, dv

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_bwd_dqdkdv_tma_dynamic(desc_dq, dk, dv,  #
                   desc_q, k, v,  #
                   desc_do,  #
                   LSE, D,  #
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   N_CTX, 
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                   offset_y,
                   QK_SCALE: tl.constexpr,  #
                   BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   DQ_ATOMIC: tl.constexpr,
                   Q_TRANSPOSED: tl.constexpr,
                   KV_TRANSPOSED: tl.constexpr,
                   SCORE_TRANSPOSED: tl.constexpr,
                   DQ_TRANSPOSED: tl.constexpr,
                   IS_DIVISIBLE: tl.constexpr,
                   PADDED_QKV: tl.constexpr,
                   PADDED_OUTPUTS: tl.constexpr,
                   CALC_DP_PRE_LOAD_DELTA: tl.constexpr,
                   MASK_VARIANT: tl.constexpr):
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    # offset_y
    offset_y = offset_y + start_m
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    # NOTE: increment start_m and create offs_m is slightly faster than 
    # increment offs_m.
    for blk_idx in range(num_steps):
        dk, dv = _attn_bwd_dqdkdv_single_tma_dynamic(
            desc_q, desc_dq, desc_do, 
            dk, dv,  #
            k, v,  #
            LSE, D,  #
            start_m, offs_n,
            # shared by Q/K/V/DO.
            offset_y,
            N_CTX, 
            BLOCK_M=BLOCK_M1,
            QK_SCALE=QK_SCALE,
            DQ_ATOMIC=DQ_ATOMIC,
            Q_TRANSPOSED=Q_TRANSPOSED,
            KV_TRANSPOSED=KV_TRANSPOSED,
            SCORE_TRANSPOSED=SCORE_TRANSPOSED,
            DQ_TRANSPOSED=DQ_TRANSPOSED,
            IS_DIVISIBLE=IS_DIVISIBLE, 
            MASK_VARIANT=MASK_VARIANT,
            PADDED_QKV=PADDED_QKV,
            PADDED_OUTPUTS=PADDED_OUTPUTS,
            CALC_DP_PRE_LOAD_DELTA=CALC_DP_PRE_LOAD_DELTA,
        )
        # Increment pointers.
        start_m += BLOCK_M1
        offset_y += BLOCK_M1
        # offs_m += BLOCK_M1

        # qT_ptrs += step_m * stride_tok
        # q_ptrs += BLOCK_M1 * stride_tok
        # do_ptrs += BLOCK_M1 * stride_tok
        # if DQ_ATOMIC:
        #     dq_ptrs += BLOCK_M1 * stride_tok
    return dk, dv

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_bwd_dqdkdv_tma_dynamic_blockcausal(desc_dq, dk, dv,  #
                   desc_q, k, v,  #
                   desc_do,  #
                   LSE, D,  #
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   N_CTX, 
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                   offset_y,
                   block_causal,
                   QK_SCALE: tl.constexpr,  #
                   BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   DQ_ATOMIC: tl.constexpr,
                   Q_TRANSPOSED: tl.constexpr,
                   KV_TRANSPOSED: tl.constexpr,
                   SCORE_TRANSPOSED: tl.constexpr,
                   DQ_TRANSPOSED: tl.constexpr,
                   IS_DIVISIBLE: tl.constexpr,
                   PADDED_QKV: tl.constexpr,
                   PADDED_OUTPUTS: tl.constexpr,
                   CALC_DP_PRE_LOAD_DELTA: tl.constexpr,
                   MASK_VARIANT: tl.constexpr):
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    # offset_y
    offset_y = offset_y + start_m
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    # NOTE: increment start_m and create offs_m is slightly faster than 
    # increment offs_m.
    for blk_idx in range(num_steps):
        dk, dv = _attn_bwd_dqdkdv_single_tma_dynamic_blockcausal(
            desc_q, desc_dq, desc_do, 
            dk, dv,  #
            k, v,  #
            LSE, D,  #
            start_m, offs_n,
            # shared by Q/K/V/DO.
            offset_y,
            N_CTX, 
            block_causal,
            BLOCK_M=BLOCK_M1,
            QK_SCALE=QK_SCALE,
            DQ_ATOMIC=DQ_ATOMIC,
            Q_TRANSPOSED=Q_TRANSPOSED,
            KV_TRANSPOSED=KV_TRANSPOSED,
            SCORE_TRANSPOSED=SCORE_TRANSPOSED,
            DQ_TRANSPOSED=DQ_TRANSPOSED,
            IS_DIVISIBLE=IS_DIVISIBLE, 
            MASK_VARIANT=MASK_VARIANT,
            PADDED_QKV=PADDED_QKV,
            PADDED_OUTPUTS=PADDED_OUTPUTS,
            CALC_DP_PRE_LOAD_DELTA=CALC_DP_PRE_LOAD_DELTA,
        )
        # Increment pointers.
        start_m += BLOCK_M1
        offset_y += BLOCK_M1
        # offs_m += BLOCK_M1

        # qT_ptrs += step_m * stride_tok
        # q_ptrs += BLOCK_M1 * stride_tok
        # do_ptrs += BLOCK_M1 * stride_tok
        # if DQ_ATOMIC:
        #     dq_ptrs += BLOCK_M1 * stride_tok
    return dk, dv

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_bwd_dq_single(dq, q, k_ptrs, v_ptrs,  #
                 do, m, Di,
                 N_CTX,  #
                 offs_m, start_n,
                 # Filled in by the wrapper.
                 BLOCK_N: tl.constexpr,
                 Q_TRANSPOSED: tl.constexpr,
                 SCORE_TRANSPOSED: tl.constexpr,
                IS_DIVISIBLE: tl.constexpr,
                CHECK_BLOCK_BOUNDARY: tl.constexpr,
                MASK_VARIANT: tl.constexpr,
                PADDED_QKV: tl.constexpr):
    offs_n = start_n + tl.arange(0, BLOCK_N)
    if IS_DIVISIBLE or PADDED_QKV:
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
    else:
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

    if SCORE_TRANSPOSED:
        if not Q_TRANSPOSED:
            qk_or_qkT = tl.dot(k, tl.trans(q))
        else:
            qk_or_qkT = tl.dot(k, q)
    else:
        if not Q_TRANSPOSED:
            qk_or_qkT = tl.dot(q, tl.trans(k))
        else:
            qk_or_qkT = tl.dot(tl.trans(q), tl.trans(k))
    if CHECK_BLOCK_BOUNDARY:
        # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
        if SCORE_TRANSPOSED:
            qk_or_qkT = tl.where(offs_n[:, None] < N_CTX, qk_or_qkT, float("-inf"))
        else:
            qk_or_qkT = tl.where(offs_n[None, :] < N_CTX, qk_or_qkT, float("-inf"))

    if SCORE_TRANSPOSED:
        p_or_pT = tl.math.exp2(qk_or_qkT - m[None, :])
    else:
        p_or_pT = tl.math.exp2(qk_or_qkT - m[:, None])
    # Autoregressive masking.
    if MASK_VARIANT == "causal":
        if SCORE_TRANSPOSED:
            mask = (offs_m[None, :] >= offs_n[:, None])
        else:
            mask = (offs_m[:, None] >= offs_n[None, :])
        p_or_pT = tl.where(mask, p_or_pT, 0.0)
    # Compute dP and dS.
    if SCORE_TRANSPOSED:
        dp_or_dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        ds_or_dsT = (p_or_pT * (dp_or_dpT - Di[None, :])).to(tl.float16)
        dq = tl.dot(tl.trans(ds_or_dsT), k, dq)
    else:
        dp_or_dpT = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds_or_dsT = (p_or_pT * (dp_or_dpT - Di[:, None])).to(tl.float16)
        dq = tl.dot(ds_or_dsT, k, dq)
    return dq


@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_bwd_dq_v2(dq, q, K, V,  #
                 do, m, D,
                 # shared by Q/K/V/DO.
                 stride_tok, stride_d,  #
                 H, N_CTX,  #
                 start_m, start_n, num_steps,  #

                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 MASK_VARIANT: tl.constexpr,
                 Q_TRANSPOSED: tl.constexpr,
                 SCORE_TRANSPOSED: tl.constexpr,
                 IS_DIVISIBLE: tl.constexpr,
                 PADDED_QKV: tl.constexpr,
                 PADDED_OUTPUTS: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    # kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    k_ptrs = K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    v_ptrs = V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d

    # D (= delta) is pre-divided by ds_scale.
    if IS_DIVISIBLE or PADDED_QKV:
        Di = tl.load(D + offs_m)
    else:
        Di = tl.load(D + offs_m, mask=offs_m < N_CTX)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    if Q_TRANSPOSED:
        q2 = tl.trans(q)
    else:
        q2 = q
    cur_n = start_n
    if not IS_DIVISIBLE:
        if num_steps >= 1:
            for blk_idx in range(num_steps - 1):
                dq = _attn_bwd_dq_single(
                    dq, q2, k_ptrs, v_ptrs,  #
                    do, m, Di,
                    N_CTX,  #
                    offs_m, cur_n,
                    # Filled in by the wrapper.
                    BLOCK_N2,
                    Q_TRANSPOSED,
                    SCORE_TRANSPOSED,
                    IS_DIVISIBLE=True,  # IS_DIVISIBLE
                    CHECK_BLOCK_BOUNDARY=False,  # CHECK_BLOCK_BOUNDARY
                    MASK_VARIANT=MASK_VARIANT,
                    PADDED_QKV=PADDED_QKV,
                )
                # Increment pointers.
                k_ptrs += BLOCK_N2 * stride_tok
                v_ptrs += BLOCK_N2 * stride_tok
                cur_n += BLOCK_N2
            dq = _attn_bwd_dq_single(
                dq, q2, k_ptrs, v_ptrs,  #
                do, m, Di,
                N_CTX,  #
                offs_m, cur_n,
                # Filled in by the wrapper.
                BLOCK_N2,
                Q_TRANSPOSED,
                SCORE_TRANSPOSED,
                IS_DIVISIBLE=False,
                CHECK_BLOCK_BOUNDARY=True,
                MASK_VARIANT=MASK_VARIANT,
                PADDED_QKV=PADDED_QKV,
            )
    else:
        for blk_idx in range(num_steps):
            dq = _attn_bwd_dq_single(
                dq, q2, k_ptrs, v_ptrs,  #
                do, m, Di,
                N_CTX,  #
                offs_m, cur_n,
                # Filled in by the wrapper.
                BLOCK_N2,
                Q_TRANSPOSED,
                SCORE_TRANSPOSED,
                IS_DIVISIBLE=True,  # IS_DIVISIBLE
                CHECK_BLOCK_BOUNDARY=False,  # CHECK_BLOCK_BOUNDARY
                MASK_VARIANT=MASK_VARIANT,
                PADDED_QKV=PADDED_QKV,

            )
            # Increment pointers.
            k_ptrs += BLOCK_N2 * stride_tok
            v_ptrs += BLOCK_N2 * stride_tok
            cur_n += BLOCK_N2
    return dq

# @triton.jit
# @tritonstd.mark_triton_compilable(is_template=True)
# def memory_fence(acc):
#     tl.inline_asm_elementwise(
#         "mov $0, $0;", "=r,+f,memory", [acc], dtype=tl.float32, is_pure=False, pack=1
#     )

@triton.jit
@tritonstd.mark_triton_compilable(inline_run_env_fn=partial(_attn_fwd_kernel_test_fn, is_fwd=False),
    real_kwargs={"N_CTX": 30720, "HEAD_DIM": 128, "H": 16, "dtype": torch.float16},
    raw_fn=_torch_bench_fn_bwd,
)
def _attn_bwd(Q, K, V, #
              DO,  #
              DQ, DK, DV,  #
              M, D,
              # shared by Q/K/V/DO.
              stride_z_pad, stride_h_pad,
              stride_z, stride_h, stride_tok, stride_d,  #
              H, N_CTX,  #
              SM_SCALE: tl.constexpr,  #
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,
              MASK_VARIANT: tl.constexpr,
              DQ_ATOMIC: tl.constexpr,
              Q_TRANSPOSED: tl.constexpr,
              KV_TRANSPOSED: tl.constexpr,
              SCORE_TRANSPOSED: tl.constexpr,
              DQ_TRANSPOSED: tl.constexpr,
              IS_DIVISIBLE: tl.constexpr,
              PARALLEL_DQ: tl.constexpr,
              PADDED_QKV: tl.constexpr,
              PADDED_OUTPUTS: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)
    bhid = tl.program_id(2)
    off_chz = tl.cast(bhid * tl.cdiv(N_CTX, 128) * 128, tl.int64)
    adj_pad = tl.cast(stride_h_pad * (bhid % H) + stride_z_pad * (bhid // H), tl.int64)
    adj = tl.cast(stride_h * (bhid % H) + stride_z * (bhid // H), tl.int64)

    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj_pad
    K += adj_pad
    V += adj_pad
    DO += adj_pad
    if PARALLEL_DQ:
        DQ += adj
    else:
        DQ += adj_pad
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz
    offs_k = tl.arange(0, HEAD_DIM)

    num_kv_block = tl.cdiv(N_CTX, BLOCK_N1)
    if not PARALLEL_DQ or pid < num_kv_block:
    # if True:

        # load scales
        start_n = pid * BLOCK_N1

        # MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
        offs_n = start_n + tl.arange(0, BLOCK_N1)

        dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

        # load K and V: they stay in SRAM throughout the inner loop.
        # kT = tl.load(K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d)
        # vT = tl.load(V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d)

        # num_steps = N_CTX // BLOCK_M1
        if IS_DIVISIBLE or PADDED_QKV:
            if KV_TRANSPOSED:
                k = tl.load(K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d)
                v = tl.load(V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d)
            else:
                k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
                v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
        else:
            if KV_TRANSPOSED:
                k = tl.load(K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d, mask=offs_n[None, :] < N_CTX, other=0.0)
                v = tl.load(V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d, mask=offs_n[None, :] < N_CTX, other=0.0)
            else:
                k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=offs_n[:, None] < N_CTX, other=0.0)
                v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=offs_n[:, None] < N_CTX, other=0.0)
        if MASK_VARIANT == "causal":
            start_m = start_n
            num_steps = tl.cdiv(N_CTX - start_m, BLOCK_M1)
            # dk, dv = _attn_bwd_dqdkdv_dynamic(  #
            #     DQ, dk, dv,  #
            #     Q, k, v,  #
            #     DO,  #
            #     M, D,  #
            #     stride_tok, stride_d,  #
            #     N_CTX,  #
            #     start_n, start_m, num_steps,  #
            #     SM_SCALE,
            #     BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
            #     MASK_VARIANT=MASK_VARIANT,  #
            #     DQ_ATOMIC=DQ_ATOMIC,
            #     Q_TRANSPOSED=Q_TRANSPOSED,
            #     KV_TRANSPOSED=KV_TRANSPOSED,
            #     SCORE_TRANSPOSED=SCORE_TRANSPOSED,
            #     DQ_TRANSPOSED=DQ_TRANSPOSED,
            #     IS_DIVISIBLE=IS_DIVISIBLE,
            #     PADDED_QKV=PADDED_QKV,  #
            #     PADDED_OUTPUTS=PADDED_OUTPUTS,
            #     CALC_DP_PRE_LOAD_DELTA=True,
            # )

            MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR

            num_steps = BLOCK_N1 // MASK_BLOCK_M1
            # calc first block
            dk, dv = _attn_bwd_dqdkdv(  #
                DQ, dk, dv,  #
                Q, k, v,  #
                DO,  #
                M, D,  #
                stride_tok, stride_d,  #
                N_CTX,  #
                start_n, start_m, num_steps,  #
                SM_SCALE,
                MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
                MASK_VARIANT="causal",  #
                DQ_ATOMIC=DQ_ATOMIC,
                Q_TRANSPOSED=Q_TRANSPOSED,
                KV_TRANSPOSED=KV_TRANSPOSED,
                SCORE_TRANSPOSED=SCORE_TRANSPOSED,
                DQ_TRANSPOSED=DQ_TRANSPOSED,
                IS_DIVISIBLE=IS_DIVISIBLE,
                PADDED_QKV=PADDED_QKV,  #
                PADDED_OUTPUTS=PADDED_OUTPUTS,
                CALC_DP_PRE_LOAD_DELTA=True,
            )
            start_m += num_steps * MASK_BLOCK_M1
            num_steps = tl.cdiv(N_CTX - start_m, BLOCK_M1)
            dk, dv = _attn_bwd_dqdkdv(  #
                DQ, dk, dv,  #
                Q, k, v,  #
                DO,  #
                M, D,  #
                stride_tok, stride_d,  #
                N_CTX,  #
                start_n, start_m, num_steps,  #
                SM_SCALE,
                BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
                MASK_VARIANT="full",  #
                DQ_ATOMIC=DQ_ATOMIC,
                Q_TRANSPOSED=Q_TRANSPOSED,
                KV_TRANSPOSED=KV_TRANSPOSED,
                SCORE_TRANSPOSED=SCORE_TRANSPOSED,
                DQ_TRANSPOSED=DQ_TRANSPOSED,
                IS_DIVISIBLE=IS_DIVISIBLE,
                PADDED_QKV=PADDED_QKV,  #
                PADDED_OUTPUTS=PADDED_OUTPUTS,
                CALC_DP_PRE_LOAD_DELTA=True,
            )
        else:
            start_m = 0
            num_steps = tl.cdiv(N_CTX, BLOCK_M1)

            dk, dv = _attn_bwd_dqdkdv(  #
                DQ, dk, dv,  #
                Q, k, v,  #
                DO,  #
                M, D,  #
                stride_tok, stride_d,  #
                N_CTX,  #
                start_n, start_m, num_steps,  #
                SM_SCALE,
                BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
                MASK_VARIANT="full",  #
                DQ_ATOMIC=DQ_ATOMIC,
                Q_TRANSPOSED=Q_TRANSPOSED,
                KV_TRANSPOSED=KV_TRANSPOSED,
                SCORE_TRANSPOSED=SCORE_TRANSPOSED,
                DQ_TRANSPOSED=DQ_TRANSPOSED,
                IS_DIVISIBLE=IS_DIVISIBLE,
                PADDED_QKV=PADDED_QKV,  #
                PADDED_OUTPUTS=PADDED_OUTPUTS,
                CALC_DP_PRE_LOAD_DELTA=True,
            )
            
        dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
        if IS_DIVISIBLE or PADDED_OUTPUTS:
            tl.store(dv_ptrs, dv)
        else:
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < N_CTX)

        # Write back dK.
        dk *= SM_SCALE
        dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
        if IS_DIVISIBLE or PADDED_OUTPUTS:
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < N_CTX)
    else:
        off_pid = pid - num_kv_block
        start_m = off_pid * BLOCK_M2
        offs_m = start_m + tl.arange(0, BLOCK_M2)
        dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
        if IS_DIVISIBLE or PADDED_QKV:
            q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
            do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

            m = tl.load(M + offs_m)
        else:
            q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=offs_m[:, None] < N_CTX, other=0.0)
            do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=offs_m[:, None] < N_CTX, other=0.0)

            m = tl.load(M + offs_m, mask=offs_m < N_CTX, other=0.0)

        if MASK_VARIANT == "causal":
            MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR

            end_n = start_m + BLOCK_M2
            num_steps = BLOCK_M2 // MASK_BLOCK_N2
            dq = _attn_bwd_dq_v2(
                dq, q, K, V,  #
                do, m, D,  #
                stride_tok, stride_d,  #
                H, N_CTX,  #
                start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,  #
                BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,  #
                MASK_VARIANT="causal",
                Q_TRANSPOSED=DQ_TRANSPOSED,
                SCORE_TRANSPOSED=DQ_TRANSPOSED,
                IS_DIVISIBLE=IS_DIVISIBLE,  #
                PADDED_QKV=PADDED_QKV,  #
                PADDED_OUTPUTS=PADDED_OUTPUTS,
            )
            end_n -= num_steps * MASK_BLOCK_N2
            # stage 2
            num_steps = end_n // BLOCK_N2
            dq = _attn_bwd_dq_v2(
                dq, q, K, V,  #
                do, m, D,  #
                stride_tok, stride_d,  #
                H, N_CTX,  #
                start_m, end_n - num_steps * BLOCK_N2, num_steps,  #
                BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
                MASK_VARIANT="full",
                Q_TRANSPOSED=DQ_TRANSPOSED,
                SCORE_TRANSPOSED=DQ_TRANSPOSED,
                IS_DIVISIBLE=IS_DIVISIBLE,  #
                PADDED_QKV=PADDED_QKV,  #
                PADDED_OUTPUTS=PADDED_OUTPUTS,
            )

        else:
            num_steps = tl.cdiv(N_CTX, BLOCK_N2)
            dq = _attn_bwd_dq_v2(
                dq, q, K, V,  #
                do, m, D,  #
                stride_tok, stride_d,  #
                H, N_CTX,  #
                start_m, 0, num_steps,  #
                BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
                MASK_VARIANT="full",
                Q_TRANSPOSED=DQ_TRANSPOSED,
                SCORE_TRANSPOSED=DQ_TRANSPOSED,
                IS_DIVISIBLE=IS_DIVISIBLE,  #
                PADDED_QKV=PADDED_QKV,  #
                PADDED_OUTPUTS=PADDED_OUTPUTS,
            )
        # Write back dQ.
        dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
        dq *= LN2
        if IS_DIVISIBLE or PADDED_OUTPUTS:
            tl.store(dq_ptrs, dq)
        else:
            tl.store(dq_ptrs, dq, mask=offs_m[:, None] < N_CTX)

    if not PARALLEL_DQ and not DQ_ATOMIC:
        start_m = pid * BLOCK_M2

        offs_m = start_m + tl.arange(0, BLOCK_M2)
        dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
        if IS_DIVISIBLE:
            q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
            do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

            m = tl.load(M + offs_m)
        else:
            q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=offs_m[:, None] < N_CTX, other=0.0)
            do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=offs_m[:, None] < N_CTX, other=0.0)

            m = tl.load(M + offs_m, mask=offs_m < N_CTX, other=0.0)

        num_steps = tl.cdiv(N_CTX, BLOCK_N2)
        dq = _attn_bwd_dq_v2(
            dq, q, K, V,  #
            do, m, D,  #
            stride_tok, stride_d,  #
            H, N_CTX,  #
            BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
            start_m, 0, num_steps,  #
            MASK_VARIANT="full",
            Q_TRANSPOSED=DQ_TRANSPOSED,
            SCORE_TRANSPOSED=DQ_TRANSPOSED,
            IS_DIVISIBLE=IS_DIVISIBLE,  #
            PADDED_QKV=PADDED_QKV,  #
            PADDED_OUTPUTS=PADDED_OUTPUTS,
        )
        # Write back dQ.
        dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
        dq *= LN2
        if IS_DIVISIBLE:
            tl.store(dq_ptrs, dq)
        else:
            tl.store(dq_ptrs, dq, mask=offs_m[:, None] < N_CTX)

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_bwd_dq_bc_single(dq, q, k_ptrs, v_ptrs,  #
                 do, m, Di,
                 N_CTX,  #
                 offs_m, start_n,
                 block_causal,
                 # Filled in by the wrapper.
                 BLOCK_N: tl.constexpr,
                 Q_TRANSPOSED: tl.constexpr,
                 SCORE_TRANSPOSED: tl.constexpr,
                IS_DIVISIBLE: tl.constexpr,
                CHECK_BLOCK_BOUNDARY: tl.constexpr,
                MASK_VARIANT: tl.constexpr,
                PADDED_QKV: tl.constexpr):
    offs_n = start_n + tl.arange(0, BLOCK_N)
    if IS_DIVISIBLE or PADDED_QKV:
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
    else:
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

    if SCORE_TRANSPOSED:
        if not Q_TRANSPOSED:
            qk_or_qkT = tl.dot(k, tl.trans(q))
        else:
            qk_or_qkT = tl.dot(k, q)
    else:
        if not Q_TRANSPOSED:
            qk_or_qkT = tl.dot(q, tl.trans(k))
        else:
            qk_or_qkT = tl.dot(tl.trans(q), tl.trans(k))
    if CHECK_BLOCK_BOUNDARY:
        # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
        if SCORE_TRANSPOSED:
            qk_or_qkT = tl.where(offs_n[:, None] < N_CTX, qk_or_qkT, float("-inf"))
        else:
            qk_or_qkT = tl.where(offs_n[None, :] < N_CTX, qk_or_qkT, float("-inf"))

    if SCORE_TRANSPOSED:
        p_or_pT = tl.math.exp2(qk_or_qkT - m[None, :])
    else:
        p_or_pT = tl.math.exp2(qk_or_qkT - m[:, None])
    # Autoregressive masking.
    if MASK_VARIANT == "causal":
        if SCORE_TRANSPOSED:
            mask = ((offs_m[None, :] // block_causal) >= (offs_n[:, None] // block_causal))
        else:
            mask = ((offs_m[:, None] // block_causal) >= (offs_n[None, :] // block_causal))
        p_or_pT = tl.where(mask, p_or_pT, 0.0)
    # Compute dP and dS.
    if SCORE_TRANSPOSED:
        dp_or_dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        ds_or_dsT = (p_or_pT * (dp_or_dpT - Di[None, :])).to(q.dtype)
        dq = tl.dot(tl.trans(ds_or_dsT), k, dq)
    else:
        dp_or_dpT = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds_or_dsT = (p_or_pT * (dp_or_dpT - Di[:, None])).to(q.dtype)
        dq = tl.dot(ds_or_dsT, k, dq)
    return dq

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_bwd_dq_bc(dq, q, K, V,  #
                 do, m, D,
                 # shared by Q/K/V/DO.
                 stride_tok, stride_d,  #
                 H, N_CTX,  #
                 start_m, start_n, num_steps,  #
                 block_causal,
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 MASK_VARIANT: tl.constexpr,
                 Q_TRANSPOSED: tl.constexpr,
                 SCORE_TRANSPOSED: tl.constexpr,
                 IS_DIVISIBLE: tl.constexpr,
                 PADDED_QKV: tl.constexpr,
                 PADDED_OUTPUTS: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    # kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    k_ptrs = K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    v_ptrs = V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d

    # D (= delta) is pre-divided by ds_scale.
    if IS_DIVISIBLE or PADDED_QKV:
        Di = tl.load(D + offs_m)
    else:
        Di = tl.load(D + offs_m, mask=offs_m < N_CTX)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    if Q_TRANSPOSED:
        q2 = tl.trans(q)
    else:
        q2 = q
    cur_n = start_n
    if not IS_DIVISIBLE:
        if num_steps >= 1:
            for blk_idx in range(num_steps - 1):
                dq = _attn_bwd_dq_bc_single(
                    dq, q2, k_ptrs, v_ptrs,  #
                    do, m, Di,
                    N_CTX,  #
                    offs_m, cur_n,
                    block_causal,
                    # Filled in by the wrapper.
                    BLOCK_N2,
                    Q_TRANSPOSED,
                    SCORE_TRANSPOSED,
                    IS_DIVISIBLE=True,  # IS_DIVISIBLE
                    CHECK_BLOCK_BOUNDARY=False,  # CHECK_BLOCK_BOUNDARY
                    MASK_VARIANT=MASK_VARIANT,
                    PADDED_QKV=PADDED_QKV,
                )
                # Increment pointers.
                k_ptrs += BLOCK_N2 * stride_tok
                v_ptrs += BLOCK_N2 * stride_tok
                cur_n += BLOCK_N2
            dq = _attn_bwd_dq_bc_single(
                dq, q2, k_ptrs, v_ptrs,  #
                do, m, Di,
                N_CTX,  #
                offs_m, cur_n,
                block_causal,
                # Filled in by the wrapper.
                BLOCK_N2,
                Q_TRANSPOSED,
                SCORE_TRANSPOSED,
                IS_DIVISIBLE=False,
                CHECK_BLOCK_BOUNDARY=True,
                MASK_VARIANT=MASK_VARIANT,
                PADDED_QKV=PADDED_QKV,
            )
    else:
        for blk_idx in range(num_steps):
            dq = _attn_bwd_dq_bc_single(
                dq, q2, k_ptrs, v_ptrs,  #
                do, m, Di,
                N_CTX,  #
                offs_m, cur_n,
                block_causal,
                # Filled in by the wrapper.
                BLOCK_N2,
                Q_TRANSPOSED,
                SCORE_TRANSPOSED,
                IS_DIVISIBLE=True,  # IS_DIVISIBLE
                CHECK_BLOCK_BOUNDARY=False,  # CHECK_BLOCK_BOUNDARY
                MASK_VARIANT=MASK_VARIANT,
                PADDED_QKV=PADDED_QKV,

            )
            # Increment pointers.
            k_ptrs += BLOCK_N2 * stride_tok
            v_ptrs += BLOCK_N2 * stride_tok
            cur_n += BLOCK_N2
    return dq


@triton.jit
@tritonstd.mark_triton_compilable(inline_run_env_fn=partial(_attn_fwd_kernel_test_fn, is_fwd=False, block_causal=32),
    real_kwargs={"N_CTX": 30000, "HEAD_DIM": 128, "H": 16, "dtype": torch.float16, "block_causal": 3000},
    raw_fn=_torch_bench_fn_bwd_bc,
)
def _attn_bwd_blockcausal(Q, K, V, #
              DO,  #
              DQ, DK, DV,  #
              M, D,
              # shared by Q/K/V/DO.
              stride_z_pad, stride_h_pad,
              stride_z, stride_h, stride_tok, stride_d,  #
              H, N_CTX,  #
              block_causal,
              SM_SCALE: tl.constexpr,  #
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,
              MASK_VARIANT: tl.constexpr,
              DQ_ATOMIC: tl.constexpr,
              Q_TRANSPOSED: tl.constexpr,
              KV_TRANSPOSED: tl.constexpr,
              SCORE_TRANSPOSED: tl.constexpr,
              DQ_TRANSPOSED: tl.constexpr,
              IS_DIVISIBLE: tl.constexpr,
              PARALLEL_DQ: tl.constexpr,
              PADDED_QKV: tl.constexpr,
              PADDED_OUTPUTS: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)
    bhid = tl.program_id(2)
    off_chz = tl.cast(bhid * tl.cdiv(N_CTX, 128) * 128, tl.int64)
    adj_pad = tl.cast(stride_h_pad * (bhid % H) + stride_z_pad * (bhid // H), tl.int64)
    adj = tl.cast(stride_h * (bhid % H) + stride_z * (bhid // H), tl.int64)

    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj_pad
    K += adj_pad
    V += adj_pad
    DO += adj_pad
    if PARALLEL_DQ:
        DQ += adj
    else:
        DQ += adj_pad
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz
    offs_k = tl.arange(0, HEAD_DIM)
    num_kv_block = tl.cdiv(N_CTX, BLOCK_N1)
    if not PARALLEL_DQ or pid < num_kv_block:
        # load scales
        start_n = pid * BLOCK_N1

        # MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
        offs_n = start_n + tl.arange(0, BLOCK_N1)

        dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

        # load K and V: they stay in SRAM throughout the inner loop.
        # kT = tl.load(K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d)
        # vT = tl.load(V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d)

        # num_steps = N_CTX // BLOCK_M1
        if IS_DIVISIBLE or PADDED_QKV:
            if KV_TRANSPOSED:
                k = tl.load(K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d)
                v = tl.load(V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d)
            else:
                k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
                v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
        else:
            if KV_TRANSPOSED:
                k = tl.load(K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d, mask=offs_n[None, :] < N_CTX, other=0.0)
                v = tl.load(V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d, mask=offs_n[None, :] < N_CTX, other=0.0)
            else:
                k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=offs_n[:, None] < N_CTX, other=0.0)
                v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=offs_n[:, None] < N_CTX, other=0.0)
        
        if MASK_VARIANT == "causal":
            total_cnt = tl.cdiv(N_CTX, BLOCK_M1)

            bc_top_left = (start_n // block_causal) * block_causal
            bc_top_right = ((start_n + BLOCK_N1) // block_causal) * block_causal
            lo = bc_top_left // BLOCK_M1
            hi = min(tl.cdiv(bc_top_right, BLOCK_M1), total_cnt)
            start_m = lo * BLOCK_M1
            num_steps = hi - lo
            # MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR

            # num_steps = BLOCK_N1 // MASK_BLOCK_M1
            # calc first block
            dk, dv = _attn_bwd_dqdkdv_blockcausal(  #
                DQ, dk, dv,  #
                Q, k, v,  #
                DO,  #
                M, D,  #
                stride_tok, stride_d,  #
                N_CTX,  #
                start_n, start_m, num_steps,  #
                block_causal,
                SM_SCALE,
                BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
                MASK_VARIANT="causal",  #
                DQ_ATOMIC=DQ_ATOMIC,
                Q_TRANSPOSED=Q_TRANSPOSED,
                KV_TRANSPOSED=KV_TRANSPOSED,
                SCORE_TRANSPOSED=SCORE_TRANSPOSED,
                DQ_TRANSPOSED=DQ_TRANSPOSED,
                IS_DIVISIBLE=IS_DIVISIBLE,
                PADDED_QKV=PADDED_QKV,  #
                PADDED_OUTPUTS=PADDED_OUTPUTS,
                CALC_DP_PRE_LOAD_DELTA=True,
            )
            start_m = hi * BLOCK_M1
            num_steps = total_cnt - hi
            dk, dv = _attn_bwd_dqdkdv_blockcausal(  #
                DQ, dk, dv,  #
                Q, k, v,  #
                DO,  #
                M, D,  #
                stride_tok, stride_d,  #
                N_CTX,  #
                start_n, start_m, num_steps,  #
                block_causal,
                SM_SCALE,
                BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
                MASK_VARIANT="full",  #
                DQ_ATOMIC=DQ_ATOMIC,
                Q_TRANSPOSED=Q_TRANSPOSED,
                KV_TRANSPOSED=KV_TRANSPOSED,
                SCORE_TRANSPOSED=SCORE_TRANSPOSED,
                DQ_TRANSPOSED=DQ_TRANSPOSED,
                IS_DIVISIBLE=IS_DIVISIBLE,
                PADDED_QKV=PADDED_QKV,  #
                PADDED_OUTPUTS=PADDED_OUTPUTS,
                CALC_DP_PRE_LOAD_DELTA=True,
            )
        else:
            start_m = 0
            num_steps = tl.cdiv(N_CTX, BLOCK_M1)

            dk, dv = _attn_bwd_dqdkdv_blockcausal(  #
                DQ, dk, dv,  #
                Q, k, v,  #
                DO,  #
                M, D,  #
                stride_tok, stride_d,  #
                N_CTX,  #
                start_n, start_m, num_steps,  #
                block_causal,
                SM_SCALE,
                BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
                MASK_VARIANT="full",  #
                DQ_ATOMIC=DQ_ATOMIC,
                Q_TRANSPOSED=Q_TRANSPOSED,
                KV_TRANSPOSED=KV_TRANSPOSED,
                SCORE_TRANSPOSED=SCORE_TRANSPOSED,
                DQ_TRANSPOSED=DQ_TRANSPOSED,
                IS_DIVISIBLE=IS_DIVISIBLE,
                PADDED_QKV=PADDED_QKV,  #
                PADDED_OUTPUTS=PADDED_OUTPUTS,
                CALC_DP_PRE_LOAD_DELTA=True,
            )
            
        dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
        if IS_DIVISIBLE or PADDED_OUTPUTS:
            tl.store(dv_ptrs, dv)
        else:
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < N_CTX)

        # Write back dK.
        dk *= SM_SCALE
        dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
        if IS_DIVISIBLE or PADDED_OUTPUTS:
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < N_CTX)
    else:
        off_pid = pid - num_kv_block
        start_m = off_pid * BLOCK_M2
        offs_m = start_m + tl.arange(0, BLOCK_M2)
        dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
        if IS_DIVISIBLE or PADDED_QKV:
            q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
            do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

            m = tl.load(M + offs_m)
        else:
            q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=offs_m[:, None] < N_CTX, other=0.0)
            do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=offs_m[:, None] < N_CTX, other=0.0)

            m = tl.load(M + offs_m, mask=offs_m < N_CTX, other=0.0)

        bc_right_top = tl.cdiv((start_m), block_causal) * block_causal

        bc_right_down = tl.cdiv((start_m + BLOCK_M2), block_causal) * block_causal
        total_cnt = tl.cdiv(N_CTX, BLOCK_N2)

        if MASK_VARIANT == "causal":
            lo, hi = min(bc_right_top // BLOCK_N2, total_cnt), min(tl.cdiv(bc_right_down, BLOCK_N2), total_cnt)

            dq = _attn_bwd_dq_bc(
                dq, q, K, V,  #
                do, m, D,  #
                stride_tok, stride_d,  #
                H, N_CTX,  #
                start_m, lo * BLOCK_N2, hi - lo,  #
                block_causal,
                BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
                MASK_VARIANT="causal",
                Q_TRANSPOSED=DQ_TRANSPOSED,
                SCORE_TRANSPOSED=DQ_TRANSPOSED,
                IS_DIVISIBLE=IS_DIVISIBLE,  #
                PADDED_QKV=PADDED_QKV,  #
                PADDED_OUTPUTS=PADDED_OUTPUTS,
            )
            # stage 2
            dq = _attn_bwd_dq_bc(
                dq, q, K, V,  #
                do, m, D,  #
                stride_tok, stride_d,  #
                H, N_CTX,  #
                start_m, 0, lo,  #
                block_causal,
                BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
                MASK_VARIANT="full",
                Q_TRANSPOSED=DQ_TRANSPOSED,
                SCORE_TRANSPOSED=DQ_TRANSPOSED,
                IS_DIVISIBLE=IS_DIVISIBLE,  #
                PADDED_QKV=PADDED_QKV,  #
                PADDED_OUTPUTS=PADDED_OUTPUTS,
            )

        else:
            num_steps = tl.cdiv(N_CTX, BLOCK_N2)
            dq = _attn_bwd_dq_bc(
                dq, q, K, V,  #
                do, m, D,  #
                stride_tok, stride_d,  #
                H, N_CTX,  #
                BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
                start_m, 0, num_steps,  #
                block_causal,
                MASK_VARIANT="full",
                Q_TRANSPOSED=DQ_TRANSPOSED,
                SCORE_TRANSPOSED=DQ_TRANSPOSED,
                IS_DIVISIBLE=IS_DIVISIBLE,  #
                PADDED_QKV=PADDED_QKV,  #
                PADDED_OUTPUTS=PADDED_OUTPUTS,
            )
        # Write back dQ.
        # print(DQ, stride_tok, stride_d)
        dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
        dq *= LN2
        if IS_DIVISIBLE or PADDED_OUTPUTS:
            tl.store(dq_ptrs, dq)
        else:
            tl.store(dq_ptrs, dq, mask=offs_m[:, None] < N_CTX)


@triton.jit
@tritonstd.mark_triton_compilable(inline_run_env_fn=partial(_attn_fwd_kernel_test_fn, is_fwd=False, is_tma=True),
    real_kwargs={"N_CTX": 30720, "HEAD_DIM": 128, "H": 16, "dtype": torch.float16},
    raw_fn=_torch_bench_fn_bwd_xformer,
)
def _attn_bwd_tma(Q, K, V, #
              DO,  #
              DQ, DK, DV,  #
              M, D,
              # shared by Q/K/V/DO.
              stride_z_pad, stride_h_pad,
              stride_z, stride_h, stride_tok, stride_d,  #
              H, N_CTX,  #
              SM_SCALE: tl.constexpr,  #
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,
              MASK_VARIANT: tl.constexpr,
              DQ_ATOMIC: tl.constexpr,
              Q_TRANSPOSED: tl.constexpr,
              KV_TRANSPOSED: tl.constexpr,
              SCORE_TRANSPOSED: tl.constexpr,
              DQ_TRANSPOSED: tl.constexpr,
              IS_DIVISIBLE: tl.constexpr,
              PARALLEL_DQ: tl.constexpr,
              PADDED_QKV: tl.constexpr,
              PADDED_OUTPUTS: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)
    bhid = tl.program_id(2)
    off_chz = tl.cast(bhid * tl.cdiv(N_CTX, 128) * 128, tl.int64)
    adj_padded = tl.cast(stride_h_pad * (bhid % H) + stride_z_pad * (bhid // H), tl.int64)
    adj = tl.cast(stride_h * (bhid % H) + stride_z * (bhid // H), tl.int64)

    pid = tl.program_id(0)
    if PADDED_QKV:
        offset_y = bhid * tl.cdiv(N_CTX, 128) * 128
    else:
        offset_y = bhid * N_CTX


    # offset pointers for batch/head
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz
    offs_k = tl.arange(0, HEAD_DIM)
    # tma bwd don't support parallel dq because tensor desc changed
    start_n = pid * BLOCK_N1

    # MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    if isinstance(K, tl.tensor_descriptor):
        if KV_TRANSPOSED:
            k = K.load([start_n + offset_y, 0]).T
            v = V.load([start_n + offset_y, 0]).T
        else:
            k = K.load([start_n + offset_y, 0])
            v = V.load([start_n + offset_y, 0])
    else:
        K += adj_padded
        V += adj_padded

        if IS_DIVISIBLE or PADDED_QKV:
            if KV_TRANSPOSED:
                k = tl.load(K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d)
                v = tl.load(V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d)
            else:
                k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
                v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
        else:
            if KV_TRANSPOSED:
                k = tl.load(K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d, mask=offs_n[None, :] < N_CTX, other=0.0)
                v = tl.load(V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d, mask=offs_n[None, :] < N_CTX, other=0.0)
            else:
                k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=offs_n[:, None] < N_CTX, other=0.0)
                v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=offs_n[:, None] < N_CTX, other=0.0)

    if MASK_VARIANT == "causal":
        start_m = start_n
        num_steps = tl.cdiv(N_CTX - start_m, BLOCK_M1)
    else:
        start_m = 0
        num_steps = tl.cdiv(N_CTX, BLOCK_M1)
    dk, dv = _attn_bwd_dqdkdv_tma_dynamic(  #
        DQ, dk, dv,  #
        Q, k, v,  #
        DO,  #
        M, D,  #
        stride_tok, stride_d,  #
        N_CTX,  #
        start_n, start_m, num_steps,  #
        offset_y,
        SM_SCALE,
        BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
        MASK_VARIANT=MASK_VARIANT,  #
        DQ_ATOMIC=DQ_ATOMIC,
        Q_TRANSPOSED=Q_TRANSPOSED,
        KV_TRANSPOSED=KV_TRANSPOSED,
        SCORE_TRANSPOSED=SCORE_TRANSPOSED,
        DQ_TRANSPOSED=DQ_TRANSPOSED,
        IS_DIVISIBLE=IS_DIVISIBLE,
        PADDED_QKV=PADDED_QKV,  #
        PADDED_OUTPUTS=PADDED_OUTPUTS,
        CALC_DP_PRE_LOAD_DELTA=False,
    )    
    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    if IS_DIVISIBLE or PADDED_OUTPUTS:
        tl.store(dv_ptrs, dv)
    else:
        tl.store(dv_ptrs, dv, mask=offs_n[:, None] < N_CTX)

    # Write back dK.
    dk *= SM_SCALE
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    if IS_DIVISIBLE or PADDED_OUTPUTS:
        tl.store(dk_ptrs, dk)
    else:
        tl.store(dk_ptrs, dk, mask=offs_n[:, None] < N_CTX)

@triton.jit
@tritonstd.mark_triton_compilable(inline_run_env_fn=partial(_attn_fwd_kernel_test_fn, is_fwd=False, is_tma=True, block_causal=192, N_CTX=384),
    real_kwargs={"N_CTX": 30000, "HEAD_DIM": 128, "H": 16, "dtype": torch.float16, "block_causal": 3000},
    raw_fn=_torch_bench_fn_bwd_bc,
)
def _attn_bwd_tma_blockcausal(Q, K, V, #
              DO,  #
              DQ, DK, DV,  #
              M, D,
              # shared by Q/K/V/DO.
              stride_z_pad, stride_h_pad,
              stride_z, stride_h, stride_tok, stride_d,  #
              H, N_CTX,  #
              block_causal,
              SM_SCALE: tl.constexpr,  #
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,
              MASK_VARIANT: tl.constexpr,
              DQ_ATOMIC: tl.constexpr,
              Q_TRANSPOSED: tl.constexpr,
              KV_TRANSPOSED: tl.constexpr,
              SCORE_TRANSPOSED: tl.constexpr,
              DQ_TRANSPOSED: tl.constexpr,
              IS_DIVISIBLE: tl.constexpr,
              PARALLEL_DQ: tl.constexpr,
              PADDED_QKV: tl.constexpr,
              PADDED_OUTPUTS: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)
    bhid = tl.program_id(2)
    off_chz = tl.cast(bhid * tl.cdiv(N_CTX, 128) * 128, tl.int64)
    adj_padded = tl.cast(stride_h_pad * (bhid % H) + stride_z_pad * (bhid // H), tl.int64)
    adj = tl.cast(stride_h * (bhid % H) + stride_z * (bhid // H), tl.int64)

    pid = tl.program_id(0)
    if PADDED_QKV:
        offset_y = bhid * tl.cdiv(N_CTX, 128) * 128
    else:
        offset_y = bhid * N_CTX


    # offset pointers for batch/head
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz
    offs_k = tl.arange(0, HEAD_DIM)
    # tma bwd don't support parallel dq because tensor desc changed
    start_n = pid * BLOCK_N1

    # MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    if isinstance(K, tl.tensor_descriptor):
        if KV_TRANSPOSED:
            k = K.load([start_n + offset_y, 0]).T
            v = V.load([start_n + offset_y, 0]).T
        else:
            k = K.load([start_n + offset_y, 0])
            v = V.load([start_n + offset_y, 0])
    else:
        K += adj_padded
        V += adj_padded

        if IS_DIVISIBLE or PADDED_QKV:
            if KV_TRANSPOSED:
                k = tl.load(K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d)
                v = tl.load(V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d)
            else:
                k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
                v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
        else:
            if KV_TRANSPOSED:
                k = tl.load(K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d, mask=offs_n[None, :] < N_CTX, other=0.0)
                v = tl.load(V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d, mask=offs_n[None, :] < N_CTX, other=0.0)
            else:
                k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=offs_n[:, None] < N_CTX, other=0.0)
                v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=offs_n[:, None] < N_CTX, other=0.0)
    bc_top_elem = (start_n // block_causal) * block_causal

    if MASK_VARIANT == "causal":
        start_m = (bc_top_elem // BLOCK_M1) * BLOCK_M1
        num_steps = tl.cdiv(N_CTX - start_m, BLOCK_M1)
    else:
        start_m = 0
        num_steps = tl.cdiv(N_CTX, BLOCK_M1)
    dk, dv = _attn_bwd_dqdkdv_tma_dynamic_blockcausal(  #
        DQ, dk, dv,  #
        Q, k, v,  #
        DO,  #
        M, D,  #
        stride_tok, stride_d,  #
        N_CTX,  #
        start_n, start_m, num_steps,  #
        offset_y,
        block_causal,
        SM_SCALE,
        BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
        MASK_VARIANT=MASK_VARIANT,  #
        DQ_ATOMIC=DQ_ATOMIC,
        Q_TRANSPOSED=Q_TRANSPOSED,
        KV_TRANSPOSED=KV_TRANSPOSED,
        SCORE_TRANSPOSED=SCORE_TRANSPOSED,
        DQ_TRANSPOSED=DQ_TRANSPOSED,
        IS_DIVISIBLE=IS_DIVISIBLE,
        PADDED_QKV=PADDED_QKV,  #
        PADDED_OUTPUTS=PADDED_OUTPUTS,
        CALC_DP_PRE_LOAD_DELTA=True,
    )    
    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    if IS_DIVISIBLE or PADDED_OUTPUTS:
        tl.store(dv_ptrs, dv)
    else:
        tl.store(dv_ptrs, dv, mask=offs_n[:, None] < N_CTX)

    # Write back dK.
    dk *= SM_SCALE
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    if IS_DIVISIBLE or PADDED_OUTPUTS:
        tl.store(dk_ptrs, dk)
    else:
        tl.store(dk_ptrs, dk, mask=offs_n[:, None] < N_CTX)

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_fwd_inner_single_blockcausal(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    offs_m, offs_n, 
                    Q_LEN, KV_LEN,
                    block_causal,
                    SM_SCALE: tl.constexpr,  #
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, 
                    HEAD_DIM: tl.constexpr, 
                    IS_DIVISIBLE: tl.constexpr,
                    CHECK_BLOCK_BOUNDARY: tl.constexpr,
                    MASK_VARIANT: tl.constexpr):
    # -- compute qk ----
    if IS_DIVISIBLE:
        k = tl.load(K_block_ptr)
    else:
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option = "zero")
    qk = tl.dot(q, k) * SM_SCALE
    if CHECK_BLOCK_BOUNDARY:
        # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
        qk = tl.where(offs_n < KV_LEN, qk, float("-inf"))
    if MASK_VARIANT == "causal":
        # only support causal currently
        mask = (offs_m[:, None] // block_causal) >= (offs_n[None, :] // block_causal)
        qk = qk + tl.where(mask, 0.0, float("-inf"))
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    # in block causal, when block_causal > BLOCK_N, m_ij may contains -inf, so we must mask it out
    masked_out_rows = (m_ij == float("-inf"))
    m_ij = tl.where(masked_out_rows, 0.0, m_ij)

    # -- update m_i and l_i
    alpha = tl.math.exp2(m_i - m_ij)

    p = tl.math.exp2(qk - m_ij[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    # -- update output accumulator --
    acc = acc * alpha[:, None]
    # update acc
    if IS_DIVISIBLE:
        v = tl.load(V_block_ptr)
    else:
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option = "zero")
    acc = tl.dot(p.to(q.dtype), v, acc)
    # update m_i and l_i
    m_i = m_ij
    return acc, l_i, m_i

@triton.jit
@tritonstd.mark_triton_compilable(is_template=True)
def _attn_fwd_inner_blockcausal(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    offs_m, offs_n,  #
                    start, end,
                    Q_LEN, KV_LEN,
                    block_causal,
                    SM_SCALE: tl.constexpr,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    IS_DIVISIBLE: tl.constexpr,
                    MASK_VARIANT: tl.constexpr):
    # range of values handled by this stage
    # loop over k, v and update accumulator
    offs_n = offs_n + start * BLOCK_N
    for start_n in range(start, end):
        acc, l_i, m_i = _attn_fwd_inner_single_blockcausal(
            acc, l_i, m_i, q, K_block_ptr, V_block_ptr, 
            offs_m, offs_n, Q_LEN, KV_LEN, block_causal,
            SM_SCALE, BLOCK_M, BLOCK_N, HEAD_DIM, 
            IS_DIVISIBLE, CHECK_BLOCK_BOUNDARY=IS_DIVISIBLE, MASK_VARIANT=MASK_VARIANT
        )
        offs_n += BLOCK_N

        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i

@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
@tritonstd.mark_triton_compilable(inline_run_env_fn=partial(_attn_fwd_kernel_test_fn, is_tma=False, block_causal=192, N_CTX=384, H=1),
    real_kwargs={"N_CTX": 30000, "HEAD_DIM": 128, "H": 16, "dtype": torch.float16, "block_causal": 3000},
    raw_fn=_flex_bench_fn_fwd,
)
def _attn_fwd_blockcausal(Q, K, V, M, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, N_CTX,  #
              block_causal,
              SM_SCALE: tl.constexpr,
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              MASK_VARIANT: tl.constexpr,
              IS_DIVISIBLE: tl.constexpr,  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    tl.static_assert(BLOCK_M >= BLOCK_N)

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = tl.cast(off_z, tl.int64) * stride_qz + tl.cast(off_h, tl.int64) * stride_qh

    # block pointers
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    QK_SCALE: tl.constexpr = SM_SCALE * 1.44269504  # 1/log(2)
    Q_USE_BLOCK_PTR: tl.constexpr = True
    if Q_USE_BLOCK_PTR:
        # load q: it will stay in SRAM throughout
        # load q: it stays in SRAM throughout the inner loop.
        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        if IS_DIVISIBLE:
            q = tl.load(Q_block_ptr)
        else:
            # boundary check is not free, so we only do it when necessary.
            q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option = "zero")
    else:
        q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + tl.arange(0, HEAD_DIM)[None, :] * stride_qk
        if IS_DIVISIBLE:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    dtype = q.dtype
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    v_order: tl.constexpr = (1, 0)
    BLOCK_M_DIV_N: tl.constexpr = BLOCK_M // BLOCK_N
    total_cnt = tl.cdiv(N_CTX, BLOCK_N)
    # assume N_CTX is divisible by block_causal
    bc_right_top = tl.cdiv((start_m) * BLOCK_M, block_causal) * block_causal

    bc_right_down = tl.cdiv((start_m + 1) * BLOCK_M, block_causal) * block_causal

    # bc_right_elem = bc_right * block_causal
    # bc_left_elem = bc_left * block_causal

    if MASK_VARIANT == "causal":
        lo, hi = min(bc_right_top // BLOCK_N, total_cnt), min(tl.cdiv(bc_right_down, BLOCK_N), total_cnt)
        # print("BC-lohi", start_m, "!", lo, hi, "|", bc_right_top, bc_right_down)

        # barrier makes it easier for compielr to schedule the
        # two loops independently
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vk, stride_vn),
            offsets=(lo * BLOCK_N, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=v_order,
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset,
            shape=(HEAD_DIM, N_CTX),
            strides=(stride_kk, stride_kn),
            offsets=(0, lo * (BLOCK_N)),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(0, 1),
        )
        acc, l_i, m_i = _attn_fwd_inner_blockcausal(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        offs_m, offs_n, lo, hi,
                                         N_CTX, N_CTX, block_causal,
                                        QK_SCALE, BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        IS_DIVISIBLE,
                                        "causal")
    if MASK_VARIANT == "causal":
        lo, hi = 0, min(bc_right_top // BLOCK_N, total_cnt)
    else:
        lo, hi = 0, total_cnt
    # print("FULL-lohi", MASK_VARIANT, lo, hi)
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
    acc, l_i, m_i = _attn_fwd_inner_blockcausal(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                    offs_m, offs_n, lo, hi,
                                    N_CTX, N_CTX, block_causal,
                                    QK_SCALE, BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    IS_DIVISIBLE,
                                    "full")

    # [Note] Handle fully masked out rows:
    # Li will be the sum(e^(-inf)) == 0.0 for masked out rows, mi will be -inf.
    # We set Li to 1.0 which will result in lse/out = 0.0 | after the log(li) + mi(0.0) step
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    # epilogue
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    m_i += tl.log2(l_i)
    OUT_USE_BLOCK_PTR: tl.constexpr = False
    if OUT_USE_BLOCK_PTR:
        O_block_ptr = tl.make_block_ptr(
            base=Out + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        if IS_DIVISIBLE:
            tl.store(m_ptrs, m_i)
            tl.store(O_block_ptr, acc.to(dtype))
        else:
            tl.store(m_ptrs, m_i, mask=offs_m < N_CTX)
            tl.store(O_block_ptr, acc.to(dtype), boundary_check=(0,))
    else:
        out_ptrs = Out + qvk_offset + offs_m[:, None] * stride_om + tl.arange(0, HEAD_DIM)[None, :]
        if IS_DIVISIBLE:
            tl.store(m_ptrs, m_i)
            tl.store(out_ptrs, acc.to(dtype))
        else:
            tl.store(m_ptrs, m_i, mask=offs_m < N_CTX)
            tl.store(out_ptrs, acc.to(dtype), offs_m[:, None] < N_CTX)

def test_attn_fwd():
    runner = tritonstd.parse_triton_compilable_to_runner(_attn_fwd)
    asyncio.run(runner.validate_kernel_by_test_data(_attn_fwd, run_triton=False))
    asyncio.run(runner.bench_kernel_in_triton_process(_attn_fwd, run_cnt=10))

def test_attn_fwd_tma():
    runner = tritonstd.parse_triton_compilable_to_runner(_attn_fwd_tma)
    asyncio.run(runner.validate_kernel_by_test_data(_attn_fwd_tma, run_triton=False))
    asyncio.run(runner.bench_kernel_in_triton_process(_attn_fwd_tma, run_cnt=20))

def test_attn_fwd_tma_v2():
    runner = tritonstd.parse_triton_compilable_to_runner(_attn_fwd_tma_v2)
    asyncio.run(runner.validate_kernel_by_test_data(_attn_fwd_tma_v2, run_triton=True))
    asyncio.run(runner.bench_kernel_in_triton_process(_attn_fwd_tma_v2, run_cnt=20))

def test_attn_fwd_tma_bshd_v2():
    runner = tritonstd.parse_triton_compilable_to_runner(_attn_fwd_tma_bshd_v2)
    asyncio.run(runner.validate_kernel_by_test_data(_attn_fwd_tma_bshd_v2, run_triton=False))
    asyncio.run(runner.bench_kernel_in_triton_process(_attn_fwd_tma_bshd_v2, run_cnt=20))

def test_attn_fwd_tma_bc():
    runner = tritonstd.parse_triton_compilable_to_runner(_attn_fwd_tma_block_causal_v2)
    asyncio.run(runner.validate_kernel_by_test_data(_attn_fwd_tma_block_causal_v2, run_triton=True))
    # runner = tritonstd.parse_triton_compilable_to_runner(_attn_fwd_blockcausal)
    # asyncio.run(runner.validate_kernel_by_test_data(_attn_fwd_blockcausal, run_triton=True))

    # asyncio.run(runner.bench_kernel_in_triton_process(_attn_fwd_tma_block_causal_v2, run_cnt=20))

def test_attn_bwd():
    runner = tritonstd.parse_triton_compilable_to_runner(_attn_bwd)
    asyncio.run(runner.validate_kernel_by_test_data(_attn_bwd, run_triton=False))
    # for DKV_VARIANT in [3, 2, 1, 0]:
    #     for DQ_VARIANT in [0, 1]:
    # return
    kwargs_set: dict[str, dict[str, Any]] = {}
    q_tr_args = [True, False]
    kv_tr_args = [True, False]
    s_tr_args = [True, False]
    dq_tr_args = [False, True]
    # fast cfg
    q_tr_args = [False]
    kv_tr_args = [True]
    s_tr_args = [False]
    dq_tr_args = [True]


    for Q_TRANSPOSED in q_tr_args:
        q_name = "qT" if Q_TRANSPOSED else "q"
        for KV_TRANSPOSED in kv_tr_args:
            kv_name = "kTvT" if KV_TRANSPOSED else "kv"
            for SCORE_TRANSPOSED in s_tr_args:
                s_name = "sT" if SCORE_TRANSPOSED else "s"
                for DQ_TRANSPOSED in dq_tr_args:
                    dq_name = "dqT" if DQ_TRANSPOSED else "dq"
                    arg_name = "_".join([
                        q_name, kv_name, s_name, dq_name,
                    ])
                    kwargs_set[arg_name] = {
                        "Q_TRANSPOSED": Q_TRANSPOSED,
                        "KV_TRANSPOSED": KV_TRANSPOSED,
                        "SCORE_TRANSPOSED": SCORE_TRANSPOSED,
                        "DQ_TRANSPOSED": DQ_TRANSPOSED,
                    }

    asyncio.run(runner.bench_kernel_in_triton_process(_attn_bwd, override_kwargs_set=kwargs_set))

    # runner = tritonstd.parse_triton_compilable_to_runner(_attn_bwd_dq_only)
    # asyncio.run(runner.validate_kernel_by_test_data(_attn_bwd_dq_only))
    # asyncio.run(runner.bench_kernel_in_triton_process(_attn_bwd_dq_only,))
def test_attn_bwd_tma():
    runner = tritonstd.parse_triton_compilable_to_runner(_attn_bwd_tma)
    asyncio.run(runner.validate_kernel_by_test_data(_attn_bwd_tma, run_triton=False))
    # return
    kwargs_set: dict[str, dict[str, Any]] = {}
    q_tr_args = [True, False]
    kv_tr_args = [True, False]
    s_tr_args = [True, False]
    dq_tr_args = [True, False]
    # fast cfg
    # q_tr_args = [False]
    # kv_tr_args = [True]
    # s_tr_args = [False]
    # dq_tr_args = [True]


    for Q_TRANSPOSED in q_tr_args:
        q_name = "qT" if Q_TRANSPOSED else "q"
        for KV_TRANSPOSED in kv_tr_args:
            kv_name = "kTvT" if KV_TRANSPOSED else "kv"
            for SCORE_TRANSPOSED in s_tr_args:
                s_name = "sT" if SCORE_TRANSPOSED else "s"
                for DQ_TRANSPOSED in dq_tr_args:
                    dq_name = "dqT" if DQ_TRANSPOSED else "dq"
                    arg_name = "_".join([
                        q_name, kv_name, s_name, dq_name,
                    ])
                    kwargs_set[arg_name] = {
                        "Q_TRANSPOSED": Q_TRANSPOSED,
                        "KV_TRANSPOSED": KV_TRANSPOSED,
                        "SCORE_TRANSPOSED": SCORE_TRANSPOSED,
                        "DQ_TRANSPOSED": DQ_TRANSPOSED,
                    }

    asyncio.run(runner.bench_kernel_in_triton_process(_attn_bwd_tma, override_kwargs_set=kwargs_set))

def test_attn_bwd_bc():
    # runner = tritonstd.parse_triton_compilable_to_runner(_attn_bwd_tma_blockcausal)
    # asyncio.run(runner.validate_kernel_by_test_data(_attn_bwd_tma_blockcausal, run_triton=False))
    runner = tritonstd.parse_triton_compilable_to_runner(_attn_bwd_blockcausal)
    asyncio.run(runner.validate_kernel_by_test_data(_attn_bwd_blockcausal, run_triton=False))

    # for DKV_VARIANT in [3, 2, 1, 0]:
    #     for DQ_VARIANT in [0, 1]:
    # return
    kwargs_set: dict[str, dict[str, Any]] = {}
    q_tr_args = [True, False]
    kv_tr_args = [True, False]
    s_tr_args = [True, False]
    dq_tr_args = [False, True]
    # fast cfg
    # q_tr_args = [False]
    # kv_tr_args = [True]
    # s_tr_args = [False]
    # dq_tr_args = [True]


    for Q_TRANSPOSED in q_tr_args:
        q_name = "qT" if Q_TRANSPOSED else "q"
        for KV_TRANSPOSED in kv_tr_args:
            kv_name = "kTvT" if KV_TRANSPOSED else "kv"
            for SCORE_TRANSPOSED in s_tr_args:
                s_name = "sT" if SCORE_TRANSPOSED else "s"
                for DQ_TRANSPOSED in dq_tr_args:
                    dq_name = "dqT" if DQ_TRANSPOSED else "dq"
                    arg_name = "_".join([
                        q_name, kv_name, s_name, dq_name,
                    ])
                    kwargs_set[arg_name] = {
                        "Q_TRANSPOSED": Q_TRANSPOSED,
                        "KV_TRANSPOSED": KV_TRANSPOSED,
                        "SCORE_TRANSPOSED": SCORE_TRANSPOSED,
                        "DQ_TRANSPOSED": DQ_TRANSPOSED,
                    }

    asyncio.run(runner.bench_kernel_in_triton_process(_attn_bwd_blockcausal, override_kwargs_set=kwargs_set))


def _main():
    # test_attn_fwd()
    # test_attn_fwd_tma()

    # test_attn_fwd_tma_v2()
    # test_attn_fwd_tma_bc()
    test_attn_bwd()
    # test_attn_bwd_tma()
    # test_attn_fwd_tma_bshd_v2()
    # test_attn_bwd_bc()
if __name__ == "__main__":
    # mask = _prepare_blockwise_causal_attn_mask("cuda", 16, 4)
    # print(mask)
    _main()
