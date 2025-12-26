
import ast
import dataclasses
from typing import Any
from tensorpc.apps.ttfs.aggtype import TRITON_AGG_FLATTEN_META_FIELD, TRITON_AGG_META_FIELD, constexpr_function
from tensorpc.apps.ttfs.utils import div_up
import triton.language as tl
import triton 
import traceback
from triton.experimental.gluon import language as gl
import tensorpc.apps.ttfs as ttfs
from tensorpc.apps.ttfs.jitx import _create_agg_from_flatten_args_ast_node, triton_jitx
@ttfs.aggregate
class TensorManager(ttfs.TensorManagerBase):
    a: ttfs.TensorDesc
    b: ttfs.TensorDesc

    M: gl.tensor
    N: gl.tensor 

@ttfs.aggregate
class TensorManagerTriton(ttfs.TensorManagerBase):
    a: ttfs.TensorDesc
    b: ttfs.TensorDesc

    M: tl.tensor
    N: tl.tensor 

@ttfs.aggregate
class A:
    a: gl.tensor

@ttfs.aggregate
class B:
    b: A

print(getattr(B, TRITON_AGG_FLATTEN_META_FIELD))
print(getattr(A, TRITON_AGG_FLATTEN_META_FIELD))

@constexpr_function
def _get_field_type(dcls, field):
    field = tl.core._unwrap_if_constexpr(field)
    agg_fields = getattr(dcls, TRITON_AGG_META_FIELD)
    return agg_fields[field].type

@ttfs.gluon_jit_kernel
def kernel_copy(a_ptr, b_ptr, M, N, stride_am: gl.constexpr, stride_an: gl.constexpr, stride_bn: gl.constexpr, stride_bm: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
    tm = TensorManager(
        a=ttfs.TensorDesc(
            a_ptr, 
            "M,N",
            (stride_am, stride_an),

        ),
        b=ttfs.TensorDesc(
            b_ptr, 
            "M,N",
            (stride_bm, stride_bn),

        ),
        M=M,
        N=N,
    )
    a_dev = B(b=_get_field_type(B, "b")(5))
    idx_m = gl.program_id(1)
    idx_n = gl.program_id(0)
    new_tm = tm.offset_all_desc("M,N", (idx_m * BLOCK_M, idx_n * BLOCK_N))
    a_blocked_desc = new_tm.get_block_desc("a", "M,N", (BLOCK_M, BLOCK_N))
    a_io = ttfs.create_io_iter(a_blocked_desc, "M")
    b_blocked_desc = new_tm.get_block_desc("b", "M,N", (BLOCK_M, BLOCK_N))
    b_io = ttfs.create_io_iter(b_blocked_desc, "M")
    a = a_io.load("a")
    b_io.store("b", a)

@ttfs.gluon_jit_kernel
def kernel_copy_b_tr(a_ptr, b_ptr, M, N, stride_am: gl.constexpr, stride_an: gl.constexpr, stride_bn: gl.constexpr, stride_bm: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
    tm = TensorManager(
        a=ttfs.TensorDesc(
            a_ptr, 
            "M,N",
            (stride_am, stride_an),

        ),
        b=ttfs.TensorDesc(
            b_ptr, 
            "M,N",
            (stride_bm, stride_bn),

        ),
        M=M,
        N=N,
    )
    idx_m = gl.program_id(1)
    idx_n = gl.program_id(0)
    new_tm = tm.offset_all_desc_ptr("M,N", (idx_m * BLOCK_M, idx_n * BLOCK_N))
    a_blocked_desc = new_tm.get_block_desc("a", "N,M", (BLOCK_N, BLOCK_M))
    b_blocked_desc = new_tm.get_block_desc("b", "N,M", (BLOCK_M, BLOCK_N))

    a_io = ttfs.create_io_iter(a_blocked_desc, "M")
    b_io = ttfs.create_io_iter(b_blocked_desc, "M")
    a = a_io.load("a")
    b_io.store("b", a)

@ttfs.triton_jit_kernel
def tt_kernel_copy_b_tr(a_ptr, b_ptr, M, N, stride_am: gl.constexpr, stride_an: gl.constexpr, stride_bn: gl.constexpr, stride_bm: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
    tm = TensorManagerTriton(
        a=ttfs.TensorDesc(
            a_ptr, 
            "M,N",

        ),
        b=ttfs.TensorDesc(
            b_ptr, 
            "N,M"
        ),
        M=M,
        N=N,
    )
    idx_m = tl.program_id(1)
    idx_n = tl.program_id(0)
    new_tm = tm.offset_all_desc_ptr("M,N", (idx_m * BLOCK_M, idx_n * BLOCK_N))
    a_blocked_desc = new_tm.get_block_desc("a", "N,M", (BLOCK_N, BLOCK_M))
    b_blocked_desc = new_tm.get_block_desc("b", "N,M", (BLOCK_M, BLOCK_N))

    a_io = ttfs.create_io_iter(a_blocked_desc, "M")
    b_io = ttfs.create_io_iter(b_blocked_desc, "M")
    a = a_io.load("a")
    b_io.store("b", a)

def test_kernel_copy():
    import torch
    M = 4096 
    N = 2048 
    a = torch.rand((M, N)).cuda().to(torch.float32)
    b = torch.zeros((M, N)).cuda().to(torch.float32)

    block_size = 128

    grid = (div_up(N, block_size), div_up(M, block_size))

    ck = kernel_copy[grid](a_ptr=a, b_ptr=b, 
        M=M, N=N,
        stride_am=N, stride_an=1,
        stride_bm=N, stride_bn=1,
        BLOCK_M=block_size, BLOCK_N=block_size
    )
    # # print(b)
    torch.testing.assert_close(a, b)

def test_kernel_copy_tr():
    import torch
    M = 4096 
    N = 2048 
    a = torch.rand((M, N)).cuda().to(torch.float32)
    b = torch.zeros((N, M)).cuda().to(torch.float32)

    block_size = 128

    grid = (div_up(N, block_size), div_up(M, block_size))

    ck = kernel_copy[grid](a_ptr=a, b_ptr=b, 
        M=M, N=N,
        stride_am=N, stride_an=1,
        stride_bn=b.stride(0), stride_bm=b.stride(1),
        BLOCK_M=block_size, BLOCK_N=block_size
    )
    # print(ck.asm["ttgir"])
    torch.testing.assert_close(a.T, b)
    b = torch.zeros((N, M)).cuda().to(torch.float32)
    tt_ck = tt_kernel_copy_b_tr[grid](a_ptr=a, b_ptr=b, 
        M=M, N=N,
        stride_am=N, stride_an=1,
        stride_bn=b.stride(0), stride_bm=b.stride(1),
        BLOCK_M=block_size, BLOCK_N=block_size
    )
    torch.testing.assert_close(a.T, b)

@ttfs.gluon_jit_kernel
def kernel(a_ptr, b_ptr, M, N, stride_am: gl.constexpr, stride_an: gl.constexpr, stride_bn: gl.constexpr, stride_bm: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
    tm = TensorManager(
        a=ttfs.TensorDesc(
            a_ptr, 
            "M,N",
            (stride_am, stride_an),

        ),
        b=ttfs.TensorDesc(
            b_ptr, 
            "N,M",
            (stride_bn, stride_bm),

        ),
        M=M,
        N=N,
    )
    idx_m = gl.program_id(0)
    idx_n = gl.program_id(1)
    a_offseted_ptr = tm.get_offseted_ptr("a", "M,N", (idx_m * BLOCK_M, idx_n * BLOCK_N))
    a_blocked = tm.get_block_desc_by_ptr(a_offseted_ptr, "a", "N,M", (BLOCK_N, BLOCK_M))
    b_offseted_ptr = tm.get_offseted_ptr("b", "N,M", (idx_n * BLOCK_N, idx_m * BLOCK_M))
    b_blocked = tm.get_block_desc_by_ptr(b_offseted_ptr, "b", "N,M", (BLOCK_N, BLOCK_M))
    gl.static_print(ttfs.mp.get_all_field_keys_with_type(tm, ttfs.TensorDesc))
    gl.static_print(a_blocked)
    all_fields_tuple = ttfs.mp.get_all_fields_with_type_gluon(tm, (ttfs.TensorDesc))
    gl.static_print(all_fields_tuple)

    tm = ttfs.mp.replace_all_fields_with_type_gluon(tm, ttfs.TensorDesc, all_fields_tuple)

    new_tm = tm.offset_all_desc_ptr("M,N", (idx_m * BLOCK_M, idx_n * BLOCK_N))


def main():
    import torch
    M = 256 
    N = 128 
    a = torch.zeros((M, N)).cuda().to(torch.float32)
    b = torch.zeros((N, M)).cuda().to(torch.float32)

    block_size = 128

    grid = (div_up(N, block_size), div_up(M, block_size))

    kernel[grid](a_ptr=a, b_ptr=b, 
        M=M, N=N,
        stride_am=N, stride_an=1,
        stride_bn=1, stride_bm=M,
        BLOCK_M=block_size, BLOCK_N=block_size
    )
    print(b)

if __name__ == "__main__":
    # test_kernel_copy()
    # test_kernel_copy_tr()
    print(ast.unparse(_create_agg_from_flatten_args_ast_node(B, "B",)))