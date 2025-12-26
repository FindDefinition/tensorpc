
import ast
import dataclasses
from typing import Annotated, Any
from tensorpc.apps.ttfs.aggtype import TRITON_AGG_FLATTEN_META_FIELD, TRITON_AGG_META_FIELD, constexpr_function
from tensorpc.apps.ttfs.tensor.base import TensorDescFieldAccessor
from tensorpc.apps.ttfs.utils import div_up
import triton.language as tl
import triton 
import traceback
from triton.experimental.gluon import language as gl
import tensorpc.apps.ttfs as ttfs
from tensorpc.apps.ttfs.jitx import _create_agg_from_flatten_args_ast_node, triton_jitx

@ttfs.aggregate
class NestedArgs:
    a: Any
    c: gl.constexpr

@ttfs.aggregate
class Args:
    a: Any
    b: NestedArgs
    c: gl.constexpr
    desc: Annotated[ttfs.TensorDesc, ttfs.tensor_desc_meta("B,H")]
    # no annotated meta, all strides calculated at runtime
    desc2: ttfs.TensorDesc


@triton_jitx
def kernel(args: Args, BLOCK_M: gl.constexpr):
    tl.static_print(args)


def main():
    import torch
    a = torch.randn(16, 16).cuda()
    aa =  torch.randn(16, 16).cuda()
    b = torch.randn(1, 1024, 12, 128).cuda()
    tdesc = ttfs.TensorDesc(
        b,
        "B,S,H,D"
    )
    desc2_ten = torch.randn(1, 1024, 12, 128).cuda()
    desc2 = ttfs.TensorDesc(
        desc2_ten,
        "B,S,H,D"
    )
    args = Args(
        a=a,
        b=NestedArgs(
            a=aa,
            c=42,
        ),
        c=7,
        desc=tdesc,
        desc2=desc2,
    )

    kernel[(1,)](args=args, BLOCK_M=8)

if __name__ == "__main__":
    main()
    # # test_kernel_copy_tr()
    # print(ast.unparse(_create_agg_from_flatten_args_ast_node(B, "B",)))