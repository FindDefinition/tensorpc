from typing import Any, Optional, Union

import rich
import numpy as np 
from tensorpc.core import pfl
from tensorpc.apps.ppcl.std import Tensor, ppcl
from tensorpc.apps.ppcl.core import TensorMeta, DTypeEnum

def func(a: Tensor, b: float, e: int):
    c = a + b
    d = ppcl.zeros([e, 1], ppcl.float32)
    f = a + d
    print(d.T, d.ndim)
    print(d[0], d[..., 0], d[None, None, :2])

if __name__ == "__main__":
    ast = pfl.parse_func_to_pfl_ast(func, backend="ppcl", parse_cfg=pfl.PFLParseConfig(allow_slice=True, allow_nd_slice=True))

    pfl.metaeval_total_tree(ast, {
        "a": TensorMeta([1, 3], DTypeEnum.float32),
        "e": 5
    }, backend="ppcl")
    for n in pfl.walk(ast):
        if isinstance(n, pfl.PFLExpr):
            print(f"\"{pfl.unparse_pfl_expr(n)}\"", n.st.metadata)
