import dataclasses 
import inspect
import math
import random
import struct
from typing import Any, Union

import rich
from tensorpc.core.annolib import Undefined, undefined
from tensorpc.core.pfl.parser import metaeval_total_tree, unparse_expr
from tensorpc.core.pfl.pfl_ast import PFLExpr
from tensorpc.core.pfl import register_pfl_std, register_meta_infer
import numpy as np 
from tensorpc.core.pfl import PFLExprInfo, parse_func_to_df_ast, pfl_ast_to_dict, pfl_ast_dump, parse_expr_to_df_ast, consteval_expr, walk


@register_pfl_std(mapped_name="NdArray", mapped=np.ndarray, backend="js")
@dataclasses.dataclass
class NdArray:
    shape: list[int]
    dtype: int
    ndim: int
    def __getitem__(self, key: int) -> np.ndarray: ...
    def tolist(self) -> list[Any]: ...
    def size(self) -> int: ...

    def __add__(self, other: Union[np.ndarray, int]) -> np.ndarray: ...

@register_pfl_std(mapped_name="Numpy", mapped=np, backend="js")
@dataclasses.dataclass
class Numpy:
    float32: int = 0
    float64: int = 4
    int8: int = 3
    int16: int = 2
    int32: int = 1
    int64: int = 8
    uint8: int = 6
    uint16: int = 9
    uint32: int = 10
    uint64: int = 11
    bool_: int = 5
    
    @staticmethod
    def array(data: list[Any]) -> np.ndarray: 
        return np.array(data)

    @staticmethod
    def zeros(shape: list[int], dtype: int) -> np.ndarray: 
        return np.zeros(shape, dtype=_JS_DTYPE_TO_NP[dtype])


@register_meta_infer(Numpy.array)
def __array_infer(data: PFLExprInfo) -> Any:
    res = np.array(data.metadata_checked)
    return {
        "shape": res.shape,
        "dtype": res.dtype,
    }  

@register_meta_infer(NdArray.__add__)
def __add_infer(this: PFLExprInfo, data: PFLExprInfo) -> Any:
    if this.has_metadata():
        if not data.has_metadata() and data == this:
            return undefined
        shape_this = this.metadata_checked["shape"]
        if data == this:
            shape_other = data.metadata_checked["shape"]
        else:
            shape_other = []
        shape_res = np.broadcast_shapes(shape_this, shape_other)
        return {
            "shape": shape_res,
            "dtype": this.metadata_checked["dtype"],
        }  
    return undefined

@register_meta_infer(Numpy.zeros)
def __zeros_infer(shape: PFLExprInfo, dtype: PFLExprInfo) -> Any:
    if shape.has_metadata() and dtype.has_metadata():
        return {
            "shape": shape.metadata_checked,
            "dtype": dtype.metadata_checked,
        }  
    return undefined


_JS_DTYPE_TO_NP = {
    0: np.float32,
    4: np.float64,
    3: np.int8,
    2: np.int16,
    1: np.int32,
    8: np.int64,
    6: np.uint8,
    9: np.uint16,
    10: np.uint32,
    11: np.uint64,
    5: np.bool_,
}


def func2(a: np.ndarray, b: float, e: int):
    c = a + b
    d = np.zeros([e, 1], np.float32)
    f = a + d

if __name__ == "__main__":
    # pflast, run_scope = parse_expr_to_df_ast("Math().sin(2)")
    # rich.print(pfl_ast_dump(pflast))
    # print(consteval_expr(pflast))
    # typing.get_type_hints
    ast, _ = parse_func_to_df_ast(func2)
    # rich.print(pfl_ast_dump(ast))

    metaeval_total_tree(ast, {
        "a": {
            "shape": [1, 3],
            "dtype": 0,  # float32
        },
        "e": 5
    })
    for n in walk(ast):
        if isinstance(n, PFLExpr):
            print(f"\"{unparse_expr(n)}\"", n.st.metadata)
    # with open("build/test.json", "w") as f:
    #     f.write(json.dumps(pfl_ast_to_dict(ast), indent=2))