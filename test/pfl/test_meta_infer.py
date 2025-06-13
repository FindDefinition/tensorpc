from collections.abc import Sequence
import dataclasses 
import inspect
import math
import random
import struct
from typing import Any, Optional, Union

import rich
from tensorpc.core.annolib import Undefined, undefined
import numpy as np 
from tensorpc.core import pfl
from typing_extensions import Self

@dataclasses.dataclass
class TensorMeta:
    shape: list[int]
    dtype: int


@pfl.register_pfl_std(mapped_name="NdArray", mapped=np.ndarray, backend="js")
@dataclasses.dataclass
class NdArray:
    shape: list[int]
    dtype: int
    @property 
    def ndim(self) -> int: 
        return len(self.shape)
    @property 
    def T(self) -> Self: ...
    def __getitem__(self, key: Any) -> Self: ...
    def tolist(self) -> list[Any]: ...
    def size(self) -> int: ...

    def __add__(self, other: Union[Self, int, float]) -> Self: ...
    def __sub__(self, other: Union[Self, int, float]) -> Self: ...
    def __mul__(self, other: Union[Self, int, float]) -> Self: ...
    def __truediv__(self, other: Union[Self, int, float]) -> Self: ...
    def __floordiv__(self, other: Union[Self, int, float]) -> Self: ...

@pfl.mark_meta_infer(NdArray.__getitem__)
def __getitem_infer(data: pfl.PFLExprInfo, slice_items: Union[tuple[pfl.PFLExprInfo, ...], pfl.PFLExprInfo]) -> Optional[pfl.PFLMetaInferResult]:
    if not data.has_metadata():
        return None 
    data_meta = data.get_metadata_checked(TensorMeta)
    if isinstance(slice_items, pfl.PFLExprInfo):
        if slice_items.type == pfl.PFLExprType.NONE_TYPE:
            new_meta = TensorMeta([1] + data_meta.shape, data_meta.dtype)
        elif slice_items.type == pfl.PFLExprType.NUMBER:
            assert len(data_meta.shape) > 0, "Cannot slice an empty tensor"
            new_meta = TensorMeta(data_meta.shape[1:], data_meta.dtype)
        else:
            raise NotImplementedError(f"Unsupported slice type: {slice_items.type}")
        return pfl.PFLMetaInferResult(new_meta)
    else:
        # from pytorch
        dim = 0
        specified_dims = 0
        for item in slice_items:
            if item.type == pfl.PFLExprType.NONE_TYPE or item.type == pfl.PFLExprType.ELLIPSIS:
                specified_dims += 1
        res_shape = data_meta.shape.copy()
        for item in slice_items:
            if item.type == pfl.PFLExprType.ELLIPSIS:
                dim += len(data_meta.shape) - specified_dims
            elif item.type == pfl.PFLExprType.SLICE:
                if not item.has_metadata():
                    return None 
                slice_obj = item.get_metadata_checked(slice)
                start = 0 if slice_obj.start is None else slice_obj.start
                stop = data_meta.shape[dim] if slice_obj.stop is None else slice_obj.stop
                step = 1 if slice_obj.step is None else slice_obj.step
                step_abs = abs(step)
                if (start < 0):
                    start += data_meta.shape[dim]
                if (stop < 0):
                    stop += data_meta.shape[dim]
                length = stop - start 
                res_dim = (length + step_abs - 1) // step_abs
                res_shape[dim] = res_dim
                dim += 1
            elif item.type == pfl.PFLExprType.NUMBER:
                res_shape.pop(dim)
            elif item.type == pfl.PFLExprType.NONE_TYPE:
                res_shape.insert(dim, 1)
                dim += 1
            else:
                raise NotImplementedError(f"Unsupported slice type: {item.type}")
        return pfl.PFLMetaInferResult(TensorMeta(res_shape, data_meta.dtype))

@pfl.mark_meta_infer(NdArray.T)
def __T_infer(data: pfl.PFLExprInfo) -> Optional[pfl.PFLMetaInferResult]:
    if data.has_metadata():
        meta = data.get_metadata_checked(TensorMeta)
        shape = meta.shape
        assert len(shape) == 2
        return pfl.PFLMetaInferResult(TensorMeta(
            shape=shape[::-1],
            dtype=meta.dtype,
        ))

@pfl.mark_meta_infer(NdArray.__add__)
@pfl.mark_meta_infer(NdArray.__sub__)
@pfl.mark_meta_infer(NdArray.__mul__)
@pfl.mark_meta_infer(NdArray.__truediv__)
@pfl.mark_meta_infer(NdArray.__floordiv__)
def __bin_op_infer(this: pfl.PFLExprInfo, data: pfl.PFLExprInfo) -> Optional[pfl.PFLMetaInferResult]:
    if this.has_metadata():
        if not data.has_metadata() and data == this:
            return
        this_meta = this.get_metadata_checked(TensorMeta)
        shape_this = this_meta.shape
        if data == this:
            data_meta = data.get_metadata_checked(TensorMeta)
            shape_other = data_meta.shape
        else:
            assert data.type == pfl.PFLExprType.NUMBER or data.type == pfl.PFLExprType.BOOL
            shape_other = []
        shape_res = np.broadcast_shapes(shape_this, shape_other)
        res_meta = TensorMeta(
            shape=list(shape_res),
            dtype=this_meta.dtype,
        )
        return pfl.PFLMetaInferResult(res_meta)

@pfl.register_pfl_std(mapped_name="Numpy", mapped=np, backend="js")
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


@pfl.mark_meta_infer(Numpy.array)
def __array_infer(data: pfl.PFLExprInfo) -> Optional[pfl.PFLMetaInferResult]:
    res = np.array(data.metadata_checked)
    return pfl.PFLMetaInferResult(TensorMeta(
        shape=list(res.shape),
        dtype=NP_DTYPE_TO_JS[res.dtype],
    ))


@pfl.mark_meta_infer(Numpy.zeros)
def __zeros_infer(shape: pfl.PFLExprInfo, dtype: pfl.PFLExprInfo) -> Any:
    if shape.has_metadata() and dtype.has_metadata():
        res_meta = TensorMeta(shape.metadata_checked, dtype.metadata_checked)
        return pfl.PFLMetaInferResult(res_meta)


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

NP_DTYPE_TO_JS = {
    np.float32: 0,
    np.float64: 4,
    np.int8: 3,
    np.int16: 2,
    np.int32: 1,
    np.int64: 8,

    np.uint8: 6,
    np.uint16: 9,
    np.uint32: 10,
    np.uint64: 11,
    np.bool_: 5,
}


def func2(a: np.ndarray, b: float, e: int):
    c = a + b
    d = np.zeros([e, 1], np.float32)
    f = a + d
    print(d.T, d.ndim, d.tolist())
    print(d[0], d[..., 0], d[None, None, :2])

if __name__ == "__main__":
    # pflast, run_scope = parse_expr_to_df_ast("Math().sin(2)")
    # rich.print(ast_dump(pflast))
    # print(consteval_expr(pflast))
    # typing.get_type_hints
    ast = pfl.parse_func_to_pfl_ast(func2, parse_cfg=pfl.PFLParseConfig(allow_slice=True, allow_nd_slice=True))
    # rich.print(ast_dump(ast))

    pfl.PFLStaticEvaluator.partial_evaulator().eval_total_tree(ast, {
        "a": TensorMeta([1, 3], 0),
        "e": 5
    })
    for n in pfl.walk(ast):
        if isinstance(n, pfl.PFLExpr):
            print(f"\"{pfl.unparse_pfl_expr(n)}\"", n.st.metadata)
    # with open("build/test.json", "w") as f:
    #     f.write(json.dumps(pfl_ast_to_dict(ast), indent=2))