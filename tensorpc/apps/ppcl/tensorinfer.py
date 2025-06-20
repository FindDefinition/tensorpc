from functools import partial
from typing import Any, Optional, Type, Union
import dataclasses
import numpy as np 
from tensorpc.core import pfl


@dataclasses.dataclass
class TensorMetaBase:
    shape: list[int]
    dtype: int
    is_pointer: bool = False
    num_element: int = 1

    @staticmethod
    def dtype_promotion(*args: int):
        raise NotImplementedError

    @staticmethod
    def get_default_dtype_from_pfl(pfl_info: pfl.PFLExprInfo) -> int:
        raise NotImplementedError

    @staticmethod
    def get_default_bool_dtype() -> int:
        raise NotImplementedError

    def is_scalar(self) -> bool:
        return len(self.shape) == 0

    def is_floating(self) -> bool:
        raise NotImplementedError

    def is_unsigned(self) -> bool:
        raise NotImplementedError

    def is_integer(self) -> bool:
        raise NotImplementedError

    def bit_size(self) -> int:
        raise NotImplementedError

def getitem_infer(data: pfl.PFLExprInfo, slice_items: Union[tuple[pfl.PFLExprInfo, ...], pfl.PFLExprInfo]) -> Optional[pfl.PFLMetaInferResult]:
    if not data.has_metadata():
        return None 
    data_meta = data.get_metadata_checked(TensorMetaBase)
    if isinstance(slice_items, pfl.PFLExprInfo):
        if slice_items.type == pfl.PFLExprType.NONE_TYPE:
            new_meta = dataclasses.replace(data_meta, shape=[1] + data_meta.shape)
        elif slice_items.type == pfl.PFLExprType.NUMBER:
            assert len(data_meta.shape) > 0, "Cannot slice an empty tensor"
            new_meta = dataclasses.replace(data_meta, shape=data_meta.shape[1:])
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
                stop = res_shape[dim] if slice_obj.stop is None else slice_obj.stop
                step = 1 if slice_obj.step is None else slice_obj.step
                step_abs = abs(step)
                if (start < 0):
                    start += res_shape[dim]
                if (stop < 0):
                    stop += res_shape[dim]
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
        data_meta = dataclasses.replace(data_meta, shape=res_shape)
        return pfl.PFLMetaInferResult(data_meta)

def matrix_transpose_infer(data: pfl.PFLExprInfo) -> Optional[pfl.PFLMetaInferResult]:
    if data.has_metadata():
        meta = data.get_metadata_checked(TensorMetaBase)
        shape = meta.shape
        assert len(shape) == 2
        res_meta = dataclasses.replace(meta, shape=shape[::-1])
        return pfl.PFLMetaInferResult(res_meta)

def bin_op_infer(left: pfl.PFLExprInfo, right: pfl.PFLExprInfo, is_compare: bool = False) -> Optional[pfl.PFLMetaInferResult]:
    if left.has_metadata():
        left_meta = left.get_metadata_checked(TensorMetaBase)
        shape_this = left_meta.shape
        is_pointer = left_meta.is_pointer

        if right.has_metadata():
            right_meta = right.get_metadata_checked(TensorMetaBase)
            if left_meta.is_pointer and right_meta.is_pointer:
                assert left_meta.dtype == right_meta.dtype, "Pointer tensors must have the same dtype"
                final_dtype = left_meta.dtype
            else:
                if left_meta.is_pointer and not right_meta.is_pointer:
                    assert not right_meta.is_floating(), "Pointer tensors cannot be compared with floating point tensors"
                elif right_meta.is_pointer and not left_meta.is_pointer:
                    assert not left_meta.is_floating(), "Pointer tensors cannot be compared with floating point tensors"
                final_dtype = left_meta.dtype_promotion(left_meta.dtype, right_meta.dtype)
            shape_other = right_meta.shape
            is_pointer = left_meta.is_pointer or right_meta.is_pointer
        else:
            assert right.type == pfl.PFLExprType.NUMBER or right.type == pfl.PFLExprType.BOOL
            shape_other = []
            dtype_other = left_meta.get_default_dtype_from_pfl(right)
            final_dtype = left_meta.dtype_promotion(left_meta.dtype, dtype_other)
        shape_res = np.broadcast_shapes(shape_this, shape_other)
        res_meta = left_meta.__class__(
            shape=list(shape_res),
            dtype=final_dtype if not is_compare else left_meta.get_default_bool_dtype(),
            is_pointer=is_pointer,
        )
        return pfl.PFLMetaInferResult(res_meta)

compare_op_infer = partial(bin_op_infer, is_compare=True)

def tensor_create_infer(shape: pfl.PFLExprInfo, dtype: pfl.PFLExprInfo, base_cls: Type[TensorMetaBase]) -> Any:
    if shape.has_metadata() and dtype.has_metadata():
        res_meta = base_cls(shape.metadata_checked, dtype.metadata_checked, False)
        return pfl.PFLMetaInferResult(res_meta)
