from typing import Any, Optional
from tensorpc.dock.jsonlike import TensorType
from tensorpc.core.moduleid import get_qualname_of_type
import numpy as np
from typing_extensions import Literal, Self


def _try_cast_tensor_dtype(obj: Any) -> Optional[np.dtype]:
    try:
        if isinstance(obj, np.ndarray):
            return obj.dtype
        elif get_qualname_of_type(type(obj)) == TensorType.TVTensor.value:
            from cumm.dtypes import get_npdtype_from_tvdtype
            return get_npdtype_from_tvdtype(obj.dtype)
        elif get_qualname_of_type(type(obj)) == TensorType.TorchTensor.value:
            import torch
            _TORCH_DTYPE_TO_NP = {
                torch.float32: np.dtype(np.float32),
                torch.float64: np.dtype(np.float64),
                torch.float16: np.dtype(np.float16),
                torch.int32: np.dtype(np.int32),
                torch.int64: np.dtype(np.int64),
                torch.int8: np.dtype(np.int8),
                torch.int16: np.dtype(np.int16),
                torch.uint8: np.dtype(np.uint8),
            }
            return _TORCH_DTYPE_TO_NP[obj.dtype]
    except:
        return None


def _get_tensor_type(obj):
    if isinstance(obj, np.ndarray):
        return TensorType.NpArray
    elif get_qualname_of_type(type(obj)) == TensorType.TVTensor.value:
        return TensorType.TVTensor
    elif get_qualname_of_type(type(obj)) == TensorType.TorchTensor.value:
        return TensorType.TorchTensor
    else:
        return TensorType.Unknown


def _cast_tensor_to_np(obj: Any) -> Optional[np.ndarray]:
    if isinstance(obj, np.ndarray):
        return obj
    elif get_qualname_of_type(type(obj)) == TensorType.TVTensor.value:
        if obj.device == 0:
            return obj.cpu().numpy()
        return obj.numpy()

    elif get_qualname_of_type(type(obj)) == TensorType.TorchTensor.value:
        if not obj.is_cpu:
            return obj.detach().cpu().numpy()
        return obj.numpy()
    return None


def _np_pooling(mat, ksize, method='max', pad=True):
    '''Non-overlapping pooling on 2D or 3D data.
    https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy
    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <method>: str, 'max for max-pooling, 
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f).

    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky, kx = ksize

    _ceil = lambda x, y: int(np.ceil(x / float(y)))

    if pad:
        ny = _ceil(m, ky)
        nx = _ceil(n, kx)
        size = (ny * ky, nx * kx) + mat.shape[2:]
        mat_pad = np.full(size, np.nan)
        mat_pad[:m, :n, ...] = mat
    else:
        ny = m // ky
        nx = n // kx
        mat_pad = mat[:ny * ky, :nx * kx, ...]

    new_shape = (ny, ky, nx, kx) + mat.shape[2:]

    if method == 'max':
        result = np.nanmax(mat_pad.reshape(new_shape), axis=(1, 3))
    elif method == 'min':
        result = np.nanmin(mat_pad.reshape(new_shape), axis=(1, 3))
    else:
        result = np.nanmean(mat_pad.reshape(new_shape), axis=(1, 3))
    return result


class TensorContainer:

    def __init__(self, obj: Any, type: TensorType, dtype: np.dtype) -> None:
        self.type = type
        self.dtype = dtype
        self._obj = obj

    def numpy(self):
        res = _cast_tensor_to_np(self._obj)
        assert res is not None
        return res

    @property
    def shape(self):
        return list(self._obj.shape)

    @property
    def ndim(self):
        return len(self._obj.shape)

    def slice_array(self, slices: tuple[Any, ...]) -> Self:
        return self.__class__(self._obj[slices], self.type, self.dtype)

    def slice_array_to_np(self, slices: tuple[Any, ...]) -> np.ndarray:
        if self.type == TensorType.NpArray:
            return self._obj[slices]
        elif self.type == TensorType.TVTensor:
            return self._obj[slices].cpu().numpy()
        elif self.type == TensorType.TorchTensor:
            return self._obj[slices].detach().cpu().numpy()
        else:
            raise ValueError(f"Unknown tensor type: {self.type}")

    def slice_2d_and_downsample_to_image_np(
        self,
        slice0: tuple[int, int, int],
        slice1: tuple[int, int, int],
        ksize: tuple[int, int],
        method: Literal['max', 'mean', "min"] = 'max'
    ) -> tuple[np.ndarray, tuple[int, int]]:
        # create slices from dims, starts and ends
        slices = []
        dims = (slice0[0], slice1[0])
        starts = (slice0[1], slice1[1])
        ends = (slice0[2], slice1[2])
        for d in dims:
            assert d < self.ndim, f"Dimension {d} out of range {self.ndim}"
        for i in range(self.ndim):
            if i not in dims:
                slices.append(slice(None))
            else:
                start = starts[dims.index(i)]
                end = ends[dims.index(i)]
                slices.append(slice(start, end))
        new_tc = self.slice_array(tuple(slices))
        ksize = (max(new_tc.shape[0], ksize[0]), max(new_tc.shape[1],
                                                     ksize[1]))
        if self.type == TensorType.NpArray:
            res = _np_pooling(new_tc._obj, ksize, method=method, pad=True)
        elif self.type == TensorType.TVTensor:
            res = _np_pooling(new_tc._obj.cpu().numpy(),
                              ksize,
                              method=method,
                              pad=True)
        elif self.type == TensorType.TorchTensor:
            import torch
            from torch.nn import functional as F
            with torch.no_grad():
                if method == "max":
                    res = F.max_pool2d(new_tc._obj, ksize,
                                       stride=ksize).detach().cpu().numpy()
                elif method == "mean":
                    res = F.avg_pool2d(new_tc._obj, ksize,
                                       stride=ksize).detach().cpu().numpy()
                else:
                    res = (-F.max_pool2d(-new_tc._obj, ksize,
                                         stride=ksize)).detach().cpu().numpy()
        else:
            raise ValueError(f"Unknown tensor type: {self.type}")
        return res, ksize


def get_tensor_container(obj) -> Optional[TensorContainer]:
    type = _get_tensor_type(obj)
    if type == TensorType.Unknown:
        return None
    dtype = _try_cast_tensor_dtype(obj)
    assert dtype is not None
    return TensorContainer(obj, type, dtype)
