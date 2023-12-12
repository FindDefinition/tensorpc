import tensorpc.core.dataclass_dispatch as dataclasses
import inspect
from pathlib import PosixPath, WindowsPath
import traceback
from typing import Any, Dict, List, Union

import numpy as np
import io 
from tensorpc.core.moduleid import get_qualname_of_type
from tensorpc.core.serviceunit import ObservedFunction
from tensorpc.flow.flowapp import appctx
from tensorpc.flow.flowapp.components import mui
from tensorpc.flow.flowapp.components.plus.canvas import SimpleCanvas
from tensorpc.flow.flowapp.components.plus.config import ConfigPanelV2

from ..common import CommonQualNames
from ..core import ALL_OBJECT_LAYOUT_HANDLERS, ObjectGridItemConfig, ObjectLayoutHandler, DataClassesType, PriorityCommon


monospace_14px = dict(fontFamily="monospace", fontSize="14px")
_MAX_STRING_IN_DETAIL = 10000


@dataclasses.dataclass
class TensorMeta:
    qualname: str
    shape: List[int]
    dtype: str
    device: str
    is_contiguous: bool
    hasnan: bool
    hasinf: bool
    is_float: bool
    min_value: Union[float, int]
    max_value: Union[float, int] 

    def get_tags(self):
        tags = [
            mui.Chip(str(self.dtype)).prop(size="small", clickable=False),
        ]
        if self.device is not None:
            tags.append(mui.Chip(self.device).prop(size="small", clickable=False))
        if self.is_contiguous:
            tags.append(
                mui.Chip("contiguous").prop(muiColor="success",
                                            size="small",
                                            clickable=False))
        else:
            tags.append(
                mui.Chip("non-contiguous").prop(muiColor="warning",
                                                size="small",
                                                clickable=False))
        if self.hasnan:
            tags.append(
                mui.Chip("nan").prop(muiColor="error",
                                    size="small",
                                    clickable=False))
        if self.hasinf:
            tags.append(
                mui.Chip("inf").prop(muiColor="error",
                                    size="small",
                                    clickable=False))
        return tags

def _get_tensor_meta(obj):
    qualname = "np.ndarray"
    device = None
    dtype = obj.dtype
    is_contig = False
    hasnan = False
    hasinf = False
    shape = list(obj.shape)
    min_value = 0
    is_float = False
    max_value = 0
    if isinstance(obj, np.ndarray):
        is_contig = obj.flags['C_CONTIGUOUS']
        device = "cpu"
        hasnan = np.isnan(obj).any().item()
        hasinf = np.isinf(obj).any().item()
        min_value = obj.min().item()
        max_value = obj.max().item()
        is_float = np.issubdtype(obj.dtype, np.floating)
    elif get_qualname_of_type(type(obj)) == CommonQualNames.TorchTensor:
        import torch
        qualname = "torch.Tensor"
        device = obj.device.type
        is_contig = obj.is_contiguous()
        hasnan = bool(torch.isnan(obj).any().item())
        hasinf = bool(torch.isinf(obj).any().item())
        min_value = obj.min().item()
        max_value = obj.max().item()
        is_float = obj.is_floating_point()
    elif get_qualname_of_type(type(obj)) == CommonQualNames.TVTensor:
        from cumm.dtypes import get_npdtype_from_tvdtype
        qualname = "tv.Tensor"
        device = "cpu" if obj.device == -1 else "cuda"
        is_contig = obj.is_contiguous()
        # TODO handle bfloat16 or fp8
        dtype = get_npdtype_from_tvdtype(obj.dtype)
        obj_cpu = obj.cpu().numpy()
        hasnan = bool(np.isnan(obj_cpu).any().item())
        hasinf = bool(np.isinf(obj_cpu).any().item())
        min_value = obj_cpu.min().item()
        max_value = obj_cpu.max().item()
        is_float = np.issubdtype(dtype, np.floating)
    else:
        raise NotImplementedError
    return TensorMeta(qualname, shape, str(dtype), device, is_contig, is_float, hasnan, hasinf, min_value, max_value)


class TensorPreview(mui.FlexBox):

    def __init__(self, obj) -> None:
        meta = _get_tensor_meta(obj)
        self.meta = meta
        self.tags = mui.FlexBox([*meta.get_tags()]).prop(flexFlow="row wrap")
        msg = f":blue[{meta.qualname}] shape: :green[`[{','.join(map(str, meta.shape))}]`]"
        if not meta.hasnan and not meta.hasinf:
            print(meta)
            if meta.is_float:
                msg += f" min: :green[{meta.min_value:.4}]"
                msg += f" max: :green[{meta.max_value:.4}]"
            else:
                msg += f" min: :green[{meta.min_value}]"
                msg += f" max: :green[{meta.max_value}]"
        self.title = mui.Markdown(msg)
        layout = [
            self.title.prop(fontSize="14px", fontFamily="monospace"),
            self.tags,
        ]
        super().__init__(layout)
        self.prop(flexDirection="column", flex=1)

@ALL_OBJECT_LAYOUT_HANDLERS.register(np.ndarray)
@ALL_OBJECT_LAYOUT_HANDLERS.register(CommonQualNames.TorchTensor)
@ALL_OBJECT_LAYOUT_HANDLERS.register(CommonQualNames.TVTensor)
class TensorHandler(ObjectLayoutHandler):
    def create_layout(self, obj: Any) -> mui.FlexBox:
        res = TensorPreview(obj)
        return res 

    def get_grid_layout_item(self, obj: Any) -> ObjectGridItemConfig:
        meta = _get_tensor_meta(obj)
        priority = 0
        if meta.hasinf or meta.hasnan:
            priority = PriorityCommon.Highest
        return ObjectGridItemConfig(1.0, 0.5, int(priority))
