from pathlib import PosixPath, WindowsPath
from typing import Any

import numpy as np

from tensorpc.core.moduleid import get_qualname_of_type
from tensorpc.core.serviceunit import ObservedFunction
from tensorpc.flow.flowapp.components import mui

from ..common import CommonQualNames
from .core import ALL_OBJECT_PREVIEW_HANDLERS, ObjectPreviewHandler

monospace_14px = dict(font_family="monospace", font_size="14px")


@ALL_OBJECT_PREVIEW_HANDLERS.register(np.ndarray)
@ALL_OBJECT_PREVIEW_HANDLERS.register(CommonQualNames.TorchTensor)
@ALL_OBJECT_PREVIEW_HANDLERS.register(CommonQualNames.TVTensor)
class TensorHandler(ObjectPreviewHandler):

    def __init__(self) -> None:
        self.tags = mui.FlexBox().prop(flex_flow="row wrap")
        self.title = mui.Typography("np.ndarray shape = []")
        self.data_print = mui.Typography("").prop(font_family="monospace",
                                                  font_size="12px",
                                                  white_space="pre-wrap")
        self.slice_val = mui.Input("Slice").prop(size="small",
                                                 mui_margin="dense")
        layout = [
            self.title.prop(font_size="14px", font_family="monospace"),
            self.tags,
            mui.Divider().prop(padding="3px"),
            mui.HBox([
                self.slice_val.prop(flex=1),
                mui.Button("show sliced", self._on_show_slice),
            ]),
            self.data_print,
        ]

        super().__init__(layout)
        self.prop(flex_direction="column")
        self.obj: Any = np.zeros([1])

    async def _on_show_slice(self):
        slice_eval_expr = f"a{self.slice_val.value}"
        res = eval(slice_eval_expr, {"a": self.obj})
        if get_qualname_of_type(type(res)) == CommonQualNames.TVTensor:
            res = res.cpu().numpy()
        else:
            res = res
        await self.data_print.write(str(res))

    async def bind(self, obj):
        # bind np object, update all metadata
        qualname = "np.ndarray"
        device = None
        dtype = self.obj.dtype
        is_contig = False
        if isinstance(obj, np.ndarray):
            is_contig = obj.flags['C_CONTIGUOUS']
            device = "cpu"
        elif get_qualname_of_type(type(obj)) == CommonQualNames.TorchTensor:
            qualname = "torch.Tensor"
            device = obj.device.type
            is_contig = obj.is_contiguous()

        elif get_qualname_of_type(type(obj)) == CommonQualNames.TVTensor:
            from cumm.dtypes import get_dtype_from_tvdtype
            qualname = "tv.Tensor"
            device = "cpu" if obj.device == -1 else "cuda"
            is_contig = obj.is_contiguous()
            dtype = get_dtype_from_tvdtype(obj.dtype)
        else:
            raise NotImplementedError
        self.obj = obj
        ev = self.data_print.update_event(value="")
        ev += self.title.update_event(
            value=f"{qualname} shape = {list(self.obj.shape)}")
        await self.send_and_wait(ev)
        tags = [
            mui.Chip(str(dtype)).prop(size="small", clickable=False),
        ]
        if device is not None:
            tags.append(mui.Chip(device).prop(size="small", clickable=False))
        if is_contig:
            tags.append(
                mui.Chip("contiguous").prop(color="success",
                                            size="small",
                                            clickable=False))
        else:
            tags.append(
                mui.Chip("non-contiguous").prop(color="warning",
                                                size="small",
                                                clickable=False))
        await self.tags.set_new_layout([*tags])


@ALL_OBJECT_PREVIEW_HANDLERS.register(str)
@ALL_OBJECT_PREVIEW_HANDLERS.register(int)
@ALL_OBJECT_PREVIEW_HANDLERS.register(float)
@ALL_OBJECT_PREVIEW_HANDLERS.register(complex)
@ALL_OBJECT_PREVIEW_HANDLERS.register(PosixPath)
@ALL_OBJECT_PREVIEW_HANDLERS.register(WindowsPath)
class StringHandler(ObjectPreviewHandler):

    def __init__(self) -> None:
        self.text = mui.Typography("").prop(font_family="monospace",
                                            font_size="14px",
                                            white_space="pre-wrap")
        super().__init__([self.text])

    async def bind(self, obj: str):
        if not isinstance(obj, str):
            str_obj = str(obj)
        else:
            str_obj = obj
        # bind np object, update all metadata
        await self.text.write(str_obj)


@ALL_OBJECT_PREVIEW_HANDLERS.register(ObservedFunction)
class ObservedFunctionHandler(ObjectPreviewHandler):

    def __init__(self) -> None:
        self.qualname = mui.Typography("").prop(word_break="break-word",
                                                **monospace_14px)
        self.path = mui.Typography("").prop(word_break="break-word",
                                            **monospace_14px)

        super().__init__(
            [self.qualname,
             mui.Divider().prop(padding="3px"), self.path])
        self.prop(flex_direction="column")

    async def bind(self, obj: ObservedFunction):
        await self.qualname.write(obj.qualname)
        await self.path.write(obj.path)
