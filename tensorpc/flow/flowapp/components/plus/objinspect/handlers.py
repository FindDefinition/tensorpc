from typing import Any
from pytools import word_wrap
from tensorpc.flow.flowapp.components import mui
from tensorpc.utils.moduleid import get_qualname_of_type
from .core import ALL_OBJECT_HANDLERS, ObjectHandler
from .tree import TORCH_TENSOR_NAME, TV_TENSOR_NAME
import numpy as np 


@ALL_OBJECT_HANDLERS.register(np.ndarray)
@ALL_OBJECT_HANDLERS.register(TORCH_TENSOR_NAME)
@ALL_OBJECT_HANDLERS.register(TV_TENSOR_NAME)
class TensorHandler(ObjectHandler):
    def __init__(self) -> None:
        self.tags = mui.FlexBox().prop(flex_flow="row wrap")
        self.title = mui.Typography("np.ndarray shape = []")
        self.data_print = mui.Typography("").prop(font_family="monospace", font_size="12px", white_space="pre-line")
        self.slice_val = mui.Input("Slice").prop(size="small", mui_margin="dense")
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
        res = eval(slice_eval_expr, {
            "a": self.obj
        }) 
        if get_qualname_of_type(type(res)) == TV_TENSOR_NAME:
            res = res.cpu().numpy()
        else:
            res = res
        await self.data_print.write(str(res))

    async def bind(self, obj):
        # bind np object, update all metadata
        qualname = "np.ndarray"
        device = None

        is_contig = False
        if isinstance(obj, np.ndarray):
            is_contig = obj.flags['C_CONTIGUOUS']

        elif get_qualname_of_type(type(obj)) == TORCH_TENSOR_NAME:
            qualname = "torch.Tensor"
            device = obj.device.type
            is_contig = obj.is_contiguous()

        elif get_qualname_of_type(type(obj)) == TV_TENSOR_NAME:
            qualname = "tv.Tensor"
            device = "cpu" if obj.device == -1 else "cuda"
            is_contig = obj.is_contiguous()
        else:
            raise NotImplementedError
        self.obj = obj 
        ev = self.data_print.update_event(value="")
        ev += self.title.update_event(value=f"{qualname} shape = {list(self.obj.shape)}")
        await self.send_and_wait(ev)
        tags = [ 
            mui.Chip(str(self.obj.dtype)).prop(size="small", clickable=False),
        ]
        if device is not None:
            tags.append(mui.Chip(device).prop(size="small", clickable=False))
        if is_contig:
            tags.append(mui.Chip("contiguous").prop(color="success", size="small", clickable=False))
        else:
            tags.append(mui.Chip("non-contiguous").prop(color="warning", size="small", clickable=False))
        await self.tags.set_new_layout([*tags])


@ALL_OBJECT_HANDLERS.register(str)
class StringHandler(ObjectHandler):
    def __init__(self) -> None:
        self.text = mui.Typography("").prop(font_family="monospace", font_size="14px", white_space="pre-line")
        super().__init__([self.text])

    async def bind(self, obj: str):
        # bind np object, update all metadata
        await self.text.write(obj)
