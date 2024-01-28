import inspect
from pathlib import PosixPath, WindowsPath
import traceback
from typing import Any, Dict

import numpy as np
import io 
from tensorpc.core.moduleid import get_qualname_of_type
from tensorpc.core.serviceunit import ObservedFunction
from tensorpc.flow.flowapp import appctx
from tensorpc.flow.flowapp.components import mui
from tensorpc.flow.flowapp.components.plus.canvas import SimpleCanvas
from tensorpc.flow.flowapp.components.plus.config import ConfigPanelV2

from ..common import CommonQualNames
from ..core import ALL_OBJECT_PREVIEW_HANDLERS, ObjectPreviewHandler, DataClassesType
from ..arraygrid import NumpyArrayGrid


monospace_14px = dict(fontFamily="monospace", fontSize="14px")
_MAX_STRING_IN_DETAIL = 10000


@ALL_OBJECT_PREVIEW_HANDLERS.register(np.ndarray)
@ALL_OBJECT_PREVIEW_HANDLERS.register(CommonQualNames.TorchTensor)
@ALL_OBJECT_PREVIEW_HANDLERS.register(CommonQualNames.TVTensor)
class TensorHandler(ObjectPreviewHandler):

    def __init__(self) -> None:
        self.tags = mui.FlexBox().prop(flexFlow="row wrap")
        self.title = mui.Typography("np.ndarray shape = []")
        self.data_print = mui.Typography("").prop(fontFamily="monospace",
                                                  fontSize="12px",
                                                  whiteSpace="pre-wrap")
        self.slice_val = mui.TextField("Slice", callback=self._slice_change).prop(size="small",
                                                 muiMargin="dense")
        self.grid_container = mui.HBox([])
        dialog = mui.Dialog([
            self.grid_container.prop(flex=1, height="70vh", width="100%")
        ]).prop(title="Array Viewer", maxWidth="xl", fullWidth=True)
        self.dialog = dialog

        layout = [
            self.title.prop(fontSize="14px", fontFamily="monospace"),
            self.tags,
            mui.Divider().prop(padding="3px"),
            mui.HBox([
                self.slice_val.prop(flex=1),
            ]),
            mui.HBox([
                mui.Button("show sliced", self._on_show_slice),
                mui.Button("3d visualization", self._on_3d_vis),
                mui.Button("Viewer", self._on_show_viewer_dialog),
                self.dialog,
            ]),
            self.data_print,
        ]

        super().__init__(layout)
        self.prop(flexDirection="column", flex=1)
        self.obj: Any = np.zeros([1])
        self.obj_uid: str = ""
        self._tensor_slices: Dict[str, str] = {}

    async def _on_show_viewer_dialog(self):
        await self.grid_container.set_new_layout([
             NumpyArrayGrid(self.obj).prop(width="100%", height="100%", overflow="hidden")
        ])
        await self.dialog.set_open(True)

    async def _on_show_slice(self):
        slice_eval_expr = f"a{self.slice_val.value}"
        try:
            res = eval(slice_eval_expr, {"a": self.obj})
        except:
            # we shouldn't raise this error because
            # it may override automatic exception in
            # tree.
            ss = io.StringIO()
            traceback.print_exc(file=ss)
            await self.data_print.write(ss.getvalue())
            return
        if get_qualname_of_type(type(res)) == CommonQualNames.TVTensor:
            res = res.cpu().numpy()
        else:
            res = res
        await self.data_print.write(str(res))

    async def _slice_change(self, value: str):
        if self.obj_uid != "":
             self._tensor_slices[self.obj_uid] = value

    async def _on_3d_vis(self):
        if self.obj_uid in self._tensor_slices:
            slice_eval_expr = f"a{self._tensor_slices[self.obj_uid]}"
        else:
            slice_eval_expr = "a"
        slice_eval_expr = f"a{self._tensor_slices[self.obj_uid]}"
        res = eval(slice_eval_expr, {"a": self.obj})
        canvas = appctx.find_component(SimpleCanvas)
        assert canvas is not None 
        await canvas._unknown_visualization(self.obj_uid, res)

    async def bind(self, obj, uid: str):
        # bind np object, update all metadata
        qualname = "np.ndarray"
        device = None
        dtype = obj.dtype
        is_contig = False
        hasnan = False
        hasinf = False

        if isinstance(obj, np.ndarray):
            is_contig = obj.flags['C_CONTIGUOUS']
            device = "cpu"
            hasnan = np.isnan(obj).any().item()
            hasinf = np.isinf(obj).any().item()
        elif get_qualname_of_type(type(obj)) == CommonQualNames.TorchTensor:
            import torch
            qualname = "torch.Tensor"
            device = obj.device.type
            is_contig = obj.is_contiguous()
            hasnan = torch.isnan(obj).any().item()
            hasinf = torch.isinf(obj).any().item()

        elif get_qualname_of_type(type(obj)) == CommonQualNames.TVTensor:
            from cumm.dtypes import get_dtype_from_tvdtype
            qualname = "tv.Tensor"
            device = "cpu" if obj.device == -1 else "cuda"
            is_contig = obj.is_contiguous()
            dtype = get_dtype_from_tvdtype(obj.dtype)
            obj_cpu = obj.cpu().numpy()
            hasnan = np.isnan(obj_cpu).any().item()
            hasinf = np.isinf(obj_cpu).any().item()
        else:
            raise NotImplementedError
        self.obj = obj
        self.obj_uid = uid
        ev = self.data_print.update_event(value="")
        ev += self.title.update_event(
            value=f"{qualname} shape = {list(self.obj.shape)}")
        if uid in self._tensor_slices:
            ev += self.slice_val.update_event(value=self._tensor_slices[uid])
        else:
            ev += self.slice_val.update_event(value="")
        await self.send_and_wait(ev)
        tags = [
            mui.Chip(str(dtype)).prop(size="small", clickable=False),
        ]
        if device is not None:
            tags.append(mui.Chip(device).prop(size="small", clickable=False))
        if is_contig:
            tags.append(
                mui.Chip("contiguous").prop(muiColor="success",
                                            size="small",
                                            clickable=False))
        else:
            tags.append(
                mui.Chip("non-contiguous").prop(muiColor="warning",
                                                size="small",
                                                clickable=False))
        if hasnan:
            tags.append(
                mui.Chip("nan").prop(muiColor="error",
                                    size="small",
                                    clickable=False))
        if hasinf:
            tags.append(
                mui.Chip("inf").prop(muiColor="error",
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
        self.text = mui.Typography("").prop(fontFamily="monospace",
                                            fontSize="14px",
                                            whiteSpace="pre-wrap")
        super().__init__([self.text])

    async def bind(self, obj: str, uid: str):
        if not isinstance(obj, str):
            str_obj = str(obj)
        else:
            str_obj = obj
        # bind np object, update all metadata
        await self.text.write(str_obj)


@ALL_OBJECT_PREVIEW_HANDLERS.register(ObservedFunction)
class ObservedFunctionHandler(ObjectPreviewHandler):

    def __init__(self) -> None:
        self.qualname = mui.Typography("").prop(wordBreak="break-word",
                                                **monospace_14px)
        self.path = mui.Typography("").prop(wordBreak="break-word",
                                            **monospace_14px)

        super().__init__(
            [self.qualname,
             mui.Divider().prop(padding="3px"), self.path])
        self.prop(flexDirection="column")

    async def bind(self, obj: ObservedFunction, uid: str):
        await self.qualname.write(obj.qualname)
        await self.path.write(obj.path)



@ALL_OBJECT_PREVIEW_HANDLERS.register(DataClassesType)
class DataclassesHandler(ObjectPreviewHandler):

    def __init__(self) -> None:
        self.cfg_ctrl_container = mui.Fragment([])
        super().__init__(
            [self.cfg_ctrl_container])
        self.prop(flexDirection="column", flex=1)

    async def bind(self, obj: Any, uid: str):
        # for uncontrolled component, use react_key to force remount.
        # TODO currently no way to update if obj dataclass def is changed with same uid.
        panel = ConfigPanelV2(obj).prop(reactKey=uid)
        await self.cfg_ctrl_container.set_new_layout([panel])

class DefaultHandler(ObjectPreviewHandler):
    """
    TODO if the object support any-layout, add a button to enable it.
    """
    def __init__(self) -> None:
        self.tags = mui.FlexBox().prop(flexFlow="row wrap")
        self.title = mui.Typography("").prop(wordBreak="break-word")
        self.path = mui.Typography("").prop(wordBreak="break-word")

        self.data_print = mui.Typography("").prop(fontFamily="monospace",
                                                  fontSize="12px",
                                                  wordBreak="break-word")
        layout = [
            self.title.prop(fontSize="14px", fontFamily="monospace"),
            self.path.prop(fontSize="14px", fontFamily="monospace"),
            self.tags,
            mui.Divider().prop(padding="3px"),
            mui.HBox([
                mui.Button("print", self._on_print),
            ]),
            self.data_print,
        ]

        super().__init__(layout)
        self.prop(flexDirection="column")
        self.obj: Any = np.zeros([1])

    async def _on_print(self):
        string = str(self.obj)
        if len(string) > _MAX_STRING_IN_DETAIL:
            string = string[:_MAX_STRING_IN_DETAIL] + "..."
        await self.data_print.write(string)

    async def bind(self, obj: Any, uid: str):
        # bind np object, update all metadata
        self.obj = obj
        ev = self.data_print.update_event(value="")
        ev += self.title.update_event(value=get_qualname_of_type(type(obj)))
        try:
            sf = inspect.getsourcefile(type(obj))
        except TypeError:
            sf = None
        if sf is None:
            sf = ""
        ev += self.path.update_event(value=sf)
        await self.send_and_wait(ev)
