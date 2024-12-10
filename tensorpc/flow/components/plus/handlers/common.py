import ast
import inspect
from pathlib import PosixPath, WindowsPath
import traceback
from typing import Any, Dict, List, Optional, Type

import numpy as np
import io
from tensorpc.core.core_io import extract_arrays_from_data, extract_object_from_data
from tensorpc.core.moduleid import get_qualname_of_type
from tensorpc.core.serviceunit import ObservedFunction
from tensorpc.core.tree_id import UniqueTreeIdForTree
from tensorpc.flow import appctx
from tensorpc.flow.components import mui, flowui
from tensorpc.flow.components.plus.canvas import SimpleCanvas
from tensorpc.flow.components.plus.config import ConfigPanel
from tensorpc.flow.components.plus.objinspect.tree import BasicObjectTree
from tensorpc.flow.components.plus.objview.script import get_frame_obj_layout_from_code, get_init_obj_convert_code
import humanize

from tensorpc.flow.jsonlike import TreeItem
from ..common import CommonQualNames
from ..pthcommon import PytorchModuleTreeItem, check_type_is_torch_module

from ..core import ALL_OBJECT_PREVIEW_HANDLERS, ObjectPreviewHandler, DataClassesType
from ..arraygrid import NumpyArrayGrid

monospace_14px = dict(fontFamily="monospace", fontSize="14px")
_MAX_STRING_IN_DETAIL = 10000


@ALL_OBJECT_PREVIEW_HANDLERS.register(np.ndarray)
@ALL_OBJECT_PREVIEW_HANDLERS.register(CommonQualNames.TorchTensor)
@ALL_OBJECT_PREVIEW_HANDLERS.register(CommonQualNames.TorchParameter)
@ALL_OBJECT_PREVIEW_HANDLERS.register(CommonQualNames.TVTensor)
class TensorHandler(ObjectPreviewHandler):

    def __init__(self) -> None:
        self.tags = mui.FlexBox().prop(flexFlow="row wrap")
        self.title = mui.Typography("np.ndarray shape = []")
        self.data_print = mui.Typography("").prop(fontFamily="monospace",
                                                  fontSize="12px",
                                                  whiteSpace="pre-wrap")
        self.slice_val = mui.TextField(
            "Slice", callback=self._slice_change).prop(size="small",
                                                       muiMargin="dense")
        self.grid_container = mui.HBox([])
        dialog = mui.Dialog([
            self.grid_container.prop(flex=1, height="70vh", width="100%")
        ]).prop(title="Array Viewer", dialogMaxWidth="xl", fullWidth=True)
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

    def _to_numpy(self, obj):
        if get_qualname_of_type(type(obj)) == CommonQualNames.TorchTensor:
            import torch 
            if obj.is_cpu:
                if obj.dtype == torch.bfloat16 or obj.dtype == torch.float16:
                    return obj.to(torch.float32).detach().numpy()
                return obj.detach().numpy()
            if obj.dtype == torch.bfloat16 or obj.dtype == torch.float16:
                return obj.to(torch.float32).detach().cpu().numpy()
            return obj.detach().cpu().numpy()
        elif get_qualname_of_type(type(obj)) == CommonQualNames.TorchParameter:
            return obj.data.cpu().numpy()
        elif get_qualname_of_type(type(obj)) == CommonQualNames.TVTensor:
            return obj.cpu().numpy()
        else:
            if obj.dtype == np.float16:
                return obj.astype(np.float32)
            return obj

    async def _on_show_viewer_dialog(self):
        await self.grid_container.set_new_layout([
            NumpyArrayGrid(self._to_numpy(self.obj)).prop(width="100%",
                                          height="100%",
                                          overflow="hidden")
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
            res = res # .cpu().numpy()
        if get_qualname_of_type(type(res)) == CommonQualNames.TorchParameter:
            res = res.data # .cpu().numpy()
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

    async def bind(self, obj, uid: Optional[str] = None):
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
        elif get_qualname_of_type(type(obj)) == CommonQualNames.TorchParameter:
            import torch
            qualname = "torch.Parameter"
            device = obj.data.device.type
            is_contig = obj.data.is_contiguous()
            hasnan = torch.isnan(obj.data).any().item()
            hasinf = torch.isinf(obj.data).any().item()

        elif get_qualname_of_type(type(obj)) == CommonQualNames.TVTensor:
            from cumm.dtypes import get_dtype_from_tvdtype # type: ignore
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
        if uid is not None:
            self.obj_uid = uid
        ev = self.data_print.update_event(value="")
        ev += self.title.update_event(
            value=f"{qualname} shape = {list(self.obj.shape)}")
        if uid is not None:
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

@ALL_OBJECT_PREVIEW_HANDLERS.register(bool)
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

    async def bind(self, obj: str, uid: Optional[str] = None):
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

    async def bind(self, obj: ObservedFunction, uid: Optional[str] = None):
        await self.qualname.write(obj.qualname)
        await self.path.write(obj.path)


@ALL_OBJECT_PREVIEW_HANDLERS.register(DataClassesType)
class DataclassesHandler(ObjectPreviewHandler):

    def __init__(self) -> None:
        self.cfg_ctrl_container = mui.Fragment([])
        self._simple_tree = BasicObjectTree(use_fast_tree=False, clear_data_when_unmount=True)

        super().__init__([self._simple_tree])
        self.prop(flexDirection="column", flex=1)

    async def bind(self, obj: Any, uid: Optional[str] = None):
        await self._simple_tree.add_object_to_tree(obj, expand_level=2)

class DefaultHandler(ObjectPreviewHandler):
    """
    TODO if the object support any-layout, add a button to enable it.
    """

    def __init__(self, tail_btns: Optional[List[mui.Button]] = None, tail_dialogs: Optional[List[mui.Dialog]] = None, 
                 external_infos: Optional[mui.LayoutType] = None) -> None:
        self.tags = mui.FlexBox().prop(flexFlow="row wrap")
        self.title = mui.Typography("").prop(wordBreak="break-word")
        self.path = mui.Typography("").prop(wordBreak="break-word")

        self.data_print = mui.Typography("").prop(fontFamily="monospace",
                                                  fontSize="12px",
                                                  wordBreak="break-word")
        self._data_container = mui.VBox([]).prop(overflow="auto", flex=1)
        if external_infos is not None:
            self._ext_info_container = mui.VBox(external_infos)
        else:
            self._ext_info_container = mui.VBox([])

        self._simple_tree = BasicObjectTree(use_fast_tree=True, clear_data_when_unmount=True).prop(flex=1)
        
        self._objscript_editor = mui.MonacoEditor("", "python", "").prop(minHeight=0)
        self._objscript_editor.event_editor_save.on(self._on_editor_save)
        self._objscript_show = mui.VBox([])
        self._uid_to_code = {}
        self._objscript_dialog = mui.Dialog([
            mui.VBox([
                self._objscript_editor.prop(flex=1),
                mui.Divider(),
                self._objscript_show.prop(flex=1),
            ]).prop(height="100%")
        ]).prop(dialogMaxWidth="xl", fullWidth=True, height="70vh")
        layout: mui.LayoutType = {
            "title": self.title.prop(fontSize="14px", fontFamily="monospace"),
            "path": self.path.prop(fontSize="14px", fontFamily="monospace"),
            "tags": self.tags,
            "ext_info": self._ext_info_container,
            "divider": mui.Divider().prop(padding="3px"),
            "buttons": mui.ButtonGroup([
                mui.Button("print", self._on_print).prop(size="small"),
                mui.Button("tree", self._on_tree_print).prop(size="small"),
                mui.Button("script", self._on_dialog_open).prop(size="small"),
                *(tail_btns or [])
            ]).prop(size="small"),
            "data": self._data_container,
            "objscript": self._objscript_dialog,
        }
        
        if tail_dialogs is not None:
            for i, d in enumerate(tail_dialogs):
                layout[f"dialog-{i}"] = d
        super().__init__(layout)
        self.prop(flexDirection="column", overflow="hidden", flex=1)
        self.obj: Any = np.zeros([1])
        self.obj_uid: Optional[str] = None

        self.event_after_unmount.on(self._on_unmount)

    async def _on_print(self):
        string = str(self.obj)
        if len(string) > _MAX_STRING_IN_DETAIL:
            string = string[:_MAX_STRING_IN_DETAIL] + "..."
        self.data_print.props.value = string
        await self._data_container.set_new_layout([self.data_print])
        # await self.data_print.write(string)

    async def _on_tree_print(self):
        await self._data_container.set_new_layout([self._simple_tree])

        await self._simple_tree.add_object_to_tree(self.obj, expand_level=2)
        await self._simple_tree.expand_all()

    async def _on_dialog_open(self):
        if self.obj is not None and self.obj_uid is not None:
            if self.obj_uid in self._uid_to_code:
                await self._objscript_editor.write(self._uid_to_code[self.obj_uid])
            else:
                code = get_init_obj_convert_code()
                await self._objscript_editor.write(code)
            await self._objscript_dialog.set_open(True)

    async def _on_dialog_close(self, ev: mui.DialogCloseEvent):
        await self._objscript_editor.write("")
        await self._objscript_show.set_new_layout([])

    async def _on_editor_save(self, ev: mui.MonacoEditorSaveEvent):
        if self.obj is not None and self.obj_uid is not None:
            _, layouts = get_frame_obj_layout_from_code(self.obj_uid, ev.value, self.obj)
            if layouts is not None:
                await self._objscript_show.set_new_layout(layouts)

    async def bind(self, obj: Any, uid: Optional[str] = None):
        # bind np object, update all metadata
        self.obj = obj
        self.obj_uid = uid
        await self._data_container.set_new_layout([self.data_print])
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

    async def _on_unmount(self):
        self.obj = None
            

@ALL_OBJECT_PREVIEW_HANDLERS.register("torch.fx.graph_module.GraphModule.__new__.<locals>.GraphModuleImpl")
class PytorchGraphHandler(DefaultHandler):
    def __init__(self):
        graph = flowui.Flow([], [], [
            flowui.MiniMap(),
            flowui.Controls(),
            flowui.Background()
        ])
        self._graph = graph
        self.view_pane_menu_items = [
            mui.MenuItem("layout",
                         "Dagre Layout"),
        ]
        self._graph.event_pane_context_menu.on(self._on_pane_contextmenu)
        self._graph.prop(paneContextMenuItems=self.view_pane_menu_items)

        self._flow_dialog = mui.Dialog([
            graph.prop(defaultLayoutSize=(150, 40))
        ]).prop(dialogMaxWidth="xl", fullWidth=True, height="70vh")
        self._tmp_graph_data = None 

        super().__init__([
            mui.Button("Open Graph", self._open_graph),
        ], tail_dialogs=[self._flow_dialog])

    async def _on_pane_contextmenu(self, data):
        item_id = data["itemId"]
        dagre = flowui.DagreLayoutOptions(
            ranksep=20,
        )
        if item_id == "layout":
            await self._graph.do_dagre_layout(dagre)

    async def _open_graph(self):
        await self._flow_dialog.set_open(True)
        dagre = flowui.DagreLayoutOptions(
            ranksep=20,
        )
        import torch.fx
        from tensorpc.flow.components.flowplus.network.pthfx import FlowUIInterpreter, PytorchExportBuilder
        gm = self.obj
        builder = PytorchExportBuilder()
        interpreter = FlowUIInterpreter(gm, builder, True)
        outputs = interpreter.run_on_graph_placeholders()
        arrs, _ = extract_object_from_data(outputs, (flowui.SymbolicImmediate,))
        graph_res = builder.build_detached_flow(arrs)
        nodes, edges, node_type_map = graph_res.nodes, graph_res.edges, graph_res.node_type_map

        await self._graph.set_flow_and_do_dagre_layout(nodes, edges, dagre)

    async def bind(self, obj: Any, uid: Optional[str] = None):
        await super().bind(obj, uid)
    
_FX_ARG_PROVIDER_KEY = "fx_args_provider"
_FX_KWARG_PROVIDER_KEY = "fx_kwargs_provider"

_INIT_FX_ARG_EDITOR = f"""
import torch 

def { _FX_ARG_PROVIDER_KEY }():
    # None or tuple
    return None

def { _FX_KWARG_PROVIDER_KEY }():
    # None or dict
    return None

"""

@ALL_OBJECT_PREVIEW_HANDLERS.register(check_type_is_torch_module)
class PytorchModuleHandler(DefaultHandler):
    def __init__(self):
        graph = flowui.Flow([], [], [
            flowui.MiniMap(),
            flowui.Controls(),
            flowui.Background()
        ])
        self._graph = graph
        self.view_pane_menu_items = [
            mui.MenuItem("layout",
                         "Dagre Layout"),
        ]
        self._graph.event_pane_context_menu.on(self._on_pane_contextmenu)
        self._graph.prop(paneContextMenuItems=self.view_pane_menu_items)
        self._model_size = mui.Typography("Model Size: 0 MB").prop(fontSize="14px")
        self._fx_trace_input_editor = mui.MonacoEditor(_INIT_FX_ARG_EDITOR, "python", "").prop(minHeight=0, flex=1)
        self._fx_trace_input_editor.event_editor_save.on(self._on_fx_trace_editor_save)
        self._flow_dialog = mui.Dialog([
            mui.HBox([
                self._fx_trace_input_editor,
                mui.VBox([
                    graph.prop(defaultLayoutSize=(150, 40))
                ]).prop(flex=3, height="100%"),
            ]).prop(height="100%")
        ]).prop(dialogMaxWidth="xl", fullWidth=True, height="70vh")
        self._tmp_graph_data = None 

        super().__init__([
            mui.Button("Module Tree", self._on_mod_tree_print),
            mui.Button("FX Graph", self._open_graph),
        ], tail_dialogs=[self._flow_dialog], external_infos=[
            self._model_size,
        ])

    async def _on_pane_contextmenu(self, data):
        item_id = data["itemId"]
        dagre = flowui.DagreLayoutOptions(
            ranksep=20,
        )
        if item_id == "layout":
            await self._graph.do_dagre_layout(dagre)

    async def bind(self, obj: Any, uid: Optional[str] = None):
        await super().bind(obj, uid)
        # obj is nn.Module
        # get weight size
        total_size = 0
        for name, param in obj.named_parameters():
            total_size += param.numel() * param.element_size()
        total_size_str = humanize.naturalsize(total_size)
        await self._model_size.write(f"Model Size: {total_size_str}")

    async def _open_graph(self, open_dialog: bool = True):
        if open_dialog:
            await self._flow_dialog.set_open(True)
        # fetch fake args from editor
        code = self._fx_trace_input_editor.props.value
        provider_fn = None
        provider_kw_fn = None
        if not isinstance(code, mui.Undefined):
            try:
                # validate code
                ast.parse(code)
                code_comp = compile(code, "test", "exec")
                module_dict = {}
                exec(code_comp, module_dict)
                provider_fn = module_dict.get(_FX_ARG_PROVIDER_KEY)
                provider_kw_fn = module_dict.get(_FX_KWARG_PROVIDER_KEY)
            except:
                traceback.print_exc()
        dagre = flowui.DagreLayoutOptions(
            ranksep=20,
        )
        import torch.fx
        import torch.export
        from tensorpc.flow.components.flowplus.network.pthfx import FlowUIInterpreter, PytorchExportBuilder
        if provider_fn is None or provider_kw_fn is None:
            gm = torch.fx.symbolic_trace(self.obj)
        else:
            with torch.device("meta"):
                args = provider_fn()
                kwargs = provider_kw_fn()
                if args is None and kwargs is None:
                    gm = torch.fx.symbolic_trace(self.obj)
                else:
                    gm = torch.export.export(self.obj.to("meta"), args=args, kwargs=kwargs)
        builder = PytorchExportBuilder()
        interpreter = FlowUIInterpreter(gm, builder, original_mod=self.obj)
        outputs = interpreter.run_on_graph_placeholders()
        arrs, _ = extract_object_from_data(outputs, (flowui.SymbolicImmediate,))
        graph_res = builder.build_detached_flow(arrs)
        nodes, edges, node_type_map = graph_res.nodes, graph_res.edges, graph_res.node_type_map
        await self._graph.set_flow_and_do_dagre_layout(nodes, edges, dagre)

    async def _on_fx_trace_editor_save(self, ev):
        await self._open_graph(False)

    async def _on_mod_tree_print(self):
        await self._data_container.set_new_layout([self._simple_tree])
        await self._simple_tree.add_object_to_tree(PytorchModuleTreeItem(self.obj), expand_level=2)
        await self._simple_tree.expand_all()
