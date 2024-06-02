import abc
from typing import Any, Dict, List, Optional, TypedDict
from typing_extensions import get_type_hints, is_typeddict
from pyparsing import abstractmethod 
import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.annocore import extract_annotated_type_and_meta, parse_annotated_function, lenient_issubclass
from tensorpc.core.moduleid import get_module_id_of_type, get_qualname_of_type
from .. import mui, flow

@dataclasses.dataclass 
class HandleMeta:
    is_array: bool = False 
    is_dict: bool = False 

class ComputeFlowClasses:
    Wrapper = "ComputeFlowWrapper"
    Header = "ComputeFlowHeader"
    InputArgs = "ComputeFlowInputArgs"
    OutputArgs = "ComputeFlowOutputArgs"
    IOHandleContainer = "ComputeFlowIOHandleContainer"
    InputHandle = "ComputeFlowInputHandle"
    OutputHandle = "ComputeFlowOutputHandle"

class HandleTypePrefix:
    Input = "inp"
    Output = "out"
    Control = "ctrl"
    Option = "opt"

class ComputeNode:

    def __init__(self, id: str, name: str) -> None:
        self.name = name
        self.id = id
        annos = self.get_compute_annotation()
        self.compute_arg_annos = annos[0]
        ranno = annos[1]
        if ranno is None or not is_typeddict(ranno.type):
            raise ValueError("Compute function must be annotated with TypedDict return type.")
        self.compute_return_anno = ranno

        self.module_id = get_module_id_of_type(type(self))

    @abc.abstractmethod
    async def compute(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def get_side_layout(self) -> Optional[mui.FlexBox]:
        return None

    def get_node_layout(self) -> Optional[mui.FlexBox]:
        return None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "module_id": self.module_id,
        }

    def get_compute_annotation(self):
        return parse_annotated_function(self.compute)

    @classmethod
    @abstractmethod
    def from_state_dict(cls, data: Dict[str, Any]):
        raise NotImplementedError

class IOHandle(mui.FlexBox):
    def __init__(self, prefix: str, id: str, name: str, is_input: bool, handle_meta: Optional[HandleMeta] = None):
        self._is_input = is_input
        htype = "target" if is_input else "source"
        hpos = "left" if is_input else "right"
        handle_classes = ComputeFlowClasses.InputHandle if is_input else ComputeFlowClasses.OutputHandle
        super().__init__([
            flow.Handle(htype, hpos, f"{prefix}-{id}").prop(className=handle_classes),
            mui.Typography(name).prop(variant="caption"),
        ]) 
        if handle_meta is None:
            handle_meta = HandleMeta()
        self.handle_meta = handle_meta
        self.prop(className=ComputeFlowClasses.IOHandleContainer)

class ComputeNodeWrapper(mui.FlexBox):
    def __init__(self, cnode: ComputeNode):

        self.header = mui.HBox([
            mui.Typography(cnode.name).prop(variant="h6")
        ]).prop(className=ComputeFlowClasses.Header)

        inp_handles, out_handles = self._func_anno_to_ioargs(cnode)
        self.inp_handles = inp_handles
        self.out_handles = out_handles
        self.input_args = mui.VBox([*inp_handles]).prop(className=ComputeFlowClasses.InputArgs)
        self.output_args = mui.VBox([*out_handles]).prop(className=ComputeFlowClasses.OutputArgs)



        self.middle_node_layout: Optional[mui.FlexBox] = None 
        node_layout = cnode.get_node_layout()
        if node_layout is not None:
            self.middle_node_layout = node_layout
        self.cnode = cnode 
        self._run_status = mui.Typography().prop(variant="caption")
        self.status_box = mui.HBox([
            self._run_status,
        ])
        super().__init__([
            self.header,
            self.input_args,
            *([self.middle_node_layout] if self.middle_node_layout is not None else []),
            self.output_args,
            self.status_box
        ])
        self.prop(className=ComputeFlowClasses.Wrapper)

    def _get_layout_from_cnode(self):
        pass 

    def _func_anno_to_ioargs(self, cnode: ComputeNode):
        arg_annos = cnode.compute_arg_annos
        inp_iohandles: List[IOHandle] = []
        out_iohandles: List[IOHandle] = []
        for arg_anno in arg_annos:
            handle_meta = None
            if arg_anno.annometa is not None and isinstance(arg_anno.annometa, HandleMeta):
                handle_meta = arg_anno.annometa
            iohandle = IOHandle(HandleTypePrefix.Input, arg_anno.name, arg_anno.param.name, True, handle_meta=handle_meta)
            inp_iohandles.append(iohandle)
        ranno = cnode.compute_return_anno.type
        assert is_typeddict(ranno)
        tdict_annos = get_type_hints(ranno, include_extras=True)
        for k, v in tdict_annos.items():
            v, anno_meta = extract_annotated_type_and_meta(v)
            handle_meta = None
            if anno_meta is not None and isinstance(anno_meta, HandleMeta):
                handle_meta = anno_meta
            ohandle = IOHandle(HandleTypePrefix.Output, k, k, False, handle_meta)
            out_iohandles.append(ohandle)

        return inp_iohandles, out_iohandles


class ComputeFlow(mui.FlexBox):
    def __init__(self):
        self.graph = flow.Flow([], [], [
            flow.MiniMap(),
            flow.Controls(),
            flow.Background()
        ])
        self.side_container = mui.VBox([]).prop(height="100%", width="100%", overflow="hidden")

        self.global_container = mui.Allotment(mui.Allotment.ChildDef([
            mui.Allotment.Pane(self.graph),
            mui.Allotment.Pane(self.side_container, visible=False),
        ])).prop(defaultSizes=[200, 100])
        self.graph.event_selection_change.on(self._on_selection)

        super().__init__([self.global_container])
        self.prop(width="100%", height="100%", overflow="hidden")

    async def _on_selection(self, selection: flow.EventSelection):
        if selection.nodes:
            node =  self.graph.get_node_by_id(selection.nodes[0])
            await self.side_container.set_new_layout([
                mui.VBox([
                    mui.Typography(f"Node {node.id}"),
                    mui.Typography(f"Label: {node.data.label}"),
                    mui.Typography(f"Type: {node.type}"),
                ]).prop(flex=1)
            ])
            await self.global_container.update_pane_props(1, {
                "visible": True
            })
        else:
            await self.side_container.set_new_layout([])
            await self.global_container.update_pane_props(1, {
                "visible": False
            })
