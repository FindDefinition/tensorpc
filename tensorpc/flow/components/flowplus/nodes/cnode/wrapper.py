import dataclasses as dataclasses_plain
import traceback
from typing import Any, Optional, TypedDict, Union
from tensorpc.flow.components import flowui, mui
from tensorpc.flow.components.flowplus.nodes.cnode.handle import parse_func_to_handle_components, IOHandle, HandleTypePrefix
from tensorpc.flow.components.flowplus.style import ComputeFlowClasses
from tensorpc.flow.components.flowplus.model import ComputeFlowNodeDrafts, ComputeFlowNodeModel, ComputeNodeType, ComputeFlowModel, ComputeNodeStatus
from .registry import ComputeNodeConfig, parse_code_to_compute_cfg
from .ctx import ComputeFlowNodeContext, enter_flow_ui_node_context_object
from tensorpc.flow.jsonlike import (as_dict_no_undefined,
                                    as_dict_no_undefined_no_deepcopy,
                                    merge_props_not_undefined)
import tensorpc.core.datamodel as D
from ..base import BaseNodeWrapper

def _error_function() -> None:
    raise NotImplementedError("This function should not be called")

@dataclasses_plain.dataclass
class ComputeNodeIOHandles:
    handle_name_to_inp_handle: dict[str, IOHandle]
    handle_name_to_out_handle: dict[str, IOHandle]

@dataclasses_plain.dataclass
class ComputeNodeWrapperComps:
    header: mui.FlexBox
    input_args: mui.Fragment
    middle_node: mui.Fragment
    output_args: mui.Fragment
    status_box: mui.FlexBox
    resizer: mui.Fragment
    io_handles: ComputeNodeIOHandles
    def get_ui_dict(self) -> mui.LayoutType:
        return {
            "header": self.header,
            "input_args": self.input_args,
            "middle_node": self.middle_node,
            "output_args": self.output_args,
            "status_box": self.status_box,
            "resizer": self.resizer,
        }

class ComputeNodeWrapper(BaseNodeWrapper):

    def __init__(self, node_id: str, cnode_cfg: Union[ComputeNodeConfig, str],
                 node_model_draft: ComputeFlowNodeDrafts):
        parsed_cfg, comps = self.create_node_child_layout(cnode_cfg, node_model_draft)
        self.io_handles = comps.io_handles
        super().__init__(
            node_id, ComputeNodeType.COMPUTE, comps.get_ui_dict())
        if parsed_cfg.box_props is not None:
            merge_props_not_undefined(self.props, parsed_cfg.box_props)
        self.node_cfg = parsed_cfg
        self.prop(
            className=
            f"{ComputeFlowClasses.NodeWrapper} {ComputeFlowClasses.NodeWrappedSelected}"
        )
        status_to_border_color = [
            [ComputeNodeStatus.Ready, "black"],
            [ComputeNodeStatus.Running, "green"],
            [ComputeNodeStatus.Done, "black"],
            [ComputeNodeStatus.Error, "red"],
        ]
        status_to_border_shadow = [
            [ComputeNodeStatus.Ready, "none"],
            [ComputeNodeStatus.Running, "0px 0px 10px 0px green"],
            [ComputeNodeStatus.Done, "none"],
            [ComputeNodeStatus.Error, "none"],
        ]

        self.bind_fields(
            borderColor=
            f"matchcase({node_model_draft.node.status}, {D.literal_val(status_to_border_color)})",
            boxShadow=
            f"matchcase({node_model_draft.node.status}, {D.literal_val(status_to_border_shadow)})",
        )
        self._ctx = ComputeFlowNodeContext(node_id)
        self.set_flow_event_context_creator(
            lambda: enter_flow_ui_node_context_object(self._ctx))

    async def set_new_cnode(self, cnode_cfg: ComputeNodeConfig,
                                 node_model_draft: ComputeFlowNodeDrafts):
        parsed_cfg, comps = self.create_node_child_layout(cnode_cfg, node_model_draft)
        self.io_handles = comps.io_handles
        self.node_cfg = parsed_cfg
        await self.update_childs(comps.get_ui_dict()) # type: ignore
        return comps.io_handles

    async def set_node_from_code(self, code: str, node_model_draft: ComputeFlowNodeDrafts):
        # raise error instead of set to error node, we only use error node in init.
        cnode_cfg = parse_code_to_compute_cfg(code)
        return await self.set_new_cnode(cnode_cfg, node_model_draft)

    def create_node_child_layout(self, cnode_cfg: Union[ComputeNodeConfig, str],
                                 node_model_draft: ComputeFlowNodeDrafts):
        if isinstance(cnode_cfg, str):
            try:
                cnode_cfg = parse_code_to_compute_cfg(cnode_cfg)
            except:
                traceback.print_exc()
                cnode_cfg = ComputeNodeConfig(_error_function, "Error Node", "Error Node", "",
                    layout=mui.HBox([
                        mui.Typography("Error Node").prop(variant="body2")
                    ]))
        
        header = mui.Typography("").prop(variant="body2")
        header.bind_fields(name=node_model_draft.node.name)
        icon_container = mui.Fragment([])
        icon_cfg = cnode_cfg.icon_cfg
        if icon_cfg is not None:
            icon_container = mui.Fragment([
                mui.Icon(mui.IconType.Add).prop(iconSize="small",
                                                icon=icon_cfg.icon,
                                                muiColor=icon_cfg.muiColor)
            ])
        header_icons = mui.HBox(
            []).prop(className=ComputeFlowClasses.HeaderIcons)
        header_container = mui.HBox([
            icon_container,
            header,
            header_icons,
        ]).prop(className=
                f"{ComputeFlowClasses.Header} {ComputeFlowClasses.NodeItem}")

        inp_handles, out_handles = parse_func_to_handle_components(
            cnode_cfg.func, cnode_cfg.is_dynamic_cls)

        inp_handles: list[IOHandle] = []
        out_handles: list[IOHandle] = []
        handle_name_to_inp_handle: dict[str, IOHandle] = {}
        handle_name_to_out_handle: dict[str, IOHandle] = {}

        input_args = mui.Fragment([*inp_handles])
        output_args = mui.Fragment([*out_handles])
        middle_node_layout: Optional[mui.FlexBox] = None
        node_layout = cnode_cfg.layout
        if node_layout is not None:
            middle_node_layout = node_layout
        _run_status = mui.Typography().prop(variant="caption").bind_fields(
            value=node_model_draft.node.msg)
        status_box = mui.HBox([
            _run_status,
        ]).prop(
            className=
            f"{ComputeFlowClasses.NodeItem} {ComputeFlowClasses.BottomStatus}")
        moddle_node_overflow = mui.undefined
        if cnode_cfg.layout_overflow is not None:
            moddle_node_overflow = cnode_cfg.layout_overflow

        middle_node_container = mui.Fragment(([
            mui.VBox([middle_node_layout]).prop(
                className=ComputeFlowClasses.NodeItem,
                flex=1,
                overflow=moddle_node_overflow)
        ] if middle_node_layout is not None else []))
        resizer = cnode_cfg.get_resizer()
        resizers: mui.LayoutType = []
        if resizer is not None:
            resizers = [resizer]
        _resizer_container = mui.Fragment([*resizers])
        io_handles = ComputeNodeIOHandles(
            handle_name_to_inp_handle=handle_name_to_inp_handle,
            handle_name_to_out_handle=handle_name_to_out_handle
        )
        res = ComputeNodeWrapperComps(
                                       header_container, input_args,
                                       middle_node_container, output_args,
                                       status_box, _resizer_container, io_handles)
        return cnode_cfg, res

