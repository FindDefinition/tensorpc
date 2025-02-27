from typing import Any, Optional
from tensorpc.flow.components import flowui, mui
from tensorpc.flow.components.flowplus.cnode.handle import parse_func_to_handle_components, IOHandle, HandleTypePrefix
from tensorpc.flow.components.flowplus.style import ComputeFlowClasses
from tensorpc.flow.components.flowplus.model import ComputeFlowNodeModel, ComputeNodeType, ComputeFlowModel, ComputeNodeStatus
from .registry import ComputeNodeConfig
from .ctx import ComputeFlowNodeContext, enter_flow_ui_node_context_object
from tensorpc.flow.jsonlike import (as_dict_no_undefined,
                                    as_dict_no_undefined_no_deepcopy,
                                    merge_props_not_undefined)
import tensorpc.core.datamodel as D

class BaseNodeWrapper(mui.FlexBox):
    def __init__(self, node_type: ComputeNodeType, children: Optional[mui.LayoutType] = None):
        super().__init__(children)
        self._node_type = node_type

class ComputeNodeWrapper(BaseNodeWrapper):
    def __init__(self,
                 node_id: str,
                 cnode_cfg: ComputeNodeConfig,
                 node_model_draft: ComputeFlowNodeModel):
        self.header = mui.Typography(cnode_cfg.name).prop(variant="body2")
        self.icon_container = mui.Fragment([])
        icon_cfg = cnode_cfg.icon_cfg
        if icon_cfg is not None:
            self.icon_container = mui.Fragment([
                mui.Icon(mui.IconType.Add).prop(iconSize="small",
                                                icon=icon_cfg.icon,
                                                muiColor=icon_cfg.muiColor)
            ])
        self.header_icons = mui.HBox(
            []).prop(className=ComputeFlowClasses.HeaderIcons)
        self.header_container = mui.HBox([
            self.icon_container,
            self.header,
            self.header_icons,
        ]).prop(className=
                f"{ComputeFlowClasses.Header} {ComputeFlowClasses.NodeItem}")

        inp_handles, out_handles = parse_func_to_handle_components(cnode_cfg.func, cnode_cfg.is_dynamic_cls)

        self.inp_handles: list[IOHandle] = []
        self.out_handles: list[IOHandle] = []
        self.handle_name_to_inp_handle: dict[str, IOHandle] = {}
        self.handle_name_to_out_handle: dict[str, IOHandle] = {}

        self.input_args = mui.Fragment([*inp_handles])
        self.output_args = mui.Fragment([*out_handles])

        self.middle_node_layout: Optional[mui.FlexBox] = None
        node_layout = cnode_cfg.layout
        if node_layout is not None:
            self.middle_node_layout = node_layout
        self._run_status = mui.Typography().prop(variant="caption").bind_fields(value=node_model_draft.msg)
        self.status_box = mui.HBox([
            self._run_status,
        ]).prop(
            className=
            f"{ComputeFlowClasses.NodeItem} {ComputeFlowClasses.BottomStatus}")
        moddle_node_overflow = mui.undefined 
        if cnode_cfg.layout_overflow is not None:
            moddle_node_overflow = cnode_cfg.layout_overflow
        
        self.middle_node_container = mui.Fragment(([
            mui.VBox([self.middle_node_layout]).prop(
                className=ComputeFlowClasses.NodeItem,
                flex=1,
                overflow=moddle_node_overflow)
        ] if self.middle_node_layout is not None else []))
        resizer = cnode_cfg.get_resizer()
        self.resizers: mui.LayoutType = []
        if resizer is not None:
            self.resizers = [resizer]
        self._resizer_container = mui.Fragment([*self.resizers])
        super().__init__(ComputeNodeType.COMPUTE, [
            flowui.Handle("target", "top", f"{HandleTypePrefix.DriverInput}-driver").prop(className=f"{ComputeFlowClasses.DriverIOHandleBase} {ComputeFlowClasses.DriverInputHandle}"),
            self.header_container, self.input_args, self.middle_node_container,
            self.output_args, self.status_box, self._resizer_container
        ])
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
            borderColor=f"matchcase({node_model_draft.status}, {D.literal_val(status_to_border_color)})",
            boxShadow=f"matchcase({node_model_draft.status}, {D.literal_val(status_to_border_shadow)})",
        )
        if cnode_cfg.box_props is not None:
            merge_props_not_undefined(self.props,
                                    cnode_cfg.box_props)


        self._ctx = ComputeFlowNodeContext(node_id)
        self.set_flow_event_context_creator(
            lambda: enter_flow_ui_node_context_object(self._ctx))

def to_ui_node(model: ComputeFlowModel, node_model: ComputeFlowNodeModel):
    if node_model.node_type == ComputeNodeType.COMPUTE:
        if node_model.code_key is not None:
            pass 
            # shared code
        pass 
    else:
        raise NotImplementedError
    pass 