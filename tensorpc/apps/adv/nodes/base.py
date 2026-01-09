from functools import partial
from tensorpc.apps.adv.model import ADVHandlePrefix, ADVRoot
from tensorpc.dock.components import flowui, mui
from tensorpc.apps.adv.model import ADVNodeType
from typing import Any, Optional, TypedDict, Union
from tensorpc.dock.components.flowplus.style import ComputeFlowClasses

class BaseHandle(mui.FlexBox):
    def __init__(self, node_gid: str,
                 dm: mui.DataModel[ADVRoot]):
        get_handle_fn = partial(ADVRoot.get_handle, node_gid=node_gid)
        handle = flowui.Handle("target", "left", "")
        handle.prop(className=f"{ComputeFlowClasses.IOHandleBase} {ComputeFlowClasses.InputHandle}")
        handle.bind_pfl_query(dm, 
            type=(get_handle_fn, "type"),
            handledPosition=(get_handle_fn, "hpos"),
            id=(get_handle_fn, "id"),
            border=(get_handle_fn, "hborder"),
            # className=(get_handle_fn, "className"),
        )
        handle_left_cond = mui.MatchCase.binary_selection(True, 
            success=handle
        )
        handle_right_cond = mui.MatchCase.binary_selection(False, 
            success=handle
        )
        handle_left_cond.bind_pfl_query(dm, 
            condition=(get_handle_fn, "is_input"))
        handle_right_cond.bind_pfl_query(dm, 
            condition=(get_handle_fn, "is_input"))
        handle_desc = mui.Typography("").prop(
                variant="caption",
                flex=1,
                marginLeft="8px",
                marginRight="8px",
                className=ComputeFlowClasses.CodeTypography)
        handle_desc.bind_pfl_query(dm,
            value=(get_handle_fn, "name"),
            textAlign=(get_handle_fn, "textAlign"),
        )
        layout: mui.LayoutType = [
            handle_left_cond,
            handle_desc,
            handle_right_cond,
        ]
        super().__init__(layout)
        self.prop(
            className=
            f"{ComputeFlowClasses.IOHandleContainer} {ComputeFlowClasses.NodeItem}"
        )


class BaseNodeWrapper(mui.FlexBox):

    def __init__(self,
                 node_gid: str,
                 dm: mui.DataModel[ADVRoot],
                 node_type: ADVNodeType,
                 children: Optional[mui.LayoutType] = None,
                 child_overflow: Optional[mui.OverflowType] = None):
        get_node_fn = partial(ADVRoot.get_node_frontend_props, node_gid=node_gid)
        header = mui.Typography("").prop(variant="body2", flex=1)
        # header.bind_fields(value=node_model_draft.node.name)

        header.bind_pfl_query(dm, 
            value=(get_node_fn, "header"))
        icon = mui.Icon(mui.IconType.Add).prop(iconSize="small")
        icon_container = mui.Fragment([
            icon
        ])
        icon.bind_pfl_query(dm, 
            icon=(get_node_fn, "iconType"))
        icon_is_shortcut = mui.Icon(mui.IconType.Shortcut).prop(iconSize="small", muiColor="primary")
        icon_is_main_flow = mui.Icon(mui.IconType.AccountTree).prop(iconSize="small", muiColor="primary")
        icon_is_shortcut.bind_pfl_query(dm, show=(get_node_fn, "isRef"))
        icon_is_main_flow.bind_pfl_query(dm, show=(get_node_fn, "isMainFlow"))

        header_icons = mui.HBox([
            icon_is_shortcut,
            icon_is_main_flow,
        ])
        header_container = mui.HBox([
            icon_container,
            header,
            header_icons,
        ]).prop(className=
                f"{ComputeFlowClasses.Header} {ComputeFlowClasses.NodeItem}")
        handles = mui.DataFlexBox(BaseHandle(
            node_gid, dm))
        handles.prop(variant="fragment")
        handles.bind_pfl_query(dm, 
            dataList=(get_node_fn, "handles"))
        moddle_node_overflow = mui.undefined
        if child_overflow is not None:
            moddle_node_overflow = child_overflow
        _run_status = mui.Typography().prop(variant="caption").bind_pfl_query(dm, 
            value=(get_node_fn, "bottomMsg"))
        status_box = mui.HBox([
            _run_status,
        ]).prop(
            className=
            f"{ComputeFlowClasses.NodeItem} {ComputeFlowClasses.BottomStatus}")

        middle_node_container = mui.Fragment(([
            mui.VBox(children).prop(
                className=ComputeFlowClasses.NodeItem,
                flex=1,
                overflow=moddle_node_overflow)
        ] if children is not None else []))
        ui_dict = {
            "header": header_container,
            # "input_args": self.input_args,
            "middle_node": middle_node_container,
            # "output_args": self.output_args,
            "handles": handles,
            "status_box": status_box,
            # "resizer": self.resizer,
        }
        super().__init__(ui_dict)
        self.prop(
            className=
            f"{ComputeFlowClasses.NodeWrapper} {ComputeFlowClasses.NodeWrappedSelected}"
        )
        self.prop(minWidth="150px")

        self._node_type = node_type
        self._node_gid = node_gid

class IndicatorWrapper(mui.FlexBox):
    def __init__(self, node_gid: str,
                 dm: mui.DataModel[ADVRoot],
                 handle_id: str):
        get_node_fn = partial(ADVRoot.get_node_frontend_props, node_gid=node_gid)
        handle = flowui.Handle("target", "left", handle_id)
        handle.prop(className=f"{ComputeFlowClasses.IOHandleBase} {ComputeFlowClasses.InputHandle}")
        handle_desc = mui.Typography("").prop(
                variant="caption",
                flex=1,
                marginLeft="8px",
                marginRight="8px",
                className=ComputeFlowClasses.CodeTypography)
        handle_desc.bind_pfl_query(dm,
            value=(get_node_fn, "header"),
            # textAlign=(get_handle_fn, "textAlign"),
        )
        icon = mui.Icon(mui.IconType.Output).prop(iconSize="small")

        layout: mui.LayoutType = [
            handle,
            handle_desc,
            icon,

        ]
        super().__init__(layout)
        self.prop(
            className=
            f"{ComputeFlowClasses.IOHandleContainer} {ComputeFlowClasses.NodeItem}"
        )
