from functools import partial
from tensorpc.apps.adv.model import ADVHandlePrefix, ADVRoot
from tensorpc.dock.components import flowui, mui
from tensorpc.apps.adv.model import ADVNodeType
from typing import Any, Optional, TypedDict, Union
from tensorpc.dock.components.flowplus.style import ComputeFlowClasses

class BaseHandle(mui.TooltipFlexBox):
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
                className=ComputeFlowClasses.CodeTypography,
                enableTooltipWhenOverflow=True)
        handle_desc.bind_pfl_query(dm,
            value=(get_handle_fn, "name"),
            textAlign=(get_handle_fn, "textAlign"),
            color=(get_handle_fn, "textColor"),

        )
        layout: mui.LayoutType = [
            handle_left_cond,
            handle_desc,
            handle_right_cond,
        ]
        super().__init__("", layout)
        self.prop(
            className=
            f"{ComputeFlowClasses.IOHandleContainer} {ComputeFlowClasses.NodeItem}",
            enterDelay=300,
            placement="right",
            arrow=True,
        )
        self.bind_pfl_query(dm,
            outline=(get_handle_fn, "outline"),
        )
        self.bind_pfl_query(dm,
            title=(get_handle_fn, "tooltip"),
        )


class BaseNodeWrapper(mui.FlexBox):
    def __init__(self,
                 node_gid: str,
                 node_type: ADVNodeType,
                 children: Optional[mui.LayoutType] = None):
        self.node_gid = node_gid
        self.node_type = node_type
        super().__init__(children)

class TagWithInfo(mui.TooltipFlexBox):
    def __init__(self, node_gid: str,
                 dm: mui.DataModel[ADVRoot]):
        get_tag_fn = partial(ADVRoot.get_tag, node_gid=node_gid)
        chip = mui.Chip().prop(
            size="small", muiColor="success", variant="outlined")
        chip.bind_pfl_query(dm,
            label=(get_tag_fn, "label"),
        )
        super().__init__("", [
            chip,
        ])

        self.bind_pfl_query(dm,
            title=(get_tag_fn, "tooltip"),
        )


class IONodeWrapper(BaseNodeWrapper):

    def __init__(self,
                 node_gid: str,
                 dm: mui.DataModel[ADVRoot],
                 node_type: ADVNodeType,
                 children: Optional[mui.LayoutType] = None,
                 child_overflow: Optional[mui.OverflowType] = None):
        get_node_fn = partial(ADVRoot.get_node_frontend_props, node_gid=node_gid)
        header = mui.Typography("").prop(variant="body2", flex=1, 
            enableTooltipWhenOverflow=True, paddingLeft="2px", paddingRight="2px")
        # header.bind_fields(value=node_model_draft.node.name)

        header.bind_pfl_query(dm, 
            value=(get_node_fn, "header"),
            color=(get_node_fn, "headerColor"),
        )
        icon = mui.Icon(mui.IconType.Add).prop(iconSize="small")
        icon_container = mui.Fragment([
            icon
        ])
        icon.bind_pfl_query(dm, 
            icon=(get_node_fn, "iconType"))
        icon_is_shortcut = mui.Icon(mui.IconType.Shortcut).prop(iconSize="small", muiColor="primary")
        icon_is_shortcut.bind_pfl_query(dm, show=(get_node_fn, "isRef"))
        
        icon_is_inline_flow = mui.Icon(mui.IconType.AccountTree).prop(iconSize="small")
        icon_is_inline_flow.bind_pfl_query(dm, show=(get_node_fn, "isMainFlow"), color=(get_node_fn, "ifColor"),)

        header_icons = mui.HBox([
            icon_is_shortcut,
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
        
        tags = mui.DataFlexBox(TagWithInfo(
            node_gid, dm
        ))
        tags.prop(flexFlow="row wrap", justifyContent="center",
            className=f"{ComputeFlowClasses.IOHandleContainer} {ComputeFlowClasses.NodeItem}")
        tags.bind_pfl_query(dm, 
            dataList=(get_node_fn, "tags"))
        tags_cond = mui.MatchCase.binary_selection(True,
            success=tags
        )
        tags_cond.bind_pfl_query(dm, 
            condition=(get_node_fn, "hasTag"))
        moddle_node_overflow = mui.undefined
        if child_overflow is not None:
            moddle_node_overflow = child_overflow
        _run_status = mui.Typography().prop(variant="caption").bind_pfl_query(dm, 
            value=(get_node_fn, "bottomMsg"),
            color=(get_node_fn, "ifColor"),
        )
        status_box = mui.HBox([
            icon_is_inline_flow,
            _run_status.prop(flex=1),
        ]).prop(
            className=
            f"{ComputeFlowClasses.NodeItem} {ComputeFlowClasses.BottomStatus}")
        status_box_cond = mui.MatchCase.binary_selection(True,
            success=status_box
        )
        status_box_cond.bind_pfl_query(dm, 
            condition=(get_node_fn, "isMainFlow"))
        middle_node_container = mui.Fragment(([
            mui.VBox(children).prop(
                className=ComputeFlowClasses.NodeItem,
                flex=1,
                overflow=moddle_node_overflow)
        ] if children is not None else []))
        ui_dict = {
            "header": header_container,
            "tags": tags_cond,

            # "input_args": self.input_args,
            "middle_node": middle_node_container,
            # "output_args": self.output_args,
            "handles": handles,
            "status_box": status_box_cond,
            # "resizer": self.resizer,
        }
        super().__init__(node_gid, node_type, ui_dict)
        self.prop(
            className=
            f"{ComputeFlowClasses.NodeWrapper} {ComputeFlowClasses.NodeWrappedSelected}"
        )
        self.prop(minWidth="150px")


class IndicatorWrapper(BaseNodeWrapper):
    def __init__(self, node_gid: str,
                 dm: mui.DataModel[ADVRoot]):
        get_node_fn = partial(ADVRoot.get_node_frontend_props, node_gid=node_gid)
        handle = flowui.Handle("target", "left", f"{ADVHandlePrefix.OutIndicator}-outputs")
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
        super().__init__(node_gid, ADVNodeType.OUT_INDICATOR, layout)
        self.prop(
            className=
            f"{ComputeFlowClasses.IOHandleContainer} {ComputeFlowClasses.NodeItem}"
        )

class MarkdownNodeWrapper(BaseNodeWrapper):
    def __init__(self, node_gid: str, dm: mui.DataModel[ADVRoot], init_width=200, init_height=200):
        get_node_fn = partial(ADVRoot.get_node_frontend_props, node_gid=node_gid)
        md = mui.Markdown().bind_pfl_query(dm, value=(get_node_fn, "code"))
        md.prop(emoji=True, katex=True, codeHighlight=True)
        super().__init__(
            node_gid, ADVNodeType.MARKDOWN, [
                md,
                flowui.NodeResizer().prop(minWidth=init_width, minHeight=init_height),
            ])

        self.prop(
            # className=
            # f"{ComputeFlowClasses.NodeWrapper} {ComputeFlowClasses.NodeWrappedSelected}",
            padding="3px",
            width="100%",
            height="100%",
            minWidth=init_width,
            minHeight=init_height,
            display="block",
        )
