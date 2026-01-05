from tensorpc.dock.components.plus.styles import CodeStyles


class ComputeFlowClasses:
    NodeWrapper = "ComputeFlowNodeWrapper"
    NodeWrappedSelected = "ComputeFlowNodeWrapperSelected"
    Header = "ComputeFlowHeader"
    IOHandleContainer = "ComputeFlowIOHandleContainer"
    IOHandleBase = "ComputeFlowIOHandleBase"
    DriverIOHandleBase = "ComputeFlowDriverIOHandleBase"
    InputHandle = "ComputeFlowInputHandle"
    OutputHandle = "ComputeFlowOutputHandle"
    DriverInputHandle = "ComputeFlowDriverInputHandle"
    DriverOutputHandle = "ComputeFlowDriverOutputHandle"

    NodeItem = "ComputeFlowNodeItem"
    CodeTypography = "ComputeFlowCodeTypography"
    BottomStatus = "ComputeFlowBottomStatus"
    HeaderIcons = "ComputeFlowHeaderIcons"

def default_compute_flow_css():
    return {
        f".{ComputeFlowClasses.Header}": {
            "borderTopLeftRadius": "7px",
            "borderTopRightRadius": "7px",
            # "justifyContent": "center",
            "paddingLeft": "4px",
            "backgroundColor": "#eee",
            "alignItems": "center",
        },
        f".{ComputeFlowClasses.HeaderIcons}": {
            "flex": 1,
            "justifyContent": "flex-end",
            "paddingRight": "4px",
            "alignItems": "center",
        },
        f".{ComputeFlowClasses.NodeWrapper}": {
            "flexDirection": "column",
            "borderRadius": "7px",
            "alignItems": "stretch",
            # "minWidth": "150px",
            "background": "white",
        },
        f".{ComputeFlowClasses.IOHandleContainer}": {
            "flexDirection": "row",
            "alignItems": "center",
            "position": "relative",
            "minHeight": "24px",
        },
        f".{ComputeFlowClasses.CodeTypography}": {
            "fontFamily": CodeStyles.fontFamily,
        },
        f".{ComputeFlowClasses.InputHandle}": {
            "position": "absolute",
            "top": "50%",
        },
        f".{ComputeFlowClasses.OutputHandle}": {
            "position": "absolute",
            "top": "50%",
        },
        f".{ComputeFlowClasses.DriverInputHandle}": {
            "position": "absolute",
            "left": "50%",
            "clipPath": "polygon(0 50%, 100% 50%, 100% 100%, 0 100%)",
        },
        f".{ComputeFlowClasses.NodeItem}": {
            "borderBottom": "1px solid lightgrey"
        },
        f".{ComputeFlowClasses.NodeItem}:last-child": {
            "borderBottom": "none",
        },
        f".{ComputeFlowClasses.BottomStatus}": {
            "justifyContent": "center",
            "alignItems": "center",
        },
        # ".react-flow__node.selected": {
        #     f".{ComputeFlowClasses.NodeWrappedSelected}": {
        #         "borderStyle": "dashed",
        #     }
        # },
        f".{ComputeFlowClasses.IOHandleBase}": {
            "borderRadius": "100%",
            "height": "12px",
            "width": "12px",
            "border": "1px solid grey",
            "background": "#eee"
        },
        f".{ComputeFlowClasses.DriverIOHandleBase}": {
            "borderRadius": "25%",
            "height": "8px",
            "width": "24px",
            "background": "silver",
            "border": "none",
            "opacity": 0,
            ":hover": {
                "opacity": 1,
            }
        },
        ".react-flow__node": {
            "padding": "0px",
        },
        ".react-flow__handle": {
        },
        # ".react-flow__handle.connecting": {
        #     "background": "#ff6060"
        # },
        ".react-flow__handle.valid": {
            "background": "#55dd99"
        },
        # ".react-flow__handle-left": {
        #     "left": "-6px",
        # },
        # ".react-flow__handle-right": {
        #     "right": "-6px",
        # },
        ".react-flow__resize-control.handle": {
            "width": "8px",
            "height": "8px",
        }
    }
