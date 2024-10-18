from tensorpc.flow.components import mui
from tensorpc.flow.components.plus.objinspect.tree import BasicObjectTree
from tensorpc.flow import marker

class BreakpointDebugPanel(mui.FlexBox):
    def __init__(self):
        self.header = mui.Typography("unknown").prop(variant="body1")
        self.header_actions = mui.HBox(
            []).prop(className=ComputeFlowClasses.HeaderIcons)

        self.header_container = mui.HBox([
            self.header,
            self.header_icons,
        ]).prop(className=
                f"{ComputeFlowClasses.Header} {ComputeFlowClasses.NodeItem}")

        super().__init__([

        ])
