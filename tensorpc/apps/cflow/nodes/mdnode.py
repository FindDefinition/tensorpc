from .base import BaseNodeWrapper


from tensorpc.dock.components import mui
from tensorpc.apps.cflow.model import ComputeFlowNodeDrafts, ComputeNodeType
from tensorpc.dock.components.flowplus.style import ComputeFlowClasses

class MarkdownNodeWrapper(BaseNodeWrapper):
    def __init__(self, node_id: str, node_model_draft: ComputeFlowNodeDrafts):
        super().__init__(
            node_id, ComputeNodeType.MARKDOWN, [
                mui.Markdown().bind_fields(value=node_model_draft.code)
            ])

        self.prop(
            className=
            f"{ComputeFlowClasses.NodeWrapper} {ComputeFlowClasses.NodeWrappedSelected}",
            padding="3px"
        )
