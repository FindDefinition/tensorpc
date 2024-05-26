import abc
from typing import Any, Dict, Optional 
import tensorpc.core.dataclass_dispatch as dataclasses
from .. import mui

class ComputeFlowClasses:
    Header = "ComputeFlowHeader"
    InputArgs = "ComputeFlowInputArgs"
    OutputArgs = "ComputeFlowOutputArgs"

class ComputeNode:

    def __init__(self, id: str, name: str) -> None:
        self.name = name
        self.id = id

    @abc.abstractmethod
    async def compute(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def get_side_layout(self) -> Optional[mui.FlexBox]:
        return None

    def get_node_layout(self) -> Optional[mui.FlexBox]:
        return None

class ComputeNodeWrapper(mui.FlexBox):
    def __init__(self, cnode: ComputeNode):

        self.header = mui.HBox([
            mui.Typography(cnode.name).prop(variant="h6")
        ]).prop(className=ComputeFlowClasses.Header)

        self.input_args = mui.VBox([]).prop(className=ComputeFlowClasses.InputArgs)
        self.output_args = mui.VBox([]).prop(className=ComputeFlowClasses.OutputArgs)

        self.middle_node_layout: Optional[mui.FlexBox] = None 
        node_layout = cnode.get_node_layout()
        if node_layout is not None:
            self.middle_node_layout = node_layout
        self.cnode = cnode 
        self._run_status = mui.Typography().prop(variant="caption")
        self.status_box = mui.HBox([

        ])

        super().__init__()
        pass 