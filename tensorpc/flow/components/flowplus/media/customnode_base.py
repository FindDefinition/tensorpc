from typing import Optional, TypedDict
from tensorpc.flow.components import flowui, mui, three, plus 
from tensorpc.flow.components.flowplus.compute import get_compute_flow_context, ComputeNode 

class OutputDict(TypedDict):
    output: int

class CustomNode(ComputeNode):
    def init_node(self):
        pass

    async def compute(self, a: int, b: int) -> OutputDict:
        return {
            'output': a + b
        }

    def get_node_layout(self) -> Optional[mui.FlexBox]:
        return mui.HBox([mui.Typography("Custom Node")])
