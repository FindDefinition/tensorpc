from typing import Any, Dict, Optional, TypedDict, List, Tuple

from tensorpc.flow import flowplus, flowui, mui, plus, three, appctx 


class OutputDict(TypedDict):
    output: int


class MyCustomNode(flowplus.ComputeNode):
    def init_node(self):
        self._layout_root = mui.VBox([mui.Typography(self.name)])
        disable_event_propagation = False 
        if disable_event_propagation:
            self._layout_root.event_click.disable_and_stop_propagation()
            self._layout_root.event_double_click.disable_and_stop_propagation()
            self._layout_root.event_pointer_up.disable_and_stop_propagation()
            self._layout_root.event_pointer_down.disable_and_stop_propagation()

    # we use annotation to specify the input and output handle/type.
    async def compute(self, a: int, b: int) -> OutputDict:
        return {'output': a + b}

    def get_node_layout(self) -> Optional[mui.FlexBox]:
        return self._layout_root

    def state_dict(self) -> Dict[str, Any]:
        # save state here, can be used to restore state such as textfield.
        res = super().state_dict()
        return res

    @classmethod
    async def from_state_dict(cls, data: Dict[str, Any]):
        # load state here. you can schedule this node again with restored state.
        res = cls.from_state_dict_default(data, cls)
        return res
