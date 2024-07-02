from typing import Any, AsyncGenerator, Dict, Optional, TypedDict, List, Tuple

from tensorpc.flow import flowplus, flowui, mui, plus, three


class OutputDict(TypedDict):
    output: int


class MyAsyncGeneratorCustomNode(flowplus.ComputeNode):
    def init_node(self):
        pass

    # we use annotation to specify the input and output handle/type.
    # if you use AsyncGenerator, you can yield the output multiple times,
    # special schedule rules will be applied to handle the output.
    async def compute(self, a: int,
                      b: int) -> AsyncGenerator[OutputDict, None]:

        for i in range(10):
            yield {'output': a + b + i}

    def get_node_layout(self) -> Optional[mui.FlexBox]:
        return mui.HBox([mui.Typography(self.name)])

    def state_dict(self) -> Dict[str, Any]:
        # save state here, can be used to restore state such as textfield.
        res = super().state_dict()
        return res

    @classmethod
    async def from_state_dict(cls, data: Dict[str, Any]):
        # load state here. you can schedule this node again with restored state.
        res = cls.from_state_dict_default(data, cls)
        return res
