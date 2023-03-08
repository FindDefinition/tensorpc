from typing import Any
from .inspector import ObjectInspector
from .layout import AnyFlexLayout

from tensorpc.flow.flowapp.components import mui


class InspectPanel(mui.FlexBox):

    def __init__(self, obj: Any):
        self.anylayout = AnyFlexLayout()
        self.inspector = ObjectInspector(obj).prop(width="100%",
                                                   height="100%",
                                                   overflow="hidden")
        child = mui.Allotment([
            self.inspector,
            mui.HBox([
                self.anylayout,
            ]).prop(width="100%", height="100%", overflow="hidden")
        ]).prop(default_sizes=[1, 3], width="100%", height="100%")

        super().__init__([child])
        self.prop(flex_flow="row nowrap")
