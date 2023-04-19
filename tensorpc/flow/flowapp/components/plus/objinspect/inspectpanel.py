from typing import Any, Optional
from .inspector import ObjectInspector
from .layout import AnyFlexLayout, FlexLayoutInitType

from tensorpc.flow.flowapp.components import mui


class InspectPanel(mui.FlexBox):

    def __init__(self, obj: Any, init_layout: Optional[FlexLayoutInitType] = None):
        self.anylayout = AnyFlexLayout(init_layout)
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
