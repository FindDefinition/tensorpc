from typing import Any, Optional
from .inspector import ObjectInspector
from .layout import AnyFlexLayout, FlexLayoutInitType

from tensorpc.flow.flowapp.components import mui


class InspectPanel(mui.FlexBox):

    def __init__(self,
                 obj: Any,
                 init_layout: Optional[FlexLayoutInitType] = None,
                 use_fast_tree: bool = False,
                 fixed_size: bool = False):
        self.anylayout = AnyFlexLayout(init_layout)
        self.inspector = ObjectInspector(obj, use_fast_tree=use_fast_tree, fixed_size=fixed_size)
        self.inspector.prop(width="100%", height="100%", overflow="hidden")
        child = mui.Allotment([
            self.inspector,
            mui.HBox([
                self.anylayout,
            ]).prop(width="100%", height="100%", overflow="hidden")
        ]).prop(defaultSizes=[1, 3], width="100%", height="100%")

        super().__init__([child])
        self.prop(flexFlow="row nowrap")
