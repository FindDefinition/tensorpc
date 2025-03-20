import dataclasses
from tensorpc.apps.cflow.nodes import register_compute_node, get_node_state_draft, ComputeNodeBase, SpecialHandleDict
import json 
from tensorpc.flow.components import mui, flowui
from typing import TypedDict, Any
from tensorpc.flow.components.plus.objinspect.tree import BasicObjectTree

class ReservedNodeTypes:
    JsonInput = "tensorpc.cflow.Json"
    ObjectTreeViewer = "tensorpc.cflow.ObjectTreeViewer"
    Expr = "tensorpc.cflow.Expr"
    TensorViewer = "tensorpc.cflow.TensorViewer"
    ImageViewer = "tensorpc.cflow.ImageViewer"

@dataclasses.dataclass
class JsonInputState:
    value: str = "0"


def _json_input_layout(drafts):
    editor = mui.SimpleCodeEditor("0", "json")
    editor.bind_draft_change(drafts.value)
    return mui.VBox([editor.prop(editorPadding=5)
                         ]).prop(width="200px",
                                 maxHeight="300px",
                                 overflow="auto")
class _JsonOutputDict(TypedDict):
    json: Any

@register_compute_node(key=ReservedNodeTypes.JsonInput,
                       name="Json Input",
                       icon_cfg=mui.IconProps(icon=mui.IconType.DataObject),
                       layout_creator=_json_input_layout,
                       state_dcls=JsonInputState)
def json_input_node() -> _JsonOutputDict:
    state, draft = get_node_state_draft(JsonInputState)
    data = json.loads(state.value)
    return {'json': data}

@register_compute_node(key=ReservedNodeTypes.ObjectTreeViewer,
                       name="Object Tree",
                       icon_cfg=mui.IconProps(icon=mui.IconType.Visibility),
                       resizer_props=flowui.NodeResizerProps(minWidth=250, minHeight=200),
                       box_props=mui.FlexBoxProps(width="100%",
                                      height="100%",
                                      minWidth=f"{250} !important",
                                      minHeight=200))
class ObjViewer(ComputeNodeBase):
    def __init__(self):
        self.item_tree = BasicObjectTree(use_init_as_root=True,
                                            default_expand_level=1000,
                                            use_fast_tree=False)


    def get_node_layout(self, drafts):
        res = mui.VBox(
            [self.item_tree.prop(flex=1, overflow="auto")]
        )
        return res.prop(width="100%", height="100%", overflow="hidden")

    async def compute(self, obj: SpecialHandleDict[Any]) -> None:
        # keep in mind that for uncontrolled component, 
        # ui may be unmounted.
        if self.item_tree.is_mounted():
            await self.item_tree.update_root_object_dict(obj,
                                            expand_level=1000,
                                            validator=self._expand_validator)
            await self.item_tree.expand_all()

    def _expand_validator(self, node: Any):
        if isinstance(node, (dict, )):
            return len(node) < 15
        if isinstance(node, (list, tuple, set)):
            return len(node) < 10
        return False
