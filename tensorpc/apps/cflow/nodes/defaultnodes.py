from tensorpc.apps.cflow.executors.simple import SSHTempExecutorBase
from tensorpc.apps.cflow.model import ComputeNodeModel
from tensorpc.apps.cflow.nodes.cnode.registry import ComputeNodeFlags, ComputeNodeRuntime
from tensorpc.autossh.core import SSHConnDesc
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.apps.cflow.nodes import register_compute_node, get_node_state_draft, ComputeNodeBase, SpecialHandleDict
import json 
from tensorpc.dock.components import mui, flowui
from typing import TypedDict, Any
from tensorpc.dock.components.plus.objinspect.tree import BasicObjectTree

class ReservedNodeTypes:
    JsonInput = "tensorpc.cflow.Json"
    ObjectTreeViewer = "tensorpc.cflow.ObjectTreeViewer"
    Expr = "tensorpc.cflow.Expr"
    TensorViewer = "tensorpc.cflow.TensorViewer"
    ImageViewer = "tensorpc.cflow.ImageViewer"

    SSHRunner = "tensorpc.cflow.SSHRunner"

@dataclasses.dataclass
class JsonInputState:
    value: str = "0"


def _json_input_layout(drafts):
    editor = mui.SimpleCodeEditor("0", "json")
    editor.bind_draft_change_uncontrolled(drafts.value)
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
                       state_dcls=JsonInputState,
                       flags=ComputeNodeFlags.EXEC_ALWAYS_LOCAL)
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
                                      minWidth=250,
                                      minHeight=200))
class ObjViewer(ComputeNodeBase):
    def __init__(self):
        self.item_tree = BasicObjectTree(use_init_as_root=True,
                                            default_expand_level=1000,
                                            use_fast_tree=False)


    def get_node_preview_layout(self, drafts):
        res = mui.VBox(
            [self.item_tree.prop(flex=1, overflow="auto")]
        )
        return res.prop(width="100%", height="100%", overflow="hidden")

    async def compute(self, obj: SpecialHandleDict[Any]) -> None:
        # keep in mind that for component controlled by code (not draft), 
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

@dataclasses.dataclass
class _SSHRunnerState:
    url_with_port: str = "localhost:22"
    username: str = "root"
    password: str = ""
    initCmd: str = ""

def _ssh_config_layout(drafts):
    url_with_port_ui = mui.Input("url:port").prop(debounce=300)
    url_with_port_ui.bind_draft_change_uncontrolled(drafts.url_with_port)
    username_ui = mui.Input("username").prop(debounce=300)
    username_ui.bind_draft_change_uncontrolled(drafts.username)
    password_ui = mui.Input("password").prop(type="password", debounce=300)
    password_ui.bind_draft_change_uncontrolled(drafts.password)
    init_cmd_ui = mui.Input("init cmd").prop(debounce=300)
    init_cmd_ui.bind_draft_change_uncontrolled(drafts.initCmd)
    return mui.VBox([
        url_with_port_ui,
        username_ui,
        password_ui,
        init_cmd_ui,
    ]).prop(width="200px",
            maxHeight="400px",
            overflow="auto",
            padding="5px")

class _SSHTempExecutor(SSHTempExecutorBase):
    def get_ssh_info_from_node_state(self, node_state: Any) -> SSHConnDesc:
        return SSHConnDesc(
            node_state.url_with_port,
            node_state.username,
            node_state.password,
            node_state.initCmd,
        )

def _create_temp_exec(node: ComputeNodeModel) -> SSHTempExecutorBase:
    # we need a unique id to store terminal state.
    return _SSHTempExecutor(node.id)

@register_compute_node(key=ReservedNodeTypes.SSHRunner,
                       name="SSH Runner",
                       icon_cfg=mui.IconProps(icon=mui.IconType.Terminal),
                       layout_creator=_ssh_config_layout,
                       state_dcls=_SSHRunnerState,
                       temp_executor_creator=_create_temp_exec)
def ssh_runner_node_node() -> None:
    # _SSHTempExecutor will run ssh connection, nothing to do here.
    return None
