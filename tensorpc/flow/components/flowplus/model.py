from typing import Annotated, Any, Callable, Optional, cast
from tensorpc.core.datamodel.draft import DraftFieldMeta
from tensorpc.flow.components.models.flow import BaseNodeModel, BaseEdgeModel, BaseFlowModel, BaseFlowModelBinder
import tensorpc.core.dataclass_dispatch as dataclasses
import enum 
import tensorpc.core.datamodel as D
import dataclasses as dataclasses_relaxed
from tensorpc.core.datamodel.draftstore import (DraftStoreMapMeta)
from tensorpc.flow.components.flowui import Node, Edge, Flow
class ComputeNodeType(enum.IntEnum):
    # compute node
    COMPUTE = 0
    # annotation
    MARKDOWN = 1
    # meta node that declare all used resources in a graph
    VIRTUAL_RESOURCE = 2
    # nested flow
    SUBFLOW = 3
    # handle in subflow
    SUBFLOW_HANDLE = 4

class ComputeNodeStatus(enum.IntEnum):
    Ready = 0
    Running = 1
    Error = 2
    Done = 3


@dataclasses.dataclass
class ComputeFlowNodeModel(BaseNodeModel):
    # core type
    node_type: ComputeNodeType = ComputeNodeType.COMPUTE
    # subflow props
    flow: Optional["ComputeFlowModel"] = None

    # type used by user
    # node_subtype: str = ""
    # compute node props
    module_id: str = ""
    status: ComputeNodeStatus = ComputeNodeStatus.Ready
    # msg show on bottom of node.
    msg: str = "ready"
    # compute/markdown props
    code: str = ""
    code_key: Optional[str] = None
    # if true and code_key isn't None, the code impl file is watched.
    is_watched: bool = False
    read_only: bool = False
    flow_key: Optional[str] = None
    # schedule props
    run_in_proc: bool = False # only valid when no vrc props set.

    # vrc props
    # for compute node, this indicate the resource it require
    # for virtual resource (vrc) node, this indicate the resource it provide
    vCPU: int = -1
    vMem: int = -1
    vGPU: int = -1
    vGPUMem: int = -1

@dataclasses.dataclass(kw_only=True)
class InlineCodeInfo:
    path: str 
    lineno: int

@dataclasses.dataclass(kw_only=True)
class InlineCode:
    code: str

@dataclasses.dataclass(kw_only=True)
class ComputeFlowModel(BaseFlowModel[ComputeFlowNodeModel, BaseEdgeModel]):
    nodes: Annotated[dict[str, ComputeFlowNodeModel], DraftStoreMapMeta(attr_key="n")] = dataclasses.field(default_factory=dict)
    selected_node: Optional[str] = None
    # we only store user node states in splitted store.
    node_states: Annotated[dict[str, Any], DraftStoreMapMeta(attr_key="ns")] = dataclasses.field(default_factory=dict)

@dataclasses.dataclass(kw_only=True)
class ComputeFlowModelRoot(ComputeFlowModel):
    # example: ['nodes', 'node_id_0', flow, 'nodes', 'node_id_1', 'flow']
    cur_path: list[str] = dataclasses.field(default_factory=list)
    shared_node_code: Annotated[dict[str, str], DraftStoreMapMeta(attr_key="snc")] = dataclasses.field(default_factory=dict)
    shared_node_flow: Annotated[dict[str, "ComputeFlowModel"], DraftStoreMapMeta(attr_key="snf")] = dataclasses.field(default_factory=dict)
    # backend only field, used for events.
    # e.g. use watchdog to watch file change. if file change, it will set content in this field.
    # then draft event observer will update the code editor.
    module_id_to_code_info: Annotated[dict[str, InlineCodeInfo], DraftFieldMeta(is_external=True)] = dataclasses.field(default_factory=dict)
    path_to_code: Annotated[dict[str, InlineCode], DraftFieldMeta(is_external=True)] = dataclasses.field(default_factory=dict)

@dataclasses_relaxed.dataclass 
class ComputeFlowDrafts:
    cur_model_draft: Any 
    prev_path_draft: Any
    preview_model_draft: Any
    cur_selected_node: Any
    cur_selected_node_code: Any

    def get_node_draft_by_id(self, node_id: str):
        return self.cur_model_draft.nodes[node_id]

    def get_node_userstate_draft_by_id(self, node_id: str):
        return self.cur_model_draft.node_states[node_id]

def get_compute_flow_drafts(root_draft: ComputeFlowModelRoot):
    cur_model_draft = cast(Optional[ComputeFlowModel], D.getitem_path_dynamic(root_draft, root_draft.cur_path, Optional[ComputeFlowModel]))
    prev_path_draft_if_exist = root_draft.cur_path + ["nodes"] + D.create_array(cur_model_draft.selected_node) + ["flow"] # type: ignore
    cur_selected_node = cur_model_draft.nodes[cur_model_draft.selected_node]
    is_not_subflow_node_selected = D.logical_or(cur_model_draft.selected_node == None, cur_selected_node.type != ComputeNodeType.SUBFLOW.value)
    prev_path_draft = D.where(is_not_subflow_node_selected, [], prev_path_draft_if_exist, return_type=list[str]) # type: ignore
    preview_model_draft = cast(Optional[ComputeFlowModel], D.getitem_path_dynamic(root_draft, prev_path_draft, Optional[ComputeFlowModel]))
    code_draft_may_module_id = D.where(cur_selected_node.module_id != "", root_draft.path_to_code[root_draft.module_id_to_code_info[cur_selected_node.module_id].path].code, cur_selected_node.code, return_type=str) # type: ignore
    code_draft = D.where(cur_selected_node.code_key != None, root_draft.shared_node_code[cur_selected_node.code_key], cur_selected_node.code, return_type=str) # type: ignore
    return ComputeFlowDrafts(cur_model_draft, prev_path_draft, preview_model_draft, cur_selected_node, code_draft)

class ComputeFlowBinder:
    def __init__(self, flow_comp: Flow, drafts: ComputeFlowDrafts, model_getter: Callable[[], ComputeFlowModel]):
        self.flow_comp = flow_comp
        self.drafts = drafts
        self.model_getter = model_getter