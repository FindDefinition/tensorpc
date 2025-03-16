from typing import Annotated, Any, Callable, Mapping, Optional, cast
from tensorpc.core.datamodel.draft import DraftFieldMeta
from tensorpc.core.tree_id import UniqueTreeIdForTree
from tensorpc.flow.components.models.flow import BaseNodeModel, BaseEdgeModel, BaseFlowModel, BaseFlowModelBinder
import tensorpc.core.dataclass_dispatch as dataclasses
import enum
import tensorpc.core.datamodel as D
import dataclasses as dataclasses_relaxed
from tensorpc.core.datamodel.draftstore import (DraftStoreMapMeta)
from tensorpc.utils.uniquename import UniqueNamePool


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


class DetailType(enum.IntEnum):
    NONE = 0
    SUBFLOW = 1
    USER_LAYOUT = 2


@dataclasses.dataclass
class FlowSettings:
    isPreviewVisible: bool = True
    isEditorVisible: bool = True


@dataclasses.dataclass
class ComputeFlowNodeModel(BaseNodeModel):
    # core type
    node_type: ComputeNodeType = ComputeNodeType.COMPUTE
    # subflow props
    flow: Optional["ComputeFlowModel"] = None

    # type used by user
    # node_subtype: str = ""
    # compute node props
    name: str = ""
    key: str = ""
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
    run_in_proc: bool = False  # only valid when no vrc props set.
    has_detail: bool = False

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
    nodes: Annotated[dict[str, ComputeFlowNodeModel],
                     DraftStoreMapMeta(attr_key="n")] = dataclasses.field(
                         default_factory=dict)
    selected_node: Optional[str] = None
    # we only store user node states in splitted store.
    node_states: Annotated[dict[str, Any],
                           DraftStoreMapMeta(
                               attr_key="ns")] = dataclasses.field(
                                   default_factory=dict)

    def _make_unique_name(self, target: Mapping[str, Any], name, max_count=10000) -> str:
        if name not in target:
            return name
        name_without_tail = name 
        tail = 0 
        if "_" in name and name[0] != "_":
            parts = name.split("_")
            try:
                tail = int(parts[-1])
                name_without_tail = "_".join(parts[:-1])
            except ValueError:
                pass
        for i in range(tail + 1, tail + max_count):
            new_name = name_without_tail + "_{}".format(i)
            if new_name not in target:
                return new_name
        raise ValueError("max count reached")


    def make_unique_node_name(self, name, max_count=10000) -> str:
        return self._make_unique_name(self.nodes, name, max_count)

    def make_unique_edge_name(self, name, max_count=10000) -> str:
        return self._make_unique_name(self.edges, name, max_count)

@dataclasses.dataclass(kw_only=True)
class ComputeFlowModelRoot(ComputeFlowModel):
    # example: ['nodes', 'node_id_0', flow, 'nodes', 'node_id_1', 'flow']
    cur_path: list[str] = dataclasses.field(default_factory=list)
    settings: FlowSettings = dataclasses.field(default_factory=FlowSettings)

    shared_node_code: Annotated[dict[str, str],
                                DraftStoreMapMeta(
                                    attr_key="snc")] = dataclasses.field(
                                        default_factory=dict)
    shared_node_flow: Annotated[dict[str, "ComputeFlowModel"],
                                DraftStoreMapMeta(
                                    attr_key="snf")] = dataclasses.field(
                                        default_factory=dict)
    # backend only field, used for events.
    # e.g. use watchdog to watch file change. if file change, it will set content in this field.
    # then draft event observer will update the code editor.
    module_id_to_code_info: Annotated[
        dict[str, InlineCodeInfo],
        DraftFieldMeta(is_external=True)] = dataclasses.field(
            default_factory=dict)
    path_to_code: Annotated[dict[str, InlineCode],
                            DraftFieldMeta(
                                is_external=True)] = dataclasses.field(
                                    default_factory=dict)
    unique_name_pool_node: Annotated[UniqueNamePool, DraftFieldMeta(
                                is_external=True)] = dataclasses.field(
        default_factory=UniqueNamePool)
    unique_name_pool_edge: Annotated[UniqueNamePool, DraftFieldMeta(
                                is_external=True)] = dataclasses.field(
        default_factory=UniqueNamePool)
    def get_uid_from_path(self):
        return UniqueTreeIdForTree.from_parts(self.cur_path).uid_encoded

@dataclasses_relaxed.dataclass
class ComputeFlowDrafts:
    root: ComputeFlowModelRoot
    cur_model: ComputeFlowModel
    preview_path: list[str]
    preview_model: ComputeFlowModel
    selected_node: ComputeFlowNodeModel
    selected_node_code: str
    selected_node_code_path: str
    selected_node_detail_type: int
    show_editor: bool
    show_detail: bool

    def get_node_drafts(self, node_id: str):
        node_state_draft = self.cur_model.node_states[node_id]
        selected_node = self.cur_model.nodes[node_id]
        code_draft, code_path_draft = get_code_drafts(self.root, selected_node)
        code_language = D.where(
            selected_node.node_type == ComputeNodeType.COMPUTE, "python",
            "markdown")
        return ComputeFlowNodeDrafts(selected_node, node_state_draft,
                                     code_draft, code_path_draft,
                                     code_language)


@dataclasses_relaxed.dataclass
class ComputeFlowNodeDrafts:
    node: ComputeFlowNodeModel
    node_state: Any
    code: str
    code_path: str
    code_lang: str


def get_code_drafts(root_draft: ComputeFlowModelRoot,
                    node_draft: ComputeFlowNodeModel):
    code_draft_may_module_id = D.where(
        node_draft.module_id != "",
        root_draft.path_to_code[root_draft.module_id_to_code_info[
            node_draft.module_id].path].code,
        node_draft.code,
        return_type=str)  # type: ignore
    code_draft = D.where(node_draft.code_key != None,
                         root_draft.shared_node_code[node_draft.code_key],
                         code_draft_may_module_id,
                         return_type=str)  # type: ignore

    code_path_draft = D.where(
        node_draft.code_key != None,
        D.literal_val("tensorpc://flow/shared/%s") % node_draft.code_key,
        D.where(node_draft.module_id != "",
                root_draft.module_id_to_code_info[node_draft.module_id].path,
                D.literal_val("tensorpc://flow/dynamic/%s") % node_draft.id,
                return_type=str),
        return_type=str)  # type: ignore
    return code_draft, code_path_draft


def get_compute_flow_drafts(root_draft: ComputeFlowModelRoot):
    cur_model_draft = cast(
        Optional[ComputeFlowModel],
        D.getitem_path_dynamic(root_draft, root_draft.cur_path,
                               Optional[ComputeFlowModel]))
    prev_path_draft_if_exist = root_draft.cur_path + [
        "nodes"
    ] + D.create_array(cur_model_draft.selected_node) + ["flow"
                                                         ]  # type: ignore
    selected_node = cur_model_draft.nodes[cur_model_draft.selected_node]
    is_not_subflow_node_selected = D.logical_or(
        cur_model_draft.selected_node == None, selected_node.type
        != ComputeNodeType.SUBFLOW.value)
    prev_path_draft = D.where(is_not_subflow_node_selected, [],
                              prev_path_draft_if_exist,
                              return_type=list[str])  # type: ignore
    preview_model_draft = cast(
        Optional[ComputeFlowModel],
        D.getitem_path_dynamic(root_draft, prev_path_draft,
                               Optional[ComputeFlowModel]))
    code_draft, code_path_draft = get_code_drafts(root_draft, selected_node)
    selected_node_detail_type = D.where(
        selected_node == None,
        DetailType.NONE.value,
        D.where(selected_node.type == ComputeNodeType.SUBFLOW.value,
                DetailType.SUBFLOW.value, DetailType.USER_LAYOUT.value),
        return_type=int)  # type: ignore

    show_editor = D.logical_and(
        root_draft.settings.isEditorVisible,
        D.logical_or(selected_node.type == ComputeNodeType.MARKDOWN.value,
                     selected_node.type == ComputeNodeType.COMPUTE.value))
    node_has_detail = D.logical_or(
        D.logical_and(selected_node_detail_type != DetailType.SUBFLOW.value,
                      selected_node.has_detail),
        selected_node_detail_type == DetailType.SUBFLOW.value)
    show_detail = D.logical_and(
        root_draft.settings.isPreviewVisible,
        D.logical_and(selected_node_detail_type != DetailType.NONE.value,
                      node_has_detail))
    return ComputeFlowDrafts(root_draft, cur_model_draft, prev_path_draft,
                             preview_model_draft, selected_node, code_draft,
                             code_path_draft, selected_node_detail_type,
                             show_editor, show_detail)
