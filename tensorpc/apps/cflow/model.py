from collections.abc import Sequence
import traceback
from typing import Annotated, Any, Callable, Mapping, Optional, cast
from tensorpc.core.datamodel.draft import DraftFieldMeta
from tensorpc.core.tree_id import UniqueTreeIdForTree
from tensorpc.apps.cflow.nodes.cnode.registry import ComputeNodeBase, ComputeNodeRuntime, get_compute_node_runtime, parse_code_to_compute_cfg
from tensorpc.dock.components.models.flow import BaseNodeModel, BaseEdgeModel, BaseFlowModel, BaseFlowModelBinder
import tensorpc.core.dataclass_dispatch as dataclasses
import enum
import tensorpc.core.datamodel as D
import dataclasses as dataclasses_relaxed
from tensorpc.core.datamodel.draftstore import (DraftStoreMapMeta)
from tensorpc.utils.uniquename import UniqueNamePool
import uuid
from tensorpc.apps.cflow.nodes.cnode.registry import NODE_REGISTRY
from .coremodel import ResourceDesc

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
    SUBFLOW_INP_HANDLE = 4
    SUBFLOW_OUT_HANDLE = 5


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
    isRightPanelVisible: bool = True
    isBottomPanelVisible: bool = True

@dataclasses.dataclass(kw_only=True)
class InlineCodeInfo:
    path: str
    lineno: int

@dataclasses.dataclass(kw_only=True)
class InlineCode:
    code: str = ""

DEFAULT_EXECUTOR_ID = "local"

@dataclasses.dataclass
class ComputeNodeModel(BaseNodeModel):
    # core type
    nType: ComputeNodeType = ComputeNodeType.COMPUTE
    # subflow props
    flow: Optional["ComputeFlowModel"] = None

    # type used by user
    # node_subtype: str = ""
    # compute node props
    name: str = ""
    key: str = ""
    moduleId: str = ""
    status: ComputeNodeStatus = ComputeNodeStatus.Ready
    # msg show on bottom of node.
    msg: str = "ready"
    # compute/markdown props
    impl: InlineCode = dataclasses.field(default_factory=InlineCode)
    codeKey: Optional[str] = None
    # if true and codeKey isn't None, the code impl file is watched.
    isWatched: bool = False
    isCached: bool = False
    readOnly: bool = False
    flowKey: Optional[str] = None
    hasDetail: bool = True
    # vrc props
    # for compute node, this indicate the resource it require
    # for virtual resource (vrc) node, this indicate the resource it provide
    vResource: ResourceDesc = dataclasses.field(default_factory=ResourceDesc)
    # nodes with same exec id will always be scheduled in same executor.
    vExecId: str = DEFAULT_EXECUTOR_ID
    # backend only fields
    runtime: Annotated[Optional[ComputeNodeRuntime], DraftFieldMeta(is_external=True)] = None

    def get_request_resource_desp(self):
        return self.vResource

    def get_node_runtime(self, root_model: "ComputeFlowModelRoot") -> ComputeNodeRuntime:
        if self.codeKey is not None:
            code = root_model.shared_node_code[self.codeKey].code
        
            cfg = parse_code_to_compute_cfg(code)
        elif self.key != "":
            code = ""
            cfg = NODE_REGISTRY.global_dict[self.key]
        else:
            code = self.impl.code
            cfg = parse_code_to_compute_cfg(code)

        rt = get_compute_node_runtime(cfg, code)
        return rt

    def get_node_runtime_from_remote(self, impl_code: str) -> ComputeNodeRuntime:
        assert self.codeKey is None, "codekey should be detached from remote"
        if self.key != "":
            cfg = NODE_REGISTRY.global_dict[self.key]
        else:
            code = impl_code
            cfg = parse_code_to_compute_cfg(code)
        rt = get_compute_node_runtime(cfg)
        return rt

    def get_node_without_runtime(self):
        # used to send node model to remote
        return dataclasses.replace(self, runtime=None)


@dataclasses.dataclass(kw_only=True)
class ComputeFlowModel(BaseFlowModel[ComputeNodeModel, BaseEdgeModel]):
    nodes: Annotated[dict[str, ComputeNodeModel],
                     DraftStoreMapMeta(attr_key="n")] = dataclasses.field(
                         default_factory=dict)
    selected_node: Optional[str] = None
    # we only store user node states in splitted store.
    node_states: Annotated[dict[str, Any],
                           DraftStoreMapMeta(
                               attr_key="ns")] = dataclasses.field(
                                   default_factory=dict)

    def create_or_convert_node_state(self, node_id: str):
        node_model = self.nodes[node_id]
        assert node_model.runtime is not None
        state_dcls = node_model.runtime.cfg.state_dcls
        state_d = None 
        if state_dcls is not None:
            state = self.node_states.get(node_id)
            assert state is not None 
            if isinstance(state, dict) and len(state) > 0:
                # convert it to dcls
                try:
                    state_d = state_dcls(**state)
                except:
                    traceback.print_exc()
                    state_d = state_dcls()
                self.node_states[node_model.id] = state_d
            else:
                state_d = state
        return state_d

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
        name = uuid.uuid4().hex + "N-" + name
        return self._make_unique_name(self.nodes, name, max_count)

    def make_unique_edge_name(self, name, max_count=10000) -> str:
        name = uuid.uuid4().hex + "E-" + name
        return self._make_unique_name(self.edges, name, max_count)

@dataclasses.dataclass(kw_only=True)
class ComputeFlowModelRoot(ComputeFlowModel):
    # example: ['nodes', 'node_id_0', flow, 'nodes', 'node_id_1', 'flow']
    cur_path: list[str] = dataclasses.field(default_factory=list)
    settings: FlowSettings = dataclasses.field(default_factory=FlowSettings)

    shared_node_code: Annotated[dict[str, InlineCode],
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
    def get_uid_from_path(self):
        return UniqueTreeIdForTree.from_parts(self.cur_path).uid_encoded

    def get_or_create_node_runtime(self, node: ComputeNodeModel) -> ComputeNodeRuntime:
        if node.runtime is None:
            node.runtime = node.get_node_runtime(self)
        return node.runtime

    def get_cur_flow(self) -> Optional[ComputeFlowModel]:
        cur_obj = self
        for p in self.cur_path:
            if cur_obj is None:
                return None 
            if isinstance(cur_obj, Mapping):
                cur_obj = cur_obj.get(p, None)
            elif dataclasses.is_dataclass(cur_obj):
                cur_obj = getattr(cur_obj, p, None)
            else:
                return None
        return cast(Optional[ComputeFlowModel], cur_obj)
    
@dataclasses_relaxed.dataclass
class ComputeFlowDrafts:
    root: ComputeFlowModelRoot
    cur_model: ComputeFlowModel
    preview_path: list[str]
    preview_model: ComputeFlowModel
    selected_node: ComputeNodeModel
    selected_node_code: str
    selected_node_code_path: str
    selected_node_code_language: str

    selected_node_detail_type: int
    show_editor: bool
    show_detail: bool

    def get_node_state_draft(self, node_id: str):
        return self.cur_model.node_states[node_id]

    def get_node_drafts(self, node_id: str):
        node_state_draft = self.cur_model.node_states[node_id]
        selected_node = self.cur_model.nodes[node_id]
        code_draft, code_path_draft, code_language = get_code_drafts(self.root, selected_node)
        return ComputeFlowNodeDrafts(selected_node, node_state_draft,
                                     code_draft, code_path_draft,
                                     code_language)


@dataclasses_relaxed.dataclass
class ComputeFlowNodeDrafts:
    node: ComputeNodeModel
    node_state: Any
    code: str
    code_path: str
    code_lang: str


def get_code_drafts(root_draft: ComputeFlowModelRoot,
                    node_draft: ComputeNodeModel):
    code_draft_may_module_id = D.where(
        node_draft.moduleId != "",
        root_draft.path_to_code[root_draft.module_id_to_code_info[
            node_draft.moduleId].path],
        node_draft.impl,
        return_type=InlineCode)  # type: ignore
    code_draft = D.where(node_draft.codeKey != None,
                         root_draft.shared_node_code[node_draft.codeKey],
                         code_draft_may_module_id,
                         return_type=InlineCode)  # type: ignore

    code_path_draft = D.where(
        node_draft.codeKey != None,
        D.literal_val("tensorpc://flow/shared/%s") % node_draft.codeKey,
        D.where(node_draft.moduleId != "",
                root_draft.module_id_to_code_info[node_draft.moduleId].path,
                D.literal_val("tensorpc://flow/dynamic/%s") % node_draft.id,
                return_type=str),
        return_type=str)  # type: ignore
    code_language = D.where(
        node_draft.nType == ComputeNodeType.COMPUTE.value, "python",
        "markdown")
    return code_draft.code, code_path_draft, code_language


def get_compute_flow_drafts(root_draft: ComputeFlowModelRoot):
    cur_model_draft = cast(
        Optional[ComputeFlowModel],
        D.getitem_path_dynamic(root_draft, root_draft.cur_path,
                               Optional[ComputeFlowModel]))
    prev_path_draft_if_exist = root_draft.cur_path + [
        "nodes"
    ] + D.array(cur_model_draft.selected_node) + ["flow"
                                                         ]  # type: ignore
    selected_node = cur_model_draft.nodes[cur_model_draft.selected_node]
    is_not_subflow_node_selected = D.logical_or(
        cur_model_draft.selected_node == None, selected_node.nType
        != ComputeNodeType.SUBFLOW.value)
    prev_path_draft = D.where(is_not_subflow_node_selected, [],
                              prev_path_draft_if_exist,
                              return_type=list[str])  # type: ignore
    preview_model_draft = cast(
        Optional[ComputeFlowModel],
        D.where(is_not_subflow_node_selected, D.literal_val(None), D.getitem_path_dynamic(root_draft, prev_path_draft,
                               Optional[ComputeFlowModel]), Optional[ComputeFlowModel]))
    code_draft, code_path_draft, code_language = get_code_drafts(root_draft, selected_node)
    selected_node_detail_type = D.where(
        selected_node == None,
        DetailType.NONE.value,
        D.where(selected_node.nType == ComputeNodeType.SUBFLOW.value,
                DetailType.SUBFLOW.value, DetailType.USER_LAYOUT.value),
        return_type=int)  # type: ignore

    show_editor = D.logical_and(
        root_draft.settings.isRightPanelVisible,
        D.logical_or(selected_node.nType == ComputeNodeType.MARKDOWN.value,
                     D.logical_and(selected_node.nType == ComputeNodeType.COMPUTE.value, selected_node.key == "")))
    node_has_detail = D.logical_or(
        D.logical_and(selected_node_detail_type != DetailType.SUBFLOW.value,
                      selected_node.hasDetail),
        selected_node_detail_type == DetailType.SUBFLOW.value)
    show_detail = D.logical_and(
        root_draft.settings.isBottomPanelVisible,
        D.logical_and(selected_node_detail_type != DetailType.NONE.value,
                      node_has_detail))
    
    return ComputeFlowDrafts(root_draft, cur_model_draft, prev_path_draft,
                             preview_model_draft, selected_node, code_draft,
                             code_path_draft, code_language, selected_node_detail_type,
                             show_editor, show_detail)
