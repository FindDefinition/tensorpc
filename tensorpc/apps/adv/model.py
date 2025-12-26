from collections.abc import Sequence
from pathlib import Path
import traceback
from typing import Annotated, Any, Callable, Mapping, Optional, Self, cast
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

class ADVNodeType(enum.IntEnum):
    # contains sub flow
    CLASS = 0
    # may contain sub flow. when have sub flow, don't have code.
    FRAGMENT = 1
    SYMBOL = 2



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
    code: str = "## let's write some code here..."

DEFAULT_EXECUTOR_ID = "local"

# @dataclasses.dataclass


@dataclasses.dataclass
class ADVNodeModel(BaseNodeModel):
    # core type
    nType: int = ADVNodeType.FRAGMENT.value
    # subflow props
    flow: Optional["ADVFlowModel"] = None
    name: str = ""
    # tmp field, set when load adv project
    frontend_path: list[str] = dataclasses.field(default_factory=list)
    path: list[str] = dataclasses.field(default_factory=list)

    impl: Optional[InlineCode] = None
    # if two node share same impl, this stores the key to original node.
    ref_fe_path: Optional[list[str]] = None
    ref_node_id: Optional[str] = None
    # --- fragment node props ---

    # --- class node props ---
    # fields
    # base classes
    # decorators



@dataclasses.dataclass(kw_only=True)
class ADVFlowModel(BaseFlowModel[ADVNodeModel, BaseEdgeModel]):
    selected_nodes: list[str] = dataclasses.field(default_factory=list)

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
class ADVProject:
    flow: ADVFlowModel
    path: str
    import_prefix: str
    # example: ['nodes', 'node_id_0', flow, 'nodes', 'node_id_1', 'flow']
    cur_path: list[str] = dataclasses.field(default_factory=list)
    # node id to relative fs path
    node_id_to_path: dict[str, list[str]] = dataclasses.field(
        default_factory=dict)
    # node id to path in dataclass model
    node_id_to_frontend_path: dict[str, list[str]] = dataclasses.field(
        default_factory=dict)

    def get_uid_from_path(self, prefix_parts: Optional[list[str]] = None):
        if prefix_parts is None:
            prefix_parts = []
        return UniqueTreeIdForTree.from_parts(prefix_parts + self.cur_path).uid_encoded

    def get_cur_flow(self) -> Optional[ADVFlowModel]:
        cur_obj = self.flow
        for p in self.cur_path:
            if cur_obj is None:
                return None 
            if isinstance(cur_obj, Mapping):
                cur_obj = cur_obj.get(p, None)
            elif dataclasses.is_dataclass(cur_obj):
                cur_obj = getattr(cur_obj, p, None)
            else:
                return None
        return cast(Optional[ADVFlowModel], cur_obj)

    def assign_path_to_all_node(self):
        node_id_to_path: dict[str, list[str]] = {}
        node_id_to_frontend_path: dict[str, list[str]] = {}

        def _assign(node: ADVNodeModel, frontend_path: list[str], path: list[str]):
            node.frontend_path = frontend_path
            node_id_to_path[node.id] = path
            node_id_to_frontend_path[node.id] = frontend_path
            if node.flow is not None:
                for n_id, n in node.flow.nodes.items():
                    _assign(n, frontend_path + ["nodes", n_id], path + [n.name])
        for n_id, n in self.flow.nodes.items():
            _assign(n, ["nodes", n_id], [n.name])
        return node_id_to_path, node_id_to_frontend_path

    def update_ref_path(self, node_id_to_frontend_path: dict[str, list[str]]):
        def _update(node: ADVNodeModel, path: list[str]):
            if node.ref_node_id is not None:
                assert node.ref_node_id in node_id_to_frontend_path, f"ref node id {node.ref_node_id} not found"
                node.ref_fe_path = node_id_to_frontend_path[node.ref_node_id]
            if node.flow is not None:
                for n_id, n in node.flow.nodes.items():
                    _update(n, path + ["nodes", n_id])
        for n_id, n in self.flow.nodes.items():
            _update(n, ["nodes", n_id])
        return 

    def draft_get_node_by_id(self,
                        node_id: str):
        node_fe_path = self.node_id_to_frontend_path[node_id]
        node: Optional[ADVNodeModel] = D.getitem_path_dynamic(self.flow, node_fe_path, Optional[ADVNodeModel])
        
        real_node: Optional[ADVNodeModel] = D.where(
            D.logical_and(node != None, node.ref_node_id == None),
            node,
            D.getitem_path_dynamic(self.flow, node.ref_fe_path, Optional[ADVNodeModel]),
            return_type=Optional[ADVNodeModel])  # type: ignore
        return real_node


    def draft_get_node_impl_editor(self,
                        node_id: str):
        real_node = self.draft_get_node_by_id(node_id)
        
        has_code = real_node.impl != None
        code_path = D.literal_val("%s/%s.py") % (self.path, D.literal_val("/").join(real_node.path))
        return has_code, real_node.impl.code, code_path

    def draft_get_cur_model(self):
        cur_model_draft = cast(
            Optional[ADVFlowModel],
            D.getitem_path_dynamic(self.flow, self.cur_path,
                                Optional[ADVFlowModel]))
        return cur_model_draft

    def draft_get_selected_node(self):
        cur_model_draft = self.draft_get_cur_model()
        selected_node = D.where(D.length(cur_model_draft.selected_nodes) == 1, cur_model_draft.nodes[cur_model_draft.selected_nodes[0]], None,
            return_type=Optional[ADVNodeModel])  # type: ignore
        return selected_node

    def draft_get_selected_flow_model(self):
        cur_model_draft = self.draft_get_cur_model()

        selected_node = self.draft_get_selected_node()
        prev_path_draft_if_exist = self.cur_path + [
            "nodes"
        ] + cur_model_draft.selected_nodes + ["flow"
                                                            ]  # type: ignore
        is_not_subflow_node_selected = D.logical_or(
            D.length(cur_model_draft.selected_nodes) != 1, selected_node.flow == None)

        prev_path_draft = D.where(is_not_subflow_node_selected, [],
                                prev_path_draft_if_exist,
                                return_type=list[str])  # type: ignore
        preview_model_draft = cast(
            Optional[ADVFlowModel],
            D.where(is_not_subflow_node_selected, D.literal_val(None), D.getitem_path_dynamic(self.flow, prev_path_draft,
                                Optional[ADVFlowModel]), Optional[ADVFlowModel]))
        return prev_path_draft, preview_model_draft

@dataclasses.dataclass(kw_only=True)
class ADVRoot:
    # don't support empty adv project, all project must exists before init.
    cur_adv_project: str
    adv_projects: dict[str, ADVProject] = dataclasses.field(
        default_factory=dict)
    running_adv_flows: dict[str, ADVFlowModel] = dataclasses.field(
        default_factory=dict)

    def get_cur_adv_project(self) -> ADVProject:
        return self.adv_projects[self.cur_adv_project]

    def get_cur_flow_uid(self) -> str:
        return self.get_cur_adv_project().get_uid_from_path([self.cur_adv_project])

    def draft_get_cur_adv_project(self) -> ADVProject:
        return self.adv_projects[self.cur_adv_project]

    def draft_get_cur_model(self):
        return self.draft_get_cur_adv_project().draft_get_cur_model()
