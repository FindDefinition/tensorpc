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
import uuid
from tensorpc.dock.components import mui 
import tensorpc.core.pfl as pfl 

class ADVNodeType(enum.IntEnum):
    # contains sub flow
    CLASS = 0
    # may contain sub flow. when have sub flow, don't have code.
    FRAGMENT = 1
    SYMBOLS = 2
    GLOBAL_SCRIPT = 3
    # user need to connect node output handle to this node
    # to indicate outputs of this flow.
    OUT_INDICATOR = 4



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

@dataclasses.dataclass(kw_only=True)
class Symbol:
    name: str
    type: str
    default: Optional[str] = None
    # when user select a fragment node, we will use different
    # border color to highlight it.
    fragment_selected: bool = False
    # when user select a variable in code editor,
    # we will use different style to highlight it.
    var_selected: bool = False

@dataclasses.dataclass(kw_only=True)
class ADVNodeHandle:
    id: str
    # display name
    name: str
    type: str
    is_input: bool
    symbol_name: str = ""
    default: Optional[str] = None
    # when user select a fragment node, we will use different
    # border color to highlight it.
    fragment_selected: bool = False
    # when user select a variable in code editor,
    # we will use different style to highlight it.
    var_selected: bool = False

    conn_node_id: Optional[str] = None
    conn_handle_id: Optional[str] = None


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
    # when this node have nested flow, this is the import code to import libraries.
    # if two node share same impl, this stores the key to original node.
    ref_fe_path: Optional[list[str]] = None
    ref_node_id: Optional[str] = None

    handles: list[ADVNodeHandle] = dataclasses.field(default_factory=list)

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

        def _assign(node: ADVNodeModel, frontend_path: list[str], path: list[str], depth: int):
            node.frontend_path = frontend_path
            node_id_to_path[node.id] = path
            node_id_to_frontend_path[node.id] = frontend_path
            if node.flow is not None:
                for n_id, n in node.flow.nodes.items():
                    _assign(n, frontend_path + ["flow", "nodes", n_id], path + [n.name], depth + 1)
        for n_id, n in self.flow.nodes.items():
            _assign(n, ["nodes", n_id], [n.name], 0)
        return node_id_to_path, node_id_to_frontend_path

    def update_ref_path(self, node_id_to_frontend_path: dict[str, list[str]]):
        def _update(node: ADVNodeModel, path: list[str]):
            if node.ref_node_id is not None:
                assert node.ref_node_id in node_id_to_frontend_path, f"ref node id {node.ref_node_id} not found"
                node.ref_fe_path = node_id_to_frontend_path[node.ref_node_id]
            if node.flow is not None:
                for n_id, n in node.flow.nodes.items():
                    _update(n, path + ["flow", "nodes", n_id])
        for n_id, n in self.flow.nodes.items():
            _update(n, ["nodes", n_id])
        return 

    @staticmethod 
    def get_node_id_path_from_fe_path(fe_path: list[str]) -> list[str]:
        return fe_path[1::3]

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

    @mui.DataModel.mark_pfl_query_func
    def get_cur_node_flows(self) -> dict[str, Any]:
        cur_proj = self.adv_projects[self.cur_adv_project]
        # cur_flow = cast(Optional[ADVFlowModel], pfl.js.Common.getItemPath(
        #     cur_proj.flow, cur_proj.cur_path))
        cur_flow: Optional[ADVFlowModel] = pfl.js.Common.getItemPath(
            cur_proj.flow, cur_proj.cur_path)
        res: dict[str, Any] = {
            "selectedNode": None,
            "enableCodeEditor": False,
        }
        if cur_flow is None:
            return res
        selected_node_ids = cur_flow.selected_nodes
        if len(selected_node_ids) == 1:
            selected_node = cur_flow.nodes[selected_node_ids[0]]
            if selected_node.ref_node_id is not None and selected_node.ref_fe_path is not None:
                real_node: Optional[ADVNodeModel] = pfl.js.Common.getItemPath(
                    cur_proj.flow, selected_node.ref_fe_path)
                if real_node is not None:
                    impl = real_node.impl
                    if impl is not None:
                        res["enableCodeEditor"] = True
            else:
                impl = selected_node.impl
                if impl is not None:
                    res["enableCodeEditor"] = True
            res["selectedNode"] = selected_node
            return res
        else:
            return res

    @mui.DataModel.mark_pfl_func
    def get_real_node_by_id(self, node_id: str) -> tuple[Optional[ADVNodeModel], bool]:
        cur_proj = self.adv_projects[self.cur_adv_project]
        node_frontend_path = cur_proj.node_id_to_frontend_path[node_id]
        node: Optional[ADVNodeModel] = pfl.js.Common.getItemPath(
            cur_proj.flow, node_frontend_path)
        node_is_ref = False
        if node is not None:
            if node.ref_node_id is not None and node.ref_fe_path is not None:
                node: Optional[ADVNodeModel] = pfl.js.Common.getItemPath(
                    cur_proj.flow, node.ref_fe_path)
                node_is_ref = True
        return node, node_is_ref


    @mui.DataModel.mark_pfl_query_nested_func
    def get_handle(self, paths: list[Any], node_id: str) -> dict[str, Any]:
        real_node, real_node_is_ref = self.get_real_node_by_id(node_id)
        res: dict[str, Any] = {}
        if real_node is not None:
            handle_idx: int = paths[0]
            handle = real_node.handles[handle_idx]
            is_input = handle.is_input
            res = {
                "id": handle.id,
                "name": handle.name,
                "type": handle.type,
                "htype": "target" if is_input else "source",
                "hpos": "left" if is_input else "right",
                "textAlign": "start" if is_input else "end",
                "is_input": is_input,
            }
            if is_input:
                res["hborder"] = "1px solid #4caf50"
        return res

    @mui.DataModel.mark_pfl_query_nested_func
    def get_right_icon(self, paths: list[Any], node_id: str) -> dict[str, Any]:
        real_node, real_node_is_ref = self.get_real_node_by_id(node_id)
        res: dict[str, Any] = {}
        if real_node is not None:
            icon_idx: int = paths[0]
            icons = [] if not real_node_is_ref else [mui.IconType.Shortcut]
            res = {
                "icon": icons[icon_idx],
            }
        return res

    @mui.DataModel.mark_pfl_query_func
    def get_node_frontend_props(self, node_id: str) -> dict[str, Any]:
        real_node, real_node_is_ref = self.get_real_node_by_id(node_id)
        res: dict[str, Any] = {}
        if real_node is not None:
            if real_node.nType == ADVNodeType.CLASS:
                icon_type = mui.IconType.DataObject
            elif real_node.nType == ADVNodeType.FRAGMENT:
                icon_type = mui.IconType.Code
            else:
                icon_type = mui.IconType.Info

            res = {
                "id": real_node.id,
                "header": real_node.name,
                "iconType": icon_type,
                "isRef": real_node_is_ref,
                "bottomMsg": "hello world!",
                "handles": real_node.handles,
                # "htype": "target" if is_input else "source",
                # "hpos": "left" if is_input else "right",
                # "textAlign": "start" if is_input else "end",
            }
            # if is_input:
            #     res["hborder"] = "1px solid #4caf50"
        return res
