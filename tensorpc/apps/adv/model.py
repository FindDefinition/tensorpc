from collections.abc import Sequence
from pathlib import Path
import traceback
from typing import Annotated, Any, Callable, Mapping, Optional, Self, Union, cast
from tensorpc.core.annolib import Undefined, undefined
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

@dataclasses.dataclass
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

class ADVHandlePrefix:
    Input = "inp"
    Output = "out"
    OutIndicator = "oic"

class ADVConstHandles:
    OutIndicator = "oic-outputs"

@dataclasses.dataclass(kw_only=True)
class ADVNodeHandle:
    id: str
    # display name
    name: str
    type: str
    is_input: bool
    symbol_name: str = ""
    default: Optional[str] = None
    flags: int = 0
    source_node_id: Optional[str] = None
    source_handle_id: Optional[str] = None
    is_sym_handle: bool = False
    sym_depth: int = -1
    # used when output of fragment node is dict.
    dict_key: Optional[str] = None

    


@dataclasses.dataclass
class ADVNodeModel(BaseNodeModel):
    # core type
    nType: int = ADVNodeType.FRAGMENT.value
    # subflow props
    flow: Optional["ADVFlowModel"] = None
    # set after parse
    name: str = ""
    handles: list[ADVNodeHandle] = dataclasses.field(default_factory=list)

    # tmp field, set when load adv project
    frontend_path: list[str] = dataclasses.field(default_factory=list)
    path: list[str] = dataclasses.field(default_factory=list)

    impl: Optional[InlineCode] = None
    # when this node have nested flow, this is the import code to import libraries.
    # if two node share same impl, this stores the key to original node.
    ref_fe_path: Optional[list[str]] = None
    ref_import_path: Optional[list[str]] = None
    ref_node_id: Optional[str] = None

    inlinesf_name: Optional[str] = None
    # --- fragment node props ---
    # alias_map_str: use alias->new_alias,alias2->new_alias2
    # to rename a output handle of a ref node or subflow node 
    # which don't support ADV.
    alias_map: str = ""

    # --- class node props ---
    # fields
    # base classes
    # decorators
    # --- out indicator node props ---
    oic_alias: str = ""

    @staticmethod
    def get_global_uid_ext(path: list[str], id: str):
        # TODO should we use node id list instead of names + [last_id]？
        return UniqueTreeIdForTree.from_parts(path + [id]).uid_encoded


    def get_global_uid(self):
        # TODO should we use node id list instead of names + [last_id]？
        return UniqueTreeIdForTree.from_parts(self.path + [self.id]).uid_encoded

    def get_ref_global_uid(self):
        assert self.ref_node_id is not None and self.ref_import_path is not None
        return UniqueTreeIdForTree.from_parts(self.ref_import_path + [self.ref_node_id]).uid_encoded

@dataclasses.dataclass
class ADVEdgeModel(BaseEdgeModel):
    isAutoEdge: bool = False

@dataclasses.dataclass(kw_only=True)
class ADVFlowModel(BaseFlowModel[ADVNodeModel, ADVEdgeModel]):
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

    def __post_init__(self):
        # disable runtime
        pass

@dataclasses.dataclass(kw_only=True)
class ADVProject:
    flow: ADVFlowModel
    path: str
    import_prefix: str
    # example: ['nodes', 'node_id_0', flow, 'nodes', 'node_id_1', 'flow']
    cur_path: list[str] = dataclasses.field(default_factory=list)
    # node id to relative fs path
    node_gid_to_path: dict[str, list[str]] = dataclasses.field(
        default_factory=dict)
    # node id to path in dataclass model
    node_gid_to_frontend_path: dict[str, list[str]] = dataclasses.field(
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

    @staticmethod
    def get_flow_node_by_fe_path(root_flow: ADVFlowModel, frontend_path: list[str]) -> Optional[tuple[Optional[ADVNodeModel], ADVNodeModel]]:
        id_path = ADVProject.get_node_id_path_from_fe_path(frontend_path)
        cur_parent: tuple[ADVFlowModel, Optional[ADVNodeModel]] = (root_flow, None)
        cur_node: Optional[ADVNodeModel] = None
        for i, node_id in enumerate(id_path):
            cur_node = cur_parent[0].nodes[node_id]
            if i != len(id_path) - 1:
                if cur_node.flow is None:
                    return None
                cur_parent = (cur_node.flow, cur_node)
        if cur_node is None:
            return None
        return (cur_parent[1], cur_node)


    def assign_path_to_all_node(self):
        node_gid_to_import_path: dict[str, list[str]] = {}
        node_gid_to_frontend_path: dict[str, list[str]] = {}

        def _assign(node: ADVNodeModel, frontend_path: list[str], path: list[str], depth: int):
            node.frontend_path = frontend_path
            node.path = path
            node_gid_to_import_path[node.get_global_uid()] = path
            # print(node.get_global_uid(), frontend_path)
            node_gid_to_frontend_path[node.get_global_uid()] = frontend_path
            if node.flow is not None:
                for n_id, n in node.flow.nodes.items():
                    _assign(n, frontend_path + ["flow", "nodes", n_id], path + [node.name], depth + 1)
        for n_id, n in self.flow.nodes.items():
            _assign(n, ["nodes", n_id], [], 0)
        # raise NotImplementedError
        return node_gid_to_import_path, node_gid_to_frontend_path

    def update_ref_path(self, node_gid_to_path: dict[str, list[str]], node_gid_to_frontend_path: dict[str, list[str]]):
        def _update(node: ADVNodeModel, path: list[str]):
            if node.ref_node_id is not None:
                ref_gid = node.get_ref_global_uid()
                assert ref_gid in node_gid_to_frontend_path, f"node {ref_gid} not found in {path}"
                node.ref_fe_path = node_gid_to_frontend_path[ref_gid]
                node.ref_import_path = node_gid_to_path[ref_gid]
            if node.flow is not None:
                for n_id, n in node.flow.nodes.items():
                    _update(n, path + [node.id])
        for n_id, n in self.flow.nodes.items():
            _update(n, [])
        return 

    @staticmethod 
    def get_node_id_path_from_fe_path(fe_path: list[str]) -> list[str]:
        return fe_path[1::3]

    def draft_get_node_by_id(self,
                        node_id: str):
        node_fe_path = self.node_gid_to_frontend_path[node_id]
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
    def get_real_node_by_gid(self, node_gid: str) -> tuple[Optional[ADVNodeModel], bool]:
        cur_proj = self.adv_projects[self.cur_adv_project]
        node_frontend_path = cur_proj.node_gid_to_frontend_path[node_gid]
        node: Optional[ADVNodeModel] = pfl.js.Common.getItemPath(
            cur_proj.flow, node_frontend_path)
        node_is_ref = False
        if node is not None:
            if node.ref_node_id is not None and node.ref_fe_path is not None:
                node: Optional[ADVNodeModel] = pfl.js.Common.getItemPath(
                    cur_proj.flow, node.ref_fe_path)
                node_is_ref = True
        return node, node_is_ref

    @mui.DataModel.mark_pfl_func
    def get_real_node_pair_by_gid(self, node_gid: str) -> tuple[Optional[ADVNodeModel], Optional[ADVNodeModel], bool]:
        cur_proj = self.adv_projects[self.cur_adv_project]
        node_frontend_path = cur_proj.node_gid_to_frontend_path[node_gid]
        node: Optional[ADVNodeModel] = pfl.js.Common.getItemPath(
            cur_proj.flow, node_frontend_path)
        node_is_ref = False
        real_node: Optional[ADVNodeModel] = node
        if node is not None:
            if node.ref_node_id is not None and node.ref_fe_path is not None:
                real_node: Optional[ADVNodeModel] = pfl.js.Common.getItemPath(
                    cur_proj.flow, node.ref_fe_path)
                node_is_ref = True
        return node, real_node, node_is_ref

    @mui.DataModel.mark_pfl_query_nested_func
    def get_handle(self, paths: list[Any], node_gid: str) -> dict[str, Any]:
        node, real_node, real_node_is_ref = self.get_real_node_pair_by_gid(node_gid)
        res: dict[str, Any] = {}
        if node is not None and real_node is not None:
            handle_idx: int = paths[0]
            handle = node.handles[handle_idx]
            is_input = handle.is_input
            if handle.dict_key is not None:
                # we need to show original key if dict output.
                # otherwise users won't know the real key
                # when they use this function in non-adv code.
                name = handle.name + "(" + handle.dict_key + ")"
            else:
                name = handle.name
            # if real_node_is_ref:
            #     print(node_id, real_node.id, handle)
            res = {
                "id": handle.id,
                "name": name,
                "type_anno": handle.type,
                "type": "target" if is_input else "source",
                "hpos": "left" if is_input else "right",
                "textAlign": "start" if (is_input or handle.is_sym_handle) else "end",
                "is_input": is_input,
            }
            if is_input:
                res["hborder"] = "1px solid #4caf50"
        return res

    @mui.DataModel.mark_pfl_query_func
    def get_node_frontend_props(self, node_gid: str) -> dict[str, Any]:
        real_node, real_node_is_ref = self.get_real_node_by_gid(node_gid)
        res: dict[str, Any] = {}
        if real_node is not None:
            if real_node.nType == ADVNodeType.CLASS:
                icon_type = mui.IconType.DataObject
            elif real_node.nType == ADVNodeType.FRAGMENT:
                if real_node.flow is not None:
                    icon_type = mui.IconType.Reactflow
                else:
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
                "isMainFlow": not real_node_is_ref and real_node.inlinesf_name is not None,
                # "htype": "target" if is_input else "source",
                # "hpos": "left" if is_input else "right",
                # "textAlign": "start" if is_input else "end",
            }
            # output indicator props
            if real_node.nType == ADVNodeType.OUT_INDICATOR:
                if len(real_node.handles) == 0:
                    res["header"] = "..."
                else:
                    first_handle = real_node.handles[0]
                    sym_name = first_handle.symbol_name
                    if sym_name == "":
                        res["header"] = "..."
                    
                    elif real_node.oic_alias.strip() != "":
                        res["header"] = sym_name + "->" + real_node.oic_alias.strip()
                    else:
                        res["header"] = sym_name

            # if is_input:
            #     res["hborder"] = "1px solid #4caf50"
        return res
