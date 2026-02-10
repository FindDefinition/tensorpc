from pathlib import Path
from typing import Annotated, Any, Callable, Mapping, Optional, Self, Union, cast
from tensorpc.apps.adv.constants import TENSORPC_ADV_FOLDER_FLOW_NAME
from tensorpc.core.annolib import Undefined, undefined
from tensorpc.core.datamodel import typemetas
from tensorpc.core.datamodel.draft import DraftBase, DraftFieldMeta
from tensorpc.core.tree_id import UniqueTreeIdForTree
from tensorpc.dock.components.models.flow import BaseNodeModel, BaseEdgeModel, BaseFlowModel, BaseFlowModelBinder
import tensorpc.core.dataclass_dispatch as dataclasses
import enum
import tensorpc.core.datamodel as D
import uuid
from tensorpc.dock.components import mui 
import tensorpc.core.pfl as pfl
from tensorpc.dock.components.plus.styles import CodeStyles 

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
    MARKDOWN = 5

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


class ADVHandleFlags(enum.IntFlag):
    IS_INPUT = enum.auto()
    IS_SYM_HANDLE = enum.auto()
    ERROR_IS_MISSING = enum.auto()
    IS_METHOD_SELF = enum.auto()
    IS_CLSMETHOD_CLS = enum.auto()

class ADVNodeFlags(enum.IntFlag):
    # fragment flags
    # base func types
    IS_METHOD = enum.auto()
    IS_CLASSMETHOD = enum.auto()
    IS_STATICMETHOD = enum.auto()
    # extend func types
    IS_AUTO_FIELD_FN = enum.auto()
    IS_DOCK_UI_LAYOUT_FN = enum.auto()
    # special flag for inherited fragments.
    IS_INHERITED_NODE = enum.auto()
    # special flag for inline flow description node
    # no impl.
    IS_INLINE_FLOW_DESC = enum.auto()
    # class flags
    IS_DATACLASS = enum.auto()


class ADVPaneContextMenu(enum.Enum):
    AddFragment = "Add Fragment"
    AddNestedFragment = "Add Fragment Subflow"
    AddClass = "Add Class"
    AddGlobalScript = "Add Global Script"
    AddSymbolGroup = "Add Symbol Group"
    AddOutput = "Add Output"
    AddMarkdown = "Add Markdown"
    AddInlineFlowDesc = "Add Inline Flow Desc"
    # special actions
    AddRef = "Add Ref Node"

    # misc actions
    Debug = "Debug"


@dataclasses.dataclass
class ADVHandleSourceInfo:
    node_id: str
    handle_id: str

@dataclasses.dataclass(kw_only=True)
class ADVNodeHandle:
    id: str
    # display name
    name: str
    type: str
    # symbol_name: indicate which symbol this handle bind to in symbol groups
    symbol_name: str = "" 
    default: Optional[str] = None
    # ADVHandleFlags
    flags: int = 0 
    source_info: Optional[ADVHandleSourceInfo] = None
    sym_depth: int = -1
    # used when output of fragment node is dict.
    dict_key: Union[Undefined, str] = undefined
    # used by output indicator
    out_var_name: Union[Undefined, str] = undefined

    def set_source_info_inplace(self, node_id: str, handle_id: str):
        self.source_info = ADVHandleSourceInfo(node_id, handle_id)

    def is_method_self(self) -> bool:
        return self.flags & ADVHandleFlags.IS_METHOD_SELF != 0

@dataclasses.dataclass
class ADVNodeRefInfo:
    node_id: str
    import_path: list[str]
    fe_path: Optional[list[str]] = None

    def is_equal_to(self, other: Self):
        return self.node_id == other.node_id and ".".join(self.import_path) == ".".join(other.import_path)

_NAMED_NODE_TYPES = {
    ADVNodeType.CLASS,
    ADVNodeType.FRAGMENT,
    ADVNodeType.SYMBOLS,

}

@dataclasses.dataclass
class ADVNodeModel(BaseNodeModel):
    # core type
    nType: int = ADVNodeType.FRAGMENT.value
    # subflow props
    flow: Optional["ADVFlowModel"] = None
    # set after parse
    name: str = ""
    error: Union[Undefined, str] = undefined
    # ADVNodeFlags
    flags: int = 0 

    handles: list[ADVNodeHandle] = dataclasses.field(default_factory=list)

    # tmp field, set when load adv project
    frontend_path: list[str] = dataclasses.field(default_factory=list)
    path: list[str] = dataclasses.field(default_factory=list)

    impl: Optional[InlineCode] = None
    # when this node have nested flow, this is the import code to import libraries.
    # if two node share same impl, this stores the key to original node.
    ref: Optional[ADVNodeRefInfo] = None

    inlinesf_name: Optional[str] = None
    # --- fragment node props ---
    # alias_map_str: use alias->new_alias,alias2->new_alias2
    # to rename a output handle of a ref node or subflow node 
    # which don't support ADV.
    alias_map: str = ""
    # external decorators.
    ext_decorators: Union[list[str], Undefined] = undefined

    # --- class node props ---
    # external inherits, split by comma
    ext_inherits: Union[str, Undefined] = undefined
    # inherit another class node. only support single inherit.
    cls_inherit_ref: Optional[ADVNodeRefInfo] = None

    def is_named_node(self):
        return self.nType in _NAMED_NODE_TYPES

    def is_base_props_equal_to(self, other: Self) -> bool:
        return (self.nType == other.nType and 
                self.alias_map == other.alias_map and 
                self.name == other.name and 
                self.inlinesf_name == other.inlinesf_name)

    def is_impl_equal_to(self, other: Self) -> bool:
        if self.impl is None and other.impl is None:
            return True
        if self.impl is None or other.impl is None:
            return False
        return self.impl.code == other.impl.code

    def is_defined_in_class(self):
        return self.is_defined_in_class_static(self.flags, self.nType)

    @staticmethod
    def is_defined_in_class_static(flags: int, nType: int):
        # init_fn, ui fn are all method (IS_METHOD set), so no need to check them.
        is_class_body_def = bool(flags & int(ADVNodeFlags.IS_METHOD | ADVNodeFlags.IS_CLASSMETHOD | ADVNodeFlags.IS_STATICMETHOD))
        return nType == ADVNodeType.FRAGMENT and is_class_body_def

    def is_inline_flow_desc(self):
        return self.nType == ADVNodeType.FRAGMENT and (self.flags & int(ADVNodeFlags.IS_INLINE_FLOW_DESC)) != 0

    def is_inline_flow_desc_def(self):
        return self.ref is None and self.nType == ADVNodeType.FRAGMENT and (self.flags & int(ADVNodeFlags.IS_INLINE_FLOW_DESC)) != 0

    def is_method(self):
        return self.nType == ADVNodeType.FRAGMENT and (self.flags & int(ADVNodeFlags.IS_METHOD)) != 0

    def is_class_method(self):
        return self.nType == ADVNodeType.FRAGMENT and (self.flags & int(ADVNodeFlags.IS_CLASSMETHOD)) != 0

    def is_static_method(self):
        return self.nType == ADVNodeType.FRAGMENT and (self.flags & int(ADVNodeFlags.IS_STATICMETHOD)) != 0

    def is_init_fn(self):
        return self.nType == ADVNodeType.FRAGMENT and (self.flags & int(ADVNodeFlags.IS_METHOD)) != 0 and self.name == "__init__"

    def is_auto_field_fn(self):
        return self.nType == ADVNodeType.FRAGMENT and (self.flags & int(ADVNodeFlags.IS_AUTO_FIELD_FN)) != 0

    def is_dataclass_node(self):
        return self.nType == ADVNodeType.CLASS and (self.flags & int(ADVNodeFlags.IS_DATACLASS)) != 0

    def is_inherited_node(self):
        return self.nType == ADVNodeType.FRAGMENT and (self.flags & int(ADVNodeFlags.IS_INHERITED_NODE)) != 0

    def get_out_indicator_alias(self):
        assert self.nType == ADVNodeType.OUT_INDICATOR
        return self.name

    @staticmethod
    def get_global_uid_ext(path: list[str], id: str):
        # TODO should we use node id list instead of names + [last_id]？
        return UniqueTreeIdForTree.from_parts(path + [id]).uid_encoded

    def get_global_uid(self):
        # TODO should we use node id list instead of names + [last_id]？
        return UniqueTreeIdForTree.from_parts(self.path + [self.id]).uid_encoded

    def get_ref_global_uid(self):
        assert self.ref is not None
        return UniqueTreeIdForTree.from_parts(self.ref.import_path + [self.ref.node_id]).uid_encoded

    def get_inherit_ref_global_uid(self):
        assert self.cls_inherit_ref is not None
        return UniqueTreeIdForTree.from_parts(self.cls_inherit_ref.import_path + [self.cls_inherit_ref.node_id]).uid_encoded

    def is_local_ref_node(self):
        if self.ref is None:
            return False
        return len(self.ref.import_path) == len(self.path) and all(x == y for x, y in zip(self.ref.import_path, self.path))

    def get_child_node_paths(self, new_node: Self) -> tuple[list[str], list[str]]:
        new_path = self.path + [self.name]
        new_frontend_path = self.frontend_path + ["flow", "nodes", new_node.id]
        return new_path, new_frontend_path

    @staticmethod 
    def extract_path_and_id(gid: str) -> tuple[list[str], str]:
        uti = UniqueTreeIdForTree(gid)
        parts = uti.parts
        return parts[:-1], parts[-1]


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

    @staticmethod
    def get_code_relative_path_static(path: list[str], is_folder: bool):
        # if this node don't have nested flow, it use single file.
        # otherwise a folder with __init__.py
        if not path:
            # ROOT flow, always folder, __init__.py
            return Path(f"{TENSORPC_ADV_FOLDER_FLOW_NAME}.py")
        import_path = path.copy()
        if is_folder:
            import_path.append(f"{TENSORPC_ADV_FOLDER_FLOW_NAME}.py")
        else:
            import_path[-1] += ".py"
        return Path(*import_path)

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
        """Get node and its parent node by frontend path.
        """
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

    @staticmethod
    def get_flow_node_by_fe_path_v2(root_flow: ADVFlowModel, frontend_path: list[str]) -> Optional[tuple[ADVFlowModel, Optional[ADVNodeModel], ADVNodeModel]]:
        """Get node and its parent node by frontend path.
        """
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
        return (cur_parent[0], cur_parent[1], cur_node)

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

    def update_ref_path(self, node_gid_to_frontend_path: dict[str, list[str]]):
        def _update(node: ADVNodeModel, path: list[str]):
            if node.ref is not None:
                ref_gid = node.get_ref_global_uid()
                assert ref_gid in node_gid_to_frontend_path, f"node {ref_gid} not found in {path}"
                node.ref.fe_path = node_gid_to_frontend_path[ref_gid]
            if node.flow is not None:
                for n_id, n in node.flow.nodes.items():
                    _update(n, path + [node.id])
        for n_id, n in self.flow.nodes.items():
            _update(n, [])
        return 

    @staticmethod 
    def get_node_id_path_from_fe_path(fe_path: list[str]) -> list[str]:
        return fe_path[1::3]

    def draft_get_real_node_by_id(self,
                        node_gid: str):
        node_fe_path = self.node_gid_to_frontend_path[node_gid]
        node: Optional[ADVNodeModel] = D.getitem_path_dynamic(self.flow, node_fe_path, Optional[ADVNodeModel])
        
        real_node: Optional[ADVNodeModel] = D.where(
            D.logical_and(node != None, node.ref == None),
            node,
            D.getitem_path_dynamic(self.flow, node.ref.fe_path, Optional[ADVNodeModel]),
            return_type=Optional[ADVNodeModel])  # type: ignore
        return real_node

    def draft_get_node_by_id(self,
                        node_gid: str):
        node_fe_path = self.node_gid_to_frontend_path[node_gid]
        node: Optional[ADVNodeModel] = D.getitem_path_dynamic(self.flow, node_fe_path, Optional[ADVNodeModel])
        return node

    def draft_get_node_by_fe_path(self, node_fe_path: list[str]):
        if not isinstance(node_fe_path, DraftBase):
            node_fe_path = D.literal_val(node_fe_path)
        node: ADVNodeModel = D.getitem_path_dynamic(self.flow, node_fe_path, ADVNodeModel)
        return node

    def draft_get_node_by_id_path(self, node_id_path: list[str]):
        assert not isinstance(node_id_path, DraftBase)
        assert len(node_id_path) > 0
        res = self.flow.nodes[node_id_path[0]]
        for node_id in node_id_path[1:]:
            res = res.flow.nodes[node_id]
        return res

    def draft_get_node_impl_editor(self,
                        node_id: str):
        real_node = self.draft_get_real_node_by_id(node_id)
        
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

    @pfl.mark_pfl_compilable
    @staticmethod
    def get_pane_context_menu_items(cur_flow_ntype: int) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = [
            {
                "id": ADVPaneContextMenu.AddRef.value,
            },
            {
                "id": "divider0",
                "divider": True,
            },
            {
                "id": ADVPaneContextMenu.AddFragment.value,
            },
            {
                "id": ADVPaneContextMenu.AddGlobalScript.value,
            },
            {
                "id": ADVPaneContextMenu.AddSymbolGroup.value,
            },
            {
                "id": ADVPaneContextMenu.AddOutput.value,
            },
            {
                "id": ADVPaneContextMenu.AddMarkdown.value,
            },
            {
                "id": ADVPaneContextMenu.AddInlineFlowDesc.value,
            },
        ]
        if cur_flow_ntype != ADVNodeType.CLASS:
            items.extend([
                {
                    "id": "divider_class",
                    "divider": True,
                },
                {
                    "id": ADVPaneContextMenu.AddNestedFragment.value,
                },
                {
                    "id": ADVPaneContextMenu.AddClass.value,
                },

            ])
        items.extend([
            {
                "id": "divider_final",
                "divider": True,
            },
            {
                "id": ADVPaneContextMenu.Debug.value,
            },
        ])
        return items

    @mui.DataModel.mark_pfl_query_func
    def get_cur_flow_props(self) -> dict[str, Any]:
        cur_proj = self.adv_projects[self.cur_adv_project]
        # cur_flow = cast(Optional[ADVFlowModel], pfl.js.Common.getItemPath(
        #     cur_proj.flow, cur_proj.cur_path))
        cur_flow: Optional[ADVFlowModel] = pfl.js.Common.getItemPath(
            cur_proj.flow, cur_proj.cur_path)
        res: dict[str, Any] = {
            "selectedNode": None,
            "enableCodeEditor": False,
            "paneMenuItems": [],
        }
        if cur_flow is None:
            return res
        cur_flow_parent_type = ADVNodeType.FRAGMENT
        if len(cur_proj.cur_path) >= 1:
            cur_flow_parent: Optional[ADVNodeModel] = pfl.js.Common.getItemPath(
                cur_proj.flow, cur_proj.cur_path[:-1])
            if cur_flow_parent is not None:
                cur_flow_parent_type = cur_flow_parent.nType
        res["paneMenuItems"] = self.get_pane_context_menu_items(cur_flow_parent_type)
        selected_node_ids = cur_flow.selected_nodes
        if len(selected_node_ids) == 1:
            selected_node = cur_flow.nodes[selected_node_ids[0]]
            if selected_node.ref is not None and selected_node.ref.fe_path is not None:
                real_node: Optional[ADVNodeModel] = pfl.js.Common.getItemPath(
                    cur_proj.flow, selected_node.ref.fe_path)
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
            if node.ref is not None and node.ref.fe_path is not None:
                node: Optional[ADVNodeModel] = pfl.js.Common.getItemPath(
                    cur_proj.flow, node.ref.fe_path)
                node_is_ref = True
        return node, node_is_ref

    @mui.DataModel.mark_pfl_func
    def get_real_node_pair_by_gid(self, node_gid: str) -> tuple[Optional[ADVNodeModel], Optional[ADVNodeModel], Optional[ADVNodeModel], bool]:
        cur_proj = self.adv_projects[self.cur_adv_project]
        node_frontend_path = cur_proj.node_gid_to_frontend_path[node_gid]
        node: Optional[ADVNodeModel] = pfl.js.Common.getItemPath(
            cur_proj.flow, node_frontend_path)
        node_is_ref = False
        real_node: Optional[ADVNodeModel] = node
        real_node_parent: Optional[ADVNodeModel] = None
        if node is not None:
            ref = node.ref
            if ref is not None:
                fe_path = ref.fe_path
                if fe_path is not None:
                    real_node: Optional[ADVNodeModel] = pfl.js.Common.getItemPath(
                        cur_proj.flow, fe_path)
                    if len(fe_path) >= 3:
                        real_node_parent: Optional[ADVNodeModel] = pfl.js.Common.getItemPath(
                            cur_proj.flow, fe_path[:-3])

                node_is_ref = True
        return node, real_node, real_node_parent, node_is_ref

    @mui.DataModel.mark_pfl_query_nested_func
    def get_handle(self, paths: list[Any], node_gid: str) -> dict[str, Any]:
        node, real_node, real_node_parent, real_node_is_ref = self.get_real_node_pair_by_gid(node_gid)
        res: dict[str, Any] = {}
        if node is not None and real_node is not None:
            handle_idx: int = paths[0]
            # paths may contains invalid index. see doc of `mark_pfl_query_nested_func`.
            if handle_idx >= len(node.handles):
                return res
            handle = node.handles[handle_idx]
            # print(node.id, handle.id, handle_idx, handle.name)
            is_input = (handle.flags & ADVHandleFlags.IS_INPUT) != 0
            is_sym_handle = (handle.flags & ADVHandleFlags.IS_SYM_HANDLE) != 0
            is_error_missing = (handle.flags & ADVHandleFlags.ERROR_IS_MISSING) != 0
            is_self = (handle.flags & ADVHandleFlags.IS_METHOD_SELF) != 0
            if handle.dict_key:
                # we need to show original key if dict output.
                # otherwise users won't know the real key
                # when they use this function in non-adv code.
                name = handle.name + "(" + handle.dict_key + ")"
            else:
                name = handle.name
            if is_sym_handle and not is_self:
                name += ": "
                name += handle.type
            # if real_node_is_ref:
            #     print(node_id, real_node.id, handle)
            res = {
                "id": handle.id,
                "name": name,
                "type_anno": handle.type,
                "type": "target" if is_input else "source",
                "hpos": "left" if is_input else "right",
                "textAlign": "start" if (is_input or is_sym_handle) else "end",
                "is_input": is_input,
                "tooltip": handle.type,
            }
            if is_self:
                res["textColor"] = CodeStyles.VscodeClassColorLight
            if is_error_missing:
                res["outline"] = "3px solid red"
            if is_input:
                res["hborder"] = "1px solid #4caf50"
        return res

    def get_tags_from_node(self, node: ADVNodeModel) -> list[dict[str, Any]] :
        tags: list[dict[str, Any]] = []
        flags = node.flags 
        is_method = (flags & ADVNodeFlags.IS_METHOD) != 0
        is_cls_method = (flags & ADVNodeFlags.IS_CLASSMETHOD) != 0
        is_static_method = (flags & ADVNodeFlags.IS_STATICMETHOD) != 0
        is_inherited = (flags & ADVNodeFlags.IS_INHERITED_NODE) != 0
        is_init_fn = is_method and node.name == "__init__"
        if is_method:
            tags.append({
                "id": "M",
                "tooltip": "Method",
            })
        elif is_cls_method:
            tags.append({
                "id": "CM",
                "tooltip": "Class Method",
            })
        elif is_static_method:
            tags.append({
                "id": "SM",
                "tooltip": "Static Method",
            })
        if is_inherited:
            tags.append({
                "id": "I",
                "tooltip": "Inherited Fragment",
            })
        if is_init_fn:
            tags.append({
                "id": "INIT",
                "tooltip": "__init__",
            })

        return tags

    @mui.DataModel.mark_pfl_query_nested_func
    def get_tag(self, paths: list[Any], node_gid: str) -> dict[str, Any]:
        node, real_node, real_node_parent, real_node_is_ref = self.get_real_node_pair_by_gid(node_gid)
        res: dict[str, Any] = {}
        if node is not None and real_node is not None:
            tag_idx: int = paths[0]
            # determine number of tags
            tags = self.get_tags_from_node(real_node)
            # paths may contains invalid index. see doc of `mark_pfl_query_nested_func`.
            if tag_idx >= len(tags):
                return res
            tag = tags[tag_idx]
            res = {
                "id": tag["id"],
                "label": tag["id"],
                "tooltip": tag["tooltip"],
            }
        return res

    @mui.DataModel.mark_pfl_query_func
    def get_node_frontend_props(self, node_gid: str) -> dict[str, Any]:
        node, real_node, real_node_parent, real_node_is_ref = self.get_real_node_pair_by_gid(node_gid)
        res: dict[str, Any] = {}
        if real_node is not None and node is not None:
            header = real_node.name
            header_color: Union[Undefined, str] = undefined
            if real_node.nType == ADVNodeType.CLASS:
                icon_type = mui.IconType.DataObject
            elif real_node.nType == ADVNodeType.FRAGMENT:
                # ref node name is alias

                header = node.name
                if real_node.flow is not None:
                    icon_type = mui.IconType.Reactflow
                else:
                    icon_type = mui.IconType.Code
            else:
                icon_type = mui.IconType.Info
            if real_node.nType == ADVNodeType.CLASS:
                header_color = CodeStyles.VscodeClassColorLight
            elif real_node.nType == ADVNodeType.SYMBOLS:
                header_color = CodeStyles.VscodeClassColorLight

            tags = self.get_tags_from_node(real_node)
            is_inline_flow_desc = node.flags & int(ADVNodeFlags.IS_INLINE_FLOW_DESC) != 0
            is_inline_flow_desc_def = is_inline_flow_desc and not real_node_is_ref

            if is_inline_flow_desc_def:
                header_color = CodeStyles.VscodeKeywordColorLight
            res = {
                "id": real_node.id,
                "header": header,
                "iconType": icon_type,
                "isRef": real_node_is_ref,
                "bottomMsg": "hello world!",
                # method def don't have self, but
                # method ref have self handle.
                "handles": node.handles,
                "tags": tags,
                "hasTag": len(tags) > 0,

                "headerColor": header_color,
                # "htype": "target" if is_input else "source",
                # "hpos": "left" if is_input else "right",
                # "textAlign": "start" if is_input else "end",
            }
            if is_inline_flow_desc_def:
                inlinesf_name = node.name
                if node.ref is None:
                    res["headerColor"] = pfl.js.ColorUtil.getPerfettoColor(inlinesf_name).base.cssString
            else:
                inlinesf_name = node.inlinesf_name
            if inlinesf_name is not None:
                res["ifColor"] = pfl.js.ColorUtil.getPerfettoColor(inlinesf_name).base.cssString
                res["bottomMsg"] = inlinesf_name
                if not is_inline_flow_desc_def:
                    res["isMainFlow"] = True

            # output indicator props
            if real_node.nType == ADVNodeType.OUT_INDICATOR:
                if len(real_node.handles) == 0:
                    res["header"] = "..."
                else:
                    first_handle = real_node.handles[0]
                    out_var_name = first_handle.out_var_name
                    if not out_var_name:
                        res["header"] = "..."
                    else:
                        if real_node.name.strip() != "":
                            res["header"] = out_var_name + "->" + real_node.name.strip()
                        else:
                            res["header"] = out_var_name
            if real_node.nType == ADVNodeType.MARKDOWN and real_node.impl is not None:
                res["code"] = real_node.impl.code
            # if is_input:
            #     res["hborder"] = "1px solid #4caf50"
        return res
