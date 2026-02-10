import ast
import dataclasses as dataclasses_plain
import enum
import hashlib
import heapq
import inspect
import shutil
from pathlib import Path
from typing import (Any, Callable, Literal, Optional, Self, TypeGuard, TypeVar,
                    Union)

import rich

import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.apps.adv import api as _ADV
from tensorpc.apps.adv.codemgr import markers as adv_markers
from tensorpc.apps.adv.codemgr.core import (BackendHandle, BackendNode,
                                            BaseParser, BaseParseResult,
                                            ImplCodeSpec)
from tensorpc.apps.adv.codemgr.fragment import (FragmentParser,
                                                FragmentParseResult,
                                                parse_alias_map)
from tensorpc.apps.adv.codemgr.misc import (GlobalScriptParseResult,
                                            MarkdownParseResult,
                                            OutIndicatorParseResult)
from tensorpc.apps.adv.codemgr.proj_parse import ADVProjectParser
from tensorpc.apps.adv.codemgr.symbols import SymbolParser, SymbolParseResult
from tensorpc.apps.adv.config import ADVNodeCMConfig
from tensorpc.apps.adv.constants import (TENSORPC_ADV_FOLDER_FLOW_NAME,
                                         TENSORPC_ADV_ROOT_FLOW_ID)
from tensorpc.apps.adv.logger import ADV_LOGGER
from tensorpc.apps.adv.model import (ADVConstHandles, ADVEdgeModel,
                                     ADVFlowModel, ADVHandleFlags,
                                     ADVHandlePrefix, ADVNodeFlags,
                                     ADVNodeHandle, ADVNodeModel,
                                     ADVNodeRefInfo, ADVNodeType, ADVProject)
from tensorpc.constants import PACKAGE_ROOT
from tensorpc.core.annolib import Undefined, is_undefined, undefined
from tensorpc.core.funcid import clean_source_code, get_attribute_name
from tensorpc.dock.jsonlike import camel_to_snake
from tensorpc.utils.uniquename import UniqueNamePool

_NTYPE_TO_ID_PREFIX = {
    ADVNodeType.FRAGMENT: "Frag",
    ADVNodeType.SYMBOLS: "Sym",
    ADVNodeType.GLOBAL_SCRIPT: "GS",
    ADVNodeType.OUT_INDICATOR: "Out",

}
_NODE_ALLOW_REF = set([ADVNodeType.GLOBAL_SCRIPT, ADVNodeType.SYMBOLS, ADVNodeType.FRAGMENT, ADVNodeType.CLASS])


@dataclasses.dataclass
class CodeBlock:
    lineno: int 
    column_offset: int  

T = TypeVar("T")

class NodeChangeFlag(enum.IntFlag):
    NAME = enum.auto()
    HANDLES = enum.auto()
    INLINE_FLOW = enum.auto()
    ALIAS_MAP = enum.auto()
    GLOBAL_SCRIPT = enum.auto()
    NEW = enum.auto() # indicate is new node
    ERROR_STATUS = enum.auto() # indicate error status changed
    IMPL_CODE = enum.auto()
    REF = enum.auto() # ref changed
    FLAGS = enum.auto() # node flags changed

@dataclasses.dataclass
class ADVNodeChange:
    name: str # only used for debug verbose.
    flags: NodeChangeFlag
    parse_res: BaseParseResult

    @staticmethod 
    def empty(name: str, parse_res: BaseParseResult, is_new: bool, error_status_changed: bool = False):
        flags = NodeChangeFlag(0)
        if is_new:
            flags |= NodeChangeFlag.NEW
        if error_status_changed:
            flags |= NodeChangeFlag.ERROR_STATUS
        return ADVNodeChange(name=name, flags=flags, parse_res=parse_res)

@dataclasses.dataclass
class ADVProjectChange:
    node_changes: dict[str, ADVNodeChange] = dataclasses.field(default_factory=dict)
    new_flow_edges: dict[str, list[ADVEdgeModel]] = dataclasses.field(default_factory=dict)
    flow_code_changes: dict[str, str] = dataclasses.field(default_factory=dict)
    changed_node_ids_in_cur_flow: list[str] = dataclasses.field(default_factory=list)
    new_node_gid_to_path: Optional[dict[str, list[str]]] = None 
    new_node_gid_to_frontend_path: Optional[dict[str, list[str]]] = None

    node_gid_to_path_update: Optional[dict[str, list[str]]] = None 
    node_gid_to_frontend_path_update: Optional[dict[str, list[str]]] = None 

    deleted_node_pairs: Optional[list[tuple[str, str]]] = None

    def is_empty(self):
        return (len(self.node_changes) == 0 and
                len(self.new_flow_edges) == 0 and
                len(self.flow_code_changes) == 0 and
                len(self.changed_node_ids_in_cur_flow) == 0 and
                (self.deleted_node_pairs is None or len(self.deleted_node_pairs) == 0))


    def get_short_repr(self):
        node_changes: dict[str, NodeChangeFlag] = {}
        for node_gid, change in self.node_changes.items():
            if change.name != "":
                new_name = f"{node_gid}({change.name})"
            else:
                new_name = f"{node_gid}"
            node_changes[new_name] = change.flags
        flow_id_to_new_edge_ids: dict[str, list[str]] = {}
        for flow_gid, edges in self.new_flow_edges.items():
            if flow_gid == "":
                flow_gid = "__ROOT__"
            flow_id_to_new_edge_ids[flow_gid] = [e.id for e in edges]
        changed_flow_paths = list(self.flow_code_changes.keys())
        res = {
            "node_changes": node_changes,
            "new_flow_edges": flow_id_to_new_edge_ids,
            "flow_code_changes": changed_flow_paths,
            "changed_node_ids_in_cur_flow": self.changed_node_ids_in_cur_flow,
        }
        if self.deleted_node_pairs is not None:
            res["deleted"] = self.deleted_node_pairs
        return res 


@dataclasses.dataclass
class ExtNodeDesc:
    node: ADVNodeModel
    node_def: ADVNodeModel
    type: Literal["ref", "subflow"]
    is_local_ref: bool

@dataclasses.dataclass
class InlineFlowParseResult:
    node_descs: list[BackendNode]
    input_handles: list[BackendHandle]
    output_handles: list[BackendHandle]
    desc_node: Optional[BackendNode]

@dataclasses.dataclass
class FlowConnInternals:
    inp_node_handle_to_node: dict[tuple[str, str], tuple[ADVNodeModel, BackendHandle]]
    node_id_to_inp_handles: dict[str, list[BackendHandle]]
    out_node_handle_to_node: dict[tuple[str, str], tuple[ADVNodeModel, BackendHandle]]
    node_id_to_out_handles: dict[str, list[BackendHandle]]
    auto_edges: list[ADVEdgeModel]

@dataclasses.dataclass
class FlowCodeFragment:
    code: str 
    path: str 
    code_range: tuple[int, int, int, int]
    is_ref_node: bool = False

@dataclasses.dataclass
class InlineFlowDesc:
    desc_node: Optional[BackendNode] # None for default inline flow
    flow_be_nodes: list[BackendNode]

@dataclasses.dataclass(kw_only=True)
class InlineFlowNodeParseResult(FragmentParseResult):
    pass

@dataclasses.dataclass(kw_only=True)
class FlowParseResult(FragmentParseResult):
    edges: list[ADVEdgeModel]
    flow_conn: FlowConnInternals
    symbol_dep_qnames: list[str] = dataclasses.field(default_factory=list)
    symbol_dep_classes: list[ADVNodeModel] = dataclasses.field(default_factory=list)

    isolated_be_nodes: list[BackendNode] = dataclasses.field(default_factory=list)

    inlineflow_results: dict[str, InlineFlowParseResult] = dataclasses.field(default_factory=dict)
    misc_be_nodes: dict[ADVNodeType, list[BackendNode]] = dataclasses.field(default_factory=dict)
    ext_be_nodes: list[BackendNode] = dataclasses.field(default_factory=list)
    generated_code_lines: list[str] = dataclasses.field(default_factory=list)
    generated_code: str = ""
    has_subflow: bool = False
    # class specific fields
    auto_field_be_node: Optional[BackendNode] = None
    base_inherited_be_node: Optional[BackendNode] = None
    
    def is_class(self):
        if self.node is None:
            return False 
        return self.node.nType == ADVNodeType.CLASS

    def _is_folder(self):
        if self.node is None:
            return True 
        return self.has_subflow

    def get_code_relative_path(self):
        if self.node is None:
            return ADVProject.get_code_relative_path_static([], True)
        return ADVProject.get_code_relative_path_static(self.node.path.copy() + [self.node.name], self.has_subflow)

    def get_path_list(self) -> list[str]:
        if self.node is None:
            return []
        return self.node.path + [self.node.name]

    def get_ref_dep_lines(self):
        if self.node is None:
            # root flow
            cur_import_path = []
        else:
            cur_import_path = self.node.path + [self.node.name]
        if self._is_folder():
            # add __adv_flow__.py to cur_import_path
            dot_prefix = "." * (len(cur_import_path) + 1)
        else:
            # import path don't contains cur file, so add xxx.py to cur_import_path
            dot_prefix = "." * (len(cur_import_path))

        ref_dep_lines: list[str] = []
        import_stmt_set: set[str] = set()
        for node_desc in self.ext_be_nodes:
            node = node_desc.node
            import_stmt = node_desc.get_import_stmt(dot_prefix)
            if import_stmt is not None:
                import_stmt_set.add(import_stmt)
        if self.base_inherited_be_node is not None:
            assert self.node is not None and self.node.nType == ADVNodeType.CLASS
            # add import stmt for base class
            import_stmt = self.base_inherited_be_node.get_import_stmt(dot_prefix)
            if import_stmt is not None:
                import_stmt_set.add(import_stmt)
        for cls_node in self.symbol_dep_classes:
            import_stmt = BackendNode.get_class_import_stmt(cls_node, dot_prefix)
            import_stmt_set.add(import_stmt)
        for import_stmt in import_stmt_set:
            ref_dep_lines.append(import_stmt)

        for node_desc in self.ext_be_nodes:
            node = node_desc.node
            # add marker to mark node id and position
            kwarg_parts = self.get_node_meta_kwargs(node)
            # if node_desc.type == "subflow":
            #     kwarg_parts.append(f"is_subflow=True")
            if node.inlinesf_name is not None:
                kwarg_parts.append(f'inlineflow_name="{node.inlinesf_name}"')
            if node_desc.is_subflow_def and node_desc.is_node_def_folder:
                kwarg_parts.append(f'is_folder=True')
            # ref import path is embedded to import stmt, so no need to save to marker here.
            kwargs_str = ", ".join(kwarg_parts)
            node_desc.is_node_def_folder
            if node_desc.is_subflow_def:
                if node_desc.is_class_node:
                    mark_stmt = f"ADV.{adv_markers.mark_class_node.__name__}(name=\"{node_desc.node_def.name}\", {kwargs_str})"
                else:
                    mark_stmt = f"ADV.{adv_markers.mark_subflow_node.__name__}(name=\"{node_desc.node_def.name}\", {kwargs_str})"
            else:
                mark_stmt = f"ADV.{adv_markers.mark_ref_node.__name__}({node_desc.get_qualname_from_import()}, {node.nType}, flags={int(node.flags)}, {kwargs_str})"
            ref_dep_lines.append(mark_stmt)
        res_lines: list[str] = []
        if ref_dep_lines:
            res_lines.append("# ------ ADV Ref/Subflow Nodes Dependency Region (Optional) ------")
            res_lines.append(f"ADV.{adv_markers.mark_ref_node_dep.__name__}()")
            res_lines.extend(ref_dep_lines)
            res_lines.append(f"ADV.{adv_markers.mark_ref_node_dep_end.__name__}()")
            res_lines.append(f"")

        return res_lines

    def _create_lines_for_frag_or_sym(self, descs: list[BackendNode], base_lineno: int, defined_in_class: Optional[bool] = None):
        res_lines: list[str] = []
        for node_desc in descs:
            node = node_desc.node
            if node.ref is not None:
                continue 
            if defined_in_class is not None:
                if node.is_defined_in_class() == (not defined_in_class):
                    continue
            if isinstance(node_desc.parse_res, (FragmentParseResult, SymbolParseResult)):
                parse_res = node_desc.parse_res
            else:
                raise RuntimeError("Invalid parse result type.")
            parse_res.lineno = base_lineno + len(res_lines)
            code_spec = parse_res.to_code_lines()
            parse_res.loc = code_spec

            res_lines.extend(code_spec.lines)
        return res_lines

    def to_code_lines(self):
        self_is_class = False 
        class_name = ""
        if self.node is not None:
            self_is_class = self.node.nType == ADVNodeType.CLASS
            class_name = self.node.name
        user_edges = [e for e in self.edges if not e.isAutoEdge]
        flow_name = self.node.name if self.node is not None else "__ADV_ROOT__"
        flow_import_path = self.get_path_list()
        lines = [
            f"# ADV Flow Definition. name: {flow_name}, import path: {flow_import_path}, relative fspath: {self.get_code_relative_path()}",
            "from tensorpc.apps.adv import api as ADV",
            "import dataclasses",
            "import typing",
            "import collections",
            "import typing_extensions",
        ]
        lines.extend(SymbolParser.get_typing_import_stmts())
        # if self.node is not None:
            # mark parent node meta
        # global scripts
        # from ....apps import adv
        gs_lines_all: list[str] = []
        if self.misc_be_nodes[ADVNodeType.GLOBAL_SCRIPT]:
            gs_lines_all.insert(0, "# ------ ADV Global Script Region ------")

        for be_node in self.misc_be_nodes[ADVNodeType.GLOBAL_SCRIPT]:
            parse_res = be_node.get_parse_res_checked(GlobalScriptParseResult)
            parse_res.lineno = len(lines) + len(gs_lines_all) + 1
            code_spec = parse_res.to_code_lines()
            parse_res.loc = code_spec
            # code_spec.
            gs_lines_all.extend(code_spec.lines)
        lines.extend(gs_lines_all)
        # symbol deps
        symbol_dep_lines: list[str] = []
        if self.symbol_dep_qnames:
            symbol_dep_lines.append("# ------ ADV Symbol Dependency Region ------")
            symbol_dep_lines.append(f"ADV.{adv_markers.mark_symbol_dep.__name__}()")
            for qname in self.symbol_dep_qnames:
                parts = qname.split(".")
                import_line = f"from {'.'.join(parts[:-1])} import {parts[-1]}"
                symbol_dep_lines.append(import_line)
            symbol_dep_lines.append(f"ADV.{adv_markers.mark_symbol_dep_end.__name__}()")
        lines.extend(symbol_dep_lines)
        # ref deps
        ref_dep_lines: list[str] = []
        if self.ext_be_nodes:
            ref_dep_lines = self.get_ref_dep_lines()
        lines.extend(ref_dep_lines)
        # isolate fragments
        isolate_fragment_lines: list[str] = self._create_lines_for_frag_or_sym(self.isolated_be_nodes, len(lines) + 1, defined_in_class=False)
        lines.extend(isolate_fragment_lines)
        # inline flow nodes not in class
        for inline_name, inline_desc in self.inlineflow_results.items():
            inline_ext_lines = self._create_lines_for_frag_or_sym(inline_desc.node_descs, len(lines) + 1, defined_in_class=False)
            lines.extend(inline_ext_lines)
        # class def start
        impl_spec = ImplCodeSpec(
            [], -1, -1, -1, -1
        )
        if self_is_class:
            assert self.node is not None 
            impl_spec = ImplCodeSpec(
                [], 0, 1, -1, -1
            )
            self.lineno = len(lines) + 1
            if self.base_inherited_be_node is not None:
                inode_id = self.base_inherited_be_node.id
                lines.append(f"@ADV.{adv_markers.mark_class_def.__name__}(inherit_node_id={repr(inode_id)})")
            else:
                lines.append(f"@ADV.{adv_markers.mark_class_def.__name__}()")
            if self.node.is_dataclass_node():
                lines.append("@dataclasses.dataclass(kw_only=True)")
            inherits: list[str] = []
            if self.base_inherited_be_node is not None:
                cls_name = self.base_inherited_be_node.node_def.name
                inherits.append(cls_name)
            if self.node is not None and not is_undefined(self.node.ext_inherits):
                inherits.extend(self.node.ext_inherits)
            if inherits:
                cls_names = ", ".join(inherits)
                lines.append(f"class {class_name}({cls_names}):")
            else:
                lines.append(f"class {class_name}:")
            if self.auto_field_be_node is not None:
                auto_field_parse_res = self.auto_field_be_node.parse_res
                assert auto_field_parse_res is not None and isinstance(auto_field_parse_res, FragmentParseResult)
                if self.node.is_dataclass_node():
                    # generate fields from auto field fn
                    lines.extend(auto_field_parse_res.create_field_lines_if_auto_field())
                else:
                    lines.extend(auto_field_parse_res.create_init_fn_lines_if_auto_field())
        # all inline flow in class is method.
        isolate_fragment_lines = self._create_lines_for_frag_or_sym(self.isolated_be_nodes, len(lines) + 1, defined_in_class=True)
        lines.extend(isolate_fragment_lines)

        # subflow nodes
        inlineflow_lines: list[str] = []

        for inline_name, inline_desc in self.inlineflow_results.items():
            inline_ext_lines = self._create_lines_for_frag_or_sym(inline_desc.node_descs, len(lines) + len(inlineflow_lines) + 1, defined_in_class=True)
            inlineflow_lines.extend(inline_ext_lines)
            # build subflow
            func_name = inline_name
            if inline_desc.desc_node is None:
                inlineflow_fn_lines = [
                    f"@ADV.{adv_markers.mark_inlineflow.__name__}()",
                    f"def {func_name}(",
                ]
                if self_is_class:
                    # add self
                    inlineflow_fn_lines.append(f"    self,")
                inlineflow_fn_lines += FragmentParseResult.get_signature_lines_from_handles(inline_desc.input_handles)
                inlineflow_fn_lines.append(f") -> dict[str, Any]:")
            else:
                assert inline_desc.desc_node.node.is_inline_flow_desc()
                inlineflow_fn_lines = inline_desc.desc_node.get_parse_res_checked(FragmentParseResult).get_code_lines_without_body(create_indent=False)
            body_lines: list[str] = []
            for node_desc in inline_desc.node_descs:
                node = node_desc.node
                parse_res = node_desc.get_parse_res_checked(FragmentParseResult)
                func_call_str = node_desc.get_func_call_expr(
                    parse_res.input_handles,
                    self.flow_conn.out_node_handle_to_node,
                )
                type_str = None
                if node.nType == ADVNodeType.CLASS:
                    type_str = node_desc.node_def.name
                elif node.is_class_method() and node_desc.node_def_parent is not None:
                    type_str = node_desc.node_def_parent.name
                anno = node_desc.get_ref_node_meta_anno_str(type_str)
                if parse_res.out_type == "single" or parse_res.out_type == "self":
                    var_name = parse_res.output_handles[0].name
                    body_lines.append(f"{var_name}{anno} = {func_call_str}")
                elif parse_res.out_type == "tuple":
                    out_names = ", ".join([h.handle.name for h in parse_res.output_handles])
                    body_lines.append(f"{out_names}{anno} = {func_call_str}")
                else:
                    # dict
                    body_lines.append(f"_adv_tmp_out{anno} = {func_call_str}")
                    for h in parse_res.output_handles:
                        if isinstance(h.handle.dict_key, Undefined):
                            dict_key = h.handle.name 
                        else:
                            dict_key = h.handle.dict_key 

                        body_lines.append(f"{h.handle.name} = _adv_tmp_out['{dict_key}']")
            # subflow always return dict
            body_lines.append("return {")
            for h in inline_desc.output_handles:
                # output handle always connect to a out indicator node, so we use name instead of symbol_name
                var_name = h.handle.name
                if not isinstance(h.handle.out_var_name, Undefined):
                    var_name = h.handle.out_var_name
                body_lines.append(f"    '{h.handle.name}': {var_name},")
            body_lines.append("}")
            body_lines_indented = [f"    {line}" for line in body_lines]
            inlineflow_fn_lines.extend(body_lines_indented)
            if inline_desc.desc_node is None:
                if self_is_class:
                    inlineflow_fn_lines = [f"    {line}" for line in inlineflow_fn_lines]
            else:
                if inline_desc.desc_node.node.is_defined_in_class():
                    inlineflow_fn_lines = [f"    {line}" for line in inlineflow_fn_lines]

            inlineflow_lines.extend(inlineflow_fn_lines)
        lines.extend(inlineflow_lines)
        # symbol groups
        # we write symbol groups after possible class definition
        symbol_group_lines: list[str] = []
        if self.misc_be_nodes[ADVNodeType.SYMBOLS]:
            symbol_group_lines.append("# ------ ADV Symbol Def Region ------")
        for be_node in self.misc_be_nodes[ADVNodeType.SYMBOLS]:
            node = be_node.node
            if node.ref is not None:
                continue 
            parse_res = be_node.get_parse_res_checked(SymbolParseResult)
            parse_res.lineno = len(lines) + len(symbol_group_lines) + 1
            code_spec = parse_res.to_code_lines()
            parse_res.loc = code_spec
            symbol_group_lines.extend(code_spec.lines)
        lines.extend(symbol_group_lines)

        out_indicator_lines: list[str] = []
        if self.misc_be_nodes[ADVNodeType.OUT_INDICATOR]:
            out_indicator_lines.append("# ------ ADV Out Indicator Region ------")
        for be_node in self.misc_be_nodes[ADVNodeType.OUT_INDICATOR]:
            parse_res = be_node.get_parse_res_checked(OutIndicatorParseResult)
            parse_res.lineno = len(lines) + len(out_indicator_lines) + 1
            code_spec = parse_res.to_code_lines()
            parse_res.loc = code_spec
            out_indicator_lines.extend(code_spec.lines)
        if out_indicator_lines:
            lines.extend(out_indicator_lines)
        markdown_lines: list[str] = []
        if self.misc_be_nodes[ADVNodeType.MARKDOWN]:
            markdown_lines.append("# ------ ADV Markdown Region ------")
        for be_node in self.misc_be_nodes[ADVNodeType.MARKDOWN]:
            node = be_node.node
            assert node.impl is not None 
            content = node.impl.code
            assert not isinstance(node.width, Undefined) and not isinstance(node.height, Undefined)
            content_lines = content.splitlines()
            kwarg_parts = self.get_node_meta_kwargs(node)
            kwarg_parts.append(f'width={repr(node.width)}')
            kwarg_parts.append(f'height={repr(node.height)}')
            kwargs_str = ", ".join(kwarg_parts)
            if len(content_lines) >= 2:
                lines.append(f"ADV.{adv_markers.mark_markdown_node.__name__}({kwargs_str}, content=\"\"\"" + content_lines[0])
                lines.extend(content_lines[1:-1])
                lines.append(content_lines[-1] + "\"\"\")")
            else:
                lines.append(f"ADV.{adv_markers.mark_markdown_node.__name__}({kwargs_str}, content={repr(content)})")

        for edge in user_edges:
            lines.append(f"ADV.{adv_markers.mark_user_edge.__name__}("
                         f"id=\"{edge.id}\", "
                         f"source=\"{edge.source}\", "
                         f"source_handle=\"{edge.sourceHandle}\", "
                         f"target=\"{edge.target}\", "
                         f"target_handle=\"{edge.targetHandle}\")")
        lines.append("# ------ End of ADV Flow Definition ------")
        impl_spec.lines = lines
        return impl_spec

@dataclasses_plain.dataclass
class ModifyParseCache:
    visited: set[str]
    old_parse_res: dict[str, tuple[FlowParseResult, dict[str, BackendNode], dict[str, ADVEdgeModel]]]

@dataclasses_plain.dataclass
class FlowCache:
    flow: ADVFlowModel
    parser: "FlowParser"
    parent_node: Optional[ADVNodeModel]
    named_node_name_set: set[str] = dataclasses_plain.field(default_factory=set)
    node_id_pool: UniqueNamePool = dataclasses_plain.field(default_factory=UniqueNamePool)
    edge_id_pool: UniqueNamePool = dataclasses_plain.field(default_factory=UniqueNamePool)
    child_flow_gids: set[str] = dataclasses_plain.field(default_factory=set)
    dep_flow_gids: set[str] = dataclasses_plain.field(default_factory=set)
    all_child_flow_gids: set[str] = dataclasses_plain.field(default_factory=set)

    topological_sorted_index: int = -1

    def get_code_relative_path_checked(self):
        return self.parser.get_flow_parse_result_checked().get_code_relative_path()

    def get_flow_gid(self):
        if self.parent_node is None:
            return TENSORPC_ADV_ROOT_FLOW_ID
        return self.parent_node.get_global_uid()

@dataclasses_plain.dataclass
class NodeConnCache:
    node: ADVNodeModel 
    flow_node_gid: str
    ref_gids_external: list[str] = dataclasses_plain.field(default_factory=list)
    ref_node_ids_local: list[str] = dataclasses_plain.field(default_factory=list)
    ref_gids_local: list[str] = dataclasses_plain.field(default_factory=list)

class ADVProjectFSCache:
    def __init__(self):
        self._folders: set[Path] = set()
        self._path_to_code: dict[Path, str] = {}

    def compare_and_update(self, new_folders: set[Path], new_path_to_code: dict[Path, str]):
        deleted_folders = self._folders - new_folders
        new_folders = new_folders - self._folders
        deleted_files = set(self._path_to_code.keys()) - set(new_path_to_code.keys())
        new_files = set(new_path_to_code.keys()) - set(self._path_to_code.keys())
        modify_files: set[Path] = set()
        for path, code in new_path_to_code.items():
            if path in self._path_to_code and self._path_to_code[path] != code:
                modify_files.add(path)
        # find root folders in deleted_folders
        deleted_list = [(p, True) for p in deleted_folders]
        deleted_list += [(p, False) for p in deleted_files]
        deleted_list.sort(key=lambda pair: pair[0])
        root_delete_folders: set[Path] = set()
        standalone_delete_files: set[Path] = set()
        if deleted_list:
            # now we have /a, /a/b, /a/c, /d
            cur_root_folder: Optional[Path] = None
            for path, is_folder in deleted_list:
                if is_folder:
                    if cur_root_folder is None:
                        cur_root_folder = path 
                    else:
                        if not path.is_relative_to(cur_root_folder):
                            root_delete_folders.add(cur_root_folder)
                            cur_root_folder = path
                else:
                    if cur_root_folder is None:
                        standalone_delete_files.add(path)
                    else:
                        if not path.is_relative_to(cur_root_folder):
                            standalone_delete_files.add(path)
            if cur_root_folder is not None:
                root_delete_folders.add(cur_root_folder)
        # remove files
        for folder in root_delete_folders:
            shutil.rmtree(folder)

        for file in standalone_delete_files:
            file.unlink()

        # create new files
        for folder in new_folders:
            folder.mkdir(parents=True, exist_ok=True, mode=0o755)
        for file in new_files:
            file.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            with open(file, "w", encoding="utf-8") as f:
                f.write(new_path_to_code[file])
        # modify files
        for file in modify_files:
            # write iff content change
            if self._path_to_code[file] != new_path_to_code[file]:
                with open(file, "w", encoding="utf-8") as f:
                    f.write(new_path_to_code[file])
        self._folders = new_folders
        self._path_to_code = new_path_to_code
        return root_delete_folders, standalone_delete_files, new_files, modify_files


class ADVProjectBackendManager:
    def __init__(self, root_flow_getter: Callable[[], ADVProject], root_flow_draft: ADVFlowModel):
        self._root_flow_getter = root_flow_getter
        self._node_gid_to_cache: dict[str, NodeConnCache] = {}
        self._flow_node_gid_to_cache: dict[str, FlowCache] = {
            TENSORPC_ADV_ROOT_FLOW_ID: FlowCache(root_flow_getter().flow, FlowParser(), None)
        }
        self._path_to_flow_node_gid: dict[Path, str] = {}
        self._root_flow_draft = root_flow_draft

        self._fs_cache = ADVProjectFSCache()
        self._root_path = root_flow_getter().path

    def get_subflow_parser(self, node_with_subflow: ADVNodeModel) -> "FlowParser":
        assert node_with_subflow.flow is not None, "Node has no nested flow."
        node_gid = node_with_subflow.get_global_uid()
        return self._flow_node_gid_to_cache[node_gid].parser

    def _reflesh_dependency(self, node_gid_to_cache: dict[str, NodeConnCache], flow_node_gid_to_cache: dict[str, FlowCache]):
        for gid, cache in node_gid_to_cache.items():
            cache.ref_gids_external.clear()
            cache.ref_node_ids_local.clear()
            cache.ref_gids_local.clear()
        for gid, cache in node_gid_to_cache.items():
            if cache.node.ref is not None:
                ref_cache = node_gid_to_cache[cache.node.get_ref_global_uid()]
                if cache.node.is_local_ref_node():
                    ref_cache.ref_node_ids_local.append(cache.node.id)
                    ref_cache.ref_gids_local.append(gid)
                else:
                    ref_cache.ref_gids_external.append(gid)
            if cache.node.cls_inherit_ref is not None:
                ref_cache = node_gid_to_cache[cache.node.get_inherit_ref_global_uid()]
                assert not cache.node.is_local_ref_node()
                ref_cache.ref_gids_external.append(gid)
        for flow_gid, cache in flow_node_gid_to_cache.items():
            cache.child_flow_gids.clear()
            cache.dep_flow_gids.clear()
            cache.all_child_flow_gids.clear()
            cache.topological_sorted_index = -1
        for flow_gid, cache in flow_node_gid_to_cache.items():
            for node in cache.flow.nodes.values():
                if node.ref is not None:
                    ref_node_cache = node_gid_to_cache[node.get_ref_global_uid()]
                    dep_flow_gid = ref_node_cache.flow_node_gid
                    if dep_flow_gid != flow_gid: # remove local ref
                        dep_flow_cache = flow_node_gid_to_cache[dep_flow_gid]
                        dep_flow_cache.child_flow_gids.add(flow_gid)
                        cache.dep_flow_gids.add(dep_flow_gid)
                elif node.cls_inherit_ref is not None:
                    ref_node_cache = node_gid_to_cache[node.get_inherit_ref_global_uid()]
                    dep_flow_gid = ref_node_cache.flow_node_gid
                    if dep_flow_gid != flow_gid: # remove local ref
                        dep_flow_cache = flow_node_gid_to_cache[dep_flow_gid]
                        dep_flow_cache.child_flow_gids.add(flow_gid)
                        cache.dep_flow_gids.add(dep_flow_gid)
                elif node.flow is not None:
                    dep_flow_cache = flow_node_gid_to_cache[node.get_global_uid()]
                    dep_flow_cache.child_flow_gids.add(flow_gid)
                    cache.dep_flow_gids.add(node.get_global_uid())
        # topological sort (dfs) flow caches, generate and assign sorted_index.
        sorted_flows: list[str] = []
        visited: set[str] = set()
        cycle_visited: set[str] = set()
        def _dfs_flow(flow_gid: str):
            if flow_gid in visited:
                return 
            if flow_gid in cycle_visited:
                raise RuntimeError(f"Cycle detected in flow dependencies at flow {flow_gid}|{cycle_visited}.")
            cycle_visited.add(flow_gid)
            flow_cache = flow_node_gid_to_cache[flow_gid]
            for child_flow_gid in flow_cache.child_flow_gids:
                _dfs_flow(child_flow_gid)
            visited.add(flow_gid)
            sorted_flows.append(flow_gid)
        for flow_gid, fcache in flow_node_gid_to_cache.items():
            _dfs_flow(flow_gid)
        sorted_flows.reverse()
        for index, flow_gid in enumerate(sorted_flows):
            flow_node_gid_to_cache[flow_gid].topological_sorted_index = index

        # calc all_child_flow_gids. when we create ref nodes, we can't create cycles.
        visited = set()
        def _dfs_all_child(flow_gid: str) -> set[str]:
            if flow_gid in visited:
                return flow_node_gid_to_cache[flow_gid].all_child_flow_gids
            visited.add(flow_gid)
            flow_cache = flow_node_gid_to_cache[flow_gid]
            all_children = set(flow_cache.child_flow_gids)
            for child_flow_gid in flow_cache.child_flow_gids:
                all_children.update(_dfs_all_child(child_flow_gid))
            flow_cache.all_child_flow_gids = all_children
            return all_children
        for flow_gid in flow_node_gid_to_cache.keys():
            _dfs_all_child(flow_gid)

    def _get_all_files_and_folders(self, is_relative: bool = False) -> tuple[set[Path], dict[Path, str]]:
        path_to_code: dict[Path, str] = {}
        folders: set[Path] = set()
        for flow_gid, fcache in self._flow_node_gid_to_cache.items():
            parser = fcache.parser
            assert parser._flow_parse_result is not None
            flow_parse_res = parser.get_flow_parse_result_checked()
            code_lines = flow_parse_res.to_code_lines().lines
            relative_path = flow_parse_res.get_code_relative_path()
            if is_relative:
                flow_path = relative_path
            else:
                flow_path = self._root_path / flow_parse_res.get_code_relative_path()
            path_to_code[flow_path] = "\n".join(code_lines)
            if flow_parse_res._is_folder():
                folders.add(flow_path.parent)
                path_to_code[flow_path.parent / "__init__.py"] = "# Auto-generated __init__.py for ADV flow folder.\n"
        return folders, path_to_code

    def _update_path_to_fcache(self):
        self._path_to_flow_node_gid.clear()
        for flow_gid, fcache in self._flow_node_gid_to_cache.items():
            parse_res = fcache.parser.get_flow_parse_result_checked()
            path = parse_res.get_code_relative_path()
            if path in self._path_to_flow_node_gid:
                raise RuntimeError(f"Flow path conflict: {path} used by both flow nodes {flow_gid} and {self._path_to_flow_node_gid[path]}.")
            self._path_to_flow_node_gid[path] = flow_gid

    def sync_project_model(self):
        # self._node_gid_to_node.clear()
        self._flow_node_gid_to_cache.clear()
        # self._node_gid_to_flow_node_gid.clear()
        root_flow = self._root_flow_getter().flow
        root_fcache = FlowCache(
            root_flow, 
            FlowParser(), 
            None
        )
        self._flow_node_gid_to_cache[TENSORPC_ADV_ROOT_FLOW_ID] = root_fcache
        def _traverse_node(node: ADVNodeModel):
            node_gid = node.get_global_uid()
            # self._node_gid_to_node[node_gid] = node
            if node.flow is not None:
                fcache = FlowCache(node.flow, FlowParser(), node)
                self._flow_node_gid_to_cache[node.get_global_uid()] = fcache
                # TODO check: name of symbol group/fragment/class must be unique in flow  
                for child_node in node.flow.nodes.values():
                    self._node_gid_to_cache[child_node.get_global_uid()] = NodeConnCache(child_node, node_gid)
                    # self._node_gid_to_flow_node_gid[child_node.get_global_uid()] = node_gid
                    if child_node.is_named_node() and not node.is_local_ref_node():
                        assert child_node.name not in fcache.named_node_name_set, f"Named node name conflict: {child_node.name} in flow node {node_gid}."
                        fcache.named_node_name_set.add(child_node.name)
                    fcache.node_id_pool(child_node.id)
                    _traverse_node(child_node)
        
        for node in root_flow.nodes.values():
            self._node_gid_to_cache[node.get_global_uid()] = NodeConnCache(node, TENSORPC_ADV_ROOT_FLOW_ID)
            # self._node_gid_to_flow_node_gid[node.get_global_uid()] = TENSORPC_ADV_ROOT_FLOW_ID
            _traverse_node(node)
            if node.is_named_node() and not node.is_local_ref_node():
                assert node.name not in root_fcache.named_node_name_set, f"Named node name conflict: {node.name} in root flow {root_fcache.named_node_name_set}."
                root_fcache.named_node_name_set.add(node.name)
            root_fcache.node_id_pool(node.id)
        self._reflesh_dependency(self._node_gid_to_cache, self._flow_node_gid_to_cache)


    def parse_all(self):
        # must be called after sync_project_model
        visited: set[str] = set()
        for flow_gid, fcache in self._flow_node_gid_to_cache.items():
            flow_parser = fcache.parser
            res = flow_parser._parse_flow_recursive(fcache.parent_node, flow_gid, fcache.flow, self, visited)
        self._update_path_to_fcache()

    def sync_all_files(self):
        self._update_path_to_fcache()
        folders, path_to_code = self._get_all_files_and_folders()
        deleted_folders, deleted_files, new_files, modify_files = self._fs_cache.compare_and_update(folders, path_to_code)
        return deleted_folders, deleted_files, new_files, modify_files

    def parse_by_flow_gids(self, flow_gids: list[str]):
        # must be called after sync_project_model
        visited: set[str] = set()
        for flow_gid in flow_gids:
            fcache = self._flow_node_gid_to_cache[flow_gid]
            flow_parser = fcache.parser
            res = flow_parser._parse_flow_recursive(fcache.parent_node, flow_gid, fcache.flow, self, visited)

    def _get_flow_code_lineno_by_node_gid(self, node_gid: str) -> Optional[FlowCodeFragment]:
        node_cache_origin = self._node_gid_to_cache[node_gid]
        node_cache = node_cache_origin
        is_ref_node = False
        
        if node_cache.node.ref is not None:
            ref_node_gid = node_cache.node.get_ref_global_uid()
            node_cache = self._node_gid_to_cache[ref_node_gid] 
            node_gid = ref_node_gid
            is_ref_node = True
        if node_cache.node.flow is not None and node_cache.node.nType != ADVNodeType.CLASS:
            # currently we don't support show code of subflow.
            return None 
        if node_cache.node.is_inline_flow_desc():
            # currently we don't support show code of inline flow.
            return None 

        flow_gid = self._get_flow_gid_from_node_gid(node_gid)
        fcache = self._flow_node_gid_to_cache[flow_gid]
        parser = fcache.parser
        flow_parse_res = parser.get_flow_parse_result_checked()
        if node_cache.node.nType == ADVNodeType.MARKDOWN:
            code_range = (-1, -1, -1, -1)
            assert node_cache.node.impl is not None
            flow_code = node_cache.node.impl.code
            node_gid = node_cache.node.get_global_uid()
            path = f"{node_gid}.md"
            return FlowCodeFragment(flow_code, str(path), code_range, is_ref_node) 

        parse_res = parser._be_nodes_map[self._node_gid_to_cache[node_gid].node.id].parse_res
        assert parse_res is not None 
        flow_code = "\n".join(flow_parse_res.to_code_lines().lines)
        loc = parse_res.get_global_loc()
        code_range = (loc.lineno_offset, loc.column, loc.end_lineno_offset, loc.end_column)
        return FlowCodeFragment(flow_code, str(flow_parse_res.get_code_relative_path()), code_range, is_ref_node)

    def _get_handles_from_parse_res(self, parse_res: BaseParseResult) -> Optional[list[ADVNodeHandle]]:
        if isinstance(parse_res, SymbolParseResult):
            return [bh.handle for bh in parse_res.symbols]
        elif isinstance(parse_res, FragmentParseResult):
            return [bh.handle for bh in parse_res.input_handles + parse_res.output_handles]
        elif isinstance(parse_res, OutIndicatorParseResult):
            return [parse_res.handle.handle]
        else:
            return None

    def init_all_nodes(self):
        # init all names and node handles in model
        for flow_gid, fcache in self._flow_node_gid_to_cache.items():
            parser = fcache.parser
            assert parser._flow_parse_result is not None
            for node_id, node in fcache.flow.nodes.items():
                be_node = parser._be_nodes_map.get(node_id, None)
                if be_node is not None and be_node.parse_res is not None:
                    fe_handles = self._get_handles_from_parse_res(be_node.parse_res)
                    if fe_handles is not None:
                        node.handles = fe_handles
                else:
                    # TODO 
                    ADV_LOGGER.warning(f"Node {node_id} in flow {flow_gid} has no parse result.")
            fcache.flow.edges = {e.id: e for e in parser._flow_parse_result.edges}

    def _get_flow_gid_from_node_gid(self, node_gid: str) -> str:
        return self._node_gid_to_cache[node_gid].flow_node_gid

    def _is_node_subflow(self, node_gid: str):
        node = self._node_gid_to_cache[node_gid].node
        if node.ref is None:
            return node.flow is not None
        node_def = self._node_gid_to_cache[node.get_ref_global_uid()].node
        return node_def.flow is not None

    def _reparse_changed_node(self, node_gid: str):
        # pair = ADVProject.get_flow_node_by_fe_path(root_flow, node.ref_fe_path)
        # assert pair is not None 
        flow_node_gid = self._node_gid_to_cache[node_gid].flow_node_gid
        return self._reparse_changed_node_internal(flow_node_gid, [node_gid])

    def _reparse_flow(self, flow_gid: str):
        return self._reparse_changed_node_internal(flow_gid, [])

    def _reparse_changed_node_internal(self, flow_gid: str, node_gids: list[str]) -> ADVProjectChange:
        return self._reparse_changed_nodes_internal([(flow_gid, node_gids)], {}) 

    def _is_edges_equal(self, edges1: list[ADVEdgeModel], edges2: list[ADVEdgeModel]) -> bool:
        id_to_edges1 = {e.id: e for e in edges1}
        id_to_edges2 = {e.id: e for e in edges2}
        if len(id_to_edges1) != len(id_to_edges2):
            return False
        for eid, edge1 in id_to_edges1.items():
            if eid not in id_to_edges2:
                return False
            edge2 = id_to_edges2[eid]
            if (edge1.source != edge2.source or 
                edge1.sourceHandle != edge2.sourceHandle or 
                edge1.target != edge2.target or 
                edge1.targetHandle != edge2.targetHandle or 
                edge1.isAutoEdge != edge2.isAutoEdge):
                return False
        return True 

    def _reparse_changed_nodes_internal(self, pairs: list[tuple[str, list[str]]], deleted_node_gid_map: dict[str, ADVNodeModel]) -> ADVProjectChange:
        # all modify operations should only change handles and edges
        gcache = ModifyParseCache(visited=set(), old_parse_res={})
        cur_layer: list[tuple[str, list[str]]] = pairs.copy()
        layered_parse_visited: set[str] = set()
        node_changes: dict[str, ADVNodeChange] = {}
        new_flow_edges: dict[str, list[ADVEdgeModel]] = {}
        flow_code_changes: dict[str, str] = {}
        changed_node_in_cur: set[str] = set()
        root_path = Path(self._root_flow_getter().path)
        while cur_layer:
            # sort cur layer
            cur_layer.sort(key=lambda x: self._flow_node_gid_to_cache[x[0]].topological_sorted_index)

            next_layer_nodes: set[str] = set()
            # clear all flow parse results in current layer
            # keep old parse 
            cur_flow_gids: list[str] = []
            for flow_gid, _ in cur_layer:
                # assert flow_gid not in layered_parse_visited
                cur_flow_gids.append(flow_gid)
                fcache = self._flow_node_gid_to_cache[flow_gid]
                flow_parser = fcache.parser
                if flow_parser._flow_parse_result is not None:
                    gcache.old_parse_res[flow_gid] = (flow_parser._flow_parse_result, flow_parser._be_nodes_map, fcache.flow.edges)
                flow_parser.clear_parse_result()

            for flow_gid, changed_node_gids in cur_layer:
                if flow_gid in layered_parse_visited:
                    continue
                layered_parse_visited.add(flow_gid)
                ADV_LOGGER.warning("reparse flow %s due to changed nodes %s", flow_gid if flow_gid else "__ROOT__", changed_node_gids)

                # TODO validate changed_node_gids
                fcache = self._flow_node_gid_to_cache[flow_gid]
                flow = fcache.flow 
                flow_parser = fcache.parser
                prev_flow_parse_res = None
                prev_be_node_map: dict[str, BackendNode] = {}
                prev_edges: Optional[dict[str, ADVEdgeModel]] = None
                if flow_gid in gcache.old_parse_res:
                    prev_flow_parse_res, prev_be_node_map, prev_edges = gcache.old_parse_res[flow_gid]
                flow_parse_res = flow_parser._parse_flow_recursive(fcache.parent_node, flow_gid, flow, self, gcache.visited) 
                flow_code_changed: bool = True
                if prev_flow_parse_res is not None:
                    flow_code_changed = prev_flow_parse_res.generated_code != flow_parse_res.generated_code
                if flow_code_changed:
                    flow_code_changes[str(root_path / flow_parse_res.get_code_relative_path())] = flow_parse_res.generated_code
                be_node_map = flow_parser._be_nodes_map

                symbol_group_gid_may_change: set[str] = set()
                frag_id_may_change: set[str] = set()
                remain_change_ids: set[str] = set()

                # 1. process global script nodes, extract symbol groups may change
                for node_gid in changed_node_gids:
                    if node_gid in deleted_node_gid_map:
                        # deleted gs/sym node always trigger symbol group change check
                        node = deleted_node_gid_map[node_gid]
                        if node.nType == ADVNodeType.GLOBAL_SCRIPT:
                            for node_id, node_parse_res in be_node_map.items():
                                cur_node = flow.nodes[node_id]
                                if cur_node.nType == ADVNodeType.SYMBOLS and cur_node.ref is None:
                                    symbol_group_gid_may_change.add(cur_node.get_global_uid())
                        elif node.nType == ADVNodeType.SYMBOLS:
                            symbol_group_gid_may_change.add(node_gid)
                        elif node.nType == ADVNodeType.FRAGMENT and node.is_inline_flow_desc_def():
                            # trigger all fragment node change.
                            for node_id, node_parse_res in be_node_map.items():
                                cur_node = flow.nodes[node_id]
                                if cur_node.nType == ADVNodeType.FRAGMENT and not cur_node.is_inline_flow_desc() and cur_node.id != node.id:
                                    frag_id_may_change.add(cur_node.id)
                        # we don't care about frag/other node deletion here because
                        # they only affect inline flow, which is already handled in flow parse.
                        continue
                    node_cache = self._node_gid_to_cache[node_gid]
                    node = node_cache.node
                    node_parse_res = be_node_map[node.id].get_parse_res_raw_checked()

                    if node.nType == ADVNodeType.GLOBAL_SCRIPT:
                        is_new_node = node.id not in prev_be_node_map
                        is_error_change = True 
                        if node.id in prev_be_node_map:
                            prev_node_parse_res = prev_be_node_map[node.id].get_parse_res_raw_checked()
                            is_error_change = prev_node_parse_res.succeed != node_parse_res.succeed
                        change = ADVNodeChange.empty(node.name, node_parse_res, is_new=is_new_node, error_status_changed=is_error_change)
                        if node.ref is None:
                            if node.id in prev_be_node_map:
                                prev_node_parse_res = prev_be_node_map[node.id].get_parse_res_raw_checked()
                                assert isinstance(node_parse_res, GlobalScriptParseResult)
                                assert isinstance(prev_node_parse_res, GlobalScriptParseResult)
                                is_name_changed = node.name != prev_node_parse_res.get_node_checked().name
                                is_content_changed = node_parse_res.code != prev_node_parse_res.code
                            else:
                                is_name_changed = True
                                is_content_changed = True
                        else:
                            is_name_changed = False 
                            if node_gid in node_changes:
                                ext_change = node_changes[node_gid]
                                is_content_changed = bool(ext_change.flags & NodeChangeFlag.GLOBAL_SCRIPT)
                            else:
                                # new gs node (def/ref) always trigger content changed
                                is_content_changed = True
                        if is_content_changed:
                            for node_id, node_parse_res in be_node_map.items():
                                cur_node = flow.nodes[node_id]
                                if cur_node.nType == ADVNodeType.SYMBOLS and cur_node.ref is None:
                                    symbol_group_gid_may_change.add(cur_node.get_global_uid())
                        if node.ref is None:
                            if is_content_changed:
                                # update all ref nodes for global script node
                                next_layer_nodes.update(node_cache.ref_gids_external)
                            if is_name_changed:
                                change.flags |= NodeChangeFlag.NAME
                            # TODO currently we don't use impl in frontend, so we don't need to update it.
                            if is_content_changed:
                                change.flags |= NodeChangeFlag.GLOBAL_SCRIPT
                            if change.flags != NodeChangeFlag(0):
                                node_changes[node_gid] = change

                    elif node.nType == ADVNodeType.SYMBOLS:
                        symbol_group_gid_may_change.add(node_gid)
                    elif node.nType == ADVNodeType.FRAGMENT or node.nType == ADVNodeType.CLASS:
                        frag_id_may_change.add(node.id)
                    else:
                        remain_change_ids.add(node.id)
                # 2. process all symbol group nodes
                for sym_node_gid in symbol_group_gid_may_change:
                    if sym_node_gid in deleted_node_gid_map:
                        # deleted sym node always trigger all frag node change check
                        for node_id in be_node_map.keys():
                            cur_node = flow.nodes[node_id]
                            if cur_node.nType == ADVNodeType.FRAGMENT and cur_node.ref is None and cur_node.flow is None:
                                frag_id_may_change.add(node_id)
                        continue
                    else:
                        node_cache = self._node_gid_to_cache[sym_node_gid]
                        sym_node = node_cache.node
                        sym_node_id = sym_node.id
                    sym_node = flow.nodes[sym_node_id]
                    sym_node_gid = sym_node.get_global_uid()
                    node_cache = self._node_gid_to_cache[sym_node_gid]
                    node_parse_res = be_node_map[sym_node_id].get_parse_res_raw_checked()
                    is_new_node = sym_node.id not in prev_be_node_map
                    is_error_change = True 
                    if sym_node.id in prev_be_node_map:
                        prev_node_parse_res = prev_be_node_map[sym_node.id].get_parse_res_raw_checked()
                        is_error_change = prev_node_parse_res.succeed != node_parse_res.succeed
                    change = ADVNodeChange.empty(name=sym_node.name, is_new=is_new_node, parse_res=node_parse_res, error_status_changed=is_error_change)
                    if sym_node.id in prev_be_node_map:
                        prev_node_parse_res = prev_be_node_map[sym_node.id].get_parse_res_raw_checked()
                        assert isinstance(node_parse_res, SymbolParseResult)
                        assert isinstance(prev_node_parse_res, SymbolParseResult)
                        is_sym_changed = node_parse_res.is_io_handle_changed(prev_node_parse_res)
                        is_sym_name_changed = node_parse_res.symbol_cls_name != prev_node_parse_res.symbol_cls_name
                    else:
                        is_sym_changed = True
                        is_sym_name_changed: bool = False
                    if is_sym_changed:
                        change.flags |= NodeChangeFlag.HANDLES
                        for node_id in be_node_map.keys():
                            cur_node = flow.nodes[node_id]
                            if cur_node.nType == ADVNodeType.FRAGMENT and cur_node.ref is None and cur_node.flow is None:
                                frag_id_may_change.add(node_id)
                    if sym_node.ref is None:
                        if is_sym_changed or is_sym_name_changed:
                            next_layer_nodes.update(node_cache.ref_gids_external)
                        if is_sym_name_changed:
                            change.flags |= NodeChangeFlag.NAME
                    if change.flags != NodeChangeFlag(0):
                        node_changes[sym_node_gid] = change

                # 3. prepare frag nodes
                frag_id_may_change_next: set[str] = set(frag_id_may_change)
                all_changed_inline_flows: set[str] = set()
                # add inline node descs
                for frag_node_id in frag_id_may_change:
                    frag_node = flow.nodes[frag_node_id]
                    if frag_node.inlinesf_name is not None:
                        all_changed_inline_flows.add(frag_node.inlinesf_name)
                if all_changed_inline_flows:
                    for node in flow.nodes.values():
                        if node.is_inline_flow_desc_def():
                            if node.name in all_changed_inline_flows:
                                frag_id_may_change_next.add(node.id)
                for frag_node_id in frag_id_may_change:
                    frag_node = flow.nodes[frag_node_id]
                    frag_node_cache = self._node_gid_to_cache[frag_node.get_global_uid()]
                    # always check local ref frag nodes to simplify logic
                    frag_id_may_change_next.update(frag_node_cache.ref_node_ids_local)
                # 4. process fragment/class nodes
                for node_id in frag_id_may_change_next:
                    node_parse_res = be_node_map[node_id].get_parse_res_raw_checked()
                    cur_node = flow.nodes[node_id]
                    node_gid = cur_node.get_global_uid()
                    node_cache = self._node_gid_to_cache[node_gid]
                    is_new_node = node_id not in prev_be_node_map
                    is_error_change = True 
                    if node_id in prev_be_node_map:
                        prev_node_parse_res = prev_be_node_map[node_id].get_parse_res_raw_checked()
                        is_error_change = prev_node_parse_res.succeed != node_parse_res.succeed
                    change = ADVNodeChange.empty(name=cur_node.name, is_new=is_new_node, parse_res=node_parse_res, error_status_changed=is_error_change)
                    is_ref_node = cur_node.ref is not None
                    if node_id in prev_be_node_map:
                        prev_node_parse_res = prev_be_node_map[node_id].get_parse_res_raw_checked()
                        assert isinstance(node_parse_res, FragmentParseResult)
                        assert isinstance(prev_node_parse_res, FragmentParseResult)
                        prev_node = prev_node_parse_res.get_node_checked()
                        is_name_changed = node_parse_res.func_name != prev_node_parse_res.func_name
                        is_node_io_changed = node_parse_res.is_io_handle_changed(prev_node_parse_res)
                        is_alias_map_changed = cur_node.alias_map != prev_node.alias_map
                        is_inline_flow_changed = cur_node.inlinesf_name != prev_node.inlinesf_name
                        is_flag_changed = cur_node.flags != prev_node.flags
                        if cur_node.ref is not None and prev_node.ref is not None:
                            is_ref_change = not cur_node.ref.is_equal_to(prev_node.ref)
                        elif prev_node.ref is None and cur_node.ref is None:
                            is_ref_change = False
                        else:
                            is_ref_change = True
                    else:
                        is_name_changed = not is_ref_node
                        is_node_io_changed = True
                        is_alias_map_changed = is_ref_node
                        is_inline_flow_changed = cur_node.inlinesf_name is not None
                        is_ref_change = True
                        is_flag_changed = True
                    # local changes without ref
                    if is_name_changed:
                        change.flags |= NodeChangeFlag.NAME
                    if is_node_io_changed:
                        change.flags |= NodeChangeFlag.HANDLES
                    if is_alias_map_changed:
                        change.flags |= NodeChangeFlag.ALIAS_MAP
                    if is_inline_flow_changed:
                        change.flags |= NodeChangeFlag.INLINE_FLOW
                    if is_ref_change:
                        change.flags |= NodeChangeFlag.REF
                    if is_flag_changed:
                        change.flags |= NodeChangeFlag.FLAGS
                    # 1. IO change (except nested flow), 2. name change 
                    # will trigger external ref update.
                    # nested flow IO change is triggered when we handle that flow.
                    if not is_ref_node and cur_node.flow is None and is_node_io_changed:
                        next_layer_nodes.update(node_cache.ref_gids_external)
                    if not is_ref_node and (is_name_changed or is_flag_changed):
                        next_layer_nodes.update(node_cache.ref_gids_external)
                    if change.flags != NodeChangeFlag(0):
                        node_changes[node_gid] = change
                # 5. process remain nodes
                for node_id in remain_change_ids:
                    node_parse_res = be_node_map[node_id].get_parse_res_raw_checked()
                    node = flow.nodes[node_id]
                    node_gid = node.get_global_uid()
                    is_new_node = node_id not in prev_be_node_map
                    change = ADVNodeChange.empty(name=node.name, is_new=is_new_node, parse_res=node_parse_res)
                    if node.nType == ADVNodeType.OUT_INDICATOR:
                        is_oic_alias_changed: bool = True
                        is_oic_handle_changed: bool = True
                        if node_id in prev_be_node_map:
                            prev_node_parse_res = prev_be_node_map[node.id].get_parse_res_raw_checked()
                            assert isinstance(node_parse_res, OutIndicatorParseResult)
                            assert isinstance(prev_node_parse_res, OutIndicatorParseResult)
                            is_oic_alias_changed = node.name != prev_node_parse_res.get_node_checked().name
                            is_oic_handle_changed = node_parse_res.is_io_handle_changed(prev_node_parse_res)
                        # out indicator can't be ref, only used in inline flow.
                        if is_oic_alias_changed:
                            change.flags |= NodeChangeFlag.NAME
                        if is_oic_handle_changed:
                            change.flags |= NodeChangeFlag.HANDLES
                    if node.nType == ADVNodeType.MARKDOWN:
                        change.flags |= NodeChangeFlag.IMPL_CODE
                    if change.flags != NodeChangeFlag(0):
                        node_changes[node_gid] = change
                # here we must compare flow_parse_res.edges with flow.edges instead of prev_flow_parse_res.edges
                # because frontend already change flow.edges.
                if prev_edges is None or not self._is_edges_equal(list(prev_edges.values()), flow_parse_res.edges):
                    new_flow_edges[flow_gid] = flow_parse_res.edges

                # if flow main inline flow io is changed by any reparse, reparse all ref fragment+inline nodes
                # TODO currently we don't support change base class node
                is_inlineflow_changed = True 
                if prev_flow_parse_res is not None:
                    assert isinstance(prev_flow_parse_res, FlowParseResult)
                    is_inlineflow_changed = prev_flow_parse_res.is_io_handle_changed(flow_parse_res)
                if is_inlineflow_changed:
                    # root flow can't be used as a fragment node.
                    if fcache.parent_node is not None:
                        parent_cache = self._node_gid_to_cache[fcache.parent_node.get_global_uid()]
                        next_layer_nodes.update(parent_cache.ref_gids_external)
                if flow_gid == pairs[0][0]:
                    changed_node_in_cur = set([c.parse_res.get_node_checked().id for c in node_changes.values()])

            next_layer: dict[str, list[str]] = {}
            for node_gid in next_layer_nodes:
                node_cache = self._node_gid_to_cache[node_gid]
                next_flow_node_gid = node_cache.flow_node_gid
                if next_flow_node_gid in layered_parse_visited:
                    continue 
                if next_flow_node_gid not in next_layer:
                    next_layer[next_flow_node_gid] = []
                next_layer[next_flow_node_gid].append(node_gid)
            cur_layer = list(next_layer.items())

        return ADVProjectChange(
            node_changes=node_changes,
            new_flow_edges=new_flow_edges,
            flow_code_changes=flow_code_changes,
            changed_node_ids_in_cur_flow=list(changed_node_in_cur),
        )

    def modify_code_impl(self, node_gid: str, new_code: str) -> ADVProjectChange:
        node = self._node_gid_to_cache[node_gid].node
        assert node.impl is not None 
        if node.nType == ADVNodeType.MARKDOWN:
            prev_code = node.impl.code
            if prev_code == new_code:
                return ADVProjectChange()
            node.impl.code = new_code
            return self._reparse_changed_node(node_gid)
        if node.nType == ADVNodeType.FRAGMENT:
            FragmentParseResult.early_validate_code(new_code)
        elif node.nType == ADVNodeType.SYMBOLS:
            SymbolParseResult.early_validate_code(new_code)
        elif node.nType == ADVNodeType.GLOBAL_SCRIPT:
            ast.parse(new_code)
        prev_code = node.impl.code
        prev_code_clean = clean_source_code(prev_code.splitlines())
        new_code_clean = clean_source_code(new_code.splitlines())
        node.impl.code = new_code
        if prev_code_clean == new_code_clean:
            # TODO only need to modify code in frontend/backend (file store).
            return ADVProjectChange()
        return self._reparse_changed_node(node_gid)


    def modify_node_config(self, node_gid: str, cfg: ADVNodeCMConfig):
        cache = self._node_gid_to_cache[node_gid]
        node = cache.node
        old_name = node.name
        old_inline_flow_name = node.inlinesf_name
        old_alias_map = node.alias_map
        old_flags = node.flags 
        old_func_type = "Global"
        if node.flags & int(ADVNodeFlags.IS_CLASSMETHOD):
            old_func_type = "Class Method"
        elif node.flags & int(ADVNodeFlags.IS_STATICMETHOD):
            old_func_type = "Static Method"
        elif node.flags & int(ADVNodeFlags.IS_METHOD):
            old_func_type = "Method"
        new_flags = int(old_flags)
        # reset is_method/is_classmethod/is_staticmethod bits
        new_flags &= ~int(ADVNodeFlags.IS_METHOD)
        new_flags &= ~int(ADVNodeFlags.IS_CLASSMETHOD)
        new_flags &= ~int(ADVNodeFlags.IS_STATICMETHOD)
        if cfg.func_base_type == "Method":
            new_flags |= int(ADVNodeFlags.IS_METHOD)
        elif cfg.func_base_type == "Class Method":
            new_flags |= int(ADVNodeFlags.IS_CLASSMETHOD)
        elif cfg.func_base_type == "Static Method":
            new_flags |= int(ADVNodeFlags.IS_STATICMETHOD)
        if old_name == cfg.name and old_inline_flow_name == cfg.inline_flow_name and old_alias_map == cfg.alias_map:
            return ADVProjectChange()
        if old_name != cfg.name:
            assert not node.is_local_ref_node(), "you can't change name of local ref node."
        if old_alias_map != cfg.alias_map:
            assert node.ref is not None or node.flow is not None, "Only ref node or subflow node can have alias map."

        node.name = cfg.name
        if cfg.inline_flow_name != "":
            node.inlinesf_name = cfg.inline_flow_name
        else:
            node.inlinesf_name = None
        node.alias_map = cfg.alias_map
        node.flags = new_flags
        flow_gid = cache.flow_node_gid
        if old_func_type != cfg.func_base_type and old_func_type != "Global":
            # check function must defined in class node
            fcache = self._flow_node_gid_to_cache[flow_gid]
            assert fcache.parent_node is not None and fcache.parent_node.nType == ADVNodeType.CLASS, "Only function node in class can change function type."

        changed_node_gids: list[str] = [node_gid]
        if node.is_inline_flow_desc_def():
            # change all node with same inline flow name
            fcache = self._flow_node_gid_to_cache[flow_gid]
            for node_cur in fcache.flow.nodes.values():
                if node_cur.inlinesf_name == old_name:
                    node_cur.inlinesf_name = cfg.name
                    changed_node_gids.append(node_cur.get_global_uid())
        if old_name != cfg.name:
            fcache = self._flow_node_gid_to_cache[flow_gid]
            assert not node.is_local_ref_node(), "you can't change name of local ref node."
            # we use node id instead of name, 
            # name is only used in code generation and user code.
            # flow itself (include scheduler) don't use name.
            if old_name in fcache.named_node_name_set:
                fcache.named_node_name_set.remove(old_name)
            fcache.named_node_name_set.add(cfg.name)
        return self._reparse_changed_node_internal(flow_gid, changed_node_gids)

    def add_new_node(self, flow_node_gid: str, new_node: ADVNodeModel):
        fcache = self._flow_node_gid_to_cache[flow_node_gid]
        if new_node.is_named_node() and not new_node.is_local_ref_node():
            assert new_node.name not in fcache.named_node_name_set, f"Node name {new_node.name} already exists in flow."
            if fcache.parent_node is not None:
                assert new_node.name != fcache.parent_node.name, "Symbol group/fragment name must be different from parent flow name."
            # fcache.named_node_name_set.add(new_node.name)
        new_node.id = fcache.node_id_pool(new_node.id)
        adv_proj = self._root_flow_getter()
        fcache.flow.nodes[new_node.id] = new_node

        # create paths
        if (new_node.flow is None or not new_node.flow.nodes) and new_node.ref is None:
            if fcache.parent_node is not None:
                new_path, new_frontend_path = fcache.parent_node.get_child_node_paths(new_node)
            else:
                # root flow
                new_path: list[str] = []
                new_frontend_path = ["nodes", new_node.id]
            new_node.path = new_path
            new_node.frontend_path = new_frontend_path
        else:
            # flow contains nested node. we just reflesh all path dict and 
            # send to frontend.
            ngid_to_path, ngid_to_fpath = adv_proj.assign_path_to_all_node()
            adv_proj.update_ref_path(ngid_to_fpath)
            adv_proj.node_gid_to_path = ngid_to_path
            adv_proj.node_gid_to_frontend_path = ngid_to_fpath

        # update cache in mgr
        if new_node.flow is not None:
            self._flow_node_gid_to_cache[new_node.get_global_uid()] = FlowCache(
                new_node.flow, 
                FlowParser(),
                new_node
            )
        self._node_gid_to_cache[new_node.get_global_uid()] = NodeConnCache(
            new_node, flow_node_gid)
        try:
            self._reflesh_dependency(self._node_gid_to_cache, self._flow_node_gid_to_cache)
        except RuntimeError as e:
            # rollback
            fcache.flow.nodes.pop(new_node.id)
            if new_node.flow is not None:
                self._flow_node_gid_to_cache.pop(new_node.get_global_uid())
            self._node_gid_to_cache.pop(new_node.get_global_uid())
            ngid_to_path, ngid_to_fpath = adv_proj.assign_path_to_all_node()
            adv_proj.update_ref_path(ngid_to_fpath)
            adv_proj.node_gid_to_path = ngid_to_path
            adv_proj.node_gid_to_frontend_path = ngid_to_fpath
            raise e
        if new_node.is_named_node():
            fcache.named_node_name_set.add(new_node.name)
        # add meta update to res
        res = self._reparse_changed_node(new_node.get_global_uid())
        if new_node.flow is None or not new_node.flow.nodes:
            res.node_gid_to_path_update = {
                new_node.get_global_uid(): new_node.path
            }
            res.node_gid_to_frontend_path_update = {
                new_node.get_global_uid(): new_node.frontend_path
            }
        else:
            res.new_node_gid_to_path = adv_proj.node_gid_to_path
            res.new_node_gid_to_frontend_path = adv_proj.node_gid_to_frontend_path
        return res

    def _get_all_subflow_gids(self, node: ADVNodeModel) -> set[str]:
        if node.flow is None:
            return set()
        node_gid = node.get_global_uid()
        all_subflow_gids: set[str] = set([node_gid])
        def _traverse(flow: ADVFlowModel):
            for node in flow.nodes.values():
                if node.flow is not None:
                    gid = node.get_global_uid()
                    all_subflow_gids.add(gid)
                    _traverse(node.flow)
        _traverse(node.flow)
        return all_subflow_gids

    def _get_all_ext_refs_of_node(self, node: ADVNodeModel) -> dict[str, tuple[ADVNodeModel, str]]:
        node_gid = node.get_global_uid()
        node_cache = self._node_gid_to_cache[node_gid]
        all_deleted_subflow_gids = self._get_all_subflow_gids(node)

        all_gid_to_node_pairs: dict[str, tuple[ADVNodeModel, str]] = {}
        all_gid_to_node_pairs[node_gid] = (node_cache.node, node_cache.flow_node_gid)

        def _traverse(flow: ADVFlowModel):
            for node in flow.nodes.values():
                node_cache = self._node_gid_to_cache[node.get_global_uid()]
                for ref_gid in node_cache.ref_gids_external:
                    ref_node_cache = self._node_gid_to_cache[ref_gid]
                    if ref_node_cache.flow_node_gid not in all_deleted_subflow_gids:
                        all_gid_to_node_pairs[ref_gid] = (ref_node_cache.node, ref_node_cache.flow_node_gid)
                if node.flow is not None:
                    _traverse(node.flow)
        if node.flow is not None:
            _traverse(node.flow)
        for ref_gid in node_cache.ref_gids_external + node_cache.ref_gids_local:
            ref_node_cache = self._node_gid_to_cache[ref_gid]
            all_gid_to_node_pairs[ref_gid] = (ref_node_cache.node, ref_node_cache.flow_node_gid)
        return all_gid_to_node_pairs

    def delete_node(self, node_gid: str):
        node_cache = self._node_gid_to_cache[node_gid]
        all_gid_to_node_pairs = self._get_all_ext_refs_of_node(node_cache.node)
        flow_gid_to_deleted_node_gids: dict[str, list[str]] = {}
        for gid, (node, _) in all_gid_to_node_pairs.items():
            node_cache = self._node_gid_to_cache[gid]
            flow_gid = node_cache.flow_node_gid
            if flow_gid not in flow_gid_to_deleted_node_gids:
                flow_gid_to_deleted_node_gids[flow_gid] = []
            flow_gid_to_deleted_node_gids[flow_gid].append(gid)
        # remove nodes from cache
        for gid in all_gid_to_node_pairs.keys():
            self._node_gid_to_cache.pop(gid)
        if node_gid in self._flow_node_gid_to_cache:
            self._flow_node_gid_to_cache.pop(node_gid)
        # remove node from model
        for gid, (node, flow_gid) in all_gid_to_node_pairs.items():
            fcache = self._flow_node_gid_to_cache[flow_gid]
            fcache.flow.nodes.pop(node.id)
            if node.name in fcache.named_node_name_set:
                fcache.named_node_name_set.remove(node.name)
        self._reflesh_dependency(self._node_gid_to_cache, self._flow_node_gid_to_cache)
        res = self._reparse_changed_nodes_internal(list(flow_gid_to_deleted_node_gids.items()), {k: v[0] for k, v in all_gid_to_node_pairs.items()})
        res.deleted_node_pairs = list((v[0].id, v[1]) for k, v in all_gid_to_node_pairs.items())
        adv_proj = self._root_flow_getter()
        ngid_to_path, ngid_to_fpath = adv_proj.assign_path_to_all_node()
        adv_proj.node_gid_to_path = ngid_to_path
        adv_proj.node_gid_to_frontend_path = ngid_to_fpath
        res.new_node_gid_to_path = adv_proj.node_gid_to_path
        res.new_node_gid_to_frontend_path = adv_proj.node_gid_to_frontend_path
        return res

    def create_draft_updates_from_change(self, draft: ADVProject, change: ADVProjectChange):
        if change.is_empty():
            return 
        for node_gid, node_change in change.node_changes.items():
            node_cache = self._node_gid_to_cache[node_gid]
            node = node_cache.node
            node_id_path = ADVProject.get_node_id_path_from_fe_path(node.frontend_path)
            if node_change.flags & NodeChangeFlag.NEW:
                handles = self._get_handles_from_parse_res(node_change.parse_res)
                if handles is not None:
                    node.handles = handles
                else:
                    assert node.nType == ADVNodeType.MARKDOWN
                if len(node_id_path) == 1:
                    # root flow
                    draft.flow.nodes[node.id] = node 
                else:
                    node_draft = draft.draft_get_node_by_id_path(node_id_path[:-1])
                    node_draft.flow.nodes[node.id] = node
            else:
                node_draft = draft.draft_get_node_by_id_path(node_id_path)
                if node_change.flags & NodeChangeFlag.HANDLES:
                    assert isinstance(node_change.parse_res, (SymbolParseResult, FragmentParseResult, OutIndicatorParseResult))
                    handles = self._get_handles_from_parse_res(node_change.parse_res)
                    assert handles is not None 
                    node_draft.handles = handles
                if node_change.flags & NodeChangeFlag.NAME:
                    node_draft.name = node.name
                if node_change.flags & NodeChangeFlag.ALIAS_MAP:
                    node_draft.alias_map = node.alias_map
                if node_change.flags & NodeChangeFlag.INLINE_FLOW:
                    node_draft.inlinesf_name = node.inlinesf_name
                if node_change.flags & NodeChangeFlag.IMPL_CODE:
                    assert node.nType == ADVNodeType.MARKDOWN
                    assert node.impl is not None 
                    node_draft.impl.code = node.impl.code
                if node_change.flags & NodeChangeFlag.REF:
                    node_draft.ref = node.ref
                if node_change.flags & NodeChangeFlag.FLAGS:
                    node_draft.flags = node.flags

        for flow_node_gid, edges in change.new_flow_edges.items():
            edges_dict =  {e.id: e for e in edges}
            assert len(edges_dict) == len(edges)
            if flow_node_gid == TENSORPC_ADV_ROOT_FLOW_ID:
                draft.flow.edges = edges_dict
            else:
                node_cache = self._node_gid_to_cache[flow_node_gid]
                node = node_cache.node
                node_draft = draft.draft_get_node_by_fe_path(node.frontend_path)
                node_draft.flow.edges = {e.id: e for e in edges}
        if change.node_gid_to_path_update is not None:
            draft.node_gid_to_path.update(change.node_gid_to_path_update)
        if change.node_gid_to_frontend_path_update is not None:
            draft.node_gid_to_frontend_path.update(change.node_gid_to_frontend_path_update)
        if change.new_node_gid_to_path is not None:
            draft.node_gid_to_path = change.new_node_gid_to_path
        if change.new_node_gid_to_frontend_path is not None:
            draft.node_gid_to_frontend_path = change.new_node_gid_to_frontend_path
        if change.deleted_node_pairs is not None:
            for node_id, flow_node_gid in change.deleted_node_pairs:
                if flow_node_gid == TENSORPC_ADV_ROOT_FLOW_ID:
                    flow_draft = draft.flow 
                else:
                    flow_node_cache = self._node_gid_to_cache[flow_node_gid]
                    node_id_path = ADVProject.get_node_id_path_from_fe_path(flow_node_cache.node.frontend_path)
                    flow_node_draft = draft.draft_get_node_by_id_path(node_id_path)
                    flow_draft = flow_node_draft.flow
                flow_draft.nodes.pop(node_id)
        self.sync_all_files()

    def collect_possible_ref_nodes(self, flow_node_gid: str) -> list[ADVNodeModel]:
        fcache_cur_flow = self._flow_node_gid_to_cache[flow_node_gid]
        res: list[ADVNodeModel] = []
        for flow_gid, fcache in self._flow_node_gid_to_cache.items():
            # if this flow depend on current flow, skip
            if flow_gid in fcache_cur_flow.all_child_flow_gids:
                continue
            for node in fcache.flow.nodes.values():
                if node.ref is None and node.nType in (ADVNodeType.GLOBAL_SCRIPT, ADVNodeType.SYMBOLS, ADVNodeType.FRAGMENT, ADVNodeType.CLASS):
                    if flow_node_gid == flow_gid and node.nType in (ADVNodeType.GLOBAL_SCRIPT, ADVNodeType.SYMBOLS):
                        # local ref of global script and symbol group is not allowed
                        continue
                    if flow_node_gid != flow_gid and node.is_init_fn():
                        # init fn only allowed to be local ref
                        # we use class node instead of init fn in external flows.
                        continue
                    if node.flow is not None:
                        fcache_cur_node = self._flow_node_gid_to_cache[node.get_global_uid()]
                        parse_res = fcache_cur_node.parser.get_flow_parse_result_checked()
                        if not parse_res.can_node_be_ref:
                            continue 
                    res.append(node)
        return res 

    def collect_all_inline_flow_names(self, flow_node_gid: str) -> list[str]:
        fcache_cur_flow = self._flow_node_gid_to_cache[flow_node_gid]
        flow_node = fcache_cur_flow.parent_node
        res: list[str] = []
        if flow_node is not None:
            if flow_node.nType == ADVNodeType.FRAGMENT:
                # fragment node has a main inline flow that have same name as fragment node
                res.append(flow_node.name)
        # if fcache_cur_flow.parent_node
        for node_id, node in fcache_cur_flow.flow.nodes.items():
            if node.is_inline_flow_desc_def():
                res.append(node.name)
        return list(set(res)) 

    def flow_nodes_position_change(self, flow_node_gid: str):
        # change position may change auto-generated edge, and fragment siguature, 
        # so we reparse all symbol group and fragment nodes.
        # WARNING: position is already changed outside.
        fcache = self._flow_node_gid_to_cache[flow_node_gid]
        flow = fcache.flow
        changed_node_gids: list[str] = []
        for node in flow.nodes.values():
            if node.nType in [ADVNodeType.SYMBOLS, ADVNodeType.FRAGMENT]:
                changed_node_gids.append(node.get_global_uid())
        return self._reparse_changed_node_internal(flow_node_gid, changed_node_gids)

    def flow_edge_change(self, flow_node_gid: str):
        # WARNING: edges is already changed outside.
        fcache = self._flow_node_gid_to_cache[flow_node_gid]
        flow = fcache.flow
        changed_node_gids: list[str] = []
        for node in flow.nodes.values():
            # edge change may change out indicator node handles
            if node.nType in [ADVNodeType.OUT_INDICATOR]:
                changed_node_gids.append(node.get_global_uid())
            # edge change may change inline flow desc nodes
            if node.is_inline_flow_desc_def():
                changed_node_gids.append(node.get_global_uid())

        return self._reparse_changed_node_internal(flow_node_gid, changed_node_gids)

    def fragment_node_ref_change(self, node_gid: str, new_ref: Optional[ADVNodeRefInfo] = None):
        # only allow ref node to change ref
        # WARNING: edge change is already changed outside.
        cache = self._node_gid_to_cache[node_gid]
        assert cache.node.nType == ADVNodeType.FRAGMENT, "only fragment node can change ref."
        assert cache.node.ref is not None, "only ref node can change ref."
        flow_node_gid = cache.flow_node_gid
        if new_ref is None:
            node_def_gid = cache.node.get_ref_global_uid()
            node_def = self._node_gid_to_cache[node_def_gid]
            cache.node.impl = node_def.node.impl
        # TODO we need to trigger editor change when we convert ref node to def node
        self._reflesh_dependency(self._node_gid_to_cache, self._flow_node_gid_to_cache)
        return self._reparse_changed_node_internal(flow_node_gid, [node_gid])

    def modify_flow_code_external(self, new_flow_codes: dict[str, str]):
        # only allow modify impl of fragment/symbolgroup/global script.
        # add/delete node isn't allowed via external editor.
        old_fcaches = {gid: self._flow_node_gid_to_cache[gid] for gid in new_flow_codes}

        new_path_to_code = {old_fcaches[gid].get_code_relative_path_checked(): code for gid, code in new_flow_codes.items()}
        fspath_to_old_fcaches = {v.get_code_relative_path_checked(): v for v in old_fcaches.values()}

        all_flow_nodes_no_change = {v.get_code_relative_path_checked(): (v.parent_node, v.flow) for v in self._flow_node_gid_to_cache.values()}
        for new_path in new_path_to_code:
            all_flow_nodes_no_change.pop(new_path)
        accessor: Callable[[list[str], Path], str] = lambda path, fspath: new_path_to_code[fspath]
        proj_parser = ADVProjectParser(path_code_accessor=accessor)
        parsed_flows: dict[Path, ADVFlowModel] = {}

        for flow_gid in new_flow_codes:
            fcache = self._flow_node_gid_to_cache[flow_gid]
            path = fcache.get_code_relative_path_checked()
            if path in parsed_flows:
                continue 
            if fcache.parent_node is None:
                path_with_name: list[str] = []
            else:
                path_with_name = fcache.parent_node.path + [fcache.parent_node.name]

            proj_parser._parse_desc_to_flow_model(path_with_name, path, set(), parsed_flows, all_flow_nodes_no_change)
        changed_flow_and_node_ids: list[tuple[str, list[str]]] = []
        # check first
        for fspath, parsed_flow in parsed_flows.items():
            fcache = fspath_to_old_fcaches[fspath]
            old_flow = fcache.flow
            new_node_ids = set(parsed_flow.nodes.keys()) - set(old_flow.nodes.keys())
            deleted_node_ids = set(old_flow.nodes.keys()) - set(parsed_flow.nodes.keys())
            assert not new_node_ids, f"New nodes {new_node_ids} are not allowed when external editing."
            assert not deleted_node_ids, f"Deleted nodes {deleted_node_ids} are not allowed when external editing."
            # check each node
            changed_node_gids: list[str] = []
            for node_id, new_node in parsed_flow.nodes.items():
                old_node = old_flow.nodes[node_id]
                assert new_node.nType == old_node.nType, f"Node type change is not allowed when external editing. Node id: {node_id}"
                is_ref_change = new_node.ref != old_node.ref
                assert not is_ref_change, f"Node ref change is not allowed when external editing. Node id: {node_id}"
                if new_node.nType not in (ADVNodeType.GLOBAL_SCRIPT, ADVNodeType.SYMBOLS, ADVNodeType.FRAGMENT):
                    # other node type can't change anything
                    assert new_node.is_base_props_equal_to(old_node), f"Node base change is not allowed for node except frag/sym/globalscript."

        for fspath, parsed_flow in parsed_flows.items():
            fcache = fspath_to_old_fcaches[fspath]
            flow_gid = fcache.get_flow_gid()
            old_flow = fcache.flow
            # check each node
            changed_node_gids: list[str] = []
            for node_id, new_node in parsed_flow.nodes.items():
                old_node = old_flow.nodes[node_id]
                if new_node.nType in (ADVNodeType.GLOBAL_SCRIPT, ADVNodeType.SYMBOLS, ADVNodeType.FRAGMENT):
                    is_base_equal = new_node.is_base_props_equal_to(old_node)
                    is_impl_equal = new_node.is_impl_equal_to(old_node)
                    if not is_base_equal or not is_impl_equal:
                        changed_node_gids.append(old_node.get_global_uid())
            # update flow model
            if changed_node_gids:
                if fcache.parent_node is not None:
                    fcache.parent_node.flow = parsed_flow
                else:
                    # root flow
                    adv_proj = self._root_flow_getter()
                    adv_proj.flow = parsed_flow 
                fcache.flow = parsed_flow
                changed_flow_and_node_ids.append((flow_gid, changed_node_gids))
        change = self._reparse_changed_nodes_internal(changed_flow_and_node_ids, {})
        return change


class GlobalScriptParser:
    def __init__(self) -> None:
        self._code_to_global_scope: dict[str, dict[str, Any]] = {}

    def parse_global_script(self, code: str) -> dict[str, Any]:
        if code in self._code_to_global_scope:
            return self._code_to_global_scope[code]
        global_scope: dict[str, Any] = {}
        local_scope: dict[str, Any] = {}
        compiled = compile(code, "<string>", "exec")
        exec(compiled, global_scope, local_scope)
        self._code_to_global_scope[code] = local_scope
        return local_scope

class FlowParser:
    def __init__(self):
        self._global_scope_parser = GlobalScriptParser()
        self._be_nodes_map: dict[str, BackendNode] = {}
        self._node_id_to_parser: dict[str, BaseParser] = {}
        self._flow_parse_result: Optional[FlowParseResult] = None
        self._edge_uid_pool = UniqueNamePool()

    def clear_parse_cache(self):
        self._be_nodes_map = {}
        self._node_id_to_parser = {}
        self._flow_parse_result = None
        self._edge_uid_pool.clear()

    def clear_parse_result(self):
        self._be_nodes_map = {}
        self._flow_parse_result = None
        self._edge_uid_pool.clear()

    def get_flow_parse_result_checked(self) -> FlowParseResult:
        assert self._flow_parse_result is not None
        return self._flow_parse_result

    def serialize_to_code(self):
        assert self._flow_parse_result is not None
        return self._flow_parse_result.to_code_lines()

    def _node_to_be_node(self, node: ADVNodeModel, cur_flow_node: Optional[ADVNodeModel], flow_gid: str, 
            cur_flow_is_folder: bool, mgr: ADVProjectBackendManager, visited: set[str], is_inherit_ref: bool = False):
        node_type = ADVNodeType(node.nType)

        is_subflow_def = False
        is_local_ref = False
        parent_is_folder = False
        is_node_def_folder = False
        if is_inherit_ref:
            ref = node.cls_inherit_ref
            assert ref is not None 
        else:
            ref = node.ref
        if ref is not None:
            assert node.flow is None, "Ref node cannot have nested flow."
            assert node_type in _NODE_ALLOW_REF, f"Only nodes of type {_NODE_ALLOW_REF} can be ref node, but got {node_type.name}."

            assert ref.fe_path is not None 
            root_flow = mgr._root_flow_getter().flow
            node_id_path = ADVProject.get_node_id_path_from_fe_path(ref.fe_path)
            pair = ADVProject.get_flow_node_by_fe_path(root_flow, ref.fe_path)
            assert pair is not None 
            node_parent, node_def = pair
            # if node_type == ADVNodeType.CLASS:
            #     pass 
            # node = mgr._node_id_to_node[node.ref_node_id]
            # node_parent = None
            if len(node_id_path) > 1:
                assert node_parent is not None 
                # node_parent = mgr._node_id_to_node[node_id_path[-2]]
                parent_flow = node_parent.flow
                assert parent_flow is not None
                flow_parser = mgr.get_subflow_parser(node_parent)
                flow_node_gid = node_parent.get_global_uid()
            else:
                fcache = mgr._flow_node_gid_to_cache[TENSORPC_ADV_ROOT_FLOW_ID]
                flow_parser = fcache.parser
                parent_flow = fcache.flow
                flow_node_gid = TENSORPC_ADV_ROOT_FLOW_ID
            if flow_gid == flow_node_gid:
                is_local_ref = True 
                parse_res = None
            else:
                if flow_parser._flow_parse_result is None:
                    flow_parser._parse_flow_recursive(node_parent, flow_node_gid, parent_flow, mgr, visited)
                parse_res = flow_parser._be_nodes_map[node_def.id].parse_res
                assert parse_res is not None 
                if isinstance(parse_res, FlowParseResult):
                    is_node_def_folder = parse_res._is_folder()
                assert flow_parser._flow_parse_result is not None 
                parent_is_folder = flow_parser._flow_parse_result._is_folder()
                parse_res = dataclasses.replace(parse_res, node=dataclasses.replace(node))
            assert node_def.ref is None
            # node_def, parse_res, is_local_ref, node_parent = self._get_or_compile_real_node(parent_node, flow_gid, node, mgr, visited)
            assert node.nType == node_def.nType, f"Ref node type {node.nType} does not match original node type {node_def.nType}."
        else:
            node_def = node
            node_parent = cur_flow_node
            is_local_ref = False
            parent_is_folder = cur_flow_is_folder
            if node.flow is not None:
                flow_parser = mgr.get_subflow_parser(node)
                if flow_parser._flow_parse_result is None:
                    parse_res = flow_parser._parse_flow_recursive(node, node.get_global_uid(), node.flow, mgr, visited)
                else:
                    parse_res = flow_parser._flow_parse_result
                is_subflow_def = True 
                is_node_def_folder = parse_res._is_folder()
            else:
                parse_res = None
        be_node = BackendNode(
            node_def=node_def,
            node=node,
            node_def_parent=node_parent,
            parse_res=parse_res,
            is_local_ref=is_local_ref,
            is_subflow_def=is_subflow_def,
            is_parent_folder=parent_is_folder,
            is_ext_node=parse_res is not None,
            is_node_def_folder=isinstance(parse_res, FlowParseResult) and parse_res.has_subflow,
        )
        assert be_node.is_node_def_folder is is_node_def_folder
        return be_node


    def _preprocess_nodes(self, cur_flow_node: Optional[ADVNodeModel], flow_gid: str, nodes: list[ADVNodeModel], 
            cur_flow_is_folder: bool,
            mgr: ADVProjectBackendManager, visited: set[str]) -> tuple[dict[ADVNodeType, list[BackendNode]], Optional[BackendNode], dict[str, ADVNodeModel]]:
        # currently class node can't have nested class/fragment flow node.
        res: dict[ADVNodeType, list[BackendNode]] = {
            ADVNodeType.GLOBAL_SCRIPT: [],
            ADVNodeType.SYMBOLS: [],
            ADVNodeType.FRAGMENT: [],
            ADVNodeType.OUT_INDICATOR: [],
            ADVNodeType.MARKDOWN: [],
            ADVNodeType.CLASS: [],
        }
        # run ast parse on global script/symbol/fragment code first
        class_name_to_be_node: dict[str, ADVNodeModel] = {}
        for node in nodes:
            node_type = ADVNodeType(node.nType)
            be_node = self._node_to_be_node(node, cur_flow_node, flow_gid, cur_flow_is_folder, mgr, visited)
            res[node_type].append(be_node)
            self._be_nodes_map[node.id] = be_node
            if node.nType == ADVNodeType.CLASS:
                class_name_to_be_node[be_node.node_def.name] = be_node.node_def
            elif node.nType == ADVNodeType.FRAGMENT and be_node.node_def.is_defined_in_class():
                # class dep
                assert be_node.node_def_parent is not None 
                class_name_to_be_node[be_node.node_def_parent.name] = be_node.node_def_parent
        base_inherited_be_node: Optional[BackendNode] = None
        if cur_flow_node is not None and cur_flow_node.nType == ADVNodeType.CLASS:
            if cur_flow_node.cls_inherit_ref is not None:
                base_inherited_be_node = self._node_to_be_node(
                    cur_flow_node, cur_flow_node=cur_flow_node, flow_gid=flow_gid, 
                    cur_flow_is_folder=cur_flow_is_folder, mgr=mgr, visited=visited,
                    is_inherit_ref=True)
                assert base_inherited_be_node.node.ref is None
        return res, base_inherited_be_node, class_name_to_be_node

    def _sort_flow_nodes_topological(self, 
             nodes_dict: dict[str, ADVNodeModel]) -> list[ADVNodeModel]:
        child_nodes = list(nodes_dict.values())
        # 1. sort by pos x
        child_nodes.sort(key=lambda n: n.position.x)
        # 2. build dep map
        if_name_to_desc_node: dict[str, ADVNodeModel] = {}
        for n in child_nodes:
            if n.is_inline_flow_desc_def():
                if_name_to_desc_node[n.name] = n
        node_dep_ids: dict[str, set[str]] = {}
        for n in child_nodes:
            node_dep_ids[n.id] = set()

        for n in child_nodes:
            if not n.is_inline_flow_desc_def():
                if n.inlinesf_name is not None and n.inlinesf_name in if_name_to_desc_node:
                    inline_desc_node = if_name_to_desc_node[n.inlinesf_name]
                    # inline desc node depends on this node
                    node_dep_ids[inline_desc_node.id].add(n.id) 
            if n.is_local_ref_node():
                assert n.ref is not None 
                node_dep_ids[n.id].add(n.ref.node_id) 
        # 3. do heap-based topological sort. 
        sorted_child_nodes: list[ADVNodeModel] = []

        min_heap: list[tuple[float, str, str]] = []
        id_to_node: dict[str, ADVNodeModel] = {n.id: n for n in child_nodes}
        for n in child_nodes:
            if not node_dep_ids[n.id]:
                heapq.heappush(min_heap, (n.position.x, n.name, n.id))
        while min_heap:
            _, _, n_id = heapq.heappop(min_heap)
            n = id_to_node[n_id]
            sorted_child_nodes.append(n)
            for other_id, dep_ids in node_dep_ids.items():
                if n.id in dep_ids:
                    dep_ids.remove(n.id)
                    if not dep_ids:
                        other_node = id_to_node[other_id]
                        heapq.heappush(min_heap, (other_node.position.x, other_node.name, other_node.id))
        assert len(sorted_child_nodes) == len(child_nodes), "Cycle detected in node references."
        return sorted_child_nodes
        
    def _parse_global_scripts(self, be_nodes: list[BackendNode]):
        global_scripts: dict[str, str] = {}
        for be_node in be_nodes:
            node_def = be_node.node_def
            assert node_def.nType == ADVNodeType.GLOBAL_SCRIPT
            assert node_def.impl is not None, f"GLOBAL_SCRIPT node {node_def.id} has no code."
            code = node_def.impl.code
            code_lines = code.splitlines()
            assert len(code_lines) > 0
            end_column = len(code_lines[-1]) + 1

            global_scripts[node_def.id] = (node_def.impl.code)
            parse_res = GlobalScriptParseResult(
                node=dataclasses.replace(be_node.node),
                code=node_def.impl.code,
            )
            be_node.parse_res = parse_res
        global_script = "\n".join(global_scripts.values())
        global_scope = self._global_scope_parser.parse_global_script(global_script)
        # required for all adv meta code
        global_scope.update({
            "dataclasses": dataclasses_plain,
            "ADV": _ADV,
        })
        return global_scripts, global_scope

    def _parse_symbol_groups(self, flow_node: Optional[ADVNodeModel], be_nodes: list[BackendNode], global_scripts: dict[str, str], global_scope: dict[str, Any],
            cls_node_map: dict[str, ADVNodeModel]):
        symbols: list[BackendHandle] = []
        sym_group_dep_qnames: list[str] = []
        symbol_dep_classes: dict[str, ADVNodeModel] = {}
        # use ordered symbol indexes to make sure arguments of fragment are stable
        cnt_base = 0
        for be_node in be_nodes:
            node = be_node.node 
            n_def = be_node.node_def
            cached_parse_res = be_node.parse_res
            if cached_parse_res is not None:
                assert isinstance(cached_parse_res, SymbolParseResult)
                cached_parse_res = cached_parse_res.copy(node.id, cnt_base)
                # use ref node instead of def node
                # code generator need to know whether this is external node
                cached_parse_res = dataclasses.replace(cached_parse_res, node=node)
                sym_group_dep_qnames.extend(cached_parse_res.dep_qnames_for_ext)
                symbol_dep_classes.update({k.name: k for k in cached_parse_res.dep_class_nodes})
                parse_res = cached_parse_res
                symbols.extend(parse_res.symbols)
            else:
                assert n_def.impl is not None, f"symbol node {n_def.id} has no code."
                parser = SymbolParser()
                parse_res = parser.parse_symbol_node(node, n_def.impl.code, global_scope, list(global_scripts.values()), flow_node, cls_node_map)
                parse_res = parse_res.copy(offset=cnt_base, is_sym_handle=True)
                be_node.parse_res = parse_res
                symbols.extend(parse_res.local_symbols)
            cnt_base += parse_res.num_symbols
        return symbols, sym_group_dep_qnames, list(symbol_dep_classes.values())

    def _parse_inline_flow_descs(self, flow_node: Optional[ADVNodeModel], be_nodes: list[BackendNode]) -> dict[str, Optional[BackendNode]]:
        res: dict[str, Optional[BackendNode]] = {}
        for be_node in be_nodes:
            if be_node.node.is_inline_flow_desc_def():
                res[be_node.node.name] = be_node
                # assign a default parse result here.
                be_node.parse_res = FragmentParseResult.create_empty_result(be_node.node)
        # we set `None` for default inline flow.
        if flow_node is not None and flow_node.nType == ADVNodeType.FRAGMENT:
            if flow_node.name not in res:
                res[flow_node.name] = None
        return res

    def _parse_fragments_or_classes(self, be_nodes: list[BackendNode], root_symbol_scope: dict[str, BackendHandle], 
            global_scope: dict[str, Any], parent_node: Optional[ADVNodeModel], inline_flow_desc_map: dict[str, Optional[BackendNode]]):
        # parse fragments, auto-generated edges will also be handled.
        inlineflow_descs: dict[str, InlineFlowDesc] = {}
        for k, v in inline_flow_desc_map.items():
            inlineflow_descs[k] = InlineFlowDesc(
                v, []
            )
        inlineflow_name_to_scope: dict[str, dict[str, BackendHandle]] = {}
        isolated_be_nodes: list[BackendNode] = []
        for be_node in be_nodes:
            n = be_node.node
            if n.is_inline_flow_desc_def():
                # inline flow desc node (def) don't have code, no need to parse here.
                continue 
            n_def = be_node.node_def
            ext_parse_res = be_node.parse_res
            if n.ref is None and n.flow is None:
                assert ext_parse_res is None
            alias_map = None
            # user can use output mapping to alias output symbols
            # of ref nodes or subflow nodes
            if n.alias_map != "":
                # TODO handle error here.
                alias_map = parse_alias_map(n.alias_map)
            # prepare symbol scope for each subflow
            subf_name = n.inlinesf_name
            need_clear_inlinesf_name = False
            if subf_name is not None and subf_name not in inline_flow_desc_map:
                subf_name = None # invalid inline flow name, ignore it
                need_clear_inlinesf_name = True
            if subf_name is not None:
                assert n.name != subf_name, f"Node {n.id} name cannot be same as inlineflow name."
                if subf_name not in inlineflow_name_to_scope:
                    inlineflow_name_to_scope[subf_name] = root_symbol_scope.copy()
                symbol_scope = inlineflow_name_to_scope[subf_name]
            else:
                # isolated fragment won't change symbol_scope.
                symbol_scope = root_symbol_scope
            # print("PARSE", be_node.node.id, symbol_scope.keys())
            if ext_parse_res is not None:
                assert isinstance(ext_parse_res, FragmentParseResult)
                parse_res = ext_parse_res.copy_for_ref_node(be_node)
            else:
                if be_node.is_local_ref:

                    assert n.ref is not None
                    parse_res = self._be_nodes_map[n.ref.node_id].get_parse_res_checked(FragmentParseResult)
                    parse_res = parse_res.copy_for_ref_node(be_node)
                else:
                    assert n_def.nType != ADVNodeType.CLASS, "shouldn't happen"
                    assert n_def.impl is not None, f"fragment node {n_def.id} has no code."
                    if n.id in self._node_id_to_parser:
                        parser = self._node_id_to_parser[n.id]
                        assert isinstance(parser, FragmentParser)
                    else:
                        parser = FragmentParser()
                        self._node_id_to_parser[n.id] = parser
                    parse_res = parser.parse_fragment(
                        n,
                        n_def.impl.code,
                        global_scope,
                        symbol_scope,
                        parent_node,
                    )
            if alias_map is not None:
                parse_res = parse_res.do_alias_map(alias_map)
                parse_res = dataclasses.replace(parse_res, alias_map=n.alias_map)
            if need_clear_inlinesf_name:
                be_node.node.inlinesf_name = None
            be_node.parse_res = parse_res
            # TODO better auto edge.
            # if subf_name is not None and n.ref is None and n.flow is None:
            if subf_name is not None:
                # inline flow node will contribute outputs to symbol scope
                for handle in parse_res.output_handles:
                    symbol_scope[handle.name] = handle 
            # TODO check valid nodes in inline flow, e.g. non-method inline flow can't have method fragment.
            if subf_name is not None:
                # NOTE inline flow desc def node can't be inline flow node (ignored above), you need to create ref node.
                inlineflow_descs[subf_name].flow_be_nodes.append(be_node)
            else:
                if not be_node.is_subflow_def and not be_node.is_ext_node:
                    # subflow nodes are handled separately
                    isolated_be_nodes.append(be_node)
        return inlineflow_descs, isolated_be_nodes

    def _build_flow_connection(self, be_nodes_map: dict[ADVNodeType, list[BackendNode]]) -> FlowConnInternals:
        inp_node_handle_to_node: dict[tuple[str, str], tuple[ADVNodeModel, BackendHandle]] = {}
        node_id_to_inp_handles: dict[str, list[BackendHandle]] = {}
        out_node_handle_to_node: dict[tuple[str, str], tuple[ADVNodeModel, BackendHandle]] = {}
        node_id_to_out_handles: dict[str, list[BackendHandle]] = {}
        auto_edges: list[ADVEdgeModel] = []
        frag_nodes = be_nodes_map[ADVNodeType.FRAGMENT]
        symbol_nodes = be_nodes_map[ADVNodeType.SYMBOLS]
        out_indicator_nodes = be_nodes_map[ADVNodeType.OUT_INDICATOR]
        class_nodes = be_nodes_map[ADVNodeType.CLASS]
        for be_node in (frag_nodes + symbol_nodes + out_indicator_nodes + class_nodes):
            n = be_node.node
            node_id_to_inp_handles[n.id] = []
            node_id_to_out_handles[n.id] = []
            parse_res = self._be_nodes_map[n.id].parse_res
            assert parse_res is not None
            if isinstance(parse_res, FragmentParseResult):
                # ref nodes/nested node only use user edges, no auto edges for inputs
                for handle in parse_res.input_handles:
                    if n.ref is None and handle.is_method_self:
                        # self of method def can be ignored for auto edge generation
                        continue 
                    inp_node_handle_to_node[(n.id, handle.id)] = (n, handle)
                    node_id_to_inp_handles[n.id].append(handle)
                    # inline flow desc (fragment) don't support auto edge.
                    # if n.ref is None and n.flow is None and not n.is_inline_flow_desc():
                    if handle.handle.source_info is not None:
                        # assert handle.handle.source_info, f"{n.id}({handle.id}) input handle has no source info for auto edge."
                        snid = handle.handle.source_info.node_id
                        shid = handle.handle.source_info.handle_id
                        auto_edges.append(ADVEdgeModel(
                            id=self._edge_uid_pool(f"AE-{snid}({shid})->{n.id}({handle.id})"),
                            source=snid,
                            sourceHandle=shid,
                            target=n.id,
                            targetHandle=handle.id,
                            isAutoEdge=True,
                        ))
                for handle in parse_res.output_handles:
                    out_node_handle_to_node[(n.id, handle.id)] = (n, handle)
                    node_id_to_out_handles[n.id].append(handle)
            elif isinstance(parse_res, SymbolParseResult):
                for handle in parse_res.symbols:
                    out_node_handle_to_node[(n.id, handle.id)] = (n, handle)
                    node_id_to_out_handles[n.id].append(handle)
            elif isinstance(parse_res, OutIndicatorParseResult):
                inp_node_handle_to_node[(n.id, parse_res.handle.id)] = (n, parse_res.handle)
                node_id_to_inp_handles[n.id].append(parse_res.handle)
        return FlowConnInternals(
            inp_node_handle_to_node=inp_node_handle_to_node,
            node_id_to_inp_handles=node_id_to_inp_handles,
            out_node_handle_to_node=out_node_handle_to_node,
            node_id_to_out_handles=node_id_to_out_handles,
            auto_edges=auto_edges,
        )

    def _parse_out_indicators(self, be_nodes: list[BackendNode]):
        inlineflow_out_handles: list[tuple[str, BackendHandle]] = []
        for be_node in be_nodes:
            n = be_node.node
            out_indicator_handle = ADVNodeHandle(
                id=ADVConstHandles.OutIndicator,
                name=n.name,
                flags=int(ADVHandleFlags.IS_INPUT),
                type="",
                symbol_name="",
            )
            backend_handle = BackendHandle(handle=out_indicator_handle, index=0)
            # output indicator won't be ref node
            parse_res = OutIndicatorParseResult(
                node=dataclasses.replace(n),
                succeed=True,
                handle=backend_handle,
            )
            be_node.parse_res = parse_res
        return inlineflow_out_handles

    def _parse_markdowns(self, be_nodes: list[BackendNode]):
        for be_node in be_nodes:
            n = be_node.node
            # output indicator won't be ref node
            md_parse_res = MarkdownParseResult(
                node=dataclasses.replace(n),
                succeed=True,
            )
            be_node.parse_res = md_parse_res
        return

    def _parse_user_edges(self, user_edges: list[ADVEdgeModel], flow_conn: FlowConnInternals, inline_flow_descs: dict[str, Optional[BackendNode]]):
        inlineflow_out_handles: list[tuple[str, BackendHandle]] = []
        valid_edges: list[ADVEdgeModel] = []

        for edge in user_edges:
            assert edge.sourceHandle is not None and edge.targetHandle is not None
            source_key = (edge.source, edge.sourceHandle)
            target_key = (edge.target, edge.targetHandle)
            source_key_valid = source_key in flow_conn.out_node_handle_to_node
            target_key_valid = target_key in flow_conn.inp_node_handle_to_node
            if source_key_valid and target_key_valid:
                source_node, source_handle = flow_conn.out_node_handle_to_node[source_key]
                target_node, target_handle = flow_conn.inp_node_handle_to_node[target_key]
                if source_node.nType == ADVNodeType.FRAGMENT or source_node.nType == ADVNodeType.CLASS:
                    if target_node.nType == ADVNodeType.FRAGMENT or target_node.nType == ADVNodeType.CLASS:
                        # we will remove invalid inline sf name later.
                        source_iname = source_node.inlinesf_name
                        if source_iname not in inline_flow_descs:
                            source_iname = None
                        target_iname = target_node.inlinesf_name
                        if target_iname not in inline_flow_descs:
                            target_iname = None
                        if source_iname != target_iname:
                            ADV_LOGGER.error(f"Edge from {edge.source}({edge.sourceHandle})<{source_iname}> "
                                            f"to {edge.target}({edge.targetHandle})<{target_iname}> crosses inlineflow "
                                            "and will be removed.")
                            continue
                if target_node.nType == ADVNodeType.OUT_INDICATOR:
                    source_iname = source_node.inlinesf_name
                    if source_iname not in inline_flow_descs:
                        source_iname = None

                    if source_iname is None:
                        ADV_LOGGER.error(f"Edge from {edge.source}({edge.sourceHandle}) "
                                        f"to {edge.target}({edge.targetHandle}) connects to output indicator "
                                        "but source node is not in inlineflow and will be removed.")
                        continue
                target_handle.handle.set_source_info_inplace(source_node.id, source_handle.handle.id)
                source_handle.target_node_handle_id.add((edge.target, edge.targetHandle))
                if target_node.nType == ADVNodeType.OUT_INDICATOR:
                    source_handle.is_inlineflow_out = True
                    # target_handle belongs to output indicator node
                    # update it for visualization
                    if target_node.get_out_indicator_alias() == "":
                        target_handle.handle.name = source_handle.handle.name
                    target_handle.handle.symbol_name = source_handle.handle.symbol_name
                    target_handle.handle.out_var_name = source_handle.handle.name 
                    source_handle = source_handle.copy()
                    # tell code generator the original var name to access it in current scope.
                    source_handle.handle.out_var_name = source_handle.handle.name 
                    if target_node.get_out_indicator_alias() != "":
                        source_handle.handle.name = target_node.name
                    inlineflow_out_handles.append((edge.source, source_handle))
                valid_edges.append(edge)
            else:
                ADV_LOGGER.warning(f"Edge from {edge.source}({edge.sourceHandle}-{source_key_valid}) "
                                   f"to {edge.target}({edge.targetHandle}-{target_key_valid}) is invalid "
                                   "and will be removed.")
        return valid_edges, inlineflow_out_handles

    def _parse_inlineflow(self, inlineflow_descs: dict[str, InlineFlowDesc], root_symbol_scope: dict[str, BackendHandle], 
            inlineflow_out_handles: list[tuple[str, BackendHandle]], 
            flow_conn: FlowConnInternals, nodes_dict: dict[str, ADVNodeModel]):
        sorted_subflow_descs: dict[str, list[BackendNode]] = {k: [] for k in inlineflow_descs.keys()}
        subflow_node_ids: set[str] = set()
        out_handles_per_flow: dict[str, list[tuple[str, BackendHandle]]] = {}
        for subf_name, iflow_desc in inlineflow_descs.items():
            # look for nodes that have no connection on output handles
            nodes = [nd.node for nd in iflow_desc.flow_be_nodes]
            node_id_to_nd = {nd.node.id: nd for nd in iflow_desc.flow_be_nodes}
            root_nodes: list[ADVNodeModel] = []
            for node in nodes:
                out_handles = flow_conn.node_id_to_out_handles[node.id]
                is_root_node = False
                if not out_handles:
                    is_root_node = True 
                else:
                    has_output_edges = False
                    for handle in out_handles:
                        if handle.target_node_handle_id and not handle.is_inlineflow_out:
                            has_output_edges = True
                            break
                    if has_output_edges:
                        is_root_node = False
                    else:
                        is_root_node = True
                if is_root_node:
                    root_nodes.append(node)
            out_handles_per_flow[subf_name] = []
            subf_node_id_to_nodes = {n.id: n for n in nodes}

            for source_node_id, source_handle in inlineflow_out_handles:
                if source_node_id in subf_node_id_to_nodes:
                    out_handles_per_flow[subf_name].append((source_node_id, source_handle))
            postorder_nodes: list[ADVNodeModel] = []
            visited_nodes: set[str] = set()
            for root_node in root_nodes:
                self._post_order_access_nodes(
                    lambda n: postorder_nodes.append(n),
                    root_node,
                    visited_nodes,
                    flow_conn.node_id_to_inp_handles,
                    nodes_dict,
                )
            sorted_subflow_descs[subf_name] = [node_id_to_nd[n.id] for n in postorder_nodes]
            for n in postorder_nodes:
                subflow_node_ids.add(n.id)
        # create input/output handles of flow if exists
        inlineflow_res: dict[str, InlineFlowParseResult] = {}

        for flow_name, subf_descs in sorted_subflow_descs.items():
            subf_input_handles: list[BackendHandle] = []
            subf_output_handles: list[BackendHandle] = []
            subf_node_id_to_nodes = {n.node.id: n for n in subf_descs}
            for handle in root_symbol_scope.values():
                for target_node_id, _ in handle.target_node_handle_id:
                    if target_node_id in subf_node_id_to_nodes:
                        # input handle connects to outside node
                        handle = handle.copy(prefix=ADVHandlePrefix.Input)
                        handle.handle.flags |= int(ADVHandleFlags.IS_INPUT)
                        handle.handle.source_info = None 
                        subf_input_handles.append(handle)
                        break
            for source_node_id, source_handle in out_handles_per_flow[flow_name]:
                subf_output_handles.append(source_handle)
            subf_output_handles.sort(key=lambda h: h.index)
            desc_be_node = inlineflow_descs[flow_name].desc_node
            inlineflow_res[flow_name] = InlineFlowParseResult(
                node_descs=subf_descs,
                input_handles=subf_input_handles,
                output_handles=subf_output_handles,
                desc_node=desc_be_node,
            )
            if desc_be_node is not None:
                desc_be_node.parse_res = FragmentParseResult(
                    succeed=True,
                    func_name=desc_be_node.node.name,
                    node=dataclasses.replace(desc_be_node.node),
                    input_handles=subf_input_handles,
                    output_handles=subf_output_handles,
                    alias_map=desc_be_node.node.alias_map,
                    out_type="dict",
                    out_type_anno="dict[str, Any]",
                    can_node_be_ref=len(subf_descs) > 0,
                )
        # postprocess all local ref of inline flow desc nodes
        # inline node desc is proceed here, so we have to update parse result
        # for all local ref node.
        node_id_to_inlineflow_descs = {v.desc_node.id: v for k, v in inlineflow_descs.items() if v.desc_node is not None}
        for k, v in self._be_nodes_map.items():
            if v.node is not None and v.node.is_inline_flow_desc() and v.node.ref is not None:
                desc_def_be_node = node_id_to_inlineflow_descs[v.node.ref.node_id].desc_node
                assert desc_def_be_node is not None
                parse_res = desc_def_be_node.get_parse_res_checked(FragmentParseResult)
                parse_res = parse_res.copy_for_ref_node(v)
                v.parse_res = parse_res
        return inlineflow_res

    def _parse_flow_recursive(self, flow_node: Optional[ADVNodeModel], flow_id: str, flow: ADVFlowModel, mgr: ADVProjectBackendManager, visited: set[str]):
        if self._flow_parse_result is not None:
            return self._flow_parse_result
        if flow_id in visited:
            raise ValueError(f"Cyclic flow reference detected at flow id {flow_id}, {visited}")
        ADV_LOGGER.warning(f"Parsing flow {'__ROOT__' if not flow_id else flow_id} ...")
        child_nodes = self._sort_flow_nodes_topological(flow.nodes)
        has_subflow = False
        for node in child_nodes:
            if (node.nType == ADVNodeType.FRAGMENT or node.nType == ADVNodeType.CLASS) and node.flow is not None:
                has_subflow = True 
                break
        if flow_node is not None and has_subflow:
            assert flow_node.nType != ADVNodeType.CLASS, "class node can't have subflow or subclass."
        flow_node_is_class = False
        if flow_node is not None:
            flow_node_is_class = flow_node.nType == ADVNodeType.CLASS
        be_nodes_map, base_inherited_be_node, cls_node_map = self._preprocess_nodes(flow_node, flow_id, child_nodes, has_subflow, mgr, visited)
        inline_flow_descs = self._parse_inline_flow_descs(flow_node, be_nodes_map[ADVNodeType.FRAGMENT])
        # store ref nodes and subflow nodes
        nodes_canbe_ref = (be_nodes_map[ADVNodeType.GLOBAL_SCRIPT] + be_nodes_map[ADVNodeType.SYMBOLS] + 
            be_nodes_map[ADVNodeType.FRAGMENT] + be_nodes_map[ADVNodeType.CLASS])
        ext_be_nodes: list[BackendNode] = []
        for be_node in nodes_canbe_ref:
            if be_node.is_subflow_def or be_node.is_ext_node:
                ext_be_nodes.append(be_node)
        # 1. parse global script nodes to build global scope
        global_scripts, global_scope = self._parse_global_scripts(be_nodes_map[ADVNodeType.GLOBAL_SCRIPT])
        # 2. parse symbol group to build symbol scope
        symbols, sym_group_dep_qnames, symbol_dep_classes = self._parse_symbol_groups(flow_node, be_nodes_map[ADVNodeType.SYMBOLS], global_scripts, global_scope, cls_node_map)
        root_symbol_scope: dict[str, BackendHandle] = {s.handle.symbol_name: s for s in symbols}
        # 3. parse fragments, auto-generated edges will also be handled.
        inlineflow_descs, isolated_be_nodes = self._parse_fragments_or_classes(
            be_nodes_map[ADVNodeType.FRAGMENT] + be_nodes_map[ADVNodeType.CLASS], 
            root_symbol_scope, global_scope, flow_node, inline_flow_descs)
        # now all nodes with output handle are parsed. we can build node-handle map.
        self._parse_out_indicators(be_nodes_map[ADVNodeType.OUT_INDICATOR])
        self._parse_markdowns(be_nodes_map[ADVNodeType.MARKDOWN])
        edges = list(flow.edges.values())
        edges = list(filter(lambda e: not e.isAutoEdge, edges))
        flow_conn = self._build_flow_connection(be_nodes_map)
        valid_edges, inlineflow_out_handles = self._parse_user_edges(edges, flow_conn, inline_flow_descs)
        inlineflow_res = self._parse_inlineflow(inlineflow_descs, root_symbol_scope, 
            inlineflow_out_handles, flow_conn, flow.nodes)
        all_input_handles: list[BackendHandle] = []
        all_output_handles: list[BackendHandle] = []
        can_flow_node_be_ref = False
        auto_field_be_node: Optional[BackendNode] = None
        if flow_node_is_class:
            # TODO currently user must defined a init function for class node.
            assert flow_node is not None 
            out_type = "single"
            out_type_anno = flow_node.name # TODO better name here.
            if flow_node is not None:
                init_fn_node_res: Optional[BackendNode] = None
                for be_node in be_nodes_map[ADVNodeType.FRAGMENT]:
                    if be_node.node_def.is_init_fn():
                        init_fn_node_res = be_node 
                    if be_node.node_def.is_auto_field_fn():
                        auto_field_be_node = be_node
                if init_fn_node_res is not None:
                    init_fn_parse_res = self._be_nodes_map[init_fn_node_res.node.id].get_parse_res_checked(FragmentParseResult)
                    all_input_handles = init_fn_parse_res.input_handles
                    all_output_handles = [
                        FragmentParseResult.create_self_handle(camel_to_snake(flow_node.name), init_fn_node_res.node.id, is_input=False)
                    ]
                    can_flow_node_be_ref = True
                elif auto_field_be_node is not None:
                    auto_init_fn_parse_res = self._be_nodes_map[auto_field_be_node.node.id].get_parse_res_checked(FragmentParseResult)
                    all_input_handles = auto_init_fn_parse_res.input_handles
                    all_output_handles = [
                        FragmentParseResult.create_self_handle(camel_to_snake(flow_node.name), auto_field_be_node.node.id, is_input=False)
                    ]
                    can_flow_node_be_ref = True 
        else:
            out_type = "dict"
            out_type_anno = "dict[str, Any]"

            if flow_node is not None and flow_node.name in inlineflow_res:
                all_input_handles = inlineflow_res[flow_node.name].input_handles
                all_output_handles = inlineflow_res[flow_node.name].output_handles
                can_flow_node_be_ref = len(inlineflow_res[flow_node.name].node_descs) > 0
        misc_nodes = {
            ADVNodeType.GLOBAL_SCRIPT: [p for p in be_nodes_map[ADVNodeType.GLOBAL_SCRIPT]],
            ADVNodeType.OUT_INDICATOR: [p for p in be_nodes_map[ADVNodeType.OUT_INDICATOR]],
            ADVNodeType.SYMBOLS: [p for p in be_nodes_map[ADVNodeType.SYMBOLS]],
            ADVNodeType.MARKDOWN: [p for p in be_nodes_map[ADVNodeType.MARKDOWN]],
        }
        for k, v in self._be_nodes_map.items():
            assert v.parse_res is not None, f"node {k}({ADVNodeType(v.node.nType).name}) isn't parsed, this shouldn't happen."
        flow_parse_res = FlowParseResult(
            node=dataclasses.replace(flow_node) if flow_node is not None else flow_node,
            func_name=flow_node.name if flow_node is not None else "",
            succeed=True,
            flow_conn=flow_conn,
            input_handles=all_input_handles,
            output_handles=all_output_handles,
            out_type=out_type,
            out_type_anno=out_type_anno,
            # flow related fields
            symbol_dep_qnames=list(set(sym_group_dep_qnames)),
            symbol_dep_classes=symbol_dep_classes,
            edges=valid_edges + flow_conn.auto_edges,
            misc_be_nodes=misc_nodes,
            isolated_be_nodes=isolated_be_nodes,
            inlineflow_results=inlineflow_res,
            ext_be_nodes=ext_be_nodes,
            has_subflow=has_subflow if flow_node is not None else True,
            alias_map=flow_node.alias_map if flow_node is not None else "",
            can_node_be_ref=can_flow_node_be_ref,
            auto_field_be_node=auto_field_be_node,
            base_inherited_be_node=base_inherited_be_node,
        )
        self._flow_parse_result = flow_parse_res
        code_spec = flow_parse_res.to_code_lines()
        flow_parse_res.generated_code_lines = code_spec.lines
        flow_parse_res.generated_code = "\n".join(code_spec.lines)
        flow_parse_res.loc = code_spec
        return flow_parse_res

    def _post_order_access_nodes(self, accessor: Callable[[ADVNodeModel], None], node: ADVNodeModel, 
            visited: set[str], node_id_to_inp_handles: dict[str, list[BackendHandle]],
            node_id_to_node: dict[str, ADVNodeModel]) -> None:
        if node.id in visited:
            return
        visited.add(node.id)
        inp_handles = node_id_to_inp_handles.get(node.id, [])
        for handle in inp_handles:
            if handle.handle.source_info is not None:
                source_node = node_id_to_node[handle.handle.source_info.node_id]
                if source_node.nType == ADVNodeType.FRAGMENT or source_node.nType == ADVNodeType.CLASS:
                    # only traverse fragment/class nodes
                    self._post_order_access_nodes(accessor, source_node, visited, node_id_to_inp_handles, node_id_to_node)
        accessor(node)

def _main_diff():
    from deepdiff.diff import DeepDiff

    from tensorpc.apps.adv.test_data.simple import get_simple_nested_model
    from tensorpc.core.datamodel.draft import create_draft_type_only

    model = get_simple_nested_model()

    manager = ADVProjectBackendManager(lambda: model, create_draft_type_only(type(model.flow)))
    manager.sync_project_model()
    manager.parse_all()
    manager.init_all_nodes()
    folders, path_to_code = manager._get_all_files_and_folders(is_relative=True)

    def _simple_accessor(path_parts: list[str], fspath: Path) -> str:
        return path_to_code[fspath]
    proj_parser = ADVProjectParser(path_code_accessor=_simple_accessor)

    flow_reparsed, _ = proj_parser._parse_desc_to_flow_model(
        [], Path(f"{TENSORPC_ADV_FOLDER_FLOW_NAME}.py"), set(), {}
    )
    reparsed_proj = ADVProject(
        flow=flow_reparsed,
        path="",
        import_prefix=""
    )
    ngid_to_path, ngid_to_fpath = reparsed_proj.assign_path_to_all_node()
    reparsed_proj.node_gid_to_path = ngid_to_path
    reparsed_proj.node_gid_to_frontend_path = ngid_to_fpath
    reparsed_proj.update_ref_path(ngid_to_fpath)
    manager2 = ADVProjectBackendManager(lambda: reparsed_proj, create_draft_type_only(type(reparsed_proj.flow)))
    manager2.sync_project_model()
    manager2.parse_all()
    manager2.init_all_nodes()


    print(DeepDiff(model.flow, flow_reparsed, ignore_order=True).pretty())

if __name__ == "__main__":
    _main_diff()