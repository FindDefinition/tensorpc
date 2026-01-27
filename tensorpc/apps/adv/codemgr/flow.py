import ast
import enum
import inspect
from pathlib import Path
import shutil
from typing import Any, Callable, Literal, Optional, Self, TypeGuard, TypeVar, Union

import rich
from tensorpc.apps.adv.codemgr.core import BackendHandle, BaseParseResult, BaseParser, ImplCodeSpec
from tensorpc.apps.adv.codemgr.fragment import FragmentParseResult, FragmentParser, parse_alias_map
from tensorpc.apps.adv.codemgr.proj_parse import ADVProjectParser
from tensorpc.apps.adv.codemgr.symbols import SymbolParseResult, SymbolParser
from tensorpc.apps.adv.constants import TENSORPC_ADV_FOLDER_FLOW_NAME, TENSORPC_ADV_ROOT_FLOW_ID
from tensorpc.constants import PACKAGE_ROOT
import tensorpc.core.dataclass_dispatch as dataclasses

from tensorpc.apps.adv.logger import ADV_LOGGER
from tensorpc.apps.adv.model import ADVConstHandles, ADVEdgeModel, ADVFlowModel, ADVHandleFlags, ADVHandlePrefix, ADVNodeModel, ADVNodeHandle, ADVNodeType, ADVProject
import hashlib
from tensorpc.core.annolib import Undefined, undefined
import dataclasses as dataclasses_plain

from tensorpc.apps.adv.codemgr import markers as adv_markers


from tensorpc.core.funcid import clean_source_code, get_attribute_name
from tensorpc.utils.uniquename import UniqueNamePool
from tensorpc.apps.adv import api as _ADV

_NTYPE_TO_ID_PREFIX = {
    ADVNodeType.FRAGMENT: "Frag",
    ADVNodeType.SYMBOLS: "Sym",
    ADVNodeType.GLOBAL_SCRIPT: "GS",
    ADVNodeType.OUT_INDICATOR: "Out",

}


_default_typing_imports = [
    "Literal",
    "Union",
    "Optional",
    "Any",
    "Callable",
    "Generator",
    "Iterable",
    "Iterator",
    "Type",
    "AsyncGenerator",
    "Awaitable",
    "Coroutine",
    "AsyncIterable",
    "AsyncIterator",
    "Annotated",
]
_default_typing_ext_imports = [
    "Self",
]

_default_collections_imports = [
    "Sequence",
    "Mapping",
]


ADV_MAIN_FLOW_NAME = "__adv_flow_main__"

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
class NodePrepResult:
    node: ADVNodeModel
    node_def: ADVNodeModel
    node_parent: Optional[ADVNodeModel]
    ext_parse_res: Optional[BaseParseResult] = None
    is_local_ref: bool = False
    is_subflow_def: bool = False
    # whether node def parent is folder
    is_parent_folder: bool = False

    @property 
    def is_ext_node(self):
        return self.ext_parse_res is not None

    @property 
    def is_node_def_folder(self):
        return isinstance(self.ext_parse_res, FlowParseResult) and self.ext_parse_res.has_subflow

    def get_name_in_global_scope(self):
        node = self.node
        res = node.name
        if node.ref_node_id is not None:
            res = self.node_def.name
            if node.name != self.node_def.name and node.name != "":
                res = node.name
        return res


@dataclasses.dataclass
class InlineFlowParseResult:
    node_descs: list[NodePrepResult]
    input_handles: list[BackendHandle]
    output_handles: list[BackendHandle]

@dataclasses.dataclass(kw_only=True)
class OutIndicatorParseResult(BaseParseResult):
    handle: BackendHandle

    def to_code_lines(self, id_to_parse_res: dict[str, "BaseParseResult"]):
        assert self.node is not None 
        kwargs_parts = self.get_node_meta_kwargs(self.node)
        source_node_id_str = f"\"{self.handle.handle.source_node_id}\"" if self.handle.handle.source_node_id is not None else "None"
        handle_id_str = f"\"{self.handle.handle.source_handle_id}\"" if self.handle.handle.source_handle_id is not None else "None"
        kwargs_parts.extend([
            f'conn_node_id={source_node_id_str}',
            f'conn_handle_id={handle_id_str}',
        ])
        if self.node.name != "":
            kwargs_parts.append(f'alias="{self.node.name}"')
        kwarg_str = ", ".join(kwargs_parts)
        decorator = f"ADV.{adv_markers.mark_out_indicator.__name__}({kwarg_str})"
        return ImplCodeSpec([
            f"{decorator}",
        ], -1, -1, 1, -1)

    def is_io_handle_changed(self, other_res: Self):
        """Compare io handles between two flow parse result. 
        if changed, all flows that depend on this fragment node (may flow) need to be re-parsed.
        TODO default change?
        """
        h1 = self.handle 
        h2 = other_res.handle
        if h1.symbol_name != h2.symbol_name or h1.name != h2.name or h1.handle.type != h2.handle.type or h1.handle.default != h2.handle.default:
            return True
        return False

@dataclasses.dataclass(kw_only=True)
class MarkdownParseResult(BaseParseResult):
    pass

@dataclasses.dataclass(kw_only=True)
class GlobalScriptParseResult(BaseParseResult):
    code: str

    def to_code_lines(self, id_to_parse_res: dict[str, "BaseParseResult"]):
        assert self.node is not None 
        kwargs_parts = self.get_node_meta_kwargs(self.node)
        if self.node.ref_import_path is not None:
            kwargs_parts.append(f'ref_import_path={repr(self.node.ref_import_path)}')
        kwarg_str = ", ".join(kwargs_parts)
        mark_stmt = f"ADV.{adv_markers.mark_global_script.__name__}(name=\"{self.node.name}\", {kwarg_str})"
        mark_end_stmt = f"ADV.{adv_markers.mark_global_script_end.__name__}()"
        lines = self.code.splitlines()
        assert len(lines) > 0
        end_column = len(lines[-1]) + 1
        return ImplCodeSpec([
            mark_stmt,
            *lines,
            mark_end_stmt,
        ], 1, 1, len(lines), end_column)

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

@dataclasses.dataclass(kw_only=True)
class FlowParseResult(FragmentParseResult):
    edges: list[ADVEdgeModel]
    symbol_dep_qnames: list[str] = dataclasses.field(default_factory=list)
    isolate_fragments: list[ADVNodeModel] = dataclasses.field(default_factory=list)
    inlineflow_results: dict[str, InlineFlowParseResult] = dataclasses.field(default_factory=dict)
    misc_nodes: dict[ADVNodeType, list[ADVNodeModel]] = dataclasses.field(default_factory=dict)
    ext_node_descs: list[NodePrepResult] = dataclasses.field(default_factory=list)
    generated_code_lines: list[str] = dataclasses.field(default_factory=list)
    generated_code: str = ""
    has_subflow: bool = False
    is_main_inlineflow_empty: bool = True
    
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

    def to_code_lines(self, id_to_parse_res: dict[str, "BaseParseResult"]):
        user_edges = [e for e in self.edges if not e.isAutoEdge]
        flow_name = self.node.name if self.node is not None else "__ADV_ROOT__"
        flow_import_path = self.get_path_list()
        lines = [
            f"# ADV Flow Definition. name: {flow_name}, import path: {flow_import_path}, relative fspath: {self.get_code_relative_path()}",
            "from tensorpc.apps.adv import api as ADV",
            "import dataclasses",
        ]
        default_typing_import = f"from typing import {', '.join(_default_typing_imports)}"
        default_typing_ex_import = f"from typing_extensions import {', '.join(_default_typing_ext_imports)}"
        default_collections_import = f"from collections.abc import {', '.join(_default_collections_imports)}"
        lines.extend([
            default_typing_import,
            default_typing_ex_import,
            default_collections_import,
        ])
        # if self.node is not None:
            # mark parent node meta
        # global scripts
        # from ....apps import adv
        gs_lines_all: list[str] = []
        if self.misc_nodes[ADVNodeType.GLOBAL_SCRIPT]:
            gs_lines_all.insert(0, "# ------ ADV Global Script Region ------")

        for node in self.misc_nodes[ADVNodeType.GLOBAL_SCRIPT]:
            parse_res = id_to_parse_res.get(node.id, None)
            assert parse_res is not None and isinstance(parse_res, GlobalScriptParseResult)
            parse_res.lineno = len(lines) + len(gs_lines_all) + 1
            code_spec = parse_res.to_code_lines(id_to_parse_res)
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
        if self.ext_node_descs:
            
            for node_desc in self.ext_node_descs:
                node = node_desc.node
                if node.ref_node_id is not None:
                    ref_import_path = node.ref_import_path
                    assert ref_import_path is not None 
                    if node_desc.is_node_def_folder:
                        ref_import_path = ref_import_path + [TENSORPC_ADV_FOLDER_FLOW_NAME]
                    elif node_desc.node_def.flow is not None:
                        # inline flow node
                        inline_flow_name = node_desc.node_def.name
                        ref_import_path = ref_import_path + [inline_flow_name]

                    import_stmt = f"from {dot_prefix}{'.'.join(ref_import_path)} import {node_desc.node_def.name}"
                    if node.name != "":
                        import_stmt += f" as {node.name}"
                    ref_dep_lines.append(import_stmt)
                else:
                    assert node_desc.is_subflow_def
                    # is subflow def node
                    assert node.flow is not None, "only ref node or subflow node can be external node"
                    if node.inlinesf_name is not None:
                        if node_desc.is_node_def_folder:
                            # we only need to import it when it is used in inline flow.
                            ref_dep_lines.append(f"from .{node.name}.{TENSORPC_ADV_FOLDER_FLOW_NAME} import {node.name}")
                        else:
                            ref_dep_lines.append(f"from .{node.name} import {node.name}")

            for node_desc in self.ext_node_descs:
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
                    mark_stmt = f"ADV.{adv_markers.mark_subflow_def.__name__}(name=\"{node_desc.node_def.name}\", {kwargs_str})"
                else:
                    mark_stmt = f"ADV.{adv_markers.mark_ref_node.__name__}({node_desc.get_name_in_global_scope()}, {node.nType}, {kwargs_str})"
                ref_dep_lines.append(mark_stmt)
        
        if ref_dep_lines:
            lines.append("# ------ ADV Ref/Subflow Nodes Dependency Region (Optional) ------")
            lines.append(f"ADV.{adv_markers.mark_ref_node_dep.__name__}()")
            lines.extend(ref_dep_lines)
            lines.append(f"ADV.{adv_markers.mark_ref_node_dep_end.__name__}()")
        # symbol groups
        symbol_group_lines: list[str] = []
        if self.misc_nodes[ADVNodeType.SYMBOLS]:
            symbol_group_lines.append("# ------ ADV Symbol Def Region ------")
        for node in self.misc_nodes[ADVNodeType.SYMBOLS]:
            if node.ref_node_id is not None:
                continue 
            parse_res = id_to_parse_res.get(node.id, None)
            assert parse_res is not None and isinstance(parse_res, SymbolParseResult)
            parse_res.lineno = len(lines) + len(symbol_group_lines) + 1
            code_spec = parse_res.to_code_lines(id_to_parse_res)
            parse_res.loc = code_spec
            symbol_group_lines.extend(code_spec.lines)
        lines.extend(symbol_group_lines)
        # isolate fragments
        isolate_fragment_lines: list[str] = []
        for node in self.isolate_fragments:
            parse_res = id_to_parse_res.get(node.id, None)
            assert parse_res is not None and isinstance(parse_res, FragmentParseResult)
            parse_res.lineno = len(lines) + len(isolate_fragment_lines) + 1
            code_spec = parse_res.to_code_lines(id_to_parse_res)
            parse_res.loc = code_spec
            isolate_fragment_lines.extend(code_spec.lines)
        lines.extend(isolate_fragment_lines)
        # subflow nodes
        inlineflow_lines: list[str] = []
        for inline_name, inline_desc in self.inlineflow_results.items():
            for node_desc in inline_desc.node_descs:
                node = node_desc.node
                if node.ref_node_id is not None:
                    continue 
                parse_res = id_to_parse_res.get(node.id, None)
                assert parse_res is not None and isinstance(parse_res, FragmentParseResult)
                parse_res.lineno = len(lines) + len(inlineflow_lines) + 1
                code_spec = parse_res.to_code_lines(id_to_parse_res)
                parse_res.loc = code_spec

                inlineflow_lines.extend(code_spec.lines)
            # build subflow
            func_name = inline_name
            inlineflow_fn_lines = [
                f"@ADV.{adv_markers.mark_inlineflow.__name__}()",
                f"def {func_name}(",
            ]
            inlineflow_fn_lines += FragmentParseResult.get_signature_lines_from_handles(inline_desc.input_handles)
            inlineflow_fn_lines.append(f") -> dict[str, Any]:")
            body_lines: list[str] = []
            for node_desc in inline_desc.node_descs:
                node = node_desc.node
                parse_res = id_to_parse_res.get(node.id, None)
                assert parse_res is not None and isinstance(parse_res, FragmentParseResult)
                arg_name_parts: list[str] = []
                for h in parse_res.input_handles:
                    if h.handle.source_node_id is None:
                        # TODO handle default value
                        arg_name_parts.append("ADV.MISSING")
                    else:
                        arg_name_parts.append(h.handle.name)
                arg_names = ", ".join(arg_name_parts)
                anno = ""
                if node.ref_node_id is not None:
                    # use Annotated to attach ref node meta
                    arg_parts = [f"\"{node.id}\"", f"\"{node.ref_node_id}\"", f"({node.position.x}, {node.position.y})"]
                    if node.alias_map != "":
                        arg_parts.append(f"\"{node.alias_map}\"")
                    else:
                        arg_parts.append(f"\"\"")
                    if node_desc.is_local_ref:
                        arg_parts.append(f"True")
                    arg_str = ", ".join(arg_parts)
                    anno = f": Annotated[Any, ADV.RefNodeMeta({arg_str})]" 
                if parse_res.out_type == "single":
                    var_name = parse_res.output_handles[0].name
                    body_lines.append(f"{var_name}{anno} = {parse_res.func_name}({arg_names})")
                elif parse_res.out_type == "tuple":
                    out_names = ", ".join([h.handle.name for h in parse_res.output_handles])
                    body_lines.append(f"{out_names}{anno} = {parse_res.func_name}({arg_names})")
                else:
                    # dict
                    body_lines.append(f"_adv_tmp_out{anno} = {parse_res.func_name}({arg_names})")
                    for h in parse_res.output_handles:
                        assert not isinstance(h.handle.dict_key, Undefined)
                        body_lines.append(f"{h.handle.name} = _adv_tmp_out['{h.handle.dict_key}']")
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
            inlineflow_lines.extend(inlineflow_fn_lines)
        lines.extend(inlineflow_lines)
        out_indicator_lines: list[str] = []
        if self.misc_nodes[ADVNodeType.OUT_INDICATOR]:
            out_indicator_lines.append("# ------ ADV Out Indicator Region ------")
        for node in self.misc_nodes[ADVNodeType.OUT_INDICATOR]:
            parse_res = id_to_parse_res.get(node.id, None)
            assert parse_res is not None and isinstance(parse_res, OutIndicatorParseResult)
            parse_res.lineno = len(lines) + len(out_indicator_lines) + 1
            code_spec = parse_res.to_code_lines(id_to_parse_res)
            parse_res.loc = code_spec
            out_indicator_lines.extend(code_spec.lines)
        if out_indicator_lines:
            lines.extend(out_indicator_lines)
        markdown_lines: list[str] = []

        if self.misc_nodes[ADVNodeType.MARKDOWN]:
            markdown_lines.append("# ------ ADV Markdown Region ------")
        for node in self.misc_nodes[ADVNodeType.MARKDOWN]:
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

        return ImplCodeSpec(
            lines, -1, -1, -1, -1
        )

@dataclasses_plain.dataclass
class ModifyParseCache:
    visited: set[str]
    old_parse_res: dict[str, tuple[FlowParseResult, dict[str, BaseParseResult]]]

@dataclasses_plain.dataclass
class FlowCache:
    flow: ADVFlowModel
    parser: "FlowParser"
    parent_node: Optional[ADVNodeModel]
    node_name_pool: UniqueNamePool = dataclasses_plain.field(default_factory=UniqueNamePool)
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
            if cache.node.ref_node_id is not None:
                ref_cache = node_gid_to_cache[cache.node.get_ref_global_uid()]
                if cache.node.is_local_ref_node():
                    ref_cache.ref_node_ids_local.append(cache.node.id)
                    ref_cache.ref_gids_local.append(gid)
                else:
                    ref_cache.ref_gids_external.append(gid)
        for flow_gid, cache in flow_node_gid_to_cache.items():
            cache.child_flow_gids.clear()
            cache.dep_flow_gids.clear()
            cache.all_child_flow_gids.clear()
            cache.topological_sorted_index = -1
        for flow_gid, cache in flow_node_gid_to_cache.items():
            for node in cache.flow.nodes.values():
                if node.ref_node_id is not None:
                    ref_node_cache = node_gid_to_cache[node.get_ref_global_uid()]
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
            code_lines = flow_parse_res.to_code_lines(parser._node_id_to_parse_result).lines
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
        self._flow_node_gid_to_cache[TENSORPC_ADV_ROOT_FLOW_ID] = FlowCache(
            root_flow, 
            FlowParser(), 
            None
        )
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
                    fcache.node_name_pool(child_node.name)
                    fcache.node_id_pool(child_node.id)
                    _traverse_node(child_node)
        
        for node in root_flow.nodes.values():
            self._node_gid_to_cache[node.get_global_uid()] = NodeConnCache(node, TENSORPC_ADV_ROOT_FLOW_ID)
            # self._node_gid_to_flow_node_gid[node.get_global_uid()] = TENSORPC_ADV_ROOT_FLOW_ID
            _traverse_node(node)
        self._reflesh_dependency(self._node_gid_to_cache, self._flow_node_gid_to_cache)


    def parse_all(self):
        # must be called after sync_project_model
        visited: set[str] = set()
        for flow_gid, fcache in self._flow_node_gid_to_cache.items():
            flow_parser = fcache.parser
            res = flow_parser._parse_flow_recursive(fcache.parent_node, flow_gid, fcache.flow, self, visited)
            # fcache.flow_code = "\n".join(res.to_code_lines(flow_parser._node_id_to_parse_result))
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
            # fcache.flow_code = "\n".join(res.to_code_lines(flow_parser._node_id_to_parse_result))

    def _get_flow_code_lineno_by_node_gid(self, node_gid: str) -> Optional[FlowCodeFragment]:
        node_cache_origin = self._node_gid_to_cache[node_gid]
        node_cache = node_cache_origin
        is_ref_node = False
        
        if node_cache.node.ref_node_id is not None:
            ref_node_gid = node_cache.node.get_ref_global_uid()
            node_cache = self._node_gid_to_cache[ref_node_gid] 
            node_gid = ref_node_gid
            is_ref_node = True
        if node_cache.node.flow is not None:
            # currently we don't support show code of subflow.
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

        parse_res = parser._node_id_to_parse_result[self._node_gid_to_cache[node_gid].node.id]
        flow_code = "\n".join(flow_parse_res.to_code_lines(parser._node_id_to_parse_result).lines)
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
                parse_res = parser._node_id_to_parse_result.get(node_id, None)
                if parse_res is not None:
                    fe_handles = self._get_handles_from_parse_res(parse_res)
                    if fe_handles is not None:
                        node.handles = fe_handles
                else:
                    # TODO 
                    ADV_LOGGER.warning(f"Node {node_id} in flow {flow_gid} has no parse result.")
            fcache.flow.edges = {e.id: e for e in parser._flow_parse_result.edges}

    def _get_flow_gid_from_node_gid(self, node_gid: str) -> str:
        return self._node_gid_to_cache[node_gid].flow_node_gid

    def _is_node_subflow(self, node_gid: str):
        # TODO cache result.
        node = self._node_gid_to_cache[node_gid].node
        if node.ref_node_id is None:
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
                flow_parser = self._flow_node_gid_to_cache[flow_gid].parser
                if flow_parser._flow_parse_result is not None:
                    gcache.old_parse_res[flow_gid] = (flow_parser._flow_parse_result, flow_parser._node_id_to_parse_result)
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
                prev_nid_to_parse_res: dict[str, BaseParseResult] = {}
                if flow_gid in gcache.old_parse_res:
                    prev_flow_parse_res, prev_nid_to_parse_res = gcache.old_parse_res[flow_gid]
                flow_parse_res = flow_parser._parse_flow_recursive(fcache.parent_node, flow_gid, flow, self, gcache.visited) 
                flow_code_changed: bool = True
                if prev_flow_parse_res is not None:
                    flow_code_changed = prev_flow_parse_res.generated_code != flow_parse_res.generated_code
                if flow_code_changed:
                    flow_code_changes[str(root_path / flow_parse_res.get_code_relative_path())] = flow_parse_res.generated_code
                node_id_to_parse_res = flow_parser._node_id_to_parse_result

                symbol_group_gid_may_change: set[str] = set()
                frag_id_may_change: set[str] = set()
                remain_change_ids: set[str] = set()

                # 1. process global script nodes, extract symbol groups may change
                for node_gid in changed_node_gids:
                    if node_gid in deleted_node_gid_map:
                        # deleted gs/sym node always trigger symbol group change check
                        node = deleted_node_gid_map[node_gid]
                        if node.nType == ADVNodeType.GLOBAL_SCRIPT:
                            for node_id, node_parse_res in node_id_to_parse_res.items():
                                cur_node = flow.nodes[node_id]
                                if cur_node.nType == ADVNodeType.SYMBOLS and cur_node.ref_node_id is None:
                                    symbol_group_gid_may_change.add(cur_node.get_global_uid())
                        elif node.nType == ADVNodeType.SYMBOLS:
                            symbol_group_gid_may_change.add(node_gid)
                        # we don't care about frag/other node deletion here because
                        # they only affect inline flow, which is already handled in flow parse.
                        continue

                    node_cache = self._node_gid_to_cache[node_gid]
                    node = node_cache.node
                    node_parse_res = node_id_to_parse_res[node.id]

                    if node.nType == ADVNodeType.GLOBAL_SCRIPT:
                        is_new_node = node.id not in prev_nid_to_parse_res
                        is_error_change = True 
                        if node.id in prev_nid_to_parse_res:
                            prev_node_parse_res = prev_nid_to_parse_res[node.id]
                            is_error_change = prev_node_parse_res.succeed != node_parse_res.succeed
                        change = ADVNodeChange.empty(node.name, node_parse_res, is_new=is_new_node, error_status_changed=is_error_change)
                        if node.ref_node_id is None:
                            if node.id in prev_nid_to_parse_res:
                                prev_node_parse_res = prev_nid_to_parse_res[node.id]
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
                            for node_id, node_parse_res in node_id_to_parse_res.items():
                                cur_node = flow.nodes[node_id]
                                if cur_node.nType == ADVNodeType.SYMBOLS and cur_node.ref_node_id is None:
                                    symbol_group_gid_may_change.add(cur_node.get_global_uid())
                        if node.ref_node_id is None:
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
                    elif node.nType == ADVNodeType.FRAGMENT:
                        frag_id_may_change.add(node.id)
                    else:
                        remain_change_ids.add(node.id)

                # 2. process all symbol group nodes
                for sym_node_gid in symbol_group_gid_may_change:
                    if sym_node_gid in deleted_node_gid_map:
                        # deleted sym node always trigger all frag node change check
                        for node_id in node_id_to_parse_res.keys():
                            cur_node = flow.nodes[node_id]
                            if cur_node.nType == ADVNodeType.FRAGMENT and cur_node.ref_node_id is None and cur_node.flow is None:
                                frag_id_may_change.add(node_id)
                        continue
                    else:
                        node_cache = self._node_gid_to_cache[sym_node_gid]
                        sym_node = node_cache.node
                        sym_node_id = sym_node.id
                    sym_node = flow.nodes[sym_node_id]
                    sym_node_gid = sym_node.get_global_uid()
                    node_cache = self._node_gid_to_cache[sym_node_gid]
                    node_parse_res = node_id_to_parse_res[sym_node_id]
                    is_new_node = sym_node.id not in prev_nid_to_parse_res
                    is_error_change = True 
                    if sym_node.id in prev_nid_to_parse_res:
                        prev_node_parse_res = prev_nid_to_parse_res[sym_node.id]
                        is_error_change = prev_node_parse_res.succeed != node_parse_res.succeed
                    change = ADVNodeChange.empty(name=sym_node.name, is_new=is_new_node, parse_res=node_parse_res, error_status_changed=is_error_change)
                    if sym_node.id in prev_nid_to_parse_res:
                        prev_node_parse_res = prev_nid_to_parse_res[sym_node.id]
                        assert isinstance(node_parse_res, SymbolParseResult)
                        assert isinstance(prev_node_parse_res, SymbolParseResult)
                        is_sym_changed = node_parse_res.is_io_handle_changed(prev_node_parse_res)
                        is_sym_name_changed = node_parse_res.symbol_cls_name != prev_node_parse_res.symbol_cls_name
                    else:
                        is_sym_changed = True
                        is_sym_name_changed: bool = False
                    if is_sym_changed:
                        change.flags |= NodeChangeFlag.HANDLES
                        for node_id in node_id_to_parse_res.keys():
                            cur_node = flow.nodes[node_id]
                            if cur_node.nType == ADVNodeType.FRAGMENT and cur_node.ref_node_id is None and cur_node.flow is None:
                                frag_id_may_change.add(node_id)
                    if sym_node.ref_node_id is None:
                        if is_sym_changed or is_sym_name_changed:
                            next_layer_nodes.update(node_cache.ref_gids_external)
                        if is_sym_name_changed:
                            change.flags |= NodeChangeFlag.NAME
                    if change.flags != NodeChangeFlag(0):
                        node_changes[sym_node_gid] = change

                # 3. prepare frag nodes
                frag_id_may_change_next: set[str] = set(frag_id_may_change)
                for frag_node_id in frag_id_may_change:
                    frag_node = flow.nodes[frag_node_id]
                    frag_node_cache = self._node_gid_to_cache[frag_node.get_global_uid()]
                    # always check local ref frag nodes to simplify logic
                    frag_id_may_change_next.update(frag_node_cache.ref_node_ids_local)
                # 4. process fragment nodes
                for node_id in frag_id_may_change_next:
                    node_parse_res = node_id_to_parse_res[node_id]
                    cur_node = flow.nodes[node_id]
                    node_gid = cur_node.get_global_uid()
                    node_cache = self._node_gid_to_cache[node_gid]
                    is_new_node = node_id not in prev_nid_to_parse_res
                    is_error_change = True 
                    if node_id in prev_nid_to_parse_res:
                        prev_node_parse_res = prev_nid_to_parse_res[node_id]
                        is_error_change = prev_node_parse_res.succeed != node_parse_res.succeed
                    change = ADVNodeChange.empty(name=cur_node.name, is_new=is_new_node, parse_res=node_parse_res, error_status_changed=is_error_change)
                    is_ref_node = cur_node.ref_node_id is not None
                    if node_id in prev_nid_to_parse_res:
                        prev_node_parse_res = prev_nid_to_parse_res[node_id]
                        assert isinstance(node_parse_res, FragmentParseResult)
                        assert isinstance(prev_node_parse_res, FragmentParseResult)
                        prev_node = prev_node_parse_res.get_node_checked()
                        is_name_changed = node_parse_res.func_name != prev_node_parse_res.func_name
                        is_node_io_changed = node_parse_res.is_io_handle_changed(prev_node_parse_res)
                        is_alias_map_changed = cur_node.alias_map != prev_node.alias_map
                        is_inline_flow_changed = cur_node.inlinesf_name != prev_node.inlinesf_name
                    else:
                        is_name_changed = not is_ref_node
                        is_node_io_changed = True
                        is_alias_map_changed = is_ref_node
                        is_inline_flow_changed = cur_node.inlinesf_name is not None
                    # local changes without ref
                    if is_node_io_changed:
                        change.flags |= NodeChangeFlag.HANDLES
                    if is_alias_map_changed:
                        change.flags |= NodeChangeFlag.ALIAS_MAP
                    if is_inline_flow_changed:
                        change.flags |= NodeChangeFlag.INLINE_FLOW

                    if not is_ref_node and cur_node.flow is None and is_node_io_changed:
                        next_layer_nodes.update(node_cache.ref_gids_external)
                    if not is_ref_node and is_name_changed:
                        change.flags |= NodeChangeFlag.NAME
                        next_layer_nodes.update(node_cache.ref_gids_external)
                    if change.flags != NodeChangeFlag(0):
                        node_changes[node_gid] = change
                # 5. process remain nodes
                for node_id in remain_change_ids:
                    node_parse_res = node_id_to_parse_res[node_id]
                    node = flow.nodes[node_id]
                    node_gid = node.get_global_uid()
                    is_new_node = node_id not in prev_nid_to_parse_res
                    change = ADVNodeChange.empty(name=node.name, is_new=is_new_node, parse_res=node_parse_res)
                    if node.nType == ADVNodeType.OUT_INDICATOR:
                        is_oic_alias_changed: bool = True
                        is_oic_handle_changed: bool = True
                        if node_id in prev_nid_to_parse_res:
                            prev_node_parse_res = prev_nid_to_parse_res[node.id]
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

                if prev_flow_parse_res is None or not self._is_edges_equal(prev_flow_parse_res.edges, flow_parse_res.edges):
                    new_flow_edges[flow_gid] = flow_parse_res.edges

                # if flow main inline flow io is changed by any reparse, reparse all ref fragment+inline nodes
                is_inlineflow_changed = True 
                if prev_flow_parse_res is not None:
                    assert isinstance(prev_flow_parse_res, FlowParseResult)
                    is_inlineflow_changed = prev_flow_parse_res.is_io_handle_changed(flow_parse_res)
                if is_inlineflow_changed:
                    # TODO currently root flow can't be used as a fragment node.
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
        prev_code = node.impl.code
        prev_code_clean = clean_source_code(prev_code.splitlines())
        new_code_clean = clean_source_code(new_code.splitlines())
        node.impl.code = new_code
        if prev_code_clean == new_code_clean:
            # TODO only need to modify code in frontend/backend (file store).
            return ADVProjectChange()
        return self._reparse_changed_node(node_gid)

    def modify_alias_map(self, node_gid: str, new_alias_map: str):
        # new_alias_map is local change, 
        node = self._node_gid_to_cache[node_gid].node
        assert node.ref_node_id is not None or node.flow is not None, "Only ref node or subflow node can have alias map."
        node.alias_map = new_alias_map
        flow_gid = self._get_flow_gid_from_node_gid(node_gid)
        return self._reparse_flow(flow_gid)


    def modify_node_name(self, node_gid: str, new_name: str):
        cache = self._node_gid_to_cache[node_gid]
        node = cache.node
        assert not node.is_local_ref_node(), "you can't change name of local ref node."
        old_name = node.name
        if old_name == new_name:
            return ADVProjectChange()
        node.name = new_name
        flow_gid = cache.flow_node_gid
        fcache = self._flow_node_gid_to_cache[flow_gid]
        fcache.node_name_pool.pop_if_exists(old_name)
        # we use node id instead of name, 
        # name is only used in code generation and user code.
        # flow itself (include scheduler) don't use name.
        return self._reparse_changed_node(node_gid)

    def add_new_node(self, flow_node_gid: str, new_node: ADVNodeModel):
        fcache = self._flow_node_gid_to_cache[flow_node_gid]
        if new_node.nType in [ADVNodeType.SYMBOLS, ADVNodeType.FRAGMENT]:
            assert new_node.name not in fcache.node_name_pool, f"Node name {new_node.name} already exists in flow."
            if fcache.parent_node is not None:
                assert new_node.name != fcache.parent_node.name, "Symbol group/fragment name must be different from parent flow name."
            fcache.node_name_pool(new_node.name)
        new_node.id = fcache.node_id_pool(new_node.id)
        adv_proj = self._root_flow_getter()
        fcache.flow.nodes[new_node.id] = new_node

        # create paths
        if (new_node.flow is None or not new_node.flow.nodes) and new_node.ref_node_id is None:
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
            if new_node.nType in [ADVNodeType.SYMBOLS, ADVNodeType.FRAGMENT]:
                fcache.node_name_pool.pop_if_exists(new_node.name)
            ngid_to_path, ngid_to_fpath = adv_proj.assign_path_to_all_node()
            adv_proj.update_ref_path(ngid_to_fpath)
            adv_proj.node_gid_to_path = ngid_to_path
            adv_proj.node_gid_to_frontend_path = ngid_to_fpath
            raise e
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
                    # import rich 
                    # rich.print(handles)
                    node_draft.handles = handles
                elif node_change.flags & NodeChangeFlag.NAME:
                    node_draft.name = node.name
                elif node_change.flags & NodeChangeFlag.ALIAS_MAP:
                    node_draft.alias_map = node.alias_map
                elif node_change.flags & NodeChangeFlag.INLINE_FLOW:
                    node_draft.inlinesf_name = node.inlinesf_name
                elif node_change.flags & NodeChangeFlag.IMPL_CODE:
                    assert node.nType == ADVNodeType.MARKDOWN
                    assert node.impl is not None 
                    node_draft.impl.code = node.impl.code

        for flow_node_gid, edges in change.new_flow_edges.items():
            edges_dict =  {e.id: e for e in edges}
            assert len(edges_dict) == len(edges)
            if flow_node_gid == "":
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
                if node.ref_node_id is None and node.nType in (ADVNodeType.GLOBAL_SCRIPT, ADVNodeType.SYMBOLS, ADVNodeType.FRAGMENT):
                    if flow_node_gid == flow_gid and node.nType in (ADVNodeType.GLOBAL_SCRIPT, ADVNodeType.SYMBOLS):
                        # local ref of global script and symbol group is not allowed
                        continue
                    if node.flow is not None:
                        fcache_cur_node = self._flow_node_gid_to_cache[node.get_global_uid()]
                        parse_res = fcache_cur_node.parser.get_flow_parse_result_checked()
                        if parse_res.is_main_inlineflow_empty:
                            continue 
                    res.append(node)
        return res 

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
        # WARNING: edge change is already changed outside.
        fcache = self._flow_node_gid_to_cache[flow_node_gid]
        flow = fcache.flow
        changed_node_gids: list[str] = []
        # edge change may change out indicator node handles
        for node in flow.nodes.values():
            if node.nType in [ADVNodeType.OUT_INDICATOR]:
                changed_node_gids.append(node.get_global_uid())

        return self._reparse_changed_node_internal(flow_node_gid, changed_node_gids)

    def modify_flow_code_external(self, new_flow_codes: dict[str, str]):
        # only allow modify impl of fragment/symbolgroup/global script.
        # add/delete node isn't allowed via external editor.
        old_fcaches = {gid: self._flow_node_gid_to_cache[gid] for gid in new_flow_codes}

        new_path_to_code = {old_fcaches[gid].get_code_relative_path_checked(): code for gid, code in new_flow_codes.items()}
        fspath_to_old_fcaches = {v.get_code_relative_path_checked(): v for v in old_fcaches.values()}

        all_flows_no_change = {v.get_code_relative_path_checked(): v.flow for v in self._flow_node_gid_to_cache.values()}
        for new_path in new_path_to_code:
            all_flows_no_change.pop(new_path)
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

            proj_parser._parse_desc_to_flow_model(path_with_name, path, set(), parsed_flows)
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
                is_ref_change = new_node.ref_node_id != old_node.ref_node_id or new_node.ref_import_path != old_node.ref_import_path
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
        self._code_to_global_scope: dict[str, dict[str, Any]] = {}

        self._node_id_to_parse_result: dict[str, BaseParseResult] = {}
        self._node_id_to_parser: dict[str, BaseParser] = {}

        self._flow_parse_result: Optional[FlowParseResult] = None
        

        self._edge_uid_pool = UniqueNamePool()

    def clear_parse_cache(self):
        self._node_id_to_parse_result = {}
        self._node_id_to_parser = {}
        self._flow_parse_result = None
        self._edge_uid_pool.clear()

    def clear_parse_result(self):
        self._node_id_to_parse_result = {}
        self._flow_parse_result = None
        self._edge_uid_pool.clear()

    def get_flow_parse_result_checked(self) -> FlowParseResult:
        assert self._flow_parse_result is not None
        return self._flow_parse_result

    def serialize_to_code(self):
        assert self._flow_parse_result is not None
        return self._flow_parse_result.to_code_lines(self._node_id_to_parse_result)

    def _preprocess_nodes(self, cur_flow_node: Optional[ADVNodeModel], flow_gid: str, nodes: list[ADVNodeModel], 
            cur_flow_is_folder: bool,
            mgr: ADVProjectBackendManager, visited: set[str]) -> dict[ADVNodeType, list[NodePrepResult]]:
        res: dict[ADVNodeType, list[NodePrepResult]] = {
            ADVNodeType.GLOBAL_SCRIPT: [],
            ADVNodeType.SYMBOLS: [],
            ADVNodeType.FRAGMENT: [],
            ADVNodeType.OUT_INDICATOR: [],
            ADVNodeType.MARKDOWN: [],
        }
        node_allow_ref = set([ADVNodeType.GLOBAL_SCRIPT, ADVNodeType.SYMBOLS, ADVNodeType.FRAGMENT])
        for node in nodes:
            node_type = ADVNodeType(node.nType)

            is_subflow_def = False
            is_local_ref = False
            parent_is_folder = False
            is_node_def_folder = False
            if node.ref_node_id is not None:
                assert node.flow is None, "Ref node cannot have nested flow."
                assert node_type in node_allow_ref, f"Only nodes of type {node_allow_ref} can be ref node, but got {node_type.name}."

                assert node.ref_fe_path is not None 
                root_flow = mgr._root_flow_getter().flow
                node_id_path = ADVProject.get_node_id_path_from_fe_path(node.ref_fe_path)
                pair = ADVProject.get_flow_node_by_fe_path(root_flow, node.ref_fe_path)
                assert pair is not None 
                node_parent, node_def = pair
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
                    parse_res = flow_parser._node_id_to_parse_result[node_def.id]
                    if isinstance(parse_res, FlowParseResult):
                        is_node_def_folder = parse_res._is_folder()
                    assert flow_parser._flow_parse_result is not None 
                    parent_is_folder = flow_parser._flow_parse_result._is_folder()
                    parse_res = dataclasses.replace(parse_res, node=dataclasses.replace(node))
                    self._node_id_to_parse_result[node.id] = parse_res

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
                    self._node_id_to_parse_result[node.id] = parse_res
                    is_subflow_def = True 
                    is_node_def_folder = parse_res._is_folder()
                else:
                    parse_res = None
            prep_res = NodePrepResult(
                node_def= node_def,
                node=node,
                node_parent=node_parent,
                ext_parse_res=parse_res,
                is_local_ref=is_local_ref,
                is_subflow_def=is_subflow_def,
                is_parent_folder=parent_is_folder,
                # is_node_def_folder=is_node_def_folder,
            )
            res[node_type].append(prep_res)
        return res 

    def _sort_flow_nodes(self, nodes_dict: dict[str, ADVNodeModel]) -> list[ADVNodeModel]:
        child_nodes = list(nodes_dict.values())
        node_sort_tuple: list[tuple[tuple[float, bool, float], ADVNodeModel]] = []
        for n in child_nodes:
            posx = n.position.x
            if n.ref_node_id is None:
                node_sort_tuple.append(((posx, False, posx), n))
            else:
                if n.ref_node_id in nodes_dict:
                    posx_ref = nodes_dict[n.ref_node_id].position.x
                    if posx <= posx_ref:
                        # ref node must be proceed after original node
                        node_sort_tuple.append(((posx_ref, True, posx), n))
                    else:
                        node_sort_tuple.append(((posx, True, posx), n))
                else:
                    node_sort_tuple.append(((posx, True, posx), n))
        # make sure non-referenced nodes are handled first.
        node_sort_tuple.sort(key=lambda t: t[0])
        child_nodes = [t[1] for t in node_sort_tuple]
        return child_nodes

    def _parse_global_scripts(self, node_preps: list[NodePrepResult]):
        global_scripts: dict[str, str] = {}
        for node_prep in node_preps:
            node_def = node_prep.node_def
            assert node_def.nType == ADVNodeType.GLOBAL_SCRIPT
            assert node_def.impl is not None, f"GLOBAL_SCRIPT node {node_def.id} has no code."
            code = node_def.impl.code
            code_lines = code.splitlines()
            assert len(code_lines) > 0
            end_column = len(code_lines[-1]) + 1

            global_scripts[node_def.id] = (node_def.impl.code)
            self._node_id_to_parse_result[node_def.id] = GlobalScriptParseResult(
                node=dataclasses.replace(node_prep.node),
                code=node_def.impl.code,
            )
        global_script = "\n".join(global_scripts.values())
        global_scope = self._global_scope_parser.parse_global_script(global_script)
        # required for all adv meta code
        global_scope.update({
            "dataclasses": dataclasses_plain,
            "ADV": _ADV,
        })
        return global_scripts, global_scope

    def _parse_symbol_groups(self, node_preps: list[NodePrepResult], global_scripts: dict[str, str], global_scope: dict[str, Any]):
        symbols: list[BackendHandle] = []
        sym_group_dep_qnames: list[str] = []
        # use ordered symbol indexes to make sure arguments of fragment are stable
        cnt_base = 0
        for node_prep in node_preps:
            node = node_prep.node 
            n_def = node_prep.node_def
            cached_parse_res = node_prep.ext_parse_res
            if cached_parse_res is not None:
                assert isinstance(cached_parse_res, SymbolParseResult)
                cached_parse_res = cached_parse_res.copy(node.id, cnt_base)
                # use ref node instead of def node
                # code generator need to know whether this is external node
                cached_parse_res = dataclasses.replace(cached_parse_res, node=node)
                sym_group_dep_qnames.extend(cached_parse_res.dep_qnames_for_ext)
                parse_res = cached_parse_res
            else:
                assert n_def.impl is not None, f"symbol node {n_def.id} has no code."
                parser = SymbolParser()
                parse_res = parser.parse_symbol_node(node, n_def.impl.code, global_scope, list(global_scripts.values()))
                parse_res = parse_res.copy(offset=cnt_base, is_sym_handle=True)
                self._node_id_to_parse_result[node.id] = parse_res
            symbols.extend(parse_res.symbols)
            cnt_base += parse_res.num_symbols
        return symbols, sym_group_dep_qnames

    def _parse_fragments(self, node_preps: list[NodePrepResult], root_symbol_scope: dict[str, BackendHandle], global_scope: dict[str, Any]):
        # parse fragments, auto-generated edges will also be handled.
        inlineflow_node_descs: dict[str, list[NodePrepResult]] = {}
        inlineflow_name_to_scope: dict[str, dict[str, BackendHandle]] = {}
        isolated_nodes: list[ADVNodeModel] = []
        for node_prep in node_preps:
            n = node_prep.node
            n_def = node_prep.node_def
            ext_parse_res = node_prep.ext_parse_res
            alias_map = None
            # user can use output mapping to alias output symbols
            # of ref nodes or subflow nodes
            if n.alias_map != "":
                # TODO handle error here.
                alias_map = parse_alias_map(n.alias_map)
            # prepare symbol scope for each subflow
            subf_name = n.inlinesf_name
            if subf_name is not None:
                assert n.name != n.inlinesf_name, f"Node {n.id} name cannot be same as inlineflow name."
                if subf_name not in inlineflow_name_to_scope:
                    inlineflow_name_to_scope[subf_name] = root_symbol_scope.copy()
                symbol_scope = inlineflow_name_to_scope[subf_name]
            else:
                # isolated fragment won't change symbol_scope.
                symbol_scope = root_symbol_scope
            if ext_parse_res is not None:
                assert isinstance(ext_parse_res, FragmentParseResult)
                ext_parse_res = ext_parse_res.copy(n.id)
                parse_res = ext_parse_res
            else:
                if node_prep.is_local_ref:
                    assert n.ref_node_id is not None
                    parse_res = self._node_id_to_parse_result[n.ref_node_id]
                    assert isinstance(parse_res, FragmentParseResult)
                    parse_res = parse_res.copy(n.id)
                    parse_res = dataclasses.replace(parse_res, node=dataclasses.replace(n))
                else:
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
                        symbol_scope
                    )
            if alias_map is not None:
                parse_res = parse_res.do_alias_map(alias_map)
                parse_res = dataclasses.replace(parse_res, alias_map=n.alias_map)
            self._node_id_to_parse_result[n.id] = parse_res

            if subf_name is not None:
                # only main flow node will contribute symbols to symbol scope
                # add outputs to symbol scope
                for handle in parse_res.output_handles:
                    symbol_scope[handle.name] = handle 
            if subf_name is not None:
                if subf_name not in inlineflow_node_descs:
                    inlineflow_node_descs[subf_name] = []
                inlineflow_node_descs[subf_name].append(node_prep)
            else:
                if not node_prep.is_subflow_def and not node_prep.is_ext_node:
                    # subflow nodes are handled separately
                    isolated_nodes.append(n)
        return inlineflow_node_descs, inlineflow_name_to_scope, isolated_nodes

    def _build_flow_connection(self, node_prep_res: dict[ADVNodeType, list[NodePrepResult]]) -> FlowConnInternals:
        inp_node_handle_to_node: dict[tuple[str, str], tuple[ADVNodeModel, BackendHandle]] = {}
        node_id_to_inp_handles: dict[str, list[BackendHandle]] = {}
        out_node_handle_to_node: dict[tuple[str, str], tuple[ADVNodeModel, BackendHandle]] = {}
        node_id_to_out_handles: dict[str, list[BackendHandle]] = {}
        auto_edges: list[ADVEdgeModel] = []
        frag_nodes = node_prep_res[ADVNodeType.FRAGMENT]
        symbol_nodes = node_prep_res[ADVNodeType.SYMBOLS]
        out_indicator_nodes = node_prep_res[ADVNodeType.OUT_INDICATOR]
        for node_prep in (frag_nodes + symbol_nodes + out_indicator_nodes):
            n = node_prep.node
            node_id_to_inp_handles[n.id] = []
            node_id_to_out_handles[n.id] = []
            parse_res = self._node_id_to_parse_result[n.id]
            assert parse_res is not None
            if isinstance(parse_res, FragmentParseResult):
                # ref nodes/nested node only use user edges, no auto edges for inputs
                for handle in parse_res.input_handles:
                    inp_node_handle_to_node[(n.id, handle.id)] = (n, handle)
                    node_id_to_inp_handles[n.id].append(handle)
                    if n.ref_node_id is None and n.flow is None:
                        snid = handle.handle.source_node_id
                        shid = handle.handle.source_handle_id
                        assert snid is not None and shid is not None, f"{n.id}({handle.id}) input handle has no source info for auto edge."
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
                    # print("Out handle for fragment:", n.id, handle.id, handle.handle.symbol_name)
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

    def _parse_out_indicators(self, node_preps: list[NodePrepResult]):
        inlineflow_out_handles: list[tuple[str, BackendHandle]] = []
        for node_prep in node_preps:
            n = node_prep.node
            out_indicator_handle = ADVNodeHandle(
                id=ADVConstHandles.OutIndicator,
                name=n.name,
                flags=int(ADVHandleFlags.IS_INPUT),
                type="",
                symbol_name="",
            )
            backend_handle = BackendHandle(handle=out_indicator_handle, index=0)
            # output indicator won't be ref node
            self._node_id_to_parse_result[n.id] = OutIndicatorParseResult(
                node=dataclasses.replace(n),
                succeed=True,
                handle=backend_handle,
            )
        return inlineflow_out_handles

    def _parse_markdowns(self, node_preps: list[NodePrepResult]):
        for node_prep in node_preps:
            n = node_prep.node
            # output indicator won't be ref node
            self._node_id_to_parse_result[n.id] = MarkdownParseResult(
                node=dataclasses.replace(n),
                succeed=True,
            )
        return

    def _parse_user_edges(self, user_edges: list[ADVEdgeModel], flow_conn: FlowConnInternals, ):
        inlineflow_out_handles: list[tuple[str, BackendHandle]] = []
        valid_edges: list[ADVEdgeModel] = []

        for edge in user_edges:
            assert edge.sourceHandle is not None and edge.targetHandle is not None
            source_key = (edge.source, edge.sourceHandle)
            target_key = (edge.target, edge.targetHandle)
            source_key_valid = source_key in flow_conn.out_node_handle_to_node
            target_key_valid = target_key in flow_conn.inp_node_handle_to_node
            if source_key_valid and target_key_valid:
                _, source_handle = flow_conn.out_node_handle_to_node[source_key]
                target_node, target_handle = flow_conn.inp_node_handle_to_node[target_key]
                target_handle.handle.source_handle_id = source_handle.handle.id
                target_handle.handle.source_node_id = source_handle.handle.source_node_id
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
                ADV_LOGGER.warning(f"Edge from {edge.source}({edge.sourceHandle}) "
                                   f"to {edge.target}({edge.targetHandle}) is invalid "
                                   "and will be removed.")
        return valid_edges, inlineflow_out_handles

    def _parse_inlineflow(self, inlineflow_descs: dict[str, list[NodePrepResult]], root_symbol_scope: dict[str, BackendHandle], 
            inlineflow_scopes: dict[str, dict[str, BackendHandle]],
            inlineflow_out_handles: list[tuple[str, BackendHandle]], flow_conn: FlowConnInternals, nodes_dict: dict[str, ADVNodeModel]):
        sorted_subflow_descs: dict[str, list[NodePrepResult]] = {k: [] for k in inlineflow_descs.keys()}
        subflow_node_ids: set[str] = set()
        out_handles_per_flow: dict[str, list[tuple[str, BackendHandle]]] = {}
        for subf_name, node_descs in inlineflow_descs.items():
            # look for nodes that have no connection on output handles
            nodes = [nd.node for nd in node_descs]
            node_id_to_nd = {nd.node.id: nd for nd in node_descs}
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
            # TODO should we only allow output indicator as inline flow output?
            # if not out_handles_per_flow[subf_name]:
            #     # use non-symbol handles in current scope as output handles
            #     scope = inlineflow_scopes[subf_name]
            #     for symbol_name, handle in scope.items():
            #         if not handle.handle.is_sym_handle:
            #             handle.is_inlineflow_out = True
            #             assert handle.handle.source_node_id is not None 
            #             out_handles_per_flow[subf_name].append((handle.handle.source_node_id, handle))
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
                    # print(flow_id, target_node_id, subf_node_id_to_nodes.keys())
                    if target_node_id in subf_node_id_to_nodes:
                        # input handle connects to outside node
                        handle = handle.copy(prefix=ADVHandlePrefix.Input)
                        handle.handle.flags |= int(ADVHandleFlags.IS_INPUT)
                        handle.handle.source_handle_id = None 
                        handle.handle.source_node_id = None
                        subf_input_handles.append(handle)
                        break
            for source_node_id, source_handle in out_handles_per_flow[flow_name]:
                subf_output_handles.append(source_handle)
            subf_output_handles.sort(key=lambda h: h.index)

            inlineflow_res[flow_name] = InlineFlowParseResult(
                node_descs=subf_descs,
                input_handles=subf_input_handles,
                output_handles=subf_output_handles,
            )
        return inlineflow_res


    def _parse_flow_recursive(self, flow_node: Optional[ADVNodeModel], flow_id: str, flow: ADVFlowModel, mgr: ADVProjectBackendManager, visited: set[str]):
        if self._flow_parse_result is not None:
            return self._flow_parse_result
        if flow_id in visited:
            raise ValueError(f"Cyclic flow reference detected at flow id {flow_id}, {visited}")
        ADV_LOGGER.warning(f"Parsing flow {"__ROOT__" if not flow_id else flow_id} ...")
        child_nodes = self._sort_flow_nodes(flow.nodes)
        has_subflow = False
        for node in child_nodes:
            if node.nType == ADVNodeType.FRAGMENT and node.flow is not None:
                has_subflow = True 
                break
        node_prep_res = self._preprocess_nodes(flow_node, flow_id, child_nodes, has_subflow, mgr, visited)
        # store ref nodes and subflow nodes
        ext_node_descs: list[NodePrepResult] = []
        for node_prep in (node_prep_res[ADVNodeType.GLOBAL_SCRIPT] + node_prep_res[ADVNodeType.SYMBOLS] + node_prep_res[ADVNodeType.FRAGMENT]):
            if node_prep.is_subflow_def or node_prep.is_ext_node:
                ext_node_descs.append(node_prep)
        # 1. parse global script nodes to build global scope
        global_scripts, global_scope = self._parse_global_scripts(node_prep_res[ADVNodeType.GLOBAL_SCRIPT])
        # 2. parse symbol group to build symbol scope
        symbols, sym_group_dep_qnames = self._parse_symbol_groups(node_prep_res[ADVNodeType.SYMBOLS], global_scripts, global_scope)
        root_symbol_scope: dict[str, BackendHandle] = {s.handle.symbol_name: s for s in symbols}
        # 3. parse fragments, auto-generated edges will also be handled.
        inlineflow_node_descs, inlineflow_scopes, isolated_nodes = self._parse_fragments(node_prep_res[ADVNodeType.FRAGMENT], root_symbol_scope, global_scope)
        # now all nodes with output handle are parsed. we can build node-handle map.
        self._parse_out_indicators(node_prep_res[ADVNodeType.OUT_INDICATOR])
        self._parse_markdowns(node_prep_res[ADVNodeType.MARKDOWN])
        edges = list(flow.edges.values())
        edges = list(filter(lambda e: not e.isAutoEdge, edges))
        flow_conn = self._build_flow_connection(node_prep_res)
        valid_edges, inlineflow_out_handles = self._parse_user_edges(edges, flow_conn)
        inlineflow_res = self._parse_inlineflow(inlineflow_node_descs, root_symbol_scope, inlineflow_scopes, inlineflow_out_handles, flow_conn, flow.nodes)
        all_input_handles: list[BackendHandle] = []
        all_output_handles: list[BackendHandle] = []
        is_main_inlineflow_empty = True
        if flow_node is not None and flow_node.name in inlineflow_res:
            all_input_handles = inlineflow_res[flow_node.name].input_handles
            all_output_handles = inlineflow_res[flow_node.name].output_handles
            is_main_inlineflow_empty = len(inlineflow_res[flow_node.name].node_descs) == 0
        misc_nodes = {
            ADVNodeType.GLOBAL_SCRIPT: [p.node for p in node_prep_res[ADVNodeType.GLOBAL_SCRIPT]],
            ADVNodeType.OUT_INDICATOR: [p.node for p in node_prep_res[ADVNodeType.OUT_INDICATOR]],
            ADVNodeType.SYMBOLS: [p.node for p in node_prep_res[ADVNodeType.SYMBOLS]],
            ADVNodeType.MARKDOWN: [p.node for p in node_prep_res[ADVNodeType.MARKDOWN]],
        }
        # import rich 
        # rich.print(flow_id, all_input_handles, all_output_handles)
        flow_parse_res = FlowParseResult(
            node=dataclasses.replace(flow_node) if flow_node is not None else flow_node,
            func_name=flow_node.name if flow_node is not None else "",
            succeed=True,
            input_handles=all_input_handles,
            output_handles=all_output_handles,
            out_type="dict",
            out_type_anno="dict[str, Any]",
            # flow related fields
            symbol_dep_qnames=list(set(sym_group_dep_qnames)),
            edges=valid_edges + flow_conn.auto_edges,
            misc_nodes=misc_nodes,
            isolate_fragments=isolated_nodes,
            inlineflow_results=inlineflow_res,
            ext_node_descs=ext_node_descs,
            has_subflow=has_subflow if flow_node is not None else True,
            alias_map=flow_node.alias_map if flow_node is not None else "",
            is_main_inlineflow_empty=is_main_inlineflow_empty,
        )
        self._flow_parse_result = flow_parse_res
        code_lines = flow_parse_res.to_code_lines(self._node_id_to_parse_result)
        flow_parse_res.generated_code_lines = code_lines.lines
        flow_parse_res.generated_code = "\n".join(code_lines.lines)

        return flow_parse_res

    def _post_order_access_nodes(self, accessor: Callable[[ADVNodeModel], None], node: ADVNodeModel, 
            visited: set[str], node_id_to_inp_handles: dict[str, list[BackendHandle]],
            node_id_to_node: dict[str, ADVNodeModel]) -> None:
        if node.id in visited:
            return
        visited.add(node.id)
        inp_handles = node_id_to_inp_handles.get(node.id, [])
        for handle in inp_handles:
            if handle.handle.source_node_id is not None:
                source_node = node_id_to_node[handle.handle.source_node_id]
                if source_node.nType == ADVNodeType.FRAGMENT:
                    # only traverse fragment nodes
                    self._post_order_access_nodes(accessor, source_node, visited, node_id_to_inp_handles, node_id_to_node)
        accessor(node)

def _main_diff():
    from tensorpc.apps.adv.test_data.simple import get_simple_nested_model
    from tensorpc.core.datamodel.draft import create_draft_type_only
    from deepdiff import DeepDiff

    model = get_simple_nested_model()

    manager = ADVProjectBackendManager(lambda: model, create_draft_type_only(type(model.flow)))
    manager.sync_project_model()
    manager.parse_all()
    manager.init_all_nodes()
    folders, path_to_code = manager._get_all_files_and_folders(is_relative=True)

    def _simple_accessor(path_parts: list[str], fspath: Path) -> str:
        return path_to_code[fspath]
    proj_parser = ADVProjectParser(path_code_accessor=_simple_accessor)

    flow_reparsed = proj_parser._parse_desc_to_flow_model(
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