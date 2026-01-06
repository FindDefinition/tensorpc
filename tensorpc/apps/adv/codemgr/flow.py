import ast
import inspect
from typing import Any, Callable, Literal, Optional, TypeVar
from tensorpc.apps.adv.codemgr.core import BackendHandle, BaseParseResult
from tensorpc.apps.adv.codemgr.fragment import FragmentParseResult, FragmentParser, parse_alias_map
from tensorpc.apps.adv.codemgr.symbols import SymbolParseResult, SymbolParser
import tensorpc.core.dataclass_dispatch as dataclasses

from tensorpc.apps.adv.logger import ADV_LOGGER
from tensorpc.apps.adv.model import ADVConstHandles, ADVEdgeModel, ADVFlowModel, ADVHandlePrefix, ADVNodeModel, ADVNodeHandle, ADVNodeType, ADVProject
import hashlib
from tensorpc.core.annolib import dataclass_flatten_fields_generator, unparse_type_expr
import dataclasses as dataclasses_plain

from tensorpc.utils.uniquename import UniqueNamePool
from tensorpc.apps.adv import api as _ADV
_ROOT_FLOW_ID = ""

ADV_MAIN_FLOW_NAME = "__adv_flow_main__"

@dataclasses.dataclass
class CodeBlock:
    lineno: int 
    column_offset: int  

T = TypeVar("T")

def mark_global_script(node_id: str, position: tuple[float, float], 
        ref_node_id: Optional[str] = None) -> Callable[[T], T]:
    # TODO we actually don't use this metadata, we read ast directly.
    def wrapper(fn_wrapped: T) -> T:
        return fn_wrapped   
    return wrapper

def mark_global_script_end() -> Callable[[T], T]:
    # TODO we actually don't use this metadata, we read ast directly.
    def wrapper(fn_wrapped: T) -> T:
        return fn_wrapped   
    return wrapper

def mark_symbol_dep() -> Callable[[T], T]:
    # TODO we actually don't use this metadata, we read ast directly.
    def wrapper(fn_wrapped: T) -> T:
        return fn_wrapped   
    return wrapper

def mark_symbol_dep_end() -> Callable[[T], T]:
    # TODO we actually don't use this metadata, we read ast directly.
    def wrapper(fn_wrapped: T) -> T:
        return fn_wrapped   
    return wrapper

def mark_ref_node_dep() -> Callable[[T], T]:
    # TODO we actually don't use this metadata, we read ast directly.
    def wrapper(fn_wrapped: T) -> T:
        return fn_wrapped   
    return wrapper

def mark_ref_node_dep_end() -> Callable[[T], T]:
    # TODO we actually don't use this metadata, we read ast directly.
    def wrapper(fn_wrapped: T) -> T:
        return fn_wrapped   
    return wrapper

def mark_ref_node(node_id: str, position: tuple[float, float], 
        ref_node_id: Optional[str] = None) -> Callable[[T], T]:
    # TODO we actually don't use this metadata, we read ast directly.
    def wrapper(fn_wrapped: T) -> T:
        return fn_wrapped   
    return wrapper

def mark_subflow_node(node_id: str, position: tuple[float, float]) -> Callable[[T], T]:
    # TODO we actually don't use this metadata, we read ast directly.
    def wrapper(fn_wrapped: T) -> T:
        return fn_wrapped   
    return wrapper

def mark_inlineflow() -> Callable[[T], T]:
    # TODO we actually don't use this metadata, we read ast directly.
    def wrapper(fn_wrapped: T) -> T:
        return fn_wrapped   
    return wrapper

def mark_out_indicator(node_id: str, position: tuple[float, float], conn_node_id: str, conn_handle_id: str, alias: Optional[str] = None) -> Callable[[T], T]:
    # TODO we actually don't use this metadata, we read ast directly.
    def wrapper(fn_wrapped: T) -> T:
        return fn_wrapped   
    return wrapper


@dataclasses.dataclass
class ExtNodeDesc:
    node: ADVNodeModel
    type: Literal["ref", "subflow"]

@dataclasses.dataclass
class InlineFlowParseResult:
    nodes: list[ADVNodeModel]
    input_handles: list[BackendHandle]
    output_handles: list[BackendHandle]

@dataclasses.dataclass
class NodePrepResult:
    node: ADVNodeModel
    node_def: ADVNodeModel
    ext_parse_res: Optional[BaseParseResult] = None
    is_local_ref: bool = False
    is_subflow: bool = False

    @property 
    def is_ext_node(self):
        return self.ext_parse_res is not None

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
        kwarg_str = ", ".join(kwargs_parts)
        decorator = f"ADV.{mark_out_indicator.__name__}({kwarg_str})"
        return [
            f"{decorator}",
        ]

@dataclasses.dataclass
class FlowConnInternals:
    inp_node_handle_to_node: dict[tuple[str, str], tuple[ADVNodeModel, BackendHandle]]
    node_id_to_inp_handles: dict[str, list[BackendHandle]]
    out_node_handle_to_node: dict[tuple[str, str], tuple[ADVNodeModel, BackendHandle]]
    node_id_to_out_handles: dict[str, list[BackendHandle]]
    auto_edges: list[ADVEdgeModel]

@dataclasses.dataclass(kw_only=True)
class FlowParseResult(FragmentParseResult):
    edges: list[ADVEdgeModel]
    symbol_dep_qnames: list[str] = dataclasses.field(default_factory=list)
    isolate_fragments: list[ADVNodeModel] = dataclasses.field(default_factory=list)
    inlineflow_results: dict[str, InlineFlowParseResult] = dataclasses.field(default_factory=dict)
    misc_nodes: dict[ADVNodeType, list[ADVNodeModel]] = dataclasses.field(default_factory=dict)
    ext_node_descs: list[ExtNodeDesc] = dataclasses.field(default_factory=list)
    
    def to_code_lines(self, id_to_parse_res: dict[str, "BaseParseResult"]):
        lines = [
            "from tensorpc.apps.adv import ADV",
            "import dataclasses",
        ]
        # if self.node is not None:
            # mark parent node meta
        # global scripts
        gs_lines_all: list[str] = []
        for node in self.misc_nodes[ADVNodeType.GLOBAL_SCRIPT]:
            kwarg_str = ", ".join(self.get_node_meta_kwargs(node))
            mark_stmt = f"ADV.{mark_global_script.__name__}({kwarg_str})"
            gs_lines = [
                mark_stmt
            ]
            if node.ref_node_id is None:
                impl = node.impl
                assert impl is not None 
                gs_lines.append(impl.code)
            gs_lines_all.extend(gs_lines)
        if gs_lines_all:
            gs_lines_all.insert(0, "# ------ ADV Global Script Region ------")
            gs_lines_all.append(f"ADV.{mark_global_script_end.__name__}()")
        lines.extend(gs_lines_all)
        # symbol deps
        symbol_dep_lines: list[str] = []
        if self.symbol_dep_qnames:
            symbol_dep_lines.append("# ------ ADV Symbol Dependency Region ------")
            symbol_dep_lines.append(f"ADV.{mark_symbol_dep.__name__}()")
            for qname in self.symbol_dep_qnames:
                parts = qname.split(".")
                import_line = f"from {'.'.join(parts[:-1])} import {parts[-1]}"
                symbol_dep_lines.append(import_line)
            symbol_dep_lines.append(f"ADV.{mark_symbol_dep_end.__name__}()")
        lines.extend(symbol_dep_lines)
        # ref deps
        ref_dep_lines: list[str] = []
        if self.ext_node_descs:
            
            for node_desc in self.ext_node_descs:
                node = node_desc.node
                if node_desc.type == "ref":
                    ref_import_path = node.ref_import_path
                    assert ref_import_path is not None 
                    ref_dep_lines.append(f"from {'.'.join(ref_import_path)} import {node.name}")
                else:
                    assert node.flow is not None, "only ref node or subflow node can be external node"
                    ref_dep_lines.append(f"from .{node.name} import {node.name}")
            for node_desc in self.ext_node_descs:
                node = node_desc.node
                # add marker to mark node id and position
                kwargs_str = ", ".join(self.get_node_meta_kwargs(node))
                if node_desc.type == "subflow":
                    mark_stmt = f"ADV.{mark_subflow_node.__name__}({kwargs_str})({node.name})"
                else:
                    mark_stmt = f"ADV.{mark_ref_node.__name__}({kwargs_str})({node.name})"
                ref_dep_lines.append(mark_stmt)
        
        if ref_dep_lines:
            lines.append("# ------ ADV Ref/Subflow Nodes Dependency Region (Optional) ------")
            lines.append(f"ADV.{mark_ref_node_dep.__name__}()")
            lines.extend(ref_dep_lines)
            lines.append(f"ADV.{mark_ref_node_dep_end.__name__}()")
        # symbol groups
        symbol_group_lines: list[str] = []
        for node in self.misc_nodes[ADVNodeType.SYMBOLS]:
            parse_res = id_to_parse_res.get(node.id, None)
            assert parse_res is not None and isinstance(parse_res, SymbolParseResult)
            symbol_group_lines.extend(parse_res.to_code_lines(id_to_parse_res))
        if symbol_group_lines:
            lines.append("# ------ ADV Symbol Def Region ------")
        lines.extend(symbol_group_lines)
        out_indicator_lines: list[str] = []
        for node in self.misc_nodes[ADVNodeType.OUT_INDICATOR]:
            parse_res = id_to_parse_res.get(node.id, None)
            assert parse_res is not None and isinstance(parse_res, OutIndicatorParseResult)
            out_indicator_lines.extend(parse_res.to_code_lines(id_to_parse_res))
        if out_indicator_lines:
            lines.append("# ------ ADV Out Indicator Region ------")
            lines.extend(out_indicator_lines)

        # isolate fragments
        isolate_fragment_lines: list[str] = []
        for node in self.isolate_fragments:
            parse_res = id_to_parse_res.get(node.id, None)
            assert parse_res is not None and isinstance(parse_res, FragmentParseResult)
            isolate_fragment_lines.extend(parse_res.to_code_lines(id_to_parse_res))
        lines.extend(isolate_fragment_lines)
        # subflow nodes
        inlineflow_lines: list[str] = []
        for inline_name, inline_desc in self.inlineflow_results.items():
            for node in inline_desc.nodes:
                if node.ref_node_id is not None:
                    continue 
                parse_res = id_to_parse_res.get(node.id, None)
                assert parse_res is not None and isinstance(parse_res, FragmentParseResult)
                inlineflow_lines.extend(parse_res.to_code_lines(id_to_parse_res))
            # build subflow
            func_name = inline_name
            inlineflow_fn_lines = [
                f"@ADV.{mark_inlineflow.__name__}()",
                f"def {func_name}(",
            ]
            inlineflow_fn_lines += FragmentParseResult.get_signature_lines_from_handles(inline_desc.input_handles)
            inlineflow_fn_lines.append(f") -> dict[str, Any]:")
            body_lines: list[str] = []
            for node in inline_desc.nodes:
                parse_res = id_to_parse_res.get(node.id, None)
                assert parse_res is not None and isinstance(parse_res, FragmentParseResult)
                arg_name_parts: list[str] = []
                for h in parse_res.input_handles:
                    if h.handle.source_node_id is None:
                        arg_name_parts.append("ADV.MISSING")
                    else:
                        arg_name_parts.append(h.handle.symbol_name)
                arg_names = ", ".join(arg_name_parts)
                anno = ""
                if node.ref_node_id is not None:
                    # use Annotated to attach ref node meta
                    arg_parts = [f"\"{node.id}\"", f"({node.position.x}, {node.position.y})"]
                    if node.alias_map is not None:
                        arg_parts.append(f"\"{node.alias_map}\"")
                    arg_str = ", ".join(arg_parts)
                    anno = f": Annotated[Any, ADV.RefNodeMeta({arg_str})]" 
                if parse_res.out_type == "single":
                    symbol_name = parse_res.output_handles[0].symbol_name
                    body_lines.append(f"{symbol_name}{anno} = {parse_res.func_name}({arg_names})")
                elif parse_res.out_type == "tuple":
                    out_names = ", ".join([h.handle.symbol_name for h in parse_res.output_handles])
                    body_lines.append(f"{out_names}{anno} = {parse_res.func_name}({arg_names})")
                else:
                    # dict
                    body_lines.append(f"_adv_tmp_out{anno} = {parse_res.func_name}({arg_names})")
                    for h in parse_res.output_handles:
                        assert h.handle.dict_key is not None 
                        body_lines.append(f"{h.handle.symbol_name} = _adv_tmp_out['{h.handle.dict_key}']")
            # subflow always return dict
            body_lines.append("return {")
            for h in inline_desc.output_handles:
                body_lines.append(f"    '{h.handle.symbol_name}': {h.handle.symbol_name},")
            body_lines.append("}")
            body_lines_indented = [f"    {line}" for line in body_lines]
            inlineflow_fn_lines.extend(body_lines_indented)
            inlineflow_lines.extend(inlineflow_fn_lines)
        lines.extend(inlineflow_lines)
        return lines

class ADVProjectBackendManager:
    def __init__(self, root_flow_getter: Callable[[], ADVFlowModel]):
        self._root_flow_getter = root_flow_getter
        
        self._node_id_to_node: dict[str, ADVNodeModel] = {
        }
        self._node_id_to_flow: dict[str, tuple[ADVFlowModel, Optional[ADVNodeModel]]] = {
            _ROOT_FLOW_ID: (root_flow_getter(), None)
        }

        self._flow_node_id_to_parser: dict[str, FlowParser] = {
            _ROOT_FLOW_ID: FlowParser()
        }

    def sync_project_model(self):
        self._node_id_to_node.clear()
        self._node_id_to_flow.clear()
        root_flow = self._root_flow_getter()
        self._node_id_to_flow[_ROOT_FLOW_ID] = (root_flow, None)
        flow_node_id_to_parser: dict[str, FlowParser] = {
            _ROOT_FLOW_ID: self._flow_node_id_to_parser[_ROOT_FLOW_ID]
        }
        def _traverse_node(node: ADVNodeModel):
            self._node_id_to_node[node.id] = node
            if node.flow is not None:
                self._node_id_to_flow[node.id] = (node.flow, node)
                if node.id not in self._flow_node_id_to_parser:
                    flow_node_id_to_parser[node.id] = FlowParser()
                else:
                    flow_node_id_to_parser[node.id] = self._flow_node_id_to_parser[node.id]
                for child_node in node.flow.nodes.values():
                    _traverse_node(child_node)
        
        for node in root_flow.nodes.values():
            _traverse_node(node)

        self._flow_node_id_to_parser = flow_node_id_to_parser

    def parse_all(self):
        # must be called after sync_project_model
        visited: set[str] = set()
        for flow_id, (flow, flow_parent_node) in self._node_id_to_flow.items():
            flow_parser = self._flow_node_id_to_parser[flow_id]
            flow_parser._parse_flow_recursive(flow_parent_node, flow_id, flow, self, visited)

    def init_all_nodes(self):
        # init all names and node handles in model
        for flow_id, (flow, _) in self._node_id_to_flow.items():
            parser = self._flow_node_id_to_parser[flow_id]
            assert parser._flow_parse_result is not None
            for node_id, node in flow.nodes.items():
                parse_res = parser._node_id_to_parse_result.get(node_id, None)
                if parse_res is not None:
                    if isinstance(parse_res, SymbolParseResult):
                        node.name = parse_res.symbol_cls_name
                        node.handles = [bh.handle for bh in parse_res.symbols]
                    elif isinstance(parse_res, FragmentParseResult):
                        node.handles = [bh.handle for bh in parse_res.input_handles + parse_res.output_handles]
                    elif isinstance(parse_res, OutIndicatorParseResult):
                        node.handles = [parse_res.handle.handle]
                else:
                    if node.nType != ADVNodeType.OUT_INDICATOR:
                        # TODO 
                        ADV_LOGGER.warning(f"Node {node_id} in flow {flow_id} has no parse result.")
                    else:
                        node.handles = [
                            ADVNodeHandle(
                                id=f"{ADVHandlePrefix.OutIndicator}-outputs",
                                name="inputs",
                                is_input=True,
                                type="",
                            )
                        ]
            flow.edges = {e.id: e for e in parser._flow_parse_result.edges}

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

        self._flow_parse_result: Optional[FlowParseResult] = None

        self._edge_uid_pool = UniqueNamePool()

    def clear_parse_cache(self):
        self._node_id_to_parse_result.clear()
        self._flow_parse_result = None
        self._edge_uid_pool.clear()

    def serialize_to_code(self):
        assert self._flow_parse_result is not None
        return self._flow_parse_result.to_code_lines(self._node_id_to_parse_result)

    def _get_or_compile_real_node(self, flow_id: str, node: ADVNodeModel, mgr: ADVProjectBackendManager, visited: set[str]):
        if node.ref_node_id is not None:
            assert node.ref_fe_path is not None 
            node_id_path = ADVProject.get_node_id_path_from_fe_path(node.ref_fe_path)
            node = mgr._node_id_to_node[node.ref_node_id]
            node_parent = None
            if len(node_id_path) > 1:
                node_parent = mgr._node_id_to_node[node_id_path[-2]]
                parent_flow = node_parent.flow
                assert parent_flow is not None
                flow_parser = mgr._flow_node_id_to_parser[node_parent.id]
                flow_node_id = node_parent.id
            else:
                flow_parser = mgr._flow_node_id_to_parser[_ROOT_FLOW_ID]
                parent_flow = mgr._node_id_to_flow[_ROOT_FLOW_ID][0]
                flow_node_id = _ROOT_FLOW_ID
            if flow_id == flow_node_id:
                # local ref node. handled later.
                return node, None, True
            if flow_parser._flow_parse_result is None:
                flow_parser._parse_flow_recursive(node_parent, flow_node_id, parent_flow, mgr, visited)
            parse_res = flow_parser._node_id_to_parse_result[node.id]
            return node, parse_res, False
        return node, None, False

    def _preprocess_nodes(self, flow_id: str, nodes: list[ADVNodeModel], mgr: ADVProjectBackendManager, visited: set[str]) -> dict[ADVNodeType, list[NodePrepResult]]:
        res: dict[ADVNodeType, list[NodePrepResult]] = {
            ADVNodeType.GLOBAL_SCRIPT: [],
            ADVNodeType.SYMBOLS: [],
            ADVNodeType.FRAGMENT: [],
            ADVNodeType.OUT_INDICATOR: [],
        }
        node_allow_ref = set([ADVNodeType.GLOBAL_SCRIPT, ADVNodeType.SYMBOLS, ADVNodeType.FRAGMENT])
        for node in nodes:
            node_type = ADVNodeType(node.nType)

            is_subflow = False
            if node.ref_node_id is not None:
                assert node_type in node_allow_ref, f"Only nodes of type {node_allow_ref} can be ref node, but got {node_type.name}."
                node_def, parse_res, is_local_ref = self._get_or_compile_real_node(flow_id, node, mgr, visited)
                assert node.nType == node_def.nType, f"Ref node type {node.nType} does not match original node type {node_def.nType}."
                is_subflow = node_def.flow is not None
            else:
                node_def = node
                is_local_ref = False
                if node.flow is not None:
                    flow_parser = mgr._flow_node_id_to_parser[flow_id]
                    parse_res = flow_parser._parse_flow_recursive(node, flow_id, node.flow, mgr, visited)
                    self._node_id_to_parse_result[node.id] = parse_res
                    is_subflow = True 
                else:
                    parse_res = None
            prep_res = NodePrepResult(
                node_def= node_def,
                node=node,
                ext_parse_res=parse_res,
                is_local_ref=is_local_ref,
                is_subflow=is_subflow,
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
            global_scripts[node_def.id] = (node_def.impl.code)
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
        inlineflow_nodes: dict[str, list[ADVNodeModel]] = {}
        inlineflow_name_to_scope: dict[str, dict[str, BackendHandle]] = {}
        isolated_nodes: list[ADVNodeModel] = []
        for node_prep in node_preps:
            n = node_prep.node
            n_def = node_prep.node_def
            ext_parse_res = node_prep.ext_parse_res
            alias_map = None
            # user can use output mapping to alias output symbols
            # of ref nodes or subflow nodes
            if n.alias_map is not None:
                # TODO handle error here.
                alias_map = parse_alias_map(n.alias_map)
            # prepare symbol scope for each subflow
            subf_name = n.inline_subflow_name
            if subf_name is not None:
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
                else:
                    assert n_def.impl is not None, f"fragment node {n_def.id} has no code."
                    parser = FragmentParser()
                    parse_res = parser.parse_fragment(
                        n,
                        n_def.impl.code,
                        global_scope,
                        symbol_scope
                    )
            if alias_map is not None:
                parse_res = parse_res.do_alias_map(alias_map)
            self._node_id_to_parse_result[n.id] = parse_res

            # TODO rename handle if alias map provided
            if subf_name is not None:
                # only main flow node will contribute symbols to symbol scope
                # add outputs to symbol scope
                for handle in parse_res.output_handles:
                    symbol_scope[handle.symbol_name] = handle 
            if subf_name is not None:
                if subf_name not in inlineflow_nodes:
                    inlineflow_nodes[subf_name] = []
                inlineflow_nodes[subf_name].append(n)
            else:
                if not node_prep.is_subflow:
                    # subflow nodes are handled separately
                    isolated_nodes.append(n)
        return inlineflow_nodes, inlineflow_name_to_scope, isolated_nodes

    def _build_flow_connection(self, node_prep_res: dict[ADVNodeType, list[NodePrepResult]], user_edges: list[ADVEdgeModel]) -> FlowConnInternals:
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
                name="inputs",
                is_input=True,
                type="",
                symbol_name="",
            )
            backend_handle = BackendHandle(handle=out_indicator_handle, index=0)
            # output indicator won't be ref node
            self._node_id_to_parse_result[n.id] = OutIndicatorParseResult(
                node=n,
                succeed=True,
                handle=backend_handle
            )
        return inlineflow_out_handles

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
                    inlineflow_out_handles.append((edge.source, source_handle))
                valid_edges.append(edge)
            else:
                ADV_LOGGER.warning(f"Edge from node {edge.source} handle {edge.sourceHandle} "
                                   f"to node {edge.target} handle {edge.targetHandle} is invalid "
                                   "and will be ignored.")
        return valid_edges, inlineflow_out_handles

    def _parse_inlineflow(self, inlineflow_nodes: dict[str, list[ADVNodeModel]], root_symbol_scope: dict[str, BackendHandle], 
            inlineflow_scopes: dict[str, dict[str, BackendHandle]],
            inlineflow_out_handles: list[tuple[str, BackendHandle]], flow_conn: FlowConnInternals, nodes_dict: dict[str, ADVNodeModel]):
        sorted_subflow_nodes: dict[str, list[ADVNodeModel]] = {k: [] for k in inlineflow_nodes.keys()}
        subflow_node_ids: set[str] = set()
        out_handles_per_flow: dict[str, list[tuple[str, BackendHandle]]] = {}
        for subf_name, nodes in inlineflow_nodes.items():
            # look for nodes that have no connection on output handles
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
            if not out_handles_per_flow[subf_name]:
                # use non-symbol handles in current scope as output handles
                scope = inlineflow_scopes[subf_name]
                for symbol_name, handle in scope.items():
                    if not handle.handle.is_sym_handle:
                        handle.is_inlineflow_out = True
                        assert handle.handle.source_node_id is not None 
                        out_handles_per_flow[subf_name].append((handle.handle.source_node_id, handle))
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
            # print(subf_name, [n.id for n in postorder_nodes], [n.id for n in nodes], [n.id for n in root_nodes])

            sorted_subflow_nodes[subf_name] = postorder_nodes
            for n in postorder_nodes:
                subflow_node_ids.add(n.id)
        # create input/output handles of flow if exists
        inlineflow_res: dict[str, InlineFlowParseResult] = {}

        for flow_name, subf_nodes in sorted_subflow_nodes.items():
            subf_input_handles: list[BackendHandle] = []
            subf_output_handles: list[BackendHandle] = []
            subf_node_id_to_nodes = {n.id: n for n in subf_nodes}
            for handle in root_symbol_scope.values():
                for target_node_id, _ in handle.target_node_handle_id:
                    # print(flow_id, target_node_id, subf_node_id_to_nodes.keys())
                    if target_node_id in subf_node_id_to_nodes:
                        # input handle connects to outside node
                        handle = handle.copy(prefix=ADVHandlePrefix.Input)
                        handle.handle.is_input = True
                        handle.handle.source_handle_id = None 
                        handle.handle.source_node_id = None
                        subf_input_handles.append(handle)
                        break
            for source_node_id, source_handle in out_handles_per_flow[flow_name]:
                subf_output_handles.append(source_handle)
            subf_output_handles.sort(key=lambda h: h.index)
            # print(flow_id, flow_name, [h.id for h in subf_input_handles], [h.id for h in subf_output_handles])

            inlineflow_res[flow_name] = InlineFlowParseResult(
                nodes=subf_nodes,
                input_handles=subf_input_handles,
                output_handles=subf_output_handles,
            )
        return inlineflow_res


    def _parse_flow_recursive(self, flow_node: Optional[ADVNodeModel], flow_id: str, flow: ADVFlowModel, mgr: ADVProjectBackendManager, visited: set[str]):
        if self._flow_parse_result is not None:
            return self._flow_parse_result
        if flow_id in visited:
            raise ValueError(f"Cyclic flow reference detected at flow id {flow_id}, {visited}")
        child_nodes = self._sort_flow_nodes(flow.nodes)
        node_prep_res = self._preprocess_nodes(flow_id, child_nodes, mgr, visited)
        # store ref nodes and subflow nodes
        ext_node_descs: list[ExtNodeDesc] = []
        for node_prep in (node_prep_res[ADVNodeType.GLOBAL_SCRIPT] + node_prep_res[ADVNodeType.SYMBOLS] + node_prep_res[ADVNodeType.FRAGMENT]):
            if node_prep.is_subflow or node_prep.is_ext_node:
                ext_node_descs.append(ExtNodeDesc(
                    node=node_prep.node,
                    type="subflow" if node_prep.is_subflow else "ref",
                ))
        # 1. parse global script nodes to build global scope
        global_scripts, global_scope = self._parse_global_scripts(node_prep_res[ADVNodeType.GLOBAL_SCRIPT])
        # 2. parse symbol group to build symbol scope
        symbols, sym_group_dep_qnames = self._parse_symbol_groups(node_prep_res[ADVNodeType.SYMBOLS], global_scripts, global_scope)
        root_symbol_scope: dict[str, BackendHandle] = {s.handle.symbol_name: s for s in symbols}
        # 3. parse fragments, auto-generated edges will also be handled.
        inlineflow_nodes, inlineflow_scopes, isolated_nodes = self._parse_fragments(node_prep_res[ADVNodeType.FRAGMENT], root_symbol_scope, global_scope)
        # now all nodes with output handle are parsed. we can build node-handle map.
        self._parse_out_indicators(node_prep_res[ADVNodeType.OUT_INDICATOR])
        edges = list(flow.edges.values())
        edges = list(filter(lambda e: not e.isAutoEdge, edges))

        flow_conn = self._build_flow_connection(node_prep_res, edges)
        valid_edges, inlineflow_out_handles = self._parse_user_edges(edges, flow_conn)
        # 4. parse output indicator node
        # inlineflow_out_handles = self._parse_out_indicators(node_prep_res[ADVNodeType.OUT_INDICATOR], edges, flow_conn)
        # 5. filter invalid user edges, fill target_node_handle_id from user edges
        # 6. parse inline flows. apply post-order sort to each inline flow.
        inlineflow_res = self._parse_inlineflow(inlineflow_nodes, root_symbol_scope, inlineflow_scopes, inlineflow_out_handles, flow_conn, flow.nodes)
        all_input_handles: list[BackendHandle] = []
        all_output_handles: list[BackendHandle] = []

        if ADV_MAIN_FLOW_NAME in inlineflow_res:
            all_input_handles = inlineflow_res[ADV_MAIN_FLOW_NAME].input_handles
            all_output_handles = inlineflow_res[ADV_MAIN_FLOW_NAME].output_handles
        misc_nodes = {
            ADVNodeType.GLOBAL_SCRIPT: [p.node for p in node_prep_res[ADVNodeType.GLOBAL_SCRIPT]],
            ADVNodeType.OUT_INDICATOR: [p.node for p in node_prep_res[ADVNodeType.OUT_INDICATOR]],
            ADVNodeType.SYMBOLS: [p.node for p in node_prep_res[ADVNodeType.SYMBOLS]],
        }
        # import rich 
        # rich.print(flow_id, all_input_handles, all_output_handles)
        flow_parse_res = FlowParseResult(
            node=flow_node,
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
        )
        self._flow_parse_result = flow_parse_res
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

