import ast
import inspect
from typing import Any
from tensorpc.apps.adv.codemgr.core import BackendHandle, BaseParseResult
from tensorpc.apps.adv.codemgr.fragment import FragmentParseResult, FragmentParser
from tensorpc.apps.adv.codemgr.symbols import SymbolParseResult, SymbolParser
import tensorpc.core.dataclass_dispatch as dataclasses

from tensorpc.apps.adv.logger import ADV_LOGGER
from tensorpc.apps.adv.model import ADVEdgeModel, ADVFlowModel, ADVNodeModel, ADVNodeHandle, ADVNodeType, ADVProject
import hashlib
from tensorpc.core.annolib import dataclass_flatten_fields_generator, unparse_type_expr
import dataclasses as dataclasses_plain

from tensorpc.utils.uniquename import UniqueNamePool

_ROOT_FLOW_ID = ""

@dataclasses.dataclass
class CodeBlock:
    lineno: int 
    column_offset: int  

@dataclasses.dataclass(kw_only=True)
class FlowParseResult(FragmentParseResult):
    edges: list[ADVEdgeModel]


class ADVProjectBackendManager:
    def __init__(self, root_flow: ADVFlowModel):
        self._node_id_to_node: dict[str, ADVNodeModel] = {
        }
        self._node_id_to_flow: dict[str, ADVFlowModel] = {
            _ROOT_FLOW_ID: root_flow
        }

        self._flow_node_id_to_parser: dict[str, FlowParser] = {
            _ROOT_FLOW_ID: FlowParser()
        }

class GlobalScriptParser:
    def __init__(self) -> None:
        self._code_to_global_scope: dict[str, dict[str, Any]] = {}

    def parse_global_script(self, code: str) -> dict[str, Any]:
        if code in self._code_to_global_scope:
            return self._code_to_global_scope[code]
        global_scope: dict[str, Any] = {}
        compiled = compile(code, "<string>", "exec")
        exec(compiled, global_scope, global_scope)
        self._code_to_global_scope[code] = global_scope
        return global_scope

# def _post_order_access_nodes()

class FlowParser:
    def __init__(self):
        self._global_scope_parser = GlobalScriptParser()
        self._code_to_global_scope: dict[str, dict[str, Any]] = {}

        self._node_id_to_parse_result: dict[str, BaseParseResult] = {}

        self._need_parse: bool = True

        self._edge_uid_pool = UniqueNamePool()

    def clear_parse_cache(self):
        self._node_id_to_parse_result.clear()
        self._need_parse = True
        self._edge_uid_pool.clear()

    def _get_or_compile_real_node(self, flow_id: str, node: ADVNodeModel, mgr: ADVProjectBackendManager, visited: set[str]):
        if node.ref_node_id is not None:
            assert node.ref_fe_path is not None 
            node_id_path = ADVProject.get_node_id_path_from_fe_path(node.ref_fe_path)
            node = mgr._node_id_to_node[node.ref_node_id]
            if len(node_id_path) > 1:
                node_parent = mgr._node_id_to_node[node_id_path[-2]]
                parent_flow = node_parent.flow
                assert parent_flow is not None
                flow_parser = mgr._flow_node_id_to_parser[node_parent.id]
                flow_node_id = node_parent.id
            else:
                flow_parser = mgr._flow_node_id_to_parser[_ROOT_FLOW_ID]
                parent_flow = mgr._node_id_to_flow[_ROOT_FLOW_ID]
                flow_node_id = _ROOT_FLOW_ID
            if flow_id == flow_node_id:
                # ref to node in same flow, no need to parse again
                # here we sort to make sure ref nodes are proceed after original nodes
                parse_res = flow_parser._node_id_to_parse_result[node.id]
                return node, parse_res, False
            if flow_parser._need_parse:
                flow_parser._parse_flow_recursive(flow_node_id, parent_flow, mgr, visited)
            parse_res = flow_parser._node_id_to_parse_result[node.id]
            return node, parse_res, True
        return node, None, False

    def _parse_flow_recursive(self, flow_id: str, flow: ADVFlowModel, mgr: ADVProjectBackendManager, visited: set[str]):
        if flow_id in visited:
            raise ValueError(f"Cyclic flow reference detected at flow id {flow_id}, {visited}")
        child_nodes = list(flow.nodes.values())
        # sort nodes by position.x. if we create ref that refers to a node in same graph, original
        # node must be handled first.
        node_sort_tuple: list[tuple[tuple[float, bool, float], ADVNodeModel]] = []
        for n in child_nodes:
            posx = n.position.x
            if n.ref_node_id is None:
                node_sort_tuple.append(((posx, False, posx), n))
            else:
                if n.ref_node_id in flow.nodes:
                    posx_ref = flow.nodes[n.ref_node_id].position.x
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
        # 1. parse global script nodes to build global scope
        global_scripts: dict[str, str] = {}
        for n in child_nodes:
            if n.nType == ADVNodeType.GLOBAL_SCRIPT:
                n, _, _ = self._get_or_compile_real_node(flow_id, n, mgr, visited)
                assert n.nType == ADVNodeType.GLOBAL_SCRIPT
                assert n.impl is not None, f"GLOBAL_SCRIPT node {n.id} has no code."
                global_scripts[n.id] = (n.impl.code)
        global_script = "\n".join(global_scripts.values())
        global_scope = self._global_scope_parser.parse_global_script(global_script)
        # 2. parse symbol group to build symbol scope
        symbols: list[BackendHandle] = []
        sym_group_dep_qnames: list[str] = []
        # use ordered symbol indexes to make sure arguments of fragment are stable
        cnt_base = 0
        for n in child_nodes:
            if n.nType == ADVNodeType.SYMBOLS:
                n_def, cached_parse_res, is_external_node = self._get_or_compile_real_node(flow_id, n, mgr, visited)
                if cached_parse_res is not None:
                    assert isinstance(cached_parse_res, SymbolParseResult)
                    cached_parse_res = cached_parse_res.copy(n.id, cnt_base)
                    if is_external_node:
                        sym_group_dep_qnames.extend(cached_parse_res.dep_qnames_for_ext)
                    else:
                        sym_group_dep_qnames.extend(cached_parse_res.dep_qnames)
                    parse_res = cached_parse_res
                else:
                    assert n_def.impl is not None, f"symbol node {n_def.id} has no code."
                    parser = SymbolParser()
                    parse_res = parser.parse_symbol_node(n.id, n_def.impl.code, global_scope, list(global_scripts.values()))
                    parse_res = parse_res.copy(offset=cnt_base)
                    self._node_id_to_parse_result[n.id] = parse_res
                symbols.extend(parse_res.symbols)
                cnt_base += parse_res.num_symbols
        
        root_symbol_scope: dict[str, BackendHandle] = {s.handle.symbol_name: s for s in symbols}
        # 3. parse fragments, auto-generated edges will also be handled.
        subflow_nodes: dict[str, list[ADVNodeModel]] = {}
        subflow_name_to_scope: dict[str, dict[str, BackendHandle]] = {}
        for n in child_nodes:
            if n.nType == ADVNodeType.FRAGMENT:
                subf_name = n.inline_subflow_name
                if subf_name is not None:
                    if subf_name not in subflow_name_to_scope:
                        subflow_name_to_scope[subf_name] = root_symbol_scope.copy()
                    symbol_scope = subflow_name_to_scope[subf_name]
                else:
                    # fragment parser won't change symbol_scope.
                    symbol_scope = root_symbol_scope
                n_def, cached_parse_res, is_external_node = self._get_or_compile_real_node(flow_id, n, mgr, visited)
                if cached_parse_res is not None:
                    assert isinstance(cached_parse_res, FragmentParseResult)
                    cached_parse_res = cached_parse_res.copy(n.id)

                    parse_res = cached_parse_res
                else:
                    assert n_def.impl is not None, f"fragment node {n_def.id} has no code."
                    parser = FragmentParser()
                    parse_res = parser.parse_fragment(
                        n.id,
                        n_def.impl.code,
                        global_scope,
                        symbol_scope
                    )
                    self._node_id_to_parse_result[n.id] = parse_res
                if subf_name is not None and n.ref_node_id is None:
                    # only non-ref main flow node will contribute symbols to symbol scope
                    # add outputs to symbol scope
                    for handle in parse_res.output_handles:
                        symbol_scope[handle.symbol_name] = handle 
                if subf_name is not None:
                    if subf_name not in subflow_nodes:
                        subflow_nodes[subf_name] = []
                    subflow_nodes[subf_name].append(n)
        # now all nodes with output handle are parsed. we can build node-handle map.

        
        # 4. parse output indicator node
        edges = list(flow.edges.values())
        edges = list(filter(lambda e: not e.isAutoEdge, edges))
        target_node_id_to_edges: dict[str, list[ADVEdgeModel]] = {}
        for e in edges:
            if e.target not in target_node_id_to_edges:
                target_node_id_to_edges[e.target] = []
            target_node_id_to_edges[e.target].append(e)
        for n in child_nodes:
            if n.nType == ADVNodeType.OUT_INDICATOR:
                # output indicator won't be ref node
                if n.id in target_node_id_to_edges:
                    out_edges = target_node_id_to_edges[n.id]
                
        # 5. parse sub flows. 
        target_handle_to_edge = {(e.target, e.targetHandle): e for e in edges}

        # 5.1 deal with user edges
        for subf_name, nodes in subflow_nodes.items():
            for node in nodes:
                pass 
            pass 
        self._need_parse = False