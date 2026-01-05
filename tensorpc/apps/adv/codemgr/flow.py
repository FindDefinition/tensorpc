import ast
import inspect
from typing import Any, Callable, Optional
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

@dataclasses.dataclass(kw_only=True)
class FlowParseResult(FragmentParseResult):
    edges: list[ADVEdgeModel]
    symbol_dep_qnames: list[str] = dataclasses.field(default_factory=list)
    symbol_nodes: list[ADVNodeModel] = dataclasses.field(default_factory=list)
    isolate_fragments: list[ADVNodeModel] = dataclasses.field(default_factory=list)
    subflow_nodes: dict[str, list[ADVNodeModel]] = dataclasses.field(default_factory=dict)


class ADVProjectBackendManager:
    def __init__(self, root_flow_getter: Callable[[], ADVFlowModel]):
        self._root_flow_getter = root_flow_getter
        
        self._node_id_to_node: dict[str, ADVNodeModel] = {
        }
        self._node_id_to_flow: dict[str, ADVFlowModel] = {
            _ROOT_FLOW_ID: root_flow_getter()
        }

        self._flow_node_id_to_parser: dict[str, FlowParser] = {
            _ROOT_FLOW_ID: FlowParser()
        }

    def sync_project_model(self):
        self._node_id_to_node.clear()
        self._node_id_to_flow.clear()
        root_flow = self._root_flow_getter()
        self._node_id_to_flow[_ROOT_FLOW_ID] = root_flow
        flow_node_id_to_parser: dict[str, FlowParser] = {
            _ROOT_FLOW_ID: self._flow_node_id_to_parser[_ROOT_FLOW_ID]
        }
        def _traverse_node(node: ADVNodeModel):
            self._node_id_to_node[node.id] = node
            if node.flow is not None:
                self._node_id_to_flow[node.id] = node.flow
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
        for flow_id, flow in self._node_id_to_flow.items():
            flow_parser = self._flow_node_id_to_parser[flow_id]
            flow_parser._parse_flow_recursive(flow_id, flow, self, visited)

    def init_all_nodes(self):
        # init all names and node handles in model
        for flow_id, flow in self._node_id_to_flow.items():
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

# def _post_order_access_nodes()

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
            if flow_parser._flow_parse_result is None:
                flow_parser._parse_flow_recursive(flow_node_id, parent_flow, mgr, visited)
            parse_res = flow_parser._node_id_to_parse_result[node.id]
            return node, parse_res, True
        return node, None, False


    def _parse_flow_recursive(self, flow_id: str, flow: ADVFlowModel, mgr: ADVProjectBackendManager, visited: set[str]):
        if self._flow_parse_result is not None:
            return self._flow_parse_result

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
        # required for all adv meta code
        global_scope.update({
            "dataclasses": dataclasses_plain,
            "ADV": _ADV,
        })
        # 2. parse symbol group to build symbol scope
        symbols: list[BackendHandle] = []
        sym_group_dep_qnames: list[str] = []
        # use ordered symbol indexes to make sure arguments of fragment are stable
        cnt_base = 0
        all_symbol_node_ids: set[str] = set()
        symbol_nodes: list[ADVNodeModel] = []
        for n in child_nodes:
            if n.nType == ADVNodeType.SYMBOLS:
                all_symbol_node_ids.add(n.id)
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
                    parse_res = parse_res.copy(offset=cnt_base, is_sym_handle=True)
                    self._node_id_to_parse_result[n.id] = parse_res
                symbols.extend(parse_res.symbols)
                cnt_base += parse_res.num_symbols
        
        root_symbol_scope: dict[str, BackendHandle] = {s.handle.symbol_name: s for s in symbols}
        # 3. parse fragments, auto-generated edges will also be handled.
        subflow_nodes: dict[str, list[ADVNodeModel]] = {}
        subflow_name_to_scope: dict[str, dict[str, BackendHandle]] = {}
        for n in child_nodes:
            if n.nType == ADVNodeType.FRAGMENT:
                alias_map = None
                # user can use output mapping to alias output symbols
                # of ref nodes or subflow nodes
                if n.alias_map is not None:
                    # TODO handle error here.
                    alias_map = parse_alias_map(n.alias_map)

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
                    if alias_map is not None:
                        cached_parse_res = cached_parse_res.do_alias_map(alias_map)
                    parse_res = cached_parse_res
                else:
                    if n_def.flow is not None:
                        flow_parser = mgr._flow_node_id_to_parser[flow_id]
                        parse_res = flow_parser._parse_flow_recursive(flow_id, n_def.flow, mgr, visited)
                        if alias_map is not None:
                            parse_res = parse_res.do_alias_map(alias_map)
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
                # TODO rename handle if alias map provided
                if subf_name is not None:
                    # only main flow node will contribute symbols to symbol scope
                    # add outputs to symbol scope
                    for handle in parse_res.output_handles:
                        symbol_scope[handle.symbol_name] = handle 
                if subf_name is not None:
                    if subf_name not in subflow_nodes:
                        subflow_nodes[subf_name] = []
                    subflow_nodes[subf_name].append(n)
        # now all nodes with output handle are parsed. we can build node-handle map.
        inp_node_handle_to_node: dict[tuple[str, str], tuple[ADVNodeModel, BackendHandle]] = {}
        node_id_to_inp_handles: dict[str, list[BackendHandle]] = {}
        out_node_handle_to_node: dict[tuple[str, str], tuple[ADVNodeModel, BackendHandle]] = {}
        node_id_to_out_handles: dict[str, list[BackendHandle]] = {}
        auto_edges: list[ADVEdgeModel] = []
        for n in child_nodes:
            if n.nType in (ADVNodeType.FRAGMENT, ADVNodeType.SYMBOLS):
                node_id_to_inp_handles[n.id] = []
                node_id_to_out_handles[n.id] = []
                parse_res = self._node_id_to_parse_result[n.id]
                assert parse_res is not None
                if isinstance(parse_res, FragmentParseResult):
                    # ref nodes/nested node only use user edges, no auto edges for inputs
                    if n.ref_node_id is None and n.flow is None:
                        for handle in parse_res.input_handles:
                            inp_node_handle_to_node[(n.id, handle.id)] = (n, handle)
                            node_id_to_inp_handles[n.id].append(handle)
                            snid = handle.handle.source_node_id
                            shid = handle.handle.source_handle_id
                            assert snid is not None and shid is not None
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
        
        # 4. parse output indicator node
        edges = list(flow.edges.values())
        edges = list(filter(lambda e: not e.isAutoEdge, edges))
        target_node_id_to_edges: dict[str, list[ADVEdgeModel]] = {}
        for e in edges:
            if e.target not in target_node_id_to_edges:
                target_node_id_to_edges[e.target] = []
            target_node_id_to_edges[e.target].append(e)
        out_indicators: list[ADVNodeModel] = []
        subflow_output_handles: list[tuple[str, BackendHandle]] = []
        for n in child_nodes:
            if n.nType == ADVNodeType.OUT_INDICATOR:
                out_indicators.append(n)
                # output indicator won't be ref node
                if n.id in target_node_id_to_edges:
                    out_edges = target_node_id_to_edges[n.id]
                    for edge in out_edges:
                        assert edge.sourceHandle is not None 
                        key = (edge.source, edge.sourceHandle)
                        if key in out_node_handle_to_node:
                            _, source_handle = out_node_handle_to_node[key]
                            assert edge.targetHandle is not None
                            # target_node_handle_id from output indicators are ignored.
                            # source_handle.target_node_handle_id.add((n.id, edge.targetHandle))
                            source_handle.is_subflow_output = True
                            out_indicator_handle = ADVNodeHandle(
                                id=ADVConstHandles.OutIndicator,
                                name="inputs",
                                is_input=True,
                                type="",
                            )
                            inp_node_handle_to_node[(n.id, edge.targetHandle)] = (n, BackendHandle(handle=out_indicator_handle, index=0))
                            subflow_output_handles.append((edge.source, source_handle))
        # 5. filter invalid user edges, fill target_node_handle_id from user edges
        valid_edges: list[ADVEdgeModel] = []
        for e in edges:
            assert e.sourceHandle is not None and e.targetHandle is not None
            key_src = (e.source, e.sourceHandle)
            key_tgt = (e.target, e.targetHandle)
            # TODO remove edges that cross subflow boundary
            if key_src in out_node_handle_to_node and key_tgt in inp_node_handle_to_node:
                _, out_handle = out_node_handle_to_node[key_src]
                out_handle.target_node_handle_id.add((e.target, e.targetHandle))
                valid_edges.append(e)
            else:
                ADV_LOGGER.warning(f"Edge from node {e.source} handle {e.sourceHandle} to node {e.target} handle {e.targetHandle} is invalid and will be ignored.")
        # 6. parse sub flows. apply post-order sort to each subflow.
        sorted_subflow_nodes: dict[str, list[ADVNodeModel]] = {k: [] for k in subflow_nodes.keys()}
        subflow_node_ids: set[str] = set()
        for subf_name, nodes in subflow_nodes.items():
            # look for nodes that have no connection on output handles
            root_nodes: list[ADVNodeModel] = []
            for node in nodes:
                out_handles = node_id_to_out_handles[node.id]
                is_root_node = False
                if not out_handles:
                    is_root_node = True 
                else:
                    has_output_edges = False
                    for handle in out_handles:
                        if handle.target_node_handle_id and not handle.is_subflow_output:
                            has_output_edges = True
                            break
                    if has_output_edges:
                        is_root_node = False
                    else:
                        is_root_node = True
                if is_root_node:
                    root_nodes.append(node)
            postorder_nodes: list[ADVNodeModel] = []
            visited_nodes: set[str] = set()
            for root_node in root_nodes:
                self._post_order_access_nodes(
                    lambda n: postorder_nodes.append(n),
                    root_node,
                    visited_nodes,
                    node_id_to_inp_handles,
                    mgr._node_id_to_node,
                )
            sorted_subflow_nodes[subf_name] = postorder_nodes
            for n in postorder_nodes:
                subflow_node_ids.add(n.id)

        remain_fragment_nodes: list[ADVNodeModel] = []
        for n in child_nodes:
            if n.nType == ADVNodeType.FRAGMENT and n.id not in subflow_node_ids:
                remain_fragment_nodes.append(n)
        # 7. create input/output handles of flow if exists
        final_edges = valid_edges + auto_edges
        all_input_handles: list[BackendHandle] = []
        all_output_handles: list[BackendHandle] = []

        if ADV_MAIN_FLOW_NAME in sorted_subflow_nodes:
            subflow_node_id_to_nodes = {n.id: n for n in sorted_subflow_nodes[ADV_MAIN_FLOW_NAME]}
            for output_handle in root_symbol_scope.values():
                for target_node_id, _ in output_handle.target_node_handle_id:
                    if target_node_id in subflow_node_id_to_nodes:
                        # output handle connects to main flow node
                        output_handle = output_handle.copy(prefix=ADVHandlePrefix.Input)
                        output_handle.handle.is_input = True
                        output_handle.handle.source_handle_id = None 
                        output_handle.handle.source_node_id = None
                        all_input_handles.append(output_handle)
                        break
            for source_node_id, source_handle in subflow_output_handles:
                if source_node_id in subflow_node_id_to_nodes:
                    all_output_handles.append(source_handle)
            all_output_handles.sort(key=lambda h: h.index)
        # import rich 
        # rich.print(flow_id, all_input_handles, all_output_handles)
        flow_parse_res = FlowParseResult(
            succeed=True,
            input_handles=all_input_handles,
            output_handles=all_output_handles,
            out_type="dict",
            out_type_anno="dict[str, Any]",
            # flow related fields
            symbol_dep_qnames=list(set(sym_group_dep_qnames)),
            edges=final_edges,
            symbol_nodes=symbol_nodes,
            isolate_fragments=remain_fragment_nodes,
            subflow_nodes=sorted_subflow_nodes,
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
            for tgt_node_id, _ in handle.target_node_handle_id:
                tgt_node = node_id_to_node[tgt_node_id]
                self._post_order_access_nodes(accessor, tgt_node, visited, node_id_to_inp_handles, node_id_to_node)
        accessor(node)

