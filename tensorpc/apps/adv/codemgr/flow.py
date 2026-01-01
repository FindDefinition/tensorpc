import ast
import inspect
from typing import Any
from tensorpc.apps.adv.codemgr.core import BaseParseResult
from tensorpc.apps.adv.codemgr.symbols import SymbolParser
import tensorpc.core.dataclass_dispatch as dataclasses

from tensorpc.apps.adv.logger import ADV_LOGGER
from tensorpc.apps.adv.model import ADVFlowModel, ADVNodeModel, ADVNodeHandle, ADVNodeType, ADVProject
import hashlib
from tensorpc.core.annolib import dataclass_flatten_fields_generator, unparse_type_expr
import dataclasses as dataclasses_plain

_ROOT_FLOW_ID = ""

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

class FlowParser:
    def __init__(self):
        self._global_scope_parser = GlobalScriptParser()
        self._code_to_global_scope: dict[str, dict[str, Any]] = {}

        self._node_id_to_parse_result: dict[str, BaseParseResult] = {}

        self._need_parse: bool = True

    def clear_parse_cache(self):
        self._node_id_to_parse_result.clear()
        self._need_parse = True

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

        real_node_id_tuple: list[tuple[]]
        real_node_ids = [n.id for n in child_nodes if n.ref_node_id is None else n.ref_node_id]
        # make sure non-referenced nodes are handled first.
        child_nodes.sort(key=lambda n: (n.ref_node_id is not None, n.position.x))
        # 1. scan GLOBAL_SCRIPT nodes to build global scope
        global_scripts: dict[str, str] = {}
        for n in child_nodes:
            if n.nType == ADVNodeType.GLOBAL_SCRIPT:
                n, cached_parse_res, is_external_node = self._get_or_compile_real_node(flow_id, n, mgr, visited)
                assert n.nType == ADVNodeType.GLOBAL_SCRIPT
                assert n.impl is not None, f"GLOBAL_SCRIPT node {n.id} has no code."
                global_scripts[n.id] = (n.impl.code)
        global_script = "\n".join(global_scripts.values())
        global_scope = self._global_scope_parser.parse_global_script(global_script)
        # 2. parse symbol group
        for n in child_nodes:
            if n.nType == ADVNodeType.SYMBOLS:
                n, cached_parse_res, is_external_node = self._get_or_compile_real_node(flow_id, n, mgr, visited)
                assert n.nType == ADVNodeType.SYMBOLS
                assert n.impl is not None, f"SYMBOLS node {n.id} has no code."
                parser = SymbolParser()
                parse_res = parser.parse_symbol_node(n.impl.code, global_scope, list(global_scripts.values()))
                self._node_id_to_parse_result[n.id] = parse_res

        self._need_parse = False