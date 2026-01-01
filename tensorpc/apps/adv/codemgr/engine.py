from tensorpc.apps.adv.model import ADVFlowModel, ADVNodeModel, ADVNodeHandle, ADVNodeType
from tensorpc.apps.adv.logger import ADV_LOGGER
import tensorpc.core.dataclass_dispatch as dataclasses


class ADVCodeEngine:
    def __init__(self, flow: ADVFlowModel):
        self.flow = flow

    def _get_global_symbols(self):
        symbols: dict[str, tuple[ADVNodeHandle, ADVNodeModel]] = {}
        for node_id, node in self.flow.nodes.items():
            if node.nType == ADVNodeType.SYMBOLS:
                for handle in node.handles:
                    if handle.name in symbols:
                        ADV_LOGGER.warning(f"Duplicate symbol name {handle.name} found in node {node_id}")
                    symbols[handle.name] = (handle, node)
        return symbols

    def get_fragment_deps(self, global_symbols: dict[str, tuple[ADVNodeHandle, ADVNodeModel]]):
        fragment_nodes: list[ADVNodeModel] = []
        for node in self.flow.nodes.values():
            if node.nType == ADVNodeType.FRAGMENT:
                for handle in node.handles:
                    assert handle.symbol_name in global_symbols, \
                        f"Symbol {handle.symbol_name} not found for fragment node {node.id}"
                fragment_nodes.append(node)
        fragment_nodes.sort(key=lambda n: n.position.x)
        tmp_symbols = global_symbols.copy()
        node_id_to_handle_to_conn: dict[str, dict[str, tuple[str, str]]] = {}
        for node in fragment_nodes:
            handle_id_to_conn: dict[str, tuple[str, str]] = {}
            for handle in node.handles:
                if handle.is_input:
                    assert handle.symbol_name in tmp_symbols, \
                        f"Symbol {handle.symbol_name} not found for fragment node {node.id}"
                    tgt_handle, tgt_node = tmp_symbols[handle.symbol_name]
                    handle_id_to_conn[handle.id] = (tgt_node.id, tgt_handle.id)
                else:
                    tmp_symbols[handle.symbol_name] = (handle, node)
            node_id_to_handle_to_conn[node.id] = handle_id_to_conn

        return node_id_to_handle_to_conn

class ADVCodeEngineV2:
    def __init__(self, flow: ADVFlowModel):
        self.flow = flow

    def _get_global_symbols(self):
        symbols: dict[str, tuple[ADVNodeHandle, ADVNodeModel]] = {}
        for node_id, node in self.flow.nodes.items():
            if node.nType == ADVNodeType.SYMBOLS:
                for handle in node.handles:
                    if handle.name in symbols:
                        ADV_LOGGER.warning(f"Duplicate symbol name {handle.name} found in node {node_id}")
                    symbols[handle.name] = (handle, node)
        return symbols

    def get_fragment_deps(self, global_symbols: dict[str, tuple[ADVNodeHandle, ADVNodeModel]]):
        fragment_nodes: list[ADVNodeModel] = []
        for node in self.flow.nodes.values():
            if node.nType == ADVNodeType.FRAGMENT:
                for handle in node.handles:
                    assert handle.symbol_name in global_symbols, \
                        f"Symbol {handle.symbol_name} not found for fragment node {node.id}"
                fragment_nodes.append(node)
        fragment_nodes.sort(key=lambda n: n.position.x)
        tmp_symbols = global_symbols.copy()
        node_id_to_handle_to_conn: dict[str, dict[str, tuple[str, str]]] = {}
        for node in fragment_nodes:
            handle_id_to_conn: dict[str, tuple[str, str]] = {}
            for handle in node.handles:
                if handle.is_input:
                    assert handle.symbol_name in tmp_symbols, \
                        f"Symbol {handle.symbol_name} not found for fragment node {node.id}"
                    tgt_handle, tgt_node = tmp_symbols[handle.symbol_name]
                    handle_id_to_conn[handle.id] = (tgt_node.id, tgt_handle.id)
                else:
                    tmp_symbols[handle.symbol_name] = (handle, node)
            node_id_to_handle_to_conn[node.id] = handle_id_to_conn

        return node_id_to_handle_to_conn


def generate_code(flow: ADVFlowModel):
    pass 

