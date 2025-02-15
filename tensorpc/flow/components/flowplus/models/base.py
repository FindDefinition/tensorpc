"""!!!Work in progress!!!

1. Basic Model Design for flow

@dataclass
class EdgeProps:
    # props like UI def
    id: ...
    width: ...

@dataclass
class NodeProps:
    # props like UI def
    id: ...
    width: ...

@dataclass
class FlowModel:
    nodes: dict[str, NodeProps]
    edges: dict[str, EdgeProps]
    node_user_state: dict[str, Any]

@dataclass
class FlowModelRoot(FlowModel):
    selected_node_id: str | None

2. Nested flow

@dataclass
class NodeProps:
    # props like UI def
    id: ...
    width: ...
    # props for nested flow
    nodes: Optional[dict[str, NodeProps]]
    edges: Optional[dict[str, EdgeProps]]
    node_user_state: Optional[dict[str, Any]]

@dataclass
class FlowModelRoot(FlowModel):
    cur_flow_path: list[str] # ["nodes", "node_id_1", "nodes", "node_id_2"]
    selected_node_id: str | None

3. Node template

@dataclass
class NodeProps:
    ...
    code_template_key: Optional[str]

@dataclass
class FlowModelRoot(FlowModel):
    cur_flow_path: list[str] # ["nodes", "node_id_1", "nodes", "node_id_2"]
    selected_node_id: str | None
    custom_node_codes: dict[str, str]

4. Flow Template

@dataclass
class NodeProps:
    ...
    flow_template_key: Optional[str]

@dataclass
class FlowModelRoot(FlowModel):
    ...
    custom_flows: dict[str, FlowModel]

"""

