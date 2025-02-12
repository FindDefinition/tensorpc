"""!!!Work in progress!!!

1. Basic Model Design for flow

@dataclass
class EdgeState:
    # props like UI def
    id: ...
    width: ...

@dataclass
class NodeState:
    # props like UI def
    id: ...
    width: ...
    # props for user
    user: Any

@dataclass
class FlowModel:
    nodes: dict[str, NodeState]
    edges: dict[str, EdgeState]

@dataclass
class FlowModelRoot(FlowModel):
    selected_node_id: str | None


2. Nested flow

@dataclass
class NodeState:
    # props like UI def
    id: ...
    width: ...
    # props for user
    user: Any
    # props for nested flow
    nodes: Optional[dict[str, NodeState]]
    edges: Optional[dict[str, EdgeState]]

@dataclass
class FlowModelRoot(FlowModel):
    cur_flow_path: list[str] # ["nodes", "node_id_1", "nodes", "node_id_2"]
    selected_node_id: str | None

3. Node template

@dataclass
class NodeState:
    ...
    code_template_key: Optional[str]

@dataclass
class FlowModelRoot(FlowModel):
    cur_flow_path: list[str] # ["nodes", "node_id_1", "nodes", "node_id_2"]
    selected_node_id: str | None
    custom_node_codes: dict[str, str]

4. Flow Template

@dataclass
class NodeState:
    ...
    flow_template_key: Optional[str]

@dataclass
class FlowModelRoot(FlowModel):
    ...
    custom_flows: dict[str, FlowModel]

"""

from tensorpc.core import dataclass_dispatch as dataclasses

