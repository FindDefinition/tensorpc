"""!!!Work in progress!!!

1. Basic Model Design for flow

@dataclass
class FlowModel:
    nodes: ...
    edges: ...

    node_states: dict[str, Any]
    custom_node_code: dict[str, Any]


2. Nested flow

@dataclass
class NodeState:
    nested_flow: Optional[FlowModel]
    template_flow_key: Optional[str]
    template_node_key: Optional[str]
    user: Any


@dataclass
class FlowModel:
    nodes: ...
    edges: ...

    node_states: dict[str, NodeState]

@dataclass
class FlowModelRoot(FlowModel):
    cur_flow_path: list[str]
    custom_node_code: dict[str, Any]
    selected_node_id: str | None

3. Flow template

@dataclass
class FlowModelRoot(FlowModel):
    # major flow
    cur_major_flow_state_path: list[str]

    # subflow preview
    cur_minor_flow_state_path: list[str] | None

    custom_node_code: dict[str, Any]
    custom_flow: dict[str, FlowModel]

    selected_node_id: str | None

"""