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

from typing import Callable, Generic, Optional, TypeVar, Union, Any, cast
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.core.datamodel.draft import DraftObject, get_draft_anno_type
from tensorpc.core.datamodel.events import DraftChangeEvent
from tensorpc.flow.core.datamodel import DataModel

from tensorpc.flow.components.flowui import Node, Edge, Flow

@dataclasses.dataclass
class NodeProps(Node):
    pass

@dataclasses.dataclass
class EdgeProps(Edge):
    pass

T_node_model = TypeVar("T_node_model", bound=NodeProps)
T_edge_model = TypeVar("T_edge_model", bound=EdgeProps)


@dataclasses.dataclass
class BaseFlowModel(Generic[T_node_model, T_edge_model]):
    nodes: dict[str, T_node_model]
    edges: dict[str, T_edge_model]
    node_user_state: dict[str, Any]
    selected_node_id: Optional[str]
    
T_flow_model = TypeVar("T_flow_model", bound=BaseFlowModel)

# @dataclasses.dataclass
# class BaseFlowModelRoot(BaseFlowModel[NodeProps, EdgeProps]):
#     pass

class BaseFlowModelBinder(Generic[T_flow_model, T_node_model, T_edge_model]):
    def __init__(self, flow_comp: Flow, model_getter: Callable[[], T_flow_model], draft: Any, 
            to_ui_node: Callable[[T_node_model], Node], to_ui_edge: Callable[[T_edge_model], Edge],
            to_model_edge: Callable[[Edge], T_edge_model]):
        assert isinstance(draft, DraftObject)
        draft_type = get_draft_anno_type(draft)
        assert draft_type is not None and draft_type.is_dataclass_type()
        assert issubclass(draft_type.origin_type, BaseFlowModel)
        self._model_getter = model_getter
        self._draft = cast(BaseFlowModel[T_node_model, T_edge_model], draft)
        self._to_ui_node = to_ui_node
        self._to_ui_edge = to_ui_edge 
        self._flow_comp = flow_comp
        self._to_model_edge = to_model_edge

        self._is_binded = False

    async def _sync_ui_edges_to_model(self):
        """Do basic sync between model and flow ui state. ui data is sync to data model.
        usually used to deal with rare race condition that cause flow-level out-of-sync.
        """
        cur_ui_edge_ids = set([n.id for n in self._flow_comp.edges])
        cur_model = self._model_getter()
        cur_model_edges_ids = set(cur_model.edges.keys())
        ui_node_id_to_del = cur_model_edges_ids - cur_ui_edge_ids
        ui_new_edges: list[Edge] = []
        for n in cur_ui_edge_ids:
            if n not in cur_model_edges_ids:
                ui_new_edges.append(self._to_ui_edge(cast(T_edge_model, cur_model.edges[n])))
        if ui_node_id_to_del:
            await self._flow_comp.delete_edges_by_ids(list(ui_node_id_to_del))
        if ui_new_edges:
            await self._flow_comp.add_edges(ui_new_edges)

    async def _sync_ui_nodes_to_model(self):
        """Do basic sync between model and flow ui state. ui data is sync to data model.
        usually used to deal with rare race condition that cause flow-level out-of-sync.
        """
        cur_ui_node_ids = set([n.id for n in self._flow_comp.nodes])
        cur_model = self._model_getter()
        cur_model_node_ids = set(cur_model.nodes.keys())
        ui_node_id_to_del = cur_model_node_ids - cur_ui_node_ids
        ui_new_nodes: list[Node] = []
        for n in cur_ui_node_ids:
            if n not in cur_model_node_ids:
                ui_new_nodes.append(self._to_ui_node(cast(T_node_model, cur_model.nodes[n])))
        if ui_node_id_to_del:
            await self._flow_comp.delete_nodes_by_ids(list(ui_node_id_to_del))
        if ui_new_nodes:
            await self._flow_comp.add_nodes(ui_new_nodes)

    async def _sync_ui_nodes_edges_to_model(self):
        await self._sync_ui_nodes_to_model()
        await self._sync_ui_edges_to_model()

    def _handle_node_delete(self, nodes: list[Any]):
        # assume this handler is called after default handler
        cur_model = self._model_getter()
        cur_ui_node_ids = set([n.id for n in self._flow_comp.nodes])
        # remove all deleted nodes
        for n_id in cur_model.nodes.keys():
            if n_id not in cur_ui_node_ids:
                self._draft.nodes.pop(n_id)

    def _handle_edge_delete(self, edges: list[Any]):
        # assume this handler is called after default handler
        cur_model = self._model_getter()
        cur_ui_node_ids = set([n.id for n in self._flow_comp.edges])
        # remove all deleted nodes
        for n_id in cur_model.edges.keys():
            if n_id not in cur_ui_node_ids:
                self._draft.edges.pop(n_id)

    def _handle_edge_connection(self, data: dict[str, Any]):
        # assume this handler is called after default handler
        cur_model = self._model_getter()
        for ui_edge in self._flow_comp.edges:
            e_id = ui_edge.id
            if e_id not in cur_model.edges:
                self._draft.edges[e_id] = self._to_model_edge(ui_edge)

    async def _handle_vis_change(self, change: dict):
        # update width/height/position
        # WARNING: width/height change may due to UI or manual resize.
        if "nodes" in change:
            for ui_node in self._flow_comp.nodes:
                self._draft.nodes[ui_node.id].width = ui_node.width
                self._draft.nodes[ui_node.id].height = ui_node.height
                self._draft.nodes[ui_node.id].position = ui_node.position
        await self._sync_ui_nodes_edges_to_model()

    async def _handle_node_logic_change(self, nodes: list[Any]):
        # frontend never trigger add node event, so we only need to handle like node delete
        return self._handle_node_delete(nodes)

    async def _handle_draft_nodes_change(self, ev: DraftChangeEvent):
        await self._sync_ui_nodes_to_model()

    async def _handle_draft_edges_change(self, ev: DraftChangeEvent):
        await self._sync_ui_edges_to_model()

    def bind_flow_component_with_compute_flow_model(self, dm_comp: DataModel):
        if self._is_binded:
            raise ValueError("Already binded")
        # bind flow event handlers
        self._flow_comp.event_edge_connection.on(self._handle_edge_connection)
        self._flow_comp.event_node_delete.on(self._handle_node_delete)
        self._flow_comp.event_edge_delete.on(self._handle_edge_delete)
        self._flow_comp.event_vis_change.on(self._handle_vis_change)
        self._flow_comp.event_node_logic_change.on(self._handle_node_logic_change)
        # bind draft change handlers
        dm_comp.install_draft_change_handler(self._draft.nodes, self._handle_draft_nodes_change)
        dm_comp.install_draft_change_handler(self._draft.edges, self._handle_draft_edges_change)
        self._is_binded = True 

