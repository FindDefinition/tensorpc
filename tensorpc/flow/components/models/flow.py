import contextlib
import contextvars
import enum
from functools import partial
from typing import (TYPE_CHECKING, Any, Callable, Coroutine, Generic,
                    Iterable, Optional, Sequence, Type,
                    TypeVar, Union, cast)

from typing_extensions import Literal, TypeAlias
import dataclasses as dataclasses_plain
import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.asynctools import cancel_task
from tensorpc.core.datamodel.draft import DraftBase, DraftObject, get_draft_anno_type, insert_assign_draft_op
from tensorpc.core.datamodel.events import DraftChangeEvent
from tensorpc.core.tree_id import UniqueTreeIdForTree
from tensorpc.flow.core.appcore import Event
from tensorpc.flow.core.common import handle_standard_event
from tensorpc.flow.core.datamodel import DataModel
from tensorpc.flow.jsonlike import Undefined, asdict_flatten_field_only, asdict_flatten_field_only_no_undefined, merge_props_not_undefined, undefined_dict_factory
from tensorpc.utils.uniquename import UniqueNamePool
from tensorpc.flow.components.flowui import Node, Edge, Flow, EventSelection

@dataclasses.dataclass
class BaseNodeModel(Node):
    pass

@dataclasses.dataclass
class BaseEdgeModel(Edge):
    pass

T_node_model = TypeVar("T_node_model", bound=BaseNodeModel)
T_edge_model = TypeVar("T_edge_model", bound=BaseEdgeModel)

@dataclasses.dataclass
class BaseFlowModel(Generic[T_node_model, T_edge_model]):
    nodes: dict[str, T_node_model]
    edges: dict[str, T_edge_model]
    
T_flow_model = TypeVar("T_flow_model", bound=BaseFlowModel)

def _default_to_ui_edge(edge: BaseEdgeModel):
    return Edge(**dataclasses.asdict(edge))

def _default_to_model_edge(edge: Edge):
    return BaseEdgeModel(**dataclasses.asdict(edge))

class BaseFlowModelBinder(Generic[T_flow_model, T_node_model, T_edge_model]):
    def __init__(self, flow_comp: Flow, model_getter: Callable[[], T_flow_model], draft: Any, 
            to_ui_node: Callable[[T_node_model], Node], to_ui_edge: Optional[Callable[[T_edge_model], Edge]] = None,
            to_model_edge: Optional[Callable[[Edge], T_edge_model]] = None) -> None:
        """
        Args:
            flow_comp (Flow): flow component instance.
            model_getter (Callable[[], T_flow_model]): a function to get the current model instance.
            draft (Any): a draft object to store the model data.
            to_ui_node (Callable[[T_node_model], Node]): a function to convert model node to ui node.
            to_ui_edge (Optional[Callable[[T_edge_model], Edge]], optional): a function to convert model edge to ui edge. Defaults to None.
                if not provided, your edge model must be BaseEdgeModel, no subclass.
            to_model_edge (Optional[Callable[[Edge], T_edge_model]], optional): a function to convert ui edge to model edge. Defaults to None.
                if not provided, your edge model must be BaseEdgeModel, no subclass.
        """
        
        assert isinstance(draft, DraftObject)
        draft_type = get_draft_anno_type(draft)
        assert draft_type is not None and draft_type.is_dataclass_type()
        assert issubclass(draft_type.origin_type, BaseFlowModel)
        self._model_getter = model_getter
        self._draft = cast(BaseFlowModel[T_node_model, T_edge_model], draft)
        self._to_ui_node = to_ui_node
        if to_ui_edge is None or to_model_edge is None:
            # when user don't provide to_ui_edge or to_model_edge, we assume
            # the edge model is BaseEdgeModel, no subclass.
            anno_type = get_draft_anno_type(self._draft.edges["test_key"])
            assert anno_type is not None and anno_type.origin_type == BaseEdgeModel
        if to_ui_edge is None:
            to_ui_edge = _default_to_ui_edge
        if to_model_edge is None:
            to_model_edge = cast(Callable[[Edge], T_edge_model], _default_to_model_edge)
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
        for n in cur_model_edges_ids:
            if n not in cur_ui_edge_ids:
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
        for n in cur_model_node_ids:
            if n not in cur_ui_node_ids:
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
                if not isinstance(ui_node.width, Undefined):
                    self._draft.nodes[ui_node.id].width = ui_node.width
                if not isinstance(ui_node.height, Undefined):
                    self._draft.nodes[ui_node.id].height = ui_node.height
                self._draft.nodes[ui_node.id].position = ui_node.position
        await self._sync_ui_nodes_edges_to_model()

    async def _handle_node_logic_change(self, nodes: list[Any]):
        # frontend never trigger add node event, so we only need to handle like node delete
        return self._handle_node_delete(nodes)

    async def _handle_node_selection(self, selection: EventSelection, draft: DraftBase):
        # frontend never trigger add node event, so we only need to handle like node delete
        draft_type = get_draft_anno_type(draft)
        assert draft_type is not None
        if issubclass(draft_type.origin_type, str):
            # single selection
            if selection.nodes:
                insert_assign_draft_op(draft, selection.nodes[0])
            else:
                insert_assign_draft_op(draft, None)
        else:
            # draft is list[str]
            insert_assign_draft_op(draft, selection.nodes)
        

    async def _handle_draft_nodes_change(self, ev: DraftChangeEvent):
        await self._sync_ui_nodes_to_model()

    async def _handle_draft_edges_change(self, ev: DraftChangeEvent):
        # print("???? DRAFT CHANGE", ev)
        await self._sync_ui_edges_to_model()

    def bind_flow_com_with_base_model(self, dm_comp: DataModel, selected_node_draft: Optional[Any] = None):
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
        if selected_node_draft is not None:
            draft_type = get_draft_anno_type(selected_node_draft)
            assert draft_type is not None 
            if issubclass(draft_type.origin_type, str):
                assert draft_type.is_optional, "selected node must be Optional[str]"
            elif draft_type.is_sequence_type():
                assert issubclass(draft_type.origin_type, str), "selected node must be List[str] if is list"
            else:
                raise ValueError("selected node must be Optional[str] or List[str]")
            self._flow_comp.event_selection_change.on(partial(self._handle_node_selection, draft=selected_node_draft))
        self._is_binded = True 