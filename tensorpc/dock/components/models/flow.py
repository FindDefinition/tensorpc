from functools import partial
import traceback
from typing import (TYPE_CHECKING, Annotated, Any, Callable, Coroutine, Generic,
                    Iterable, Optional, Sequence, Type,
                    TypeVar, Union, cast)
import rich
from typing_extensions import Self
from tensorpc.apps.cflow.logger import CFLOW_LOGGER
import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.datamodel.asdict import as_dict_no_undefined
from tensorpc.core.datamodel.draft import DraftBase, DraftFieldMeta, DraftObject, get_draft_anno_type, get_draft_ast_node, insert_assign_draft_op
from tensorpc.core.datamodel.draftast import evaluate_draft_ast, evaluate_draft_ast_noexcept
from tensorpc.core.datamodel.events import DraftChangeEvent, DraftEventType
from tensorpc.dock.core.datamodel import DataModel
from tensorpc.dock.jsonlike import Undefined
from tensorpc.dock.components.flowui import FlowInternals, XYPosition, NodeBase, EdgeBase, Node, Edge, Flow, EventSelection, NodeData
from tensorpc.core.annolib import undefined


@dataclasses.dataclass
class BaseNodeModel(NodeBase):
    width: Union[Undefined, Union[int, float]] = undefined
    height: Union[Undefined, Union[int, float]] = undefined
    position: XYPosition = dataclasses.field(
        default_factory=lambda: XYPosition(0, 0))


@dataclasses.dataclass
class BaseEdgeModel(EdgeBase):
    pass

T_node_model = TypeVar("T_node_model", bound=BaseNodeModel)
T_edge_model = TypeVar("T_edge_model", bound=BaseEdgeModel)

@dataclasses.dataclass
class BaseFlowModel(Generic[T_node_model, T_edge_model]):
    nodes: dict[str, T_node_model]
    edges: dict[str, T_edge_model]
    runtime: Annotated[FlowInternals[T_node_model, T_edge_model], DraftFieldMeta(is_external=True)] = dataclasses.field(default_factory=FlowInternals[T_node_model, T_edge_model])

    def __post_init__(self):
        try:
            self.runtime.set_from_nodes_edges(list(self.nodes.values()), list(self.edges.values()))
        except:
            traceback.print_exc()
            raise 

T_flow_model = TypeVar("T_flow_model", bound=BaseFlowModel)

def _default_to_ui_edge(edge: BaseEdgeModel):
    return Edge(**dataclasses.asdict(edge))

def _default_to_model_edge(edge: EdgeBase):
    return BaseEdgeModel(edge.id, edge.source, edge.target,
        edge.sourceHandle, edge.targetHandle)

class BaseFlowModelBinder(Generic[T_flow_model, T_node_model, T_edge_model]):
    def __init__(self, flow_comp: Flow, model_getter: Callable[[], Any], draft: Any, 
            to_ui_node: Callable[[T_flow_model, str], Node], to_ui_edge: Optional[Callable[[T_edge_model], Edge]] = None,
            to_model_edge: Optional[Callable[[Edge], T_edge_model]] = None,
            flow_uid_getter: Optional[Callable[[], str]] = None,
            debug_id: str = "flow") -> None:
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
        
        assert isinstance(draft, DraftObject), f"draft must be DraftObject, but got {type(draft)}"
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
        self._flow_uid_getter = flow_uid_getter

        self._is_binded = False

        self._debug_id = debug_id

    def _get_cur_model_may_nested(self):
        root_model = self._model_getter()
        # print(root_model, self._draft)
        cur_model = evaluate_draft_ast_noexcept(get_draft_ast_node(self._draft), root_model)
        return cast(Optional[T_flow_model], cur_model)

    async def _sync_ui_edges_to_model(self):
        """Do basic sync between model and flow ui state. ui data is sync to data model.
        usually used to deal with rare race condition that cause flow-level out-of-sync.
        """
        cur_ui_edge_ids = set([n.id for n in self._flow_comp.edges])
        cur_model = self._get_cur_model_may_nested()
        if cur_model is None:
            # eval failed, this usually means current node don't have a sub flow.
            await self._flow_comp.clear()
        else:
            try:
                cur_model_edges_ids = set(cur_model.edges.keys())
                ui_node_id_to_del = cur_ui_edge_ids - cur_model_edges_ids
                ui_new_edges: list[Edge] = []
                for n in cur_model_edges_ids:
                    if n not in cur_ui_edge_ids:
                        ui_new_edges.append(self._to_ui_edge(cast(T_edge_model, cur_model.edges[n])))
                if ui_node_id_to_del:
                    await self._flow_comp.delete_edges_by_ids(list(ui_node_id_to_del))
                if ui_new_edges:
                    await self._flow_comp.add_edges(ui_new_edges)
            except:
                # rich.print(as_dict_no_undefined(cur_model))
                CFLOW_LOGGER.error("Flow %s set failed.", self._debug_id)
                raise 


    async def _sync_ui_nodes_to_model(self):
        """Do basic sync between model and flow ui state. ui data is sync to data model.
        usually used to deal with rare race condition that cause flow-level out-of-sync.
        """
        cur_ui_node_ids = set([n.id for n in self._flow_comp.nodes])
        cur_model = self._get_cur_model_may_nested()
        if cur_model is None:
            # print(self._model_getter())
            # CFLOW_LOGGER.warning("_sync_ui_nodes_to_model: Flow %s eval failed, use empty flow.", self._debug_id)
            # eval failed, this usually means current node don't have a sub flow.
            await self._flow_comp.clear()
        else:
            cur_model_node_ids = set(cur_model.nodes.keys())
            ui_node_id_to_del = cur_ui_node_ids - cur_model_node_ids
            ui_new_nodes: list[Node] = []
            for n in cur_model_node_ids:
                if n not in cur_ui_node_ids:
                    ui_new_nodes.append(self._to_ui_node(cur_model, n))
            if ui_node_id_to_del:
                await self._flow_comp.delete_nodes_by_ids(list(ui_node_id_to_del))
            if ui_new_nodes:
                await self._flow_comp.add_nodes(ui_new_nodes)
        
    async def _switch_flow(self, nodes: dict[str, T_node_model], edges: dict[str, T_edge_model], flow_user_uid: Optional[str] = None):
        """Do basic sync between model and flow ui state. ui data is sync to data model.
        usually used to deal with rare race condition that cause flow-level out-of-sync.
        """
        cur_model = self._get_cur_model_may_nested()
        if cur_model is None:
            # eval failed, this usually means current node don't have a sub flow.
            await self._flow_comp.clear()
            return
        ui_nodes: list[Node] = []
        for n in nodes.values():
            ui_nodes.append(self._to_ui_node(cur_model, n.id))
        ui_edges: list[Edge] = []
        for e in edges.values():
            ui_edges.append(self._to_ui_edge(e))
        await self._flow_comp.switch_flow(ui_nodes, ui_edges, flow_user_uid=flow_user_uid)

    async def _sync_ui_nodes_edges_to_model(self):
        await self._sync_ui_nodes_to_model()
        await self._sync_ui_edges_to_model()

    def _handle_node_delete(self, data: dict):
        if not self._is_flow_user_uid_same(data.get("flowUserUid")):
            # if not same, the flow is changed, return
            return
        # assume this handler is called after default handler
        cur_model = self._get_cur_model_may_nested()
        if cur_model is not None:
            cur_ui_node_ids = set([n.id for n in self._flow_comp.nodes])
            # remove all deleted nodes
            for n_id in cur_model.nodes.keys():
                if n_id not in cur_ui_node_ids:
                    self._draft.nodes.pop(n_id)

    def _handle_edge_delete(self, data: dict):
        if not self._is_flow_user_uid_same(data.get("flowUserUid")):
            # if not same, the flow is changed, return
            return
        # assume this handler is called after default handler
        cur_model = self._get_cur_model_may_nested()
        if cur_model is not None:
            cur_ui_node_ids = set([n.id for n in self._flow_comp.edges])
            # remove all deleted nodes
            for n_id in cur_model.edges.keys():
                if n_id not in cur_ui_node_ids:
                    self._draft.edges.pop(n_id)

    def _handle_edge_connection(self, data: dict[str, Any]):
        if not self._is_flow_user_uid_same(data.get("flowUserUid")):
            # if not same, the flow is changed, return
            return
        # assume this handler is called after default handler
        cur_model = self._get_cur_model_may_nested()
        if cur_model is not None:
            for ui_edge in self._flow_comp.edges:
                e_id = ui_edge.id
                if e_id not in cur_model.edges:
                    self._draft.edges[e_id] = self._to_model_edge(ui_edge)

    async def _handle_vis_change(self, change: dict):
        if not self._is_flow_user_uid_same(change.get("flowUserUid")):
            # if not same, the flow is changed, return
            return
        # update width/height/position from debounced frontend event
        # WARNING: width/height change may due to UI or manual resize.
        # WARNING: when you switch flow before this event is finished, the width/height may
        # set to wrong nodes, so you should set global-unique name for all nodes include nested flow nodes.
        if "nodes" in change:
            cur_model = self._get_cur_model_may_nested()
            if cur_model is not None:
                for ui_node in self._flow_comp.nodes:
                    if ui_node.id in cur_model.nodes:
                        if not isinstance(ui_node.width, Undefined):
                            self._draft.nodes[ui_node.id].width = ui_node.width
                        if not isinstance(ui_node.height, Undefined):
                            self._draft.nodes[ui_node.id].height = ui_node.height
                        self._draft.nodes[ui_node.id].position = ui_node.position

    async def _handle_node_logic_change(self, data: dict):
        if not self._is_flow_user_uid_same(data.get("flowUserUid")):
            # if not same, the flow is changed, return
            return
        # frontend never trigger add node event, so we only need to handle like node delete
        return self._handle_node_delete(data)

    async def _handle_node_selection(self, selection: EventSelection, draft: DraftBase):
        if not self._is_flow_user_uid_same(selection.flowUserUid):
            # if not same, the flow is changed, return
            return
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

    def _is_flow_user_uid_same(self, event_flow_uid: Optional[str]):
        if self._flow_uid_getter is None or event_flow_uid is None:
            return True
        flow_user_uid = self._flow_uid_getter()
        return flow_user_uid == event_flow_uid

    async def _handle_draft_change(self, ev: DraftChangeEvent):
        # we observe draft.nodes and draft.edges, there will be two kind of change:
        # 1. whole nodes/edges changed (ObjectIdChange), this means evaluated nodes/edges dict is changed.
        # usually this is caused by flow change, so we need to reset the whole flow ui.
        # 2. user change nodes/edges dict by pop/clear/update etc. (DictChange), so we compare changed
        # nodes/edges with previous and sync to flow ui.
        # keep in mind that nested change (e.g. nodes["id"].content = ...) won't trigger `DictChange`.
        node_change_type = ev.type_dict["nodes"]
        edge_change_type = ev.type_dict["edges"]
        # handle whole flow change
        # TODO should we watch path draft change?
        # print(DraftEventType(node_change_type), DraftEventType(edge_change_type))
        is_mount_change = node_change_type == DraftEventType.MountChange
        is_full_change = node_change_type == DraftEventType.ObjectIdChange and edge_change_type == DraftEventType.ObjectIdChange
        if is_full_change or is_mount_change:
            flow_user_uid = None
            if self._flow_uid_getter is not None:
                flow_user_uid = self._flow_uid_getter()
            nodes = ev.new_value_dict["nodes"]
            edges = ev.new_value_dict["edges"]
            if nodes is not None and edges is not None:
                # print(ev.new_value_dict)
                return await self._switch_flow(nodes, edges, flow_user_uid)
        if node_change_type == DraftEventType.ObjectIdChange or edge_change_type == DraftEventType.ObjectIdChange:
            await self._flow_comp.clear()
            if self._flow_uid_getter is not None:
                flow_user_uid = self._flow_uid_getter()
                await self._flow_comp.send_and_wait(self._flow_comp.update_event(flowUserUid=flow_user_uid))
        is_changed = False
        if node_change_type != DraftEventType.NoChange:
            await self._sync_ui_nodes_to_model()
            is_changed = True
        if edge_change_type != DraftEventType.NoChange:
            await self._sync_ui_edges_to_model()
            is_changed = True
        if is_changed:
            cur_model = self._get_cur_model_may_nested()
            if cur_model is None:
                # eval failed, use empty flow
                await self._flow_comp.clear()
            else:
                cur_model.runtime.set_from_nodes_edges(list(cur_model.nodes.values()), list(cur_model.edges.values()))


    def bind_flow_comp_with_base_model(self, dm_comp: DataModel, selected_node_draft: Optional[Any] = None):
        if self._is_binded:
            raise ValueError("Already binded")
        # bind flow event handlers
        self._flow_comp.event_edge_connection.on(self._handle_edge_connection)
        self._flow_comp.event_node_delete.on(self._handle_node_delete)
        self._flow_comp.event_edge_delete.on(self._handle_edge_delete)
        self._flow_comp.event_vis_change.on(self._handle_vis_change)
        self._flow_comp.event_node_logic_change.on(self._handle_node_logic_change)
        # bind draft change handlers
        dm_comp.install_draft_change_handler({
            "nodes": self._draft.nodes,
            "edges": self._draft.edges
        }, self._handle_draft_change)
        if selected_node_draft is not None:
            draft_type = get_draft_anno_type(selected_node_draft)
            assert draft_type is not None 
            if issubclass(draft_type.origin_type, str):
                assert draft_type.is_optional, "selected node must be Optional[str]"
            elif draft_type.is_sequence_type():
                assert issubclass(draft_type.child_types[0], str), "selected node must be List[str] if is list"
                assert not draft_type.is_optional, "selected node not be Optional if it's list[str]"
            else:
                raise ValueError("selected node must be Optional[str] or List[str]")
            self._flow_comp.event_selection_change.on(partial(self._handle_node_selection, draft=selected_node_draft))
        self._is_binded = True 
