# Copyright 2024 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import contextvars
import enum
from typing import (TYPE_CHECKING, Any, Callable, Coroutine, Dict, Generic, Iterable,
                    List, Optional, Set, Tuple, Type, TypeVar, Union)

from typing_extensions import Literal, TypeAlias

import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.asynctools import cancel_task
from tensorpc.flow.core.appcore import Event
from tensorpc.flow.core.common import handle_standard_event
from tensorpc.flow.jsonlike import merge_props_not_undefined
from tensorpc.utils.uniquename import UniqueNamePool

from ..core.core import (AppEvent, AppEventType, BasicProps, Component,
                    DataclassType, FrontendEventType, NumberType, UIType,
                    Undefined, undefined)
from .mui import (ContainerBaseProps, LayoutType, MUIBasicProps, MUIComponentBase,
                  MUIComponentBaseProps, MUIComponentType, MUIContainerBase,
                  MUIFlexBoxProps, MenuItem, Theme, ValueType)

_T = TypeVar("_T", bound=Component)

@dataclasses.dataclass
class FlowFitViewOptions:
    minZoom: Union[Undefined, int] = undefined
    maxZoom: Union[Undefined, int] = undefined


@dataclasses.dataclass
class FlowProps(ContainerBaseProps):
    className: Union[Undefined, str] = undefined
    nodeDragThreshold: Union[Undefined, int] = undefined
    nodesDraggable: Union[Undefined, bool] = undefined
    nodesConnectable: Union[Undefined, bool] = undefined
    nodesFocusable: Union[Undefined, bool] = undefined
    edgesFocusable: Union[Undefined, bool] = undefined
    elementsSelectable: Union[Undefined, bool] = undefined
    autoPanOnConnect: Union[Undefined, bool] = undefined
    autoPanOnNodeDrag: Union[Undefined, bool] = undefined
    selectionOnDrag: Union[Undefined, bool] = undefined
    selectionMode: Union[Undefined, Literal["partial", "full"]] = undefined
    selectNodesOnDrag: Union[Undefined, bool] = undefined
    connectOnClick: Union[Undefined, bool] = undefined
    connectionMode: Union[Undefined, Literal["loose", "strict"]] = undefined
    panOnDrag: Union[Undefined, bool] = undefined
    panOnScroll: Union[Undefined, bool] = undefined
    panOnScrollSpeed: Union[Undefined, int] = undefined
    panOnScrollMode: Union[Undefined, Literal["horizontal", "vertical",
                                              "free"]] = undefined
    snapToGrid: Union[Undefined, bool] = undefined
    snapGrid: Union[Undefined, Tuple[int, int]] = undefined
    fitView: Union[Undefined, bool] = undefined
    fitViewOptions: Union[Undefined, FlowFitViewOptions] = undefined
    zoomOnScroll: Union[Undefined, bool] = undefined
    zoomOnPinch: Union[Undefined, bool] = undefined
    zoomOnDoubleClick: Union[Undefined, bool] = undefined
    attributionPosition: Union[Undefined,
                               Literal["top-left", "top-right", "bottom-left",
                                       "bottom-right"]] = undefined
    connectionRadius: Union[Undefined, int] = undefined
    ConnectionLineStyle: Union[Undefined, Any] = undefined
    style: Union[Undefined, Any] = undefined
    onlyRenderVisibleElements: Union[Undefined, bool] = undefined
    preventScrolling: Union[Undefined, bool] = undefined
    elevateEdgesOnSelect: Union[Undefined, bool] = undefined
    defaultMarkerColor: Union[Undefined, str] = undefined
    edgeUpdaterRadius: Union[Undefined, int] = undefined
    edgesUpdatable: Union[Undefined, bool] = undefined

    defaultEdgeOptions: Union[Undefined, Any] = undefined
    deleteKeyCode: Union[Undefined, Union[str, List[str], None]] = undefined
    selectionKeyCode: Union[Undefined, Union[str, List[str], None]] = undefined
    multiSelectionKeyCode: Union[Undefined, Union[str, List[str],
                                                  None]] = undefined
    zoomActivationKeyCode: Union[Undefined, Union[str, List[str],
                                                  None]] = undefined
    panActivationKeyCode: Union[Undefined, Union[str, List[str],
                                                 None]] = undefined
    disableKeyboardA11y: Union[Undefined, bool] = undefined
    connectionLineType: Union[Undefined, Literal["default", "straight", "step",
                                                 "smoothstep",
                                                 "simplebezier"]] = undefined
    selectedBoxSxProps: Union[Undefined, Dict[str, Any]] = undefined
    debounce: Union[Undefined, NumberType] = undefined

    droppable: Union[bool, Undefined] = undefined
    allowedDndTypes: Union[List[str], Undefined] = undefined
    allowFile: Union[bool, Undefined] = undefined
    sourceValidConnectMap: Union[Dict[str, Dict[str, Any]], Undefined] = undefined
    targetValidConnectMap: Union[Dict[str, Dict[str, Any]], Undefined] = undefined
    paneContextMenuItems: Union[Undefined, List[MenuItem]] = undefined
    nodeContextMenuItems: Union[Undefined, List[MenuItem]] = undefined
    nodeTypeMap: Union[Undefined, Dict[str, str]] = undefined
    preventCycle: Union[Undefined, bool] = undefined

    invisiblizeAllResizer: Union[Undefined, bool] = undefined
    invisiblizeAllToolbar: Union[Undefined, bool] = undefined

@dataclasses.dataclass
class XYPosition:
    x: NumberType
    y: NumberType


@dataclasses.dataclass
class NodeData:
    component: Union[Undefined, Component] = undefined
    selectedTheme: Union[Undefined, Theme] = undefined
    selectedBoxSxProps: Union[Undefined, Dict[str, Any]] = undefined
    data: Union[Undefined, Any] = undefined
    label: Union[Undefined, str] = undefined
    sourceEdgeOverrides: Union[Undefined, Dict[str, Any]] = undefined


@dataclasses.dataclass
class Node:
    id: str
    data: Union[Undefined, NodeData] = undefined
    # type: Union[Undefined,
    #             Literal["app", "appTemplate", "input", "default", "output",
    #                     "group", "annotation"]] = undefined
    type: Union[Undefined, str] = undefined

    position: XYPosition = dataclasses.field(
        default_factory=lambda: XYPosition(0, 0))
    style: Union[Undefined, Any] = undefined
    className: Union[Undefined, str] = undefined
    dragHandle: Union[Undefined, str] = undefined
    hidden: Union[Undefined, bool] = undefined
    draggable: Union[Undefined, bool] = undefined
    selectable: Union[Undefined, bool] = undefined
    connectable: Union[Undefined, bool] = undefined
    deletable: Union[Undefined, bool] = undefined
    width: Union[Undefined, NumberType] = undefined
    height: Union[Undefined, NumberType] = undefined
    parentId: Union[Undefined, str] = undefined
    focusable: Union[Undefined, bool] = undefined
    extent: Union[Undefined, Literal["parent"],
                  Tuple[Tuple[NumberType, NumberType],
                        Tuple[NumberType, NumberType]]] = undefined
    sourcePosition: Union[Undefined, Literal["left", "top", "right",
                                                "bottom"]] = undefined
    targetPosition: Union[Undefined, Literal["left", "top", "right",
                                                "bottom"]] = undefined
    def get_component(self) -> Optional[Component]:
        if not isinstance(self.data, Undefined):
            if not isinstance(self.data.component, Undefined):
                return self.data.component
        return None

    def get_component_checked(self, type: Type[_T]) -> _T:
        if not isinstance(self.data, Undefined):
            if not isinstance(self.data.component, Undefined):
                if isinstance(self.data.component, type):
                    return self.data.component
        raise ValueError(f"node don't contain component with type {type}")



@dataclasses.dataclass
class EdgeMarker:
    type: Literal["arrow", "arrowclosed"]
    color: Union[Undefined, str] = undefined
    width: Union[Undefined, NumberType] = undefined
    height: Union[Undefined, NumberType] = undefined
    markerUnits: Union[Undefined, str] = undefined
    orient: Union[Undefined, str] = undefined
    strokeWidth: Union[Undefined, NumberType] = undefined


@dataclasses.dataclass
class Edge:
    id: str
    source: str
    target: str
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None
    type: Union[Undefined, Literal["default", "straight", "step",
                                   "smoothstep"]] = undefined
    style: Union[Undefined, Any] = undefined
    animated: Union[Undefined, bool] = undefined
    hidden: Union[Undefined, bool] = undefined
    focusable: Union[Undefined, bool] = undefined
    label: Union[Undefined, str] = undefined
    markerStart: Union[Undefined, EdgeMarker, str] = undefined
    markerEnd: Union[Undefined, EdgeMarker, str] = undefined


@dataclasses.dataclass
class _NodesHelper:
    nodes: List[Node]


@dataclasses.dataclass
class _EdgesHelper:
    edges: List[Edge]


class FlowControlType(enum.IntEnum):
    DagreLayout = 0
    FitView = 1
    AddNewNodes = 2
    DeleteNodeByIds = 3
    UpdateNodeInternals = 4
    UpdateNodeProps = 5
    UpdateNodeData = 6
    UpdateNodeStyle = 7
    DeleteEdgeByIds = 8
    UpdatePaneContextMenuItem = 9

@dataclasses.dataclass
class DagreLayoutOptions:
    rankdir: Union[Undefined, Literal["TB", "BT", "LR", "RL"]] = undefined
    align: Union[Undefined, Literal["UL", "UR", "DL", "DR"]] = undefined
    nodesep: Union[Undefined, NumberType] = undefined
    ranksep: Union[Undefined, NumberType] = undefined
    marginx: Union[Undefined, NumberType] = undefined
    marginy: Union[Undefined, NumberType] = undefined
    edgesep: Union[Undefined, NumberType] = undefined
    acyclicer: Union[Undefined, Literal["greedy"]] = undefined
    ranker: Union[Undefined, Literal["network-simplex", "tight-tree",
                                     "longest-path"]] = undefined

@dataclasses.dataclass
class EventSelection:
    nodes: List[str]
    edges: List[str]

class Flow(MUIContainerBase[FlowProps, MUIComponentType]):

    @dataclasses.dataclass
    class ChildDef:
        nodes: List[Node]
        edges: List[Edge]
        extraChilds: Union[Undefined, List[Component]] = undefined
        componentTemplate: Union[Undefined, Component] = undefined

    def __init__(
            self,
            nodes: List[Node],
            edges: List[Edge],
            extra_childs: Union[Undefined, List[Component]] = undefined,
            component_template: Union[Undefined,
                                      Component] = undefined) -> None:
        super().__init__(UIType.Flow,
                         FlowProps,
                         Flow.ChildDef(nodes, edges, extra_childs,
                                       component_template),
                         allowed_events=[
                             FrontendEventType.FlowSelectionChange.value,
                             FrontendEventType.FlowNodesInitialized.value,
                             FrontendEventType.FlowEdgeConnection.value,
                             FrontendEventType.FlowEdgeDelete.value,
                             FrontendEventType.FlowNodeDelete.value,
                             FrontendEventType.Drop.value,
                             FrontendEventType.FlowPaneContextMenu.value,
                             FrontendEventType.FlowNodeContextMenu.value,
                             FrontendEventType.FlowNodeLogicChange.value,
                         ])

        self.event_change = self._create_event_slot(
            FrontendEventType.Change)
        self.event_selection_change = self._create_event_slot(
            FrontendEventType.FlowSelectionChange, lambda x: EventSelection(**x))
        self.event_nodes_initialized = self._create_event_slot(
            FrontendEventType.FlowNodesInitialized)
        self.event_edge_connection = self._create_event_slot(
            FrontendEventType.FlowEdgeConnection)
        self.event_edge_delete = self._create_event_slot(
            FrontendEventType.FlowEdgeDelete)
        self.event_node_delete = self._create_event_slot(
            FrontendEventType.FlowNodeDelete)
        self.event_node_logic_change = self._create_event_slot(
            FrontendEventType.FlowNodeLogicChange)

        self.event_drop = self._create_event_slot(FrontendEventType.Drop)
        self.event_pane_context_menu = self._create_event_slot(FrontendEventType.FlowPaneContextMenu)
        self.event_node_context_menu = self._create_event_slot(FrontendEventType.FlowNodeContextMenu)
        self._update_graph_data()
        # we must due with delete event because it comes earlier than change event.
        self.event_node_delete.on(self._handle_node_delete)
        self.event_edge_delete.on(self._handle_edge_delete)
        self.event_edge_connection.on(self._handle_new_edge)
        self.event_node_logic_change.on(self._handle_node_logic_change)

        self._unique_name_pool_node = UniqueNamePool()
        self._unique_name_pool_edge = UniqueNamePool()

        self.set_flow_event_context_creator(lambda: enter_flow_ui_context(self))

    @property
    def childs_complex(self):
        assert isinstance(self._child_structure, Flow.ChildDef)
        return self._child_structure

    @property
    def nodes(self):
        return self.childs_complex.nodes

    @property
    def edges(self):
        return self.childs_complex.edges

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    def create_unique_node_id(self, id: str):
        return self._unique_name_pool_node(id)

    def create_unique_edge_id(self, id: str):
        return self._unique_name_pool_edge(id)

    def _find_comps_in_dataclass(self, child: "Flow.ChildDef"):
        unique_name_pool = UniqueNamePool()
        res: List[Tuple[Component, str]] = []
        for node in child.nodes:
            if not isinstance(node.data, Undefined) and not isinstance(
                    node.data.component, Undefined):
                comp = node.data.component
                unique_name_pool(node.id)
                res.append((comp, node.id))
        if not isinstance(child.componentTemplate, Undefined):
            res.append((child.componentTemplate,
                        unique_name_pool("__flow_template__")))
        if not isinstance(child.extraChilds, Undefined):
            for i, c in enumerate(child.extraChilds):
                res.append((c, unique_name_pool(f"extraChilds:{i}")))
        return res

    def _update_graph_data(self):
        # node id must unique
        self._id_to_node = {node.id: node for node in self.nodes}
        assert len(self._id_to_node) == len(self.nodes)
        self._id_to_edge = {edge.id: edge for edge in self.edges}
        # edge id must unique
        assert len(self._id_to_edge) == len(self.edges)
        self._source_id_to_edge = {edge.source: edge for edge in self.edges}
        self._target_id_to_edge = {edge.target: edge for edge in self.edges}
        self._node_id_to_sources: Dict[str, List[Tuple[str, Optional[str], Optional[str]]]] = {node.id: [] for node in self.nodes}
        self._node_id_to_targets: Dict[str, List[Tuple[str, Optional[str], Optional[str]]]] = {node.id: [] for node in self.nodes}
        self._node_id_to_handle_to_edges: Dict[str, Dict[Optional[str], List[Edge]]] = {node.id: {} for node in self.nodes}
        for edge in self.edges:
            self._node_id_to_targets[edge.source].append(
                (self._id_to_node[edge.target].id, edge.sourceHandle, edge.targetHandle))
            self._node_id_to_sources[edge.target].append(
                (self._id_to_node[edge.source].id, edge.sourceHandle, edge.targetHandle))
            if edge.sourceHandle not in self._node_id_to_handle_to_edges[edge.source]:
                self._node_id_to_handle_to_edges[edge.source][edge.sourceHandle] = []
            self._node_id_to_handle_to_edges[edge.source][edge.sourceHandle].append(edge)
            if edge.targetHandle not in self._node_id_to_handle_to_edges[edge.target]:
                self._node_id_to_handle_to_edges[edge.target][edge.targetHandle] = []
            self._node_id_to_handle_to_edges[edge.target][edge.targetHandle].append(edge)

        # TODO detection cycle
        for n in self.nodes:
            if not isinstance(n, Undefined):
                assert n.id in self._id_to_node
        all_node_ids = set(self._id_to_node.keys())
        self._unique_name_pool_node = UniqueNamePool(init_set=all_node_ids)
        all_edge_ids = set(self._id_to_edge.keys())
        self._unique_name_pool_edge = UniqueNamePool(init_set=all_edge_ids)

    def get_node_by_id(self, node_id: str):
        return self._id_to_node[node_id]

    def has_node_id(self, node_id: str):
        return node_id in self._id_to_node 

    def get_source_nodes(self, node_id: str):
        return [
            self._id_to_node[idh[0]] for idh in self._node_id_to_sources[node_id]
        ]

    def get_target_nodes(self, node_id: str):
        return [
            self._id_to_node[idh[0]] for idh in self._node_id_to_targets[node_id]
        ]

    def get_source_node_and_handles(self, node_id: str):
        return [
            (self._id_to_node[idh[0]], idh[1], idh[2]) for idh in self._node_id_to_sources[node_id]
        ]

    def get_target_node_and_handles(self, node_id: str):
        return [
            (self._id_to_node[idh[0]], idh[1], idh[2]) for idh in self._node_id_to_targets[node_id]
        ]

    def get_edges_by_node_and_handle_id(self, node_id: str, handle_id: Optional[str]):
        content = self._node_id_to_handle_to_edges[node_id]
        return content.get(handle_id, [])

    def get_all_nodes_in_connected_graph(self, node: Node):
        visited: Set[str] = set()
        stack = [node]
        res: List[Node] = []
        while stack:
            cur = stack.pop()
            if cur.id in visited:
                continue
            visited.add(cur.id)
            res.append(cur)
            all_connected = self.get_source_nodes(cur.id) + self.get_target_nodes(
                cur.id)
            for n in all_connected:
                stack.append(n)
        return res

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        # print("flow event", ev.type, ev.data)
        return await handle_standard_event(self, ev, is_sync=is_sync, sync_state_after_change=False, change_status=False)
   
    def _handle_node_logic_change(self, nodes: List[Any]):
        cur_id_to_comp: Dict[str, Component] = {}
        for n in self.nodes:
            if not isinstance(n.data, Undefined) and not isinstance(
                    n.data.component, Undefined):
                assert n.data.component._flow_uid is not None
                cur_id_to_comp[n.data.component._flow_uid.
                               uid_encoded] = n.data.component
        for node_raw in nodes:
            if "data" in node_raw:
                data = node_raw["data"]
                if "component" in data:
                    assert data["component"] in cur_id_to_comp
                    data["component"] = cur_id_to_comp[data["component"]]
        self.childs_complex.nodes = _NodesHelper(nodes).nodes

    def state_change_callback(
            self,
            value: dict,
            type: ValueType = FrontendEventType.Change.value):
        if "nodes" in value:
            # print(value)
            cur_id_to_comp: Dict[str, Component] = {}
            for n in self.nodes:
                if not isinstance(n.data, Undefined) and not isinstance(
                        n.data.component, Undefined):
                    assert n.data.component._flow_uid is not None
                    cur_id_to_comp[n.data.component._flow_uid.
                                   uid_encoded] = n.data.component
            for node_raw in value["nodes"]:
                if "data" in node_raw:
                    data = node_raw["data"]
                    if "component" in data:
                        assert data["component"] in cur_id_to_comp
                        data["component"] = cur_id_to_comp[data["component"]]
            self.childs_complex.nodes = _NodesHelper(value["nodes"]).nodes
        if "edges" in value:
            self.childs_complex.edges = _EdgesHelper(value["edges"]).edges
        self._update_graph_data()

    async def _handle_node_delete(self, nodes: List[Any]):
        """triggered when you use frontend api to delete nodes such as deleteKeyCode
        """
        return await self.delete_nodes_by_ids([n["id"] for n in nodes], _internal_dont_send_comp_event=True) 

    async def _handle_new_edge(self, data: Dict[str, Any]):
        new_edge = Edge(**data["newEdge"])
        self.childs_complex.edges.append(new_edge)
        self._update_graph_data()

    def _validate_node_ids(self, node_ids: List[str]):
        for node_id in node_ids:
            assert node_id in self._id_to_node, f"node id {node_id} not exists"

    async def update_node_internals(self, node_ids: List[str]):
        self._validate_node_ids(node_ids)
        res = {
            "type": FlowControlType.UpdateNodeInternals.value,
            "nodeIds": node_ids,
        }
        return await self.send_and_wait(self.create_comp_event(res))

    async def update_node_props(self, node_id: str, props: Dict[str, Any]):
        self._validate_node_ids([node_id])
        assert "data" not in props, "you can't update data via this api, use update_node_data instead"
        res = {
            "type": FlowControlType.UpdateNodeProps.value,
            "nodeId": node_id,
            "data": props,
        }
        return await self.send_and_wait(self.create_comp_event(res))

    async def update_node_data(self, node_id: str, data: Dict[str, Any]):
        assert "component" not in data, "you can't update component via this api"
        self._validate_node_ids([node_id])
        res = {
            "type": FlowControlType.UpdateNodeData.value,
            "nodeId": node_id,
            "data": data,
        }
        return await self.send_and_wait(self.create_comp_event(res))

    async def update_node_style(self, node_id: str, data: Dict[str, Any]):
        self._validate_node_ids([node_id])
        res = {
            "type": FlowControlType.UpdateNodeStyle.value,
            "nodeId": node_id,
            "data": data,
        }
        return await self.send_and_wait(self.create_comp_event(res))

    def _handle_edge_delete(self, edges: List[Any]):
        edge_ids_set = set([e["id"] for e in edges])
        new_edges: List[Edge] = []
        for edge in self.edges:
            if edge.id in edge_ids_set:
                continue
            new_edges.append(edge)
        self.childs_complex.edges = new_edges

    async def do_dagre_layout(self,
                              options: Optional[DagreLayoutOptions] = None,
                              fit_view: bool = False):
        if options is None:
            options = DagreLayoutOptions()
        res = {
            "type": FlowControlType.DagreLayout,
            "graphOptions": options,
            "fitView": fit_view,
        }
        return await self.send_and_wait(self.create_comp_event(res))

    async def fit_view(self):
        res = {
            "type": FlowControlType.FitView,
            "fitView": True,
        }
        return await self.send_and_wait(self.create_comp_event(res))

    async def update_pane_context_menu_items(self, items: List[MenuItem]):
        """Update pane context menu items based on id.
        this function won't add or remove items, only update the existing items.
        """
        if not isinstance(self.props.paneContextMenuItems, Undefined):
            all_item_id_to_items = {item.id: item for item in self.props.paneContextMenuItems}
            for item in items:
                if item.id not in all_item_id_to_items:
                    raise ValueError(f"item id {item.id} not exists")
                merge_props_not_undefined(all_item_id_to_items[item.id], item)
            res = {
                "type": FlowControlType.UpdatePaneContextMenuItem,
                "menuItems": items,
            }
            return await self.send_and_wait(self.create_comp_event(res))

    async def add_nodes(self, nodes: List[Node], screen_to_flow: Optional[bool] = None):
        """Add new nodes to the flow.

        Args:
            nodes (Node): nodes to add.
            screen_to_flow (Optional[bool], optional): Whether the node position is in screen coordinates. Defaults to None.
                you should use this when you use position from pane context menu or drag-drop to add a node.
        """

        new_layout: Dict[str, Component] = {}
        for node in nodes:
            assert node.id not in self._id_to_node, f"node id {node.id} already exists"
            comp = node.get_component()
            if comp is not None:
                new_layout[node.id] = comp
            self.nodes.append(node)
        self._update_graph_data()
        ev_new_node = {
            "type": FlowControlType.AddNewNodes,
            "nodes": nodes,
        }
        if screen_to_flow is not None:
            ev_new_node["screenToFlowPosition"] = screen_to_flow
        if new_layout:
            return await self.update_childs(new_layout,
                                            update_child_complex=False,
                                            additional_ev_creator=lambda: self.
                                            create_comp_event(ev_new_node))
        else:
            return await self.send_and_wait(self.create_comp_event(ev_new_node))

    async def add_node(self, node: Node, screen_to_flow: Optional[bool] = None):
        """Add a new node to the flow.

        Args:
            node (Node): The node to add.
            screen_to_flow (Optional[bool], optional): Whether the node position is in screen coordinates. Defaults to None.
                you should use this when you use position from pane context menu or drag-drop to add a node.
        """
        await self.add_nodes([node], screen_to_flow)

    async def delete_nodes_by_ids(self, node_ids: List[str], *, _internal_dont_send_comp_event: bool = False):
        node_ids_set = set(node_ids)
        new_nodes: List[Node] = []
        del_node_id_with_comp: List[str] = []
        for node in self.nodes:
            if node.id not in node_ids_set:
                new_nodes.append(node)
            else:
                if not isinstance(node.data, Undefined):
                    if not isinstance(node.data.component, Undefined):
                        del_node_id_with_comp.append(node.id)
        self.childs_complex.nodes = new_nodes
        # remove edges
        new_edges: List[Edge] = []
        for edge in self.edges:
            if edge.source in node_ids_set or edge.target in node_ids_set:
                continue
            new_edges.append(edge)
        self.childs_complex.edges = new_edges
        self._update_graph_data()
        ev_del_node = {
            "type": FlowControlType.DeleteNodeByIds,
            "nodeIds": node_ids,
        }
        if del_node_id_with_comp:
            if _internal_dont_send_comp_event:
                return await self.remove_childs_by_keys(
                    del_node_id_with_comp,
                    update_child_complex=False)
            else:
                return await self.remove_childs_by_keys(
                    del_node_id_with_comp,
                    update_child_complex=False,
                    additional_ev_creator=lambda: self.create_comp_event(ev_del_node))
        else:
            if not _internal_dont_send_comp_event:
                return await self.send_and_wait(self.create_comp_event(ev_del_node))

    async def delete_edges_by_ids(self, edge_ids: List[str]):
        edge_ids_set = set(edge_ids)
        new_edges: List[Edge] = []
        for edge in self.edges:
            if edge.id not in edge_ids_set:
                new_edges.append(edge)
        self.childs_complex.edges = new_edges
        self._update_graph_data()
        ev_del_edge = {
            "type": FlowControlType.DeleteEdgeByIds.value,
            "edgeIds": edge_ids,
        }
        return await self.send_and_wait(self.create_comp_event(ev_del_edge))

class FlowUIContext:

    def __init__(self, flow: Flow) -> None:
        self.flow = flow


FLOW_CONTEXT_VAR: contextvars.ContextVar[
    Optional[FlowUIContext]] = contextvars.ContextVar("simpleflowui_context",
                                                   default=None)


def get_flow_ui_context() -> Optional[FlowUIContext]:
    return FLOW_CONTEXT_VAR.get()


@contextlib.contextmanager
def enter_flow_ui_context(flow: "Flow"):
    ctx = FlowUIContext(flow)
    token = FLOW_CONTEXT_VAR.set(ctx)
    try:
        yield ctx
    finally:
        FLOW_CONTEXT_VAR.reset(token)

@dataclasses.dataclass
class HandleProps(MUIFlexBoxProps):
    type: Union[Literal["source", "target"], Undefined] = undefined
    handledPosition: Union[Literal["left", "top", "right", "bottom"],
                    Undefined] = undefined
    isConnectable: Union[bool, Undefined] = undefined
    style: Union[Undefined, Any] = undefined
    id: Union[Undefined, str] = undefined
    className: Union[Undefined, str] = undefined
    connectionLimit: Union[Undefined, int] = undefined

class Handle(MUIComponentBase[HandleProps]):

    def __init__(self, type: Literal["source", "target"],
                 position: Literal["left", "top", "right", "bottom"],
                 id: Union[Undefined, str] = undefined) -> None:
        super().__init__(UIType.Handle, HandleProps, [])
        self.prop(type=type, handledPosition=position, id=id)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class NodeColorMap:
    app: Union[Undefined, str] = undefined
    input: Union[Undefined, str] = undefined
    default: Union[Undefined, str] = undefined
    output: Union[Undefined, str] = undefined
    group: Union[Undefined, str] = undefined
    annotation: Union[Undefined, str] = undefined


@dataclasses.dataclass
class MiniMapProps(MUIBasicProps):
    nodeColorMap: Union[Undefined, NodeColorMap] = undefined
    nodeStrokeColorMap: Union[Undefined, NodeColorMap] = undefined
    nodeBorderRadius: Union[Undefined, int] = undefined
    nodeStrokeWidth: Union[Undefined, int] = undefined
    maskColor: Union[Undefined, str] = undefined
    maskStrokeColor: Union[Undefined, str] = undefined
    maskStrokeWidth: Union[Undefined, int] = undefined
    position: Union[Undefined, Literal["top-left", "top-right", "bottom-left",
                                       "bottom-right", "top-center",
                                       "bottom-center"]] = undefined
    pannable: Union[Undefined, bool] = undefined
    zoomable: Union[Undefined, bool] = undefined
    inversePan: Union[Undefined, bool] = undefined
    zoomStep: Union[Undefined, int] = undefined
    offsetScale: Union[Undefined, int] = undefined


class MiniMap(MUIComponentBase[MiniMapProps]):

    def __init__(self) -> None:
        super().__init__(UIType.FlowMiniMap, MiniMapProps, [])

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class ControlsProps(MUIBasicProps):
    position: Union[Undefined, Literal["top-left", "top-right", "bottom-left",
                                       "bottom-right", "top-center",
                                       "bottom-center"]] = undefined
    showZoom: Union[Undefined, bool] = undefined
    showFitView: Union[Undefined, bool] = undefined
    showInteractive: Union[Undefined, bool] = undefined
    fitViewOptions: Union[Undefined, FlowFitViewOptions] = undefined


class Controls(MUIComponentBase[ControlsProps]):

    def __init__(self) -> None:
        super().__init__(UIType.FlowControls, ControlsProps, [])

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class BackgroundProps(MUIBasicProps):
    id: Union[Undefined, str] = undefined
    variant: Union[Undefined, Literal["lines", "dots", "cross"]] = undefined
    color: Union[Undefined, str] = undefined
    gap: Union[Undefined, NumberType] = undefined
    size: Union[Undefined, NumberType] = undefined
    offset: Union[Undefined, NumberType] = undefined
    lineWidth: Union[Undefined, NumberType] = undefined


class Background(MUIComponentBase[BackgroundProps]):

    def __init__(self) -> None:
        super().__init__(UIType.FlowBackground, BackgroundProps, [])

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class NodeResizerProps(MUIBasicProps):
    minWidth: Union[Undefined, NumberType] = undefined
    minHeight: Union[Undefined, NumberType] = undefined
    keepAspectRatio: Union[Undefined, bool] = undefined
    maxWidth: Union[Undefined, NumberType] = undefined
    maxHeight: Union[Undefined, NumberType] = undefined
    isVisible: Union[Undefined, bool] = undefined
    color: Union[Undefined, str] = undefined
    handleClassName: Union[Undefined, str] = undefined
    lineClassName: Union[Undefined, str] = undefined
    handleStyle: Union[Undefined, Any] = undefined
    lineStyle: Union[Undefined, Any] = undefined


class NodeResizer(MUIComponentBase[NodeResizerProps]):

    def __init__(self) -> None:
        super().__init__(UIType.FlowNodeResizer, NodeResizerProps, [])

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class NodeToolbarProps(ContainerBaseProps):
    position: Union[Undefined, Literal["top", "bottom", "left",
                                       "right"]] = undefined
    isVisible: Union[Undefined, bool] = undefined
    offset: Union[Undefined, NumberType] = undefined
    align: Union[Undefined, Literal["center", "start", "end"]] = undefined


class NodeToolbar(MUIContainerBase[NodeToolbarProps, MUIComponentType]):

    def __init__(self, children: LayoutType) -> None:
        if isinstance(children, list):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.FlowNodeToolBar,
                         NodeToolbarProps,
                         children,
                         allowed_events=[])

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)
