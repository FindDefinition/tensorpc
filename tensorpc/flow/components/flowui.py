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
from typing import (TYPE_CHECKING, Any, Callable, Coroutine, Dict, Generic,
                    Iterable, List, Optional, Sequence, Set, Tuple, Type,
                    TypeVar, Union, cast)

from typing_extensions import Literal, TypeAlias
import dataclasses as dataclasses_plain
import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.asynctools import cancel_task
from tensorpc.core.tree_id import UniqueTreeIdForTree
from tensorpc.flow.core.appcore import Event
from tensorpc.flow.core.common import handle_standard_event
from tensorpc.flow.jsonlike import asdict_flatten_field_only, asdict_flatten_field_only_no_undefined, merge_props_not_undefined, undefined_dict_factory
from tensorpc.utils.uniquename import UniqueNamePool

from ..core.component import (AppEvent, AppEventType, BasicProps, Component,
                              DataclassType, FrontendEventType, NumberType,
                              UIType, Undefined, undefined)
from .mui import (ContainerBaseProps, LayoutType, MUIBasicProps,
                  MUIComponentBase, MUIComponentBaseProps, MUIComponentType,
                  MUIContainerBase, MUIFlexBoxProps, MenuItem, Theme,
                  ValueType)

_T = TypeVar("_T", bound=Component)


@dataclasses.dataclass
class FlowFitViewOptions:
    minZoom: Union[Undefined, int] = undefined
    maxZoom: Union[Undefined, int] = undefined


NodeTypeLiteral: TypeAlias = Literal["app", "appTemplate", "input", "default", "output",
                            "group"]

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
    connectionLineStyle: Union[Undefined, Any] = undefined
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
    sourceValidConnectMap: Union[Dict[str, Dict[str, Any]],
                                 Undefined] = undefined
    targetValidConnectMap: Union[Dict[str, Dict[str, Any]],
                                 Undefined] = undefined
    paneContextMenuItems: Union[Undefined, List[MenuItem]] = undefined
    nodeContextMenuItems: Union[Undefined, List[MenuItem]] = undefined
    nodeTypeMap: Union[Undefined,
                       Dict[str, NodeTypeLiteral]] = undefined
    preventCycle: Union[Undefined, bool] = undefined

    invisiblizeAllResizer: Union[Undefined, bool] = undefined
    invisiblizeAllToolbar: Union[Undefined, bool] = undefined

    defaultLayoutSize: Union[Undefined, Tuple[NumberType,
                                              NumberType]] = undefined


@dataclasses.dataclass
class XYPosition:
    x: NumberType
    y: NumberType


@dataclasses.dataclass
class NodeData:
    component: Union[Undefined, Component] = undefined
    selectedTheme: Union[Undefined, Theme] = undefined
    selectedBoxSxProps: Union[Undefined, Dict[str, Any]] = undefined
    userdata: Union[Undefined, Any] = undefined
    label: Union[Undefined, str] = undefined
    sourceEdgeOverrides: Union[Undefined, Dict[str, Any]] = undefined
    contextMenuItems: Union[Undefined, List[MenuItem]] = undefined
    # used by default nodes with multiple handles
    targetHandleIds: Union[Undefined, List[Optional[str]]] = undefined
    sourceHandleIds: Union[Undefined, List[Optional[str]]] = undefined
    partition: Union[Undefined, int] = undefined

@dataclasses.dataclass
class Node:
    id: str
    data: Union[Undefined, NodeData] = undefined
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
    initialWidth: Union[Undefined, NumberType] = undefined
    initialHeight: Union[Undefined, NumberType] = undefined

    parentId: Union[Undefined, str] = undefined
    focusable: Union[Undefined, bool] = undefined
    extent: Union[Undefined, Literal["parent"],
                  Tuple[Tuple[NumberType, NumberType],
                        Tuple[NumberType, NumberType]]] = undefined
    sourcePosition: Union[Undefined, Literal["left", "top", "right",
                                             "bottom"]] = undefined
    targetPosition: Union[Undefined, Literal["left", "top", "right",
                                             "bottom"]] = undefined

    def set_component(self, comp: Component):
        if isinstance(self.data, Undefined):
            self.data = NodeData()
        self.data.component = comp

    def set_component_replaced(self, comp: Component):
        if isinstance(self.data, Undefined):
            node_data = NodeData()
        else:
            node_data = dataclasses.replace(self.data, component=comp)
        return dataclasses.replace(self, data=node_data)

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

    def get_user_data(self) -> Optional[Any]:
        if not isinstance(self.data, Undefined):
            if not isinstance(self.data.userdata, Undefined):
                return self.data.userdata
        return None

    def get_node_data(self):
        if not isinstance(self.data, Undefined):
            return self.data
        return None


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
                                   "smoothstep", "multistep", "multismoothstep"]] = undefined
    style: Union[Undefined, Any] = undefined
    animated: Union[Undefined, bool] = undefined
    hidden: Union[Undefined, bool] = undefined
    focusable: Union[Undefined, bool] = undefined
    label: Union[Undefined, str] = undefined
    markerStart: Union[Undefined, EdgeMarker, str] = undefined
    markerEnd: Union[Undefined, EdgeMarker, str] = undefined
    # only available when type is svgstep
    # if undefined, it becomes a smoothstep edge
    data: Union[Undefined, Any] = undefined


@dataclasses.dataclass
class _NodesHelper:
    nodes: List[Node]


@dataclasses.dataclass
class _EdgesHelper:
    edges: List[Edge]

class LayoutAlgoType(enum.IntEnum):
    Dagre = 0
    Elk = 1

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
    SetFlowAndDagreLayout = 10
    LocateNode = 11
    SetFlowAndElkLayout = 12
    ElkLayout = 13


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
class ElkGlobalOptions:
    direction: Union[Undefined, Literal["DOWN", "UP", "RIGHT", "LEFT"]] = "DOWN"
    padding: Union[Undefined, str] = undefined

_PortAlignmentLiteral: TypeAlias = Literal["BEGIN", "END", "CENTER", "JUSTIFIED", "DISTRIBUTED"]

@dataclasses.dataclass
class ElkPortAlignment:
    default: Union[_PortAlignmentLiteral, Undefined] = undefined 
    west: Union[_PortAlignmentLiteral, Undefined] = undefined
    east: Union[_PortAlignmentLiteral, Undefined] = undefined
    north: Union[_PortAlignmentLiteral, Undefined] = undefined
    south: Union[_PortAlignmentLiteral, Undefined] = undefined

@dataclasses.dataclass
class ElkDefaultNodeProps:
    portAlignment: Union[Undefined, ElkPortAlignment] = undefined
    portConstraints: Union[Undefined, str] = undefined

@dataclasses.dataclass
class ElkSpacing:
    commentComment: Union[Undefined, NumberType] = undefined
    commentNode: Union[Undefined, NumberType] = undefined
    nodeNodeBetweenLayers: Union[Undefined, NumberType] = undefined
    nodeNode: Union[Undefined, NumberType] = undefined
    edgeNode: Union[Undefined, NumberType] = undefined
    edgeNodeBetweenLayers: Union[Undefined, NumberType] = undefined
    edgeEdge: Union[Undefined, NumberType] = undefined
    edgeEdgeBetweenLayers: Union[Undefined, NumberType] = undefined
    componentComponent: Union[Undefined, NumberType] = undefined
    portPort: Union[Undefined, NumberType] = undefined
    portsSurrounding: Union[Undefined, str] = undefined

@dataclasses.dataclass
class ElkConsiderModelOrder:
    components: Union[Undefined, Literal["NONE", "INSIDE_PORT_SIDE_GROUPS", "GROUP_MODEL_ORDER", "MODEL_ORDER"]] = undefined
    strategy: Union[Undefined, Literal["NONE", "NODES_AND_EDGES", "PREFER_EDGES", "PREFER_NODES"]] = undefined
    portModelOrder: Union[Undefined, bool] = undefined
    noModelOrder: Union[Undefined, bool] = undefined
    longEdgeStrategy: Union[Undefined, Literal["DUMMY_NODE_OVER", "DUMMY_NODE_UNDER", "EQUAL"]] = undefined
    crossingCounterNodeInfluence: Union[Undefined, NumberType] = undefined
    crossingCounterPortInfluence: Union[Undefined, NumberType] = undefined

@dataclasses.dataclass
class ElkPartitioning:
    activate: Union[Undefined, bool] = undefined

@dataclasses.dataclass
class ElkNodePlacement:
    strategy: Union[Undefined, Literal["LINEAR_SEGMENTS", "SIMPLE", "BRANDES_KOEPF", "NETWORK_SIMPLEX"]] = undefined
    favorStraightEdges: Union[Undefined, bool] = undefined

@dataclasses.dataclass
class ElkLayoutOptions:
    algorithm: Literal["layered", "mrtree"] = "layered"
    elk: Union[Undefined, ElkGlobalOptions] = dataclasses.field(default_factory=ElkGlobalOptions)
    considerModelOrder: Union[Undefined, ElkConsiderModelOrder] = undefined
    nodePlacement: Union[Undefined, ElkNodePlacement] = undefined
    partitioning: Union[Undefined, ElkPartitioning] = undefined
    spacing: Union[Undefined, ElkSpacing] = undefined
    defaultNodeProps: Union[Undefined, ElkDefaultNodeProps] = undefined

@dataclasses.dataclass
class EventSelection:
    nodes: List[str]
    edges: List[str]

_T_node_data_dict = TypeVar("_T_node_data_dict", bound=Optional[Dict[str, Any]])
_T_edge_data_dict = TypeVar("_T_edge_data_dict", bound=Optional[Dict[str, Any]])

@dataclasses_plain.dataclass
class FlowInternals:
    id_to_node: Dict[str, Node] = dataclasses_plain.field(default_factory=dict)
    id_to_edge: Dict[str, Edge] = dataclasses_plain.field(default_factory=dict)
    node_id_to_sources: Dict[str, List[Tuple[
        str, Optional[str],
        Optional[str]]]] = dataclasses_plain.field(default_factory=dict)
    node_id_to_targets: Dict[str, List[Tuple[
        str, Optional[str],
        Optional[str]]]] = dataclasses_plain.field(default_factory=dict)
    node_id_to_inp_handle_to_edges: Dict[
        str, Dict[Optional[str],
                  List[Edge]]] = dataclasses_plain.field(default_factory=dict)
    node_id_to_out_handle_to_edges: Dict[
        str, Dict[Optional[str],
                  List[Edge]]] = dataclasses_plain.field(default_factory=dict)
    unique_name_pool_node: UniqueNamePool = dataclasses_plain.field(
        default_factory=UniqueNamePool)
    unique_name_pool_edge: UniqueNamePool = dataclasses_plain.field(
        default_factory=UniqueNamePool)

    @property
    def nodes(self):
        return list(self.id_to_node.values())

    @property
    def edges(self):
        return list(self.id_to_edge.values())

    def set_from_nodes_edges(self, nodes: List[Node], edges: List[Edge]):
        # node id must unique
        self.id_to_node = {node.id: node for node in nodes}
        assert len(self.id_to_node) == len(nodes)

        self.id_to_edge = {edge.id: edge for edge in edges}
        # edge id must unique
        assert len(self.id_to_edge) == len(edges)
        self.node_id_to_sources = {node.id: [] for node in nodes}
        self.node_id_to_targets = {node.id: [] for node in nodes}
        self.node_id_to_inp_handle_to_edges = {node.id: {} for node in nodes}
        self.node_id_to_out_handle_to_edges = {node.id: {} for node in nodes}
        for edge in edges:
            self.node_id_to_targets[edge.source].append(
                (edge.target, edge.sourceHandle, edge.targetHandle))
            self.node_id_to_sources[edge.target].append(
                (edge.source, edge.sourceHandle, edge.targetHandle))
            if edge.sourceHandle not in self.node_id_to_out_handle_to_edges[
                    edge.source]:
                self.node_id_to_out_handle_to_edges[edge.source][
                    edge.sourceHandle] = []
            self.node_id_to_out_handle_to_edges[edge.source][
                edge.sourceHandle].append(edge)
            if edge.targetHandle not in self.node_id_to_inp_handle_to_edges[
                    edge.target]:
                self.node_id_to_inp_handle_to_edges[edge.target][
                    edge.targetHandle] = []
            self.node_id_to_inp_handle_to_edges[edge.target][
                edge.targetHandle].append(edge)
        all_node_ids = set(self.id_to_node.keys())
        self.unique_name_pool_node = UniqueNamePool(init_set=all_node_ids)
        all_edge_ids = set(self.id_to_node.keys())
        self.unique_name_pool_edge = UniqueNamePool(init_set=all_edge_ids)

    def _calculate_node_group_meta(
        self,
        node_ids: List[str],
        group_id: Optional[str] = None,
        merged_out_to_merged_id: Optional[Dict[Tuple[str, Optional[str]],
                                               str]] = None):
        inside_node_id_out_handle_to_edges: Dict[Tuple[str, Optional[str]],
                                                  List[Edge]] = {}
        outside_node_id_out_handle_to_edges: Dict[Tuple[str, Optional[str]],
                                                  List[Edge]] = {}
        # used for input edges order correction
        inside_node_id_inp_handle_to_edges: Dict[Tuple[str, Optional[str]],
                                                  List[Edge]] = {}
        outside_node_id_inp_handle_to_edges: Dict[Tuple[str, Optional[str]],
                                                  List[Edge]] = {}

        node_id_to_inp_handle_to_edges = self.node_id_to_inp_handle_to_edges.copy(
        )
        node_id_to_out_handle_to_edges = self.node_id_to_out_handle_to_edges.copy(
        )

        node_ids_set = set(node_ids)
        for node_id_to_merge in node_ids:
            inp_handle_to_edges = node_id_to_inp_handle_to_edges[
                node_id_to_merge]
            out_handle_to_edges = node_id_to_out_handle_to_edges[
                node_id_to_merge]
            for handle, edges in inp_handle_to_edges.items():
                # check edge connect to outside
                for edge in edges:
                    if edge.source not in node_ids_set:
                        key = (edge.source, edge.sourceHandle)
                        if key not in outside_node_id_out_handle_to_edges:
                            outside_node_id_out_handle_to_edges[key] = []
                        outside_node_id_out_handle_to_edges[key].append(edge)
                        key_inp = (edge.target, edge.targetHandle)
                        if key_inp not in inside_node_id_inp_handle_to_edges:
                            inside_node_id_inp_handle_to_edges[key_inp] = []
                        inside_node_id_inp_handle_to_edges[key_inp].append(edge)
            for handle, edges in out_handle_to_edges.items():
                # check edge connect to outside
                for edge in edges:
                    if edge.target not in node_ids_set:
                        key = (edge.source, handle)
                        if key not in inside_node_id_out_handle_to_edges:
                            inside_node_id_out_handle_to_edges[key] = []
                        inside_node_id_out_handle_to_edges[key].append(edge)
                        key_out = (edge.target, edge.targetHandle)
                        if key_out not in outside_node_id_inp_handle_to_edges:
                            outside_node_id_inp_handle_to_edges[key_out] = []
                        outside_node_id_inp_handle_to_edges[key_out].append(edge)
                        if merged_out_to_merged_id is not None and group_id is not None:
                            if key not in merged_out_to_merged_id:
                                merged_out_to_merged_id[key] = group_id

        return (inside_node_id_out_handle_to_edges, outside_node_id_out_handle_to_edges,
            inside_node_id_inp_handle_to_edges, outside_node_id_inp_handle_to_edges)

    def _get_order_corrected_outside_out(self, outside_out: Dict[Tuple[str, Optional[str]],
                                                  List[Edge]], inside_in: Dict[Tuple[str, Optional[str]],
                                                  List[Edge]]):
        outside_out_visited: Set[Tuple[str, Optional[str]]] = set()
        new_outside_out: Dict[Tuple[str, Optional[str]], List[Edge]] = {}
        for _, edges in inside_in.items():
            for edge in edges:
                key = (edge.source, edge.sourceHandle)
                if key in outside_out_visited:
                    continue
                outside_out_visited.add(key)
                new_outside_out[key] = outside_out[key]
        return new_outside_out 

    def _get_order_corrected_inside_out(self, inside_out: Dict[Tuple[str, Optional[str]],
                                                  List[Edge]], outside_in: Dict[Tuple[str, Optional[str]],
                                                  List[Edge]]):
        inside_out_visited: Set[Tuple[str, Optional[str]]] = set()
        new_inside_out: Dict[Tuple[str, Optional[str]], List[Edge]] = {}
        for _, edges in outside_in.items():
            for edge in edges:
                key = (edge.source, edge.sourceHandle)
                if key in inside_out_visited:
                    continue
                inside_out_visited.add(key)
                new_inside_out[key] = inside_out[key]
        return new_inside_out 

    def merge_nodes_with_data(
        self,
        merge_list: List[Tuple[Node, List[str]]],
        merged_data: Optional[List[Any]] = None,
        node_id_to_data: _T_node_data_dict = None,
        edge_id_to_data: _T_edge_data_dict = None,
        correct_inp_edge_order: bool = True
    ) -> Tuple["FlowInternals", Dict[str, List[Edge]], _T_node_data_dict, _T_edge_data_dict]:
        """merge nodes, then return a new `FlowInternals`, remain self unchanged.

        this API ensures that nodes in `merge_list` won't be changed except their id.
        """
        # check merged node id is valid and have no intersection
        node_id_set_to_merge: Set[str] = set()
        for _, merge_ids in merge_list:
            for merge_id in merge_ids:
                assert merge_id in self.id_to_node
                assert merge_id not in node_id_set_to_merge, f"node id {merge_id} already merged"
                node_id_set_to_merge.add(merge_id)
        node_ids_not_to_merge = set(
            self.id_to_node.keys()) - node_id_set_to_merge
        not_to_merge_name_pool = UniqueNamePool(init_set=node_ids_not_to_merge)
        for merged_node, _ in merge_list:
            merged_node.id = not_to_merge_name_pool(merged_node.id)
        # append nodes and edges that not in merge list
        new_nodes: List[Node] = [x[0] for x in merge_list]
        new_edges: List[Edge] = []
        node_id_to_merged_id: Dict[str, str] = {}
        for merged_node, merge_ids in merge_list:
            for merge_id in merge_ids:
                node_id_to_merged_id[merge_id] = merged_node.id
        prev_no_merge_nodes: List[Node] = []
        for node in self.id_to_node.values():
            if node.id not in node_id_set_to_merge:
                new_nodes.append(node)
                prev_no_merge_nodes.append(node)
        if not correct_inp_edge_order:
            # if we want correct order, we can't add non-merged edges first
            for edge in self.id_to_edge.values():
                if edge.source not in node_id_set_to_merge and edge.target not in node_id_set_to_merge:
                    new_edges.append(edge)
        edge_name_pool = UniqueNamePool(
            init_set=set([edge.id for edge in new_edges]))
        merged_id_to_outside_out_to_edges: Dict[str, Dict[Tuple[str,
                                                                Optional[str]],
                                                          List[Edge]]] = {}
        merged_id_to_inside_out_to_edges: Dict[str, Dict[Tuple[str,
                                                                Optional[str]],
                                                          List[Edge]]] = {}
        merged_id_to_outside_inp_to_edges: Dict[str, Dict[Tuple[str,
                                                                Optional[str]],
                                                          List[Edge]]] = {}
        merged_id_to_inside_inp_to_edges: Dict[str, Dict[Tuple[str,
                                                                Optional[str]],
                                                          List[Edge]]] = {}

        
        merged_out_to_merged_id: Dict[Tuple[str, Optional[str]], str] = {}

        outside_out_to_merged_handle: Dict[Tuple[str, Optional[str], str],
                                           Optional[str]] = {}
        inside_out_to_merged_handle: Dict[Tuple[str, Optional[str]],
                                           Optional[str]] = {}
        for merged_node, merged_node_ids in merge_list:
            inside_out_handle_to_edges, outside_out_handle_to_edges, in_i, out_i = self._calculate_node_group_meta(
                merged_node_ids, merged_node.id,
                merged_out_to_merged_id)
            merged_id_to_inside_out_to_edges[
                merged_node.id] = inside_out_handle_to_edges
            merged_id_to_outside_out_to_edges[
                merged_node.id] = outside_out_handle_to_edges
            merged_id_to_outside_inp_to_edges[merged_node.id] = out_i
            merged_id_to_inside_inp_to_edges[merged_node.id] = in_i
            # determine new handle id
            handle_unique_name_pool = UniqueNamePool()
            for (node_id,
                 handle), edges in outside_out_handle_to_edges.items():
                first_edge_handle = edges[0].targetHandle
                if handle is None:
                    new_handle = None
                else:
                    new_handle = handle_unique_name_pool(first_edge_handle if first_edge_handle is not None else handle)
                outside_out_to_merged_handle[(node_id, handle, merged_node.id)] = new_handle
            for (node_id,
                 handle), edges in inside_out_handle_to_edges.items():
                if handle is None:
                    new_handle = None
                else:
                    new_handle = handle_unique_name_pool(handle)
                inside_out_to_merged_handle[(node_id, handle)] = new_handle

        # we get all merged handles, now we need to construct new edges

        new_edge_id_to_edges: Dict[str, List[Edge]] = {}
        node_id_handle_pair_set: Set[Tuple[str, str, Optional[str], Optional[str]]] = set()

        if correct_inp_edge_order:
            # iterate all non-merged nodes, add edges first to make sure the edge order 
            # of these nodes are correct
            node_id_to_inp = self.node_id_to_inp_handle_to_edges
            node_id_to_out = self.node_id_to_out_handle_to_edges
            non_merge_edge_id_set: Set[str] = set()
            for node in prev_no_merge_nodes:
                inp_handle_to_edges = node_id_to_inp[node.id]
                for handle, edges in inp_handle_to_edges.items():
                    for edge in edges:
                        if edge.source in node_id_to_merged_id:
                            merged_node_id = node_id_to_merged_id[edge.source]
                            merged_handle_key = (edge.source, edge.sourceHandle)
                            merged_handle = inside_out_to_merged_handle[merged_handle_key]

                            edge_dup_key = (merged_node_id, node.id, merged_handle, edge.targetHandle)
                            if edge_dup_key in node_id_handle_pair_set:
                                continue
                            new_edge_id = edge_name_pool(
                                f"{merged_node_id}-{node.id}")
                            new_edge = Edge(new_edge_id,
                                        source=merged_node_id,
                                        target=node.id,
                                        sourceHandle=merged_handle,
                                        targetHandle=edge.targetHandle)
                            node_id_handle_pair_set.add(edge_dup_key)
                            new_edge_id_to_edges[new_edge.id] = [edge]
                            new_edges.append(new_edge)
                        else:
                            if edge.id not in non_merge_edge_id_set:
                                non_merge_edge_id_set.add(edge.id)
                                new_edges.append(edge)
            for node in prev_no_merge_nodes:
                out_handle_to_edges = node_id_to_out[node.id]
                for handle, edges in out_handle_to_edges.items():
                    for edge in edges:
                        if edge.target not in node_id_to_merged_id:
                            if edge.id not in non_merge_edge_id_set:
                                non_merge_edge_id_set.add(edge.id)
                                new_edges.append(edge)

        for merged_node, merged_node_ids in merge_list:
            outside_node_id_out_handle_to_edges = merged_id_to_outside_out_to_edges[
                merged_node.id]
            inside_node_id_out_handle_to_edges = merged_id_to_inside_out_to_edges[
                merged_node.id]

            outside_node_id_inp_handle_to_edges = merged_id_to_outside_inp_to_edges[
                merged_node.id]
            inside_node_id_inp_handle_to_edges = merged_id_to_inside_inp_to_edges[
                merged_node.id]
            if correct_inp_edge_order:
                outside_node_id_out_handle_to_edges = self._get_order_corrected_outside_out(
                    outside_node_id_out_handle_to_edges, inside_node_id_inp_handle_to_edges)
                inside_node_id_out_handle_to_edges = self._get_order_corrected_inside_out(
                    inside_node_id_out_handle_to_edges, outside_node_id_inp_handle_to_edges)
            for (node_id,
                 handle), edges in outside_node_id_out_handle_to_edges.items():                
                key = (node_id, handle)
                cur_merge_handle = outside_out_to_merged_handle[(key[0], key[1], merged_node.id)]
                edge: Optional[Edge] = None
                if key in merged_out_to_merged_id:
                    # connect to another merged node
                    new_node_id = merged_out_to_merged_id[key]
                    new_handle = inside_out_to_merged_handle[key]
                    edge_dup_key = (new_node_id, merged_node.id, new_handle, cur_merge_handle)
                    if edge_dup_key in node_id_handle_pair_set:
                        continue
                    new_edge_id = edge_name_pool(
                        f"{new_node_id}-{merged_node.id}")
                    edge = Edge(new_edge_id,
                                source=new_node_id,
                                target=merged_node.id,
                                sourceHandle=new_handle,
                                targetHandle=cur_merge_handle)
                    node_id_handle_pair_set.add(edge_dup_key)
                else:
                    # connect to outside original node
                    edge_dup_key = (edges[0].source, merged_node.id, edges[0].sourceHandle, cur_merge_handle)
                    if edge_dup_key in node_id_handle_pair_set:
                        continue
                    new_edge_id = edge_name_pool(f"{edges[0].source}-{merged_node.id}")
                    edge = Edge(new_edge_id,
                                source=edges[0].source,
                                target=merged_node.id,
                                sourceHandle=edges[0].sourceHandle,
                                targetHandle=cur_merge_handle)
                if edge is not None:
                    new_edge_id_to_edges[edge.id] = edges
                    new_edges.append(edge)

            for (node_id,
                 handle), edges in inside_node_id_out_handle_to_edges.items():
                cur_merge_handle = inside_out_to_merged_handle[(node_id, handle)]

                for prev_edge in edges:
                    prev_edge_target = prev_edge.target 
                    edge: Optional[Edge] = None
                    if prev_edge_target in node_id_to_merged_id:
                        new_node_id = node_id_to_merged_id[prev_edge_target]
                        new_handle = outside_out_to_merged_handle[(node_id, handle, new_node_id)]
                        edge_dup_key = (merged_node.id, new_node_id, cur_merge_handle, new_handle)
                        if edge_dup_key in node_id_handle_pair_set:
                            continue
                        new_edge_id = edge_name_pool(
                            f"{merged_node.id}-{new_node_id}")
                        edge = Edge(new_edge_id,
                                    source=merged_node.id,
                                    target=new_node_id,
                                    sourceHandle=cur_merge_handle,
                                    targetHandle=new_handle)
                        node_id_handle_pair_set.add(edge_dup_key)
                    else:
                        # connect to outside original node
                        edge_dup_key = (merged_node.id, prev_edge.target, cur_merge_handle, prev_edge.targetHandle)
                        if edge_dup_key in node_id_handle_pair_set:
                            continue

                        new_edge_id = edge_name_pool(f"{merged_node.id}-{prev_edge.target}")
                        edge = Edge(new_edge_id,
                                    source=merged_node.id,
                                    target=prev_edge.target,
                                    sourceHandle=cur_merge_handle,
                                    targetHandle=prev_edge.targetHandle)
                    new_edge_id_to_edges[new_edge_id] = edges
                    new_edges.append(edge)
        res_internals = FlowInternals()

        res_internals.set_from_nodes_edges(new_nodes, new_edges)
        prev_node_id_to_data: _T_node_data_dict = node_id_to_data
        if node_id_to_data is not None:
            assert merged_data is not None
            assert len(merged_data) == len(merge_list)
            prev_node_id_to_data = cast(_T_node_data_dict, node_id_to_data.copy())
            assert prev_node_id_to_data is not None 
            # remove merged node datas
            for node_id, meta in node_id_to_data.items():
                if node_id not in res_internals.id_to_node:
                    del prev_node_id_to_data[node_id]
            for j in range(len(merge_list)):
                prev_node_id_to_data[merge_list[j][0].id] = merged_data[j]
        prev_edge_id_to_data: _T_edge_data_dict = edge_id_to_data
        if edge_id_to_data is not None:
            prev_edge_id_to_data = cast(_T_edge_data_dict, edge_id_to_data.copy())
            assert prev_edge_id_to_data is not None 
            for edge_id, meta in edge_id_to_data.items():
                if edge_id not in res_internals.id_to_edge:
                    del prev_edge_id_to_data[edge_id]
            for edge_id, prev_edges in new_edge_id_to_edges.items():
                if prev_edges[0].id in edge_id_to_data:
                    prev_edge_id_to_data[edge_id] = edge_id_to_data[
                        prev_edges[0].id]

        return res_internals, new_edge_id_to_edges, prev_node_id_to_data, prev_edge_id_to_data

    def merge_nodes(
        self, merge_list: List[Tuple[Node, List[str]]]
    ) -> Tuple["FlowInternals", Dict[str, List[Edge]]]:
        res = self.merge_nodes_with_data(merge_list)
        return res[0], res[1]

    def create_sub_flow(
        self,
        node_ids: List[str],
        input_type: str = "input",
        output_type: str = "output"
    ) -> Tuple["FlowInternals", List[Tuple[Node, List[Edge]]], List[Tuple[
            Node, List[Edge]]]]:
        node_ids_set = set(node_ids)
        assert len(node_ids_set) == len(node_ids), "node ids must be unique"

        for n in node_ids:
            assert n in self.id_to_node, f"node id {n} not exists"
        inside_out_to_edges, outside_out_to_edges, _, _ = self._calculate_node_group_meta(
            node_ids)
        # outside_out_to_edges: node outputs (top nodes) to subflow
        # outside_inp_to_edges: node inputs (bottom nodes) to subflow
        new_nodes: List[Node] = [self.id_to_node[n] for n in node_ids]
        new_edges: List[Edge] = []
        for edge in self.id_to_edge.values():
            if edge.source in node_ids_set and edge.target in node_ids_set:
                new_edges.append(edge)
                # print(edge.id, edge.source, edge.target)
        node_uniq_pool = UniqueNamePool(init_set=node_ids_set)
        input_node_edge_pairs: List[Tuple[Node, List[Edge]]] = []
        output_node_edge_pairs: List[Tuple[Node, List[Edge]]] = []
        for (node_id,
             handle), edges in outside_out_to_edges.items():
            original_node = self.id_to_node[node_id]
            new_node_id = node_uniq_pool(node_id)
            inp_node = dataclasses.replace(original_node, id=new_node_id, type=input_type, )
            
            if not isinstance(inp_node.data, Undefined):
                inp_node.data = dataclasses.replace(inp_node.data)
                # we copy outside node as input node, so we need to correct handle ids
                inp_node.data.targetHandleIds = undefined
            node_new_edges: List[Edge] = []
            for edge in edges:
                new_edge = dataclasses.replace(edge,
                                               source=inp_node.id,
                                               sourceHandle=handle)
                new_edges.append(new_edge)
                node_new_edges.append(new_edge)
            new_nodes.append(inp_node)
            input_node_edge_pairs.append((inp_node, node_new_edges))
        for (node_id,
             handle), edges in inside_out_to_edges.items():
            original_node = self.id_to_node[node_id]
            new_node_id = node_uniq_pool(node_id)
            out_node = dataclasses.replace(original_node, id=new_node_id, type=output_type)
            if not isinstance(out_node.data, Undefined):
                # we copy inside output node as subflow output node, so we need to correct handle ids
                out_node.data = dataclasses.replace(out_node.data)
                out_node.data.targetHandleIds = out_node.data.sourceHandleIds
                out_node.data.sourceHandleIds = undefined

            node_new_edges: List[Edge] = []
            for edge in edges:
                new_edge = dataclasses.replace(edge,
                                               target=out_node.id,
                                               targetHandle=handle)
                new_edges.append(new_edge)
                node_new_edges.append(new_edge)
            new_nodes.append(out_node)
            output_node_edge_pairs.append((out_node, node_new_edges))
        res_internals = FlowInternals()
        res_internals.set_from_nodes_edges(new_nodes, new_edges)
        return res_internals, input_node_edge_pairs, output_node_edge_pairs


    def create_internals_with_none_handle(self):
        res_internals = FlowInternals()
        new_edges: List[Edge] = []
        for edge in self.id_to_edge.values():
            new_edges.append(dataclasses.replace(edge, sourceHandle=None, targetHandle=None))
        res_internals.set_from_nodes_edges(list(self.id_to_node.values()), new_edges)
        return res_internals

class Flow(MUIContainerBase[FlowProps, MUIComponentType]):
    """
    ## Style
    you can use official css to style the flow component.
    We also provide `react-flow__node__content` to style texts in reactflow default nodes.
    """
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
                             FrontendEventType.ComponentReady.value,
                         ])

        self.event_change = self._create_event_slot(FrontendEventType.Change)
        self.event_selection_change = self._create_event_slot(
            FrontendEventType.FlowSelectionChange,
            lambda x: EventSelection(**x))
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
        # ready to receive component event.
        self.event_component_ready = self._create_event_slot_noarg(
            FrontendEventType.ComponentReady)
        self._internals = FlowInternals()

        self.event_drop = self._create_event_slot(FrontendEventType.Drop)
        self.event_pane_context_menu = self._create_event_slot(
            FrontendEventType.FlowPaneContextMenu)
        self.event_node_context_menu = self._create_event_slot(
            FrontendEventType.FlowNodeContextMenu)
        self._update_graph_data()
        # we must due with delete event because it comes earlier than change event.
        self.event_node_delete.on(self._handle_node_delete)
        self.event_edge_delete.on(self._handle_edge_delete)
        self.event_edge_connection.on(self._handle_new_edge)
        self.event_node_logic_change.on(self._handle_node_logic_change)

        self.set_flow_event_context_creator(
            lambda: enter_flow_ui_context(self))

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
        return self._internals.unique_name_pool_node(id)

    def create_unique_edge_id(self, id: str):
        return self._internals.unique_name_pool_edge(id)

    def _find_comps_in_dataclass(self, child: DataclassType):
        assert isinstance(child, Flow.ChildDef)
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
        self._internals.set_from_nodes_edges(self.nodes, self.edges)
        # TODO detection cycle
        for n in self.nodes:
            if not isinstance(n, Undefined):
                assert n.id in self._internals.id_to_node

    def set_nodes_edges_locally(self, nodes: List[Node], edges: List[Edge]):
        self.childs_complex.nodes = nodes
        self.childs_complex.edges = edges
        self._update_graph_data()

    def get_node_by_id(self, node_id: str):
        return self._internals.id_to_node[node_id]

    def has_node_id(self, node_id: str):
        return node_id in self._internals.id_to_node

    def get_source_nodes(self, node_id: str):
        return [
            self._internals.id_to_node[idh[0]]
            for idh in self._internals.node_id_to_sources[node_id]
        ]

    def get_target_nodes(self, node_id: str):
        return [
            self._internals.id_to_node[idh[0]]
            for idh in self._internals.node_id_to_targets[node_id]
        ]

    def get_source_node_and_handles(self, node_id: str):
        return [(self._internals.id_to_node[idh[0]], idh[1], idh[2])
                for idh in self._internals.node_id_to_sources[node_id]]

    def get_target_node_and_handles(self, node_id: str):
        return [(self._internals.id_to_node[idh[0]], idh[1], idh[2])
                for idh in self._internals.node_id_to_targets[node_id]]

    def get_edges_by_node_and_handle_id(self, node_id: str,
                                        handle_id: Optional[str]):
        inp_content = self._internals.node_id_to_inp_handle_to_edges[node_id]
        out_content = self._internals.node_id_to_out_handle_to_edges[node_id]
        if handle_id in inp_content:
            return inp_content.get(handle_id, [])
        else:
            return out_content.get(handle_id, [])

    def get_all_parent_nodes(self, node_id: str):
        res: List[Node] = []
        accessed: Set[str] = set()
        cur_parents = self.get_source_nodes(node_id)
        while cur_parents:
            res.extend(cur_parents)
            new_parents = []
            for parent in cur_parents:
                if parent.id in accessed:
                    continue
                accessed.add(parent.id)
                new_parents.extend(self.get_source_nodes(parent.id))
            cur_parents = new_parents
        return res

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
            all_connected = self.get_source_nodes(
                cur.id) + self.get_target_nodes(cur.id)
            for n in all_connected:
                stack.append(n)
        return res

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self,
                                           ev,
                                           is_sync=is_sync,
                                           sync_state_after_change=False,
                                           change_status=False)

    def _handle_node_logic_change(self, nodes: List[Any]):
        cur_id_to_comp: Dict[str, Component] = {}
        for n in self.nodes:
            if not isinstance(n.data, Undefined) and not isinstance(
                    n.data.component, Undefined):
                assert n.data.component._flow_uid is not None
                cur_id_to_comp[
                    n.data.component._flow_uid.uid_encoded] = n.data.component
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
        return await self.delete_nodes_by_ids(
            [n["id"] for n in nodes], _internal_dont_send_comp_event=True)

    async def _handle_new_edge(self, data: Dict[str, Any]):
        new_edge = Edge(**data["newEdge"])
        self.childs_complex.edges.append(new_edge)
        self._update_graph_data()

    def _validate_node_ids(self, node_ids: List[str]):
        for node_id in node_ids:
            assert node_id in self._internals.id_to_node, f"node id {node_id} not exists"

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

    async def set_node_style(self, node_id: str, data: Dict[str, Any]):
        self._validate_node_ids([node_id])
        res = {
            "type": FlowControlType.UpdateNodeStyle.value,
            "nodeId": node_id,
            "data": data,
            "override": True,
        }
        return await self.send_and_wait(self.create_comp_event(res))

    async def select_nodes(self, node_ids: List[str]):
        # TODO support controlled select in frontend
        self._validate_node_ids(node_ids)
        node_ids_set = set(node_ids)
        res = {
            "type": FlowControlType.UpdateNodeProps.value,
            "nodeId": node_ids,
            "data": {
                "selected": True
            },
        }
        node_ids_unselected_set = self._internals.id_to_node.keys(
        ) - node_ids_set
        res_unselected = {
            "type": FlowControlType.UpdateNodeProps.value,
            "nodeId": list(node_ids_unselected_set),
            "data": {
                "selected": False
            },
        }
        await self.send_and_wait(self.create_comp_event(res))
        return await self.send_and_wait(self.create_comp_event(res_unselected))

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

    async def do_elk_layout(self,
                              options: Optional[ElkLayoutOptions] = None,
                              fit_view: bool = False):
        if options is None:
            options = ElkLayoutOptions()
        opt_dict = asdict_flatten_field_only_no_undefined(options)
        res = {
            "type": FlowControlType.ElkLayout,
            "graphOptions": opt_dict,
            "fitView": fit_view,
        }
        return await self.send_and_wait(self.create_comp_event(res))

    async def _set_flow_and_do_layout(
            self,
            nodes: List[Node],
            edges: List[Edge],
            algo_type: LayoutAlgoType,
            options: Optional[Any] = None,
            fit_view: bool = False,
            duration: Optional[NumberType] = None):
        """Inorder to handle init static flow layout, you should use this function to set flow and do dagre layout.
        """
        new_layout: Dict[str, Component] = {}
        for node in nodes:
            comp = node.get_component()
            if comp is not None:
                new_layout[node.id] = comp
        self.childs_complex.nodes = nodes
        self.childs_complex.edges = edges
        self._update_graph_data()
        if options is None:
            if algo_type == LayoutAlgoType.Dagre:
                options = DagreLayoutOptions()
            elif algo_type == LayoutAlgoType.Elk:
                options = asdict_flatten_field_only_no_undefined(ElkLayoutOptions())
        else:
            if algo_type == LayoutAlgoType.Elk:
                options = asdict_flatten_field_only_no_undefined(options)

        ev_new_node = {
            "type": FlowControlType.SetFlowAndDagreLayout if algo_type == LayoutAlgoType.Dagre else FlowControlType.SetFlowAndElkLayout,
            "nodes": nodes,
            "edges": edges,
            "graphOptions": options,
            "fitView": fit_view,
        }
        if duration is not None:
            ev_new_node["fitViewDuration"] = duration
        if new_layout:
            return await self.update_childs(
                new_layout,
                update_child_complex=False,
                post_ev_creator=lambda: self.create_comp_event(ev_new_node))
        else:
            return await self.send_and_wait(self.create_comp_event(ev_new_node)
                                            )

    async def set_flow_and_do_dagre_layout(
            self,
            nodes: List[Node],
            edges: List[Edge],
            options: Optional[DagreLayoutOptions] = None,
            fit_view: bool = False,
            duration: Optional[NumberType] = None):
        """Inorder to handle init static flow layout, you should use this function to set flow and do dagre layout.
        """
        return await self._set_flow_and_do_layout(
            nodes, edges, LayoutAlgoType.Dagre, options, fit_view, duration)

    async def set_flow_and_do_elk_layout(
            self,
            nodes: List[Node],
            edges: List[Edge],
            options: Optional[ElkLayoutOptions] = None,
            fit_view: bool = False,
            duration: Optional[NumberType] = None):
        """Inorder to handle init static flow layout, you should use this function to set flow and do dagre layout.
        """
        return await self._set_flow_and_do_layout(
            nodes, edges, LayoutAlgoType.Elk, options, fit_view, duration)

    async def locate_node(self,
                          node_id: str,
                          keep_zoom: Optional[bool] = False,
                          duration: Optional[NumberType] = None):
        return await self.locate_nodes([node_id], keep_zoom, duration)

    async def locate_nodes(self,
                           node_ids: List[str],
                           keep_zoom: Optional[bool] = False,
                           duration: Optional[NumberType] = None):
        res = {
            "type": FlowControlType.LocateNode,
            "nodeId": node_ids,
        }
        if keep_zoom is not None:
            res["fitViewKeepZoom"] = keep_zoom
        if duration is not None:
            res["fitViewDuration"] = duration
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
            all_item_id_to_items = {
                item.id: item
                for item in self.props.paneContextMenuItems
            }
            for item in items:
                if item.id not in all_item_id_to_items:
                    raise ValueError(f"item id {item.id} not exists")
                merge_props_not_undefined(all_item_id_to_items[item.id], item)
            res = {
                "type": FlowControlType.UpdatePaneContextMenuItem,
                "menuItems": items,
            }
            return await self.send_and_wait(self.create_comp_event(res))

    async def update_node_context_menu_items(self, node_id: str,
                                             items: List[MenuItem]):
        """Update node context menu items based on id.
        this function won't add or remove items, only update the existing items.
        """
        node = self._internals.id_to_node[node_id]
        if isinstance(node.data, Undefined):
            return
        if not isinstance(node.data.contextMenuItems, Undefined):
            all_item_id_to_items = {
                item.id: item
                for item in node.data.contextMenuItems
            }
            for item in items:
                if item.id not in all_item_id_to_items:
                    raise ValueError(f"item id {item.id} not exists")
                merge_props_not_undefined(all_item_id_to_items[item.id], item)
            return await self.update_node_data(
                node_id, {"contextMenuItems": node.data.contextMenuItems})

    async def set_node_context_menu_items(self, node_id: str,
                                          items: List[MenuItem]):
        """set node context menu items based on id.
        """
        await self.update_node_data(node_id, {
            "contextMenuItems": items,
        })

    async def add_nodes(self,
                        nodes: List[Node],
                        screen_to_flow: Optional[bool] = None):
        """Add new nodes to the flow.

        Args:
            nodes (Node): nodes to add.
            screen_to_flow (Optional[bool], optional): Whether the node position is in screen coordinates. Defaults to None.
                you should use this when you use position from pane context menu or drag-drop to add a node.
        """

        new_layout: Dict[str, Component] = {}
        for node in nodes:
            assert node.id not in self._internals.id_to_node, f"node id {node.id} already exists"
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
            return await self.update_childs(
                new_layout,
                update_child_complex=False,
                post_ev_creator=lambda: self.create_comp_event(ev_new_node))
        else:
            return await self.send_and_wait(self.create_comp_event(ev_new_node)
                                            )

    async def add_node(self,
                       node: Node,
                       screen_to_flow: Optional[bool] = None):
        """Add a new node to the flow.

        Args:
            node (Node): The node to add.
            screen_to_flow (Optional[bool], optional): Whether the node position is in screen coordinates. Defaults to None.
                you should use this when you use position from pane context menu or drag-drop to add a node.
        """
        await self.add_nodes([node], screen_to_flow)

    async def delete_nodes_by_ids(
            self,
            node_ids: List[str],
            *,
            _internal_dont_send_comp_event: bool = False):
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
                    del_node_id_with_comp, update_child_complex=False)
            else:
                return await self.remove_childs_by_keys(
                    del_node_id_with_comp,
                    update_child_complex=False,
                    post_ev_creator=lambda: self.create_comp_event(ev_del_node)
                )
        else:
            if not _internal_dont_send_comp_event:
                return await self.send_and_wait(
                    self.create_comp_event(ev_del_node))

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

    def __init__(self,
                 type: Literal["source", "target"],
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
        if isinstance(children, Sequence):
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


T = TypeVar("T", bound=Any)
T_edge = TypeVar("T_edge", bound=Any)


@dataclasses.dataclass
class SymbolicImmediate:
    id: str
    source_id: str
    source_handle: Optional[str] = None
    name: Optional[str] = None  # for ui only
    userdata: Optional[Any] = None
    is_input: bool = False


@dataclasses_plain.dataclass
class SymbolicGraphOutput(Generic[T, T_edge]):
    nodes: List[Node]
    edges: List[Edge]
    node_type_map: Union[Undefined,
                         Dict[str, NodeTypeLiteral]] = undefined
    node_id_to_data: Dict[str, T] = dataclasses.field(default_factory=dict)
    edge_id_to_data: Dict[str,
                          T_edge] = dataclasses.field(default_factory=dict)


class SymbolicFlowBuilder(Generic[T, T_edge]):
    """A symbolic flow builder to help you build symbolic flow."""

    def __init__(self, use_multiple_handle_node: bool = False) -> None:
        # self._internals.id_to_node: Dict[str, Node] = {}
        self._id_to_node_data: Dict[str, T] = {}
        # _id_to_edge_data use userdata in SymbolicImmediate, added when a new edge is created
        self._id_to_edge_data: Dict[str, T_edge] = {}

        self._id_to_immedinate: Dict[str, SymbolicImmediate] = {}
        # (edge_id, source_handle, target_handle)
        # if handle is None, means default handle
        self._internals = FlowInternals()

        self._node_id_to_immedinates: Dict[str, List[SymbolicImmediate]] = {}

        self._unique_name_pool_imme = UniqueNamePool()

        self._unique_name_pool = UniqueNamePool()
        self._unique_name_pool_edge = UniqueNamePool()
        self._use_multiple_handle_node = use_multiple_handle_node
        self._input_node_type = "input"
        self._output_node_type = "output"

    def create_input(self,
                     name: Optional[str] = None,
                     id: Optional[str] = None,
                     node_data: Optional[T] = None,
                     default_input_handle: Optional[str] = None):
        if id is not None:
            assert id not in self._id_to_immedinate, f"immedinate id {id} already exists"
        imme_id = self._unique_name_pool_imme(
            id if id is not None else "Immedinate")
        node_id = self._unique_name_pool(imme_id)
        node = self.create_op_node(name if name is not None else "Input", None,
                                   [default_input_handle],
                                   type=self._input_node_type,
                                   node_id=node_id)
        res = SymbolicImmediate(id=imme_id,
                                source_id=node.id,
                                name=name,
                                is_input=True)
        self._id_to_immedinate[imme_id] = res
        self._node_id_to_immedinates[node_id] = [res]
        if node_data is not None:
            self._id_to_node_data[node_id] = node_data
        return res, node

    def get_immedinate_node(self, immedinate: SymbolicImmediate):
        source_id = immedinate.source_id
        return self._internals.id_to_node[source_id]

    def create_op_node(self,
                       name: str,
                       inp_handles: Optional[List[Optional[str]]],
                       out_handles: List[Optional[str]],
                       type: Optional[str] = None,
                       node_id: Optional[str] = None,
                       node_data: Optional[T] = None):
        if node_id is None:
            node_id = self._unique_name_pool(name)
        else:
            assert node_id not in self._internals.id_to_node, f"node id {node_id} already exists"
        assert node_id not in self._internals.node_id_to_inp_handle_to_edges
        assert node_id not in self._internals.node_id_to_out_handle_to_edges
        self._internals.node_id_to_inp_handle_to_edges[node_id] = {}
        self._internals.node_id_to_out_handle_to_edges[node_id] = {}
        node_data_base = NodeData(label=name)
        node = Node(id=node_id, data=node_data_base)
        if type is not None:
            node.type = type
        if self._use_multiple_handle_node:
            node_data_base.sourceHandleIds = out_handles
            if inp_handles is not None:
                node_data_base.targetHandleIds = inp_handles
        self._internals.id_to_node[node_id] = node
        # fill _node_id_to_out_handle_to_edges and _node_id_to_inp_handle_to_edges
        if inp_handles is not None:
            for handle in inp_handles:
                if handle not in self._internals.node_id_to_inp_handle_to_edges[
                        node_id]:
                    self._internals.node_id_to_inp_handle_to_edges[node_id][
                        handle] = []
        for handle in out_handles:
            if handle not in self._internals.node_id_to_out_handle_to_edges[
                    node_id]:
                self._internals.node_id_to_out_handle_to_edges[node_id][
                    handle] = []
        self._node_id_to_immedinates[node_id] = []
        self._internals.node_id_to_sources[node_id] = []
        self._internals.node_id_to_targets[node_id] = []
        if node_data is not None:
            self._id_to_node_data[node_id] = node_data
        return node

    def call_op_node(
        self, op_node: Node,
        op_inp_handle_to_imme: Dict[Optional[str],
                                    Union[SymbolicImmediate,
                                          List[SymbolicImmediate]]]):
        assert op_node.id in self._internals.id_to_node
        inp_handle_to_edges = self._internals.node_id_to_inp_handle_to_edges[
            op_node.id]
        out_handle_to_edges = self._internals.node_id_to_out_handle_to_edges[
            op_node.id]
        for handle in op_inp_handle_to_imme.keys():
            assert handle in inp_handle_to_edges
        # for handle in outputs:
        #     assert handle in out_handle_to_edges
        # connect source node to op node
        for handle, immes in op_inp_handle_to_imme.items():
            if not isinstance(immes, list):
                immes = [immes]
            for imme in immes:
                edge_id = self._unique_name_pool_edge(
                    f"{imme.source_id}=>{op_node.id}")
                edge = Edge(id=edge_id,
                            source=imme.source_id,
                            target=op_node.id,
                            sourceHandle=imme.source_handle,
                            targetHandle=handle)
                if imme.userdata is not None:
                    self._id_to_edge_data[edge_id] = imme.userdata
                self._internals.node_id_to_inp_handle_to_edges[
                    op_node.id][handle].append(edge)
                self._internals.node_id_to_sources[op_node.id].append(
                    (edge_id, imme.source_handle, handle))
                self._internals.node_id_to_out_handle_to_edges[imme.source_id][
                    imme.source_handle].append(edge)
                # add to source node metas
                self._internals.node_id_to_targets[imme.source_id].append(
                    (edge_id, handle, imme.source_handle))
        res_immes: List[SymbolicImmediate] = []
        # create output immedinate node
        op_node_output_handles = list(out_handle_to_edges.keys())
        for handle in op_node_output_handles:
            imme_id = self._unique_name_pool_imme(f"{op_node.id}-{handle}")
            imme = SymbolicImmediate(id=imme_id,
                                     name=handle,
                                     source_id=op_node.id,
                                     source_handle=handle)
            self._id_to_immedinate[imme_id] = imme
            self._node_id_to_immedinates[op_node.id].append(imme)
            res_immes.append(imme)
        return res_immes

    def get_immedinate_by_id(self, id: str):
        return self._id_to_immedinate[id]

    def change_immedinate_id(self, imme: SymbolicImmediate, new_id: str):
        if imme.id == new_id:
            return imme
        assert new_id not in self._id_to_immedinate, f"immedinate id {new_id} already exists"
        del self._id_to_immedinate[imme.id]
        imme = dataclasses.replace(imme, id=new_id)
        self._id_to_immedinate[new_id] = imme
        return imme

    def is_node_input(self, node_id: str):
        node_immes = self._node_id_to_immedinates[node_id]
        return len(node_immes) > 0 and node_immes[0].is_input

    def build_detached_flow(self,
                            out_immedinates: Sequence[SymbolicImmediate],
                            disable_handle: bool = True,
                            out_node_datas: Optional[List[T]] = None,
                            default_out_handle: Optional[str] = "out"):
        """Build flow with different config without modifying 
        the current symbolic flow states.
        Args:
            out_immedinates (Sequence[SymbolicImmediate]): output immedinates of flow graph.
            disable_handle (bool, optional): disable all handle logic and set all handle to None. used if you don't provide 
                custom node ui. Defaults to True.
        Returns:
            node_and_edges: nodes and edges of the flow graph.
        """
        # validate inputs
        for imme in out_immedinates:
            assert imme.id in self._id_to_immedinate, f"immedinate id {imme.id} not exists"
        if out_node_datas is not None:
            assert len(out_node_datas) == len(out_immedinates)
        # create output nodes
        out_nodes = []
        out_edges = []
        node_umap_copy = self._unique_name_pool.copy()
        edge_umap_copy = self._unique_name_pool_edge.copy()
        for i, imme in enumerate(out_immedinates):
            imme_id = imme.name if imme.name is not None else imme.id
            node_id = node_umap_copy(imme_id)
            out_node_data = NodeData(label=imme_id)
            node = Node(id=node_id,
                        data=out_node_data,
                        type=self._output_node_type)
            out_nodes.append(node)
            if out_node_datas is not None:
                self._id_to_node_data[node_id] = out_node_datas[i]
            if self._use_multiple_handle_node:
                out_node_data.targetHandleIds = [default_out_handle]
            # connect to immedinate
            edge_id = edge_umap_copy(f"{imme.source_id}=>{node_id}")
            edge = Edge(id=edge_id,
                        source=imme.source_id,
                        target=node_id,
                        sourceHandle=imme.source_handle,
                        targetHandle=default_out_handle)
            if imme.userdata is not None:
                self._id_to_edge_data[edge_id] = imme.userdata
            if disable_handle:
                edge.sourceHandle = None
                edge.targetHandle = None
            out_edges.append(edge)
        # get nodes and edges
        all_nodes = list(self._internals.id_to_node.values())
        all_edges: List[Edge] = []
        for node_id, handle_to_edges in self._internals.node_id_to_inp_handle_to_edges.items(
        ):
            for edges in handle_to_edges.values():
                if disable_handle:
                    for edge in edges:
                        edge = dataclasses.replace(edge,
                                                   sourceHandle=None,
                                                   targetHandle=None)
                        all_edges.append(edge)
                else:
                    all_edges.extend(edges)
        # remap node types if you use type to store op meta such as
        # Conv2d, we will remap these types to "default" if you
        # don't provide custom node ui.

        default_node_types = set(
            ["app", "input", "default", "output", "group", "appTemplate"])
        node_type_map: Dict[str, NodeTypeLiteral] = {}
        for node in all_nodes:
            if not isinstance(node.type, Undefined):
                if node.type not in default_node_types:
                    is_inp = self.is_node_input(node.id)
                    node_type_map[node.type] = "input" if is_inp else "default"
        if node_type_map:
            node_type_map_res = node_type_map
        else:
            node_type_map_res = undefined
        return SymbolicGraphOutput(all_nodes + out_nodes,
                                   all_edges + out_edges, node_type_map_res,
                                   self._id_to_node_data.copy(),
                                   self._id_to_edge_data.copy())
