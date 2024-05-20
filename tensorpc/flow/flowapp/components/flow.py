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

import tensorpc.core.dataclass_dispatch as dataclasses
from typing import (TYPE_CHECKING, Any, Callable, Coroutine, Dict, Iterable,
                    List, Optional, Tuple, Type, TypeVar, Union)

from tensorpc.core.asynctools import cancel_task
from tensorpc.flow.flowapp.appcore import Event
from tensorpc.flow.flowapp.components.common import (handle_standard_event)
from typing_extensions import Literal, TypeAlias

from ..core import (AppEvent, AppEventType, BasicProps, Component,
                    FrontendEventType, NumberType, UIType, Undefined,
                    undefined)
from .mui import ContainerBaseProps, MUIContainerBase, MUIComponentType, Theme, MUIComponentBaseProps, MUIComponentBase, ValueType


@dataclasses.dataclass
class FlowFitViewOptions:
    minZoom: Union[Undefined, int] = undefined
    maxZoom: Union[Undefined, int] = undefined


@dataclasses.dataclass
class FlowProps(ContainerBaseProps):
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
    debounce: Union[Undefined, NumberType] = undefined

@dataclasses.dataclass
class XYPosition:
    x: NumberType
    y: NumberType


@dataclasses.dataclass
class NodeData:
    component: Union[Undefined, Component] = undefined
    selectedTheme: Union[Undefined, Theme] = undefined
    data: Union[Undefined, Any] = undefined
    label: Union[Undefined, str] = undefined
    
@dataclasses.dataclass
class Node:
    id: str
    data: NodeData
    type: Union[Undefined, Literal["app", "appTemplate", "input", "default",
                                   "output", "group"]] = undefined
    position: Union[Undefined, XYPosition] = undefined
    style: Union[Undefined, Any] = undefined
    className: Union[Undefined, str] = undefined
    dragHandle: Union[Undefined, bool] = undefined
    hidden: Union[Undefined, bool] = undefined
    draggable: Union[Undefined, bool] = undefined
    selectable: Union[Undefined, bool] = undefined
    connectable: Union[Undefined, bool] = undefined
    deletable: Union[Undefined, bool] = undefined
    width: Union[Undefined, NumberType] = undefined
    height: Union[Undefined, NumberType] = undefined
    # parentNode
    focusable: Union[Undefined, bool] = undefined


@dataclasses.dataclass
class Edge:
    id: str
    source: str
    target: str
    type: Union[Undefined, Literal["default", "straight", "step",
                                   "smoothstep"]] = undefined
    style: Union[Undefined, Any] = undefined
    animated: Union[Undefined, bool] = undefined
    hidden: Union[Undefined, bool] = undefined
    focusable: Union[Undefined, bool] = undefined


@dataclasses.dataclass
class _NodesHelper:
    nodes: List[Node]


@dataclasses.dataclass
class _EdgesHelper:
    edges: List[Edge]


class Flow(MUIContainerBase[FlowProps, MUIComponentType]):

    @dataclasses.dataclass
    class ChildDef:
        nodes: List[Node]
        edges: List[Edge]
        extraChilds: Union[Undefined, List[Component]] = undefined
        componentTemplate: Union[Undefined, str] = undefined

    def __init__(
            self,
            nodes: List[Node],
            edges: List[Edge],
            extra_childs: Union[Undefined, List[Component]] = undefined,
            component_template: Union[Undefined, str] = undefined) -> None:
        super().__init__(UIType.Flow,
                         FlowProps,
                         Flow.ChildDef(nodes, edges, extra_childs,
                                       component_template),
                         allowed_events=[FrontendEventType.FlowSelectionChange.value])
        self.event_selection_change = self._create_event_slot(
            FrontendEventType.FlowSelectionChange)
        self._update_graph_data()

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

    def _update_graph_data(self):
        # node id must unique
        self._id_to_node = {node.id: node for node in self.nodes}
        assert len(self._id_to_node) == len(self.nodes)
        self._id_to_edge = {edge.id: edge for edge in self.edges}
        # edge id must unique
        assert len(self._id_to_edge) == len(self.edges)
        self._source_id_to_edge = {edge.source: edge for edge in self.edges}
        self._target_id_to_edge = {edge.target: edge for edge in self.edges}
        self._node_id_to_sources = {node.id: [] for node in self.nodes}
        self._node_id_to_targets = {node.id: [] for node in self.nodes}
        for edge in self.edges:
            self._node_id_to_sources[edge.source].append(self._id_to_node[edge.target].id)
            self._node_id_to_targets[edge.target].append(self._id_to_node[edge.source].id)

    def get_source_nodes(self, node_id: str):
        return [self._id_to_node[id] for id in self._node_id_to_sources[node_id]]

    def get_target_nodes(self, node_id: str):
        return [self._id_to_node[id] for id in self._node_id_to_targets[node_id]]

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)

    def state_change_callback(self, value: dict, type: ValueType = FrontendEventType.Change.value):
        if "nodes" in value:
            self.childs_complex.nodes = _NodesHelper(value["nodes"]).nodes
        if "edges" in value:
            self.childs_complex.edges = _EdgesHelper(value["edges"]).edges
        self._update_graph_data()

@dataclasses.dataclass
class HandleProps(MUIComponentBaseProps):
    type: Union[Literal["source", "target"], Undefined] = undefined
    position: Union[Literal["left", "top", "right", "bottom"], Undefined] = undefined
    isConnectable: Union[bool, Undefined] = undefined
    style: Union[Undefined, Any] = undefined


class Handle(MUIComponentBase[HandleProps]):

    def __init__(self, type: Literal["source", "target"], position: Literal["left", "top", "right", "bottom"]) -> None:
        super().__init__(
            UIType.Handle, HandleProps,[])
        self.prop(type=type, position=position)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)
