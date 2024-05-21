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

import enum
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
from .mui import ContainerBaseProps, LayoutType, MUIContainerBase, MUIComponentType, Theme, MUIComponentBaseProps, MUIComponentBase, ValueType


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
    selectedBoxSxProps: Union[Undefined, Dict[str, Any]] = undefined
    data: Union[Undefined, Any] = undefined
    label: Union[Undefined, str] = undefined
    
@dataclasses.dataclass
class Node:
    id: str
    data: Union[Undefined, NodeData] = undefined
    type: Union[Undefined, Literal["app", "appTemplate", "input", "default",
                                   "output", "group", "annotation"]] = undefined
    position: XYPosition = dataclasses.field(default_factory=lambda: XYPosition(0, 0))
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
    parentId: Union[Undefined, str] = undefined
    focusable: Union[Undefined, bool] = undefined
    extent: Union[Undefined, Literal["parent"], Tuple[Tuple[NumberType, NumberType], Tuple[NumberType, NumberType]]] = undefined

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
    ranker: Union[Undefined, Literal["network-simplex", "tight-tree", "longest-path"]] = undefined

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
        # TODO detection cycle
        for n in self.nodes:
            if not isinstance(n, Undefined):
                assert n.id in self._id_to_node

    def get_source_nodes(self, node_id: str):
        return [self._id_to_node[id] for id in self._node_id_to_sources[node_id]]

    def get_target_nodes(self, node_id: str):
        return [self._id_to_node[id] for id in self._node_id_to_targets[node_id]]

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        print(ev.type)
        return await handle_standard_event(self, ev, is_sync=is_sync)

    def state_change_callback(self, value: dict, type: ValueType = FrontendEventType.Change.value):
        if "nodes" in value:
            cur_id_to_comp: Dict[str, Component] = {}
            for n in self.nodes:
                if not isinstance(n.data, Undefined) and not isinstance(n.data.component, Undefined):
                    assert n.data.component._flow_uid is not None 
                    cur_id_to_comp[n.data.component._flow_uid.uid_encoded] = n.data.component
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

    async def do_dagre_layout(self, options: Optional[DagreLayoutOptions] = None, fit_view: bool = False):
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

@dataclasses.dataclass
class HandleProps(MUIComponentBaseProps):
    type: Union[Literal["source", "target"], Undefined] = undefined
    position: Union[Literal["left", "top", "right", "bottom"], Undefined] = undefined
    isConnectable: Union[bool, Undefined] = undefined
    style: Union[Undefined, Any] = undefined
    id: Union[Undefined, str] = undefined


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

@dataclasses.dataclass
class NodeColorMap:
    app: Union[Undefined, str] = undefined
    input: Union[Undefined, str] = undefined
    default: Union[Undefined, str] = undefined
    output: Union[Undefined, str] = undefined
    group: Union[Undefined, str] = undefined
    annotation: Union[Undefined, str] = undefined


@dataclasses.dataclass
class MiniMapProps(MUIComponentBaseProps):
    nodeColorMap: Union[Undefined, NodeColorMap] = undefined
    nodeStrokeColorMap: Union[Undefined, NodeColorMap] = undefined
    nodeBorderRadius: Union[Undefined, int] = undefined
    nodeStrokeWidth: Union[Undefined, int] = undefined
    maskColor: Union[Undefined, str] = undefined
    maskStrokeColor: Union[Undefined, str] = undefined
    maskStrokeWidth: Union[Undefined, int] = undefined
    position: Union[Undefined, Literal["top-left", "top-right", "bottom-left", "bottom-right", "top-center", "bottom-center"]] = undefined
    pannable: Union[Undefined, bool] = undefined
    zoomable: Union[Undefined, bool] = undefined
    inversePan: Union[Undefined, bool] = undefined
    zoomStep: Union[Undefined, int] = undefined
    offsetScale: Union[Undefined, int] = undefined


class MiniMap(MUIComponentBase[MiniMapProps]):

    def __init__(self) -> None:
        super().__init__(
            UIType.FlowMiniMap, MiniMapProps, [])

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class ControlsProps(MUIComponentBaseProps):
    position: Union[Undefined, Literal["top-left", "top-right", "bottom-left", "bottom-right", "top-center", "bottom-center"]] = undefined
    showZoom: Union[Undefined, bool] = undefined
    showFitView: Union[Undefined, bool] = undefined
    showInteractive: Union[Undefined, bool] = undefined
    fitViewOptions: Union[Undefined, FlowFitViewOptions] = undefined

class Controls(MUIComponentBase[ControlsProps]):

    def __init__(self) -> None:
        super().__init__(
            UIType.FlowControls, ControlsProps, [])

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class BackgroundProps(MUIComponentBaseProps):
    id: Union[Undefined, str] = undefined
    variant: Union[Undefined, Literal["lines", "dots", "cross"]] = undefined
    color: Union[Undefined, str] = undefined
    gap: Union[Undefined, NumberType] = undefined
    size: Union[Undefined, NumberType] = undefined
    offset: Union[Undefined, NumberType] = undefined
    lineWidth: Union[Undefined, NumberType] = undefined

class Background(MUIComponentBase[BackgroundProps]):

    def __init__(self) -> None:
        super().__init__(
            UIType.FlowBackground, BackgroundProps, [])

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

@dataclasses.dataclass
class NodeResizerProps(MUIComponentBaseProps):
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
        super().__init__(
            UIType.FlowNodeResizer, NodeResizerProps, [])

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
    position: Union[Undefined, Literal["top", "bottom", "left", "right"]] = undefined
    isVisible: Union[Undefined, bool] = undefined
    offset: Union[Undefined, NumberType] = undefined
    align: Union[Undefined, Literal["center", "start", "end"]] = undefined


class NodeToolbar(MUIContainerBase[NodeToolbarProps, MUIComponentType]):

    def __init__(
            self,
            children: LayoutType) -> None:
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

