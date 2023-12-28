# Copyright 2022 Yan Yan
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

import asyncio
import base64
import enum
import tensorpc.core.dataclass_dispatch as dataclasses
from typing import (TYPE_CHECKING, Any, Callable, Coroutine, Dict, Iterable,
                    List, Optional, Tuple, Type, TypeVar, Union)

from tensorpc.core.asynctools import cancel_task
from tensorpc.flow.flowapp.components.common import (handle_standard_event)
from typing_extensions import Literal, TypeAlias

from ..core import (AppEvent, AppEventType, BasicProps, Component,
                    ContainerBase, FrontendEventType, NumberType, T_child, TaskLoopEvent, UIEvent,
                    UIRunStatus, UIType, Undefined, undefined,
                    as_dict_no_undefined)
from .mui import MUIComponentBase


@dataclasses.dataclass
class Font:
    size: Union[Undefined, NumberType] = undefined
    family: Union[Undefined, str] = undefined
    color: Union[Undefined, str] = undefined


@dataclasses.dataclass
class Marker:
    color: Union[Undefined, str, List[str]] = undefined
    size: Union[Undefined, NumberType, List[NumberType]] = undefined
    opacity: Union[Undefined, NumberType, List[NumberType]] = undefined


@dataclasses.dataclass
class Line:
    color: Union[Undefined, str] = undefined
    width: Union[Undefined, NumberType] = undefined
    shape: Union[Undefined, Literal["linear", "spline", "vhv", "hvh", "vh",
                                    "hv"]] = undefined
    dash: Union[Undefined, Literal["solid", "dashdot", "dot"]] = undefined


@dataclasses.dataclass
class Trace:
    x: Union[Undefined, List[Union[NumberType, str]]] = undefined
    y: Union[Undefined, List[Union[NumberType, str]]] = undefined
    z: Union[Undefined, List[Union[NumberType, str]]] = undefined
    text: Union[Undefined, str, List[str]] = undefined
    type: Union[Undefined, Literal["scatter", "scattergl", "bar", "image"]] = undefined
    mode: Union[Undefined, Literal["markers", "lines", "lines+markers"]] = undefined
    visible: Union[Undefined, bool] = undefined
    name: Union[Undefined, str] = undefined
    line: Union[Undefined, Line] = undefined
    marker: Union[Undefined, Marker] = undefined
    values: Union[Undefined, List[Union[NumberType, str]]] = undefined
    labels: Union[Undefined, List[Union[NumberType, str]]] = undefined

@dataclasses.dataclass
class Margin:
    l: Union[Undefined, NumberType] = undefined
    r: Union[Undefined, NumberType] = undefined
    t: Union[Undefined, NumberType] = undefined
    b: Union[Undefined, NumberType] = undefined
    pad: Union[Undefined, NumberType] = undefined
    autoexpand: Union[Undefined, bool] = undefined


@dataclasses.dataclass
class Annotation:
    xref: Union[Undefined, Literal["paper", "container"]] = undefined
    yref: Union[Undefined, Literal["paper", "container"]] = undefined
    x: Union[Undefined, NumberType] = undefined
    y: Union[Undefined, NumberType] = undefined
    xanchor: Union[Undefined, Literal["left", "auto", "right",
                                      "center"]] = undefined
    yanchor: Union[Undefined, Literal["top", "auto", "middle",
                                      "bottom"]] = undefined
    text: Union[Undefined, str] = undefined
    font: Union[Undefined, Font] = undefined
    showarrow: Union[Undefined, bool] = undefined


@dataclasses.dataclass
class Legend:
    x: Union[Undefined, NumberType] = undefined
    y: Union[Undefined, NumberType] = undefined
    xanchor: Union[Undefined, Literal["left", "auto", "right",
                                      "center"]] = undefined
    yanchor: Union[Undefined, Literal["top", "auto", "middle",
                                      "bottom"]] = undefined
    title: Union[Undefined, str] = undefined
    showgrid: Union[Undefined, bool] = undefined
    zeroline: Union[Undefined, bool] = undefined
    showline: Union[Undefined, bool] = undefined
    font: Union[Undefined, Font] = undefined


@dataclasses.dataclass
class Axis:
    title: Union[Undefined, str] = undefined
    showgrid: Union[Undefined, bool] = undefined
    zeroline: Union[Undefined, bool] = undefined
    showline: Union[Undefined, bool] = undefined
    range: Union[Undefined, Tuple[NumberType, NumberType]] = undefined
    autorange: Union[Undefined, bool] = undefined
    showticklabels: Union[Undefined, bool] = undefined
    linecolor: Union[Undefined, str] = undefined
    linewidth: Union[Undefined, NumberType] = undefined
    autotick: Union[Undefined, bool] = undefined
    ticks: Union[Undefined, Literal["outside"]] = undefined
    tickcolor: Union[Undefined, str] = undefined
    tickwidth: Union[Undefined, NumberType] = undefined
    ticklen: Union[Undefined, NumberType] = undefined
    tickfont: Union[Undefined, Font] = undefined
    automargin: Union[Undefined, bool] = undefined


@dataclasses.dataclass
class Layout:
    title: Union[Undefined, str] = undefined
    width: Union[Undefined, NumberType] = undefined
    height: Union[Undefined, NumberType] = undefined
    showlegend: Union[Undefined, bool] = undefined
    autosize: Union[Undefined, bool] = undefined
    margin: Union[Undefined, Margin] = undefined
    annotations: Union[Undefined, List[Annotation]] = undefined
    font: Union[Undefined, Font] = undefined
    hovermode: Union[Undefined, Literal["x", "y", "closest", "x unified",
                                        "y unified", False]] = undefined
    clickmode: Union[Undefined, Literal["event", "select", "event+select",
                                        "none", False]] = undefined
    dragmode: Union[Undefined,
                    Literal["zoom", "pan", "select", "lasso", "drawclosedpath",
                            "drawopenpath", "drawline", "drawrect",
                            "drawcircle", "orbit", "turntable",
                            False]] = undefined
    xaxis: Union[Undefined, Axis] = undefined
    yaxis: Union[Undefined, Axis] = undefined


@dataclasses.dataclass
class PlotlyProps(BasicProps):
    data: List[Trace] = dataclasses.field(default_factory=list)
    layout: Layout = dataclasses.field(default_factory=Layout)


class ChartControlType(enum.IntEnum):
    ExtendData = 0
    ClearData = 1
    UpdateTrace = 2

@dataclasses.dataclass
class PlotlyTraceDataUpdate:
    traceUpdateIndex: int 
    traceUpdateMaximumNumber: Union[Undefined, int] = undefined
    x: Union[Undefined, List[Union[NumberType, str]]] = undefined
    y: Union[Undefined, List[Union[NumberType, str]]] = undefined
    z: Union[Undefined, List[Union[NumberType, str]]] = undefined
    text: Union[Undefined, List[str]] = undefined
    markerSize: Union[Undefined, List[NumberType]] = undefined
    markerColor: Union[Undefined, List[str]] = undefined
    markerOpacity: Union[Undefined, List[NumberType]] = undefined
    values: Union[Undefined, List[Union[NumberType, str]]] = undefined
    labels: Union[Undefined, List[Union[NumberType, str]]] = undefined

class Plotly(MUIComponentBase[PlotlyProps]):
    """see https://plotly.com/javascript/ for documentation"""

    def __init__(self) -> None:
        super().__init__(UIType.Plotly, PlotlyProps, allowed_events=[FrontendEventType.Click])
        self.event_click = self._create_event_slot(FrontendEventType.Click)

    async def show_raw(self, data: List[Trace], layout: Layout):
        self.props.data = data
        self.props.layout = layout
        await self.put_app_event(self.update_event(data=data, layout=layout))

    def get_sync_props(self) -> Dict[str, Any]:
        res = super().get_sync_props()
        res["data"] = self.props.data
        res["layout"] = self.props.layout
        return res

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    @staticmethod
    def layout_no_margin():
        return Layout(autosize=True,
                      margin=Margin(l=0, r=0, b=0, t=0),
                      xaxis=Axis(automargin=True),
                      yaxis=Axis(automargin=True))

    async def extend_data(self, updates: List[PlotlyTraceDataUpdate]):
        ev = self.create_comp_event({
            "type": ChartControlType.ExtendData.value,
            "updates": [as_dict_no_undefined(u) for u in updates],
        })
        data = self.props.data 
        for update in updates:
            if update.traceUpdateIndex >= len(data) or update.traceUpdateIndex < 0:
                continue 
            trace = data[update.traceUpdateIndex]
            if not isinstance(update.x, Undefined) and not isinstance(trace.x, Undefined):
                trace.x.extend(update.x)
            if not isinstance(update.y, Undefined) and not isinstance(trace.y, Undefined):
                trace.y.extend(update.y)
            if not isinstance(update.z, Undefined) and not isinstance(trace.z, Undefined):
                trace.z.extend(update.z)
            if not isinstance(update.text, Undefined) and not isinstance(trace.text, Undefined) and isinstance(trace.text, list):
                trace.text.extend(update.text)
            if not isinstance(update.markerSize, Undefined) and not isinstance(trace.marker, Undefined) and isinstance(trace.marker, Marker):
                if isinstance(trace.marker.size, list):
                    trace.marker.size.extend(update.markerSize)
            if not isinstance(update.markerColor, Undefined) and not isinstance(trace.marker, Undefined) and isinstance(trace.marker, Marker):
                if isinstance(trace.marker.color, list):
                    trace.marker.color.extend(update.markerColor)
            if not isinstance(update.markerOpacity, Undefined) and not isinstance(trace.marker, Undefined) and isinstance(trace.marker, Marker):
                if isinstance(trace.marker.opacity, list):
                    trace.marker.opacity.extend(update.markerOpacity)
            if not isinstance(update.values, Undefined) and not isinstance(trace.values, Undefined):
                trace.values.extend(update.values)
            if not isinstance(update.labels, Undefined) and not isinstance(trace.labels, Undefined):
                trace.labels.extend(update.labels)
        await self.send_and_wait(ev)
