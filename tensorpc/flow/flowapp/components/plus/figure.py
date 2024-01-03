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

from typing import Any, Dict, List, Optional, Tuple, Union
from typing_extensions import Literal
from ..mui import FlexBox, Component, NumberType, AppEvent
from .. import plotly
from ...core import DataClassWithUndefined, NumberType, Undefined, ValueType, undefined, as_dict_no_undefined
import dataclasses


def as_dict_no_undefined_first_level(obj: Any):
    res: Dict[str, Any] = {}
    for field in dataclasses.fields(obj):
        val = getattr(obj, field.name)
        if not isinstance(val, Undefined):
            res[field.name] = val
    return res


class HomogeneousMetricFigure(FlexBox):
    """multiple figures with same layout, and same number of data trace,
    only data value / type varies.
    Often be used in metrics.
    """

    def __init__(self, width: int, height: int):
        super().__init__()
        self.props.flexFlow = "row wrap"
        self.base_layout = plotly.Layout(width=width, height=height)
        self.traces: List[plotly.Trace] = []
        self._trace_dict: Dict[str, Dict[str, plotly.Trace]] = {}

    async def update_figures(self, keys_to_title: Dict[str, str]):
        if not keys_to_title:
            return
        plots: Dict[str, Component] = {}
        for k, v in keys_to_title.items():
            new_layout = dataclasses.replace(self.base_layout, title=v)
            plots[k] = plotly.Plotly().prop(data=[], layout=new_layout)
        await self.update_childs(plots)

    async def clear_figures(self):
        await self.set_new_layout({})

    async def set_traces_visible(self, trace_ids: List[str], visible: bool):
        ev = AppEvent("", {})
        for trace_id in trace_ids:
            assert trace_id in self._trace_dict, "your trace id not exists"
            for k in self.props.childs:
                trace = self._trace_dict[trace_id][k]
                trace.visible = visible
                plot = self[k]
                assert isinstance(plot, plotly.Plotly)
                ev += plot.update_event(data=plot.props.data)
        return await self.send_and_wait(ev)

    async def set_trace_visible(self, trace_id: str, visible: bool):
        return await self.set_traces_visible([trace_id], visible)

    async def update_metric(self, x: NumberType, trace_id: str, color: str,
                            metric_dict: Dict[str, NumberType]):
        metric_dict_once = {x: [v] for x, v in metric_dict.items()}
        await self.update_metrics([x], trace_id, color, metric_dict_once)

    async def update_metrics(self, x: List[NumberType], trace_id: str,
                             color: str, metric_dict: Dict[str,
                                                           List[NumberType]]):
        figure_to_update: Dict[str, str] = {}
        for k in metric_dict.keys():
            if k not in self.props.childs:
                figure_to_update[k] = k
        await self.update_figures(figure_to_update)
        ev = AppEvent("", {})
        for k, v in metric_dict.items():
            plot = self[k]
            assert isinstance(plot, plotly.Plotly)
            if trace_id not in self._trace_dict:
                self._trace_dict[trace_id] = {}
            if k not in self._trace_dict[trace_id]:
                new_trace = plotly.Trace([], [], [],
                                         "scatter",
                                         "lines",
                                         line=plotly.Line(color=color),
                                         name=trace_id)
                plot.props.data.append(new_trace)
                self._trace_dict[trace_id][k] = new_trace
            trace = self._trace_dict[trace_id][k]
            trace.x.extend(x)
            trace.y.extend(v)
            ev += plot.update_event(data=plot.props.data)
        return await self.send_and_wait(ev)

    async def update_plotly_layout(self, layout: plotly.Layout):
        """merge new key to existed base_layout, only support depth-1 merge"""
        layout_dict = as_dict_no_undefined_first_level(layout)
        self.base_layout = dataclasses.replace(self.base_layout, **layout_dict)
        ev = AppEvent("", {})
        for k in self.props.childs:
            plot = self[k]
            assert isinstance(plot, plotly.Plotly)
            new_layout = dataclasses.replace(plot.props.layout, **layout_dict)
            plot.props.layout = new_layout
            ev += plot.update_event(layout=new_layout)
        return await self.send_and_wait(ev)
