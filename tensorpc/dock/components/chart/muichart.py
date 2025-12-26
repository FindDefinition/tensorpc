import asyncio
import base64
from collections.abc import Sequence
import enum
from tensorpc.core import core_io
import tensorpc.core.dataclass_dispatch as dataclasses
from typing import (TYPE_CHECKING, Any, Callable, Coroutine, Dict, Iterable,
                    List, Optional, Tuple, Type, TypeVar, Union)

from tensorpc.core.asynctools import cancel_task
from tensorpc.dock.core.appcore import Event
from tensorpc.dock.core.common import (handle_standard_event)
from typing_extensions import Literal, TypeAlias

from tensorpc.dock.core.component import (AppEvent, AppEventType, BasicProps, Component,
                    ContainerBase, FrontendEventType, NumberType, T_child,
                    TaskLoopEvent, UIEvent, UIRunStatus, UIType, Undefined,
                    undefined, as_dict_no_undefined)
from tensorpc.dock.components.mui import IFrame, MUIComponentBase, ValueType

@dataclasses.dataclass
class ScatterValueType:
    x: NumberType 
    y: NumberType
    z: Union[Any, Undefined] = undefined
    id: Union[str, NumberType, Undefined] = undefined

@dataclasses.dataclass
class ScatterDatasetKeyType:
    x: str 
    y: str 
    z: Union[str, Undefined] = undefined
    id: Union[str, Undefined] = undefined

@dataclasses.dataclass
class CommonSeriesType:
    id: Union[str, NumberType, Undefined] = undefined
    color: Union[str, Undefined] = undefined
    labelMarkType: Union[Literal["circle", "square", "square"], Undefined] = undefined
    xAxisId: Union[str, Undefined] = undefined
    yAxisId: Union[str, Undefined] = undefined
    stack: Union[str, Undefined] = undefined
    stackOffset: Union[Literal["none", "expand", "diverging", "silhouette", "wiggle"], Undefined] = undefined
    stackOrder: Union[Literal["reverse", "none", "appearance", "ascending", "descending", "insideOut"], Undefined] = undefined

@dataclasses.dataclass
class BarSeries(CommonSeriesType):
    data: Union[Sequence[Optional[NumberType]], Undefined] = undefined
    dataKey: Union[str, Undefined] = undefined
    label: Union[str, Undefined] = undefined
    layout: Union[Literal["horizontal", "vertical"], Undefined] = undefined
    stackOffset: Union[Literal["none", "expand", "diverging", "silhouette", "wiggle"], Undefined] = undefined
    minBarSize: Union[NumberType, Undefined] = undefined
    
@dataclasses.dataclass
class LineSeries(CommonSeriesType):
    data: Union[Sequence[Optional[NumberType]], Undefined] = undefined
    dataKey: Union[str, Undefined] = undefined
    label: Union[str, Undefined] = undefined
    curve: Union[Literal["linear", "step", "catmullRom", "monotoneX", "monotoneY", "natural", "stepBefore", "stepAfter", "bumpY", "bumpX"], Undefined] = undefined
    strictStepCurve: Union[bool, Undefined] = undefined
    showMark: Union[bool, Undefined] = undefined
    shape: Union[Literal['circle', 'cross', 'diamond', 'square', 'star', 'triangle', 'wye'], Undefined] = undefined
    disableHighlight: Union[bool, Undefined] = undefined
    connectNulls: Union[bool, Undefined] = undefined
    stackOffset: Union[Literal["none", "expand", "diverging", "silhouette", "wiggle"], Undefined] = undefined
    baseline: Union[NumberType, Literal["min", "max"], Undefined] = undefined
    
@dataclasses.dataclass
class ScatterSeriesPreview:
    markerSize: Union[NumberType, Undefined] = undefined


@dataclasses.dataclass
class ScatterSeries(CommonSeriesType):
    data: Union[Sequence[ScatterValueType], Undefined] = undefined
    markerSize: Union[NumberType, Undefined] = undefined
    datasetKeys: Union[ScatterDatasetKeyType, Undefined] = undefined
    label: Union[str, Undefined] = undefined
    disableHover: Union[bool, Undefined] = undefined
    zAxisId: Union[str, Undefined] = undefined
    preview: Union[ScatterSeriesPreview, Undefined] = undefined

@dataclasses.dataclass
class ChartsAxisHighlightProps:
    x: Union[Literal['none', 'line', 'band'], Undefined] = undefined
    y: Union[Literal['none', 'line', 'band'], Undefined] = undefined

@dataclasses.dataclass
class ChartsGridProps:
    vertical: Union[bool, Undefined] = undefined
    horizontal: Union[bool, Undefined] = undefined

@dataclasses.dataclass
class AxisItemIdentifier:
    axisId: Union[str, NumberType]
    dataIndex: NumberType

@dataclasses.dataclass
class HighlightItemData:
    seriesId: Union[str, NumberType]
    dataIndex: NumberType

@dataclasses.dataclass
class ChartMargin:
    left: NumberType 
    right: NumberType
    top: NumberType
    bottom: NumberType

@dataclasses.dataclass
class CommonAxisConfig:
    id: Union[str, NumberType, Undefined] = undefined
    min: Union[NumberType, Undefined] = undefined
    max: Union[NumberType, Undefined] = undefined
    data: Union[Sequence[Any], Undefined] = undefined
    dataKey: Union[str, Undefined] = undefined
    hideTooltip: Union[bool, Undefined] = undefined
    reverse: Union[bool, Undefined] = undefined
    domainLimit: Union[Literal['nice', 'strict'], Undefined] = undefined
    ignoreTooltip: Union[bool, Undefined] = undefined

@dataclasses.dataclass
class ChartsAxisProps:
    tickMaxStep: Union[NumberType, Undefined] = undefined
    tickMinStep: Union[NumberType, Undefined] = undefined
    tickNumber: Union[NumberType, Undefined] = undefined
    tickPlacement: Union[Literal['start', 'end', 'middle', 'extremities'], Undefined] = undefined
    tickLabelPlacement: Union[Literal['middle', 'tick'], Undefined] = undefined
    axisId: Union[str, NumberType, Undefined] = undefined
    disableLine: Union[bool, Undefined] = undefined
    disableTicks: Union[bool, Undefined] = undefined
    # tickLabelStyle: Union[Dict[str, Any], Undefined] = undefined
    # labelStyle: Union[Dict[str, Any], Undefined] = undefined
    # tickLabelInterval: Union[Literal['auto'], Callable[[Any, NumberType], bool], Undefined] = undefined
    label: Union[str, Undefined] = undefined
    tickSize: Union[NumberType, Undefined] = undefined
    scaleType: Union[Literal['time', 'utc', 'linear', 'sqrt', 'pow', 'log', 'symlog', 'point', 'band'], Undefined] = undefined

    position: Union[Literal['top', 'bottom', 'left', 'right', 'none'], Undefined] = undefined
    # only available when type is band (bar chart)
    categoryGapRatio: Union[NumberType, Undefined] = undefined
    barGapRatio: Union[NumberType, Undefined] = undefined

@dataclasses.dataclass
class XAxis(CommonAxisConfig, ChartsAxisProps):
    tickLabelMinGap: Union[NumberType, Undefined] = undefined
    
@dataclasses.dataclass
class YAxis(CommonAxisConfig, ChartsAxisProps):
    pass

@dataclasses.dataclass
class ZAxis:
    id: Union[str, Undefined] = undefined
    data: Union[Sequence[Any], Undefined] = undefined
    dataKey: Union[str, Undefined] = undefined
    min: Union[NumberType, Undefined] = undefined
    max: Union[NumberType, Undefined] = undefined


@dataclasses.dataclass
class BarChartProps(BasicProps):
    series: Sequence[BarSeries] = dataclasses.field(default_factory=list)
    dataset: Union[Sequence[Any], Undefined] = undefined
    axisHighlight: Union[ChartsAxisHighlightProps, Undefined] = undefined
    borderRadius: Union[NumberType, Undefined] = undefined
    disableAxisListener: Union[bool, Undefined] = undefined
    grid: Union[ChartsGridProps, Undefined] = undefined
    width: Union[NumberType, Undefined] = undefined
    height: Union[NumberType, Undefined] = undefined
    margin: Union[ChartMargin, Undefined] = undefined
    hideLegend: Union[bool, Undefined] = undefined
    highlightedAxis: Union[Sequence[AxisItemIdentifier], Undefined] = undefined
    highlightedItem: Union[HighlightItemData, Undefined] = undefined
    layout: Union[Literal['horizontal', 'vertical'], Undefined] = undefined
    showToolbar: Union[bool, Undefined] = undefined
    skipAnimation: Union[bool, Undefined] = undefined
    xAxis: Union[Sequence[XAxis], Undefined] = undefined
    yAxis: Union[Sequence[YAxis], Undefined] = undefined

@dataclasses.dataclass
class LineChartProps(BasicProps):
    series: Sequence[LineSeries] = dataclasses.field(default_factory=list)
    dataset: Union[Sequence[Any], Undefined] = undefined
    axisHighlight: Union[ChartsAxisHighlightProps, Undefined] = undefined
    disableAxisListener: Union[bool, Undefined] = undefined
    grid: Union[ChartsGridProps, Undefined] = undefined
    width: Union[NumberType, Undefined] = undefined
    height: Union[NumberType, Undefined] = undefined
    margin: Union[ChartMargin, Undefined] = undefined
    hideLegend: Union[bool, Undefined] = undefined
    highlightedAxis: Union[Sequence[AxisItemIdentifier], Undefined] = undefined
    highlightedItem: Union[HighlightItemData, Undefined] = undefined
    showToolbar: Union[bool, Undefined] = undefined
    skipAnimation: Union[bool, Undefined] = undefined
    xAxis: Union[Sequence[XAxis], Undefined] = undefined
    yAxis: Union[Sequence[YAxis], Undefined] = undefined
    lineWidth: Union[NumberType, Undefined] = undefined

@dataclasses.dataclass
class ScatterChartProps(BasicProps):
    series: Sequence[ScatterSeries] = dataclasses.field(default_factory=list)
    dataset: Union[Sequence[Any], Undefined] = undefined
    axisHighlight: Union[ChartsAxisHighlightProps, Undefined] = undefined
    disableAxisListener: Union[bool, Undefined] = undefined
    grid: Union[ChartsGridProps, Undefined] = undefined
    width: Union[NumberType, Undefined] = undefined
    height: Union[NumberType, Undefined] = undefined
    margin: Union[ChartMargin, Undefined] = undefined
    hideLegend: Union[bool, Undefined] = undefined
    highlightedAxis: Union[Sequence[AxisItemIdentifier], Undefined] = undefined
    highlightedItem: Union[HighlightItemData, Undefined] = undefined
    showToolbar: Union[bool, Undefined] = undefined
    skipAnimation: Union[bool, Undefined] = undefined
    xAxis: Union[Sequence[XAxis], Undefined] = undefined
    yAxis: Union[Sequence[YAxis], Undefined] = undefined
    zAxis: Union[Sequence[ZAxis], Undefined] = undefined

@dataclasses.dataclass
class SparkLineClipAreaOffset:
    top: Union[NumberType, Undefined] = undefined
    bottom: Union[NumberType, Undefined] = undefined
    left: Union[NumberType, Undefined] = undefined
    right: Union[NumberType, Undefined] = undefined

@dataclasses.dataclass
class SparkLineChartProps(BasicProps):
    data: Sequence[NumberType] = dataclasses.field(default_factory=list)
    dataset: Union[Sequence[Any], Undefined] = undefined
    axisHighlight: Union[ChartsAxisHighlightProps, Undefined] = undefined
    disableAxisListener: Union[bool, Undefined] = undefined
    grid: Union[ChartsGridProps, Undefined] = undefined
    width: Union[NumberType, Undefined] = undefined
    height: Union[NumberType, Undefined] = undefined
    margin: Union[ChartMargin, Undefined] = undefined
    highlightedAxis: Union[Sequence[AxisItemIdentifier], Undefined] = undefined
    highlightedItem: Union[HighlightItemData, Undefined] = undefined
    skipAnimation: Union[bool, Undefined] = undefined
    xAxis: Union[Sequence[XAxis], Undefined] = undefined
    yAxis: Union[Sequence[YAxis], Undefined] = undefined
    clipAreaOffset: Union[SparkLineClipAreaOffset, Undefined] = undefined
    disableClipping: Union[bool, Undefined] = undefined
    disableVoronoi: Union[bool, Undefined] = undefined
    plotType: Union[Literal["line", "bar"], Undefined] = undefined
    showHighlight: Union[bool, Undefined] = undefined
    showTooltip: Union[bool, Undefined] = undefined
    voronoiMaxRadius: Union[NumberType, Undefined] = undefined
    # only available when plotType is line
    curve: Union[Literal["linear", "step", "catmullRom", "monotoneX", "monotoneY", "natural", "stepBefore", "stepAfter", "bumpY", "bumpX"], Undefined] = undefined
    baseline: Union[NumberType, Literal["min", "max"], Undefined] = undefined
    area: Union[bool, Undefined] = undefined
    lineWidth: Union[NumberType, Undefined] = undefined

@dataclasses.dataclass
class BarItemIdentifier:
    type: str 
    seriesId: Union[str, NumberType]
    dataIndex: NumberType


@dataclasses.dataclass
class LineItemIdentifier:
    type: str 
    seriesId: Union[str, NumberType]
    dataIndex: Union[NumberType, Undefined] = undefined

@dataclasses.dataclass
class ChartsAxisData:
    dataIndex: NumberType
    axisValue: Union[NumberType, str]
    seriesValues: dict[str, Optional[NumberType]]

@dataclasses.dataclass
class ScatterItemIdentifier:
    type: str 
    seriesId: Union[str, NumberType]
    dataIndex: NumberType

class BarChart(MUIComponentBase[BarChartProps]):
    def __init__(self) -> None:
        super().__init__(UIType.MUIBarChart,
                         BarChartProps,
                         allowed_events=[
                            FrontendEventType.ChartItemClick,
                            FrontendEventType.ChartAxisClick,
                        ])
        self.event_item_click = self._create_event_slot(FrontendEventType.ChartItemClick,
            converter=lambda x: BarItemIdentifier(**x))
        self.event_axis_click = self._create_event_slot(FrontendEventType.ChartAxisClick,
            converter=lambda x: ChartsAxisData(**x) if x is not None else None)
    
    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)

class LineChart(MUIComponentBase[LineChartProps]):
    def __init__(self) -> None:
        super().__init__(UIType.MUILineChart,
                         LineChartProps,
                         allowed_events=[
                            FrontendEventType.ChartLineClick,
                            FrontendEventType.ChartAreaClick,
                            FrontendEventType.ChartAxisClick,
                            FrontendEventType.ChartMarkClick,
                        ])
        self.event_line_click = self._create_event_slot(FrontendEventType.ChartLineClick,
            converter=lambda x: LineItemIdentifier(**x))
        self.event_area_click = self._create_event_slot(FrontendEventType.ChartAreaClick,
            converter=lambda x: LineItemIdentifier(**x))
        self.event_axis_click = self._create_event_slot(FrontendEventType.ChartAxisClick,
            converter=lambda x: ChartsAxisData(**x) if x is not None else None)
        self.event_mark_click = self._create_event_slot(FrontendEventType.ChartMarkClick,
            converter=lambda x: LineItemIdentifier(**x))

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)

class ScatterChart(MUIComponentBase[ScatterChartProps]):
    def __init__(self) -> None:
        super().__init__(UIType.MUIScatterChart,
                         ScatterChartProps,
                         allowed_events=[
                            FrontendEventType.ChartItemClick,
                            FrontendEventType.ChartAxisClick
                        ])
        self.event_item_click = self._create_event_slot(FrontendEventType.ChartItemClick,
            converter=lambda x: ScatterItemIdentifier(**x))
        self.event_axis_click = self._create_event_slot(FrontendEventType.ChartAxisClick,
            converter=lambda x: ChartsAxisData(**x) if x is not None else None)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)

class SparkLineChart(MUIComponentBase[SparkLineChartProps]):
    def __init__(self) -> None:
        super().__init__(UIType.MUISparkLineChart,
                         SparkLineChartProps,
                         allowed_events=[
                            FrontendEventType.ChartItemClick,
                            FrontendEventType.ChartAxisClick
                        ])
        self.event_item_click = self._create_event_slot(FrontendEventType.ChartItemClick,
            converter=lambda x: ScatterItemIdentifier(**x))
        self.event_axis_click = self._create_event_slot(FrontendEventType.ChartAxisClick,
            converter=lambda x: ChartsAxisData(**x) if x is not None else None)
    
    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync=is_sync)


def _main():
    BarChartProps(
        series=[
            BarSeries(data=[4, 3, 5]),
            BarSeries(data=[1, 6, 3]),
            BarSeries(data=[2, 5, 6]),
        ],
        xAxis=[XAxis(data=["A", "B", "C"])],
    )

if __name__ == "__main__":
    _main()