from typing import Dict, Union, Any, List, Tuple, Optional

import numpy as np
import abc


class FigureObject(abc.ABC):
    @abc.abstractmethod
    def data(self) -> dict:
        raise NotImplementedError

    def bound(self) -> Optional[List[float]]:
        return None


class FigureObject2d(FigureObject):
    @abc.abstractmethod
    def move(self, x: float, y: float):
        raise NotImplementedError


class FigureObject3d(FigureObject):
    @abc.abstractmethod
    def move(self, x: float, y: float, z: float):
        raise NotImplementedError


class ImagePolygon(FigureObject2d):
    Type = "polygon"

    def __init__(self,
                 points: np.ndarray,
                 color: str,
                 hovertext: str = "",
                 query: bool = False,
                 select: bool = True,
                 closed: bool = True):
        # points: [N, 2]
        self.points = points.reshape(-1, 2)
        self.color = color
        self.hovertext = hovertext
        self.query = query
        self.select = select
        self.closed = closed

    def move(self, x: float, y: float):
        self.points[:, 0] += x
        self.points[:, 1] += y

    def bound(self):
        mins = self.points.min(axis=0)
        maxs = self.points.max(axis=0)
        return [mins[0], mins[1], maxs[0], maxs[1]]

    def data(self) -> dict:
        return {
            "type": ImagePolygon.Type,
            "points": self.points.reshape(-1).tolist(),
            "color": self.color,
            "hovertext": self.hovertext,
            "need_query": self.query,
            "select": self.select,
            "closed": self.closed,
        }


class Lines2d(FigureObject2d):
    Type = "lines"

    def __init__(self,
                 lines: np.ndarray,
                 color: str,
                 width: float = 1,
                 opacity: float = 0.5):
        self.lines = lines.reshape(-1, 2, 2).astype(np.float32)
        self.color = color
        self.width = width
        self.opacity = opacity

    def move(self, x: float, y: float):
        self.lines[..., 0] += float(x)
        self.lines[..., 1] += float(y)

    def bound(self):
        if self.lines.size == 0:
            return None
        mins = self.lines.reshape(-1, 2).min(axis=0)
        maxs = self.lines.reshape(-1, 2).max(axis=0)
        return [mins[0], mins[1], maxs[0], maxs[1]]

    def data(self) -> dict:
        return {
            "type": self.Type,
            "lines": self.lines,
            "color": self.color,
            "width": self.width,
            "opacity": self.opacity,
            "need_query": False,
        }

class Points2d(FigureObject2d):
    Type = "points"
    def __init__(self,
                points: np.ndarray,
                color: str,
                width: float = 1,
                opacity: float = 0.5):
        self.points = points.reshape(-1, 2).astype(np.float32)
        self.color = color
        self.width = width
        self.opacity = opacity


    def move(self, x: float, y: float):
        self.points[..., 0] += float(x)
        self.points[..., 1] += float(y)

    def bound(self):
        if self.points.size == 0:
            return None 
        mins = self.points.reshape(-1, 2).min(axis=0)
        maxs = self.points.reshape(-1, 2).max(axis=0)
        return [mins[0], mins[1], maxs[0], maxs[1]]


    def data(self) -> dict:
        return {
            "type": self.Type,
            "points": self.points,
            "color": self.color,
            "width": self.width,
            "opacity": self.opacity,
            "need_query": False,
        }


class _Plotly(FigureObject2d):
    Type = "plotly"

    def __init__(self, data: list, layout: Optional[Any] = None):
        if layout is None:
            layout = {}
        self._plotly_data = data
        self._layout = layout

    def data(self) -> dict:
        return {
            "type": self.Type,
            "data": self._plotly_data,
            "layout": self._layout,
        }

    def move(self, x: float, y: float):
        return

    def bound(self):
        return [0, 0, 0, 0]


class PlotlyPlot(_Plotly):
    def __init__(self, width: int, height: int, title: str = ""):
        layout: Dict[str, Any] = {
            "width": width,
            "height": height,
            # "margin": {
            #     "l": 10,
            #     "r": 10,
            #     "b": 10,
            #     "t": 10,
            # }
        }
        if title:
            layout["title"] = title
        super().__init__([], layout)

    def margin(self, l: int, r: int, b: int, t: int):
        self._layout["margin"] = {
            "l": l,
            "r": r,
            "b": b,
            "t": t,
        }

        return self

    def grid(self,
             rows: int,
             cols: int,
             pattern: str = 'independent',
             roworder: str = 'top to bottom'):
        self._layout["grid"] = {
            "rows": rows,
            "columns": cols,
            "pattern": pattern,
            "roworder": roworder,
        }
        return self

    def _to_list(self, x: Union[np.ndarray, list]):
        if isinstance(x, np.ndarray):
            return x.tolist()
        assert isinstance(x, list)
        return x

    def scatter(self,
                x: Union[np.ndarray, list],
                y: Union[np.ndarray, list],
                color: Optional[str] = None,
                size: float = 1,
                **props):
        marker = {"size": size}  # type: Dict[str, Union[str, int, float]]
        if color is not None:
            marker["color"] = color

        data = {
            "x": self._to_list(x),
            "y": self._to_list(y),
            "mode": 'markers',
            "type": 'scatter',
            "marker": marker,
            **props
        }
        self._plotly_data.append(data)
        return self

    def line(self,
             x,
             y: Optional[Any] = None,
             color: Optional[str] = None,
             title: Optional[str] = None,
             width: float = 1,
                **props):
        line = {"width": width}  # type: Dict[str, Union[str, int, float]]
        if color is not None:
            line["color"] = color
        data = {
            "x": self._to_list(x),
            "y": list(range(0, len(x))) if y is None else y,
            "mode": 'lines',
            "line": line,
            **props
        }
        self._plotly_data.append(data)
        return self


class Texts2d(FigureObject2d):
    Type = "texts"

    def __init__(self, locs: np.ndarray, texts: List[str], color: str,
                 fontSize: float):
        self.locs = locs.reshape(-1, 2).astype(np.float32)
        self.color = color
        self.texts = texts
        self.fontSize = fontSize

    def move(self, x: float, y: float):
        self.locs[..., 0] += float(x)
        self.locs[..., 1] += float(y)

    def bound(self):
        if self.locs.size == 0:
            return None
        mins = self.locs.reshape(-1, 2).min(axis=0)
        maxs = self.locs.reshape(-1, 2).max(axis=0)
        return [mins[0], mins[1], maxs[0], maxs[1]]

    def data(self) -> dict:
        return {
            "type": self.Type,
            "locs": self.locs.reshape(-1).tolist(),
            "color": self.color,
            "texts": self.texts,
            "fontSize": self.fontSize,
            "need_query": False,
        }


class Lines3d(FigureObject3d):
    Type = "lines"

    def __init__(self,
                 lines: np.ndarray,
                 color: str,
                 width: float = 4,
                 opacity: float = 0.5):
        self.lines = lines.reshape(-1, 2, 3)
        self.color = color
        self.width = width
        self.opacity = opacity

    def move(self, x: float, y: float, z: float):
        self.lines[..., 0] += x
        self.lines[..., 1] += y
        self.lines[..., 2] += z

    def bound(self):
        mins = self.lines.reshape(-1, 3).min(axis=0)
        maxs = self.lines.reshape(-1, 3).max(axis=0)
        return [mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2]]

    def data(self) -> dict:
        return {
            "type": self.Type,
            "lines": self.lines.reshape(-1).tolist(),
            "color": self.color,
            "width": self.width,
            "opacity": self.opacity,
            "need_query": False,
        }


class Polygon3d(Lines3d):
    def __init__(self,
                 polygon: np.ndarray,
                 color: str,
                 width: float = 4,
                 opacity: float = 0.5,
                 closed: bool = True):
        polygon = np.array(polygon)
        assert len(polygon.shape) == 2 and polygon.shape[1] == 3
        polygon_next = polygon[[-1, *range(0, len(polygon) - 1)]]
        lines = np.stack([polygon, polygon_next], axis=1)
        if not closed:
            lines = lines[:-1]
        super().__init__(lines, color, width, opacity)


class BoundingBox(FigureObject3d):
    Type = "boundingbox"

    def __init__(self,
                 dim: Tuple[float, float, float],
                 pos: Tuple[float, float, float],
                 rot: Tuple[float, float, float],
                 color: str,
                 hovertext: str = "",
                 query: bool = True,
                 **props):
        self.dim = dim
        self.pos = list(pos)
        self.rot = rot
        self.color = color
        self.hovertext = hovertext
        self.query = query
        self.props = props

    def move(self, x: float, y: float, z: float):
        self.pos[0] += x
        self.pos[1] += y
        self.pos[2] += z

    def bound(self):
        # TODO better bound
        pos = np.array(self.pos)
        dim = np.array(self.dim) / 2
        return [*(pos - dim), *(pos + dim)]

    def data(self) -> dict:
        return {
            "type": self.Type,
            "dim": self.dim,
            "pos": self.pos,
            "rot": self.rot,
            "color": self.color,
            "hovertext": self.hovertext,
            "need_query": self.query,
            **self.props,
        }


class Element3d(FigureObject3d):
    Type = "Custom"

    def __init__(self, elem_name: str, dynamic_props: dict, **props):
        self.elem_name = elem_name
        self.dynamic_props = dynamic_props
        self.props = props

    def move(self, x: float, y: float, z: float):
        pass

    def data(self) -> dict:
        return {
            "type": self.Type,
            "dynamicName": self.elem_name,
            "dynamicProps": self.dynamic_props,
            **self.props,
        }


class Sphere3d(FigureObject3d):
    Type = "sphere"

    def __init__(self,
                 pos: Tuple[float, float, float],
                 radius: float,
                 color: str,
                 label: Optional[str] = None,
                 labelColor: Optional[str] = None):
        self.radius = radius
        self.pos = list(pos)
        self.color = color
        self.label = label
        self.labelColor = labelColor

    def move(self, x: float, y: float, z: float):
        self.pos[0] += x
        self.pos[1] += y
        self.pos[2] += z

    def data(self) -> dict:
        return {
            "pos": self.pos,
            "radius": self.radius,
            "color": self.color,
            "label": self.label,
            "need_query": False,
            "labelColor": self.labelColor,
        }
