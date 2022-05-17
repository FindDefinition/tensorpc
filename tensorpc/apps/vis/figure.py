import base64
import contextlib
from typing import Any, List, Optional

import numpy as np
from tensorpc.apps.vis import objects
import abc 

class Layer(abc.ABC):
    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def data(self):
        raise NotImplementedError

    def bound(self) -> Optional[List[float]]:
        return None 

class Layer2d(Layer):
    def __init__(self, name: str):
        super().__init__(name)
        self.objects = [] # type: List[objects.FigureObject2d]

    def move(self, x: float, y: float):
        # move the whole layer to another location.
        for obj in self.objects:
            obj.move(x, y)

    def add_object(self, obj: objects.FigureObject2d):
        assert isinstance(obj, objects.FigureObject2d)
        self.objects.append(obj)
        return self

    def data(self):
        return {
            "name": self.name ,
            "objects": [obj.data() for obj in self.objects],
        }

class Layer3d(Layer):
    def __init__(self, name: str):
        super().__init__(name)
        self.objects = [] # type: List[objects.FigureObject3d]

    def move(self, x: float, y: float, z: float):
        # move the whole layer to another location.
        for obj in self.objects:
            obj.move(x, y, z)

    def add_object(self, obj: objects.FigureObject3d):
        assert isinstance(obj, objects.FigureObject3d), str(type(obj))
        self.objects.append(obj)
        return self

    def data(self):
        return {
            "name": self.name ,
            "objects": [obj.data() for obj in self.objects],
        }

class ImageLayer(Layer2d):
    def polygon(self,
                points: np.ndarray,
                color: str,
                hovertext: str = "",
                query=False,
                select=True,
                closed=True):
        return self.add_object(objects.ImagePolygon(points, color, hovertext, query, select, closed))
    
    def lines(self, lines, color: str, width=1, opacity=0.5):
        obj = objects.Lines2d(lines, color, width, opacity)
        return self.add_object(obj)

    def points(self, points, color: str, width=1, opacity=0.5):
        obj = objects.Points2d(points, color, width, opacity)
        return self.add_object(obj)

    def box_minmax(self,
                   box: list,
                   color: str,
                   hovertext: str = "",
                   query=False,
                   select=True):
        x0, y0, x1, y1 = box
        points = [x0, y0, x0, y1, x1, y1, x1, y0]
        return self.polygon(np.array(points), color, hovertext, query, select=select)

    def bound(self) -> Optional[List[float]]:
        bounds = [] # type: List[List[float]]
        for obj in self.objects:
            bound = obj.bound()
            if bound is not None:
                bounds.append(bound)
        bounds_arr = np.array(bounds)
        return [*bounds_arr[:, :2].min(axis=0), *bounds_arr[:, 2:].max(axis=0)]

def lines_to_polygon(lines: np.ndarray):
    assert len(lines.shape) == 2
    lines_next = lines[[-1, *range(0, len(lines) - 1)]]
    lines = np.stack([lines, lines_next], axis=1)
    return lines


class D3Layer(Layer3d):
    def box(self,
            dim,
            pos,
            rot,
            color: str,
            hovertext: str = "",
            query=True,
            **props):
        obj = objects.BoundingBox(dim, pos, rot, color, hovertext, query, **props)
        return self.add_object(obj)

    def element(self, elem_name: str, dynamic_props: dict, **props):
        obj = objects.Element3d(elem_name, dynamic_props, **props)
        return self.add_object(obj)

    def lines(self, lines, color: str, width=4, opacity=0.5):
        obj = objects.Lines3d(lines, color, width, opacity)
        return self.add_object(obj)

    def sphere(self, pos, radius, color, label=None, labelColor=None):
        obj = objects.Sphere3d(pos, radius, color, label, labelColor)
        return self.add_object(obj)

    def polygon(self, polygon, color: str, width=4, opacity=0.5, closed=True):
        obj = objects.Polygon3d(polygon, color, width, opacity, closed)
        return self.add_object(obj)

    def bound(self) -> Optional[List[float]]:
        bounds = [] # type: List[List[float]]
        for obj in self.objects:
            bound = obj.bound()
            if bound is not None:
                bounds.append(bound)
        if not bounds:
            return None 
        bounds_arr = np.array(bounds)
        return [*bounds_arr[:, :3].min(axis=0), *bounds_arr[:, 3:].max(axis=0)]

class Figure(object):
    def __init__(self, uid, data, name=None, **props):
        if name is None:
            name = uid
        
        self._data = {
            "uid": uid,
            "name": name,
            "base": {
                "data": data,
            },
            "rpc": {
                "address": "",
                "httpAddress": "",
                "service": "",
                "arguments": [],
                "usehttp": True,
            },
            "layers": [],
            "props": props,
            "tags": [],
        }
        self.layers = [] # type: List[Layer]
        self.props = self._data["props"]
        self.tags = self._data["tags"]

    def add_layer(self, layer):
        self.layers.append(layer)

    @property 
    def num_layers(self):
        return len(self.layers)

    def data(self):
        self._data["layers"] = [l.data() for l in self.layers]
        return self._data

    def set_overlay_rpc(self, addr: str, http_addr: str, service: str, 
                        args: List[Any], use_http: bool = False):
        self._data["overlay_rpc"] = {
            "address": addr,
            "httpAddress": http_addr,
            "service": service,
            "arguments": args,
            "usehttp": use_http,
        }

    def set_rpc(self, addr: str, http_addr: str, service: str, 
                args: List[Any], use_http: bool = False):
        self._data["rpc"] = {
            "address": addr,
            "httpAddress": http_addr,
            "service": service,
            "arguments": args,
            "usehttp": use_http,
        }


class ImageFigure(Figure):
    def __init__(self, uid, data=None, autoscale=True, encode_suffix="jpg",
                 **props):
        if data is not None:
            if isinstance(data, np.ndarray):
                import cv2
                assert len(data.shape) in (3, 2)
                _, img_str = cv2.imencode(".{}".format(encode_suffix), data)
                encoded_string = base64.b64encode(img_str)
                web_image = 'data:image/{};base64,'.format(encode_suffix)
                web_image += encoded_string.decode("utf-8")
                data = web_image
        super(ImageFigure, self).__init__(uid, data, **props)
        self._data["type"] = "image"
        self._data["base"]["autoscale"] = autoscale

    @contextlib.contextmanager
    def layer(self, name):
        layer = ImageLayer(name)
        assert isinstance(layer, ImageLayer)
        yield layer
        self.add_layer(layer)


class RPCImageFigure(ImageFigure):
    def __init__(self, uid, addr, service: str, arguments: List[Any]):
        super(ImageFigure, self).__init__(uid, None)
        self._data["type"] = "image"
        self._data["rpc"]["address"] = addr
        self._data["rpc"]["service"] = service
        self._data["rpc"]["arguments"] = arguments


class PointCloudFigure(Figure):
    def __init__(self,
                 uid,
                 data=None,
                 color=None,
                 opacity=None,
                 pc_type="pointcloud",
                 factor=100,
                 **props):
        super(PointCloudFigure, self).__init__(uid, data, name=None, **props)
        if data is not None:
            data = self._propress_pointcloud(data, factor)

        self._data["type"] = pc_type
        self._data["base"]["factor"] = factor
        self._data["base"]["data"] = data
        self._data["base"]["color"] = color
        self._data["base"]["opacity"] = opacity

    @contextlib.contextmanager
    def layer(self, name):
        layer = D3Layer(name)
        assert isinstance(layer, D3Layer)
        yield layer
        self.add_layer(layer)

    def _propress_pointcloud(self, data, factor):
        assert isinstance(data, np.ndarray)
        assert data.shape[1] >= 3
        data = data[:, :3]
        if data.dtype == np.float32 or data.dtype == np.float64:
            if factor > 0:
                # convert to int16
                data = (data * factor).astype(np.int16)
        return data

    def update_data(self, data, factor=100):
        self._data["base"]["data"] = self._propress_pointcloud(data, factor)


class RPCPointCloudFigure(PointCloudFigure):
    def __init__(self,
                 uid,
                 addr: str,
                 service: str,
                 arguments: List[Any],
                 pc_type="pointcloud"):
        super(PointCloudFigure, self).__init__(uid, None)
        self._data["type"] = pc_type
        self._data["rpc"]["address"] = addr
        self._data["rpc"]["service"] = service
        self._data["rpc"]["arguments"] = arguments


if __name__ == "__main__":
    import json
    from pathlib import Path
    datas = []
    idx = 0
    paths = list(
        Path("/mnt/truenas/scratch/yan.yan/tasks/200102-lidarline/pc").glob(
            "*.npy"))
    paths = sorted(paths)
    for npy in paths:
        figure = PointCloudFigure(idx, str(npy))
        figure._data["type"] = "pointcloud-npz"
        datas.append(figure.data())
        idx += 1
    with open("/home/tusimple/test_pc.json", "w") as f:
        json.dump(datas, f)
