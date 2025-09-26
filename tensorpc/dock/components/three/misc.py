import asyncio
import json
from typing_extensions import Annotated, Literal, TypeAlias
from typing import (Callable, Union, Any, Optional, Coroutine)
import enum 
import base64 

import numpy as np 
from pydantic import field_validator
import urllib.request

import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.annolib import Undefined, undefined
from tensorpc.core.datamodel import typemetas
from tensorpc.dock.components.three.geometry import Shape
from tensorpc.dock.core.component import (UIType, FrontendEventType, ContainerBaseProps)
from collections.abc import Sequence
from tensorpc.dock.core import colors
from tensorpc.dock.core.appcore import Event, EventDataType
from tensorpc.dock.core.common import handle_standard_event
from tensorpc.core.datamodel.typemetas import RangedFloat, RangedInt
from tensorpc.dock.components.mui import (Image as MUIImage, PointerEventsProperties, MUIComponentType)
from pydantic import field_validator, model_validator
from typing_extensions import Self

from .base import (PyDanticConfigForNumpy, NumberType, ThreeBasicProps, ThreeComponentBase, Object3dBaseProps, Vector3Type, 
    Object3dWithEventBase, InteractableProps, ValueType, ThreeContainerBase, ThreeComponentType,
    Object3dContainerBase, Object3dContainerBaseProps, ThreeLayoutType)
from tensorpc.core.httpservers.core import JS_MAX_SAFE_INT

_CORO_NONE: TypeAlias = Union[Coroutine[None, None, None], None]

@dataclasses.dataclass
class ColorMap:
    type: Literal["jet"] = "jet"
    min: Union[NumberType, Undefined] = undefined
    max: Union[NumberType, Undefined] = undefined


@dataclasses.dataclass(config=PyDanticConfigForNumpy)
class PointProps(Object3dBaseProps):
    limit: int = 0
    points: Union[np.ndarray, Undefined] = undefined
    colors: Union[np.ndarray, str, Undefined] = undefined
    attrs: Union[np.ndarray, Undefined] = undefined
    attrFields: Union[Sequence[str], Undefined] = undefined
    labels: Union[np.ndarray, Undefined] = undefined
    labelColorArray: Union[Sequence[Vector3Type], Undefined] = undefined
    sizeAttenuation: bool = False
    size: float = 3.0
    sizes: Union[np.ndarray, Undefined] = undefined
    encodeMethod: Union[Literal["none", "int16"], Undefined] = undefined
    encodeScale: Union[NumberType, Undefined] = undefined
    enableBvh: Union[bool, Undefined] = undefined
    colorMap: Union[ColorMap, Undefined] = undefined

    @field_validator('points')
    def points_validator(cls, v: Union[np.ndarray, Undefined]):
        if not isinstance(v, Undefined):
            assert v.dtype == np.float32 or v.dtype == np.uint8
            assert v.ndim == 2 and v.shape[1] in [3, 4]
        return v

    @field_validator('labelColorArray')
    def label_color_array_validator(cls, v: Union[Sequence[Vector3Type], Undefined]):
        if not isinstance(v, Undefined):
            assert len(v) <= 20, "max 20 label color"
        return v

class PointsControlType(enum.Enum):
    SetColors = 0


class Points(Object3dWithEventBase[PointProps]):

    def __init__(self, limit: int) -> None:
        super().__init__(UIType.ThreePoints, PointProps)
        self.props.points = np.zeros((0, 3), np.float32)
        self.props.limit = limit

    def get_sync_props(self) -> dict[str, Any]:
        res = super().get_sync_props()
        res["points"] = self.props.points
        res["labels"] = self.props.labels
        res["colors"] = self.props.colors
        res["sizes"] = self.props.sizes
        res["attrs"] = self.props.attrs
        res["attrFields"] = self.props.attrFields
        res["labelColorArray"] = self.props.labelColorArray
        return res

    def _get_ui_label_color_array(self):
        return [colors.str_to_rgb_float(color) for color in colors.RANDOM_COLORS_FOR_UI]

    def validate_props(self, props: dict[str, Any]):
        if "points" in props:
            return props["points"].shape[0] <= self.props.limit
        return False

    @staticmethod
    def _check_colors(colors, points: Optional[np.ndarray] = None):
        if isinstance(colors, np.ndarray):
            if colors.ndim == 1:
                assert colors.dtype == np.uint8, "when gray, must be uint8"
            else:
                assert colors.ndim == 2 and colors.shape[1] == 3
            if points is not None:
                assert points.shape[0] == colors.shape[0]

    async def set_colors_in_range(self, colors: Union[str, np.ndarray],
                                  begin: int, end: int):
        """
        Args: 
            cam2world: camera to world matrix, 4x4 ndaray or 16 list
            distance: camera orbit target distance.
        """
        assert begin >= 0 and end >= begin and end <= self.props.limit
        self._check_colors(colors)
        if isinstance(colors, np.ndarray):
            assert colors.shape[0] == end - begin
        return await self.send_and_wait(
            self.create_comp_event({
                "type": PointsControlType.SetColors.value,
                "offset": [begin, end],
                "colors": colors,
            }))

    async def clear(self):
        self.props.points = np.zeros((0, 3), np.float32)
        self.props.colors = undefined
        self.props.attrs = undefined
        self.props.attrFields = undefined
        self.props.sizes = undefined

        return await self.send_and_wait(
            self.update_event(points=self.props.points,
                              colors=undefined,
                              attrs=undefined,
                              attrFields=undefined,
                              sizes=undefined))

    async def update_points(self,
                            points: np.ndarray,
                            colors: Optional[Union[np.ndarray, str,
                                                   Undefined]] = None,
                            attrs: Optional[Union[np.ndarray,
                                                  Undefined]] = None,
                            attr_fields: Optional[list[str]] = None,
                            limit: Optional[int] = None,
                            sizes: Optional[Union[np.ndarray,
                                                  Undefined]] = None,
                            size_attenuation: bool = False,
                            size: Optional[Union[NumberType,
                                                 Undefined]] = None,
                            encode_method: Optional[Union[Literal["none",
                                                                  "int16"],
                                                          Undefined]] = None,
                            encode_scale: Optional[Union[NumberType,
                                                         Undefined]] = 50,
                            color_map: Optional[Union[ColorMap, Undefined]] = None,
                            labels: Optional[Union[np.ndarray,
                                                         Undefined]] = None,
                            label_color_array: Optional[Union[list[tuple[Vector3Type]], Undefined]] = None):
        # TODO better check, we must handle all errors before sent to frontend.
        assert points.ndim == 2 and points.shape[1] in [
            3, 4
        ], "only support 3 or 4 features for points"
        if limit is not None:
            assert points.shape[
                0] <= limit, f"your points size {points.shape[0]} must smaller than limit {limit}"
        else:
            assert points.shape[
                0] <= self.props.limit, f"your points size {points.shape[0]} must smaller than limit {self.props.limit}"

        assert points.dtype == np.float32, "only support fp32 points"
        if points.shape[1] == 4 and colors is None:
            colors = points[:, 3].astype(np.uint8)
            points = points[:, :3]
        self._check_colors(colors, points)
        if encode_method == "int16":
            upd: dict[str, Any] = {
                "points": (points * encode_scale).astype(np.int16),
                "encodeMethod": "int16",
                "encodeScale": encode_scale,
                "sizeAttenuation": size_attenuation,
            }
        else:
            upd: dict[str, Any] = {
                "points": points,
                "sizeAttenuation": size_attenuation,
            }
        if size is not None:
            upd["size"] = size
        if color_map is not None:
            upd["colorMap"] = color_map
        if label_color_array is not None:
            if not isinstance(label_color_array, Undefined):
                assert len(label_color_array) <= 20, "max 20 label color"
            upd["labelColorArray"] = label_color_array

        if sizes is not None:
            if not isinstance(sizes, Undefined):
                assert sizes.shape[0] == points.shape[
                    0] and sizes.dtype == np.float32
            upd["sizes"] = sizes
            self.props.sizes = sizes
        if labels is not None:
            if not isinstance(labels, Undefined):
                assert labels.shape[0] == points.shape[
                    0] and (labels.dtype == np.int32 or labels.dtype == np.uint8)
                if label_color_array is None and isinstance(self.props.labelColorArray, Undefined):
                    self.props.labelColorArray = self._get_ui_label_color_array()
                    upd["labelColorArray"] = self.props.labelColorArray
            upd["labels"] = labels
            self.props.labels = labels

        if colors is not None:
            upd["colors"] = colors
            self.props.colors = colors
        if attrs is not None:
            self.props.attrs = attrs
            if not isinstance(attrs, Undefined):
                if attrs.ndim == 1:
                    attrs = attrs.reshape(-1, 1)
                if attr_fields is None:
                    attr_fields = [f"{i}" for i in range(attrs.shape[1])]
            upd["attrs"] = attrs
            upd["attrFields"] = attr_fields
            if attr_fields is not None:
                self.props.attrFields = attr_fields
        if limit is not None:
            assert limit > 0
            upd["limit"] = limit
            self.props.limit = limit
        self.props.points = points
        await self.send_and_wait(self.create_update_event(upd))

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass(config=PyDanticConfigForNumpy)
class SegmentsProps(ThreeBasicProps):
    limit: int = 0
    lines: Union[np.ndarray, Undefined] = undefined
    colors: Union[np.ndarray, Undefined] = undefined
    lineWidth: float = 1.0
    color: Annotated[Union[str, Undefined], typemetas.ColorRGB()] = undefined
    transparent: Union[bool, Undefined] = undefined
    opacity: Annotated[Union[NumberType, Undefined],
                       typemetas.CommonObject(default=1.0)] = undefined
    layers: Annotated[Union[int, Undefined],
                      typemetas.RangedInt(0, 31, 1, default=0)] = undefined

    @staticmethod
    def lines_validator(v: np.ndarray):
        assert v.dtype == np.float32
        assert v.ndim == 3 and v.shape[1] == 2 and v.shape[2] == 3

    @field_validator('lines')
    def _lines_validator(cls, v: Union[np.ndarray, Undefined]):
        if not isinstance(v, Undefined):
            cls.lines_validator(v)
        return v


class Segments(ThreeComponentBase[SegmentsProps]):

    def __init__(
        self,
        limit: int,
        line_width: float = 1.0,
        color: Annotated[Union[str, Undefined],
                         typemetas.ColorRGB()] = undefined
    ) -> None:
        super().__init__(UIType.ThreeSegments, SegmentsProps)
        self.props.lines = np.zeros((0, 2, 3), np.float32)
        self.props.lineWidth = line_width
        self.props.limit = limit
        self.props.colors = undefined
        self.props.color = color

    def get_sync_props(self) -> dict[str, Any]:
        res = super().get_sync_props()
        res["lines"] = self.props.lines
        res["colors"] = self.props.colors
        return res

    def validate_props(self, props: dict[str, Any]):
        if "lines" in props:
            return props["lines"].shape[0] <= self.props.limit
        return False

    async def clear(self):
        self.props.lines = np.zeros((0, 2, 3), np.float32)
        self.props.colors = undefined
        return self.send_and_wait(
            self.update_event(lines=undefined, colors=undefined))

    async def update_lines(self,
                           lines: np.ndarray,
                           colors: Optional[Union[np.ndarray,
                                                  Undefined]] = None,
                           limit: Optional[int] = None):
        assert lines.ndim == 3 and lines.shape[1] == 2 and lines.shape[
            2] == 3, f"{lines.shape} lines must be [N, 2, 3]"
        if limit is not None:
            assert lines.shape[
                0] <= limit, f"your points size {lines.shape[0]} must smaller than limit {limit}"
        else:
            assert lines.shape[
                0] <= self.props.limit, f"your points size {lines.shape[0]} must smaller than limit {self.props.limit}"
        upd: dict[str, Any] = {
            "lines": lines,
        }
        if colors is not None:
            if not isinstance(colors, Undefined):
                assert colors.shape[0] == lines.shape[
                    0], "color shape not valid"
            upd["colors"] = colors
            self.props.colors = colors
        if limit is not None:
            assert limit > 0
            upd["limit"] = limit
            self.props.limit = limit

        self.props.lines = lines.astype(np.float32)

        await self.send_and_wait(self.create_update_event(upd))

    async def update_mesh_lines(self, mesh: np.ndarray):
        mesh = mesh.reshape(-1, 3, 3)
        indexes = [0, 1, 1, 2, 2, 0]
        lines = np.stack([mesh[:, i] for i in indexes],
                         axis=1).reshape(-1, 2, 3)
        await self.update_lines(lines)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass(config=PyDanticConfigForNumpy)
class Boxes2DProps(Object3dBaseProps):
    centers: Union[np.ndarray, Undefined] = undefined
    dimensions: Union[np.ndarray, Undefined] = undefined
    colors: Union[np.ndarray, Undefined] = undefined
    attrs: Union[list[str], Undefined] = undefined
    color: Annotated[Union[str, Undefined], typemetas.ColorRGB()] = undefined
    alpha: Union[NumberType, Undefined] = undefined
    lineColor: Union[str, Undefined] = undefined
    lineWidth: Union[NumberType, Undefined] = undefined
    hoverLineColor: Union[str, Undefined] = undefined
    hoverLineWidth: Union[NumberType, Undefined] = undefined


class Boxes2D(Object3dWithEventBase[Boxes2DProps]):

    def __init__(self, limit: int) -> None:
        super().__init__(UIType.ThreeBoxes2D, Boxes2DProps)
        self.props.centers = np.zeros((0, 2), np.float32)
        self.props.dimensions = np.zeros((0, 2), np.float32)
        self.limit = limit

    def to_dict(self):
        res = super().to_dict()
        res["limit"] = self.limit
        return res

    def get_sync_props(self) -> dict[str, Any]:
        res = super().get_sync_props()
        res["centers"] = self.props.centers
        res["dimensions"] = self.props.dimensions
        res["colors"] = self.props.colors
        return res

    def validate_props(self, props: dict[str, Any]):
        if "centers" in props:
            res = props["centers"].shape[0] <= self.limit
        else:
            res = False
        return res

    async def update_boxes(self,
                           centers: Optional[np.ndarray] = None,
                           dimensions: Optional[np.ndarray] = None,
                           colors: Optional[Union[np.ndarray,
                                                  Undefined]] = None,
                           attrs: Optional[Union[list[str],
                                                 Undefined]] = None):
        # TODO check props in
        assert not isinstance(self.props.centers, Undefined)
        upd: dict[str, Any] = {}
        if centers is not None:
            assert centers.shape[
                0] <= self.limit, f"your centers size must smaller than limit {self.limit}"
            self.props.centers = centers
            upd["centers"] = centers
        if dimensions is not None:
            if dimensions.ndim == 1:
                assert dimensions.shape[0] in [
                    1, 2
                ], "dimersion must be [1] or [2]"
            else:
                assert dimensions.shape[
                    0] <= self.limit, f"your dimensions size must smaller than limit {self.limit}"
            self.props.dimensions = dimensions
            if dimensions.shape != self.props.centers.shape:
                # check broadcastable
                np.broadcast_shapes(self.props.centers.shape, dimensions.shape)
            upd["dimensions"] = dimensions
        if colors is not None:
            if not isinstance(colors, Undefined):
                assert colors.shape[
                    0] <= self.limit, f"your colors size must smaller than limit {self.limit}"
            self.props.colors = colors
            upd["colors"] = colors
        if attrs is not None:
            self.props.attrs = attrs
            upd["attrs"] = attrs
        await self.send_and_wait(self.create_update_event(upd))

    def get_props_dict(self):
        state = super().get_props_dict()
        dims = self.props.dimensions
        centers = self.props.centers
        if not isinstance(dims, Undefined) and not isinstance(
                centers, Undefined):
            if dims.shape != centers.shape:
                dims = np.broadcast_to(dims, centers.shape)
        state.update({
            "colors": self.props.colors,
            "centers": self.props.centers,
            "dimensions": dims,
            "attrs": self.props.attrs,
        })
        return state

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class BoundingBoxProps(Object3dBaseProps, InteractableProps):
    dimension: Union[Vector3Type, Undefined] = undefined
    edgeWidth: Union[float, Undefined] = undefined
    edgeColor: Annotated[Union[str, int, Undefined],
                         typemetas.ColorRGB(default="green")] = undefined
    emissive: Annotated[Union[str, int, Undefined],
                        typemetas.ColorRGB(default="red")] = undefined
    color: Annotated[Union[str, int, Undefined],
                     typemetas.ColorRGB(default="red")] = undefined
    opacity: Annotated[Union[NumberType, Undefined],
                       typemetas.CommonObject(default=0.3)] = undefined
    edgeOpacity: Annotated[Union[NumberType, Undefined],
                           typemetas.CommonObject(default=0.5)] = undefined
    checked: bool = False
    addCross: bool = True


class BoundingBox(Object3dWithEventBase[BoundingBoxProps]):

    def __init__(self,
                 dimension: Vector3Type,
                 edge_width: float = 1,
                 edge_color: str = "green",
                 emissive: str = "red",
                 color: str = "red",
                 opacity: float = 0.3,
                 edge_opacity: float = 0.5) -> None:
        super().__init__(UIType.ThreeBoundingBox, BoundingBoxProps)
        self.props.dimension = dimension
        self.props.edgeWidth = edge_width
        self.props.edgeColor = edge_color
        self.props.emissive = emissive
        self.props.color = color
        self.props.opacity = opacity
        self.props.edgeOpacity = edge_opacity

    def get_sync_props(self) -> dict[str, Any]:
        res = super().get_sync_props()
        res["dimension"] = self.props.dimension
        return res

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        await handle_standard_event(self, ev, is_sync=is_sync)

    def state_change_callback(
            self,
            value: bool,
            type: ValueType = FrontendEventType.Change.value):
        if isinstance(value, bool):
            self.props.checked = value
        elif isinstance(value, dict):
            assert "position" in value
            assert "rotation" in value
            self.props.position = value["position"]
            self.props.rotation = value["rotation"]


@dataclasses.dataclass
class AxesHelperProps(Object3dBaseProps):
    length: NumberType = 10


class AxesHelper(ThreeComponentBase[AxesHelperProps]):

    def __init__(self, length: float) -> None:
        super().__init__(UIType.ThreeAxesHelper, AxesHelperProps)
        self.props.length = length

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class EdgesProps(ThreeBasicProps):
    threshold: Union[NumberType, Undefined] = undefined
    color: Union[ValueType, Undefined] = undefined
    scale: Union[NumberType, Undefined] = undefined


class Edges(ThreeComponentBase[EdgesProps]):

    def __init__(
        self,
        threshold: Union[NumberType, Undefined] = undefined,
        color: Union[ValueType, Undefined] = undefined,
        scale: Union[NumberType, Undefined] = undefined,
    ) -> None:
        super().__init__(UIType.ThreeEdges, EdgesProps)
        self.props.threshold = threshold
        self.props.color = color
        self.props.scale = scale

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class WireframeProps(ThreeBasicProps):
    fillOpacity: Union[NumberType, Undefined] = undefined
    fillMix: Union[NumberType, Undefined] = undefined
    strokeOpacity: Union[NumberType, Undefined] = undefined
    thickness: Union[NumberType, Undefined] = undefined
    colorBackfaces: Union[bool, Undefined] = undefined
    dashInvert: Union[bool, Undefined] = undefined
    dash: Union[bool, Undefined] = undefined
    dashRepeats: Union[NumberType, Undefined] = undefined
    dashLength: Union[NumberType, Undefined] = undefined
    squeeze: Union[bool, Undefined] = undefined
    squeezeMin: Union[NumberType, Undefined] = undefined
    squeezeMax: Union[NumberType, Undefined] = undefined
    stroke: Union[ValueType, Undefined] = undefined
    backfaceStroke: Union[ValueType, Undefined] = undefined
    fill: Union[ValueType, Undefined] = undefined


class Wireframe(ThreeComponentBase[WireframeProps]):
    """used in Mesh childs.
    """

    def __init__(self) -> None:
        super().__init__(UIType.ThreeWireframe, WireframeProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class InfiniteGridHelperProps(Object3dBaseProps):
    size1: Annotated[Union[NumberType, Undefined],
                     RangedFloat(1, 50, 0.1)] = undefined
    size2: Annotated[Union[NumberType, Undefined],
                     RangedFloat(1, 200, 0.5)] = undefined
    color: Annotated[Union[str, Undefined], typemetas.ColorRGB()] = undefined
    distance: Union[NumberType, Undefined] = undefined
    axes: Union[str, Undefined] = undefined


class InfiniteGridHelper(ThreeComponentBase[InfiniteGridHelperProps]):

    def __init__(self,
                 size1: float,
                 size2: float,
                 color: str,
                 distance: float = 8000) -> None:
        super().__init__(UIType.ThreeInfiniteGridHelper,
                         InfiniteGridHelperProps)
        self.props.size1 = size1
        self.props.size2 = size2
        self.props.color = color
        self.props.distance = distance

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass(config=PyDanticConfigForNumpy)
class ImageProps(Object3dBaseProps, InteractableProps):
    image: Union[np.ndarray, str,
                 bytes] = dataclasses.field(default_factory=str)
    segments: Union[NumberType, Undefined] = undefined
    color: Annotated[Union[str, int, Undefined],
                     typemetas.ColorRGB()] = undefined
    zoom: Union[NumberType, Undefined] = undefined
    grayscale: Union[NumberType, Undefined] = undefined
    toneMapped: Union[bool, Undefined] = undefined
    transparent: Union[bool, Undefined] = undefined
    opacity: Union[NumberType, Undefined] = undefined
    # if image isn't dynamic, you should cache it, otherwise
    # don't cache it. False by default.
    cached: Union[bool, Undefined] = undefined

class Image(Object3dWithEventBase[ImageProps]):

    def __init__(self) -> None:
        super().__init__(UIType.ThreeImage, ImageProps)

    async def show(self, image: np.ndarray):
        encoded = MUIImage.encode_image_bytes(image)
        self.props.image = encoded
        await self.send_and_wait(self.create_update_event({
            "image": encoded,
        }))

    async def clear(self):
        self.props.image = b''
        await self.send_and_wait(self.update_event(image=b''))

    async def show_raw(self, image_bytes: bytes, suffix: str):
        await self.send_and_wait(self.show_raw_event(image_bytes, suffix))

    def encode_raw_to_web(self, raw: bytes, suffix: str):
        return b'data:image/' + suffix.encode(
            "utf-8") + b';base64,' + base64.b64encode(raw)

    def show_raw_event(self, image_bytes: bytes, suffix: str):
        raw = b'data:image/' + suffix.encode(
            "utf-8") + b';base64,' + base64.b64encode(image_bytes)
        self.props.image = raw
        return self.create_update_event({
            "image": raw,
        })

    def get_sync_props(self) -> dict[str, Any]:
        res = super().get_sync_props()
        res["image"] = self.props.image
        return res

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

@dataclasses.dataclass
class ScreenShotProps(ThreeBasicProps):
    pass


class ScreenShot(ThreeComponentBase[ScreenShotProps]):
    """a special ui to get screen shot. steps:
    1. use trigger_screen_shot with userdata
    2. get image and userdata you provided from callback.
    currently impossible to get image from one function call.
    """

    def __init__(self, callback: Callable[[tuple[str, Any]],
                                          _CORO_NONE]) -> None:
        super().__init__(UIType.ThreeScreenShot,
                         ScreenShotProps,
                         allowed_events=[FrontendEventType.Change.value])
        self.register_event_handler(FrontendEventType.Change.value, callback)
        self.event_change = self._create_event_slot(FrontendEventType.Change)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def trigger_screen_shot(self, data: Optional[Any] = None):
        """when you provide a data, we will use image and 
        this data to call your callback
        """
        # check data is can be converted to json
        x = json.dumps(data)
        assert len(x) < 1000 * 1000
        await self.send_and_wait(
            self.create_comp_event({
                "type": 0,
                "data": data,
            }))

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self,
                                           ev,
                                           sync_status_first=True,
                                           is_sync=is_sync)


class _PendingState:

    def __init__(self,
                 ev: asyncio.Event,
                 result: Optional[Any] = None) -> None:
        self.ev = ev
        self.result = result


class ScreenShotSyncReturn(ThreeComponentBase[ScreenShotProps]):
    """a special ui to get screen shot. steps:
    1. use trigger_screen_shot with userdata
    2. get image and userdata you provided from callback.
    currently impossible to get image from one function call.
    """

    def __init__(self) -> None:
        super().__init__(UIType.ThreeScreenShot,
                         ScreenShotProps,
                         allowed_events=[FrontendEventType.Change.value])
        self.register_event_handler(FrontendEventType.Change.value,
                                    self._on_callback)
        self._pending_rpc: dict[int, _PendingState] = {}
        self._uid_index = 0
        self.event_change = self._create_event_slot(FrontendEventType.Change)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def _on_callback(self, data: tuple[str, Any]):
        img_data = data[0]
        uid = data[1]
        if uid in self._pending_rpc:
            self._pending_rpc[uid].ev.set()
            self._pending_rpc[uid].result = img_data

    async def get_screen_shot(self, timeout=2):
        uid = self._uid_index % JS_MAX_SAFE_INT
        await self.send_and_wait(
            self.create_comp_event({
                "type": 0,
                "data": uid,
            }))
        self._uid_index += 1
        ev = asyncio.Event()
        self._pending_rpc[uid] = _PendingState(ev, None)
        try:
            await asyncio.wait_for(ev.wait(), timeout=timeout)
            res = self._pending_rpc.pop(uid).result
            assert res is not None
            if isinstance(res, bytes):
                return res
            return urllib.request.urlopen(res).read()
        except:
            if uid in self._pending_rpc:
                self._pending_rpc.pop(uid)
            raise

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self,
                                           ev,
                                           sync_status_first=True,
                                           sync_state_after_change=False,
                                           is_sync=is_sync)

@dataclasses.dataclass
class HtmlProps(Object3dContainerBaseProps):
    prepend: Union[bool, Undefined] = undefined
    center: Union[bool, Undefined] = undefined
    fullscreen: Union[bool, Undefined] = undefined
    eps: Union[float, Undefined] = undefined
    distanceFactor: Union[float, Undefined] = undefined
    sprite: Union[bool, Undefined] = undefined
    transform: Union[bool, Undefined] = undefined
    zIndexRange: Union[list[Union[int, float]], Undefined] = undefined
    wrapperClass: Union[str, Undefined] = undefined
    pointerEvents: Union[PointerEventsProperties, Undefined] = undefined
    occlude: Union[bool, Undefined] = undefined
    insideFlex: Union[bool, Undefined] = undefined


class Html(Object3dContainerBase[HtmlProps, MUIComponentType]):
    """we can use MUI components only in Html.
    TODO reject invalid component
    """

    def __init__(self, children: dict[str, MUIComponentType]) -> None:
        super().__init__(UIType.ThreeHtml, HtmlProps, children)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class TextProps(Object3dBaseProps):
    value: Union[str, list[str]] = ""
    positions: Union[np.ndarray, list[Vector3Type], Undefined] = undefined
    colors: Union[np.ndarray, list[Vector3Type], Undefined] = undefined
    characters: Union[str, Undefined] = undefined
    color: Annotated[Union[str, int, Undefined],
                     typemetas.ColorRGB(default="white")] = undefined
    fontSize: Annotated[
        Union[NumberType, Undefined],
        typemetas.RangedFloat(0.1, 20, 0.02, default=1)] = undefined
    maxWidth: Union[NumberType, Undefined] = undefined
    lineHeight: Union[NumberType, Undefined] = undefined
    letterSpacing: Union[NumberType, Undefined] = undefined
    textAlign: Union[Literal["left", "right", "center", "justify"],
                     Undefined] = undefined
    font: Union[str, Undefined] = undefined
    anchorX: Union[NumberType, Literal["left", "center", "right"],
                   Undefined] = undefined
    anchorY: Union[NumberType, Literal["top", "top-baseline", "middle",
                                       "bottom-baseline", "bottom"],
                   Undefined] = undefined
    clipRect: Union[tuple[NumberType, NumberType, NumberType, NumberType],
                    Undefined] = undefined
    depthOffset: Union[NumberType, Undefined] = undefined
    direction: Union[Literal["auto", "ltr", "rtl"], Undefined] = undefined
    overflowWrap: Union[Literal["normal", "break-word"], Undefined] = undefined
    whiteSpace: Union[Literal['normal', 'nowrap'], Undefined] = undefined
    outlineWidth: Union[ValueType, Undefined] = undefined
    outlineOffsetX: Union[ValueType, Undefined] = undefined
    outlineOffsetY: Union[ValueType, Undefined] = undefined
    outlineBlur: Union[ValueType, Undefined] = undefined
    outlineColor: Union[str, Undefined] = undefined
    outlineOpacity: Union[NumberType, Undefined] = undefined
    strokeWidth: Union[ValueType, Undefined] = undefined
    strokeColor: Union[NumberType, Undefined] = undefined
    strokeOpacity: Union[NumberType, Undefined] = undefined
    fillOpacity: Union[NumberType, Undefined] = undefined

    @model_validator(mode="after")
    def _check_value(self) -> Self:
        if not isinstance(self.value, str):
            assert isinstance(self.value, list)
            assert isinstance(self.positions, (list, np.ndarray))
            assert len(self.value) == len(self.positions), "value and positions must have same length"
            if isinstance(self.positions, np.ndarray):
                assert self.positions.ndim == 2 and self.positions.shape[1] in [2, 3], "positions must be [N, 2/3] array"
        return self

class Text(Object3dWithEventBase[TextProps]):

    def __init__(self, init: Union[str, list[str]], positions: Optional[Union[np.ndarray, list[Vector3Type]]] = None) -> None:
        super().__init__(UIType.ThreeText, TextProps)
        if isinstance(init, str):
            assert positions is None 
        else:
            assert positions is not None
        self.props.value = init
        if positions is not None:
            self.props.positions = positions

    def get_sync_props(self) -> dict[str, Any]:
        res = super().get_sync_props()
        res["value"] = self.props.value
        return res

    async def update_value(self, value: str):
        assert isinstance(value, str)
        self.props.value = value
        upd: dict[str, Any] = {"value": value}
        await self.send_and_wait(self.create_update_event(upd))

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class LineProps(Object3dBaseProps):
    points: Union[list[tuple[NumberType, NumberType,
                       NumberType]], np.ndarray] = dataclasses.field(default_factory=list)
    color: Annotated[Union[str, Undefined], typemetas.ColorRGB()] = undefined
    dashed: Union[bool, Undefined] = undefined
    dashSize: Union[NumberType, Undefined] = undefined
    gapSize: Union[NumberType, Undefined] = undefined
    vertexColors: Union[tuple[NumberType, NumberType, NumberType], np.ndarray,
                        Undefined] = undefined
    lineWidth: Union[NumberType, Undefined] = undefined
    segments: Union[bool, Undefined] = undefined
    transparent: Union[bool, Undefined] = undefined
    opacity: Annotated[Union[NumberType, Undefined],
                       typemetas.CommonObject(default=1.0)] = undefined
    variant: Union[Literal["default", "aabb"], Undefined] = undefined
    aabbSizes: Union[tuple[NumberType, NumberType,
                       NumberType], np.ndarray, Undefined] = undefined

class Line(Object3dWithEventBase[LineProps]):

    def __init__(
        self, points: Union[np.ndarray, list[tuple[NumberType, NumberType,
                                                   NumberType]]]
    ) -> None:
        super().__init__(UIType.ThreeLine, LineProps)
        if isinstance(points, np.ndarray):
            nelem = points.shape[1]
            assert points.ndim == 2 and (nelem == 3 or nelem == 2) and points.dtype == np.float32, "must be [N, 3]/[N, 2] array"
            self.props.points = points
        else:
            self.props.points = points

    def get_sync_props(self) -> dict[str, Any]:
        res = super().get_sync_props()
        res["points"] = self.props.points
        return res

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class LineShapeProps(Object3dBaseProps):
    pathOps: list[tuple[int, list[Union[float, bool]]]] = dataclasses.field(
        default_factory=list)
    color: Annotated[Union[str, Undefined], typemetas.ColorRGB()] = undefined
    dashed: Union[bool, Undefined] = undefined
    dashSize: Union[NumberType, Undefined] = undefined
    gapSize: Union[NumberType, Undefined] = undefined
    lineWidth: Union[NumberType, Undefined] = undefined
    transparent: Union[bool, Undefined] = undefined
    opacity: Annotated[Union[NumberType, Undefined],
                       typemetas.CommonObject(default=1.0)] = undefined
    divisions: Union[NumberType, Undefined] = undefined

class LineShape(Object3dWithEventBase[LineShapeProps]):

    def __init__(self, shape: Shape) -> None:
        super().__init__(UIType.ThreeLineShape, LineShapeProps)
        self.props.pathOps = shape.ops

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

@dataclasses.dataclass
class ContactShadowsProps(Object3dBaseProps):
    opacity: Annotated[Union[NumberType, Undefined],
                       typemetas.CommonObject(default=1.0)] = undefined
    width: Union[NumberType, Undefined] = undefined
    height: Union[NumberType, Undefined] = undefined
    blur: Union[NumberType, Undefined] = undefined
    near: Union[NumberType, Undefined] = undefined
    far: Union[NumberType, Undefined] = undefined
    smooth: Union[bool, Undefined] = undefined
    resolution: Union[NumberType, Undefined] = undefined
    frames: Union[int, Undefined] = undefined
    scale: Union[NumberType, tuple[NumberType, NumberType],
                 Undefined] = undefined
    color: Annotated[Union[int, str, Undefined],
                     typemetas.ColorRGB()] = undefined
    depthWrite: Union[bool, Undefined] = undefined


class ContactShadows(ThreeComponentBase[ContactShadowsProps]):

    def __init__(self) -> None:
        super().__init__(UIType.ThreeContactShadows, ContactShadowsProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class GizmoHelperShadowsProps(ThreeBasicProps):
    alignment: Union[Literal["top-left", "top-right", "bottom-right",
                             "bottom-left", "bottom-center", "center-right",
                             "center-left", "center-center", "top-center"],
                     Undefined] = undefined
    margin: Union[tuple[NumberType, NumberType], Undefined] = undefined
    renderPriority: Union[int, Undefined] = undefined
    autoClear: Union[bool, Undefined] = undefined
    axisColors: Union[tuple[str, str, str], Undefined] = undefined
    axisScale: Union[tuple[NumberType, NumberType, NumberType],
                     Undefined] = undefined
    labels: Union[tuple[str, str, str], Undefined] = undefined
    axisHeadScale: Union[NumberType, Undefined] = undefined
    labelColor: Annotated[Union[str, Undefined],
                          typemetas.ColorRGB()] = undefined
    hideNegativeAxes: Union[bool, Undefined] = undefined
    hideAxisHeads: Union[bool, Undefined] = undefined
    disabled: Union[bool, Undefined] = undefined
    font: Union[str, Undefined] = undefined


class GizmoHelper(ThreeComponentBase[GizmoHelperShadowsProps]):

    def __init__(self) -> None:
        super().__init__(UIType.ThreeGizmoHelper, GizmoHelperShadowsProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class SelectionContextProps(ContainerBaseProps):
    multiple: Union[bool, Undefined] = undefined
    box: Union[bool, Undefined] = undefined
    border: Union[str, Undefined] = undefined
    backgroundColor: Union[str, Undefined] = undefined
    useOutline: Union[bool, Undefined] = undefined


class SelectionContext(ThreeContainerBase[SelectionContextProps,
                                          ThreeComponentType]):

    def __init__(
            self,
            children: Optional[ThreeLayoutType] = None,
            callback: Optional[Callable[[Any], _CORO_NONE]] = None) -> None:
        if children is None:
            children = {}
        if isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}

        super().__init__(UIType.ThreeSelectionContext,
                         SelectionContextProps, {**children},
                         allowed_events=[FrontendEventType.Change.value])
        self.event_change = self._create_event_slot(FrontendEventType.Change)
        if callback is not None:
            self.event_change.on(callback)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        await handle_standard_event(self,
                                    ev,
                                    sync_state_after_change=False,
                                    change_status=False,
                                    is_sync=is_sync)

@dataclasses.dataclass
class OutlinesProps(ThreeBasicProps):
    color: Annotated[Union[str, Undefined], typemetas.ColorRGB()] = undefined
    opacity: Union[Undefined, NumberType] = undefined
    transparent: Union[Undefined, bool] = undefined
    thickness: Union[Undefined, NumberType] = undefined
    angle: Union[Undefined, NumberType] = undefined


class Outlines(ThreeComponentBase[OutlinesProps]):

    def __init__(self) -> None:
        super().__init__(UIType.ThreeOutlines, OutlinesProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

@dataclasses.dataclass
class BvhProps(ContainerBaseProps):
    splitStrategy: Union[Literal["CENTER", "AVERAGE", "SAH"], Undefined] = undefined
    verbose: Union[bool, Undefined] = undefined
    setBoundingBox: Union[bool, Undefined] = undefined
    maxDepth: Union[int, Undefined] = undefined
    maxLeafTris: Union[int, Undefined] = undefined
    indirect: Union[bool, Undefined] = undefined


class Bvh(ThreeContainerBase[BvhProps, ThreeComponentType]):
    def __init__(
            self,
            children: Optional[ThreeLayoutType] = None) -> None:
        if children is None:
            children = {}
        if isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}

        super().__init__(UIType.ThreeBVH,
                         BvhProps, {**children},
                         allowed_events=[])

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

