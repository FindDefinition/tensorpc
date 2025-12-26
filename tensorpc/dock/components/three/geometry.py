from typing_extensions import Annotated
from typing import (Union)

import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.annolib import Undefined, undefined
from tensorpc.core.datamodel import typemetas
from tensorpc.dock.core.component import (UIType)

from .base import ThreeGeometryPropsBase, NumberType, ThreeGeometryBase, GeometryType, PathOpType
import numpy as np 

@dataclasses.dataclass
class SimpleGeometryProps(ThreeGeometryPropsBase):
    shapeType: int = 0
    shapeArgs: Union[list[Union[int, float, bool]], Undefined] = undefined


@dataclasses.dataclass
class PathShapeProps(ThreeGeometryPropsBase):
    pathOps: list[tuple[int, list[Union[float, bool]]]] = dataclasses.field(
        default_factory=list)
    curveSegments: Union[NumberType, Undefined] = undefined

class SimpleGeometry(ThreeGeometryBase[SimpleGeometryProps]):

    def __init__(self, type: GeometryType, args: list[Union[int, float,
                                                            bool]]) -> None:
        super().__init__(UIType.ThreeSimpleGeometry, SimpleGeometryProps)
        self.props.shapeType = type.value
        self.props.shapeArgs = args

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class BoxGeometry(SimpleGeometry):

    def __init__(self,
                 width: float = 1,
                 height: float = 1,
                 depth: float = 1,
                 width_segments: int = 1,
                 height_segments: int = 1,
                 depth_segments: int = 1) -> None:
        args: list[Union[int, float, bool]] = [
            width, height, depth, width_segments, height_segments,
            depth_segments
        ]
        super().__init__(GeometryType.Box, args)


class CapsuleGeometry(SimpleGeometry):

    def __init__(self,
                 radius: float = 1,
                 length: float = 1,
                 cap_subdivisions: int = 4,
                 radial_segments: int = 8) -> None:
        args: list[Union[int, float, bool]] = [
            radius, length, cap_subdivisions, radial_segments
        ]
        super().__init__(GeometryType.Capsule, args)


class PlaneGeometry(SimpleGeometry):

    def __init__(
        self,
        width: float = 1,
        height: float = 1,
        width_segments: int = 1,
        height_segments: int = 1,
    ) -> None:
        args: list[Union[int, float, bool]] = [
            width, height, width_segments, height_segments
        ]
        super().__init__(GeometryType.Plane, args)


class CircleGeometry(SimpleGeometry):

    def __init__(self,
                 radius: float = 1,
                 segments: int = 8,
                 theta_start: float = 0,
                 theta_length: float = np.pi * 2) -> None:
        args: list[Union[int, float, bool]] = [
            radius, segments, theta_start, theta_length
        ]
        super().__init__(GeometryType.Circle, args)


class ConeGeometry(SimpleGeometry):

    def __init__(self,
                 radius: float = 1,
                 height: float = 1,
                 radial_segments: int = 32,
                 height_segments: int = 1,
                 open_ended: bool = False,
                 theta_start: float = 0,
                 theta_length: float = np.pi * 2) -> None:
        args: list[Union[int, float, bool]] = [
            radius, height, radial_segments, height_segments, open_ended,
            theta_start, theta_length
        ]
        super().__init__(GeometryType.Cone, args)


class CylinderGeometry(SimpleGeometry):

    def __init__(self,
                 radius_top: float = 1,
                 radius_bottom: float = 1,
                 height: float = 1,
                 radial_segments: int = 32,
                 height_segments: int = 1,
                 open_ended: bool = False,
                 theta_start: float = 0,
                 theta_length: float = np.pi * 2) -> None:
        args: list[Union[int, float, bool]] = [
            radius_top, radius_bottom, height, radial_segments,
            height_segments, open_ended, theta_start, theta_length
        ]
        super().__init__(GeometryType.Cylinder, args)


class DodecahedronGeometry(SimpleGeometry):

    def __init__(self, radius: float = 1, detail: int = 0) -> None:
        args: list[Union[int, float, bool]] = [radius, detail]
        super().__init__(GeometryType.Dodecahedron, args)


class IcosahedronGeometry(SimpleGeometry):

    def __init__(self, radius: float = 1, detail: int = 0) -> None:
        args: list[Union[int, float, bool]] = [radius, detail]
        super().__init__(GeometryType.Icosahedron, args)


class OctahedronGeometry(SimpleGeometry):

    def __init__(self, radius: float = 1, detail: int = 0) -> None:
        args: list[Union[int, float, bool]] = [radius, detail]
        super().__init__(GeometryType.Octahedron, args)


class TetrahedronGeometry(SimpleGeometry):

    def __init__(self, radius: float = 1, detail: int = 0) -> None:
        args: list[Union[int, float, bool]] = [radius, detail]
        super().__init__(GeometryType.Tetrahedron, args)


class RingGeometry(SimpleGeometry):

    def __init__(self,
                 inner_radius: float = 0.5,
                 outer_radius: float = 1,
                 theta_segments: int = 32,
                 phi_segments: int = 1,
                 theta_start: float = 0,
                 theta_length: float = np.pi * 2) -> None:
        args: list[Union[int, float, bool]] = [
            inner_radius, outer_radius, theta_segments, phi_segments,
            theta_start, theta_length
        ]
        super().__init__(GeometryType.Ring, args)


class SphereGeometry(SimpleGeometry):

    def __init__(self,
                 radius: float = 1,
                 widthSegments: int = 32,
                 heightSegments: int = 16,
                 phi_start: float = 0,
                 phi_length: float = np.pi * 2,
                 theta_start: float = 0,
                 theta_length: float = np.pi) -> None:
        args: list[Union[int, float, bool]] = [
            radius, widthSegments, heightSegments, phi_start, phi_length,
            theta_start, theta_length
        ]
        super().__init__(GeometryType.Sphere, args)


class TorusGeometry(SimpleGeometry):

    def __init__(self,
                 radius: float = 1,
                 tube: float = 0.4,
                 radial_segments: int = 12,
                 tubular_segments: int = 48,
                 arc: float = np.pi * 2) -> None:
        args: list[Union[int, float, bool]] = [
            radius, tube, radial_segments, tubular_segments, arc
        ]
        super().__init__(GeometryType.Torus, args)


class TorusKnotGeometry(SimpleGeometry):

    def __init__(self,
                 radius: float = 1,
                 tube: float = 0.4,
                 tubular_segments: int = 64,
                 radial_segments: int = 8,
                 p: int = 2,
                 q: int = 3) -> None:
        args: list[Union[int, float, bool]] = [
            radius, tube, tubular_segments, radial_segments, p, q
        ]
        super().__init__(GeometryType.TorusKnot, args)

class Shape:

    def __init__(self) -> None:
        self.ops: list[tuple[int, list[Union[float, bool]]]] = []

    def move_to(self, x: float, y: float):
        self.ops.append((PathOpType.Move.value, [x, y]))

    def line_to(self, x: float, y: float):
        self.ops.append((PathOpType.Line.value, [x, y]))

    def absarc(self,
               x: float,
               y: float,
               radius: float,
               startAngle: float,
               endAngle: float,
               clockwise: bool = False):
        self.ops.append((PathOpType.AbsArc.value,
                         [x, y, radius, startAngle, endAngle, clockwise]))

    def arc(self,
            x: float,
            y: float,
            radius: float,
            startAngle: float,
            endAngle: float,
            clockwise: bool = False):
        self.ops.append((PathOpType.Arc.value,
                         [x, y, radius, startAngle, endAngle, clockwise]))

    def bezier_curve_to(self, cp1X: float, cp1Y: float, cp2X: float,
                        cp2Y: float, x: float, y: float):
        self.ops.append((PathOpType.Arc.value, [cp1X, cp1Y, cp2X, cp2Y, x, y]))

    def quadratic_curve_to(self, cpX: float, cpY: float, x: float, y: float):
        self.ops.append((PathOpType.QuadraticCurve.value, [cpX, cpY, x, y]))

    @classmethod 
    def from_aabb(cls, center_x: float, center_y: float, width: float, height: float):
        x = center_x - width / 2
        y = center_y - height / 2
        ctx = cls()
        ctx.move_to(x, y)
        ctx.line_to(x + width, y)
        ctx.line_to(x + width, y + height)
        ctx.line_to(x, y + height)
        ctx.line_to(x, y)
        return ctx

class ShapeGeometry(ThreeGeometryBase[PathShapeProps]):

    def __init__(self, shape: Shape) -> None:
        super().__init__(UIType.ThreeShape, PathShapeProps)
        self.props.pathOps = shape.ops

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


def _rounded_shape(x: float, y: float, w: float, h: float, r: float):
    ctx = Shape()
    ctx.move_to(x, y + r)
    ctx.line_to(x, y + h - r)
    ctx.quadratic_curve_to(x, y + h, x + r, y + h)
    ctx.line_to(x + w - r, y + h)
    ctx.quadratic_curve_to(x + w, y + h, x + w, y + h - r)
    ctx.line_to(x + w, y + r)
    ctx.quadratic_curve_to(x + w, y, x + w - r, y)
    ctx.line_to(x + r, y)
    ctx.quadratic_curve_to(x, y, x, y + r)
    return ctx


def _rounded_shape_v2(x: float, y: float, w: float, h: float, r: float):
    ctx = Shape()
    eps = 1e-5
    r -= eps
    ctx.absarc(eps, eps, eps, -np.pi / 2, -np.pi, True)
    ctx.absarc(eps, h - r * 2, eps, np.pi, np.pi / 2, True)
    ctx.absarc(w - r * 2, h - r * 2, eps, np.pi / 2, 0, True)
    ctx.absarc(w - r * 2, eps, eps, 0, -np.pi / 2, True)
    return ctx


class RoundedRectGeometry(ShapeGeometry):

    def __init__(self, width: float, height: float, radius: float) -> None:
        shape = _rounded_shape(-width / 2, -height / 2, width, height, radius)
        super().__init__(shape)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)
