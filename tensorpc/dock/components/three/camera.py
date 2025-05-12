from typing_extensions import Annotated, Literal, TypeAlias
from typing import (Union, Any, Optional)
import enum 
import base64 

import numpy as np 
from pydantic import field_validator

import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.annolib import Undefined, undefined
from tensorpc.core.datamodel import typemetas
from tensorpc.dock.core.component import (UIType, FrontendEventType)
from collections.abc import Sequence
from tensorpc.dock.core import colors
from tensorpc.dock.core.appcore import Event, EventDataType
from tensorpc.dock.core.common import handle_standard_event
from tensorpc.core.datamodel.typemetas import RangedFloat, RangedInt

from .base import PerspectiveCameraProps, Object3dContainerBaseProps, NumberType, ThreeBasicProps, ThreeComponentBase, Object3dContainerBase, Vector3Type, ThreeComponentType, Object3dBase


class PerspectiveCamera(Object3dContainerBase[PerspectiveCameraProps,
                                              ThreeComponentType]):

    def __init__(
        self,
        make_default: bool = True,
        fov: Union[float, Undefined] = undefined,
        aspect: Union[float, Undefined] = undefined,
        near: Union[float, Undefined] = undefined,
        far: Union[float, Undefined] = undefined,
        position: Vector3Type = (0, 0, 1),
        up: Vector3Type = (0, 0, 1),
        children: Optional[Union[list[ThreeComponentType],
                                 dict[str, ThreeComponentType]]] = None
    ) -> None:
        if children is None:
            children = {}
        if isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.ThreePerspectiveCamera, PerspectiveCameraProps,
                         children)
        self.props.fov = fov
        self.props.aspect = aspect
        self.props.near = near
        self.props.far = far
        self.props.position = position
        self.props.up = up
        self.props.makeDefault = make_default

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class OrthographicCameraProps(Object3dContainerBaseProps):
    makeDefault: Union[bool, Undefined] = undefined
    zoom: Union[float, Undefined] = undefined
    near: Union[float, Undefined] = undefined
    far: Union[float, Undefined] = undefined


class OrthographicCamera(Object3dContainerBase[OrthographicCameraProps,
                                               ThreeComponentType]):

    def __init__(
        self,
        make_default: bool = True,
        near: Union[float, Undefined] = undefined,
        far: Union[float, Undefined] = undefined,
        zoom: Union[float, Undefined] = undefined,
        position: Vector3Type = (0, 0, 1),
        up: Vector3Type = (0, 0, 1),
        children: Optional[Union[list[ThreeComponentType],
                                 dict[str, ThreeComponentType]]] = None
    ) -> None:
        if children is None:
            children = {}
        if isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.ThreeOrthographicCamera,
                         OrthographicCameraProps, children)

        self.props.zoom = zoom
        self.props.near = near
        self.props.far = far
        self.props.position = position
        self.props.up = up
        self.props.makeDefault = make_default

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class OrbitControlProps(ThreeBasicProps):
    enableDamping: Union[bool, Undefined] = undefined
    dampingFactor: Union[NumberType, Undefined] = undefined
    minDistance: Union[NumberType, Undefined] = undefined
    maxDistance: Union[NumberType, Undefined] = undefined
    minPolarAngle: Union[NumberType, Undefined] = undefined
    maxPolarAngle: Union[NumberType, Undefined] = undefined
    minZoom: Union[NumberType, Undefined] = undefined
    maxZoom: Union[NumberType, Undefined] = undefined
    enableZoom: Union[bool, Undefined] = undefined
    zoomSpeed: Union[NumberType, Undefined] = undefined
    enableRotate: Union[bool, Undefined] = undefined
    rotateSpeed: Union[NumberType, Undefined] = undefined
    enablePan: Union[bool, Undefined] = undefined
    panSpeed: Union[NumberType, Undefined] = undefined
    keyPanSpeed: Union[NumberType, Undefined] = undefined
    makeDefault: Union[bool, Undefined] = undefined


# "rotate" | "dolly" | "truck" | "offset" | "zoom" | "none"
MouseButtonType: TypeAlias = Literal["rotate", "dolly", "truck", "offset",
                                     "zoom", "none"]


@dataclasses.dataclass
class MouseButtonConfig:
    left: Union[Undefined, MouseButtonType] = undefined
    middle: Union[Undefined, MouseButtonType] = undefined
    right: Union[Undefined, MouseButtonType] = undefined
    wheel: Union[Undefined, MouseButtonType] = undefined


@dataclasses.dataclass
class CameraControlProps(ThreeBasicProps):
    dampingFactor: Union[NumberType, Undefined] = undefined
    smoothTime: Union[NumberType, Undefined] = undefined
    draggingSmoothTime: Union[NumberType, Undefined] = undefined

    minDistance: Union[NumberType, Undefined] = undefined
    maxDistance: Union[NumberType, Undefined] = undefined
    minPolarAngle: Union[NumberType, Undefined] = undefined
    maxPolarAngle: Union[NumberType, Undefined] = undefined
    minZoom: Union[NumberType, Undefined] = undefined
    maxZoom: Union[NumberType, Undefined] = undefined
    polarRotateSpeed: Union[NumberType, Undefined] = undefined
    azimuthRotateSpeed: Union[NumberType, Undefined] = undefined
    truckSpeed: Union[NumberType, Undefined] = undefined
    dollySpeed: Union[NumberType, Undefined] = undefined
    verticalDragToForward: Union[bool, Undefined] = undefined
    keyboardFront: Union[bool, Undefined] = undefined
    keyboardMoveSpeed: Union[NumberType, Undefined] = undefined
    keyboardElevateSpeed: Union[NumberType, Undefined] = undefined

    infinityDolly: Union[bool, Undefined] = undefined
    makeDefault: Union[bool, Undefined] = undefined
    mouseButtons: Union[MouseButtonConfig, Undefined] = undefined
    # used to sync object 3ds based on camera position and rotation.
    # keep in mind that this won't affact position/rotation of those objects in BACKEND.
    # you need to due with them in backend.
    syncObject3ds: Union[list[Union[Object3dBase, Object3dContainerBase]],
                         Undefined] = undefined


class MapControl(ThreeComponentBase[OrbitControlProps]):

    def __init__(self) -> None:
        super().__init__(UIType.ThreeMapControl, OrbitControlProps)
        self.props.enableDamping = True
        self.props.dampingFactor = 0.25
        self.props.minDistance = 1
        self.props.maxDistance = 100

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class CameraUserControlType(enum.Enum):
    SetCamPose = 0
    SetLookAt = 1
    Reset = 2
    SetCamPoseRaw = 3
    RotateTo = 4
    LookAtObject = 5


class CameraControl(ThreeComponentBase[CameraControlProps]):
    """default values: https://github.com/yomotsu/camera-controls#properties
    threejs camera default axes:
        x: right
        y: up
        z: negative forward
    when we look at negative z axis and keep screen is y up and x right, 
    the rotation of camera is zero.
    """

    def __init__(self) -> None:
        super().__init__(UIType.ThreeCameraControl, CameraControlProps,
                         [FrontendEventType.Change.value])

        # self.props.enableDamping = True
        # self.props.dampingFactor = 1
        self.props.draggingSmoothTime = 0
        # self.props.smoothTime = 0
        # self.props.minDistance = 1
        # self.props.maxDistance = 100
        self.event_change = self._create_event_slot(FrontendEventType.Change)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        await handle_standard_event(self,
                                    ev,
                                    sync_state_after_change=False,
                                    is_sync=is_sync)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def set_cam2world(self,
                            cam2world: Union[list[float], np.ndarray],
                            distance: float,
                            fov_angle: float = -1,
                            update_now: bool = False):
        """
        TODO handle OrthographicCamera
        TODO currently we use a simple way to set cam2world, the
            rotation is limited by fixed camera up. so only cam2world[:, 2]
            is used in set_cam2world.
        Args: 
            cam2world: camera to world matrix, 4x4 ndaray or 16 list, R|T, not R/T
                the coordinate system is right hand, x right, y up, z negative forward
            distance: camera orbit target distance.
        """
        cam2world = np.array(cam2world, np.float32).reshape(4, 4)
        cam2world = cam2world.T  # R|T to R/T
        return await self.send_and_wait(
            self.create_comp_event({
                "type":
                CameraUserControlType.SetCamPose.value,
                "pose":
                list(map(float,
                         cam2world.reshape(-1).tolist())),
                "targetDistance":
                distance,
                "fov":
                fov_angle,
                "updateNow":
                update_now,
            }))

    async def set_lookat(self, origin: list[float], target: list[float]):
        """
        Args: 
            origin: camera position
            target: lookat target
        """
        return await self.send_and_wait(
            self.create_comp_event({
                "type": CameraUserControlType.SetLookAt.value,
                "lookat": origin + target,
            }))

    async def reset_camera(self):
        return await self.send_and_wait(
            self.create_comp_event({
                "type": CameraUserControlType.Reset.value,
            }))

    async def lookat_object(self, obj_uid: str):
        return await self.send_and_wait(
            self.create_comp_event({
                "type": CameraUserControlType.LookAtObject.value,
                "objectName": obj_uid,
            }))

    async def rotate_to(self, azimuth: float, polar: float):
        return await self.send_and_wait(
            self.create_comp_event({
                "type": CameraUserControlType.RotateTo.value,
                "azimuth": azimuth,
                "polar": polar,
            }))

    @staticmethod
    def fov_size_to_intrinsic(fov_angle: float, width: NumberType,
                              height: NumberType):
        size_wh = [int(width), int(height)]
        fov = (np.pi / 180) * fov_angle
        tanHalfFov = np.tan((fov / 2))
        f = size_wh[1] / 2 / tanHalfFov
        intrinsic = np.zeros((3, 3), np.float32)
        intrinsic[0, 0] = f
        intrinsic[1, 1] = f
        intrinsic[0, 2] = size_wh[0] / 2
        intrinsic[1, 2] = size_wh[1] / 2
        intrinsic[2, 2] = 1
        return intrinsic

    @staticmethod
    def intrinsic_size_to_fov(intrinsic: np.ndarray, width: NumberType,
                              height: NumberType):
        f = intrinsic[0][0]
        size_wh = [int(width), int(height)]
        tanHalfFov = size_wh[1] / 2 / f
        fov = np.arctan(tanHalfFov) * 2
        fov_angle = fov / (np.pi / 180)
        return fov_angle


class OrbitControl(ThreeComponentBase[OrbitControlProps]):

    def __init__(self) -> None:
        super().__init__(UIType.ThreeOrbitControl, OrbitControlProps)
        self.props.enableDamping = True
        self.props.dampingFactor = 0.25
        self.props.minDistance = 1
        self.props.maxDistance = 100

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

# @dataclasses.dataclass
# class PointerLockControlProps(Object3dBaseProps):
#     enabled: Union[bool, Undefined] = undefined
#     minPolarAngle: Union[float, Undefined] = undefined
#     maxPolarAngle: Union[float, Undefined] = undefined
#     makeDefault: Union[bool, Undefined] = undefined

# class PointerLockControl(ThreeComponentBase[PointerLockControlProps]):
#     def __init__(self,
#                  enabled: Union[bool, Undefined] = undefined,
#                  min_polar_angle: Union[float, Undefined] = undefined,
#                  max_polar_angle: Union[float, Undefined] = undefined) -> None:
#         super().__init__(UIType.ThreePointerLockControl,
#                          PointerLockControlProps)
#         self.props.enabled = enabled
#         self.props.minPolarAngle = min_polar_angle
#         self.props.maxPolarAngle = max_polar_angle

#     @property
#     def prop(self):
#         propcls = self.propcls
#         return self._prop_base(propcls, self)

#     @property
#     def update_event(self):
#         propcls = self.propcls
#         return self._update_props_base(propcls)

# @dataclasses.dataclass
# class FirstPersonControlProps(ThreeBasicProps):
#     enabled: Union[bool, Undefined] = undefined
#     movementSpeed: Union[float, Undefined] = undefined
#     autoForward: Union[bool, Undefined] = undefined
#     lookSpeed: Union[float, Undefined] = undefined
#     lookVertical: Union[bool, Undefined] = undefined
#     activeLook: Union[bool, Undefined] = undefined
#     heightSpeed: Union[bool, Undefined] = undefined
#     heightCoef: Union[float, Undefined] = undefined
#     heightMin: Union[float, Undefined] = undefined
#     heightMax: Union[float, Undefined] = undefined
#     constrainVertical: Union[bool, Undefined] = undefined
#     verticalMin: Union[float, Undefined] = undefined
#     verticalMax: Union[float, Undefined] = undefined
#     mouseDragOn: Union[bool, Undefined] = undefined
#     makeDefault: Union[bool, Undefined] = undefined

# class FirstPersonControl(ThreeComponentBase[FirstPersonControlProps]):
#     def __init__(self) -> None:
#         super().__init__(UIType.ThreeFirstPersonControl,
#                          FirstPersonControlProps)

#     @property
#     def prop(self):
#         propcls = self.propcls
#         return self._prop_base(propcls, self)

#     @property
#     def update_event(self):
#         propcls = self.propcls
#         return self._update_props_base(propcls)

