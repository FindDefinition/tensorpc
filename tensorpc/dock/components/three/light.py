from collections.abc import Sequence
from typing_extensions import Annotated, Literal
from typing import (Optional, Union)

import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.annolib import Undefined, undefined
from tensorpc.core.datamodel import typemetas
from tensorpc.dock.core.component import (UIType, ContainerBaseProps)

from .base import (Object3dBaseProps, ThreeLayoutType, NumberType, Object3dBase, Vector3Type,
                ThreeBasicProps, ThreeComponentBase, Object3dContainerBaseProps, 
                ThreeContainerBase, ThreeComponentType, Object3dContainerBase)


@dataclasses.dataclass
class PointLightProps(Object3dBaseProps):
    color: Annotated[Union[int, str, Undefined],
                     typemetas.ColorRGB()] = undefined
    intensity: Union[NumberType, Undefined] = undefined
    distance: Union[NumberType, Undefined] = undefined
    decay: Union[NumberType, Undefined] = undefined
    castShadow: Union[bool, Undefined] = undefined
    power: Union[NumberType, Undefined] = undefined
    helperSize: Union[NumberType, Undefined] = undefined


class PointLight(Object3dBase[PointLightProps]):

    def __init__(self,
                 position: Union[Vector3Type, Undefined] = undefined,
                 color: Annotated[Union[int, str, Undefined],
                                  typemetas.ColorRGB()] = undefined,
                 intensity: Union[NumberType, Undefined] = undefined) -> None:
        super().__init__(UIType.ThreePointLight, PointLightProps)
        self.props.color = color
        self.props.intensity = intensity
        self.props.position = position

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class AmbientLightProps(Object3dBaseProps):
    color: Annotated[Union[int, str, Undefined],
                     typemetas.ColorRGB()] = undefined
    intensity: Union[NumberType, Undefined] = undefined


class AmbientLight(Object3dBase[AmbientLightProps]):

    def __init__(self,
                 color: Annotated[Union[int, str, Undefined],
                                  typemetas.ColorRGB()] = undefined,
                 intensity: Union[NumberType, Undefined] = undefined) -> None:
        super().__init__(UIType.ThreeAmbientLight, AmbientLightProps)
        self.props.color = color
        self.props.intensity = intensity

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class HemisphereLightProps(Object3dBaseProps):
    color: Annotated[Union[int, str, Undefined],
                     typemetas.ColorRGB()] = undefined
    intensity: Union[NumberType, Undefined] = undefined
    groundColor: Union[NumberType, str, Undefined] = undefined


class HemisphereLight(Object3dBase[HemisphereLightProps]):

    def __init__(
            self,
            color: Annotated[Union[int, str, Undefined],
                             typemetas.ColorRGB()] = undefined,
            intensity: Union[NumberType, Undefined] = undefined,
            ground_color: Union[NumberType, str,
                                Undefined] = undefined) -> None:
        super().__init__(UIType.ThreeHemisphereLight, HemisphereLightProps)
        self.props.color = color
        self.props.intensity = intensity
        self.props.groundColor = ground_color

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class DirectionalLightProps(Object3dBaseProps):
    color: Annotated[Union[int, str, Undefined],
                     typemetas.ColorRGB()] = undefined
    intensity: Union[NumberType, Undefined] = undefined
    castShadow: Union[bool, Undefined] = undefined
    targetPosition: Union[Vector3Type, Undefined] = undefined
    helperColor: Union[NumberType, Undefined] = undefined
    helperSize: Union[NumberType, Undefined] = undefined


class DirectionalLight(Object3dBase[DirectionalLightProps]):

    def __init__(
            self,
            position: Union[Vector3Type, Undefined] = undefined,
            color: Annotated[Union[int, str, Undefined],
                             typemetas.ColorRGB()] = undefined,
            intensity: Union[NumberType, Undefined] = undefined,
            target_position: Union[Vector3Type,
                                   Undefined] = undefined) -> None:
        super().__init__(UIType.ThreeDirectionalLight, DirectionalLightProps)
        self.props.color = color
        self.props.intensity = intensity
        self.props.targetPosition = target_position
        self.props.position = position

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class SpotLightProps(Object3dBaseProps):
    color: Annotated[Union[int, str, Undefined],
                     typemetas.ColorRGB()] = undefined
    intensity: Union[NumberType, Undefined] = undefined
    distance: Union[NumberType, Undefined] = undefined
    decay: Union[NumberType, Undefined] = undefined
    castShadow: Union[bool, Undefined] = undefined
    angle: Union[NumberType, Undefined] = undefined
    penumbra: Union[NumberType, Undefined] = undefined
    power: Union[NumberType, Undefined] = undefined
    targetPosition: Union[Vector3Type, Undefined] = undefined
    helperColor: Union[NumberType, Undefined] = undefined


class SpotLight(Object3dBase[SpotLightProps]):

    def __init__(
            self,
            position: Union[Vector3Type, Undefined] = undefined,
            color: Annotated[Union[int, str, Undefined],
                             typemetas.ColorRGB()] = undefined,
            intensity: Union[NumberType, Undefined] = undefined,
            target_position: Union[Vector3Type,
                                   Undefined] = undefined) -> None:
        super().__init__(UIType.ThreeSpotLight, SpotLightProps)
        self.props.color = color
        self.props.intensity = intensity
        self.props.targetPosition = target_position
        self.props.position = position

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class SkyProps(ThreeBasicProps):
    distance: Union[NumberType, Undefined] = undefined
    sunPosition: Union[Vector3Type, Undefined] = undefined
    inclination: Union[NumberType, Undefined] = undefined
    azimuth: Union[NumberType, Undefined] = undefined
    mieCoefficient: Union[NumberType, Undefined] = undefined
    mieDirectionalG: Union[NumberType, Undefined] = undefined
    rayleigh: Union[NumberType, Undefined] = undefined
    turbidity: Union[NumberType, Undefined] = undefined


class Sky(ThreeComponentBase[SkyProps]):

    def __init__(self) -> None:
        super().__init__(UIType.ThreeSky, SkyProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class EnvGround:
    radius: Union[NumberType, Undefined] = undefined
    height: Union[NumberType, Undefined] = undefined
    scale: Union[Vector3Type, Undefined] = undefined


@dataclasses.dataclass
class EnvironmentProps(ContainerBaseProps):
    files: Union[list[str], str, Undefined] = undefined
    resolution: Union[int, Undefined] = undefined
    background: Union[bool, Literal["only"], Undefined] = undefined
    blur: Union[int, Undefined] = undefined
    preset: Union[Literal["sunset", "dawn", "night", "warehouse", "forest",
                          "apartment", "studio", "city", "park", "lobby"],
                  Undefined] = undefined
    ground: Union[EnvGround, bool, Undefined] = undefined
    path: Union[str, Undefined] = undefined


class Environment(ThreeContainerBase[EnvironmentProps, ThreeComponentType]):

    def __init__(self, children: Optional[ThreeLayoutType] = None) -> None:
        if children is None:
            children = {}
        if isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}

        super().__init__(UIType.ThreeEnvironment, EnvironmentProps,
                         {**children})

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

