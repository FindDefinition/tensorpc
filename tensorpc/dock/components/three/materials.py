from collections.abc import Sequence
import enum
from typing_extensions import Annotated
from typing import (Any, Union)

import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.annolib import Undefined, undefined
from tensorpc.core.datamodel import typemetas
from tensorpc.dock.core.component import (UIType, ContainerBaseProps)
from tensorpc.dock.components.threecore import TextureFormat, TextureMappingType, TextureType, TextureWrappingMode
import numpy as np 
from .base import ThreeComponentType, ThreeMaterialBase, ThreeMaterialContainerBase, ThreeMaterialPropsBaseProps, ThreeMaterialPropsBase, NumberType, ThreeBasicProps, ThreeComponentBase
from pydantic import field_validator

@dataclasses.dataclass
class MeshBasicMaterialProps(ThreeMaterialPropsBase):
    color: Annotated[Union[str, Undefined], typemetas.ColorRGB()] = undefined
    wireframe: Union[bool, Undefined] = undefined
    vertexColors: Union[bool, Undefined] = undefined
    fog: Union[bool, Undefined] = undefined


@dataclasses.dataclass
class MeshStandardMaterialProps(MeshBasicMaterialProps):
    emissive: Union[str, Undefined] = undefined
    roughness: Union[NumberType, Undefined] = undefined
    metalness: Union[NumberType, Undefined] = undefined
    flagShading: Union[bool, Undefined] = undefined


@dataclasses.dataclass
class MeshLambertMaterialProps(MeshBasicMaterialProps):
    emissive: Union[str, Undefined] = undefined


@dataclasses.dataclass
class MeshMatcapMaterialProps(ThreeMaterialPropsBase):
    flagShading: Union[bool, Undefined] = undefined
    color: Annotated[Union[str, Undefined], typemetas.ColorRGB()] = undefined


@dataclasses.dataclass
class MeshNormalMaterialProps(ThreeMaterialPropsBase):
    flagShading: Union[bool, Undefined] = undefined
    wireframe: Union[bool, Undefined] = undefined


@dataclasses.dataclass
class MeshDepthMaterialProps(ThreeMaterialPropsBase):
    wireframe: Union[bool, Undefined] = undefined


@dataclasses.dataclass
class MeshPhongMaterialProps(MeshBasicMaterialProps):
    reflectivity: Union[NumberType, Undefined] = undefined
    refractionRatio: Union[NumberType, Undefined] = undefined
    emissive: Union[str, Undefined] = undefined
    specular: Union[str, Undefined] = undefined
    shininess: Union[NumberType, Undefined] = undefined


@dataclasses.dataclass
class MeshPhysicalMaterialProps(MeshStandardMaterialProps):
    reflectivity: Union[NumberType, Undefined] = undefined
    clearcoat: Union[NumberType, Undefined] = undefined
    clearcoatRoughness: Union[NumberType, Undefined] = undefined
    metalness: Union[NumberType, Undefined] = undefined
    roughness: Union[NumberType, Undefined] = undefined
    sheen: Union[NumberType, Undefined] = undefined
    transmission: Union[NumberType, Undefined] = undefined
    ior: Union[NumberType, Undefined] = undefined
    attenuationColor: Union[str, NumberType, Undefined] = undefined
    attenuationDistance: Union[NumberType, Undefined] = undefined
    specularIntensity: Union[NumberType, Undefined] = undefined
    specularColor: Union[str, NumberType, Undefined] = undefined
    sheenRoughness: Union[NumberType, Undefined] = undefined
    sheenColor: Union[str, NumberType, Undefined] = undefined


@dataclasses.dataclass
class MeshToonMaterialProps(ThreeMaterialPropsBase):
    color: Annotated[Union[str, Undefined], typemetas.ColorRGB()] = undefined


@dataclasses.dataclass
class MeshTransmissionMaterialProps(MeshPhysicalMaterialProps):
    transmission: Union[NumberType, Undefined] = undefined
    thickness: Union[NumberType, Undefined] = undefined
    backsideThickness: Union[NumberType, Undefined] = undefined
    roughness: Union[NumberType, Undefined] = undefined
    chromaticAberration: Union[NumberType, Undefined] = undefined
    anisotropy: Union[NumberType, Undefined] = undefined
    distortion: Union[NumberType, Undefined] = undefined
    distortion_scale: Union[NumberType, Undefined] = undefined
    temporalDistortion: Union[NumberType, Undefined] = undefined
    transmission_sampler: Union[bool, Undefined] = undefined
    backside: Union[bool, Undefined] = undefined
    resolution: Union[NumberType, Undefined] = undefined
    backsideResolution: Union[NumberType, Undefined] = undefined
    samples: Union[NumberType, Undefined] = undefined


@dataclasses.dataclass
class MeshDiscardMaterialProps(ThreeBasicProps):
    pass

@dataclasses.dataclass
class MeshPortalMaterialProps(ThreeMaterialPropsBaseProps, ContainerBaseProps):
    blend: Union[NumberType, Undefined] = undefined
    blur: Union[NumberType, Undefined] = undefined
    resolution: Union[NumberType, Undefined] = undefined
    worldUnits: Union[bool, Undefined] = undefined
    eventPriority: Union[NumberType, Undefined] = undefined
    renderPriority: Union[NumberType, Undefined] = undefined
    events: Union[bool, Undefined] = undefined
    

class MeshBasicMaterial(ThreeComponentBase[MeshBasicMaterialProps]):

    def __init__(self) -> None:
        super().__init__(UIType.ThreeMeshBasicMaterial, MeshBasicMaterialProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshStandardMaterial(ThreeMaterialBase[MeshStandardMaterialProps]):

    def __init__(self) -> None:
        super().__init__(UIType.ThreeMeshStandardMaterial,
                         MeshStandardMaterialProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshLambertMaterial(ThreeMaterialBase[MeshLambertMaterialProps]):

    def __init__(self, ) -> None:
        super().__init__(UIType.ThreeMeshLambertMaterial,
                         MeshLambertMaterialProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshMatcapMaterial(ThreeMaterialBase[MeshMatcapMaterialProps]):

    def __init__(self, ) -> None:
        super().__init__(UIType.ThreeMeshMatcapMaterial,
                         MeshMatcapMaterialProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshNormalMaterial(ThreeMaterialBase[MeshNormalMaterialProps]):

    def __init__(self, ) -> None:
        super().__init__(UIType.ThreeMeshNormalMaterial,
                         MeshNormalMaterialProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshDepthMaterial(ThreeMaterialBase[MeshDepthMaterialProps]):

    def __init__(self, ) -> None:
        super().__init__(UIType.ThreeMeshDepthMaterial, MeshDepthMaterialProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshPhongMaterial(ThreeMaterialBase[MeshPhongMaterialProps]):

    def __init__(self, ) -> None:
        super().__init__(UIType.ThreeMeshPhongMaterial, MeshPhongMaterialProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshPhysicalMaterial(ThreeMaterialBase[MeshPhysicalMaterialProps]):

    def __init__(self, ) -> None:
        super().__init__(UIType.ThreeMeshPhysicalMaterial,
                         MeshPhysicalMaterialProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshToonMaterial(ThreeMaterialBase[MeshToonMaterialProps]):

    def __init__(self, ) -> None:
        super().__init__(UIType.ThreeMeshToonMaterial, MeshToonMaterialProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshTransmissionMaterial(ThreeMaterialBase[MeshTransmissionMaterialProps]
                               ):

    def __init__(self, ) -> None:
        super().__init__(UIType.ThreeMeshTransmissionMaterial,
                         MeshTransmissionMaterialProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class MeshDiscardMaterial(ThreeMaterialBase[MeshDiscardMaterialProps]):

    def __init__(self, ) -> None:
        super().__init__(UIType.ThreeMeshDiscardMaterial,
                         MeshDiscardMaterialProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


class ShaderUniformType(enum.IntEnum):
    # Int32Array = 0
    # Float32Array = 1
    Matrix4 = 2
    Matrix3 = 3
    Quaternion = 4
    Vector4 = 5
    Vector3 = 6
    Vector2 = 7
    Color = 8
    Number = 9
    Boolean = 10
    Array = 11
    DataTexture = 12
    DataArrayTexture = 13
    Data3DTexture = 14


@dataclasses.dataclass
class ShaderUniform:
    name: str
    type: ShaderUniformType
    value: Any


@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class DataTexture:
    data: np.ndarray
    texType: Union[TextureType, Undefined] = undefined
    format: Union[TextureFormat, Undefined] = undefined
    mapping: Union[TextureMappingType, Undefined] = undefined
    wrapS: Union[TextureWrappingMode, Undefined] = undefined
    wrapT: Union[TextureWrappingMode, Undefined] = undefined

    @field_validator('data')
    def uniform_data_validator(cls, v: np.ndarray):
        assert isinstance(
            v, np.ndarray) and v.dtype == np.uint8 and v.ndim == 3 and v.shape[
                2] == 4, f"uniform data must be [H, W, 4] RGBA uint8 array"
        return v


# type TextureValue = {
#     data: np.NdArray
#     texType?: TextureType
#     format?: TextureFormat
#     mapping?: TextureMappingType
#     wrapS?: TextureWrappingMode
#     wrapT?: TextureWrappingMode
# }


@dataclasses.dataclass
class MeshShaderMaterialProps(ThreeMaterialPropsBase):
    uniforms: list[ShaderUniform] = dataclasses.field(default_factory=list)
    vertexShader: str = ""
    fragmentShader: str = ""
    timeUniformKey: Union[Undefined, str] = undefined

    @staticmethod
    def _validator_single_uniform(u: ShaderUniform, value: Any):
        uv = value
        assert u.name.isidentifier(
        ), f"uniform name {u.name} must be identifier"
        if u.type == ShaderUniformType.Matrix4:
            assert isinstance(
                uv,
                np.ndarray) and uv.dtype == np.float32 and uv.shape == (4, 4)
        elif u.type == ShaderUniformType.Matrix3:
            assert isinstance(
                uv,
                np.ndarray) and uv.dtype == np.float32 and uv.shape == (3, 3)
        elif u.type == ShaderUniformType.Quaternion:
            assert isinstance(
                uv,
                np.ndarray) and uv.dtype == np.float32 and uv.shape == (4, )
        elif u.type == ShaderUniformType.Vector4:
            assert isinstance(
                uv,
                np.ndarray) and uv.dtype == np.float32 and uv.shape == (4, )
        elif u.type == ShaderUniformType.Vector3:
            assert isinstance(
                uv,
                np.ndarray) and uv.dtype == np.float32 and uv.shape == (3, )
        elif u.type == ShaderUniformType.Vector2:
            assert isinstance(
                uv,
                np.ndarray) and uv.dtype == np.float32 and uv.shape == (2, )
        elif u.type == ShaderUniformType.Color:
            assert isinstance(uv, (str, int))
        elif u.type == ShaderUniformType.Number:
            assert isinstance(uv, (int, float))
        elif u.type == ShaderUniformType.Boolean:
            assert isinstance(uv, (bool))
        elif u.type == ShaderUniformType.Array:
            assert isinstance(uv, (list))
        elif u.type == ShaderUniformType.DataTexture:
            assert isinstance(uv, DataTexture)

    @field_validator('uniforms')
    def uniform_validator(cls, v: list[ShaderUniform]):
        for u in v:
            cls._validator_single_uniform(u, u.value)
        return v

    @field_validator('timeUniformKey')
    def time_uniform_key_validator(cls, v):
        if isinstance(v, Undefined):
            return v
        assert isinstance(
            v,
            str) and v.isidentifier(), f"timeUniformKey {v} must be identifier"
        return v

class _ShaderControlType(enum.IntEnum):
    UpdateUniform = 0


class MeshShaderMaterial(ThreeMaterialBase[MeshShaderMaterialProps]):
    """don't forget to add some stmt in fragment shader:

    #include <tonemapping_fragment>

    #include <colorspace_fragment>
    """

    def __init__(self, uniforms: list[ShaderUniform], vertex_shader: str, fragment_shader: str) -> None:
        super().__init__(UIType.ThreeMeshShaderMaterial,
                         MeshShaderMaterialProps)
        self.prop(uniforms=uniforms,
                  vertexShader=vertex_shader,
                  fragmentShader=fragment_shader)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    def update_uniform_values_event(self, uniform_values: dict[str, Any]):
        uniform_def_dict = {u.name: u for u in self.props.uniforms}
        res: list[ShaderUniform] = []
        for k, v in uniform_values.items():
            if k not in uniform_def_dict:
                raise ValueError(f"uniform {k} not defined")
            u = uniform_def_dict[k]
            MeshShaderMaterialProps._validator_single_uniform(u, v)
            res.append(ShaderUniform(name=k, type=u.type, value=v))
        # here we update value of backend. note that value in frontend
        # won't be updated since it will trigger material recreation.
        for k, v in uniform_values.items():
            u = uniform_def_dict[k]
            u.value = v
        return self.create_comp_event({
            "type": _ShaderControlType.UpdateUniform,
            "uniforms": res
        })

    async def update_uniform_values(self, uniform_values: dict[str, Any]):
        await self.send_and_wait(
            self.update_uniform_values_event(uniform_values))

class MeshPortalMaterial(ThreeMaterialContainerBase[MeshPortalMaterialProps, ThreeComponentType]):

    def __init__(
        self, children: Union[dict[str, ThreeComponentType],
                              list[ThreeComponentType]]
    ) -> None:
        if isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.ThreeMeshPortalMaterial, MeshPortalMaterialProps, children)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)
