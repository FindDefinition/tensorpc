from typing import (Union)

import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.annolib import Undefined, undefined
from tensorpc.core.datamodel import typemetas
from tensorpc.dock.core.component import (UIType)
from collections.abc import Sequence
from .base import BlendFunction, NumberType, ThreeBasicProps, ToneMapppingMode, ThreeEffectBase, ContainerBaseProps, ThreeContainerBase, ThreeEffectType

@dataclasses.dataclass
class EffectComposerProps(ContainerBaseProps):
    enabled: Union[bool, Undefined] = undefined
    depthBuffer: Union[bool, Undefined] = undefined
    disableNormalPass: Union[bool, Undefined] = undefined
    stencilBuffer: Union[bool, Undefined] = undefined
    autoClear: Union[bool, Undefined] = undefined
    multisampling: Union[int, Undefined] = undefined
    resolutionScale: Union[NumberType, Undefined] = undefined
    renderPriority: Union[int, Undefined] = undefined


class EffectComposer(ThreeContainerBase[EffectComposerProps, ThreeEffectBase]):
    """when you use postprocessing, you need to set flat=False in canvas (disable default tonemapping) and 
    use ToneMapping effect in postprocessing.
    high-precision frame buffer is enabled by default.
    """

    def __init__(self, children: ThreeEffectType) -> None:
        if children is None:
            children = {}
        if isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}
        for v in children.values():
            assert isinstance(
                v, ThreeEffectBase), "child of effect composer must be effect."
        super().__init__(UIType.ThreeEffectComposer, EffectComposerProps,
                         {**children})

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class ToneMappingProps(ThreeBasicProps):
    blendFunction: Union[BlendFunction, Undefined] = undefined
    mode: Union[ToneMapppingMode, Undefined] = undefined
    adaptive: Union[bool, Undefined] = undefined
    resolution: Union[NumberType, Undefined] = undefined
    maxLuminance: Union[NumberType, Undefined] = undefined
    whitePoint: Union[NumberType, Undefined] = undefined
    middleGrey: Union[NumberType, Undefined] = undefined
    minLuminance: Union[NumberType, Undefined] = undefined
    averageLuminance: Union[NumberType, Undefined] = undefined
    adaptationRate: Union[NumberType, Undefined] = undefined


class ToneMapping(ThreeEffectBase[ToneMappingProps]):

    def __init__(self) -> None:
        super().__init__(UIType.ThreeEffectToneMapping, ToneMappingProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class OutlineProps(ThreeBasicProps):
    selectionLayer: Union[int, Undefined] = undefined
    edgeStrength: Union[NumberType, Undefined] = undefined
    pulseSpeed: Union[NumberType, Undefined] = undefined
    visibleEdgeColor: Union[NumberType, Undefined] = undefined
    hiddenEdgeColor: Union[NumberType, Undefined] = undefined
    width: Union[NumberType, Undefined] = undefined
    height: Union[NumberType, Undefined] = undefined
    kernelSize: Union[NumberType, Undefined] = undefined
    blur: Union[bool, Undefined] = undefined
    xRay: Union[bool, Undefined] = undefined
    blendFunction: Union[BlendFunction, Undefined] = undefined


class Outline(ThreeEffectBase[OutlineProps]):

    def __init__(self) -> None:
        super().__init__(UIType.ThreeEffectOutline, OutlineProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class BloomProps(ThreeBasicProps):
    luminanceThreshold: Union[NumberType, Undefined] = undefined
    luminanceSmoothing: Union[NumberType, Undefined] = undefined
    blendFunction: Union[BlendFunction, Undefined] = undefined
    intensity: Union[NumberType, Undefined] = undefined
    resolutionX: Union[NumberType, Undefined] = undefined
    resolutionY: Union[NumberType, Undefined] = undefined
    kernelSize: Union[NumberType, Undefined] = undefined
    mipMap: Union[bool, Undefined] = undefined


class Bloom(ThreeEffectBase[BloomProps]):

    def __init__(self) -> None:
        super().__init__(UIType.ThreeEffectBloom, BloomProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class DepthOfFieldProps(ThreeBasicProps):
    focusDistance: Union[NumberType, Undefined] = undefined
    focalLength: Union[NumberType, Undefined] = undefined
    bokehScale: Union[NumberType, Undefined] = undefined
    height: Union[NumberType, Undefined] = undefined
    width: Union[NumberType, Undefined] = undefined
    blendFunction: Union[BlendFunction, Undefined] = undefined


class DepthOfField(ThreeEffectBase[DepthOfFieldProps]):

    def __init__(self) -> None:
        super().__init__(UIType.ThreeEffectDepthOfField, DepthOfFieldProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)
