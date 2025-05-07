
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
from tensorpc.dock.core.component import (UIType, ContainerBaseProps)
from collections.abc import Sequence
from tensorpc.dock.core import colors
from tensorpc.dock.core.appcore import Event, EventDataType
from tensorpc.dock.core.common import handle_standard_event
from tensorpc.core.datamodel.typemetas import RangedFloat, RangedInt
from tensorpc.dock.components.mui import (Image as MUIImage, PointerEventsProperties, MUIComponentType)

from .base import (ThreeContainerBase, NumberType, ThreeComponentType, ThreeLayoutType,
    Object3dContainerBase, Object3dContainerBaseProps)

class URILoaderType(enum.IntEnum):
    GLTF = 0
    FBX = 1
    RGBE = 2
    TEXTURE = 3


@dataclasses.dataclass
class LoaderContextProps(ContainerBaseProps):
    uri: str = ""
    loaderType: URILoaderType = URILoaderType.GLTF
    dataKey: Union[str, Undefined] = undefined  # default: URILoader


class URILoaderContext(ThreeContainerBase[LoaderContextProps,
                                          ThreeComponentType]):
    """create a context with template data.
    default dataKey: "" (empty), this means the data itself is passed to children
    """

    def __init__(self,
                 type: URILoaderType,
                 uri: str,
                 children: Optional[ThreeLayoutType] = None) -> None:
        if children is None:
            children = {}
        if isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}

        super().__init__(UIType.ThreeURILoaderContext, LoaderContextProps,
                         {**children})
        self.props.uri = uri
        self.props.loaderType = type

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class CubeCameraProps(Object3dContainerBaseProps):
    frames: Union[int, Undefined] = undefined
    resolution: Union[NumberType, Undefined] = undefined
    near: Union[NumberType, Undefined] = undefined
    far: Union[NumberType, Undefined] = undefined
    dataKey: Union[str, Undefined] = undefined


class CubeCamera(Object3dContainerBase[CubeCameraProps, ThreeComponentType]):
    """create a context with template data. 
    default dataKey: CubeCameraTexture
    """

    def __init__(self, children: ThreeLayoutType) -> None:
        if children is None:
            children = {}
        if isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}
        # assert children, "CubeCamera must have children"
        super().__init__(UIType.ThreeCubeCamera, CubeCameraProps, {**children})

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

