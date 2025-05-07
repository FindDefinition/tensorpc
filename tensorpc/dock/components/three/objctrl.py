

from typing_extensions import Annotated, Literal, TypeAlias
from typing import (Callable, Union, Any, Optional, Coroutine)
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

from .base import ThreeContainerBase, NumberType, PivotControlsCommonProps, ThreeComponentBase, Object3dContainerBase, Vector3Type, ThreeComponentType, ContainerBaseProps

_CORO_ANY: TypeAlias = Union[Coroutine[Any, None, None], Any]

@dataclasses.dataclass
class TransformControlsProps(ContainerBaseProps):
    enabled: Union[bool, Undefined] = undefined
    axis: Union[str, Undefined] = undefined
    mode: Union[str, Undefined] = undefined
    translationSnap: Union[NumberType, Undefined] = undefined
    rotationSnap: Union[NumberType, Undefined] = undefined
    scaleSnap: Union[NumberType, Undefined] = undefined
    space: Union[str, Undefined] = undefined
    size: Union[NumberType, Undefined] = undefined
    showX: Union[bool, Undefined] = undefined
    showY: Union[bool, Undefined] = undefined
    showZ: Union[bool, Undefined] = undefined
    object3dUid: Union[str, Undefined] = undefined


class TransformControls(ThreeComponentBase[TransformControlsProps]):

    def __init__(self) -> None:
        super().__init__(UIType.ThreeTransformControl, TransformControlsProps)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)


@dataclasses.dataclass
class PivotControlsProps(ContainerBaseProps, PivotControlsCommonProps):
    offset: Union[Vector3Type, Undefined] = undefined
    rotation: Union[Vector3Type, Undefined] = undefined

    matrix: Union[list[float], Undefined] = undefined
    autoTransform: Union[bool, Undefined] = undefined


class PivotControls(ThreeContainerBase[PivotControlsProps,
                                       ThreeComponentType]):

    def __init__(self,
                 children: Optional[Union[dict[str, ThreeComponentType],
                                          list[ThreeComponentType]]] = None,
                 callback: Optional[Callable[[bool], _CORO_ANY]] = None,
                 debounce: float = 100) -> None:
        if children is None:
            children = []
        if isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}

        super().__init__(UIType.ThreePivotControl,
                         PivotControlsProps,
                         allowed_events=[FrontendEventType.Change.value],
                         _children=children)
        if callback is not None:
            self.register_event_handler(FrontendEventType.Change.value,
                                        callback,
                                        debounce=debounce)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self,
                                           ev,
                                           sync_status_first=True,
                                           sync_state_after_change=False,
                                           is_sync=is_sync)

