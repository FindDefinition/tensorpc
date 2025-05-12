import asyncio
from collections.abc import Sequence
import json
from typing_extensions import Annotated, Literal, TypeAlias
from typing import (Callable, Iterable, Type, TypeVar, TypedDict, Union, Any, Optional, Coroutine)
import enum 
import numpy as np 
from pydantic import field_validator

import tensorpc.core.dataclass_dispatch as dataclasses
from tensorpc.core.annolib import Undefined, undefined
from tensorpc.dock.core.component import (UIType, ContainerBaseProps, Component, ContainerBase, T_base_props, T_container_props, T_child, Fragment, FrontendEventType)
from .base import (PyDanticConfigForNumpy, NumberType, ThreeBasicProps, 
    ValueType, ThreeComponentType, DataPortal)
from tensorpc.dock.core.appcore import Event, EventDataType
from tensorpc.dock.core.common import handle_standard_event


FlexAlignment: TypeAlias = Literal[
    'auto',
    'flex-start',
    'center',
    'flex-end',
    'stretch',
    'baseline',
    'space-between',
    'space-around'
]
FlexJustify: TypeAlias = Literal[
    'flex-start',
    'center',
    'flex-end',
    'space-between',
    'space-around',
    'space-evenly'
]
FlexDirection: TypeAlias = Literal[
    'row',
    'row-reverse',
    'column',
    'column-reverse'
]
FlexWrap: TypeAlias = Literal[
    'nowrap', 'wrap', 'wrap-reverse'
]


@dataclasses.dataclass
class UIKitFlexCoreProps:
    visibility: Union[Literal["visible", "hidden"], Undefined] = undefined

    margin: Union[NumberType, Undefined] = undefined
    marginX: Union[NumberType, Undefined] = undefined
    marginY: Union[NumberType, Undefined] = undefined
    marginTop: Union[NumberType, Undefined] = undefined
    marginLeft: Union[NumberType, Undefined] = undefined
    marginRight: Union[NumberType, Undefined] = undefined
    marginBottom: Union[NumberType, Undefined] = undefined

    transformTranslateX: Union[NumberType, Undefined] = undefined
    transformTranslateY: Union[NumberType, Undefined] = undefined
    transformTranslateZ: Union[NumberType, Undefined] = undefined
    transformScaleX: Union[NumberType, Undefined] = undefined
    transformScaleY: Union[NumberType, Undefined] = undefined
    transformScaleZ: Union[NumberType, Undefined] = undefined
    transformRotateX: Union[NumberType, Undefined] = undefined
    transformRotateY: Union[NumberType, Undefined] = undefined
    transformRotateZ: Union[NumberType, Undefined] = undefined

    positionType: Union[str, Undefined] = undefined
    inset: Union[ValueType, Undefined] = undefined
    positionTop: Union[ValueType, Undefined] = undefined
    positionLeft: Union[ValueType, Undefined] = undefined
    positionRight: Union[ValueType, Undefined] = undefined
    positionBottom: Union[ValueType, Undefined] = undefined

    alignContent: Union[FlexAlignment, Undefined] = undefined
    alignItems: Union[FlexAlignment, Undefined] = undefined
    alignSelf: Union[FlexAlignment, Undefined] = undefined

    flexDirection: Union[FlexDirection, Undefined] = undefined
    flexWrap: Union[FlexWrap, Undefined] = undefined
    justifyContent: Union[FlexJustify, Undefined] = undefined

    flexBasis: Union[ValueType, Undefined] = undefined
    flexGrow: Union[NumberType, Undefined] = undefined
    flexShrink: Union[NumberType, Undefined] = undefined
    width: Union[ValueType, Undefined] = undefined
    height: Union[ValueType, Undefined] = undefined
    minWidth: Union[ValueType, Undefined] = undefined
    minHeight: Union[ValueType, Undefined] = undefined
    maxWidth: Union[ValueType, Undefined] = undefined
    maxHeight: Union[ValueType, Undefined] = undefined

    aspectRatio: Union[NumberType, Undefined] = undefined
    border: Union[NumberType, Undefined] = undefined
    borderX: Union[NumberType, Undefined] = undefined
    borderY: Union[NumberType, Undefined] = undefined
    borderTop: Union[NumberType, Undefined] = undefined
    borderLeft: Union[NumberType, Undefined] = undefined
    borderRight: Union[NumberType, Undefined] = undefined
    borderBottom: Union[NumberType, Undefined] = undefined
    overflow: Union[Literal["visible", "hidden", "scroll"], Undefined] = undefined

    padding: Union[ValueType, Undefined] = undefined
    paddingX: Union[ValueType, Undefined] = undefined
    paddingY: Union[ValueType, Undefined] = undefined
    paddingTop: Union[ValueType, Undefined] = undefined
    paddingLeft: Union[ValueType, Undefined] = undefined
    paddingRight: Union[ValueType, Undefined] = undefined
    paddingBottom: Union[ValueType, Undefined] = undefined

    gap: Union[NumberType, Undefined] = undefined
    gapRow: Union[NumberType, Undefined] = undefined
    gapColumn: Union[NumberType, Undefined] = undefined


@dataclasses.dataclass
class UIKitContainerCoreProps(UIKitFlexCoreProps):
    receiveShadow: Union[bool, Undefined] = undefined
    castShadow: Union[bool, Undefined] = undefined
    depthTest: Union[bool, Undefined] = undefined
    depthWrite: Union[bool, Undefined] = undefined
    renderOrder: Union[NumberType, Undefined] = undefined
    backgroundColor: Union[ValueType, Undefined] = undefined
    backgroundOpacity: Union[NumberType, Undefined] = undefined
    panelMaterialClass: Union[str, Undefined] = undefined

    borderOpacity: Union[NumberType, Undefined] = undefined
    borderColor: Union[ValueType, Undefined] = undefined
    borderRadius: Union[NumberType, Undefined] = undefined
    borderLeftRadius: Union[NumberType, Undefined] = undefined
    borderTopRadius: Union[NumberType, Undefined] = undefined
    borderRightRadius: Union[NumberType, Undefined] = undefined
    borderBottomRadius: Union[NumberType, Undefined] = undefined
    borderTopLeftRadius: Union[NumberType, Undefined] = undefined
    borderTopRightRadius: Union[NumberType, Undefined] = undefined
    borderBottomLeftRadius: Union[NumberType, Undefined] = undefined
    borderBottomRightRadius: Union[NumberType, Undefined] = undefined
    borderBend: Union[NumberType, Undefined] = undefined
    borderWidth: Union[NumberType, Undefined] = undefined
    scrollbarWidth: Union[NumberType, Undefined] = undefined
    scrollbarColor: Union[ValueType, Undefined] = undefined

    scrollbarPanelMaterialClass: Union[str, Undefined] = undefined
    scrollbarBackgroundOpacity: Union[NumberType, Undefined] = undefined
    scrollbarBackgroundColor: Union[ValueType, Undefined] = undefined
    scrollbarBorderRadius: Union[NumberType, Undefined] = undefined
    scrollbarBorderLeftRadius: Union[NumberType, Undefined] = undefined
    scrollbarBorderTopRadius: Union[NumberType, Undefined] = undefined
    scrollbarBorderRightRadius: Union[NumberType, Undefined] = undefined
    scrollbarBorderBottomRadius: Union[NumberType, Undefined] = undefined
    scrollbarBorderTopLeftRadius: Union[NumberType, Undefined] = undefined
    scrollbarBorderTopRightRadius: Union[NumberType, Undefined] = undefined
    scrollbarBorderBottomLeftRadius: Union[NumberType, Undefined] = undefined
    scrollbarBorderBottomRightRadius: Union[NumberType, Undefined] = undefined
    zIndexOffset: Union[NumberType, dict[str, NumberType], Undefined] = undefined



class UIKitComponentBase(Component[T_base_props, "UIKitComponentType"]):
    pass

class UIKitContainerBase(ContainerBase[T_container_props, T_child]):
    pass

UIKitComponentType: TypeAlias = Union[UIKitComponentBase, UIKitContainerBase, Fragment, "DataPortal"]

@dataclasses.dataclass
class UIKitContainerBaseProps(ContainerBaseProps):
    pass

@dataclasses.dataclass
class UIKitBaseProps(ThreeBasicProps):
    pass

T_uk_prop = TypeVar("T_uk_prop", bound=UIKitBaseProps)
T_uk_container_prop = TypeVar("T_uk_container_prop",
                              bound=UIKitContainerBaseProps)

class UIKitWithEventBase(UIKitComponentBase[T_uk_prop]):

    def __init__(
            self,
            base_type: UIType,
            prop_cls: Type[T_uk_prop],
            allowed_events: Optional[Iterable[EventDataType]] = None) -> None:
        if allowed_events is None:
            allowed_events = []

        super().__init__(base_type,
                         prop_cls,
                         allowed_events=[
                             FrontendEventType.Click.value,
                             FrontendEventType.DoubleClick.value,
                             FrontendEventType.Enter.value,
                             FrontendEventType.Leave.value,
                             FrontendEventType.Over.value,
                             FrontendEventType.Out.value,
                             FrontendEventType.Up.value,
                             FrontendEventType.Down.value,
                             FrontendEventType.ContextMenu.value,
                             FrontendEventType.Change.value,
                         ] + list(allowed_events))
        self.event_double_click = self._create_event_slot(
            FrontendEventType.DoubleClick)
        self.event_click = self._create_event_slot_noarg(FrontendEventType.Click)
        self.event_enter = self._create_event_slot(FrontendEventType.Enter)
        self.event_leave = self._create_event_slot(FrontendEventType.Leave)
        self.event_over = self._create_event_slot(FrontendEventType.Over)
        self.event_out = self._create_event_slot(FrontendEventType.Out)
        self.event_up = self._create_event_slot(FrontendEventType.Up)
        self.event_down = self._create_event_slot(FrontendEventType.Down)
        self.event_context_menu = self._create_event_slot(
            FrontendEventType.ContextMenu)
        self.event_change = self._create_event_slot(FrontendEventType.Change)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync)

class UIKitContainerWithEventBase(UIKitContainerBase[T_uk_container_prop,
                                                      T_child]):

    def __init__(
            self,
            base_type: UIType,
            prop_cls: Type[T_uk_container_prop],
            children: dict[str, T_child],
            allowed_events: Optional[Iterable[EventDataType]] = None) -> None:
        if allowed_events is None:
            allowed_events = []
        super().__init__(base_type,
                         prop_cls,
                         children,
                         allowed_events=[
                             FrontendEventType.Click.value,
                             FrontendEventType.DoubleClick.value,
                             FrontendEventType.Enter.value,
                             FrontendEventType.Leave.value,
                             FrontendEventType.Over.value,
                             FrontendEventType.Out.value,
                             FrontendEventType.Up.value,
                             FrontendEventType.Down.value,
                             FrontendEventType.ContextMenu.value,
                         ] + list(allowed_events))
        self.event_double_click = self._create_event_slot(
            FrontendEventType.DoubleClick)
        self.event_click = self._create_event_slot_noarg(FrontendEventType.Click)
        self.event_enter = self._create_event_slot(FrontendEventType.Enter)
        self.event_leave = self._create_event_slot(FrontendEventType.Leave)
        self.event_over = self._create_event_slot(FrontendEventType.Over)
        self.event_out = self._create_event_slot(FrontendEventType.Out)
        self.event_up = self._create_event_slot(FrontendEventType.Up)
        self.event_down = self._create_event_slot(FrontendEventType.Down)
        self.event_context_menu = self._create_event_slot(
            FrontendEventType.ContextMenu)

    async def handle_event(self, ev: Event, is_sync: bool = False):
        return await handle_standard_event(self, ev, is_sync)


@dataclasses.dataclass
class RootProps(UIKitContainerCoreProps, UIKitContainerBaseProps):
    anchorX: Union[NumberType, Undefined] = undefined
    anchorY: Union[NumberType, Undefined] = undefined
    sizeX: Union[NumberType, Undefined] = undefined
    sizeY: Union[NumberType, Undefined] = undefined
    pixelSizeX: Union[NumberType, Undefined] = undefined

@dataclasses.dataclass
class FullscreenProps(RootProps):
    attachCamera: Union[bool, Undefined] = undefined
    distanceToCamera: Union[NumberType, Undefined] = undefined

@dataclasses.dataclass
class ContainerProps(UIKitContainerCoreProps, UIKitContainerBaseProps):
    pass

@dataclasses.dataclass
class ContentProps(ContainerProps):
    depthAlign: Union[Literal["back", "center", "front"], Undefined] = undefined
    keepAspectRatio: Union[bool, Undefined] = undefined


class Root(UIKitContainerWithEventBase[RootProps, UIKitComponentType]):
    # TODO can/should group accept event?
    def __init__(
        self, children: Union[dict[str, UIKitComponentType],
                              list[UIKitComponentType]]
    ) -> None:
        if isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.UIKitRoot, RootProps, children)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

class Container(UIKitContainerWithEventBase[ContainerProps, UIKitComponentType]):
    # TODO can/should group accept event?
    def __init__(
        self, children: Union[dict[str, UIKitComponentType],
                              list[UIKitComponentType]]
    ) -> None:
        if isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.UIKitContainer, ContainerProps, children)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

class Fullscreen(UIKitContainerWithEventBase[FullscreenProps, UIKitComponentType]):
    # TODO can/should group accept event?
    def __init__(
        self, children: Union[dict[str, UIKitComponentType],
                              list[UIKitComponentType]]
    ) -> None:
        if isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.UIKitFullscreen, FullscreenProps, children)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

class Content(UIKitContainerWithEventBase[ContentProps, ThreeComponentType]):
    # TODO can/should group accept event?
    def __init__(
        self, children: Union[dict[str, ThreeComponentType],
                              list[ThreeComponentType]]
    ) -> None:
        if isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}
        for v in children.values():
            assert not isinstance(v, UIKitComponentBase), "component inside Content must be regular three component."
        super().__init__(UIType.UIKitContent, ContentProps, children)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)
