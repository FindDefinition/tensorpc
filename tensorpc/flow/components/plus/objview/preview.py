import abc
from functools import partial
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Optional,
                    Sequence, Set, Tuple, Type, TypeVar, Union)

from tensorpc.core.serviceunit import AppFuncType, ReloadableDynamicClass, ServFunctionMeta
from tensorpc.flow.components.plus.styles import CodeStyles
from tensorpc.flow.core.appcore import Event, get_app, get_editable_app
from tensorpc.flow.components import mui
from tensorpc.flow.components.plus.reload_utils import preview_layout_reload
from tensorpc.flow.core.component import FlowSpecialMethods, FrontendEventType
from tensorpc.flow.core.objtree import UserObjTreeProtocol
from ..handlers.common import DefaultHandler
from ..core import (ObjectPreviewHandler, DataClassesType,
                    ObjectPreviewLayoutHandleManager)

class ObjectPreviewBase(mui.FlexBox, abc.ABC):
    @abc.abstractmethod
    async def clear_preview_layout(self): ...

    @abc.abstractmethod
    async def set_obj_preview_layout(
            self,
            obj: Any,
            uid: Optional[str] = None,
            root: Optional[UserObjTreeProtocol] = None,
            header: Optional[str] = None): ...

    @abc.abstractmethod
    async def set_preview_layout(
            self,
            layout: mui.LayoutType): ...

class ObjectPreview(ObjectPreviewBase):
    def __init__(self, enable_reload: bool = True):
        self.preview_container = mui.VBox([]).prop(overflow="hidden",
                                                   flex=1,
                                                   alignItems="stretch")
        self.preview_header = mui.Typography("").prop(
            variant="caption", fontFamily=CodeStyles.fontFamily)
        super().__init__([
            self.preview_header,
            mui.Divider(),
            self.preview_container,
        ])
        self.prop(overflow="hidden",
                padding="3px",
                flexFlow="column nowrap",
                flex=1,
                width="100%",
                height="100%")

        self._cached_preview_handler = ObjectPreviewLayoutHandleManager()
        self._current_preview_layout: Optional[mui.FlexBox] = None
        self._cached_preview_layouts: Dict[str, Tuple[mui.FlexBox, int]] = {}
        self._default_handler = DefaultHandler()
        self._enable_reload = enable_reload

    async def clear_preview_layout(self):
        await self.preview_container.set_new_layout([])

    async def set_obj_preview_layout(
            self,
            obj: Any,
            uid: Optional[str] = None,
            root: Optional[UserObjTreeProtocol] = None,
            header: Optional[str] = None):

        preview_layout: Optional[mui.FlexBox] = None
        obj_type: Type = type(obj)

        # preview layout is checked firstly, then preview handler.
        if self._cached_preview_handler.is_in_cache(obj):
            handler = self._cached_preview_handler.query_handler(obj)
            assert handler is not None
        else:
            metas = self.flow_app_comp_core.reload_mgr.query_type_method_meta(
                obj_type, True, include_base=True)
            special_methods = FlowSpecialMethods(metas)
            if special_methods.create_preview_layout is not None:
                if uid is not None and uid in self._cached_preview_layouts:
                    preview_layout, obj_id = self._cached_preview_layouts[uid]
                    if obj_id != id(obj):
                        preview_layout = None
                        self._cached_preview_layouts.pop(uid)
                if preview_layout is None:
                    if root is None:
                        preview_layout = mui.flex_preview_wrapper(
                            obj, metas, self.flow_app_comp_core.reload_mgr)
                    else:
                        with root.enter_context(root):
                            preview_layout = mui.flex_preview_wrapper(
                                obj, metas, self.flow_app_comp_core.reload_mgr)
                handler = self._default_handler
            else:
                handler = self._cached_preview_handler.query_handler(obj)
                if handler is None:
                    handler = self._default_handler
            # if preview_layout is None:
            #     self._type_to_handler_object[modified_obj_type] = handler
        if preview_layout is not None:
            if root is not None:
                preview_layout.set_flow_event_context_creator(
                    lambda: root.enter_context(root))
            # preview_layout.event_emitter.remove_listener()
            if uid is not None and self._enable_reload:
                if self._current_preview_layout is None:
                    get_editable_app().observe_layout(
                        preview_layout,
                        partial(self._on_preview_layout_reload,
                                uid=uid,
                                obj_id=id(obj)))
                else:
                    get_editable_app().observe_layout(
                        preview_layout,
                        partial(self._on_preview_layout_reload,
                                uid=uid,
                                obj_id=id(obj)))
            self._current_preview_layout = preview_layout
            if uid is not None:
                self._cached_preview_layouts[uid] = (preview_layout, id(obj))
            # self.__install_preview_event_listeners(preview_layout)
            await self.preview_container.set_new_layout([preview_layout])
        else:
            childs = list(self.preview_container._child_comps.values())
            if not childs or childs[0] is not handler:
                await self.preview_container.set_new_layout([handler])
            await handler.bind(obj, uid)
        if header is not None:
            await self.preview_header.write(header)

    async def set_preview_layout(
            self,
            layout: mui.LayoutType,
            header: Optional[str] = None):
        await self.preview_container.set_new_layout(layout) # type: ignore
        if header is not None:
            await self.preview_header.write(header)

    async def _on_preview_layout_reload(self, layout: mui.FlexBox,
                                        create_layout: ServFunctionMeta,
                                        uid: str, obj_id: int):
        layout_flex = await preview_layout_reload(
            lambda x: self.preview_container.set_new_layout([x]), layout,
            create_layout)
        if layout_flex is not None:
            get_editable_app().observe_layout(
                layout_flex,
                partial(self._on_preview_layout_reload, uid=uid,
                        obj_id=obj_id))
            self._cached_preview_layouts[uid] = (layout_flex, obj_id)
            return layout_flex
