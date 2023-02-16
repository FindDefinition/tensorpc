from functools import partial
import types
from tensorpc.flow.flowapp.components import mui, three
from typing import Any, Callable, Dict, Hashable, Iterable, Optional, Set, Tuple, Type, Union, List
import numpy as np
from tensorpc.flow.flowapp.core import FrontendEventType
from tensorpc.utils.moduleid import get_qualname_of_type
import inspect
import enum
from tensorpc.core.inspecttools import get_members
from .tree import ObjectTree, _DEFAULT_OBJ_NAME
from .core import ALL_OBJECT_HANDLERS, ObjectHandler


class ObjectInspector(mui.FlexBox):

    def __init__(self,
                 init: Optional[Any] = None,
                 cared_types: Optional[Set[Type]] = None,
                 ignored_types: Optional[Set[Type]] = None,
                 with_detail: bool = True) -> None:
        self.detail_container = mui.FlexBox([]).prop(flex=1,
                                                     overflow="auto",
                                                     padding="3px")
        self.with_detail = with_detail
        self.tree = ObjectTree(init, cared_types, ignored_types)
        layout: mui.LayoutType = [self.tree.prop(flex=1)]
        if with_detail:
            layout.append(mui.Divider())
            layout.append(self.detail_container)
        super().__init__(layout)
        self.prop(flex_direction="column", flex=1, overflow="hidden", min_height=0, min_width=0)
        if with_detail:
            self.tree.tree.register_event_handler(
                FrontendEventType.TreeItemSelect.value, self._on_select)
        self._type_to_handler_object: Dict[Type[Any], ObjectHandler] = {}

    async def _on_select(self, uid: str):
        obj, found = self.tree._get_obj_by_uid(uid)
        if not found:
            raise ValueError(
                f"your object {uid} is invalid, may need to reflesh")
        obj_type = type(obj)
        if obj_type in self._type_to_handler_object:
            handler = self._type_to_handler_object[obj_type]
        else:
            obj_qualname = get_qualname_of_type(type(obj))
            handler_type: Optional[Type[ObjectHandler]] = None
            if obj_type in ALL_OBJECT_HANDLERS:
                handler_type = ALL_OBJECT_HANDLERS[obj_type]
            elif obj_qualname in ALL_OBJECT_HANDLERS:
                handler_type = ALL_OBJECT_HANDLERS[obj_qualname]
            else:
                # TODO use a base handler here.
                await self.detail_container.set_new_layout([])
                return
            handler = handler_type()
            self._type_to_handler_object[obj_type] = handler
        childs = list(self.detail_container._child_comps.values())
        if not childs or childs[0] is not handler:
            await self.detail_container.set_new_layout([handler])
        await handler.bind(obj)

    async def set_object(self, obj, key: str = _DEFAULT_OBJ_NAME):
        await self.tree.set_object(obj, key)
    
    async def update_tree(self):
        await self.tree.update_tree()

    async def remove_object(self, key: str):
        await self.tree.remove_object(key)
