import asyncio
import enum
import inspect
import types
from functools import partial
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Optional,
                    Set, Tuple, Type, Union)

import numpy as np

from tensorpc.core.inspecttools import get_members
from tensorpc.flow.flowapp.components import mui, three
from tensorpc.flow.flowapp.core import FrontendEventType
from tensorpc.core.moduleid import get_qualname_of_type

from .core import ALL_OBJECT_PREVIEW_HANDLERS, ObjectPreviewHandler
from .tree import _DEFAULT_OBJ_NAME, ObjectTree

_DEFAULT_LOCALS_NAME = "locals"

_MAX_STRING_IN_DETAIL = 10000


class DefaultHandler(ObjectPreviewHandler):
    """
    TODO if the object support any-layout, add a button to enable it.
    """

    def __init__(self) -> None:
        self.tags = mui.FlexBox().prop(flex_flow="row wrap")
        self.title = mui.Typography("").prop(word_break="break-word")
        self.path = mui.Typography("").prop(word_break="break-word")

        self.data_print = mui.Typography("").prop(font_family="monospace",
                                                  font_size="12px",
                                                  white_space="pre-line")
        layout = [
            self.title.prop(font_size="14px", font_family="monospace"),
            self.path.prop(font_size="14px", font_family="monospace"),
            self.tags,
            mui.Divider().prop(padding="3px"),
            mui.HBox([
                mui.Button("print", self._on_print),
            ]),
            self.data_print,
        ]

        super().__init__(layout)
        self.prop(flex_direction="column")
        self.obj: Any = np.zeros([1])

    async def _on_print(self):
        string = str(self.obj)
        if len(string) > _MAX_STRING_IN_DETAIL:
            string = string[:_MAX_STRING_IN_DETAIL] + "..."
        await self.data_print.write(string)

    async def bind(self, obj):
        # bind np object, update all metadata
        self.obj = obj
        ev = self.data_print.update_event(value="")
        ev += self.title.update_event(value=get_qualname_of_type(type(obj)))
        try:
            sf = inspect.getsourcefile(type(obj))
        except TypeError:
            sf = None
        if sf is None:
            sf = ""
        ev += self.path.update_event(value=sf)
        await self.send_and_wait(ev)
        # await self.tags.set_new_layout([*tags])


class ObjectInspector(mui.FlexBox):

    def __init__(self,
                 init: Optional[Any] = None,
                 cared_types: Optional[Set[Type]] = None,
                 ignored_types: Optional[Set[Type]] = None,
                 with_detail: bool = True,
                 use_allotment: bool = False) -> None:
        
        self.detail_container = mui.FlexBox([]).prop(overflow="auto",
                                                     padding="3px")
        if use_allotment:
            self.detail_container.prop(height="100%")
        else:
            self.detail_container.prop(flex=1)

        self.with_detail = with_detail
        self.tree = ObjectTree(init, cared_types, ignored_types)
        if use_allotment:
            layout: mui.LayoutType = [self.tree.prop(overflow="auto", height="100%",)]
        else:
            layout: mui.LayoutType = [self.tree.prop(flex=1)]

        if with_detail:
            if not use_allotment:
                layout.append(mui.Divider())
            layout.append(self.detail_container)
        self.default_handler = DefaultHandler()
        super().__init__(layout)
        if use_allotment:
            self.prop(overflow="hidden",
                    default_sizes=[1, 1] if with_detail else [1],
                    vertical=True)
        else:
            self.prop(flex_direction="column",
                    flex=1,
                    overflow="hidden",
                    min_height=0,
                    min_width=0)

        if with_detail:
            self.tree.tree.register_event_handler(
                FrontendEventType.TreeItemSelect.value, self._on_select)
        self._type_to_handler_object: Dict[Type[Any], ObjectPreviewHandler] = {}

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
            handler_type: Optional[Type[ObjectPreviewHandler]] = None
            if obj_type in ALL_OBJECT_PREVIEW_HANDLERS:
                handler_type = ALL_OBJECT_PREVIEW_HANDLERS[obj_type]
            elif obj_qualname in ALL_OBJECT_PREVIEW_HANDLERS:
                handler_type = ALL_OBJECT_PREVIEW_HANDLERS[obj_qualname]
            # else:
            #     handler_type = DefaultHandler
            # TODO use a base handler here.
            # await self.detail_container.set_new_layout([])
            # return
            if handler_type is not None:
                handler = handler_type()
            else:
                handler = self.default_handler
            self._type_to_handler_object[obj_type] = handler
        childs = list(self.detail_container._child_comps.values())
        if not childs or childs[0] is not handler:
            await self.detail_container.set_new_layout([handler])
        await handler.bind(obj)

    async def set_object(self, obj, key: str = _DEFAULT_OBJ_NAME):
        await self.tree.set_object(obj, key)

    async def update_locals(self,
                            key: str = _DEFAULT_LOCALS_NAME,
                            *,
                            _frame_cnt: int = 1):
        cur_frame = inspect.currentframe()
        assert cur_frame is not None
        frame = cur_frame
        while _frame_cnt > 0:
            frame = cur_frame.f_back
            assert frame is not None
            cur_frame = frame
            _frame_cnt -= 1
        # del frame
        local_vars = cur_frame.f_locals.copy()
        del frame
        del cur_frame
        await self.tree.set_object(local_vars, key)

    async def update_locals_sync(self, key: str = _DEFAULT_LOCALS_NAME, *,
                                    _frame_cnt: int = 1):
        """update locals in sync manner, usually used on non-sync code via appctx.
        """
        fut = asyncio.run_coroutine_threadsafe(self.update_locals(key, _frame_cnt=1 + _frame_cnt), asyncio.get_running_loop())
        return fut.result()

    async def set_object_sync(self, obj, key: str = _DEFAULT_OBJ_NAME):
        """set object in sync manner, usually used on non-sync code via appctx.
        """
        fut = asyncio.run_coroutine_threadsafe(self.set_object(obj, key), asyncio.get_running_loop())
        return fut.result()

    async def update_tree(self):
        await self.tree.update_tree()

    async def remove_object(self, key: str):
        await self.tree.remove_object(key)
