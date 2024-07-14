"""
## CustomTreeItemHandler vs TreeItem vs UserObjTree

* Tree Item To Node

CustomTreeItemHandler: full control

TreeItem: full control

UserObjTree: none

* child of obj

CustomTreeItemHandler: full control

TreeItem: full control

UserObjTree: only support sync

* Event Handling

CustomTreeItemHandler: full control

TreeItem: control self and direct child

UserObjTree: none
"""

import abc
import dataclasses
import enum
import inspect
import types
from functools import partial
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Optional,
                    Set, Tuple, Type)

import numpy as np

from tensorpc.core.inspecttools import get_members
from tensorpc.flow.components import mui
from tensorpc.core.moduleid import get_qualname_of_type
from tensorpc.flow.core.objtree import UserObjTree, UserObjTreeProtocol
from tensorpc.flow.jsonlike import JsonLikeNode
from tensorpc.utils.registry import HashableRegistryKeyOnly


class PriorityCommon(enum.IntEnum):
    Lowest = 0
    Low = 20
    Normal = 40
    High = 60
    Highest = 80


@dataclasses.dataclass
class ObjectGridItemConfig:
    width: float = 1.0
    height: float = 1.0
    priority: int = 0

    # used for internal layout only
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0


USER_OBJ_TREE_TYPES: Set[Any] = {UserObjTree}


def register_user_obj_tree_type(type):
    USER_OBJ_TREE_TYPES.add(type)


class ObjectPreviewHandler(mui.FlexBox):

    @abc.abstractmethod
    async def bind(self, obj: Any, uid: str):
        pass


class ObjectLayoutHandler(abc.ABC):

    @abc.abstractmethod
    def create_layout(self, obj: Any) -> mui.FlexBox:
        raise NotImplementedError

    def get_grid_layout_item(self, obj: Any) -> ObjectGridItemConfig:
        return ObjectGridItemConfig(1.0, 1.0)


class ObjectLayoutCreator(abc.ABC):

    @abc.abstractmethod
    def create(self) -> mui.FlexBox:
        raise NotImplementedError


class ObjectLayoutHandlerRegistry(
        HashableRegistryKeyOnly[Type[ObjectLayoutHandler]]):

    def check_type_exists(self, type: Type) -> bool:
        qname = get_qualname_of_type(type)
        if type in self:
            return True
        return qname in self


class ObjectLayoutHandleManager:
    def __init__(self):
        self._type_to_handler_object: Dict[Type[Any], ObjectLayoutHandler] = {}

    def is_in_cache(self, obj: Any) -> bool:
        obj_type = type(obj)
        is_dcls = dataclasses.is_dataclass(obj)
        if obj_type in self._type_to_handler_object:
            return True 
        elif is_dcls and DataClassesType in self._type_to_handler_object:
            return True 
        return False 

    def query_handler(self, obj: Any) -> Optional[ObjectLayoutHandler]:
        obj_type = type(obj)
        is_dcls = dataclasses.is_dataclass(obj)
        handler: Optional[ObjectLayoutHandler] = None
        if obj_type in self._type_to_handler_object:
            handler = self._type_to_handler_object[obj_type]
        elif is_dcls and DataClassesType in self._type_to_handler_object:
            handler = self._type_to_handler_object[DataClassesType]
        else:
            obj_qualname = get_qualname_of_type(type(obj))
            handler_type: Optional[Type[ObjectLayoutHandler]] = None
            modified_obj_type = obj_type

            if obj is not None:
                # check standard type first, if not found, check datasetclass type.
                if obj_type in ALL_OBJECT_LAYOUT_HANDLERS:
                    handler_type = ALL_OBJECT_LAYOUT_HANDLERS[obj_type]
                elif obj_qualname in ALL_OBJECT_LAYOUT_HANDLERS:
                    handler_type = ALL_OBJECT_LAYOUT_HANDLERS[obj_qualname]
                elif is_dcls and DataClassesType in ALL_OBJECT_LAYOUT_HANDLERS:
                    handler_type = ALL_OBJECT_LAYOUT_HANDLERS[
                        DataClassesType]
                    modified_obj_type = DataClassesType
            if handler_type is not None:
                handler = handler_type()
                self._type_to_handler_object[modified_obj_type] = handler
        return handler
class ObjectPreviewLayoutHandleManager:
    def __init__(self):
        self._type_to_handler_object: Dict[Type[Any], ObjectPreviewHandler] = {}

    def is_in_cache(self, obj: Any) -> bool:
        obj_type = type(obj)
        is_dcls = dataclasses.is_dataclass(obj)
        if obj_type in self._type_to_handler_object:
            return True 
        elif is_dcls and DataClassesType in self._type_to_handler_object:
            return True 
        return False 

    def query_handler(self, obj: Any) -> Optional[ObjectPreviewHandler]:
        obj_type = type(obj)
        is_dcls = dataclasses.is_dataclass(obj)
        handler: Optional[ObjectPreviewHandler] = None
        if obj_type in self._type_to_handler_object:
            handler = self._type_to_handler_object[obj_type]
        elif is_dcls and DataClassesType in self._type_to_handler_object:
            handler = self._type_to_handler_object[DataClassesType]
        else:
            obj_qualname = get_qualname_of_type(type(obj))
            handler_type: Optional[Type[ObjectPreviewHandler]] = None
            modified_obj_type = obj_type

            if obj is not None:
                # check standard type first, if not found, check datasetclass type.
                if obj_type in ALL_OBJECT_PREVIEW_HANDLERS:
                    handler_type = ALL_OBJECT_PREVIEW_HANDLERS[obj_type]
                elif obj_qualname in ALL_OBJECT_PREVIEW_HANDLERS:
                    handler_type = ALL_OBJECT_PREVIEW_HANDLERS[obj_qualname]
                elif is_dcls and DataClassesType in ALL_OBJECT_PREVIEW_HANDLERS:
                    handler_type = ALL_OBJECT_PREVIEW_HANDLERS[
                        DataClassesType]
                    modified_obj_type = DataClassesType
            if handler_type is not None:
                handler = handler_type()
                self._type_to_handler_object[modified_obj_type] = handler
        return handler

ALL_OBJECT_PREVIEW_HANDLERS: HashableRegistryKeyOnly[
    Type[ObjectPreviewHandler]] = HashableRegistryKeyOnly(allow_duplicate=True)

ALL_OBJECT_LAYOUT_HANDLERS: ObjectLayoutHandlerRegistry = ObjectLayoutHandlerRegistry(
    allow_duplicate=True)

ALL_OBJECT_LAYOUT_CREATORS: HashableRegistryKeyOnly[
    Type[ObjectLayoutCreator]] = HashableRegistryKeyOnly()


class ContextMenuType(enum.Enum):
    DataStorageStore = 0
    DataStorageItemDelete = 1
    DataStorageItemCommand = 2

    CopyReadItemCode = 3


class DataClassesType:
    """a placeholder that used for custom handlers.
    user need to register this type to make sure
    handler is used if object is dataclass.
    """
    pass


class CustomTreeItemHandler(abc.ABC):
    """
    TODO should we use lazy load in TreeItem?
    """

    @abc.abstractmethod
    async def get_childs(self, obj: Any) -> Optional[Dict[str, Any]]:
        """if return None, we will use default method to extract childs
        of object.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def patch_node(self, obj: Any,
                   node: JsonLikeNode) -> Optional[JsonLikeNode]:
        """modify/patch node created from `parse_obj_to_tree_node`
        """

    async def handle_button(self, obj_trace: List[Any],
                            node_trace: List[JsonLikeNode],
                            button_id: str) -> Optional[bool]:
        return None

    async def handle_context_menu(self, obj_trace: List[Any],
                                  node_trace: List[JsonLikeNode],
                                  userdata: Dict[str, Any]) -> Optional[bool]:
        return None


def register_obj_preview_handler(cls):
    return ALL_OBJECT_PREVIEW_HANDLERS.register(cls)


def register_obj_layout_handler(cls):
    return ALL_OBJECT_LAYOUT_HANDLERS.register(cls)
