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
from tensorpc.flow.flowapp.components import mui
from tensorpc.core.moduleid import get_qualname_of_type
from tensorpc.flow.flowapp.objtree import UserObjTree, UserObjTreeProtocol
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
    width: float  = 1.0
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

class ObjectLayoutHandlerRegistry(HashableRegistryKeyOnly[Type[ObjectLayoutHandler]]):
    def check_type_exists(self, type: Type) -> bool:
        qname = get_qualname_of_type(type)
        if type in self:
            return True 
        return qname in self
 

ALL_OBJECT_PREVIEW_HANDLERS: HashableRegistryKeyOnly[Type[ObjectPreviewHandler]] = HashableRegistryKeyOnly(allow_duplicate=True)

ALL_OBJECT_LAYOUT_HANDLERS: ObjectLayoutHandlerRegistry = ObjectLayoutHandlerRegistry(allow_duplicate=True)

ALL_OBJECT_LAYOUT_CREATORS: HashableRegistryKeyOnly[Type[ObjectLayoutCreator]] = HashableRegistryKeyOnly()

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
    def patch_node(self, obj: Any, node: JsonLikeNode) -> Optional[JsonLikeNode]:
        """modify/patch node created from `parse_obj_to_tree_node`
        """
    
    async def handle_button(self, obj_trace: List[Any], node_trace: List[JsonLikeNode], button_id: str) -> Optional[bool]:
        return None
    
    async def handle_context_menu(self, obj_trace: List[Any], node_trace: List[JsonLikeNode], userdata: Dict[str, Any]) -> Optional[bool]:
        return None


def register_obj_preview_handler(cls):
    return ALL_OBJECT_PREVIEW_HANDLERS.register(cls)


def register_obj_layout_handler(cls):
    return ALL_OBJECT_LAYOUT_HANDLERS.register(cls)


