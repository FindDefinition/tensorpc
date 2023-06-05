import abc
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
from tensorpc.utils.registry import HashableRegistryKeyOnly

USER_OBJ_TREE_TYPES: Set[Any] = {UserObjTree}

def register_user_obj_tree_type(type):
    USER_OBJ_TREE_TYPES.add(type)

class ObjectPreviewHandler(mui.FlexBox):
    @abc.abstractmethod
    async def bind(self, obj: Any, uid: str):
        pass


class ObjectLayoutHandler(mui.FlexBox):

    @classmethod
    def from_object(cls, obj) -> mui.FlexBox:
        raise NotImplementedError


class ObjectLayoutCreator(abc.ABC):

    @abc.abstractmethod
    def create(self) -> mui.FlexBox:
        raise NotImplementedError

ALL_OBJECT_PREVIEW_HANDLERS: HashableRegistryKeyOnly[Type[ObjectPreviewHandler]] = HashableRegistryKeyOnly()

ALL_OBJECT_LAYOUT_HANDLERS: HashableRegistryKeyOnly[Type[ObjectLayoutHandler]] = HashableRegistryKeyOnly()

ALL_OBJECT_LAYOUT_CREATORS: HashableRegistryKeyOnly[Type[ObjectLayoutCreator]] = HashableRegistryKeyOnly()

class ContextMenuType(enum.Enum):
    DataStorageStore = 0
    DataStorageItemDelete = 1
    DataStorageItemCommand = 2

    CopyReadItemCode = 3
