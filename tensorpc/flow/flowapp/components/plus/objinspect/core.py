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

USER_OBJ_TREE_TYPES: Set[Any] = {UserObjTree}

class ObjectPreviewHandler(mui.FlexBox):

    async def bind(self, obj):
        pass


class ObjectLayoutHandler(mui.FlexBox):

    @classmethod
    def from_object(cls, obj) -> mui.FlexBox:
        raise NotImplementedError


class ObjectLayoutCreator(abc.ABC):

    @abc.abstractmethod
    def create(self) -> mui.FlexBox:
        raise NotImplementedError


class ObjectPreviewHandlerRegistry:

    def __init__(self, allow_duplicate: bool = True):
        self.global_dict: Dict[Hashable, Type[ObjectPreviewHandler]] = {}
        self.allow_duplicate = allow_duplicate

    def register(self, key: Optional[Hashable] = None):

        def wrapper(func: Type[ObjectPreviewHandler]):
            key_ = key
            if key is None:
                key_ = func.__name__
            if not self.allow_duplicate and key_ in self.global_dict:
                raise KeyError("key {} already exists".format(key_))
            self.global_dict[key_] = func
            return func

        return wrapper

    def __contains__(self, key: Hashable):
        return key in self.global_dict

    def __getitem__(self, key: Hashable):
        return self.global_dict[key]

    def items(self):
        yield from self.global_dict.items()


class ObjectLayoutHandlerRegistry:

    def __init__(self, allow_duplicate: bool = True):
        self.global_dict: Dict[Hashable, Type[ObjectLayoutHandler]] = {}
        self.allow_duplicate = allow_duplicate

    def register(self, key: Optional[Hashable] = None):

        def wrapper(func: Type[ObjectLayoutHandler]):
            key_ = key
            if key is None:
                key_ = func.__name__
            if not self.allow_duplicate and key_ in self.global_dict:
                raise KeyError("key {} already exists".format(key_))
            self.global_dict[key_] = func
            return func

        return wrapper

    def __contains__(self, key: Hashable):
        return key in self.global_dict

    def __getitem__(self, key: Hashable):
        return self.global_dict[key]

    def items(self):
        yield from self.global_dict.items()

class ObjectLayoutCreatorRegistry:

    def __init__(self, allow_duplicate: bool = True):
        self.global_dict: Dict[Hashable, Type[ObjectLayoutCreator]] = {}
        self.allow_duplicate = allow_duplicate

    def register(self, key: Optional[Hashable] = None):

        def wrapper(func: Type[ObjectLayoutCreator]):
            key_ = key
            if key is None:
                key_ = func.__name__
            if not self.allow_duplicate and key_ in self.global_dict:
                raise KeyError("key {} already exists".format(key_))
            self.global_dict[key_] = func
            return func

        return wrapper

    def __contains__(self, key: Hashable):
        return key in self.global_dict

    def __getitem__(self, key: Hashable):
        return self.global_dict[key]

    def items(self):
        yield from self.global_dict.items()

ALL_OBJECT_PREVIEW_HANDLERS = ObjectPreviewHandlerRegistry()

ALL_OBJECT_LAYOUT_HANDLERS = ObjectLayoutHandlerRegistry()

ALL_OBJECT_LAYOUT_CREATORS = ObjectLayoutCreatorRegistry()

class ContextMenuType(enum.Enum):
    DataStorageStore = 0
    DataStorageItemDelete = 1
    DataStorageItemCommand = 2

