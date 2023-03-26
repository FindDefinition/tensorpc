import abc
import enum
import inspect
import types
from functools import partial
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Optional,
                    Set, Tuple, Type, Union)

import numpy as np

from tensorpc.core.inspecttools import get_members
from tensorpc.flow.flowapp.components import mui
from tensorpc.core.moduleid import get_qualname_of_type


class ObjectPreviewHandler(mui.FlexBox):

    async def bind(self, obj):
        pass


class ObjectLayoutHandler(mui.FlexBox):

    @classmethod
    def from_object(cls, obj) -> mui.FlexBox:
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


ALL_OBJECT_PREVIEW_HANDLERS = ObjectPreviewHandlerRegistry()

ALL_OBJECT_LAYOUT_HANDLERS = ObjectLayoutHandlerRegistry()

class ContextMenuType(enum.Enum):
    DataStorage = 0
    DataStorageItem = 1

class TreeItem(abc.ABC):
    @abc.abstractmethod
    async def get_child_desps(self, parent_ns: str) -> Dict[str, mui.JsonLikeNode]:
        raise NotImplementedError
    
    @abc.abstractmethod
    async def get_child(self, key: str) -> Any:
        raise NotImplementedError

    def get_json_like_node(self, parent_ns: str) -> Optional[mui.JsonLikeNode]:
        return None 
    
    async def handle_button(self, button_key: str):
        return 
    
    async def handle_child_button(self, button_key: str, child_key: str):
        return 
    
    async def handle_context_menu(self, type: ContextMenuType, userdata: Optional[Any]):
        return 
    
    async def handle_child_context_menu(self, type: ContextMenuType, child_key: str, userdata: Optional[Any]):
        return 
