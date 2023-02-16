from functools import partial
import types
from tensorpc.flow.flowapp.components import mui
from typing import Any, Callable, Dict, Hashable, Iterable, Optional, Set, Tuple, Type, Union, List
import numpy as np
from tensorpc.utils.moduleid import get_qualname_of_type
import inspect 
import enum
from tensorpc.core.inspecttools import get_members

class ObjectHandler(mui.FlexBox):
    async def bind(self, obj):
        pass

class ObjectHandlerRegistry:
    def __init__(self, allow_duplicate: bool = False):
        self.global_dict: Dict[Hashable, Type[ObjectHandler]] = {} 
        self.allow_duplicate = allow_duplicate

    def register(self, key: Optional[Hashable] = None):
        def wrapper(func: Type[ObjectHandler]):
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


ALL_OBJECT_HANDLERS = ObjectHandlerRegistry()