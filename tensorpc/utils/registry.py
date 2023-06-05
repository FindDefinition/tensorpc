# Copyright 2022 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, Hashable, List, Optional, Generic, Type, TypeVar, Union

T = TypeVar("T", bound=Union[Type, Callable])


class HashableRegistry(Generic[T]):

    def __init__(self, allow_duplicate: bool = False):
        self.global_dict: Dict[Hashable, T] = {}
        self.allow_duplicate = allow_duplicate

    def register(self, func=None, key: Optional[Hashable] = None):

        def wrapper(func: T) -> T:
            key_ = key
            if key is None:
                key_ = func.__name__
            if not self.allow_duplicate and key_ in self.global_dict:
                raise KeyError("key {} already exists".format(key_))
            self.global_dict[key_] = func
            return func

        if func is None:
            return wrapper
        else:
            return wrapper(func)

    def __contains__(self, key: Hashable):
        return key in self.global_dict

    def __getitem__(self, key: Hashable):
        return self.global_dict[key]

    def items(self):
        yield from self.global_dict.items()


class HashableRegistryKeyOnly(Generic[T]):

    def __init__(self, allow_duplicate: bool = False):
        self.global_dict: Dict[Hashable, T] = {}
        self.allow_duplicate = allow_duplicate

    def register(self, key: Optional[Hashable] = None):

        def wrapper(func: T) -> T:
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


class HashableSeqRegistryKeyOnly(Generic[T]):

    def __init__(self):
        self.global_dict: Dict[Hashable, List[T]] = {}

    def register(self, key: Optional[Hashable] = None):
        def wrapper(func: T) -> T:
            key_ = key
            if key is None:
                key_ = func.__name__
            if key_ not in self.global_dict:
                self.global_dict[key_] = []
            self.global_dict[key_].append(func)
            return func

        return wrapper

    def __contains__(self, key: Hashable):
        return key in self.global_dict

    def __getitem__(self, key: Hashable) -> List[T]:
        return self.global_dict[key]

    def items(self):
        yield from self.global_dict.items()
