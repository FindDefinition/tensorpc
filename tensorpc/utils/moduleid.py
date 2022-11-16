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
from typing import Type 
import inspect 
from typing import Callable

def get_qualname_of_type(klass: Type) -> str:
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + '.' + klass.__qualname__

def is_lambda(obj: Callable):
    if not inspect.isfunction(obj) and not inspect.ismethod(obj):
        return False
    return obj.__qualname__ == "<lambda>"

def is_valid_function(obj: Callable):
    return inspect.isfunction(obj) or inspect.ismethod(obj)

def get_function_qualname(obj: Callable):
    return obj.__qualname__