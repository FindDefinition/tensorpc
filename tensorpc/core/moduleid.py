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
import traceback
from typing import Type 
import inspect 
from typing import Callable
import dataclasses
import importlib
from pathlib import Path
from typing import Any, List, Optional
import uuid

from tensorpc.core.serviceunit import ServFunctionMeta
import inspect 
from typing import (Any, Callable, Deque, Dict, List, Optional, Set, Tuple,
                    Type, Union)
import importlib.util
import sys

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



@dataclasses.dataclass
class TypeMeta:
    module_key: str 
    local_key: str 
    is_path: bool 
    @property 
    def module_id(self):
        return self.module_key + "::" + self.local_key

    def get_reloaded_module_dict(self):
        if not self.is_path:
            # use importlib to reload module
            module = importlib.import_module(self.module_key)
            if module is None:
                return None
            try:
                importlib.reload(module)
            except:
                traceback.print_exc()
                return None
            module_dict = module.__dict__
            return module_dict
        else:
            mod_name = Path(self.module_key).stem + "_" + uuid.uuid4().hex
            spec = importlib.util.spec_from_file_location(mod_name, self.module_key)
            assert spec is not None, f"your {self.module_key} not exists"
            standard_module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None, "shouldn't happen"
            spec.loader.exec_module(standard_module)
            # do we need to add this module to sys?
            # sys.modules[mod_name] = standard_module
            return standard_module.__dict__

    def get_local_type_from_module_dict(self, module_dict: Dict[str, Any]):
        parts = self.local_key.split("::")
        obj = module_dict[parts[0]]
        for part in parts[1:]:
            obj = getattr(obj, part)
        return obj

def get_obj_type_meta(
        obj_type) -> Optional[TypeMeta]:
    qualname = get_qualname_of_type(obj_type)
    spec = importlib.util.find_spec(qualname.split(".")[0])
    is_standard_module = True
    module_path = ""

    if spec is None or spec.origin is None:
        is_standard_module = False
        try:
            module_path_p =  Path(inspect.getfile(obj_type)).resolve()
            module_path = str(module_path_p)
        except:
            return None
    # else:
    #     try:
    #         module_path_p =  Path(inspect.getfile(obj_type)).resolve()
    #         module_path = str(module_path_p)
    #         try:
    #             module_path_p.relative_to(Path(spec.origin).parent.resolve())
    #         except:
    #             is_standard_module = False
    #     except:
    #         return None
    parts = qualname.split(".")
    res_import_path = ""
    res_import_idx = -1
    cur_mod_import_path = parts[0]
    # cur_mod = None 
    if cur_mod_import_path in sys.modules:
        # cur_mod = sys.modules[cur_mod_import_path]
        res_import_path = cur_mod_import_path
        res_import_idx = 1
    count = 1 
    for part in parts[1:]:
        cur_mod_import_path += f".{part}"
        if cur_mod_import_path in sys.modules:
            # cur_mod = sys.modules[cur_mod_import_path]
            res_import_path = cur_mod_import_path
            res_import_idx = count + 1
        count += 1
    assert res_import_path is not None  
    module_import_path = res_import_path
    local_import_path = "::".join(parts[res_import_idx:])
    if not is_standard_module:
        module_import_path = module_path
    return TypeMeta(module_import_path, local_import_path, not is_standard_module)
