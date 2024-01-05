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
import dataclasses
import importlib
import importlib.util
import inspect
import sys
import traceback
import types
import uuid
from pathlib import Path
from typing import (Any, Callable, Deque, Dict, List, Optional, Set, Tuple,
                    Type, Union)
import time
from typing_extensions import TypeAlias
import importlib.machinery
from tensorpc.constants import TENSORPC_FILE_NAME_PREFIX


def get_qualname_of_type(klass: Type) -> str:
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + '.' + klass.__qualname__

def get_mro_qualnames_of_type(klass: Type) -> Set[str]:
    mros = inspect.getmro(klass)
    return set(get_qualname_of_type(mro) for mro in mros)

def is_lambda(obj: Callable):
    if not inspect.isfunction(obj) and not inspect.ismethod(obj):
        return False
    return obj.__qualname__ == "<lambda>"


def is_valid_function(obj: Callable):
    return inspect.isfunction(obj) or inspect.ismethod(obj)


def get_function_qualname(obj: Callable):
    return obj.__qualname__

if sys.version_info >= (3, 10):
    _ClassInfo: TypeAlias = type | types.UnionType | tuple["_ClassInfo", ...]
else:
    _ClassInfo: TypeAlias = Union[type, Tuple["_ClassInfo", ...]] 

def loose_isinstance(obj, _class_or_tuple: _ClassInfo):
    """for reloaded code, the type of obj may be different from the type of the class in the current module.
    """
    obj_qnames = get_mro_qualnames_of_type(type(obj))
    if not isinstance(_class_or_tuple, (list, tuple)):
        _class_or_tuple = (_class_or_tuple,)

    for c in _class_or_tuple:
        if get_qualname_of_type(c) in obj_qnames:
            return True 
    return False 

@dataclasses.dataclass
class InMemoryFSItem:
    path: str 
    st_size: int 
    st_mtime: float
    st_ctime: float
    content: str

class InMemoryFS:
    def __init__(self):
        self.fs_dict: Dict[str, InMemoryFSItem] = {}

    def add_file(self, path: str, content: str):
        self.fs_dict[path] = InMemoryFSItem(path, len(content), time.time(), time.time(), content)

    def modify_file(self, path: str, content: str):
        if path not in self.fs_dict:
            raise ValueError("file not exist")
        self.fs_dict[path] = InMemoryFSItem(path, len(content), time.time(), self.fs_dict[path].st_ctime, content)
    
    def add_or_modify_file(self, path: str, content: str):
        if path not in self.fs_dict:
            return self.add_file(path, content)
        self.fs_dict[path] = InMemoryFSItem(path, len(content), time.time(), self.fs_dict[path].st_ctime, content)

    def stat(self, path: str):
        if path not in self.fs_dict:
            raise OSError(f"file not exist {path}")
        return self.fs_dict[path]

    def __contains__(self, path: str):
        return path in self.fs_dict
    
    def __getitem__(self, path: str):
        return self.fs_dict[path]

    def load_in_memory_module(self, path: str):
        module = types.ModuleType(path)
        spec = importlib.machinery.ModuleSpec(path, None)
        module.__file__ = path
        module.__spec__ = spec
        code_comp = compile(self[path].content, path, "exec")
        exec(code_comp, module.__dict__)
        # we need to add module to sys.modules to get inspect.getfile work.
        sys.modules[path] = module
        return module
    


def is_tensorpc_dynamic_path(path: str):
    return path.startswith(f"<{TENSORPC_FILE_NAME_PREFIX}")

@dataclasses.dataclass
class TypeMeta:
    module_key: str
    local_key: str
    is_path: bool
    is_in_memory: bool = False

    @property
    def module_id(self):
        return self.module_key + "::" + self.local_key

    def get_reloaded_module(self, in_memory_fs: Optional[InMemoryFS] = None):
        if not self.is_path:
            module = sys.modules.get(self.module_key, None)
            # use importlib to reload module
            # module = importlib.import_module(self.module_key)
            if module is None:
                return None
            try:
                importlib.reload(module)
            except:
                traceback.print_exc()
                return None
            module_dict = module.__dict__
            return module_dict, module
        else:
            if self.is_in_memory:
                assert in_memory_fs is not None
            if in_memory_fs is not None and self.is_in_memory:
                standard_module = in_memory_fs.load_in_memory_module(self.module_key)
            else:
                mod_name = Path(self.module_key).stem + "_" + uuid.uuid4().hex
                mod_name = f"<{mod_name}>"
                spec = importlib.util.spec_from_file_location(
                    mod_name, self.module_key)
                assert spec is not None, f"your {self.module_key} not exists"
                standard_module = importlib.util.module_from_spec(spec)
                standard_module.__file__ = self.module_key
                assert spec.loader is not None, "shouldn't happen"
                spec.loader.exec_module(standard_module)
            # do we need to add this module to sys?
            # sys.modules[mod_name] = standard_module
            return standard_module.__dict__, standard_module

    def get_reloaded_module_dict(self):
        res = self.get_reloaded_module()
        if res is not None:
            return res[0]
        return None

    @staticmethod
    def get_local_type_from_module_dict_qualname(qualname: str, module_dict: Dict[str, Any]):
        parts = qualname.split(".")
        obj = module_dict[parts[0]]
        for part in parts[1:]:
            obj = getattr(obj, part)
        return obj

    def get_local_type_from_module_dict(self, module_dict: Dict[str, Any]):
        parts = self.local_key.split("::")
        obj = module_dict[parts[0]]
        for part in parts[1:]:
            obj = getattr(obj, part)
        return obj

def get_obj_type_meta(obj_type) -> Optional[TypeMeta]:
    qualname = get_qualname_of_type(obj_type)
    spec = importlib.util.find_spec(qualname.split(".")[0])
    is_standard_module = True
    is_in_memory = False
    module_path = ""
    if spec is None or spec.origin is None:
        is_standard_module = False
        try:
            path = inspect.getfile(obj_type)
            if path.startswith(f"<{TENSORPC_FILE_NAME_PREFIX}"):
                module_path = path
                is_in_memory = True 
            else:
                module_path_p = Path(path).resolve()
                module_path = str(module_path_p)
        except:
            # all tensorpc dynamic class store path in __module__
            type_path = obj_type.__module__
            if type_path.startswith(f"<{TENSORPC_FILE_NAME_PREFIX}"):
                module_path = type_path
                is_in_memory = True 
            else:
                return None
    # assert spec is not None 
    if spec is not None and spec.origin is not None:
        if "<" in spec.name:
            is_standard_module = False
            module_path = spec.origin
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
    return TypeMeta(module_import_path, local_import_path,
                    not is_standard_module, is_in_memory)
