import dataclasses
import importlib
from pathlib import Path
from typing import Any, List, Optional

from tensorpc.core.serviceunit import ServFunctionMeta
import inspect
import rtx
from typing import (Any, Callable, Deque, Dict, List, Optional, Set, Tuple,
                    Type, Union)
import importlib.util
import sys
# def reload_object_methods(obj: Any, previous_metas: Optional[List[ServFunctionMeta]] = None):

#     pass


class A:
    pass


def get_qualname_of_type(klass) -> str:
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + '.' + klass.__qualname__


@dataclasses.dataclass
class TypeMeta:
    module_key: str
    local_key: str
    is_path: bool

    @property
    def module_id(self):
        return self.module_key + "::" + self.local_key


def get_obj_type_meta(obj_type) -> Optional[TypeMeta]:
    qualname = get_qualname_of_type(obj_type)
    spec = importlib.util.find_spec(qualname.split(".")[0])
    is_standard_module = True
    if spec is None or spec.origin is None:
        is_standard_module = False
    module_path = ""
    # TODO handle ipython
    if spec is None or spec.origin is None:
        is_standard_module = False
        try:
            module_path_p = Path(inspect.getfile(obj_type)).resolve()
            module_path = str(module_path_p)
        except:
            return None
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
                    not is_standard_module)


aa = A()
print(
    get_qualname_of_type(A),
    get_qualname_of_type(rtx.B),
)
print(get_obj_type_meta(ServFunctionMeta), get_obj_type_meta(rtx.B))
# if __name__ == "__main__":
#     a = A()

#     print(aa)
