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

from functools import partial
import importlib
import traceback
from typing import Any, Callable
import types
import inspect
import importlib.machinery


def reload_method(method: Callable, module_dict: dict, prev_code: str = ""):
    if isinstance(method, partial):
        method_wrapped = method.func
    else:
        method_wrapped = method
    if not inspect.ismethod(method_wrapped):
        return None, ""
    bound = method_wrapped.__self__
    method_qualname = method_wrapped.__qualname__
    qual_parts = method_qualname.split(".")
    # module = inspect.getmodule(method)
    # if module is None:
    #     return False
    # try:
    #     if module.__spec__ is not None:
    #         # ignore source-file module
    #         if isinstance(module.__spec__.loader, importlib.machinery.SourceFileLoader):
    #             return False
    #     importlib.reload(module)
    #     module_dict = module.__dict__
    # except:
    #     traceback.print_exc()
    #     return False
    new_method_unbound = module_dict[qual_parts[0]]
    for p in qual_parts[1:]:
        new_method_unbound = getattr(new_method_unbound, p)
    new_method_code = None
    if prev_code != "":
        new_method_code_lines, _ = inspect.getsourcelines(new_method_unbound)
        new_method_code = "\n".join(new_method_code_lines)
        if new_method_code == prev_code:
            return None, new_method_code
    # now new_method_unbound should be a unbound method
    if inspect.ismethod(method_wrapped):
        new_method = types.MethodType(new_method_unbound, bound)
        setattr(bound, method_wrapped.__name__, new_method)
        if isinstance(method, partial):
            new_method = partial(new_method, *method.args, **method.keywords)
    else:
        new_method = new_method_unbound
    return new_method, new_method_code


def rebind_object(obj: Any):
    pass
