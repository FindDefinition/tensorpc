import asyncio
import json
import typing
import rich 
import dataclasses
from tensorpc.core import pfl
from tensorpc.core.moduleid import get_module_id_of_type, get_qualname_of_type
import tensorpc.core.pfl.backends.js as stl 
import numpy as np 

def fn_dict(a: dict[str, int]):
    for k, v in a.items():
        print(k, v)

if __name__ == "__main__":
    lib2 = pfl.parse_func_to_pfl_library(fn_dict, parse_cfg=pfl.PFLParseConfig(allow_kw=True))
    print(pfl.unparse_pfl_ast(lib2.get_compiled_unit_specs(fn_dict)))
    print(json.dumps(pfl.pfl_ast_to_dict(lib2.get_compiled_unit_specs(fn_dict))))
