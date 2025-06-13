import asyncio
import json
import typing
import rich 
import dataclasses
from tensorpc.core import pfl
from tensorpc.core.moduleid import get_module_id_of_type, get_qualname_of_type
import tensorpc.core.pfl.pfl_std as stl 
import numpy as np 
@dataclasses.dataclass
class Model:
    a: int 
    b: list[int]

def func1(a: int, b: str, m: Model):
    a += 1
    c = {
        "a": 1,
        "b": 2,
        "c": 3
    }
    c["a"] += 1
    if b.startswith("a"):
        c["c"] -= 5
    m.b.append(a)
    ff = len(m.b)
    print(a)
    m.a = stl.Math.max(a, a, 1)
    d = False if a > 5 else True
    e = stl.Math()

def func2(a: float, b: float):
    a = stl.Math.sin(b)
    dd = np.array([2])
    e = dd[0].tolist()
    f: np.ndarray = np.zeros([1, 2])
    # print(dd, e, dd.shape[0])
    # for j in range(5):
    #     print(j)
    return f

@pfl.mark_pfl_compilable
def add(x: int, y: int) -> int:
    return x + y

def func3(a: float, b: float):
    c = stl.Math.pow(a, y=b)
    if a > 10:
        d = 5 
    else:
        d = 3
    for j in range(10):
        d += 1
    return add(d, 3)

if __name__ == "__main__":
    # pflast, run_scope = parse_expr_to_df_ast("Math().sin(2)")
    # rich.print(ast_dump(pflast))
    # print(consteval_expr(pflast))
    # typing.get_type_hints
    lib = pfl.parse_func_to_pfl_library(func3, parse_cfg=pfl.PFLParseConfig(allow_kw=True))
    print(lib.all_compiled.keys())
    runner = pfl.PFLAsyncRunner(lib)
    print(asyncio.run(runner.run_func(get_module_id_of_type(func3), {
        "a": 5,
        "b": 3
    })))
    # for k, v in all_compiled.items():
    #     print(k, v.ret_st)
    #     rich.print(pfl.ast_dump(v))

    # ast, _ = pfl.parse_expr_to_df_ast("a[None, :2]", {
    #     "a": np.zeros([2, 2])
    # }, parse_cfg=pfl.PFLParseConfig(allow_slice=True, allow_nd_slice=True))
    # rich.print(pfl.ast_dump(ast))
    # rich.print(pfl.unparse_pfl_expr(ast))

    # with open("build/test.json", "w") as f:
    #     f.write(json.dumps(pfl_ast_to_dict(ast), indent=2))