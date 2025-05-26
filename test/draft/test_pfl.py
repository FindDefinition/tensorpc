import json
from tensorpc.core.datamodel.pfl.pfl_ast import parse_func_to_df_ast, pfl_ast_to_dict, pfl_ast_dump
import rich 
import dataclasses
from tensorpc.core.datamodel.pfl.pfl_std import Math
import tensorpc.core.datamodel.pfl.pfl_std as stl 
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
    a = Math.sin(b)
    dd = np.array([2])
    e = dd[0].tolist()
    print(dd, e, dd.shape[0])

if __name__ == "__main__":
    ast, _ = parse_func_to_df_ast(func2)
    rich.print(pfl_ast_dump(ast))
    with open("build/test.json", "w") as f:
        f.write(json.dumps(pfl_ast_to_dict(ast), indent=2))