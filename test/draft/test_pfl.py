import json
from tensorpc.core.datamodel.pfl.pfl_ast import parse_func_to_df_ast, DFStaticType, pfl_ast_to_dict, pfl_ast_dump
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
    a = len(m.b)
    print(a)
    m.a = stl.Math.max(a, a, 1)
    d = False if a > 5 else True

def func2(a: float, b: float, m: Model):
    a = Math.sin(b)
    b = np.array([2])

if __name__ == "__main__":
    ast, _ = parse_func_to_df_ast(func1)
    rich.print(pfl_ast_dump(ast))
    # with open("test.json", "w") as f:
    #     f.write(json.dumps(pfl_ast_to_dict(ast), indent=2))