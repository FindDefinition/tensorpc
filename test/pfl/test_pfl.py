import json
import typing
from tensorpc.core.pfl import parse_func_to_df_ast, pfl_ast_to_dict, pfl_ast_dump, parse_expr_to_df_ast, consteval_expr
import rich 
import dataclasses
from tensorpc.core.pfl.pfl_std import Math
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
    a = Math.sin(b)
    dd = np.array([2])
    e = dd[0].tolist()
    f: np.ndarray = np.zeros([1, 2])
    # print(dd, e, dd.shape[0])
    # for j in range(5):
    #     print(j)

if __name__ == "__main__":
    # pflast, run_scope = parse_expr_to_df_ast("Math().sin(2)")
    # rich.print(pfl_ast_dump(pflast))
    # print(consteval_expr(pflast))
    # typing.get_type_hints
    ast, _ = parse_func_to_df_ast(func2)
    rich.print(pfl_ast_dump(ast))
    # with open("build/test.json", "w") as f:
    #     f.write(json.dumps(pfl_ast_to_dict(ast), indent=2))