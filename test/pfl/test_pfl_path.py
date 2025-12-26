from tensorpc.core import pfl
import tensorpc.core.pfl.backends.js as stl 
import dataclasses
from tensorpc.core.pfl.pflpath import search
@dataclasses.dataclass
class Model:
    a: int 
    b: list[int]

def test_simple_expr():

    expr = "root.a + 5"
    node = pfl.PFLParser().parse_expr_string_to_pfl_ast(expr, {
        "root": Model(1, [2])
    }, {})
    print(node)
    node2 = pfl.PFLParser().parse_expr_string_to_pfl_ast(expr, {}, {
        "root": Model
    })
    print(node2)

def test_partial_expr():

    exprs = [
        "root.a + 5",
        "not b[0]",
        "b[0] > a['asd']",

    ]
    for expr in exprs:
        node = pfl.PFLParser().parse_expr_string_to_pfl_ast(expr, {}, {},
            partial_type_infer=True)
        print(expr, node)

def test_simple_path():
    exprs = [
        ["a + b", {"a": 5, "b": 3}],
        # ["a.c", {"a": 5}],
        ["a.b", {"a": {"b": 10}}],
        ["Math.max(a, b) + Math.E", {"a": 5, "b": 3}],
        ["getRoot().a", {"a": 5, "b": 3}],
        ["len(a)", {"a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}],
        ["a[1:4]", {"a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}],
        ["a[7:2:-2]", {"a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}],
        ["a[4:]", {"a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}],
        ["a_lst + b_lst", {"a_lst": [5], "b_lst": [3]}],

    ]
    for expr, data in exprs:
        res = search(expr, data)
        print(expr, res)

if __name__ == "__main__":
    # test_partial_expr()
    test_simple_path()