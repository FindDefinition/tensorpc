import dataclasses
import json
from typing import Any
import tensorpc.core.pfl.backends.js as stl 
from tensorpc.core import pfl


@dataclasses.dataclass
class Foo:
    val: float = 5.0
    
    def method_dep(self, a: float, b: float) -> float:
        return a + b

    def method(self, a: float, b: float) -> float:
        c = stl.Math.pow(a, b)
        if a > 10:
            d = 5 
        else:
            d = 3
        j = 1
        for j in range(10):
            d += j * self.val
        return self.method_dep(c, d)

@dataclasses.dataclass
class FooWithInit:
    val: float

    def __init__(self, vv: float):
        self.val = vv + 10.0

@pfl.mark_pfl_compilable(is_template=True)
@dataclasses.dataclass
class FooT:
    valX: Any

    def __init__(self, vv: float):
        self.valX = vv + 10.0


def main(a: int):
    # foo = Foo()
    # foo2 = FooWithInit(5.0)

    foo3 = FooT(5.0)
    print(foo3.valX)

    # return foo.method(1.0, 2.0)

def test_custom_dcls():
    lib = pfl.parse_func_to_pfl_library(main)
    print(lib.all_compiled.keys())
    # print(json.dumps(lib.dump_to_json_dict()))
    # foo = Foo()
    # print(foo.method(1.0, 2.0))

if __name__ == "__main__":
    test_custom_dcls()
