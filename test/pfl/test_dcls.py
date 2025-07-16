import dataclasses
import json
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


def test_dcls_basic():
    lib = pfl.parse_func_to_pfl_library(Foo.method)
    print(lib.all_compiled.keys())
    print(json.dumps(lib.dump_to_json_dict()))
    foo = Foo()
    print(foo.method(1.0, 2.0))

if __name__ == "__main__":
    test_dcls_basic()
