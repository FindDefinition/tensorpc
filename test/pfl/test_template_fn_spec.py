import dataclasses
import json
from typing import Annotated, Any
from tensorpc.core import pfl

pfl.register_backend("custom", pfl.PFLParseConfig(
    allow_var_union=False,
    allow_kw=False,
    allow_nd_slice=False,
    allow_slice=False,
    allow_new_var_after_if=True,
    tuple_assign_must_be_homogeneous=True,
    allow_custom_class=True,
))

@pfl.register_pfl_std(mapped_name="Custom", backend="custom")
@dataclasses.dataclass
class CustomStd:
    @staticmethod 
    def warp_specialize(default_args: Annotated[Any, pfl.PFLTemplateFnSpecArgsMeta("default")], 
                default_partition: Annotated[Any, pfl.PFLTemplateFnSpecMeta("default")], 
                worker_args: Annotated[Any, pfl.PFLTemplateFnSpecArgsMeta("worker")], 
                worker_partitions: Annotated[list[Any], pfl.PFLTemplateFnSpecMeta("worker")], 
                worker_num_warps: list[int],
                worker_num_regs: list[int]) -> None:
        pass


@pfl.mark_pfl_compilable(is_template=True)
def add(a, b):
    return a + b 

@pfl.mark_pfl_compilable(is_template=True)
def sub(a, b):
    return a - b 

def main():

    CustomStd.warp_specialize((1, 2), add, (5.0, 3.0), [add, sub], [32, 32], [32, 32])

def test_template_fn_spec():
    lib = pfl.parse_func_to_pfl_library(main, backend="custom")
    print(lib.all_compiled.keys())

if __name__ == "__main__":
    test_template_fn_spec()
