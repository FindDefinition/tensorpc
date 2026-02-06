from typing import Optional
from tensorpc.core.moduleid import get_module_id_of_type, get_qualname_of_type
import tensorpc.core.pfl.backends.js as stl 
from tensorpc.core.annolib import Undefined, undefined

from tensorpc.core import pfl


@pfl.mark_pfl_compilable
def _pfl_rm_0(a: int, b: Optional[int]):
    if b is not None:
        return a + b
    else:
        return a + 3

def test_pfl_remove_optional():
    lib = pfl.parse_func_to_pfl_library(_pfl_rm_0, parse_cfg=pfl.PFLParseConfig(allow_kw=True, allow_remove_optional_based_on_cond=True))

if __name__ == "__main__":
    test_pfl_remove_optional()

