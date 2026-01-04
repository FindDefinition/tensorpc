"""ADV Flow Serialization Format

from tensorpc.apps.adv import api as ADV
import dataclasses



# ------ ADV Flow Meta Region ------

ADV.mark_flow_meta(node_id="some_node_id...", position=[0, 0])

# ------ ADV Global Script Region ------

ADV.mark_global_script("some_node_id...", [0, 100])
import numpy as np 
...

ADV.mark_global_script("other_node_id...", [0, 300])
import numpy as np 

ADV.mark_global_end()

# ------ ADV Symnol Dependency Region ------
ADV.mark_symbol_dep()

from torch import Tensor
...

ADV.mark_symbol_dep_end()

# ------ ADV Ref Nodes Dependency Region (Optional) ------
ADV.mark_ref_node_dep()

from ... import Node1_main_function

ADV.mark_ref_node_dep_end()
# ------ ADV Nested Flow (fragment) Import Region ------
ADV.mark_nested_flow_dep()

from .nested_flow1 import nested_flow1

ADV.mark_nested_flow_dep_end()

# ------ ADV Symbol Def Region ------

@ADV.mark_symbol_def(node_id="some_node_id...", position=[100, 100])
@dataclasses.dataclass
class SymbolGroup1:
    a: int 
    b: float

# ------ ADV Fragment Region ------

@ADV.mark_fragment(node_id="some_node_id...", position=[100, 100])
def fragment_function(a: int, b: float) -> dict[str, Any]:
    ADV.mark_outputs(...)
    ...

    return {
        "output1": ...,
    }


# ------ ADV Inline Flow Region ------

@ADV.mark_inline_flow
def inline_flow1(...):
    ...

"""


from tensorpc.apps.adv.codemgr.flow import FlowParseResult


