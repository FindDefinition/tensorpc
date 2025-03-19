from typing import Any, Dict, Optional, TypedDict, List, Tuple
from typing_extensions import NotRequired # for TypedDict
import dataclasses
from tensorpc.flow import flowplus, flowui, mui, plus, three, appctx 
from tensorpc.flow.components.flowplus.nodes.cnode.registry import register_compute_node


class OutputDict(TypedDict):
    output: Any


@register_compute_node(name="My Custom Node")
def my_custom_node(a: int, b: int) -> OutputDict:
    return {'output': a + b}