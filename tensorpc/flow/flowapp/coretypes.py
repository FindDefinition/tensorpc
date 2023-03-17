import dataclasses
from typing import Any


@dataclasses.dataclass
class TreeDragTarget:
    obj: Any
    tree_id: str
    tab_id: str = ""
    source_comp_uid: str = ""
