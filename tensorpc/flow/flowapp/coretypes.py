import dataclasses
from typing import Any, Optional, Callable 

from typing_extensions import ContextManager


@dataclasses.dataclass
class TreeDragTarget:
    obj: Any
    tree_id: str
    tab_id: str = ""
    source_comp_uid: str = ""

    context_creator: Optional[Callable[[], ContextManager]] = None
