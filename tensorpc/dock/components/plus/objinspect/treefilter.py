import abc 
import dataclasses
from typing import Any, Optional
from tensorpc.dock.components import mui 

class TreeExpandFilter(abc.ABC):

    def get_all_expand_item_ids(self):
        return [item.id for item in self.get_expand_menu_items()]

    @abc.abstractmethod
    def get_expand_menu_items(self) -> list[mui.MenuItem]: ...

    @abc.abstractmethod
    def is_cared_type(self, obj: Any) -> bool: ...

    @abc.abstractmethod
    def is_leaf(self, obj: Any, menu_item_id: str) -> bool:
        """Check if the object is a leaf node. if so, stop expanding.
        only used when recursive expand is True.
        """
        return True

    @abc.abstractmethod
    def should_keep(self, obj: Any, menu_item_id: str) -> bool:
        """Check if the object should be kept
        """
        return True

    def expand_dict(self, obj: Any) -> Optional[dict]:
        """expand object which is cared type. if None, use default expand.
        """
        return None

@dataclasses.dataclass
class TreeExpandFilterDesc:
    filter: TreeExpandFilter
    is_recursive: bool = False
