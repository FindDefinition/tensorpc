import enum
from ..treefilter import TreeExpandFilter
from tensorpc.dock.components import mui 
from typing import Any, Optional 
import sys 

class PthExpandActions(enum.Enum):
    WEIGHTS = "Expand Weights"

class PthModuleExpandFilter(TreeExpandFilter):
    """
    Filter to expand PyTorch module objects in the tree view.
    """

    def get_expand_menu_items(self) -> list[mui.MenuItem]:
        return [
            mui.MenuItem(
                id=PthExpandActions.WEIGHTS.value,
                label=PthExpandActions.WEIGHTS.value,
            ),
        ]

    def is_cared_type(self, obj: Any) -> bool:
        if sys.modules.get("torch") is not None:
            import torch
            if isinstance(obj, torch.nn.Module):
                return True
        return False

    def is_leaf(self, obj: Any, menu_item_id: str):
        """Check if the object is a leaf node. if so, stop expanding.
        only used when recursive expand is True.
        """
        if menu_item_id == PthExpandActions.WEIGHTS.value:
            if sys.modules.get("torch") is not None:
                import torch
                if isinstance(obj, torch.nn.Module):
                    return False 
        return True

    def should_keep(self, obj: Any, menu_item_id: str):
        """Check if the object should be kept
        """
        if menu_item_id == PthExpandActions.WEIGHTS.value:
            if sys.modules.get("torch") is not None:
                import torch
                if isinstance(obj, torch.nn.Module):
                    all_params = list(obj.parameters())
                    return len(all_params) > 0
                if isinstance(obj, (torch.nn.Module, torch.Tensor)):
                    return True 
        return False 

    def expand_dict(self, obj: Any) -> Optional[dict]:
        """expand object which is cared type. if None, use default expand.
        """
        if sys.modules.get("torch") is not None:
            import torch
            if isinstance(obj, (torch.nn.ModuleList)):
                return {str(i): v for i, v in enumerate(obj)}
            elif isinstance(obj, torch.nn.ModuleDict):
                return {k: v for k, v in obj.items()}
