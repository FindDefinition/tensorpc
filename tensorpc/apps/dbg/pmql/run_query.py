from typing import Any
from .parser import ModuleStackQuery, SingleQuery, ModuleVariableQuery

def _convert_pth_module_to_pytree(module: Any):
    from torch.nn import Module, ModuleList, ModuleDict