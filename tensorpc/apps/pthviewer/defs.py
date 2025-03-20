
import dataclasses
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import torch

from tensorpc.core.tree_id import UniqueTreeIdForTree

@dataclasses.dataclass
class PytorchNodeMeta:
    op: str
    module_id: Optional[UniqueTreeIdForTree] = None
    # (module_id, module_qname)
    module_stack: Optional[List[Tuple[str, str]]] = None
    module_qname: Optional[str] = None
    output_desps: Optional[Sequence[Any]] = None
    is_merged: bool = False
    ftree_id: Optional[str] = None
    is_io_node: bool = False
    io_name: Optional[str] = None
    stack_trace: Optional[str] = None
    additional_args: Optional[Dict[str, Any]] = None
    op_sig: Optional[str] = None

@dataclasses.dataclass
class EdgeTensorMeta:
    raw: Any  # FakeTensor or SymInt

    def get_memory_size(self) -> int:
        if isinstance(self.raw, torch.Tensor):
            return self.raw.nbytes
        return 0