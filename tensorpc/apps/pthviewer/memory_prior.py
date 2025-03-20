

from tensorpc.utils.registry import HashableRegistry
from typing import Callable, Dict, Any, List, Optional, Tuple
import torch 
from .defs import EdgeTensorMeta

REGISTRY: HashableRegistry[Callable[[List[EdgeTensorMeta], List[EdgeTensorMeta], Optional[torch.nn.Module]], Tuple[int, int]]] = HashableRegistry()



@REGISTRY.register_with_key(key="aten::linear")
def _get_linear_train_memory(inputs: List[EdgeTensorMeta], outputs: List[EdgeTensorMeta], module: Optional[torch.nn.Module] = None) -> Tuple[int, int]:
    return outputs[0].get_memory_size(), inputs[0].get_memory_size() + outputs[0].get_memory_size()

@REGISTRY.register_with_key(key="aten::conv1d")
@REGISTRY.register_with_key(key="aten::conv3d")
@REGISTRY.register_with_key(key="aten::conv2d")
def _get_conv_train_memory(inputs: List[EdgeTensorMeta], outputs: List[EdgeTensorMeta], module: Optional[torch.nn.Module] = None) -> Tuple[int, int]:
    return outputs[0].get_memory_size(), inputs[0].get_memory_size() + outputs[0].get_memory_size()