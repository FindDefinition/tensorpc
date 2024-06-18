import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Type
from .compute import ComputeNode, get_compute_flow_context

from tensorpc.flow.components import flowui, mui
from .media import customnode_base

USER_CUSTOM_NODE_CLASS_NAME = "CustomNode"

class CustomNode(ComputeNode):
    def init_node(self):
        base_code_path = Path(customnode_base.__file__).resolve()
        with open(base_code_path, "r") as f:
            base_code = f.read()
        self._code_editor = mui.MonacoEditor(base_code, "python", "test")

        self._cnode = self._get_cnode_cls_from_code(base_code)
        
    def _get_cnode_cls_from_code(self, code: str):
        mod_dict = {}
        code_comp = compile(code, "test", "exec")
        exec(code_comp, mod_dict)
        cnode_cls: Type[ComputeNode] = mod_dict[USER_CUSTOM_NODE_CLASS_NAME]
        cnode = cnode_cls(self.id, self.name, self._node_type, self._init_pos)
        return cnode 

    @property 
    def is_async_gen(self):
        return inspect.isasyncgenfunction(self._cnode.compute)

    def get_side_layout(self) -> Optional[mui.FlexBox]:
        return self._cnode.get_side_layout()

    def get_node_layout(self) -> Optional[mui.FlexBox]:
        return self._cnode.get_node_layout()

    def state_dict(self) -> Dict[str, Any]:
        return self._cnode.state_dict()

    def get_compute_annotation(self):
        return self._cnode.get_compute_annotation()
