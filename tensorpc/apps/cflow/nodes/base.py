from tensorpc.dock.components import flowui, mui
from tensorpc.apps.cflow.model import ComputeNodeType
from typing import Any, Optional, TypedDict, Union

class BaseNodeWrapper(mui.FlexBox):

    def __init__(self,
                 node_id: str,
                 node_type: ComputeNodeType,
                 children: Optional[mui.LayoutType] = None):
        super().__init__(children)
        self._node_type = node_type
        self._node_id = node_id
