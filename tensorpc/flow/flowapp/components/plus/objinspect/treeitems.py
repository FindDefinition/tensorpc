from tensorpc.core.tracer import FrameResult
from tensorpc.flow.jsonlike import (CommonQualNames, ContextMenuData,
                                    IconButtonData, JsonLikeNode, JsonLikeType, parse_obj_to_jsonlike, TreeItem)
from typing import Any, Callable, Dict, Generic, Hashable, List, Optional, TypeVar, Union, Tuple
from tensorpc.core import inspecttools
from .analysis import parse_obj_dict

class TraceTreeItem(TreeItem):
    def __init__(self, frame_res: FrameResult) -> None:
        super().__init__()

        self.local_vars = inspecttools.filter_local_vars(frame_res.local_vars)
        self.qname = frame_res.qualname

    async def get_child_desps(self, parent_ns: str) -> Dict[str, JsonLikeNode]:
        res = parse_obj_dict(self.local_vars, parent_ns, lambda x: True) 
        return {x.name: x for x in res}
    async def get_child(self, key: str) -> Any:
        return self.local_vars[key]

    def get_json_like_node(self, id: str) -> Optional[JsonLikeNode]:
        return JsonLikeNode(id,
                            self.qname,
                            JsonLikeType.Object.value,
                            typeStr="TraceRes",
                            cnt=len(self.local_vars),
                            drag=False)
