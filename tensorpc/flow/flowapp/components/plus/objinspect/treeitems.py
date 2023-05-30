from tensorpc.core.tracer import FrameResult, TraceType
from tensorpc.flow.jsonlike import (CommonQualNames, ContextMenuData,
                                    IconButtonData, JsonLikeNode, JsonLikeType, parse_obj_to_jsonlike, TreeItem)
from typing import Any, Callable, Dict, Generic, Hashable, List, Optional, TypeVar, Union, Tuple
from tensorpc.core import inspecttools
from .analysis import parse_obj_dict, GLOBAL_SPLIT, parse_obj_item

def parse_frame_result_to_trace_item(frame_results: List[FrameResult]):
    fr_stack: List[Tuple[FrameResult, TraceTreeItem]] = []
    res: List[TraceTreeItem] = []
    print([x.qualname for x in frame_results])
    for fr in frame_results:
        if fr.type == TraceType.Call:
            item = TraceTreeItem(fr)
            fr_stack.append((fr, item))
        
        elif fr.type == TraceType.Return:
            poped = fr_stack.pop()
            if len(fr_stack) == 0:
                res.append(poped[1])
            else:
                fr_stack[-1][1].append_child(poped[1])
    return res 

class TraceTreeItem(TreeItem):
    def __init__(self, frame_res: FrameResult) -> None:
        super().__init__()
        self.local_vars = inspecttools.filter_local_vars(frame_res.local_vars)
        self.qname = frame_res.qualname
        self.filename = frame_res.filename
        self.lineno = frame_res.lineno
        self.child_trace_res: List[TraceTreeItem] = []

    async def get_child_desps(self, parent_ns: str) -> Dict[str, JsonLikeNode]:
        res: Dict[str, JsonLikeNode] = {}
        for v in self.child_trace_res:
            id = f"{parent_ns}{GLOBAL_SPLIT}{v.get_uid()}"
            node = v.get_json_like_node(id)
            res[v.get_uid()] = node
        res_list = parse_obj_dict(self.local_vars, parent_ns, lambda x: True) 
        res.update({x.name: x for x in res_list})
        return res
    
    async def get_child(self, key: str) -> Any:
        return self.local_vars[key]

    def get_json_like_node(self, id: str) -> JsonLikeNode:
        return JsonLikeNode(id,
                            id.split(GLOBAL_SPLIT)[-1],
                            JsonLikeType.Object.value,
                            typeStr="Frame",
                            cnt=len(self.local_vars),
                            drag=False,
                            alias=self.qname)

    def append_child(self, item: "TraceTreeItem"):
        self.child_trace_res.append(item) 

    def __repr__(self):
        return f"{self.filename}::{self.qname}"
    
    def get_uid(self):
        return f"{self.filename}@{self.qname}"