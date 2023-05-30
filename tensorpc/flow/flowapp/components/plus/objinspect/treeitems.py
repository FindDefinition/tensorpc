from tensorpc.core.tracer import FrameResult, TraceType
from tensorpc.flow.flowapp import appctx
from tensorpc.flow.flowapp.components import mui
from tensorpc.flow.flowapp.reload import reload_object_methods
from tensorpc.flow.jsonlike import (CommonQualNames, ContextMenuData,
                                    IconButtonData, JsonLikeNode, JsonLikeType,
                                    parse_obj_to_jsonlike, TreeItem)
from typing import Any, Callable, Dict, Generic, Hashable, List, Optional, TypeVar, Union, Tuple
from tensorpc.core import inspecttools
from tensorpc.flow.marker import mark_create_preview_layout
from .analysis import parse_obj_dict, GLOBAL_SPLIT, parse_obj_item


def parse_frame_result_to_trace_item(frame_results: List[FrameResult], use_return_locals: bool = False):
    fr_stack: List[Tuple[FrameResult, TraceTreeItem]] = []
    res: List[TraceTreeItem] = []
    # print([(x.qualname, x.type, x.depth) for x in frame_results])
    for fr in frame_results:
        if fr.type == TraceType.Call:
            item = TraceTreeItem(fr)
            fr_stack.append((fr, item))

        elif fr.type == TraceType.Return:
            poped = fr_stack.pop()
            if len(fr_stack) == 0:
                res.append(poped[1])
            else:
                if use_return_locals:
                    poped[1].set_frame_result(fr)
                fr_stack[-1][1].append_child(poped[1])
    return res


class TraceTreeItem(TreeItem):
    def __init__(self, frame_res: FrameResult) -> None:
        super().__init__()
        self.set_frame_result(frame_res)
        self.child_trace_res: List[TraceTreeItem] = []

    def set_frame_result(self, frame_res: FrameResult):
        self.local_vars = inspecttools.filter_local_vars(frame_res.local_vars)
        self.qname = frame_res.qualname
        self.name = self.qname.split(".")[-1]
        self.filename = frame_res.filename
        self.lineno = frame_res.lineno

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
        child_trace_keys = [x.get_uid() for x in self.child_trace_res]
        if key in child_trace_keys:
            return self.child_trace_res[child_trace_keys.index(key)]
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

    @mark_create_preview_layout
    def preview_layout(self):
        btn = mui.Button("Run Frame", self._on_run_frame)
        reload_btn = mui.Button("Reload Object", self._on_reload_self)
        font = dict(font_family="monospace",
                    font_size="14px",
                    word_break="break-word")
        return mui.VBox([
            mui.Typography(f"Frame: {self.qname}").prop(**font),
            mui.Typography(f"Path: {self.filename}:{self.lineno}").prop(
                **font),
            mui.HBox([btn, reload_btn]),
        ]).prop(flex=1)

    def _on_run_frame(self):
        if "self" not in self.local_vars:
            raise ValueError(
                "self not in local vars, currently only support run frame with self"
            )
        getattr(self.local_vars["self"], self.name)(**self.local_vars)

    def _on_reload_self(self):
        if "self" not in self.local_vars:
            raise ValueError(
                "self not in local vars, currently only support reload object with self"
            )
        reload_object_methods(self.local_vars["self"],
                              reload_mgr=appctx.get_reload_manager())
