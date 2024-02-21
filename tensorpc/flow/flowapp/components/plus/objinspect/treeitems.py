import ast
import sys
import tokenize
from tensorpc.core.funcid import find_toplevel_func_node_by_lineno
from tensorpc.core.tracer import FrameResult, TraceType
from tensorpc.core.tree_id import UniqueTreeIdForTree
from tensorpc.flow.flowapp import appctx
from tensorpc.flow.flowapp.components import mui
from tensorpc.flow.flowapp.reload import reload_object_methods
from tensorpc.flow.jsonlike import (CommonQualNames, ContextMenuData,
                                    IconButtonData, JsonLikeNode, JsonLikeType,
                                    parse_obj_to_jsonlike, TreeItem)
from typing import Any, Callable, Dict, Generic, Hashable, List, Optional, TypeVar, Union, Tuple
from tensorpc.core import inspecttools
from tensorpc.flow.marker import mark_create_preview_layout
from .analysis import ObjectTreeParser, get_tree_context, get_tree_context_noexcept
import humanize
import datetime as dt

_DELTA_SHORTCUTS = [
    ("microseconds", "microsecond", "us"),
    ("milliseconds", "millisecond", "ms"),
]

def _delta_shortcut(string: str):
    for ss, s, sh in _DELTA_SHORTCUTS:
        string = string.replace(ss, sh).replace(s, sh)
    return string

def parse_frame_result_to_trace_item(frame_results: List[FrameResult], use_return_locals: bool = False):
    fr_stack: List[Tuple[FrameResult, TraceTreeItem]] = []
    res: List[TraceTreeItem] = []
    # print([(x.qualname, x.type, x.depth) for x in frame_results])
    for fr in frame_results:
        if fr.type == TraceType.Call:
            item = TraceTreeItem(fr)
            if fr_stack:
                prev = fr_stack[-1][1]
                item.parent_cls_name = prev.cls_name
            fr_stack.append((fr, item))

        elif fr.type == TraceType.Return:
            poped = fr_stack.pop()
            if use_return_locals:
                poped[1].set_return_frame_result(fr)
            else:
                poped[1].end_ts = fr.timestamp
            if len(fr_stack) == 0:
                res.append(poped[1])
            else:
                fr_stack[-1][1].append_child(poped[1])
    return res


class TraceTreeItem(TreeItem):
    def __init__(self, frame_res: FrameResult) -> None:
        super().__init__()
        self.set_return_frame_result(frame_res)
        self.depth = frame_res.depth
        self.call_var_names: List[str] = list(frame_res.local_vars.keys())
        self.start_ts = frame_res.timestamp
        self.end_ts = -1
        self.child_trace_res: List[TraceTreeItem] = []
        self.parent_cls_name: str = ""

    def set_return_frame_result(self, frame_res: FrameResult):
        self.local_vars = inspecttools.filter_local_vars(frame_res.local_vars)
        self.is_method = "self" in self.local_vars
        if self.is_method:
            self.cls_name = type(self.local_vars["self"]).__name__
        else:
            self.cls_name = ""
        self.qname = frame_res.qualname
        self.name = self.qname.split(".")[-1]
        self.filename = frame_res.filename
        self.lineno = frame_res.lineno
        self.module_qname = frame_res.module_qname
        self.end_ts = frame_res.timestamp

    def get_display_name(self):
        # if method, use "self.xxx" instead of full qualname
        if self.is_method and self.parent_cls_name == self.cls_name:
            return f"self.{self.name}"
        else:
            return self.qname

    async def get_child_desps(self, parent_ns: UniqueTreeIdForTree) -> Dict[str, JsonLikeNode]:
        res: Dict[str, JsonLikeNode] = {}
        for v in self.child_trace_res:
            id = parent_ns.append_part(v.get_uid())
            node = v.get_json_like_node(id)
            res[v.get_uid()] = node
        res_list = await get_tree_context_noexcept().parser.parse_obj_dict_to_nodes(self.local_vars, parent_ns)
        res.update({x.name: x for x in res_list})
        return res

    async def get_child(self, key: str) -> Any:
        child_trace_keys = [x.get_uid() for x in self.child_trace_res]
        if key in child_trace_keys:
            return self.child_trace_res[child_trace_keys.index(key)]
        return self.local_vars[key]

    def get_json_like_node(self, id: UniqueTreeIdForTree) -> JsonLikeNode:
        return JsonLikeNode(id,
                            id.parts[-1],
                            JsonLikeType.Object.value,
                            typeStr="Frame",
                            cnt=len(self.local_vars),
                            drag=False,
                            alias=self.get_display_name(),
                            value=self.get_delta_str())

    def append_child(self, item: "TraceTreeItem"):
        self.child_trace_res.append(item)

    def __repr__(self):
        return f"{self.filename}::{self.qname}"

    def get_uid(self):
        return f"{self.filename}:{self.lineno}@{self.qname}"

    def get_delta_str(self):
        delta = 0
        if self.end_ts != -1 and self.start_ts != -1:
            delta = (self.end_ts - self.start_ts) / 1e6
        delta_str = _delta_shortcut(humanize.naturaldelta(dt.timedelta(milliseconds=delta), minimum_unit="milliseconds"))
        if delta == 0:
            delta_str = "undefined"
        return delta_str

    @mark_create_preview_layout
    def preview_layout(self):
        btn = mui.Button("Run Frame", self._on_run_frame)
        reload_btn = mui.Button("Reload Object", self._on_reload_self)
        font = dict(fontFamily="monospace",
                    fontSize="14px",
                    wordBreak="break-word")
        return mui.VBox([
            mui.Typography(f"Frame<{self.depth}>: {self.qname}").prop(**font),
            mui.Typography(f"Path: {self.filename}:{self.lineno}").prop(
                **font),
            mui.Typography(f"Time: {self.get_delta_str()}").prop(
                **font, variant="caption"),
            mui.HBox([btn, reload_btn]),
        ]).prop(flex=1)

    def _get_qname(self):
        if sys.version_info[:2] >= (3, 11):
            return self.qname 
        else:
            # use ast parse 
            with tokenize.open(self.filename) as f:
                data = f.read()
            tree = ast.parse(data)
            res = find_toplevel_func_node_by_lineno(tree, self.lineno)
            if res is None:
                return None 
            if res[0].name != self.name:
                return None 
            ns = ".".join([x.name for x in res[1]])
            return f"{ns}.{res[0]}"

    def _get_static_method(self):
        qname = self._get_qname()
        if qname is None:
            return None 
        module = sys.modules.get(self.module_qname)
        if module is None:
            return None 
        parts = qname.split(".")
        obj = module.__dict__[parts[0]]
        for part in parts[1:]:
            obj = getattr(obj, part)
        return obj

    async def _on_run_frame(self):
        """rerun this function with return trace.
        """
        if "self" not in self.local_vars:
            # try find method via qualname
            method = self._get_static_method()
            if method is None:
                raise ValueError(
                    "self not in local vars, currently only support run frame with self"
                )
            async with appctx.inspector.trace([], f"trace-{self.name}", traced_names=set([self.name]), use_return_locals=True):
                method(**self.local_vars)
        else:
            local_vars = {k: v for k, v in self.local_vars.items()}
            local_vars.pop("self")
            fn = getattr(self.local_vars["self"], self.name)
            async with appctx.inspector.trace([], f"trace-{self.name}", traced_names=set([self.name]), use_return_locals=True):
                fn(**local_vars)

    def _on_reload_self(self):
        if "self" not in self.local_vars:
            raise ValueError(
                "self not in local vars, currently only support reload object with self"
            )
        reload_object_methods(self.local_vars["self"],
                              reload_mgr=appctx.get_reload_manager())
