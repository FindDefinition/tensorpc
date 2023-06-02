import ast
import sys
import tokenize
from tensorpc.core.funcid import find_toplevel_func_node_by_lineno
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
            if use_return_locals:
                poped[1].set_frame_result(fr)
            if len(fr_stack) == 0:
                res.append(poped[1])
            else:
                fr_stack[-1][1].append_child(poped[1])
    return res


class TraceTreeItem(TreeItem):
    def __init__(self, frame_res: FrameResult) -> None:
        super().__init__()
        self.set_frame_result(frame_res)
        self.call_var_names: List[str] = list(frame_res.local_vars.keys())

        self.child_trace_res: List[TraceTreeItem] = []

    def set_frame_result(self, frame_res: FrameResult):
        self.local_vars = inspecttools.filter_local_vars(frame_res.local_vars)
        self.is_method = "self" in self.local_vars
        self.qname = frame_res.qualname
        self.name = self.qname.split(".")[-1]
        self.filename = frame_res.filename
        self.lineno = frame_res.lineno
        self.module_qname = frame_res.module_qname

    def get_display_name(self):
        # if method, use "self.xxx" instead of full qualname
        if self.is_method:
            return f"self.{self.name}"
        else:
            return self.qname

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
                            alias=self.get_display_name())

    def append_child(self, item: "TraceTreeItem"):
        self.child_trace_res.append(item)

    def __repr__(self):
        return f"{self.filename}::{self.qname}"

    def get_uid(self):
        return f"{self.filename}:{self.lineno}@{self.qname}"

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
            async with appctx.trace(f"trace-{self.name}", traced_names=set([self.name]), use_return_locals=True):
                method(**self.local_vars)
        else:
            local_vars = {k: v for k, v in self.local_vars.items()}
            local_vars.pop("self")
            fn = getattr(self.local_vars["self"], self.name)
            async with appctx.trace(f"trace-{self.name}", traced_names=set([self.name]), use_return_locals=True):
                fn(**local_vars)

    def _on_reload_self(self):
        if "self" not in self.local_vars:
            raise ValueError(
                "self not in local vars, currently only support reload object with self"
            )
        reload_object_methods(self.local_vars["self"],
                              reload_mgr=appctx.get_reload_manager())