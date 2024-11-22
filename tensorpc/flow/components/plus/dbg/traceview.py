import ast
import asyncio
import dataclasses
import datetime
import enum
import gzip
import io
import json
import time
import traceback
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, Union
import zipfile

import grpc
import psutil
import rich
import yaml
import os

from tensorpc.compat import InWindows
from tensorpc.constants import TENSORPC_BG_PROCESS_NAME_PREFIX
from tensorpc.core.asyncclient import (simple_chunk_call_async,
                                       simple_remote_call_async)
from tensorpc.core.client import simple_remote_call
from tensorpc.core.funcid import find_toplevel_func_node_container_by_lineno
from tensorpc.core.tree_id import UniqueTreeIdForTree
from tensorpc.dbg.constants import (
    TENSORPC_DBG_FRAME_INSPECTOR_KEY, TENSORPC_DBG_SPLIT, DebugDistributedInfo,
    DebugFrameInfo, DebugInfo, RecordFilterConfig, RecordMode,
    RemoteDebugEventType, RemoteDebugTargetTrace, TracerConfig, TraceResult,
    TracerSingleResult, TracerType, TracerUIConfig)
from tensorpc.dbg.serv_names import serv_names as dbg_serv_names
from tensorpc.flow import appctx, marker
from tensorpc.flow.components import chart, mui
from tensorpc.flow.components.plus.config import ConfigDialogEvent, ConfigPanelDialog, ConfigPanelDialogPersist
from tensorpc.flow.components.plus.objview.preview import ObjectPreview
from tensorpc.flow.components.plus.styles import (CodeStyles,
                                                  get_tight_icon_tab_theme)
from tensorpc.flow.core.appcore import AppSpecialEventType
from tensorpc.flow.jsonlike import JsonLikeNode, JsonLikeType, as_dict_no_undefined
from tensorpc.flow.vscode.coretypes import (VscodeBreakpoint,
                                            VscodeTensorpcMessage,
                                            VscodeTensorpcMessageType)
import re
import numpy as np


def _get_site_pkg_prefix():
    return os.path.abspath(os.path.dirname(os.path.dirname(np.__file__)))


@dataclasses.dataclass
class VariableTraceCfg:
    code: str
    is_distributed: bool
    max_num_variable: int = 1


def parse_viztracer_trace_events_to_raw_tree(
    trace_events: List[Dict[str, Any]],
    modify_events_func: Optional[Callable] = None
) -> Tuple[dict, List[Dict[str, Any]], mui.JsonLikeTreeFieldMap]:
    viz_pattern = re.compile(r"(.*)\((.*):([0-9]*)\)")
    duration_events: List[Dict[str, Any]] = []
    cnt = 0
    for event in trace_events:
        ph = event["ph"]
        if event["pid"] == event["tid"]:
            if ph == "X":
                # only care about main thread and duration events
                try:
                    m = viz_pattern.match(event["name"])
                    if m is not None:
                        func_qname = m.group(1).strip()
                        file_name = m.group(2)
                        lineno = int(m.group(3))
                        duration_events.append({
                            "id": cnt,
                            "name": func_qname,
                            "fname": file_name,
                            "lineno": lineno,
                            "ts": event["ts"],
                            "dur": event["dur"],
                        })
                        cnt += 1
                except Exception:
                    continue
            if ph == "i" or ph == "I":
                if "args" in event:
                    args = event["args"]
                    if isinstance(
                            args,
                            dict) and "path" in args and "lineno" in args:
                        data = {
                            "id": cnt,
                            "name": "DebugEventI",
                            "fname": args["path"],
                            "lineno": args["lineno"],
                            "ts": event["ts"],
                            "dur": 0,
                        }
                        if "msg" in args:
                            value = args["msg"]
                            data["value"] = value
                        duration_events.append(data)
                        cnt += 1

    assert len(duration_events) > 0, "No duration events found"
    duration_events.sort(key=lambda x: x["ts"])
    if modify_events_func is not None:
        modify_events_func(duration_events)
    min_ts = 0
    max_ts = 0
    for event in duration_events:
        min_ts = min(min_ts, event["ts"])
        max_ts = max(max_ts, event["ts"] + event["dur"])
    root = {
        "children": [],
        "ts": min_ts - 1e6,
        "dur": max_ts + 1e6,
    }
    stack = [root]
    path_map: Dict[str, str] = {}
    name_map: Dict[str, str] = {}
    type_str_map: Dict[str, str] = {}
    obj_type_node = JsonLikeType.Object.value
    instant_type_node = JsonLikeType.Constant.value
    for event in duration_events:
        while stack and stack[-1]["ts"] + stack[-1]["dur"] <= event["ts"]:
            stack.pop()
        parts = event["name"].split(".")
        type_name = ".".join(parts[:-1])
        # we use raw json like tree here.
        value = event["fname"]
        type_node = obj_type_node
        if "value" in event:
            value = event["value"]
            type_node = instant_type_node
        if value not in path_map:
            value_id = str(len(path_map))
            path_map[value] = value_id
        else:
            value_id = path_map[value]
        if parts[-1] not in name_map:
            name_id = str(len(name_map))
            name_map[parts[-1]] = name_id
        else:
            name_id = name_map[parts[-1]]
        if type_name not in type_str_map:
            type_id = str(len(type_str_map))
            type_str_map[type_name] = type_id
        else:
            type_id = type_str_map[type_name]
        node = {
            "id": str(event["id"]),
            "name": name_id,
            "value": value_id,
            "type": type_node,
            "typeStr": type_id,
            "children": [],
            "ts": event["ts"],
            "dur": event["dur"],
        }
        if stack:
            stack[-1]["children"].append(node)
        stack.append(node)
    name_map = {v: k for k, v in name_map.items()}
    type_str_map = {v: k for k, v in type_str_map.items()}
    return stack[0], duration_events, mui.JsonLikeTreeFieldMap(
        name_map, type_str_map, {
            v: k
            for k, v in path_map.items()
        })


@dataclasses.dataclass
class TraceState:
    id_to_duration_events: Dict[str, dict]
    min_ts: float
    max_ts: float
    has_trace_data: bool = False


class EditorActionType(enum.Enum):
    TraceVar = "Trace Variable"
    TraceVarDialog = "Trace Variable Advanced"
    TraceVarAllRank = "Trace Variable All Rank"


class TraceView(mui.FlexBox):

    def __init__(self, dist_info: DebugDistributedInfo):
        self._dist_meta = dist_info
        self._obj_preview = ObjectPreview(enable_reload=False)
        self.tree = mui.RawTanstackJsonLikeTree()
        # use globalFilterContiguousOnly to match whole word only.
        # we usually don't need too much fuzzy search, usually copy-paste class name or file name
        # from other places.
        self.tree.prop(
            ignoreRoot=True,
            expansionIconTrigger=True,
            fixedSize=True,
            filterFromLeafRows=True,
            filterNodeValue=True,
            rowFilterMatchProps=mui.FlexBoxProps(backgroundColor="beige"),
            globalFilterContiguousOnly=True)
        self.tree.props.tree = mui.JsonLikeNode.create_dummy_dict_binary()
        filter_inp = mui.Input("filter").prop(
            valueChangeTarget=(self.tree, "globalFilter"),
            debounce=500,
            value=mui.undefined)
        # update filter prop in backend only because we use `valueChangeTarget` to change
        # globalFilter directly. we modify backend value to keep state when remount.
        filter_inp.event_change.on(
            lambda val: self.tree.prop(globalFilter=val))
        self._editor = mui.MonacoEditor("", "python", "")
        self._editor.prop(readOnly=True, minHeight=0, minWidth=0)
        self._editor.prop(actions=[
            mui.MonacoEditorAction(
                id=EditorActionType.TraceVar.name,
                label=EditorActionType.TraceVar.value,
                contextMenuOrder=1.5,
                contextMenuGroupId="tensorpc-traceview-action",
            ),
            mui.MonacoEditorAction(
                id=EditorActionType.TraceVarDialog.name,
                label=EditorActionType.TraceVarDialog.value,
                contextMenuOrder=1.5,
                contextMenuGroupId="tensorpc-traceview-action",
            ),
            mui.MonacoEditorAction(
                id=EditorActionType.TraceVarAllRank.name,
                label=EditorActionType.TraceVarAllRank.value,
                contextMenuOrder=1.5,
                contextMenuGroupId="tensorpc-traceview-action",
            ),

        ])
        self._editor.event_editor_action.on(self._handle_editor_action)

        self._code_header = mui.Typography().prop(variant="caption",
                                                  paddingLeft="10px")
        self._perfetto = chart.Perfetto()
        child = mui.Allotment([
            mui.Allotment([
                mui.VBox([
                    filter_inp,
                    self.tree,
                ]).prop(width="100%", height="100%", overflow="hidden"),
                self._obj_preview,
            ]).prop(overflow="hidden", defaultSizes=[2, 1], vertical=True),
            # mui.VBox([
            #     filter_inp,
            #     self.tree,
            # ]).prop(width="100%", height="100%", overflow="hidden"),
            mui.VBox([
                self._perfetto.prop(flex=1),
                self._code_header,
                self._editor.prop(flex=1),
            ]).prop(width="100%", height="100%", alignItems="stretch"),
        ]).prop(defaultSizes=[10, 24], width="100%", height="100%")
        self._variable_trace_cfg = VariableTraceCfg("", False, 1)

        self._variable_trace_cfg_dialog = ConfigPanelDialogPersist(
            self._variable_trace_cfg, self._on_variable_trace_dialog)
        super().__init__([child, self._variable_trace_cfg_dialog])
        self.prop(flexFlow="row nowrap", width="100%", height="100%")

        self.tree.event_select.on(self._tree_item_select)
        self._state: Optional[TraceState] = None

    async def _tree_item_select(self, selected: Dict[str, bool]):
        if not selected:
            return
        uid = list(selected.keys())[0]
        uid_int = int(uid)
        if self._state is not None:
            ev = self._state.id_to_duration_events[str(uid_int)]
            path = ev["original_fname"]
            lineno = ev["lineno"]
            assert os.path.exists(path)
            with open(path, "r") as f:
                code = f.read()
            write_ev = self._code_header.update_event(value=f"{path}:{lineno}")
            await self.send_and_wait(
                self._editor.update_event(value=code, path=path) + write_ev)
            await self._editor.set_line_number(lineno)
            if self._state.has_trace_data:
                duration = self._state.max_ts - self._state.min_ts
                margin = ev["dur"] * 1.0
                if ev["dur"] == 0:
                    margin = duration * 0.05
                start_ts = max(ev["ts"] - margin, self._state.min_ts)
                end_ts = min(ev["ts"] + ev["dur"] + margin, self._state.max_ts)
                start_ts_second = start_ts / 1e6
                end_ts_second = end_ts / 1e6
                await self._perfetto.scroll_to_range(start_ts_second,
                                                     end_ts_second, 1.0)

    def _modify_trace_events(self, events):
        # remove prefix of fname
        path_remove_prefix = _get_site_pkg_prefix()
        if path_remove_prefix is not None:
            for event in events:
                event["original_fname"] = event["fname"]
                event["fname"] = event["fname"].replace(path_remove_prefix, "")

    async def set_trace_events(self, tracer_result: TracerSingleResult):
        events = tracer_result.trace_events
        if events is not None:
            root, duration_events, fieldmap = parse_viztracer_trace_events_to_raw_tree(
                events, self._modify_trace_events)
            id_to_ev = {str(ev["id"]): ev for ev in duration_events}
            min_ts = float('inf')
            max_ts = 0
            for event in duration_events:
                min_ts = min(min_ts, event["ts"])
                max_ts = max(max_ts, event["ts"] + event["dur"])

            self._state = TraceState(id_to_ev,
                                     min_ts,
                                     max_ts,
                                     has_trace_data=True)
            editor_ev = self._editor.update_event(value="")
            selected_state = {
                k: True
                for k in self.tree.get_all_expandable_node_ids(
                    root["children"])
            }
            editor_ev += self.tree.update_event(tree=json.dumps(root).encode(),
                                                expanded=selected_state,
                                                fieldMap=fieldmap)
            await self.send_and_wait(editor_ev)
            zip_ss = io.BytesIO()
            with zipfile.ZipFile(zip_ss,
                                 mode="w",
                                 compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=9) as zf:
                zf.writestr(f"main.json", tracer_result.data)
                if tracer_result.external_events:
                    zf.writestr(
                        f"extra.json",
                        json.dumps(
                            {"traceEvents": tracer_result.external_events}))
            res = zip_ss.getvalue()
            await self._perfetto.set_trace_data(res, "trace")

    async def _run_variable_trace(self, code: str, lineno: int,
                                  is_distributed: bool,
                                  max_num_variable: int = 1):
        if not isinstance(self._editor.props.value,
                          mui.Undefined) and not isinstance(
                              self._editor.props.path, mui.Undefined):
            # parse code via ast to avoid early error
            ast.parse(code)
            tree = ast.parse(self._editor.props.value)
            nodes = find_toplevel_func_node_container_by_lineno(tree, lineno)
            if nodes is not None:
                node_qname = ".".join([n.name for n in nodes])
                trace_ev = RemoteDebugTargetTrace(
                    RemoteDebugEventType.DIST_TARGET_VARIABLE_TRACE,
                    self._dist_meta, self._editor.props.path, node_qname, code,
                    is_distributed, max_num_variable)
                await self.put_app_event(
                    self.create_remote_comp_event(
                        RemoteDebugEventType.DIST_TARGET_VARIABLE_TRACE.value,
                        trace_ev))

    async def _on_variable_trace_dialog(
            self, cfg_ev: ConfigDialogEvent[VariableTraceCfg]):
        cfg = cfg_ev.cfg
        assert cfg_ev.userdata is not None
        await self._run_variable_trace(cfg.code, cfg_ev.userdata,
                                       cfg.is_distributed, cfg.max_num_variable)

    async def set_variable_trace_result(self, var_name: str, var_obj: Any):
        await self._obj_preview.set_obj_preview_layout(var_obj,
                                                       uid=var_name,
                                                       header=var_name)

    async def _handle_editor_action(self, act_ev: mui.MonacoEditorActionEvent):
        action = act_ev.action
        if action == EditorActionType.TraceVar.name or action == EditorActionType.TraceVarAllRank.name:
            is_dist = action == EditorActionType.TraceVarAllRank.name
            if is_dist:
                assert self._dist_meta.backend == "pytorch", "Only support torch backend to all gather variable if all rank"
            sel = act_ev.selection
            if sel is not None:
                lineno = sel.selections[0].startLineNumber
                code = sel.selectedCode
                if code.strip() != "":
                    await self._run_variable_trace(code, lineno, is_dist)

        if action == EditorActionType.TraceVarDialog.name:
            sel = act_ev.selection
            if sel is not None:
                lineno = sel.selections[0].startLineNumber
                code = sel.selectedCode
                if code.strip() != "":
                    self._variable_trace_cfg.code = code
                    await self._variable_trace_cfg_dialog.open_config_dialog(
                        userdata=lineno)
