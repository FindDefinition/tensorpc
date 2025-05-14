import math
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, Union
from tensorpc.dock.components import chart, mui
from tensorpc.dock.jsonlike import JsonLikeNode, JsonLikeType, as_dict_no_undefined
import re

def build_depth_from_trace_events(
    trace_events: List[Dict[str, Any]]):
    duration_events: List[Dict[str, Any]] = []
    cnt = 0
    tid_cnt_map = {}
    for event in trace_events:
        if event["tid"] not in tid_cnt_map:
            tid_cnt_map[event["tid"]] = 0
        tid_cnt_map[event["tid"]] += 1

    max_tid = -1

    max_tid_cnt = 0
    for tid, cnt in tid_cnt_map.items():
        if cnt > max_tid_cnt:
            max_tid = tid
            max_tid_cnt = cnt
    assert max_tid_cnt != -1

    for event in trace_events:
        ph = event["ph"]
        if max_tid == event["tid"]:
            if ph == "X":
                duration_events.append(event)
    assert len(duration_events) > 0, f"No duration events found, {len(trace_events)}, center event: {trace_events[len(trace_events) // 2]}"
    duration_events.sort(key=lambda x: x["ts"])
    min_ts = math.inf
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
    for event in duration_events:
        while stack and stack[-1]["ts"] + stack[-1]["dur"] <= event["ts"]:
            stack.pop()
        node = {
            "children": [],
            "ts": event["ts"],
            "dur": event["dur"],
            "depth": len(stack),
        }
        event["depth"] = len(stack)
        if stack:
            stack[-1]["children"].append(node)
        stack.append(node)
    return duration_events

def parse_viztracer_trace_events_to_raw_tree(
    trace_events: List[Dict[str, Any]],
    modify_events_func: Optional[Callable] = None,
    add_depth_to_event: bool = False,
    parse_viztracer_name: bool = True,
) -> Tuple[dict, List[Dict[str, Any]], mui.JsonLikeTreeFieldMap]:
    viz_pattern = re.compile(r"(.*)\((.*):([0-9]*)\)")
    duration_events: List[Dict[str, Any]] = []
    cnt = 0
    tid_cnt_map = {}
    for event in trace_events:
        if event["tid"] not in tid_cnt_map:
            tid_cnt_map[event["tid"]] = 0
        tid_cnt_map[event["tid"]] += 1
    max_tid = -1
    max_tid_cnt = 0
    for tid, cnt in tid_cnt_map.items():
        if cnt > max_tid_cnt:
            max_tid = tid
            max_tid_cnt = cnt
    assert max_tid_cnt != -1
    for event in trace_events:
        ph = event["ph"]
        if max_tid == event["tid"]:
            if ph == "X":
                # only care about main thread and duration events
                if parse_viztracer_name:
                    try:
                        m = viz_pattern.match(event["name"])
                        if m is not None:
                            func_qname = m.group(1).strip()
                            file_name = m.group(2)
                            lineno = int(m.group(3))
                            data = {
                                "id": cnt,
                                "name": func_qname,
                                "fname": file_name,
                                "lineno": lineno,
                                "ts": event["ts"],
                                "dur": event["dur"],
                            }
                            duration_events.append(data)
                            if "args" in event:
                                data["args"] = event["args"]
                            cnt += 1
                    except Exception:
                        continue
                else:
                    data = {
                        "id": cnt,
                        "name": event["name"],
                        "fname": "unknown",
                        "lineno": -1,
                        "ts": event["ts"],
                        "dur": event["dur"],
                    }
                    if "args" in event:
                        data["args"] = event["args"]

                    duration_events.append(data)
                    cnt += 1
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

    assert len(duration_events) > 0, f"No duration events found, {len(trace_events)}, center event: {trace_events[len(trace_events) // 2]}"
    duration_events.sort(key=lambda x: x["ts"])
    if modify_events_func is not None:
        modify_events_func(duration_events)
    min_ts = math.inf
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
            "depth": len(stack),
        }
        if add_depth_to_event:
            event["depth"] = len(stack)
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
