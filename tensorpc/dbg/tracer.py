import contextlib
import copy
import dataclasses
import io
import math
import os
import random
import tempfile
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import Literal

from .constants import TENSORPC_DBG_USER_DURATION_EVENT_KEY, DebugDistributedInfo, TracerSingleResult, TracerType, TracerConfig

try:
    import orjson as json  # type: ignore

    def json_dump_to_bytes(obj: Any) -> bytes:
        # json dump/load is very slow when trace data is large
        # so we use orjson if available
        return json.dumps(obj)
except ImportError:
    import json  # type: ignore

    def json_dump_to_bytes(obj: Any) -> bytes:
        return json.dumps(obj).encode()


class VizTracerAndPytorchTracer:

    def __init__(self, tracer_viz: Any, tracer_pth: Any) -> None:
        self._tracer_viz = tracer_viz
        self._tracer_pth = tracer_pth

    def start(self):
        # pth profiler should start first
        self._tracer_pth.__enter__()
        self._tracer_viz.start()

    def stop(self):
        self._tracer_viz.stop()
        self._tracer_pth.__exit__(None, None, None)


class DebugTracerWrapper:

    def __init__(self) -> None:
        self._tracer: Any = None
        self._tracer_type: TracerType = TracerType.VIZTRACER

        self._tracer_proc_name: Optional[str] = None
        self._trace_cfg: Optional[TracerConfig] = None
        self._trace_dist_meta: Optional[DebugDistributedInfo] = None

        self._trace_events_external: List[Any] = []
        self._trace_tid = None
        self._trace_lock = None

    def set_tracer(self, cfg: Optional[TracerConfig], tracer: Any,
                   tracer_type: TracerType, proc_name: str,
                   meta: DebugDistributedInfo) -> None:
        self._tracer = tracer
        self._tracer_type = tracer_type
        self._tracer_proc_name = proc_name
        self._trace_cfg = cfg
        self._trace_dist_meta = meta
        self._trace_lock = threading.Lock()
        self._trace_tid = threading.get_ident()

    def reset_tracer(self) -> None:
        # TODO if fork during tracing...
        self._tracer = None
        self._tracer_type = TracerType.VIZTRACER
        self._tracer_proc_name = None
        self._trace_events_external = []
        self._trace_cfg = None
        self._trace_dist_meta = None
        self._trace_lock = None
        self._trace_tid = None

    def _get_site_packages_by_profiler_location(self):
        if self._tracer_type == TracerType.VIZTRACER or self._tracer_type == TracerType.VIZTRACER_PYTORCH:
            import viztracer
            return os.path.abspath(
                os.path.dirname(os.path.dirname(viztracer.__file__)))
        elif self._tracer_type == TracerType.PYTORCH:
            import torch
            return os.path.abspath(os.path.dirname(torch.__file__))
        else:
            raise ValueError(f"Invalid tracer type: {self._tracer_type}")

    def log_instant(self,
                    name: str,
                    args: Any = None,
                    scope: str = "p") -> None:
        if self._tracer_type == TracerType.TARGET_TRACER:
            return 

        is_diff_thread = threading.get_ident() != self._trace_tid
        if self._tracer is not None and self._trace_lock is not None:
            if self._tracer_type == TracerType.VIZTRACER and not is_diff_thread:
                self._tracer.log_instant(name, args, scope)
            elif self._tracer_type == TracerType.VIZTRACER_PYTORCH and not is_diff_thread:
                self._tracer._tracer_viz.log_instant(name, args, scope)
            else:
                """breakpoint based trace can't trace already started thread.
                so we need to log instant event manually if the tracer isn't started
                in current thread.
                """
                pid = os.getpid()
                # pid == tid in pytorch profiler
                if self._tracer_type == TracerType.VIZTRACER:
                    ts = time.monotonic_ns() // 1000  # us
                else:
                    ts = time.time_ns() // 1000  # us
                with self._trace_lock:
                    self._trace_events_external.append({
                        "name": name,
                        "args": args,
                        "s": scope,
                        "pid": pid,
                        "tid": pid,
                        "ph": "i",
                        "ts": ts,
                    })
    
    @contextlib.contextmanager
    def log_duration(self,
                    name: str,
                    args: Any = None,
                    thread_id: int = 0,
                    cat: Optional[str] = None):
        if self._tracer_type == TracerType.TARGET_TRACER:
            yield 
            return 

        if self._tracer is not None and self._trace_lock is not None:
            pid = os.getpid()
            # pid == tid in pytorch profiler
            if self._tracer_type == TracerType.VIZTRACER:
                ts = time.monotonic_ns() / 1000  # us
            else:
                ts = time.time_ns() / 1000  # us
            yield 
            if self._tracer_type == TracerType.VIZTRACER:
                ts_end = time.monotonic_ns() / 1000  # us
            else:
                ts_end = time.time_ns() / 1000  # us
            res = {
                "name": name,
                "pid": pid,
                # "tid": pid,
                "tid": thread_id, # TODO pid or 0?
                "ph": "X",
                "ts": ts,
                "dur": ts_end - ts,
                "cat": cat or TENSORPC_DBG_USER_DURATION_EVENT_KEY,
            }
            if args is not None:
                res["args"] = args
            with self._trace_lock:
                self._trace_events_external.append(res)

    def start(self):
        if self._tracer is not None:
            if self._tracer_type == TracerType.VIZTRACER or self._tracer_type == TracerType.VIZTRACER_PYTORCH:
                self._tracer.start()
            elif self._tracer_type == TracerType.PYTORCH:
                self._tracer.__enter__()
            elif self._tracer_type == TracerType.TARGET_TRACER:
                self._tracer.start()
            else:
                raise ValueError(f"Invalid tracer type: {self._tracer_type}")

    def stop(self):
        if self._tracer is not None:
            if self._tracer_type == TracerType.VIZTRACER or self._tracer_type == TracerType.VIZTRACER_PYTORCH:
                self._tracer.stop()
            elif self._tracer_type == TracerType.PYTORCH:
                self._tracer.__exit__(None, None, None)
            elif self._tracer_type == TracerType.TARGET_TRACER:
                self._tracer.stop()
            else:
                raise ValueError(f"Invalid tracer type: {self._tracer_type}")

    def _save_pth(
            self,
            tracer_pth: Any,
            external_events: List[Any],
            proc_name_for_pth: Optional[str] = None,
            extract_base_ts: bool = True,
            suppress_user_anno: bool = False) -> Tuple[bytes, Optional[int]]:
        assert self._trace_lock is not None
        MAX_FLOW_ID_NUM_PAD = 6
        fp = tempfile.NamedTemporaryFile("w+t", suffix=".json", delete=False)
        fp.close()
        tracer_pth.export_chrome_trace(fp.name)
        with open(fp.name, "rb") as f:
            data = f.read()
        os.remove(fp.name)
        # for large json file (> 50MB), load and dump is very slow even with orjson
        # so we use a ugly but fast way to modify events and extract baseTimeNanoseconds
        # pytorch may modify json format so we currently need to check for each pytorch version.
        # checked: pytorch 2.1 and 2.5
        if proc_name_for_pth is not None and self._tracer_proc_name is not None:
            data = data.replace(proc_name_for_pth.encode(),
                                f"{self._tracer_proc_name}".encode())
            if self._trace_dist_meta is not None and self._trace_dist_meta.backend is not None:
                meta = self._trace_dist_meta
                # correct flow event id
                if meta.world_size > 0:
                    digits = int(math.log10(meta.world_size)) + 1
                elif meta.world_size == 0:
                    digits = 1
                else:
                    digits = 1
                # pad zeros to meta.rank
                rank_padded_str = str(
                    meta.rank + 1).zfill(digits) + "0" * MAX_FLOW_ID_NUM_PAD
                if suppress_user_anno:
                    data = data.replace(
                        b'"ph": "X", "cat": "user_annotation"',
                        b'"ph": "M", "cat": "user_annotation"')
                if meta.world_size > 1:
                    # fix duplicated flow event across ranks
                    # TODO pytorch may remove beautiful json dump (remove whitespace) in future
                    data = data.replace(
                        b'"ph": "f", "id": ',
                        f'"ph": "f", "id": {rank_padded_str}'.encode())
                    data = data.replace(
                        b'"ph": "s", "id": ',
                        f'"ph": "s", "id": {rank_padded_str}'.encode())
                    # suppress all pytorch process_name meta events
                    data = data.replace(
                        b'"process_name"',
                        b'"process_name_change_name_to_be_invalid"')
                    # supress all user annotation events when use v+p tracer
                    # append our process name meta event
                    pid = os.getpid()
                    external_events.append({
                        "name": "process_name",
                        "ph": "M",
                        "pid": pid,
                        "tid": 0,
                        "args": {
                            "name": self._tracer_proc_name,
                        },
                    })
                    external_events.append({
                        "name": "process_name",
                        "ph": "M",
                        "pid": meta.rank,
                        "tid": 0,
                        "args": {
                            "name": f"{self._tracer_proc_name}-device",
                        },
                    })

        base_ts: Optional[int] = None
        if extract_base_ts:
            fast_find_segment = data[-2000:]
            key = b"\"baseTimeNanoseconds\":"
            # step 1: do find on last 2000 char to get "baseTimeNanoseconds"
            index = fast_find_segment.find(key)
            if index != -1:
                # step 2: find integer after that
                index += len(key)
                segment_contains_time_ns = fast_find_segment[index:]
                segment_contains_time_ns_str = segment_contains_time_ns.decode(
                )
                is_first_digit_find = False
                digits = []
                for c in segment_contains_time_ns_str:
                    if c.isdigit():
                        is_first_digit_find = True
                    if is_first_digit_find:
                        if c.isdigit():
                            digits.append(c)
                        else:
                            break
                if is_first_digit_find:
                    base_ts = int("".join(map(str, digits)))
            # if base_ts is None:
                # fast check failed, load entire json file
                # pth_trace_dict = json.loads(data)
                # if "baseTimeNanoseconds" in pth_trace_dict:
                #     base_ts = pth_trace_dict["baseTimeNanoseconds"]
        if self._trace_cfg is not None and self._trace_cfg.profile_memory:
            # load pytorch data, filter memory instant events, then construct counter events
            try:
                pth_trace_dict = json.loads(data)
                if "baseTimeNanoseconds" in pth_trace_dict:
                    base_ts = pth_trace_dict["baseTimeNanoseconds"]
            except:
                # in some pytorch version, json is not valid
                traceback.print_exc()
                return data, base_ts
            pid = os.getpid()
            if base_ts is not None:
                base_ts_us = base_ts / 1000.0
            else:
                base_ts_us = 0
            for ev in pth_trace_dict["traceEvents"]:
                if (ev["ph"] == "I" or ev["ph"] == "i") and "args" in ev:
                    if "Total Allocated" in ev["args"]:
                        total_alloc = ev["args"]["Total Allocated"]
                        total_reserv = ev["args"]["Total Reserved"]
                        if ev["args"]["Device Id"] >= 0:
                            external_events.append({
                                "name": "pth_memory",
                                "ph": "C",
                                "pid": pid,
                                "tid": 0,
                                "ts": ev["ts"] + base_ts_us,
                                "args": {
                                    "Total Allocated": total_alloc,
                                    "Total Reserved": total_reserv,
                                },
                            })
        return data, base_ts

    def _filter_viztracer_data_inplace(self, data: Dict[str, Any]):
        if self._trace_cfg is not None:
            if self._trace_cfg.record_filter.exclude_name_prefixes:
                exclude_name_prefixes = self._trace_cfg.record_filter.exclude_name_prefixes
                data["traceEvents"] = [
                    ev for ev in data["traceEvents"]
                    if "name" not in ev or not any(
                        ev["name"].startswith(prefix)
                        for prefix in exclude_name_prefixes)
                ]
            if self._trace_cfg.record_filter.exclude_file_names:
                exclude_file_names = self._trace_cfg.record_filter.exclude_file_names
                data["traceEvents"] = [
                    ev for ev in data["traceEvents"]
                    if "name" not in ev or not any(
                        prefix in ev["name"]
                        for prefix in exclude_file_names)
                ]

    def save(
        self,
        proc_name_for_pth: Optional[str] = None
    ) -> Optional[List[TracerSingleResult]]:
        if self._tracer_type == TracerType.TARGET_TRACER:
            return None 
        if self._tracer is not None and self._trace_lock is not None:
            with self._trace_lock:
                ext_events = copy.deepcopy(self._trace_events_external)
            if self._tracer_type == TracerType.VIZTRACER:
                ss = io.BytesIO()
                sss = io.StringIO()
                self._tracer.parse()
                self._filter_viztracer_data_inplace(self._tracer.data)
                self._tracer.save(sss)
                data = sss.getvalue().encode()
                site_pkg = None
                if self._trace_cfg is not None and self._trace_cfg.replace_sitepkg_prefix:
                    site_pkg = self._get_site_packages_by_profiler_location()
                    data = data.replace(site_pkg.encode(), b"")
                ss.write(data)
                tr_res = TracerSingleResult(
                    data=ss.getvalue(),
                    tracer_type=self._tracer_type,
                    trace_events=self._tracer.data["traceEvents"],
                    site_packages_prefix=site_pkg,
                    external_events=ext_events)
                return [tr_res]
            elif self._tracer_type == TracerType.PYTORCH:
                extract_bts = bool(ext_events)
                data, base_ts = self._save_pth(self._tracer, ext_events, proc_name_for_pth,
                                               extract_bts)
                if ext_events and base_ts is not None:
                    for ev in ext_events:
                        if "ts" in ev:
                            ev["ts"] -= base_ts / 1000.0
                tr_res = TracerSingleResult(data=data,
                                            tracer_type=self._tracer_type,
                                            external_events=ext_events)
                return [tr_res]

            elif self._tracer_type == TracerType.VIZTRACER_PYTORCH:
                # handle pytorch
                data, base_ts = self._save_pth(self._tracer._tracer_pth,
                                               ext_events,
                                               proc_name_for_pth,
                                               True,
                                               suppress_user_anno=True)
                pth_tr_res = TracerSingleResult(data=data,
                                                tracer_type=TracerType.PYTORCH)

                ss = io.BytesIO()
                sss = io.StringIO()
                # align viztracer timestamp from monotonic time to epoch time (or pytorch base time if exists)
                # TODO better align
                if base_ts is not None:
                    mono_pth_diff = time.time_ns() - time.monotonic_ns(
                    ) - base_ts
                else:
                    mono_pth_diff = time.time_ns() - time.monotonic_ns()
                if ext_events and base_ts is not None:
                    # if use PYTORCH or VIZTRACER_PYTORCH, we need to align viztracer timestamp
                    for ev in ext_events:
                        if "ts" in ev:
                            ev["ts"] -= base_ts / 1000.0
                self._tracer._tracer_viz.parse()
                self._filter_viztracer_data_inplace(
                    self._tracer._tracer_viz.data)
                for ev in self._tracer._tracer_viz.data["traceEvents"]:
                    if "ts" in ev:
                        ev["ts"] = (int(ev["ts"] * 1000) +
                                    mono_pth_diff) / 1000.0
                self._tracer._tracer_viz.save(sss)
                vizdata = sss.getvalue().encode()
                site_pkg = None

                if self._trace_cfg is not None and self._trace_cfg.replace_sitepkg_prefix:
                    site_pkg = self._get_site_packages_by_profiler_location()
                    vizdata = vizdata.replace(site_pkg.encode(), b"")
                ss.write(vizdata)
                viz_res = ss.getvalue()
                viz_tr_res = TracerSingleResult(
                    data=viz_res,
                    tracer_type=TracerType.VIZTRACER,
                    site_packages_prefix=site_pkg,
                    trace_events=self._tracer._tracer_viz.data["traceEvents"],
                    external_events=ext_events)
                return [viz_tr_res, pth_tr_res]
            else:
                raise ValueError(f"Invalid tracer type: {self._tracer_type}")
