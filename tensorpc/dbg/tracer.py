import dataclasses
import io
import os
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import Literal

from .constants import TracerType

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

        self._trace_instant_events_for_pth: List[Any] = []

    def set_tracer(self, tracer: Any, tracer_type: TracerType, proc_name: str) -> None:
        self._tracer = tracer
        self._tracer_type = tracer_type
        self._tracer_proc_name = proc_name

    def reset_tracer(self) -> None:
        self._tracer = None
        self._tracer_type = TracerType.VIZTRACER
        self._tracer_proc_name = None
        self._trace_instant_events_for_pth = []

    def log_instant(self, name: str, args: Any = None, scope: str = "p") -> None:
        if self._tracer is not None:
            if self._tracer_type == TracerType.VIZTRACER:
                self._tracer.log_instant(name, args, scope)
            elif self._tracer_type == TracerType.VIZTRACER_PYTORCH:
                self._tracer._tracer_viz.log_instant(name, args, scope)
            # else:
            #     pid = os.getpid()
            #     self._trace_instant_events_for_pth.append({
            #         "name": name,
            #         "args": args,
            #         "s": scope,
            #         "pid": pid,
            #         "tid": pid, # pid == tid in pytorch profiler
            #         "ph": "I",
            #         "ts": time.time_ns() // 1000, # us
            #     })

    def start(self):
        if self._tracer is not None:
            if self._tracer_type == TracerType.VIZTRACER or self._tracer_type == TracerType.VIZTRACER_PYTORCH:
                self._tracer.start()
            elif self._tracer_type == TracerType.PYTORCH:
                self._tracer.__enter__()
            else:
                raise ValueError(f"Invalid tracer type: {self._tracer_type}")

    def stop(self):
        if self._tracer is not None:
            if self._tracer_type == TracerType.VIZTRACER or self._tracer_type == TracerType.VIZTRACER_PYTORCH:
                self._tracer.stop()
            elif self._tracer_type == TracerType.PYTORCH:
                self._tracer.__exit__(None, None, None)
            else:
                raise ValueError(f"Invalid tracer type: {self._tracer_type}")

    def save(self, proc_name_for_pth: Optional[str] = None) -> Optional[List[bytes]]:
        if self._tracer is not None:
            if self._tracer_type == TracerType.VIZTRACER:
                ss = io.BytesIO()
                sss = io.StringIO()
                self._tracer.save(sss)
                ss.write(sss.getvalue().encode())
                return [ss.getvalue()]
            elif self._tracer_type == TracerType.PYTORCH:
                fp = tempfile.NamedTemporaryFile("w+t", suffix=".json", delete=False)
                fp.close()
                self._tracer.export_chrome_trace(fp.name)
                with open(fp.name, "rb") as f:
                    data = f.read()
                    if proc_name_for_pth is not None and self._tracer_proc_name is not None:
                        data = data.replace(proc_name_for_pth.encode(), self._tracer_proc_name.encode())
                # remove temp file
                os.remove(fp.name)
                return [data]

            elif self._tracer_type == TracerType.VIZTRACER_PYTORCH:
                # handle pytorch
                fp = tempfile.NamedTemporaryFile("w+t", suffix=".json", delete=False)
                fp.close()
                self._tracer._tracer_pth.export_chrome_trace(fp.name)
                with open(fp.name, "rb") as f:
                    data = f.read()
                os.remove(fp.name)
                if proc_name_for_pth is not None and self._tracer_proc_name is not None:
                    data = data.replace(proc_name_for_pth.encode(), self._tracer_proc_name.encode())
                fast_find_segment = data[-2000:]
                key = b"\"baseTimeNanoseconds\":"
                base_ts: Optional[int] = None
                # step 1: do find on last 2000 char to get "baseTimeNanoseconds"
                index = fast_find_segment.find(key)
                if index != -1:
                    # step 2: find integer after that
                    index += len(key)
                    segment_contains_time_ns = fast_find_segment[index:]
                    segment_contains_time_ns_str = segment_contains_time_ns.decode()
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
                if base_ts is None:
                    # fast check failed, load entire json file
                    pth_trace_dict = json.loads(data)
                    if "baseTimeNanoseconds" in pth_trace_dict:
                        base_ts = pth_trace_dict["baseTimeNanoseconds"]
                ss = io.BytesIO()
                sss = io.StringIO()
                # align viztracer timestamp from monotonic time to epoch time (or pytorch base time if exists)
                # TODO better align
                if base_ts is not None:
                    mono_pth_diff = time.time_ns() - time.monotonic_ns() - base_ts
                else:
                    mono_pth_diff = time.time_ns() - time.monotonic_ns()

                self._tracer._tracer_viz.parse()
                for ev in self._tracer._tracer_viz.data["traceEvents"]:
                    if "ts" in ev:
                        ev["ts"] = (int(ev["ts"] * 1000) + mono_pth_diff) / 1000.0
                self._tracer._tracer_viz.save(sss)
                ss.write(sss.getvalue().encode())
                viz_res = ss.getvalue()
                return [viz_res, data]
            else:
                raise ValueError(f"Invalid tracer type: {self._tracer_type}")