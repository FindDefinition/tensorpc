import dataclasses
import io
import os
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import Literal

from .constants import ExternalTrace, TracerType

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
            else:
                pid = os.getpid()
                self._trace_instant_events_for_pth.append({
                    "name": name,
                    "args": args,
                    "s": scope,
                    "pid": pid,
                    "tid": pid, # pid == tid in pytorch profiler
                    "ph": "I",
                    "ts": time.time_ns() // 1000, # us
                })

    def start(self):
        if self._tracer is not None:
            if self._tracer_type == TracerType.VIZTRACER:
                self._tracer.start()
            else:
                self._tracer.__enter__()

    def stop(self):
        if self._tracer is not None:
            if self._tracer_type == TracerType.VIZTRACER:
                self._tracer.stop()
            else:
                self._tracer.__exit__(None, None, None)

    def save(self, ss: io.BytesIO, proc_name_for_pth: Optional[str] = None):
        if self._tracer is not None:
            if self._tracer_type == TracerType.VIZTRACER:
                sss = io.StringIO()
                self._tracer.save(sss)
                ss.write(sss.getvalue().encode())
            else:
                fp = tempfile.NamedTemporaryFile("w+t", suffix=".json", delete=False)
                fp.close()
                self._tracer.export_chrome_trace(fp.name)
                with open(fp.name, "rb") as f:
                    data = f.read()
                    if proc_name_for_pth is not None and self._tracer_proc_name is not None:
                        data = data.replace(proc_name_for_pth.encode(), self._tracer_proc_name.encode())
                    ss.write(data)
                # remove temp file
                if self._tracer_proc_name is not None:
                    with open(f"{self._tracer_proc_name}.json", "wb") as f:
                        f.write(data)

                os.remove(fp.name)
