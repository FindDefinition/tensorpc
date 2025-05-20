import json
import math

import yaml
from tensorpc.apps.dbg.components.perfmonitor import PerfMonitor
from tensorpc.constants import TENSORPC_DEV_SECRET_PATH
from tensorpc.dock import mui, three, plus, appctx, mark_create_layout
import dataclasses
from typing import Any 
import numpy as np
import tensorpc.core.datamodel as D

class App:
    @mark_create_layout
    def my_layout(self):
        self.monitor = PerfMonitor()
        return mui.VBox([
            mui.Button("Load Trace", self._set_data),
            self.monitor
        ]).prop(minHeight=0,
                minWidth=0,
                width="100%",
                height="100%",
                overflow="hidden")

    async def _set_data(self):
        with open(TENSORPC_DEV_SECRET_PATH, "r") as f:
            path = yaml.safe_load(f)["perfetto_debug"]["trace_path"]
        with open(path, "r") as f:
            trace = json.load(f)
        trace_events = trace["traceEvents"]

        await self.monitor.append_perf_data(0, [trace_events], [None], max_depth=10)