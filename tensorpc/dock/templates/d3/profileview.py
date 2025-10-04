import copy
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
        self.monitor = PerfMonitor(use_view=True)
        
        # self.monitor2 = mui.HBox([mui.Markdown("## PerfMonitor"),])
        # self.monitor2 = PerfMonitor(use_view=True)

        return mui.VBox([
            mui.Button("Load Trace", self._set_data),
            three.ViewCanvas([
                self.monitor.prop(flex=1),
                # self.monitor2.prop(flex=1),

            ]).prop(display="flex",
                flexDirection="column", width="100%", height="100%", overflow="hidden"),
            
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
        print(type(trace))
        trace_events = trace["traceEvents"]
        trace_events2 = copy.deepcopy(trace_events)
        print(len(trace_events))
        await self.monitor.append_perf_data(0, [trace_events2], [None], max_depth=15)
        # await self.monitor2.append_perf_data(0, [trace_events], [None], max_depth=4)