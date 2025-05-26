import copy
import json
import math

import cv2
import yaml
from tensorpc.apps.scivis.tensor import TensorPanel
from tensorpc.constants import TENSORPC_DEV_SECRET_PATH
from tensorpc.dock import mui, three, plus, appctx, mark_create_layout
import dataclasses
from typing import Any 
import numpy as np
import tensorpc.core.datamodel as D

class App:
    @mark_create_layout
    def my_layout(self):
        self.monitor = TensorPanel()
        # self.monitor2 = mui.HBox([mui.Markdown("## PerfMonitor"),])
        # self.monitor2 = PerfMonitor(use_view=True)

        return mui.VBox([
            mui.Button("Load Trace", self._set_data),
            self.monitor.prop(flex=1),

            # three.ViewCanvas([
            #     self.monitor.prop(flex=1),
            #     # self.monitor2.prop(flex=1),

            # ]).prop(display="flex",
            #     flexDirection="column", width="100%", height="100%", overflow="hidden"),
            
        ]).prop(minHeight=0,
                minWidth=0,
                width="100%",
                height="100%",
                overflow="hidden")

    async def _set_data(self):
        img = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
        print(img.shape, img.dtype)
        await self.monitor.set_new_tensor(img)
        # await self.monitor2.append_perf_data(0, [trace_events], [None], max_depth=4)