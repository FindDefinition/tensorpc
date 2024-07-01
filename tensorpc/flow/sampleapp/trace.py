import asyncio
from pathlib import Path
from typing import Optional

import aiohttp
from tensorpc.core.tracers.codefragtracer import CodeFragTracerResult
from tensorpc.flow import mui, three, plus, mark_create_layout, appctx
import sys
from tensorpc import PACKAGE_ROOT
import numpy as np

from tensorpc.flow.marker import mark_did_mount
from tensorpc import prim
from tensorpc.flow.sampleapp.tracesample.sample import trace_test_2

class TraceDevApp:

    @mark_create_layout
    def my_layout(self):
        btn = mui.Button("Prepare Trace")
        self.md = mui.Markdown()
        self.od = plus.BasicObjectTree()
        tracer_box = plus.VscodeTracerBox([
            btn,
            self.md,
        ]).prop(flexDirection="column")
        btn.event_click.on(self._prepare_trace)
        self.tracer_box = tracer_box
        tracer_box.event_trace_start.on(self._trace_start)
        tracer_box.event_trace_end.on(self._trace_end)
        return tracer_box

    async def _prepare_trace(self):
        print("WTFRTX")
        await self.tracer_box.prepare_trace(trace_test_2, (1, 2), {})

    async def _trace_start(self):
        await self.md.write("Tracing started")

    async def _trace_end(self, result: Optional[CodeFragTracerResult]):
        await self.md.write("Tracing End")
        if result is not None:
            print(result.line_result.eval_result)
