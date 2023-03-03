# Copyright 2023 Yan Yan
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import inspect
import types
from functools import partial
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Optional,
                    Set, Tuple, Type, Union, TypeVar)
import subprocess
from tensorpc.flow.flowapp.components import mui, three
from tensorpc.flow import marker
import asyncio
import psutil 
import io 
import csv 
from dataclasses import dataclass
@dataclass
class GPUMeasure:
    name: str 
    gpuusage: int 
    memusage: int
    temperature: int
    memused: int
    memtotal: int

    def to_string(self):
        msg = f"gpu={self.gpuusage}%,mem={self.memused}/{self.memtotal}MB,"
        msg += f"{self.temperature}\u2103,io={self.memusage}%"
        return msg 

@dataclass
class GPUMonitor:
    util: mui.CircularProgress
    mem: mui.CircularProgress

class ComputeResourceMonitor(mui.FlexBox):
    def __init__(self):
        self.cpu = mui.CircularProgress().prop(color="green", variant="determinate")
        self.mem = mui.CircularProgress().prop(color="aqua", variant="determinate")
        num_gpu = len(self._get_gpu_measures())

        self.gpus: List[GPUMonitor] = []
        gpu_uis = []
        for i in range(num_gpu):
            util = mui.CircularProgress().prop(color="blue", variant="determinate")
            mem = mui.CircularProgress().prop(color="sliver", variant="determinate")
            self.gpus.append(GPUMonitor(util, mem))
            gpu_uis.extend([mui.Divider("vertical"), util, mem])
        super().__init__([
            self.cpu,
            self.mem,
            *gpu_uis,
        ])
        self.prop(flex_flow="row wrap")

        self.shutdown_ev = asyncio.Event()

    @marker.mark_did_mount
    def _on_mount(self):
        self.shutdown_ev.clear()
        self._resource_task = asyncio.create_task(self.get_resource())

    @marker.mark_will_unmount
    def _on_unmount(self):
        self.shutdown_ev.set()

    def _get_gpu_measures(self) -> List[GPUMeasure]:
        querys = [
            "gpu_name",
            "utilization.gpu",
            "utilization.memory",
            "temperature.gpu",
            "memory.used",
            "memory.total",
        ]
        try:
            output = subprocess.check_output(["nvidia-smi", f"--query-gpu={','.join(querys)}", "--format=csv"])
            output_str = output.decode("utf-8")
            output_str_file = io.StringIO(output_str)
            csv_data = csv.reader(output_str_file, delimiter=',', quotechar=',')
            rows = list(csv_data)[1:]
            rows = [[r.strip() for r in row] for row in rows]
            gpumeasures: List[GPUMeasure] = []
            for r in rows:
                query = dict(zip(querys, r))
                gpuusage = int(query["utilization.gpu"].split(" ")[0])
                memusage = int(query["utilization.memory"].split(" ")[0])
                memused = int(query["memory.used"].split(" ")[0])
                memtotal = int(query["memory.total"].split(" ")[0])
                temp = int(query["temperature.gpu"])
                gpumeasure = GPUMeasure(query["gpu_name"], gpuusage, memusage, temp, memused, memtotal)
                gpumeasures.append(gpumeasure)
            return gpumeasures
        except:
            return []

    async def get_resource(self):
        while True:
            cpu_percent = psutil.cpu_percent()
            vm_percent = psutil.virtual_memory().percent
            ev = self.cpu.update_event(value=cpu_percent)
            ev += self.mem.update_event(value=vm_percent)
            if len(self.gpus) > 0:
                gpumeasures: List[GPUMeasure] = self._get_gpu_measures()
                for g, m in zip(gpumeasures, self.gpus):
                    ev += m.util.update_event(value=g.gpuusage)
                    ev += m.mem.update_event(value=g.memused / g.memtotal * 100)
            await self.send_and_wait(ev)
            await asyncio.sleep(2.0)
            if self.shutdown_ev.is_set():
                break

