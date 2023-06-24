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
import time
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
import humanize

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
class NetworkMeasure:
    name: str 
    bytes_recv: int
    bytes_sent: int
    recv_speed: float 
    send_speed: float
    timestamp_second: float

@dataclass
class GPUMonitor:
    util: mui.CircularProgress
    mem: mui.CircularProgress


class ComputeResourceMonitor(mui.FlexBox):

    def __init__(self):
        self.cpu = mui.CircularProgress().prop(color="green",
                                               variant="determinate")
        self.mem = mui.CircularProgress().prop(color="aqua",
                                               variant="determinate")
        num_gpu = len(self._get_gpu_measures())

        self.gpus: List[GPUMonitor] = []
        gpu_uis = []
        for i in range(num_gpu):
            util = mui.CircularProgress().prop(color="blue",
                                               variant="determinate")
            mem = mui.CircularProgress().prop(color="sliver",
                                              variant="determinate")
            self.gpus.append(GPUMonitor(util, mem))
            gpu_uis.extend([mui.Divider("vertical"), util, mem])
        self.net_info_str = mui.Typography("").prop(whiteSpace="pre-wrap", fontSize="14px", fontFamily="monospace")
        super().__init__([
            mui.HBox([
                self.cpu,
                self.mem,
                *gpu_uis,
            ]),
            self.net_info_str,
        ])
        self.prop(flexFlow="column")
        self.prev_net_measures: Dict[str, NetworkMeasure] = {}
        self.period = 2.0

        self.shutdown_ev = asyncio.Event()

    @marker.mark_did_mount
    def _on_mount(self):
        self.shutdown_ev.clear()
        self._resource_task = asyncio.create_task(self.get_resource())

    @marker.mark_will_unmount
    def _on_unmount(self):
        self.shutdown_ev.set()

    def _get_network_measure(self) -> List[NetworkMeasure]:
        all_stats = psutil.net_if_stats()
        all_io_counters = psutil.net_io_counters(pernic=True)
        network_measures: List[NetworkMeasure] = []
        prev_measures = self.prev_net_measures
        for nic, stats in all_stats.items():
            if stats.isup:
                io_counters = all_io_counters[nic]
                ts = time.time()
                if nic in self.prev_net_measures:
                    prev_measure = prev_measures[nic]
                    recv_speed = (io_counters.bytes_recv -
                                  prev_measure.bytes_recv) / (ts - prev_measure.timestamp_second)
                    send_speed = (io_counters.bytes_sent -
                                  prev_measure.bytes_sent) / (ts - prev_measure.timestamp_second)
                else:
                    recv_speed = 0
                    send_speed = 0
                measure = NetworkMeasure(nic, io_counters.bytes_recv, io_counters.bytes_sent, recv_speed, send_speed, ts)
                self.prev_net_measures[nic] = measure
                network_measures.append(measure)
        return network_measures
        
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
            output = subprocess.check_output([
                "nvidia-smi", f"--query-gpu={','.join(querys)}", "--format=csv"
            ])
            output_str = output.decode("utf-8")
            output_str_file = io.StringIO(output_str)
            csv_data = csv.reader(output_str_file,
                                  delimiter=',',
                                  quotechar=',')
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
                gpumeasure = GPUMeasure(query["gpu_name"], gpuusage, memusage,
                                        temp, memused, memtotal)
                gpumeasures.append(gpumeasure)
            return gpumeasures
        except:
            return []

    async def get_resource(self):
        while True:
            cpu_percent = psutil.cpu_percent()
            
            vm_percent = psutil.virtual_memory().percent
            net_measures = self._get_network_measure()
            net_info_strs = [f"{m.name}: recv={humanize.naturalsize(m.recv_speed)}/s, send={humanize.naturalsize(m.send_speed)}/s" for m in net_measures]
            ev = self.cpu.update_event(value=cpu_percent)
            ev += self.mem.update_event(value=vm_percent)
            ev += self.net_info_str.update_event(value="\n".join(net_info_strs))
            if len(self.gpus) > 0:
                gpumeasures: List[GPUMeasure] = self._get_gpu_measures()
                for g, m in zip(gpumeasures, self.gpus):
                    ev += m.util.update_event(value=g.gpuusage)
                    ev += m.mem.update_event(value=g.memused / g.memtotal *
                                             100)
            await self.send_and_wait(ev)
            await asyncio.sleep(self.period)
            if self.shutdown_ev.is_set():
                break
