# Copyright 2022 Yan Yan
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

from dataclasses import dataclass
from typing import List
from tensorpc.apps.flow.client import update_node_status

import csv 
import subprocess
import fire 
import io 
import asyncio 

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

async def main_async(duration: float = 1):
    while True:
        await asyncio.sleep(duration)
        querys = [
            "gpu_name",
            "utilization.gpu",
            "utilization.memory",
            "temperature.gpu",
            "memory.used",
            "memory.total",
        ]
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
        gpu_names = ",".join(set([r[0] for r in rows]))
        measures = [f"{i}: {gm.to_string()}" for i, gm in enumerate(gpumeasures)]
        measure = "\n".join(measures)
        content = f"{gpu_names}\n{measure}"
        update_node_status(content)

def main(duration: float = 1):
    try:
        asyncio.run(main_async(duration))
    except KeyboardInterrupt:
        return 0

if __name__ == "__main__":
    fire.Fire(main)