from typing import Dict, List, Optional, Tuple
import subprocess 
import dataclasses
import io 
import csv 

@dataclasses.dataclass
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

def get_nvidia_gpu_measures() -> List[GPUMeasure]:
    gpumeasures: List[GPUMeasure] = []

    querys = [
        "gpu_name",
        "utilization.gpu",
        "utilization.memory",
        "temperature.gpu",
        "memory.used",
        "memory.total",
    ]
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={','.join(querys)}", "--format=csv"])
    except:
        return gpumeasures
    output_str = output.decode("utf-8")
    output_str_file = io.StringIO(output_str)
    csv_data = csv.reader(output_str_file, delimiter=',', quotechar=',')
    rows = list(csv_data)[1:]
    rows = [[r.strip() for r in row] for row in rows]
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