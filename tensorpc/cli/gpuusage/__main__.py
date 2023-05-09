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
from tensorpc.flow.client import update_node_status

import csv
import subprocess
import fire
import io
import asyncio
from tensorpc.utils.gpuusage import get_nvidia_gpu_measures


async def main_async(duration: float = 2):
    while True:
        await asyncio.sleep(duration)
        gpu_measures = get_nvidia_gpu_measures()
        gpu_names = ",".join(set([r.name for r in gpu_measures]))
        measures = [
            f"{i}: {gm.to_string()}" for i, gm in enumerate(gpu_measures)
        ]
        measure = "\n".join(measures)
        content = f"{gpu_names}\n{measure}"
        update_node_status(content)


def main(duration: float = 2):
    try:
        asyncio.run(main_async(duration))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    fire.Fire(main)
