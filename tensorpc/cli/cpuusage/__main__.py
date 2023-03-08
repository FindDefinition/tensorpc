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

from tensorpc.flow.client import update_node_status

import psutil
import fire

import asyncio


async def main_async(duration: float = 2):
    while True:
        await asyncio.sleep(duration)
        cpu_percent = psutil.cpu_percent()
        vm_percent = psutil.virtual_memory().percent

        content = f"cpu={cpu_percent:.2f}%,mem={vm_percent:.2f}%"
        update_node_status(content)


def main(duration: float = 2):
    asyncio.run(main_async(duration))


if __name__ == "__main__":
    fire.Fire(main)
