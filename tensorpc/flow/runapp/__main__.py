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

import fire
import asyncio

from tensorpc.flow.flowapp import App
from tensorpc.flow.serv.flowapp import FlowApp


async def main_async(module_name: str, **config):
    app_serv = FlowApp(module_name, config, True)
    await app_serv.init()
    await app_serv.app.headless_main()
    app_serv.shutdown_ev.set()
    assert app_serv._send_loop_task is not None
    await app_serv._send_loop_task


def main(module_name: str, **config):
    asyncio.run(main_async(module_name, **config))


if __name__ == "__main__":
    fire.Fire(main)