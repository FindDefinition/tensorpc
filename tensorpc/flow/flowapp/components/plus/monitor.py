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

from tensorpc.flow.flowapp.components import mui, three
from tensorpc.flow import marker
import asyncio
from typing_extensions import ParamSpec
import psutil 

class ComputeResourceMonitor(mui.FlexBox):
    def __init__(self):
        self.cpu = mui.CircularProgress().prop(color="green", variant="determinate")
        self.mem = mui.CircularProgress().prop(color="aqua", variant="determinate")
        self.gpus = mui.Fragment([])
        super().__init__([
            self.cpu,
            self.mem,
            mui.Divider("vertical"),
            self.gpus,
        ])
        self.prop(flex_flow="row wrap")

        self.shutdown_ev = asyncio.Event()

    @marker.mark_did_mount
    def _on_mount(self):
        print("MOUNT")
        self.shutdown_ev.clear()
        self._resource_task = asyncio.create_task(self.get_resource())

    @marker.mark_will_unmount
    def _on_unmount(self):
        print("UNMOUNT")

        self.shutdown_ev.set()

    async def get_resource(self):
        while True:
            await asyncio.sleep(1.0)
            if self.shutdown_ev.is_set():
                break
            cpu_percent = psutil.cpu_percent()
            vm_percent = psutil.virtual_memory().percent
            ev = self.cpu.update_event(value=cpu_percent)
            ev += self.mem.update_event(value=vm_percent)
            await self.send_and_wait(ev)
