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

from typing import Any
import asyncio
from tensorpc.flow.flowapp.core import Component, create_ignore_usr_msg, Undefined, UIRunStatus


async def handle_raw_event(ev: Any, comp: Component, just_run: bool = False):
    # ev: [type, data]
    type, data = ev
    if type in comp._flow_event_handlers:
        handler = comp._flow_event_handlers[type]
        if isinstance(handler, Undefined):
            return
    else:
        return
    if comp.props.status == UIRunStatus.Running.value:
        msg = create_ignore_usr_msg(comp)
        await comp.send_app_event_and_wait(msg)
        return
    elif comp.props.status == UIRunStatus.Stop.value:
        comp.state_change_callback(data)
        def ccb(cb):
            return lambda: cb(data)
        if not just_run:
            comp._task = asyncio.create_task(
                comp.run_callback(ccb(handler.cb), True))
        else:
            await comp.run_callback(ccb(handler.cb), True)
            
async def handle_standard_event(comp: Component, data: Any, sync_first: bool = False):
    if comp.props.status == UIRunStatus.Running.value:
        msg = create_ignore_usr_msg(comp)
        await comp.send_app_event_and_wait(msg)
        return
    elif comp.props.status == UIRunStatus.Stop.value:
        cb1 = comp.get_callback()
        comp.state_change_callback(data)
        if cb1 is not None:
            def ccb(cb):
                return lambda: cb(data)
            comp._task = asyncio.create_task(comp.run_callback(ccb(cb1), True, sync_first=sync_first))
        else:
            await comp.sync_status(True)

async def handle_standard_event_no_arg(comp: Component, sync_first: bool = False):
    if comp.props.status == UIRunStatus.Running.value:
        msg = create_ignore_usr_msg(comp)
        await comp.send_app_event_and_wait(msg)
        return
    elif comp.props.status == UIRunStatus.Stop.value:
        cb2 = comp.get_callback()
        if cb2 is not None:
            comp._task = asyncio.create_task(comp.run_callback(cb2, sync_first=sync_first))
