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

from typing import Any, Tuple, Union
import asyncio
from tensorpc.flow.flowapp.core import Component, EventType, create_ignore_usr_msg, Undefined, UIRunStatus, FrontendEventType

_ONEARG_EVENTS = set([FrontendEventType.Change.value, FrontendEventType.InputChange.value,
    FrontendEventType.DialogClose.value])

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
        await comp.send_and_wait(msg)
        return
    elif comp.props.status == UIRunStatus.Stop.value:
        comp.state_change_callback(data, type)
        def ccb(cb):
            return lambda: cb(data)
        if not just_run:
            comp._task = asyncio.create_task(
                comp.run_callback(ccb(handler.cb), True))
        else:
            await comp.run_callback(ccb(handler.cb), True)
            
async def handle_standard_event(comp: Component, data: EventType, sync_first: bool = False):
    if comp.props.status == UIRunStatus.Running.value:
        msg = create_ignore_usr_msg(comp)
        await comp.send_and_wait(msg)
        return
    elif comp.props.status == UIRunStatus.Stop.value:
        if data[0] in _ONEARG_EVENTS:
            handler = comp.get_event_handler(data[0])
            comp.state_change_callback(data[1], data[0])
            if handler is not None:
                def ccb(cb):
                    return lambda: cb(data[1])
                comp._task = asyncio.create_task(comp.run_callback(ccb(handler.cb), True, sync_first=sync_first))
            else:
                await comp.sync_status(True)
        elif data[0] == FrontendEventType.Click.value:
            handler = comp.get_event_handler(FrontendEventType.Click.value)
            if handler is not None:
                comp._task = asyncio.create_task(comp.run_callback(handler.cb, sync_first=sync_first))
        else:
            raise NotImplementedError

# async def handle_change_event_no_arg(comp: Component, sync_first: bool = False):
#     if comp.props.status == UIRunStatus.Running.value:
#         msg = create_ignore_usr_msg(comp)
#         await comp.send_and_wait(msg)
#         return
#     elif comp.props.status == UIRunStatus.Stop.value:
#         cb2 = comp.get_callback()
#         if cb2 is not None:
#             comp._task = asyncio.create_task(comp.run_callback(cb2, sync_first=sync_first))
