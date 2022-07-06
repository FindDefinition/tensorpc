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

from typing import Any, Dict, List, Optional
from tensorpc.apps.flow.coretypes import get_uid
from tensorpc.apps.flow.flowapp import App, AppEvent, AppEventType, LayoutEvent, UIEvent
import asyncio
from tensorpc.core.httpclient import http_remote_call
from tensorpc.core.serviceunit import get_cls_obj_from_module_name
import tensorpc
from ..client import MasterMeta
from tensorpc import prim 
from tensorpc.apps.flow.serv_names import serv_names
import traceback


class FlowApp:
    """this service must run inside devflow.
    """
    def __init__(self, module_name: str, config: Dict[str, Any]) -> None:
        print(module_name, config)
        self.module_name = module_name
        self.config = config
        self.shutdown_ev = asyncio.Event()
        self.master_meta = MasterMeta()
        assert self.master_meta.is_inside_devflow, "this service must run inside devflow"
        assert self.master_meta.is_http_valid
        obj_type, alias, module_key = get_cls_obj_from_module_name(
            module_name)
        self.app: App = obj_type(**self.config)
        self._send_loop_queue: "asyncio.Queue[AppEvent]" = asyncio.Queue(
        )
        self._send_loop_task: Optional[asyncio.Task] = None
        self._need_to_send_env: Optional[AppEvent] = None
        self.shutdown_ev.clear()
        self._send_loop_task = asyncio.create_task(self._send_loop())

        self._uid = get_uid(self.master_meta.graph_id, self.master_meta.node_id)
        lay = self.app._get_app_layout()
        print(lay)
        asyncio.run_coroutine_threadsafe(self._send_loop_queue.put(AppEvent("", AppEventType.UpdateLayout, LayoutEvent(lay))), loop=asyncio.get_running_loop())

    def _get_app(self):
        return self.app

    async def run_ui_event(self, data):
        ev = UIEvent.from_dict(data)
        return await self.app._handle_control_event(ev)
        # await self.app._queue.put(UIEvent.from_dict(data))

    def get_layout(self):
        return self.app._get_app_layout()

    async def _http_remote_call(self, key: str, *args, **kwargs):
        return await http_remote_call(prim.get_http_client_session(),
                                      self.master_meta.http_url, key, *args,
                                      **kwargs)

    async def _send_http_event(self, ev: AppEvent):
        if self.master_meta.is_worker:
            return await self._http_remote_call(serv_names.FLOWWORKER_PUT_APP_EVENT, self.master_meta.graph_id, ev.to_dict())
        else:
            return await self._http_remote_call(serv_names.FLOW_PUT_APP_EVENT, ev.to_dict())

    async def _send_grpc_event(self, ev: AppEvent, robj: tensorpc.AsyncRemoteManager):
        if self.master_meta.is_worker:
            return await robj.remote_call(serv_names.FLOWWORKER_PUT_APP_EVENT, self.master_meta.graph_id, ev.to_dict())
        else:
            return await robj.remote_call(serv_names.FLOW_PUT_APP_EVENT, ev.to_dict())

    async def _send_loop(self):
        # TODO unlike flowworker, the app shouldn't disconnect to master/worker.
        # so we should just use retry here.
        shut_task = asyncio.create_task(self.shutdown_ev.wait())
        # async with tensorpc.AsyncRemoteManager(url) as robj:
        if self._need_to_send_env is not None:
            # TODO if this fail?
            await self._send_http_event(self._need_to_send_env)
            self._need_to_send_env = None
        send_task = asyncio.create_task(self._send_loop_queue.get())
        wait_tasks: List[asyncio.Task] = [shut_task, send_task]
        while True:
            # TODO if send fail, save this ev and send after reconnection
            # ev = await self._send_loop_queue.get()
            (done, pending) = await asyncio.wait(
                wait_tasks, return_when=asyncio.FIRST_COMPLETED)
            if shut_task in done:
                break
            ev: AppEvent = send_task.result()
            # assign uid here.
            ev.uid = self._uid
            send_task = asyncio.create_task(self._send_loop_queue.get())
            wait_tasks: List[asyncio.Task] = [shut_task, send_task]
            try:
                await self._send_http_event(ev)
            except Exception as e:
                # remote call may fail by connection broken
                # TODO retry for reconnection
                traceback.print_exc()
                self._send_loop_task = None
                self._need_to_send_env = ev
                # when disconnect to master, enter slient mode
                break
        self._send_loop_task = None
