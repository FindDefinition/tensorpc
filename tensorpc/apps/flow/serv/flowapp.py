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

import pickle
from typing import Any, Dict, List, Optional
from tensorpc.apps.flow.coretypes import ScheduleEvent, get_uid
from tensorpc.apps.flow.flowapp.core import AppEditorFrontendEvent, AppEvent, AppEventType, LayoutEvent, NotifyEvent, NotifyType, ScheduleNextForApp, UIEvent, UISaveStateEvent
from tensorpc.apps.flow.flowapp.app import App
import asyncio
from tensorpc.core import marker
from tensorpc.core.httpclient import http_remote_call
from tensorpc.core.serviceunit import ReloadableDynamicClass, ServiceUnit
import tensorpc
from ..client import MasterMeta
from tensorpc import prim
from tensorpc.apps.flow.serv_names import serv_names
import traceback
import time 

class FlowApp:
    """this service must run inside devflow.
    if headless is enabled, all event sent to frontend will be ignored.
    """

    def __init__(self, module_name: str, config: Dict[str, Any], headless: bool = False) -> None:
        print(module_name, config)
        self.module_name = module_name
        self.config = config
        self.shutdown_ev = asyncio.Event()
        self.master_meta = MasterMeta()
        assert self.master_meta.is_inside_devflow, "this service must run inside devflow"
        assert self.master_meta.is_http_valid
        self._send_loop_task: Optional[asyncio.Task] = None
        self._need_to_send_env: Optional[AppEvent] = None
        self.shutdown_ev.clear()

        self._uid = get_uid(self.master_meta.graph_id,
                            self.master_meta.node_id)
        self.headless = headless
        self.dynamic_app_cls = ReloadableDynamicClass(module_name)
        self.app: App = self.dynamic_app_cls.obj_type(**self.config)
        self.app_su = ServiceUnit(module_name, config)
        self.app_su.init_service(self.app)
        self.app._app_dynamic_cls = self.dynamic_app_cls
        self.app._app_service_unit = self.app_su

        self._send_loop_queue: "asyncio.Queue[AppEvent]" = self.app._queue
        self.app._send_callback = self._send_http_event
        self._send_loop_task = asyncio.create_task(self._send_loop())
        # print(lay)
        print(self.master_meta.http_url)

    @marker.mark_async_init
    async def init(self):
        self.app.app_initialize()
        await self.app.app_initialize_async()
        if self.app._force_special_layout_method:
            await self.app._app_run_layout_function()
        lay = self.app._get_app_layout()
        await self._send_loop_queue.put(
            AppEvent("", {AppEventType.UpdateLayout: LayoutEvent(lay)}))
        # TODO should we just use grpc client to query init state here?
        await self._send_loop_queue.put(
            AppEvent("", {AppEventType.Notify: NotifyEvent(NotifyType.AppStart)}))

    def _get_app(self):
        return self.app

    async def run_single_event(self, type, data):
        if type == AppEventType.AppEditor.value:
            ev = AppEditorFrontendEvent.from_dict(data)
            return await self.app._handle_code_editor_event_system(ev)
        elif type == AppEventType.UIEvent.value:
            ev = UIEvent.from_dict(data)
            return await self.app._handle_event_with_ctx(ev)
        elif type == AppEventType.ScheduleNext.value:
            asyncio.create_task(self._run_schedule_event_task(data))
        elif type == AppEventType.UISaveStateEvent.value:
            ev = UISaveStateEvent.from_dict(data)
            return await self.app._restore_simple_app_state(ev.uid_to_data)

    async def run_app_service(self, key: str, *args, **kwargs):
        serv, meta = self.app_su.get_service_and_meta(key)
        res_or_coro = serv(*args, **kwargs)
        if meta.is_async:
            return await res_or_coro
        else:
            return res_or_coro

    async def run_app_async_gen_service(self, key: str, *args, **kwargs):
        serv, meta = self.app_su.get_service_and_meta(key)
        assert meta.is_async and meta.is_gen
        async for x in serv(*args, **kwargs):
            yield x

    async def _run_schedule_event_task(self, data):
        ev = ScheduleEvent.from_dict(data)
        res = await self.app.flow_run(ev)
        if res is not None:
            ev = ScheduleEvent(time.time_ns(), res, {})
            appev = ScheduleNextForApp(ev.to_dict())
            await self._send_loop_queue.put(AppEvent(self._uid, {
                AppEventType.ScheduleNext: appev,
            }))

    def get_layout(self):
        return self.app._get_app_layout()

    async def _http_remote_call(self, key: str, *args, **kwargs):
        return await http_remote_call(prim.get_http_client_session(),
                                      self.master_meta.http_url, key, *args,
                                      **kwargs)

    async def _send_http_event(self, ev: AppEvent):
        ev.uid = self._uid
        if self.master_meta.is_worker:
            return await self._http_remote_call(
                serv_names.FLOWWORKER_PUT_APP_EVENT, self.master_meta.graph_id,
                ev.to_dict())
        else:
            return await self._http_remote_call(serv_names.FLOW_PUT_APP_EVENT,
                                                ev.to_dict())

    async def _send_grpc_event(self, ev: AppEvent,
                               robj: tensorpc.AsyncRemoteManager):
        if self.master_meta.is_worker:
            return await robj.remote_call(serv_names.FLOWWORKER_PUT_APP_EVENT,
                                          self.master_meta.graph_id,
                                          ev.to_dict())
        else:
            return await robj.remote_call(serv_names.FLOW_PUT_APP_EVENT,
                                          ev.to_dict())

    async def _send_grpc_event_large(self, ev: AppEvent,
                                     robj: tensorpc.AsyncRemoteManager):
        if self.master_meta.is_worker:
            return await robj.chunked_remote_call(
                serv_names.FLOWWORKER_PUT_APP_EVENT, self.master_meta.graph_id,
                ev.to_dict())
        else:
            return await robj.chunked_remote_call(
                serv_names.FLOW_PUT_APP_EVENT, ev.to_dict())

    def _send_grpc_event_large_sync(self, ev: AppEvent,
                                     robj: tensorpc.RemoteManager):
        if self.master_meta.is_worker:
            return robj.chunked_remote_call(
                serv_names.FLOWWORKER_PUT_APP_EVENT, self.master_meta.graph_id,
                ev.to_dict())
        else:
            return robj.chunked_remote_call(
                serv_names.FLOW_PUT_APP_EVENT, ev.to_dict())


    async def _send_loop(self):
        # TODO unlike flowworker, the app shouldn't disconnect to master/flowworker.
        # so we should just use retry here.
        shut_task = asyncio.create_task(self.shutdown_ev.wait())
        grpc_url = self.master_meta.grpc_url
        async with tensorpc.AsyncRemoteManager(grpc_url) as robj:
            send_task = asyncio.create_task(self._send_loop_queue.get())
            wait_tasks: List[asyncio.Task] = [shut_task, send_task]
            master_disconnect = 0.0
            retry_duration = 2.0 # 2s
            previous_event = AppEvent(self._uid, {})
            while True:
                # if send fail, MERGE incoming app events, and send again after some time.
                # all app event is "replace" in frontend.
                (done,
                pending) = await asyncio.wait(wait_tasks,
                                            return_when=asyncio.FIRST_COMPLETED)
                if shut_task in done:
                    break
                ev: AppEvent = send_task.result()
                if self.headless:
                    for k, v in ev.type_to_event.items():
                        if k == AppEventType.UIEvent:
                            assert isinstance(v, UIEvent)
                            await self.app._handle_event_with_ctx(v)
                    continue 
                ts = time.time()
                # assign uid here.
                ev.uid = self._uid
                send_task = asyncio.create_task(self._send_loop_queue.get())
                wait_tasks: List[asyncio.Task] = [shut_task, send_task]
                if master_disconnect >= 0:
                    previous_event = previous_event.merge_new(ev)
                    if ts - master_disconnect > retry_duration:
                        try:
                            # await self._send_http_event(previous_event)
                            await self._send_grpc_event_large(previous_event, robj)
                            master_disconnect = -1
                            previous_event = AppEvent(self._uid, {})
                        except Exception as e:
                            print("Retry connection Fail.")
                            master_disconnect = ts   
                else:
                    try:
                        # print("SEND", ev.type)
                        # await self._send_http_event(ev)
                        await self._send_grpc_event_large(ev, robj)
                        # print("SEND", ev.type, "FINISH")
                    except Exception as e:
                        traceback.print_exc()
                        # remote call may fail by connection broken
                        # when disconnect to master/remote worker, enter slient mode
                        previous_event = previous_event.merge_new(ev)
                        master_disconnect = ts
                # trigger sent event here.
                if ev.sent_event is not None:
                    ev.sent_event.set()

        self._send_loop_task = None

    @marker.mark_exit
    async def on_exit(self):
        # save simple state to master
        try:
            grpc_url = self.master_meta.grpc_url
            uiev = UISaveStateEvent(self.app._get_simple_app_state())
            ev = AppEvent(self._uid, {AppEventType.UISaveStateEvent: uiev})
            # TODO remove this dump
            # check user error, user can't store invalid
            # object that exists after reload module.
            pickle.dumps(ev)
            async with tensorpc.AsyncRemoteManager(grpc_url) as robj:
                await self._send_grpc_event_large(ev, robj)

        except:
            traceback.print_exc()
