# Copyright 2024 Yan Yan
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

import copy
from functools import partial
import inspect
import io
import os
from pathlib import Path
import pickle
from runpy import run_path
from typing import Any, Dict, List, Optional

import grpc
from tensorpc.core.asyncclient import simple_chunk_call_async
from tensorpc.core.defs import FileDesp, FileResource, FileResourceRequest
from tensorpc.flow.constants import TENSORPC_APP_ROOT_COMP, TENSORPC_LSP_EXTRA_PATH
from tensorpc.flow.coretypes import ScheduleEvent, get_unique_node_id
from tensorpc.core.tree_id import UniqueTreeId, UniqueTreeIdForComp
from tensorpc.flow.serv.common import handle_file_resource
from tensorpc.flow.vscode.coretypes import VscodeTensorpcMessage, VscodeTensorpcQuery
from tensorpc.flow import appctx
from tensorpc.flow.core.appcore import ALL_OBSERVED_FUNCTIONS, RemoteCompEvent, enter_app_context
from tensorpc.flow.components.mui import FlexBox, flex_wrapper
from tensorpc.flow.core.component import AppEditorEvent, AppEditorFrontendEvent, AppEvent, AppEventType, InitLSPClientEvent, LayoutEvent, NotifyEvent, NotifyType, RemoteComponentBase, ScheduleNextForApp, UIEvent, UIExceptionEvent, UISaveStateEvent, UserMessage
from tensorpc.flow.flowapp.app import App, EditableApp
import asyncio
from tensorpc.core import marker
from tensorpc.core.httpclient import http_remote_call
from tensorpc.core.serviceunit import AppFuncType, ReloadableDynamicClass, ServiceUnit
import tensorpc
from tensorpc.flow.core.reload import AppReloadManager, FlowSpecialMethods

from tensorpc.flow.jsonlike import Undefined
from tensorpc.flow.langserv import close_tmux_lang_server, get_tmux_lang_server_info_may_create
from ..client import AppLocalMeta, MasterMeta
from tensorpc import prim
from tensorpc.flow.serv_names import serv_names
from tensorpc.core.serviceunit import ServiceEventType
import traceback
import time
import sys
from urllib import parse

_grpc_status_master_disconnect = set([
     grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
])

class FlowApp:
    """this service must run inside devflow.
    if headless is enabled, all event sent to frontend will be ignored.
    if external_argv is enabled, it will be used as sys.argv and launched
        as a python script after app init
    """

    def __init__(self,
                 module_name: str,
                 config: Dict[str, Any],
                 headless: bool = False,
                 external_argv: Optional[List[str]] = None) -> None:
        # print(module_name, config)
        if external_argv is not None:
            print("external_argv", external_argv)
        self.module_name = module_name
        self.config = config
        self.shutdown_ev = asyncio.Event()
        self.master_meta = MasterMeta()
        self.app_meta = AppLocalMeta()
        assert not prim.get_server_is_sync(), "only support async server"
        process_title = self.master_meta.process_title
        try:
            import setproctitle  # type: ignore
            setproctitle.setproctitle(process_title)
        except ImportError:
            pass
        if not headless:
            assert self.master_meta.is_inside_devflow, "this service must run inside devflow"
            # assert self.master_meta.is_http_valid
        self._send_loop_task: Optional[asyncio.Task] = None
        self._need_to_send_env: Optional[AppEvent] = None
        self.shutdown_ev.clear()
        if not headless:
            self._uid = get_unique_node_id(self.master_meta.graph_id,
                                self.master_meta.node_id)
        else:
            self._uid = ""
        self.headless = headless
        reload_mgr = AppReloadManager(ALL_OBSERVED_FUNCTIONS)
        self.dynamic_app_cls = ReloadableDynamicClass(module_name, reload_mgr)
        static_creator = self.dynamic_app_cls.get_object_creator_if_exists()
        if static_creator is not None:
            obj = static_creator()
        else:
            obj = self.dynamic_app_cls.obj_type(**self.config)
        if isinstance(obj, App):
            self.app: App = obj
        elif isinstance(obj, FlexBox):
            # external root
            external_root = obj
            self.app: App = EditableApp(external_root=external_root,
                                        reload_manager=reload_mgr)
        else:
            # other object, must declare a tensorpc_flow_layout
            # external_root = flex_wrapper(obj)
            self.app: App = EditableApp(external_wrapped_obj=obj,
                                        reload_manager=reload_mgr)
            self.app._app_force_use_layout_function()
        self.app._flow_app_comp_core.reload_mgr = reload_mgr
        self.app_su = ServiceUnit(module_name, config)
        self.app_su.init_service(obj)
        self.app._app_dynamic_cls = self.dynamic_app_cls
        self.app._app_service_unit = self.app_su
        self.app._flow_app_is_headless = headless
        self._send_loop_queue: "asyncio.Queue[AppEvent]" = self.app._queue
        # self.app._send_callback = self._send_http_event
        self._send_loop_task = asyncio.create_task(self._send_loop_v2())
        self.lsp_port = self.master_meta.lsp_port
        if self.lsp_port is not None:
            assert self.master_meta.lsp_fwd_port is not None
            self.lsp_fwd_port = self.master_meta.lsp_fwd_port
        else:
            self.lsp_fwd_port = None
        self.external_argv = external_argv
        self._external_argv_task: Optional[asyncio.Future] = None

    @marker.mark_server_event(event_type=marker.ServiceEventType.Init)
    async def init(self):
        if self.app._force_special_layout_method:
            layout_created = False
            special_methods = FlowSpecialMethods(self.app_su.serv_metas)
            if special_methods.create_layout is not None:
                await self.app._app_run_layout_function(
                    decorator_fn=special_methods.create_layout.get_binded_fn())
                layout_created = True
            if not layout_created:
                await self.app._app_run_layout_function()
        else:
            self.app.root._attach(UniqueTreeIdForComp.from_parts([TENSORPC_APP_ROOT_COMP]),
                                  self.app._flow_app_comp_core)
        # print(lay["layout"])
        self.app.app_initialize()
        await self.app.app_initialize_async()
        enable_lsp = self.lsp_port is not None and self.app._flowapp_enable_lsp
        print(enable_lsp, self.lsp_port)
        if enable_lsp:
            assert self.lsp_port is not None
            try:
                get_tmux_lang_server_info_may_create("pyright",
                                                    self.master_meta.node_id,
                                                    self.lsp_port)
            except:
                traceback.print_exc()
        lay = self.app._get_app_layout()
        self.app._flowapp_is_inited = True
        await self._send_loop_queue.put(
            AppEvent("", {AppEventType.UpdateLayout: LayoutEvent(lay)}))
        # TODO should we just use grpc client to query init state here?
        init_event: Dict[AppEventType, Any] = {
            AppEventType.Notify: NotifyEvent(NotifyType.AppStart)
        }
        if self.lsp_fwd_port is not None and enable_lsp:
            cfg = copy.deepcopy(self.app._flowapp_internal_lsp_config)
            extra_path = os.getenv(TENSORPC_LSP_EXTRA_PATH, None)
            if extra_path is not None:
                extra_paths = extra_path.split(":")
                if not isinstance(cfg.python.analysis.extraPaths, Undefined):
                    cfg.python.analysis.extraPaths.extend(extra_paths)
                else:
                    cfg.python.analysis.extraPaths = extra_paths
            init_event[AppEventType.InitLSPClient] = InitLSPClientEvent(
                self.lsp_fwd_port,
                cfg.get_dict())
        await self._send_loop_queue.put(AppEvent("", init_event))
        if self.external_argv is not None:
            with enter_app_context(self.app):
                self._external_argv_task = asyncio.create_task(
                    appctx.run_in_executor_with_exception_inspect(
                        partial(self._run_app_script,
                                argv=self.external_argv), ))

    def _run_app_script(self, argv: List[str]):
        argv_bkp = sys.argv
        sys.argv = argv
        print("???", argv)
        try:
            run_path(argv[0], run_name="__main__")
        finally:
            sys.argv = argv_bkp
            self._external_argv_task = None

    def _get_app(self):
        return self.app

    async def run_single_event(self, type, data, is_sync: bool = False):
        """is_sync: only used for ui event.
        """
        if type == AppEventType.AppEditor.value:
            ev = AppEditorFrontendEvent.from_dict(data)
            return await self.app._handle_code_editor_event_system(ev)
        elif type == AppEventType.UIEvent.value:
            ev = UIEvent.from_dict(data)
            return await self.app._handle_event_with_ctx(ev, is_sync)
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
            await self._send_loop_queue.put(
                AppEvent(self._uid, {
                    AppEventType.ScheduleNext: appev,
                }))

    async def handle_vscode_event(self, data: dict):
        """run event come from vscode, you need to install vscode-tensorpc-bridge extension first,
        then enable it in machine which run this app.
        """
        ev = VscodeTensorpcMessage(
            type=data["type"],
            currentUri=data["currentUri"],
            workspaceUri=data["workspaceUri"],
            selections=data["selections"] if "selections" in data else None,
            selectedCode=data["selectedCode"] if "selectedCode" in data else None,
        )
        await self.app.handle_vscode_event(ev)

    async def handle_msg_from_remote_comp(self, key: str, event: RemoteCompEvent):
        return await self.app.handle_msg_from_remote_comp(key, event)

    async def handle_vscode_query(self, data: dict):
        """run event come from vscode, you need to install vscode-tensorpc-bridge extension first,
        then enable it in machine which run this app.
        """
        ev = VscodeTensorpcQuery(
            type=data["type"],
            workspaceUri=data["workspaceUri"],
            data=data["data"],
        )
        res = await self.app.handle_vscode_query(ev)
        if res is not None:
            return {
                "queryResult": res,
                "appNodeId": self.master_meta.node_readable_id
            }
        return None

    async def handle_vscode_queries(self, datas: List[dict]):
        """run event come from vscode, you need to install vscode-tensorpc-bridge extension first,
        then enable it in machine which run this app.
        """
        res_list = []
        for data in datas:
            qres = await self.handle_vscode_query(data)
            res_list.append(qres)
        return res_list

    async def handle_simple_rpc(self, event: str, *args, **kwargs):
        return await self.app._flowapp_simple_rpc_handlers.call_event(event, *args, **kwargs)

    def get_vscode_breakpoints(self):
        state = self.app.get_vscode_state()
        if not state.is_init:
            return None 
        return state.get_all_breakpoints()

    async def relay_app_event_from_remote_component(self, app_event_dict: Dict[str, Any]):
        assert "remotePrefixes" in app_event_dict
        prefixes = app_event_dict["remotePrefixes"]
        uid = UniqueTreeIdForComp.from_parts(prefixes)
        # sync some state from remote component
        for ev_type, ev_dict in app_event_dict["typeToEvents"]:
            if ev_type == AppEventType.UpdateComponents:
                # when layout in remote changed, we must keep comp uids in main app.
                assert "remoteComponentAllChilds" in ev_dict
                remote_comp = self.app.root._get_comp_by_uid(uid.uid_encoded)
                assert isinstance(remote_comp, RemoteComponentBase)
                remote_comp.set_cur_child_uids(ev_dict["remoteComponentAllChilds"])
        app_ev = AppEvent.from_dict(app_event_dict)
        # print("relay_app_event_from_remote_component", app_event_dict)
        await self._send_loop_queue.put(app_ev)

    async def relay_app_storage_from_remote_comp(self, serv_name: str, args, kwargs):
        return await simple_chunk_call_async(self.master_meta.grpc_url,
                                             serv_name, *args, **kwargs)

    async def remote_comp_shutdown(self, prefixes: List[str]):
        uid = UniqueTreeIdForComp.from_parts(prefixes)
        remote_comp = self.app.root._get_comp_by_uid(uid.uid_encoded)
        assert isinstance(remote_comp, RemoteComponentBase)
        print("DISCONNECT")
        await remote_comp.disconnect()

    def get_layout(self, editor_only: bool = False):
        if editor_only:
            res = self.app._get_app_editor_state()
        else:
            res = self.app._get_app_layout()
        if self.app._flowapp_enable_lsp:
            res["lspPort"] = self.lsp_port
        return res

    def _get_file_path_stat(
        self, path: str
    ) -> os.stat_result:
        """Return the file path, stat result, and gzip status.

        This method should be called from a thread executor
        since it calls os.stat which may block.
        """
        return Path(path).stat()

    async def get_file_metadata(self, file_key: str, comp_uid: Optional[str] = None):
        url = parse.urlparse(file_key)
        base = url.path
        file_key_qparams = parse.parse_qs(url.query)
        if comp_uid is not None:
            comp = self.app.root._get_comp_by_uid(comp_uid)
            if isinstance(comp, RemoteComponentBase):
                return await comp.get_file_metadata(file_key)
        if self.app._flowapp_file_resource_handlers.has_event_handler(base):
            # we only use first value
            if len(file_key_qparams) > 0:
                file_key_qparams = {
                    k: v[0]
                    for k, v in file_key_qparams.items()
                }
            else:
                file_key_qparams = {}
            try:
                res = self.app._flowapp_file_resource_handlers.call_event(base, FileResourceRequest(base, True, None, file_key_qparams))
                if inspect.iscoroutine(res):
                    res = await res
                assert isinstance(res, FileResource)
                if res._empty:
                    return res
                if res.path is not None and res.stat is None:
                    loop = asyncio.get_event_loop()
                    st = await loop.run_in_executor(
                        None, self._get_file_path_stat, res.path
                    )
                    res.stat = st
                else:
                    if res.content is not None:
                        assert res.content is not None and isinstance(res.content, bytes)
                        res.length = len(res.content)
                        res.content = None
                        if res.stat is None:
                            # mtime must exist to resolve cache problem
                            if res.modify_timestamp_ns is None:
                                res.modify_timestamp_ns = time.time_ns()
                    msg = "file metadata must return stat or length if not path"
                    assert res.stat is not None or res.length is not None, msg
                return res  
            except:
                traceback.print_exc()
                raise
        else:
            raise KeyError(f"File key {file_key} not found.")

    async def get_file(self, file_key: str, offset: int, count: Optional[int] = None, chunk_size=2**16, comp_uid: Optional[str] = None):
        url = parse.urlparse(file_key)
        base = url.path
        file_key_qparams = parse.parse_qs(url.query)
        if comp_uid is not None:
            comp = self.app.root._get_comp_by_uid(comp_uid)
            if isinstance(comp, RemoteComponentBase):
                async for x in comp.get_file(file_key, offset, count, chunk_size):
                    yield x
                return 
        if self.app._flowapp_file_resource_handlers.has_event_handler(base):
            # we only use first value
            if len(file_key_qparams) > 0:
                file_key_qparams = {
                    k: v[0]
                    for k, v in file_key_qparams.items()
                }
            else:
                file_key_qparams = {}
            try:
                req = FileResourceRequest(base, False, offset, file_key_qparams)
                handler = self.app._flowapp_file_resource_handlers.get_event_handler(base).handler
                async for chunk in handle_file_resource(req, handler, chunk_size, count):
                    yield chunk
            except:
                traceback.print_exc()
                raise
        else:
            raise KeyError(f"File key {file_key} not found.")

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
        # import rich
        # rich.print(ev.to_dict())
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
            return robj.chunked_remote_call(serv_names.FLOW_PUT_APP_EVENT,
                                            ev.to_dict())

    def _create_exception_event(self, e: Exception):
        ss = io.StringIO()
        traceback.print_exc(file=ss)
        user_exc = UserMessage.create_error("UNKNOWN", f"AppServiceError: {repr(e)}", ss.getvalue())
        return AppEvent("", {AppEventType.UIException: UIExceptionEvent([user_exc])})

    async def _send_loop(self):
        # TODO unlike flowworker, the app shouldn't disconnect to master/flowworker.
        # so we should just use retry here.
        shut_task = asyncio.create_task(self.shutdown_ev.wait())
        grpc_url = self.master_meta.grpc_url
        async with tensorpc.AsyncRemoteManager(grpc_url) as robj:
            send_task = asyncio.create_task(self._send_loop_queue.get())
            wait_tasks: List[asyncio.Task] = [shut_task, send_task]
            master_disconnect = 0.0
            retry_duration = 2.0  # 2s
            previous_event = AppEvent(self._uid, {})
            while True:
                # if send fail, MERGE incoming app events, and send again after some time.
                # all app event is "replace" in frontend.
                (done, pending) = await asyncio.wait(
                    wait_tasks, return_when=asyncio.FIRST_COMPLETED)
                if shut_task in done:
                    break
                ev: AppEvent = send_task.result()
                if ev.is_loopback:
                    for k, v in ev.type_to_event.items():
                        if k == AppEventType.UIEvent:
                            assert isinstance(v, UIEvent)
                            await self.app._handle_event_with_ctx(v)
                    send_task = asyncio.create_task(
                        self._send_loop_queue.get())
                    wait_tasks: List[asyncio.Task] = [shut_task, send_task]
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
                            await self._send_grpc_event_large(
                                previous_event, robj)
                            master_disconnect = -1
                            previous_event = AppEvent(self._uid, {})
                        except Exception as e:
                            # TODO send error event to frontend
                            traceback.print_exc()
                            # print("Retry connection Fail.")
                            master_disconnect = ts
                else:
                    try:
                        # print("SEND", ev.type)
                        # await self._send_http_event(ev)
                        await self._send_grpc_event_large(ev, robj)
                        # print("SEND", ev.type, "FINISH")
                    except grpc.aio.AioRpcError as e:
                        traceback.print_exc()
                        if e.code() in _grpc_status_master_disconnect:
                            # print("Connection Fail.")
                            # remote call may fail by connection broken
                            # when disconnect to master/remote worker, enter slient mode
                            previous_event = previous_event.merge_new(ev)
                            master_disconnect = ts
                        else:
                            # other error. send exception
                            exc_ev = self._create_exception_event(e)
                            exc_ev.uid = self._uid
                            try:
                                await self._send_grpc_event_large(exc_ev, robj)
                            except Exception as e:
                                traceback.print_exc()
                    except Exception as e:
                        traceback.print_exc()
                        exc_ev = self._create_exception_event(e)
                        exc_ev.uid = self._uid
                        try:
                            await self._send_grpc_event_large(exc_ev, robj)
                        except Exception as e:
                            traceback.print_exc()

                # trigger sent event here.
                if ev.sent_event is not None:
                    ev.sent_event.set()

        self._send_loop_task = None

    async def _send_loop_stream_main(self, robj: tensorpc.AsyncRemoteManager):
        # TODO unlike flowworker, the app shouldn't disconnect to master/flowworker.
        # so we should just use retry here.
        shut_task = asyncio.create_task(self.shutdown_ev.wait())
        send_task = asyncio.create_task(self._send_loop_queue.get())
        wait_tasks: List[asyncio.Task] = [shut_task, send_task]
        previous_event = AppEvent(self._uid, {})
        while True:
            # if send fail, MERGE incoming app events, and send again after some time.
            # all app event is "replace" in frontend.
            (done, pending) = await asyncio.wait(
                wait_tasks, return_when=asyncio.FIRST_COMPLETED)
            if shut_task in done:
                break
            ev: AppEvent = send_task.result()
            if ev.is_loopback:
                for k, v in ev.type_to_event.items():
                    if k == AppEventType.UIEvent:
                        assert isinstance(v, UIEvent)
                        await self.app._handle_event_with_ctx(v)
                send_task = asyncio.create_task(
                    self._send_loop_queue.get())
                wait_tasks: List[asyncio.Task] = [shut_task, send_task]
                continue
            ts = time.time()
            # assign uid here.
            ev.uid = self._uid
            send_task = asyncio.create_task(self._send_loop_queue.get())
            wait_tasks: List[asyncio.Task] = [shut_task, send_task]
            # this is stream remote call, will use stream data to call a remote function, 
            # so we must yield (args, kwargs) instead of data.
            yield [ev.to_dict()], {}
            # trigger sent event here.
            if ev.sent_event is not None:
                ev.sent_event.set()

        self._send_loop_task = None

    async def _send_loop_v2(self):
        # TODO unlike flowworker, the app shouldn't disconnect to master/flowworker.
        # so we should just use retry here.
        grpc_url = self.master_meta.grpc_url
        try:
            async with tensorpc.AsyncRemoteManager(grpc_url) as robj:
                async for _ in robj.chunked_stream_remote_call(serv_names.FLOW_PUT_APP_EVENT, self._send_loop_stream_main(robj)):
                    pass
        finally:
            traceback.print_exc()
            self._send_loop_task = None 

    @marker.mark_server_event(event_type=ServiceEventType.Exit)
    async def on_exit(self):
        # save simple state to master
        try:
            self.app.app_terminate()
            await self.app.app_terminate_async()
        except:
            traceback.print_exc()
        # we can't close language server here
        # because we must wait for frontend shutdown client.
        # close_tmux_lang_server(self.master_meta.node_id)
        try:
            grpc_url = self.master_meta.grpc_url
            uiev = UISaveStateEvent(self.app._get_simple_app_state())
            editorev = self.app.set_editor_value_event("")
            ev = AppEvent(
                self._uid, {
                    AppEventType.UISaveStateEvent: uiev,
                    AppEventType.AppEditor: editorev
                })
            # TODO remove this dump
            # check user error, user can't store invalid
            # object that exists after reload module.
            pickle.dumps(ev)
            async with tensorpc.AsyncRemoteManager(grpc_url) as robj:
                await self._send_grpc_event_large(ev, robj)

        except:
            traceback.print_exc()
