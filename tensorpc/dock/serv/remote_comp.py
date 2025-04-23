import asyncio
import collections
import collections.abc
import dataclasses
import inspect
import io
import os
from pathlib import Path
import time
import traceback
from typing import Any, Dict, List, Optional

import tensorpc
from tensorpc.core import marker
from tensorpc.core.asyncclient import (AsyncRemoteManager,
                                       simple_chunk_call_async)
from tensorpc.core.asynctools import cancel_task
from tensorpc.core.defs import FileDesc, FileResource, FileResourceRequest
from tensorpc.core.serviceunit import ServiceEventType
from tensorpc.core.tree_id import UniqueTreeId, UniqueTreeIdForComp
from tensorpc.dock.components.mui import FlexBox
from tensorpc.dock.constants import TENSORPC_APP_ROOT_COMP
from tensorpc.dock.core.appcore import ALL_OBSERVED_FUNCTIONS, AppSpecialEventType, RemoteCompEvent, enter_app_context
from tensorpc.dock.core.component import (AppEvent, AppEventType,
                                          FrontendEventType, LayoutEvent, RemoteComponentBase,
                                          UIEvent, UpdateComponentsEvent,
                                          patch_uid_keys_with_prefix,
                                          patch_uid_list_with_prefix,
                                          patch_unique_id)
from tensorpc.dock.core.reload import AppReloadManager, FlowSpecialMethods
from tensorpc.dock.coretypes import split_unique_node_id
from tensorpc.dock.flowapp.app import App, EditableApp
from tensorpc.dock.serv.common import handle_file_resource, REMOTE_APP_LOGGER
from tensorpc.dock.serv_names import serv_names
from urllib import parse


@dataclasses.dataclass
class MountedAppMeta:
    url: str
    port: int
    key: str
    prefixes: List[str]
    remote_gen_queue: Optional[asyncio.Queue] = None

    @property
    def url_with_port(self):
        return f"{self.url}:{self.port}"


@dataclasses.dataclass
class AppObject:
    app: App
    send_loop_queue: "asyncio.Queue[AppEvent]"
    shutdown_ev: asyncio.Event
    mount_ev: asyncio.Event
    send_loop_task: Optional[asyncio.Task] = None
    obj: Any = None
    mounted_app_meta: Optional[MountedAppMeta] = None


_PATCH_IN_REMOTE = False

class RemoteComponentService:

    def __init__(self) -> None:
        self._app_objs: Dict[str, AppObject] = {}
        self.shutdown_ev = asyncio.Event()
        self.shutdown_ev.clear()

    async def remove_layout_object(self, key: str):
        app_obj = self._app_objs[key]
        app_obj.shutdown_ev.set()
        app_obj.mounted_app_meta = None
        app_obj.mount_ev.clear()
        if app_obj.send_loop_task is not None:
            await app_obj.send_loop_task
        app_obj.app.app_terminate()
        await app_obj.app.app_terminate_async()
        app_obj.app.app_storage.set_remote_grpc_url(None)

    def has_layout_object(self, key: str):
        return key in self._app_objs

    async def set_layout_object(self, key: str, obj):
        reload_mgr = AppReloadManager(ALL_OBSERVED_FUNCTIONS)
        if isinstance(obj, FlexBox):
            # external root
            external_root = obj
            app: App = EditableApp(external_root=external_root,
                                   reload_manager=reload_mgr,
                                   is_remote_component=True)
        else:
            # other object, must declare a tensorpc_flow_layout
            # external_root = flex_wrapper(obj)
            app: App = EditableApp(external_wrapped_obj=obj,
                                   reload_manager=reload_mgr,
                                   is_remote_component=True)
            app._app_force_use_layout_function()
        app._flow_app_comp_core.reload_mgr = reload_mgr
        send_loop_queue: "asyncio.Queue[AppEvent]" = app._queue
        app_obj = AppObject(app, send_loop_queue, asyncio.Event(),
                            asyncio.Event())
        app_obj.shutdown_ev.clear()

        if not isinstance(obj, FlexBox):
            app_obj.obj = obj

        if app._force_special_layout_method:
            layout_created = False
            metas = reload_mgr.query_type_method_meta(type(obj),
                                                      no_code=True,
                                                      include_base=True)
            special_methods = FlowSpecialMethods(metas)
            special_methods.bind(obj)
            if special_methods.create_layout is not None:
                await app._app_run_layout_function(
                    decorator_fn=special_methods.create_layout.get_binded_fn())
                layout_created = True
            if not layout_created:
                await app._app_run_layout_function()
        else:
            app.root._attach(UniqueTreeIdForComp.from_parts([TENSORPC_APP_ROOT_COMP]),
                             app._flow_app_comp_core)
        send_loop_task = asyncio.create_task(self._send_loop(app_obj), name=f"send loop {key}")
        app.app_initialize()
        await app.app_initialize_async()
        app_obj.send_loop_task = send_loop_task
        self._app_objs[key] = app_obj

    def get_layout_root_and_app_by_key(self, key: str):
        if key not in self._app_objs:
            raise KeyError(f"key {key} not found, available keys: {list(self._app_objs.keys())}")
        return self._app_objs[key].app.root, self._app_objs[key].app

    async def mount_app(self, key: str, url: str, port: int,
                        prefixes: List[str], remote_gen_queue: Optional[asyncio.Queue] = None):
        assert key in self._app_objs, key
        app_obj = self._app_objs[key]
        if app_obj.mounted_app_meta is not None:
            if app_obj.mounted_app_meta.remote_gen_queue is not None:
                # mount app via remote generator (client -> remote comp server)
                # server can't access client url.
                raise ValueError("app is already mounted")
            # mount app that support server -> client connection
            # check is same
            if (url == app_obj.mounted_app_meta.url
                and port == app_obj.mounted_app_meta.port):
                return
            # check mounted app is alive
            try:
                async with AsyncRemoteManager(
                        app_obj.mounted_app_meta.url_with_port) as robj:
                    await robj.health_check()
            except:
                # mounted app dead. use new one
                traceback.print_exc()
                # with app_obj.app._enter
                await self.unmount_app(app_obj.mounted_app_meta.key)

        assert app_obj.mounted_app_meta is None, "already mounted"
        app_obj.mounted_app_meta = MountedAppMeta(url, port, key,
                                                  prefixes, remote_gen_queue)
        app_obj.mount_ev.set()
        app_obj.app.app_storage.set_remote_grpc_url(
            app_obj.mounted_app_meta.url_with_port)
        # gid, nid = split_unique_node_id(node_uid)
        # app_obj.app.app_storage.set_graph_node_id(gid, nid)
        with enter_app_context(app_obj.app):
            await app_obj.app._flowapp_special_eemitter.emit_async(AppSpecialEventType.RemoteCompMount, app_obj.mounted_app_meta)

        # app_obj.send_loop_task = send_loop_task

    async def unmount_app(self, key: str, is_local_call: bool = False):
        assert key in self._app_objs
        app_obj = self._app_objs[key]
        if not is_local_call:
            if app_obj.mounted_app_meta is not None and app_obj.mounted_app_meta.remote_gen_queue is not None:
                app_obj.shutdown_ev.set()
                with enter_app_context(app_obj.app):
                    await app_obj.app._flowapp_special_eemitter.emit_async(AppSpecialEventType.RemoteCompUnmount, None)
                return
        if not is_local_call:
            REMOTE_APP_LOGGER.warning("Unmount remote comp %s", key)
            # raise ValueError("app is mounted via remote generator, you can't call unmount")
        with enter_app_context(app_obj.app):
            await app_obj.app._flowapp_special_eemitter.emit_async(AppSpecialEventType.RemoteCompUnmount, None)

        if app_obj.mounted_app_meta is not None:
            app_obj.mounted_app_meta = None
        app_obj.mount_ev.clear()
        app_obj.app.app_storage.set_remote_grpc_url(None)
        # app_obj.shutdown_ev.set()
        # if app_obj.send_loop_task is not None:
        #     await app_obj.send_loop_task

    async def get_layout_dict(self, key: str, prefixes: List[str]):
        assert key in self._app_objs
        app_obj = self._app_objs[key]
        lay = await app_obj.app._get_app_layout()
        root_uid = app_obj.app.root._flow_uid
        assert root_uid is not None
        layout_dict = lay["layout"]
        if _PATCH_IN_REMOTE:
            layout_dict = patch_uid_keys_with_prefix(layout_dict, prefixes)
            for k, v in layout_dict.items():
                layout_dict[k] = patch_unique_id(v, prefixes)
        lay["layout"] = layout_dict
        # # print("APP layout_dict", layout_dict)
        if _PATCH_IN_REMOTE:
            lay["remoteRootUid"] = UniqueTreeIdForComp.from_parts(
                prefixes + root_uid.parts).uid_encoded
        else:
            lay["remoteRootUid"] = UniqueTreeIdForComp.from_parts(
                root_uid.parts).uid_encoded
        return lay

    async def mount_app_generator(self, key: str,
                        prefixes: List[str], url: str = "", port: int = -1):
        REMOTE_APP_LOGGER.warning("Mount remote comp %s (Generator)", key)
        assert key in self._app_objs, key
        app_obj = self._app_objs[key]
        try:
            queue = asyncio.Queue()
            await self.mount_app(key, url, port, prefixes, queue)
            shutdown_task = asyncio.create_task(app_obj.shutdown_ev.wait(), name="shutdown")
            wait_queue_task = asyncio.create_task(queue.get(), name="wait for queue")
            yield await self.get_layout_dict(key, prefixes)
            while True:
                try:
                    (done,
                    pending) = await asyncio.wait([shutdown_task, wait_queue_task],
                                                return_when=asyncio.FIRST_COMPLETED)
                    if shutdown_task in done:
                        for task in pending:
                            await cancel_task(task)
                        break
                    try:

                        ev = wait_queue_task.result()
                        if isinstance(ev, AppEvent):
                            ev_dict = self._patch_app_event(ev, prefixes, app_obj.app)
                            yield ev_dict
                        elif isinstance(ev, RemoteCompEvent):
                            yield ev 
                        wait_queue_task = asyncio.create_task(queue.get(), name="wait for queue")
                    except StopAsyncIteration:
                        for task in pending:
                            await cancel_task(task)
                        break
                except asyncio.CancelledError:
                    REMOTE_APP_LOGGER.warning("Remote comp %s (Generator) cancelled", key)
                    await cancel_task(wait_queue_task)
                    await cancel_task(shutdown_task)
                    break
        except:
            traceback.print_exc()
            raise 
        finally:
            REMOTE_APP_LOGGER.warning("Unmount remote comp %s (Generator)", key)
            await self.unmount_app(key, is_local_call=True)

    async def _send_loop(self, app_obj: AppObject):
        # mount_meta = app_obj.mounted_app_meta
        app = app_obj.app
        retry_cnt = 2
        # assert mount_meta is not None
        shut_task = asyncio.create_task(app_obj.shutdown_ev.wait(), name="shutdown")
        # grpc_url = mount_meta.url_with_port
        # async with tensorpc.AsyncRemoteManager(grpc_url) as robj:
        send_task = asyncio.create_task(app_obj.send_loop_queue.get(), name="wait for queue")
        wait_for_mount_task = asyncio.create_task(app_obj.mount_ev.wait(), name=f"wait for mount-{os.getpid()}")
        robj: Optional[tensorpc.AsyncRemoteManager] = None
        wait_tasks: List[asyncio.Task] = [
            shut_task, send_task, wait_for_mount_task
        ]
        while True:
            (done,
             pending) = await asyncio.wait(wait_tasks,
                                           return_when=asyncio.FIRST_COMPLETED)
            if shut_task in done:
                for task in pending:
                    await cancel_task(task)
                # print("!!!", "send loop closed by event", last_key, os.getpid())
                break
            if wait_for_mount_task in done:
                assert app_obj.mounted_app_meta is not None, "shouldn't happen"
                # print("ROBJ", app_obj.mounted_app_meta.url_with_port)
                if app_obj.mounted_app_meta.remote_gen_queue is None:
                    robj = tensorpc.AsyncRemoteManager(
                        app_obj.mounted_app_meta.url_with_port)
                wait_tasks: List[asyncio.Task] = [shut_task, send_task]
                if send_task not in done:
                    continue
            ev: AppEvent = send_task.result()
            if ev.is_loopback:
                raise NotImplementedError("loopback not implemented")
            ts = time.time()
            # print(app_obj.mounted_app_meta, robj)
            if app_obj.mounted_app_meta is None:
                # must clear robj here because app is unmounted
                robj = None
            if app_obj.mounted_app_meta is None or (robj is None and app_obj.mounted_app_meta.remote_gen_queue is None):
                # we got app event, but
                # remote component isn't mounted, ignore app event
                send_task = asyncio.create_task(app_obj.send_loop_queue.get(), name="wait for queue")
                if not wait_for_mount_task.done():
                    await cancel_task(wait_for_mount_task)
                wait_for_mount_task = asyncio.create_task(
                    app_obj.mount_ev.wait(), name=f"wait for mount-{os.getpid()}-{time.time_ns()}-1")
                wait_tasks: List[asyncio.Task] = [
                    shut_task, wait_for_mount_task, send_task
                ]
                if ev.sent_event is not None:
                    ev.sent_event.set()
                continue
            # print("WHY", app_obj.mounted_app_meta, robj)

            # assign uid here.
            # print("WTF", ev.to_dict(), app_obj.mounted_app_meta.node_uid)
            # ev.uid = app_obj.mounted_app_meta.node_uid
            send_task = asyncio.create_task(app_obj.send_loop_queue.get(), name="wait for queue")
            wait_tasks: List[asyncio.Task] = [shut_task, send_task]
            succeed = False
            # when user use additional event such as RemoteCompEvent, regular app event may be empty.
            if ev.type_to_event:
                if robj is None:
                    assert app_obj.mounted_app_meta.remote_gen_queue is not None
                    await app_obj.mounted_app_meta.remote_gen_queue.put(ev)
                else:
                    # retry
                    for _ in range(retry_cnt):
                        try:
                            await self._send_grpc_event_large(
                                ev,
                                robj,
                                app_obj.mounted_app_meta.prefixes,
                                app,
                                timeout=10)
                            succeed = True
                            break
                        except BaseException as e:
                            traceback.print_exc()
                    if not succeed:
                        print("Master disconnect for too long, unmount")
                        if robj is not None:
                            await robj.close()
                            robj = None
                        await self.unmount_app(app_obj.mounted_app_meta.key)
                        wait_for_mount_task = asyncio.create_task(
                            app_obj.mount_ev.wait(), name=f"wait for mount-{os.getpid()}")
                        wait_tasks: List[asyncio.Task] = [
                            shut_task, wait_for_mount_task, send_task
                        ]
                        continue
            # trigger sent event here.
            if ev.sent_event is not None:
                ev.sent_event.set()
            # handle additional events
            for addi_ev in ev._additional_events:
                if isinstance(addi_ev, RemoteCompEvent):
                    if robj is None:
                        assert app_obj.mounted_app_meta.remote_gen_queue is not None
                        await app_obj.mounted_app_meta.remote_gen_queue.put(addi_ev)
                    else:
                        try:
                            await robj.remote_call(serv_names.APP_RUN_REMOTE_COMP_EVENT, addi_ev.key, addi_ev, rpc_timeout=1)
                        except BaseException as e:
                            traceback.print_exc()

        app_obj.send_loop_task = None
        app_obj.mounted_app_meta = None
        if robj is not None:
            await robj.close()
        # print("!!!", "send loop closed", last_key, os.getpid())

    async def handle_msg_from_remote_comp(self, key: str, rpc_key: str, event: RemoteCompEvent):
        app_obj = self._app_objs[key]
        return await app_obj.app.handle_msg_from_remote_comp(rpc_key, event)

    def _patch_app_event(self, ev: AppEvent, prefixes: List[str], app: App,):
        if _PATCH_IN_REMOTE:
            ev._remote_prefixes = prefixes
            ev.patch_keys_prefix_inplace(prefixes)
        for ui_ev in ev.type_to_event.values():
            if isinstance(ui_ev, UpdateComponentsEvent):
                comp_dict = app.root._get_uid_encoded_to_comp_dict()
                if _PATCH_IN_REMOTE:
                    uids = list(comp_dict.keys())
                    for v in comp_dict.values():
                        if isinstance(v, RemoteComponentBase):
                            uids.extend(v._cur_child_uids)
                    ui_ev.remote_component_all_childs = patch_uid_list_with_prefix(
                        list(comp_dict.keys()), prefixes)
                else:
                    ui_ev.remote_component_all_childs = list(comp_dict.keys())
        ev_dict = ev.to_dict()
        if _PATCH_IN_REMOTE:
            ev_dict = patch_unique_id(ev_dict, prefixes)
        return ev_dict

    async def _send_grpc_event_large(self,
                                     ev: AppEvent,
                                     robj: tensorpc.AsyncRemoteManager,
                                     prefixes: List[str],
                                     app: App,
                                     timeout: Optional[int] = None):
        
        ev_dict = self._patch_app_event(ev, prefixes, app)
        # print("APP EVENT", ev_dict)
        return await robj.chunked_remote_call(
            serv_names.APP_RELAY_APP_EVENT_FROM_REMOTE,
            ev_dict,
            rpc_timeout=timeout)

    async def run_single_event(self,
                               key: str,
                               type,
                               data,
                               is_sync: bool = False):
        """is_sync: only used for ui event.
        """
        app_obj = self._app_objs[key]
        assert app_obj.mounted_app_meta is not None
        prefixes = app_obj.mounted_app_meta.prefixes
        if type == AppEventType.UIEvent.value:
            ev = UIEvent.from_dict(data)
            if _PATCH_IN_REMOTE:
                ev.unpatch_keys_prefix_inplace(prefixes)
            return await app_obj.app._handle_event_with_ctx(ev, is_sync)

    async def handle_simple_rpc(self, key: str, event: str, *args, **kwargs):
        app_obj = self._app_objs[key]
        return await app_obj.app._flowapp_simple_rpc_handlers.call_event(event, *args, **kwargs)

    @marker.mark_server_event(event_type=ServiceEventType.Exit)
    async def on_exit(self):
        for app_obj in self._app_objs.values():
            try:
                if app_obj.mounted_app_meta is not None and app_obj.mounted_app_meta.remote_gen_queue is None:
                    prefixes = app_obj.mounted_app_meta.prefixes
                    await simple_chunk_call_async(
                        app_obj.mounted_app_meta.url_with_port,
                        serv_names.APP_REMOTE_COMP_SHUTDOWN, prefixes)
            except:
                traceback.print_exc()
            # send loop must be shutdown after all ui unmount.
            try:
                app_obj.app.app_terminate()
                await app_obj.app.app_terminate_async()
            except:
                traceback.print_exc()
            app_obj.shutdown_ev.set()
            if app_obj.send_loop_task is not None:
                await app_obj.send_loop_task

    def _get_file_path_stat(
        self, path: str
    ) -> os.stat_result:
        """Return the file path, stat result, and gzip status.

        This method should be called from a thread executor
        since it calls os.stat which may block.
        """
        return Path(path).stat()

    async def get_file_metadata(self, key: str, file_key: str, comp_uid: Optional[str] = None):
        app_obj = self._app_objs[key]
        assert app_obj.mounted_app_meta is not None
        if comp_uid is not None:
            comp = app_obj.app.root._get_comp_by_uid(comp_uid)
            if isinstance(comp, RemoteComponentBase):
                return await comp.get_file_metadata(file_key, comp_uid)
        url = parse.urlparse(file_key)
        base = url.path
        file_key_qparams = parse.parse_qs(url.query)
        # if comp_uid is not None:
        #     comp = self.app.root._get_comp_by_uid(comp_uid)
        #     if isinstance(comp, RemoteComponentBase):
        #         return await comp.get_file_metadata(file_key)

        if app_obj.app._flowapp_file_resource_handlers.has_event_handler(base):
            # we only use first value
            if len(file_key_qparams) > 0:
                file_key_qparams = {
                    k: v[0]
                    for k, v in file_key_qparams.items()
                }
            else:
                file_key_qparams = {}
            try:
                res = app_obj.app._flowapp_file_resource_handlers.call_event(base, FileResourceRequest(base, True, None, file_key_qparams))
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

    async def get_file(self, key: str, file_key: str, offset: int, count: Optional[int] = None, chunk_size=2**16):
        app_obj = self._app_objs[key]
        assert app_obj.mounted_app_meta is not None
        url = parse.urlparse(file_key)
        base = url.path
        file_key_qparams = parse.parse_qs(url.query)

        if app_obj.app._flowapp_file_resource_handlers.has_event_handler(base):
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
                handler = app_obj.app._flowapp_file_resource_handlers.get_event_handler(base).handler
                async for chunk in handle_file_resource(req, handler, chunk_size, count):
                    yield chunk
            except GeneratorExit:
                return 
            except:
                traceback.print_exc()
                raise
        else:
            raise KeyError(f"File key {file_key} not found.")
