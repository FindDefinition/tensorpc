import asyncio
import collections
import collections.abc
import dataclasses
import time
import traceback
from typing import Any, Dict, List, Optional

import tensorpc
from tensorpc.core import marker
from tensorpc.core.asyncclient import (AsyncRemoteManager,
                                       simple_chunk_call_async)
from tensorpc.core.serviceunit import ServiceEventType
from tensorpc.core.tree_id import UniqueTreeId
from tensorpc.flow.components.mui import FlexBox
from tensorpc.flow.constants import TENSORPC_APP_ROOT_COMP
from tensorpc.flow.core.appcore import ALL_OBSERVED_FUNCTIONS, AppSpecialEventType, enter_app_context
from tensorpc.flow.core.component import (AppEvent, AppEventType,
                                          FrontendEventType, LayoutEvent,
                                          UIEvent, UpdateComponentsEvent,
                                          patch_uid_keys_with_prefix,
                                          patch_uid_list_with_prefix)
from tensorpc.flow.core.reload import AppReloadManager, FlowSpecialMethods
from tensorpc.flow.coretypes import split_unique_node_id
from tensorpc.flow.flowapp.app import App, EditableApp
from tensorpc.flow.serv_names import serv_names


@dataclasses.dataclass
class MountedAppMeta:
    node_uid: str
    url: str
    port: int
    key: str
    prefixes: List[str]

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


def patch_unique_id(data: Any, prefixes: List[str]):
    # can't use abc.Sequence because string is sequence too.
    if isinstance(data, list):
        new_data = []
        for i in range(len(data)):
            d = data[i]
            if isinstance(d, UniqueTreeId):
                d = d.copy()
                d.set_parts_inplace(prefixes + d.parts)
            else:
                d = patch_unique_id(d, prefixes)
            new_data.append(d)
        return new_data
    elif isinstance(data, tuple):
        new_data = []
        for i in range(len(data)):
            d = data[i]
            if isinstance(d, UniqueTreeId):
                d = d.copy()
                d.set_parts_inplace(prefixes + d.parts)
            else:
                d = patch_unique_id(d, prefixes)
            new_data.append(d)
        return tuple(new_data)
    elif isinstance(data, collections.abc.Mapping):
        new_data = {}
        for k, d in data.items():
            if isinstance(d, UniqueTreeId):
                d = d.copy()
                d.set_parts_inplace(prefixes + d.parts)
            else:
                d = patch_unique_id(d, prefixes)
            new_data[k] = d
        return new_data
    elif isinstance(data, UniqueTreeId):
        # data.parts[1:]: remote the ROOT part
        data = data.copy()
        data.set_parts_inplace(prefixes + data.parts)
        return data
    else:
        return data


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
            app.root._attach(UniqueTreeId.from_parts([TENSORPC_APP_ROOT_COMP]),
                             app._flow_app_comp_core)
        send_loop_task = asyncio.create_task(self._send_loop(app_obj))
        app.app_initialize()
        await app.app_initialize_async()
        app_obj.send_loop_task = send_loop_task
        self._app_objs[key] = app_obj

    def get_layout_root_and_app_by_key(self, key: str):
        return self._app_objs[key].app.root, self._app_objs[key].app

    async def mount_app(self, node_uid: str, key: str, url: str, port: int,
                        prefixes: List[str]):
        print("MOUNT", key)
        assert key in self._app_objs
        app_obj = self._app_objs[key]
        #
        if app_obj.mounted_app_meta is not None:
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
        app_obj.mounted_app_meta = MountedAppMeta(node_uid, url, port, key,
                                                  prefixes)
        app_obj.mount_ev.set()
        app_obj.app.app_storage.set_remote_grpc_url(
            app_obj.mounted_app_meta.url_with_port)
        gid, nid = split_unique_node_id(node_uid)
        app_obj.app.app_storage.set_graph_node_id(gid, nid)
        with enter_app_context(app_obj.app):
            await app_obj.app._flowapp_special_eemitter.emit_async(AppSpecialEventType.RemoteCompMount, app_obj.mounted_app_meta)

        # app_obj.send_loop_task = send_loop_task

    async def unmount_app(self, key: str):
        print("UNMOUNT", key)
        assert key in self._app_objs
        app_obj = self._app_objs[key]
        with enter_app_context(app_obj.app):
            await app_obj.app._flowapp_special_eemitter.emit_async(AppSpecialEventType.RemoteCompUnmount, None)

        if app_obj.mounted_app_meta is not None:
            app_obj.mounted_app_meta = None
        app_obj.mount_ev.clear()
        app_obj.app.app_storage.set_remote_grpc_url(None)
        # app_obj.shutdown_ev.set()
        # if app_obj.send_loop_task is not None:
        #     await app_obj.send_loop_task

    def get_layout_dict(self, key: str, prefixes: List[str]):
        assert key in self._app_objs
        app_obj = self._app_objs[key]
        lay = app_obj.app._get_app_layout()
        root_uid = app_obj.app.root._flow_uid
        assert root_uid is not None
        layout_dict = lay["layout"]
        layout_dict = patch_uid_keys_with_prefix(layout_dict, prefixes)
        for k, v in layout_dict.items():
            layout_dict[k] = patch_unique_id(v, prefixes)
        lay["layout"] = layout_dict
        # print("APP layout_dict", layout_dict)
        lay["remoteRootUid"] = UniqueTreeId.from_parts(
            prefixes + root_uid.parts).uid_encoded
        return lay

    async def _send_loop(self, app_obj: AppObject):
        # mount_meta = app_obj.mounted_app_meta
        app = app_obj.app
        retry_cnt = 2
        # assert mount_meta is not None
        shut_task = asyncio.create_task(app_obj.shutdown_ev.wait(), name="shutdown")
        # grpc_url = mount_meta.url_with_port
        # async with tensorpc.AsyncRemoteManager(grpc_url) as robj:
        send_task = asyncio.create_task(app_obj.send_loop_queue.get(), name="wait for queue")
        wait_for_mount_task = asyncio.create_task(app_obj.mount_ev.wait(), name="wait for mount")
        robj: Optional[tensorpc.AsyncRemoteManager] = None
        wait_tasks: List[asyncio.Task] = [
            shut_task, send_task, wait_for_mount_task
        ]
        while True:
            (done,
             pending) = await asyncio.wait(wait_tasks,
                                           return_when=asyncio.FIRST_COMPLETED)
            if shut_task in done:
                break
            if wait_for_mount_task in done:
                assert app_obj.mounted_app_meta is not None, "shouldn't happen"
                # print("ROBJ", app_obj.mounted_app_meta.url_with_port)
                robj = tensorpc.AsyncRemoteManager(
                    app_obj.mounted_app_meta.url_with_port)
                wait_tasks: List[asyncio.Task] = [shut_task, send_task]
                if send_task not in done:
                    continue
            ev: AppEvent = send_task.result()
            if ev.is_loopback:
                raise NotImplementedError("loopback not implemented")
                for k, v in ev.type_to_event.items():
                    if k == AppEventType.UIEvent:
                        assert isinstance(v, UIEvent)
                        await app._handle_event_with_ctx(v)
                send_task = asyncio.create_task(app_obj.send_loop_queue.get())
                wait_for_mount_task = asyncio.create_task(
                    app_obj.mount_ev.wait())
                wait_tasks: List[asyncio.Task] = [
                    shut_task, send_task, wait_for_mount_task
                ]
                continue
            ts = time.time()
            # print(app_obj.mounted_app_meta, robj)
            if app_obj.mounted_app_meta is None:
                # must clear robj here because app is unmounted
                robj = None
            if app_obj.mounted_app_meta is None or robj is None:
                # we got app event, but
                # remote component isn't mounted, ignore app event
                send_task = asyncio.create_task(app_obj.send_loop_queue.get())
                wait_for_mount_task = asyncio.create_task(
                    app_obj.mount_ev.wait())
                wait_tasks: List[asyncio.Task] = [
                    shut_task, wait_for_mount_task, send_task
                ]
                if ev.sent_event is not None:
                    ev.sent_event.set()
                continue
            # print("WHY", app_obj.mounted_app_meta, robj)

            # assign uid here.
            # print("WTF", ev.to_dict(), app_obj.mounted_app_meta.node_uid)
            ev.uid = app_obj.mounted_app_meta.node_uid
            send_task = asyncio.create_task(app_obj.send_loop_queue.get())
            wait_tasks: List[asyncio.Task] = [shut_task, send_task]
            succeed = False
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
                except Exception as e:
                    traceback.print_exc()
            if not succeed:
                print("Master disconnect for too long, unmount")
                if robj is not None:
                    await robj.close()
                    robj = None
                await self.unmount_app(app_obj.mounted_app_meta.key)
                wait_for_mount_task = asyncio.create_task(
                    app_obj.mount_ev.wait())
                wait_tasks: List[asyncio.Task] = [
                    shut_task, wait_for_mount_task, send_task
                ]
                continue
            # trigger sent event here.
            if ev.sent_event is not None:
                ev.sent_event.set()
        app_obj.send_loop_task = None
        app_obj.mounted_app_meta = None
        if robj is not None:
            await robj.close()

    async def _send_grpc_event_large(self,
                                     ev: AppEvent,
                                     robj: tensorpc.AsyncRemoteManager,
                                     prefixes: List[str],
                                     app: App,
                                     timeout: Optional[int] = None):
        ev._remote_prefixes = prefixes
        ev.patch_keys_prefix_inplace(prefixes)
        for ui_ev in ev.type_to_event.values():
            if isinstance(ui_ev, UpdateComponentsEvent):
                comp_dict = app.root._get_uid_encoded_to_comp_dict()
                ui_ev.remote_component_all_childs = patch_uid_list_with_prefix(
                    list(comp_dict.keys()), prefixes)
        ev_dict = ev.to_dict()
        ev_dict = patch_unique_id(ev_dict, prefixes)
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
            ev.unpatch_keys_prefix_inplace(prefixes)
            return await app_obj.app._handle_event_with_ctx(ev, is_sync)

    @marker.mark_server_event(event_type=ServiceEventType.Exit)
    async def on_exit(self):
        for app_obj in self._app_objs.values():
            try:
                if app_obj.mounted_app_meta is not None:
                    prefixes = app_obj.mounted_app_meta.prefixes
                    await simple_chunk_call_async(
                        app_obj.mounted_app_meta.url_with_port,
                        serv_names.APP_REMOTE_COMP_SHUTDOWN, prefixes)
            except:
                traceback.print_exc()
            app_obj.shutdown_ev.set()
            if app_obj.send_loop_task is not None:
                await app_obj.send_loop_task
            try:
                app_obj.app.app_terminate()
                await app_obj.app.app_terminate_async()
            except:
                traceback.print_exc()