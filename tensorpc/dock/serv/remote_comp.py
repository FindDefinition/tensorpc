import asyncio
import dataclasses
import inspect
import io
import os
from pathlib import Path
import time
import traceback
from typing import Any, Coroutine, Dict, List, Optional, Union
import uuid

import tensorpc
from tensorpc.core import marker
from tensorpc.core import prim
from tensorpc.core.asyncclient import (AsyncRemoteManager,
                                       simple_chunk_call_async)
from tensorpc.core.asynctools import cancel_task
from tensorpc.core.defs import FileDesc, FileResource, FileResourceRequest
from tensorpc.core.distributed.ftgroup import FTGroupConfig, FTStateBase, FaultToleranceRPCGroup
from tensorpc.core.prim import get_server_exposed_props, get_server_meta
from tensorpc.core.server_core import ServerDistributedMeta
from tensorpc.core.serviceunit import ReloadableDynamicClass, ServiceEventType, ServiceUnit
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
from tensorpc.dock.serv.common import handle_file_resource
from tensorpc.dock.loggers import REMOTE_APP_SERV_LOGGER
from tensorpc.dock.serv_names import serv_names
from urllib import parse

from tensorpc.utils import get_service_key_by_type


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

"""
Distributed test cmd:
torchrun --nproc_per_node=2 -m tensorpc.serve --port 50051,50052 --distributed_mode torch_nccl \
    --serv_config_json '{"tensorpc.dock.serv.remote_comp::RemoteComponentService": {"init_apps": {"dev": "tensorpc.apps.sample.dist::TorchDistributedApp"}}}'
torchrun --nproc_per_node=2 -m tensorpc.dock.serve_remote --port 50051,50052 --distributed_mode torch_nccl \
    tensorpc.apps.sample.dist::TorchDistributedApp

"""

class RemoteComponentService:
    def __init__(self, init_apps: Optional[dict[str, str]] = None) -> None:
        self._app_objs: Dict[str, AppObject] = {}
        self.shutdown_ev = asyncio.Event()
        self.shutdown_ev.clear()
        self._ft_group: Optional[FaultToleranceRPCGroup] = None
        self._dist_urls: Optional[List[str]] = None
        self._is_master = True 
        self._rank_prefix = ""
        self._init_apps = init_apps
        self._dist_meta: Optional[ServerDistributedMeta] = None

        self._dist_call_tasks_dict: dict[str, asyncio.Task] = {}

    @marker.mark_server_event(event_type=marker.ServiceEventType.AfterServerStart)
    async def _after_start(self):
        if self._ft_group is not None:
            await self._ft_group.check_connection()

    @marker.mark_server_event(event_type=marker.ServiceEventType.Init)
    async def init(self):
        dist_meta = get_server_exposed_props().dist_meta
        if dist_meta is not None:
            self._dist_meta = dist_meta
            dist_meta.init_backend()
            rank = dist_meta.rank 
            self._rank_prefix = f"[{rank}]"
            port = get_server_meta().port 
            url = f"localhost:{port}"
            dist_urls = dist_meta.all_gather_object(url)
            set_worker_state_serv_key = get_service_key_by_type(RemoteComponentService, RemoteComponentService.set_worker_state_on_master.__name__)
            self._ft_group = FaultToleranceRPCGroup(prim.get_async_shutdown_event(), rank, dist_meta.world_size, "localhost", port,
                set_worker_state_serv_key, dist_urls,
                cfg=FTGroupConfig(disconnect_total_retry=5))
            self._is_master = (rank == 0)
            self._ft_group.init()
        if self._init_apps is not None:
            for key, cls_path in self._init_apps.items():
                await self._set_layout_object_internal(key, cls_path, raise_on_fail=True)

    async def set_worker_state_on_master(self, state: FTStateBase):
        assert self._ft_group is not None 
        return await self._ft_group.set_worker_state(state)

    async def remove_layout_object(self, key: str):
        assert prim.is_loopback_call()
        return await self._remove_layout_object_internal(key)

    async def _remove_layout_object_internal(self, key: str):
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

    async def set_layout_object(self, key: str, obj: Union[str, FlexBox, Any], **app_create_kwargs):
        assert prim.is_loopback_call()
        return await self._set_layout_object_internal(key, obj, raise_on_fail=True, **app_create_kwargs)
    
    async def _set_layout_object_internal(self, key: str, obj: Union[str, FlexBox, Any], raise_on_fail: bool, **app_create_kwargs):
        reload_mgr = AppReloadManager(ALL_OBSERVED_FUNCTIONS)
        disable_auto_reload = True
        dynamic_app_cls: Optional[ReloadableDynamicClass] = None
        app_su: Optional[ServiceUnit] = None
        if isinstance(obj, str):
            # enable auto reload for regular app.
            REMOTE_APP_SERV_LOGGER.warning("%sCreate remote comp app %s from class %s", self._rank_prefix, key, obj)
            dynamic_app_cls = ReloadableDynamicClass(obj, reload_mgr)
            static_creator = dynamic_app_cls.get_object_creator_if_exists()
            module_id = obj

            if static_creator is not None:
                obj = static_creator()
            else:
                obj = dynamic_app_cls.obj_type(**app_create_kwargs)
            disable_auto_reload = False
            app_su = ServiceUnit(module_id, {})
            app_su.init_service(obj)

        else:
            REMOTE_APP_SERV_LOGGER.warning("%sCreate remote comp app %s", self._rank_prefix, key)

        if isinstance(obj, FlexBox):
            # external root
            external_root = obj
            app: App = EditableApp(external_root=external_root,
                                   reload_manager=reload_mgr,
                                   disable_auto_reload=disable_auto_reload,
                                   is_remote_comp=True)
        else:
            # other object, must declare a tensorpc_flow_layout
            # external_root = flex_wrapper(obj)
            app: App = EditableApp(external_wrapped_obj=obj,
                                   reload_manager=reload_mgr,
                                   disable_auto_reload=disable_auto_reload,
                                   is_remote_comp=True)
            app._app_force_use_layout_function()
        app._flow_app_comp_core.reload_mgr = reload_mgr
        if dynamic_app_cls is not None:
            app._app_dynamic_cls = dynamic_app_cls
        if app_su is not None:
            app._app_service_unit = app_su
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
                    decorator_fn=special_methods.create_layout.get_binded_fn(),
                    raise_on_fail=raise_on_fail)
                layout_created = True
            if not layout_created:
                await app._app_run_layout_function(raise_on_fail=raise_on_fail)
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
        REMOTE_APP_SERV_LOGGER.warning("%sMount remote comp %s to %s", self._rank_prefix, key, app_obj.mounted_app_meta.url_with_port)
        # event (remote mount) must be handled async, because it may send event to frontend, 
        # the app send loop isn't ready before mount_app returns.
        asyncio.create_task(self._send_remote_mount(app_obj), name="send remote mount")
        # with enter_app_context(app_obj.app):
        #     await app_obj.app._flowapp_special_eemitter.emit_async(AppSpecialEventType.RemoteCompMount, app_obj.mounted_app_meta)

    async def _send_remote_mount(self, app_obj: AppObject):
        with enter_app_context(app_obj.app):
            await app_obj.app._flowapp_special_eemitter.emit_async(AppSpecialEventType.RemoteCompMount, app_obj.mounted_app_meta)

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
            REMOTE_APP_SERV_LOGGER.warning("Unmount remote comp %s", key)
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
        lay["layout"] = layout_dict
        # # print("APP layout_dict", layout_dict)
        lay["remoteRootUid"] = UniqueTreeIdForComp.from_parts(
            root_uid.parts).uid_encoded
        return lay

    async def mount_app_generator(self, key: str,
                        prefixes: List[str], url: str = "", port: int = -1):
        REMOTE_APP_SERV_LOGGER.warning("Start remote comp %s (Generator)", key)
        assert key in self._app_objs, key
        app_obj = self._app_objs[key]
        assert self._is_master
        try:
            queue = asyncio.Queue()
            if self._ft_group is not None:
                client_coro = await self._run_dist_call_get_coro(
                    key, RemoteComponentService.mount_app.__name__, 
                    key, url, port, prefixes
                )
                await asyncio.gather(client_coro, self.mount_app(key, url, port, prefixes, queue))
            else:
                await self.mount_app(key, url, port, prefixes, queue)
            shutdown_task = asyncio.create_task(app_obj.shutdown_ev.wait(), name="shutdown")
            wait_queue_task = asyncio.create_task(queue.get(), name="wait for queue")
            # REMOTE_APP_SERV_LOGGER.warning("Start remote comp %s (Generator) step 2", key)
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
                            if ev.sent_event is not None:
                                ev.sent_event.set()
                                ev.sent_event = None
                        elif isinstance(ev, RemoteCompEvent):
                            yield ev 
                        wait_queue_task = asyncio.create_task(queue.get(), name="wait for queue")
                    except StopAsyncIteration:
                        for task in pending:
                            await cancel_task(task)
                        break
                except asyncio.CancelledError:
                    REMOTE_APP_SERV_LOGGER.warning("Remote comp %s (Generator) cancelled", key)
                    await cancel_task(wait_queue_task)
                    await cancel_task(shutdown_task)
                    break
        except:
            traceback.print_exc()
            raise 
        finally:
            REMOTE_APP_SERV_LOGGER.warning("Unmount remote comp %s (Generator)", key)
            if self._ft_group is not None:
                client_coro = await self._run_dist_call_get_coro(
                    key, RemoteComponentService.unmount_app.__name__, 
                    key, is_local_call=True
                )
                await asyncio.gather(client_coro, self.unmount_app(key, is_local_call=True))
            else:
                await self.unmount_app(key, is_local_call=True)


    async def _send_loop(self, app_obj: AppObject):
        # assert mount_meta is not None
        shut_task = asyncio.create_task(app_obj.shutdown_ev.wait(), name="shutdown")
        send_task = asyncio.create_task(app_obj.send_loop_queue.get(), name="wait for queue")
        wait_tasks: List[asyncio.Task] = [
            shut_task, send_task
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
            ev: AppEvent = send_task.result()
            if ev.is_loopback:
                raise NotImplementedError("loopback not implemented")
            if app_obj.mounted_app_meta is None:
                # we got app event, but
                # remote component isn't mounted, ignore app event
                send_task = asyncio.create_task(app_obj.send_loop_queue.get(), name="wait for queue")
                wait_tasks: List[asyncio.Task] = [
                    shut_task, send_task
                ]
                if ev.sent_event is not None:
                    ev.sent_event.set()
                continue
            # ev.uid = app_obj.mounted_app_meta.node_uid
            send_task = asyncio.create_task(app_obj.send_loop_queue.get(), name="wait for queue")
            wait_tasks: List[asyncio.Task] = [shut_task, send_task]
            # when user use additional event such as RemoteCompEvent, regular app event may be empty.
            shouldn_t_set_sent_ev = False
            if self._ft_group is None or self._is_master:
                # only master send event to UI
                if ev.type_event_tuple:
                    assert app_obj.mounted_app_meta.remote_gen_queue is not None
                    await app_obj.mounted_app_meta.remote_gen_queue.put(ev)
                    shouldn_t_set_sent_ev = True 
            # trigger sent event here.
            if not shouldn_t_set_sent_ev:
                if ev.sent_event is not None:
                    ev.sent_event.set()
                    ev.sent_event = None
            # handle additional events
            if self._ft_group is None or self._is_master:
                # only master send event to UI
                for addi_ev in ev._additional_events:
                    if isinstance(addi_ev, RemoteCompEvent):
                        assert app_obj.mounted_app_meta.remote_gen_queue is not None
                        await app_obj.mounted_app_meta.remote_gen_queue.put(addi_ev)

        app_obj.send_loop_task = None
        app_obj.mounted_app_meta = None

    async def run_dist_call_create_task(self, method_name: str,
                                       task_uuid: str,
                                       *args,
                                       **kwargs):
        method = getattr(self, method_name)
        task = asyncio.create_task(method(*args, **kwargs), name=f"dist call {method_name}")
        self._dist_call_tasks_dict[task_uuid] = task

    async def run_dist_call_wait_task(self, task_uuid: str):
        assert task_uuid in self._dist_call_tasks_dict, f"handle {task_uuid} not found"
        task = self._dist_call_tasks_dict.pop(task_uuid)
        res = await task
        return res

    async def _run_dist_call_get_coro(self, call_key: str, method_name: str, *args, **kwargs):
        assert self._ft_group is not None 
        task_uuid = f"{call_key}-{uuid.uuid4().hex}"
        await (self._ft_group.master_call_all_client(
            get_service_key_by_type(RemoteComponentService, RemoteComponentService.run_dist_call_create_task.__name__), 
            set(), method_name, task_uuid, *args, **kwargs))
        return self._ft_group.master_call_all_client(
            get_service_key_by_type(RemoteComponentService, RemoteComponentService.run_dist_call_wait_task.__name__), 
            set(), task_uuid)

    async def handle_msg_from_remote_comp(self, key: str, rpc_key: str, event: RemoteCompEvent):
        app_obj = self._app_objs[key]
        tasks: list[Coroutine[None, None, Any]] = []
        if self._ft_group is not None and self._is_master:
            client_coro = await self._run_dist_call_get_coro(
                key, RemoteComponentService.handle_msg_from_remote_comp.__name__, 
                key, rpc_key, event
            )
            tasks.append(client_coro)

        tasks.append(app_obj.app.handle_msg_from_remote_comp(rpc_key, event))
        if len(tasks) == 1:
            return await tasks[0]
        else:
            return (await asyncio.gather(*tasks))[0]

    def _patch_app_event(self, ev: AppEvent, prefixes: List[str], app: App,):
        for _, ui_ev in ev.type_event_tuple:
            if isinstance(ui_ev, UpdateComponentsEvent):
                comp_dict = app.root._get_uid_encoded_to_comp_dict()
                ui_ev.remote_component_all_childs = list(comp_dict.keys())
        ev_dict = ev.to_dict()
        return ev_dict

    async def run_single_event(self,
                               key: str,
                               type,
                               data,
                               is_sync: bool = False):
        """is_sync: only used for ui event.
        """
        app_obj = self._app_objs[key]
        assert app_obj.mounted_app_meta is not None
        if type == AppEventType.UIEvent.value:
            tasks: list[asyncio.Task] = []
            if self._ft_group is not None:
                # distributed should always use sync ui handler.
                is_sync = True
            if self._ft_group is not None and self._is_master:
                client_coro = await self._run_dist_call_get_coro(
                    key, RemoteComponentService.run_single_event.__name__, 
                    key, type, data, is_sync
                )
                tasks.append(asyncio.create_task(client_coro, name="ui event client"))
            ev = UIEvent.from_dict(data)
            # if handler contains distributed sync op, it will block master thread.
            tasks.append(asyncio.create_task( 
                app_obj.app._handle_event_with_ctx(ev, is_sync), name="ui event master"))
            if len(tasks) == 1:
                res = await tasks[0]
            else:
                res = (await asyncio.gather(*tasks))[0]
            return res 

    async def handle_simple_rpc(self, key: str, event: str, *args, **kwargs):
        app_obj = self._app_objs[key]
        coros = []
        if self._ft_group is not None and self._is_master:
            client_coro = await self._run_dist_call_get_coro(
                key, RemoteComponentService.handle_simple_rpc.__name__, 
                key, event, *args, **kwargs
            )
            coros.append(client_coro)
        coros.append(app_obj.app._flowapp_simple_rpc_handlers.call_event(event, *args, **kwargs))
        if len(coros) == 1:
            return await coros[0]
        else:
            return (await asyncio.gather(*coros))[0]

    @marker.mark_server_event(event_type=ServiceEventType.Exit)
    async def on_exit(self):
        for app_obj in self._app_objs.values():
            # send loop must be shutdown after all ui unmount.
            try:
                app_obj.app.app_terminate()
                await app_obj.app.app_terminate_async()
            except:
                traceback.print_exc()
            app_obj.shutdown_ev.set()
            if app_obj.send_loop_task is not None:
                await app_obj.send_loop_task
        # if self._ft_group is not None:
        #     await self._ft_group.close()
        dist_meta = self._dist_meta
        if dist_meta is not None:
            dist_meta.cleanup()
        
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
