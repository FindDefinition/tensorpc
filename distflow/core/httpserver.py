import asyncio
import json
import sys
import threading
import traceback
from functools import partial
from typing import Awaitable, Callable, Dict, List, Optional, Set, Type, Union

import aiohttp
from aiohttp import web

from distflow import compat
from distflow.core import core_io
from distflow.core.client import RemoteException, format_stdout
from distflow.core.serviceunit import ServiceType

from distflow.core.server_core import ProtobufServiceCore, ServiceCore

from distflow.protos import remote_object_pb2
from distflow.protos import remote_object_pb2 as remote_object_pb2
from distflow.protos import rpc_message_pb2

from distflow.protos import wsdef_pb2
from distflow.utils import df_logging
from contextlib import suppress
import numpy as np
import time

LOGGER = df_logging.get_logger()


class WebsocketClient(object):
    # TODO peer client use a async queue instead of recv because
    # aiohttp don't allow parallel recv
    def __init__(self,
                 ws: web.WebSocketResponse,
                 serv_id_to_name: Dict[int, str],
                 uid: Optional[int] = None):
        self.ws = ws
        self._uid = uid  # type: Optional[int]
        self._serv_id_to_name = serv_id_to_name
        self._name_to_serv_id = {v: k for k, v in serv_id_to_name.items()}

    async def send_raw(self, data):
        return await self.ws.send_bytes(data)

    def get_client_id(self):
        return id(self.ws)

    def get_event_id(self):
        return int(time.time() * 1e9)

    def __hash__(self):
        return self.get_client_id()

    async def send(self,
                   data,
                   msg_type: core_io.SocketMsgType,
                   service_key: str = "",
                   request_id: int = 0):
        if self._uid is not None:
            request_id = self._uid
        sid = 0
        if service_key != "":
            sid = self._name_to_serv_id[service_key]
        if msg_type.value & core_io.SocketMsgType.ErrorMask.value:
            req = wsdef_pb2.Header(service_id=sid,
                                   data=data,
                                   rpc_id=request_id)
        else:
            req = wsdef_pb2.Header(service_id=sid, rpc_id=request_id)
        # if use list of bytes, browser raise "invalid frame buffer"
        # so we handle chunks by ourself.
        max_size = self.ws._max_msg_size
        encoder = core_io.SocketMessageEncoder(data)
        tasks: List[asyncio.Task] = []
        for chunk in encoder.get_message_chunks(msg_type, req, max_size):
            tasks.append(asyncio.create_task(self.ws.send_bytes(chunk)))
        await asyncio.gather(*tasks)

    async def send_exception(self, exc: BaseException,
                             type: core_io.SocketMsgType, request_id: int):
        return await self.send(core_io.get_exception_json(exc),
                               type,
                               request_id=request_id)

    async def send_error_string(self, err: str, detail: str,
                                type: core_io.SocketMsgType, request_id: int):
        return await self.send(core_io.get_error_json(err, detail),
                               type,
                               request_id=request_id)

    async def send_user_error_string(self, err: str, detail: str,
                                     request_id: int):
        return await self.send_error_string(err, detail,
                                            core_io.SocketMsgType.UserError,
                                            request_id)

    async def send_user_error(self, exc, request_id: int):
        return await self.send_exception(exc, core_io.SocketMsgType.UserError,
                                         request_id)

    async def send_event_error(self, exc, request_id: int):
        return await self.send_exception(exc, core_io.SocketMsgType.EventError,
                                         request_id)

    async def send_subscribe_error(self, exc, request_id: int):
        return await self.send_exception(exc,
                                         core_io.SocketMsgType.SubscribeError,
                                         request_id)


def create_task(coro):
    if compat.Python3_7AndLater:
        return asyncio.create_task(coro)
    else:
        return asyncio.ensure_future(coro)


async def _cancel(task):
    # more info: https://stackoverflow.com/a/43810272/1113207
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


class AllWebsocketHandler:
    def __init__(self, service_core: ProtobufServiceCore):
        self.service_core = service_core
        self.clients = []

        self.delete_event_ev = asyncio.Event()
        self.new_event_ev = asyncio.Event()
        self.shutdown_ev = service_core.async_shutdown_event
        self._shutdown_task: Optional[asyncio.Task] = None

        self.all_ev_providers = service_core.service_units.get_all_event_providers(
        )
        self.event_to_clients: Dict[str, Set[WebsocketClient]] = {}
        self.client_to_events: Dict[WebsocketClient, Set[str]] = {}

        self._serv_id_to_name = service_core.service_units.get_service_id_to_name(
        )
        self._name_to_serv_id = {
            v: k
            for k, v in self._serv_id_to_name.items()
        }

        self._new_events: Set[str] = set()
        self._delete_events: Set[str] = set()

    async def _handle_rpc(self, client: WebsocketClient, service_key: str,
                          data, req_id: int, is_notification: bool):
        _, serv_meta = self.service_core.service_units.get_service_and_meta(
            service_key)
        args, kwargs = data
        res = None
        is_exception = False
        if serv_meta.is_async:
            # we cant capture stdouts in async service
            assert not serv_meta.is_gen
            res, is_exception = await self.service_core.execute_async_service(
                service_key, args, kwargs, json_call=True)
        elif serv_meta.type == ServiceType.Normal:
            res, is_exception = self.service_core.execute_service(
                service_key, args, kwargs, json_call=True)
        else:
            is_exception = True
            exc_str = "not implemented rpc service type, available: unary, async unary"
            res = json.dumps({"error": exc_str, "detail": ""})
        if is_exception:
            msg_type = core_io.SocketMsgType.RPCError
        else:
            msg_type = core_io.SocketMsgType.RPC
        if is_notification and not is_exception:
            return
        await client.send([res], msg_type=msg_type, request_id=req_id)

    async def handle_new_connection(self, request):
        service_core = self.service_core
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        client = WebsocketClient(
            ws, service_core.service_units.get_service_id_to_name())
        with service_core._enter_exec_context():
            try:
                service_core.service_units.websocket_onconnect(client)
            except Exception as e:
                await client.send_user_error(e, 0)
        try:
            # TODO we should wait shutdown here
            # TODO handle send error
            async for ws_msg in ws:
                if ws_msg.type == aiohttp.WSMsgType.BINARY:
                    data = ws_msg.data
                    try:
                        msg_type, req, data = core_io.parse_message_chunks(
                            [data])
                    except Exception as e:
                        await client.send_user_error(e, 0)
                        continue
                    if req.service_id not in self._serv_id_to_name:
                        await client.send_user_error_string(
                            "ServiceNotFound",
                            f"can't find your service {req.service_id}",
                            req.rpc_id)
                        continue
                    serv_key: str = self._serv_id_to_name[req.service_id]
                    assert req.chunk_index == 0, "this should't happen"
                    if msg_type == core_io.SocketMsgType.Subscribe:
                        event_key: str = serv_key
                        if event_key not in self.all_ev_providers:
                            await client.send_subscribe_error(
                                KeyError(f"event key {event_key} not found"),
                                req.rpc_id)
                            continue
                        if event_key not in self.event_to_clients:
                            self.event_to_clients[event_key] = set()
                            # we set this to tell event provider this event is new.
                            self._new_events.add(event_key)
                            # trigger event provider to add new event.
                            self.new_event_ev.set()
                        self.event_to_clients[event_key].add(client)
                        if client not in self.client_to_events:
                            self.client_to_events[client] = set()
                        self.client_to_events[client].add(event_key)

                    elif msg_type == core_io.SocketMsgType.UnSubscribe:
                        event_key: str = serv_key
                        if event_key not in self.all_ev_providers:
                            await client.send_subscribe_error(
                                KeyError("service id not found"), req.rpc_id)
                            continue
                        # remove events
                        if event_key in self.event_to_clients:
                            clients = self.event_to_clients[event_key]
                            if client not in clients:
                                await client.send_subscribe_error(
                                    KeyError(
                                        f"you haven't sub event {event_key} yet."
                                    ), req.rpc_id)
                                continue
                            clients.remove(client)
                            if not clients:
                                self.event_to_clients.pop(event_key)
                                # we set this to tell event provider
                                # this event doesn't have any subscriber.
                                self._delete_events.add(event_key)
                                # trigger event provider to delete event in loop.
                                self.delete_event_ev.set()

                        client_evs = self.client_to_events[client]
                        client_evs.remove(event_key)
                        if not client_evs:
                            self.client_to_events.pop(client)

                    elif msg_type == core_io.SocketMsgType.RPC:
                        await self._handle_rpc(client, serv_key, data,
                                               req.rpc_id, False)
                    elif msg_type == core_io.SocketMsgType.Notification:
                        await self._handle_rpc(client, serv_key, data,
                                               req.rpc_id, True)
                    elif msg_type == core_io.SocketMsgType.QueryServiceIds:
                        await client.send(self._name_to_serv_id,
                                          msg_type,
                                          request_id=req.rpc_id)
                    else:
                        raise NotImplementedError
                elif ws_msg.type == aiohttp.WSMsgType.ERROR:
                    print("ERROR")
                else:
                    raise NotImplementedError
        finally:
            # remove all sub events for this websocket
            if client in self.client_to_events:
                for ev in self.client_to_events[client]:
                    clients = self.event_to_clients[ev]
                    clients.remove(client)
                    if not clients:
                        self._delete_events.add(ev)
                        self.event_to_clients.pop(ev)
            self.client_to_events.pop(client)
            # tell event executor remove task for this client
            if self._delete_events:
                self.delete_event_ev.set()
            try:
                service_core.service_units.websocket_ondisconnect(client)
            except:
                traceback.print_exc()

    async def event_provide_executor(self):
        subed_evs = [(k, self.all_ev_providers[k])
                     for k in self.event_to_clients.keys()]
        ev_tasks = {
            k: asyncio.create_task(ev.fn(), name=k)
            for k, ev in subed_evs
        }
        task_to_ev: Dict[asyncio.Task,
                         str] = {v: k
                                 for k, v in ev_tasks.items()}
        wait_new_ev_task = asyncio.create_task(self.new_event_ev.wait(),
                                               name="new_event")
        wait_del_ev_task = asyncio.create_task(self.delete_event_ev.wait(),
                                               name="delete_event")
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(self.shutdown_ev.wait())
        wait_tasks: List[asyncio.Task] = [
            *ev_tasks.values(),
            wait_new_ev_task,
            wait_del_ev_task,
            self._shutdown_task,
        ]
        while True:
            (done,
             pending) = await asyncio.wait(wait_tasks,
                                           return_when=asyncio.FIRST_COMPLETED)
            if self.shutdown_ev.is_set():
                for task in pending:
                    await _cancel(task)
                break
            new_tasks: Dict[str, asyncio.Task] = {}
            new_task_to_ev: Dict[asyncio.Task, str] = {}
            wait_tasks = [
                self._shutdown_task,
            ]
            # determine events waited next.
            if self.new_event_ev.is_set():
                for new_ev in self._new_events:
                    # add new event to loop
                    ev = self.all_ev_providers[new_ev]
                    new_task = asyncio.create_task(ev.fn(), name=new_ev)
                    new_tasks[new_ev] = new_task
                    new_task_to_ev[new_task] = new_ev
                self._new_events.clear()
                self.new_event_ev.clear()
                wait_new_ev_task = asyncio.create_task(
                    self.new_event_ev.wait(), name="new_event")
            if self.delete_event_ev.is_set():
                self.delete_event_ev.clear()
                wait_del_ev_task = asyncio.create_task(
                    self.delete_event_ev.wait(), name="delete_event")
            for task in done:
                if task in task_to_ev:
                    ev_key = task_to_ev[task]
                    if ev_key not in self._delete_events:
                        ev = self.all_ev_providers[ev_key]
                        new_task = asyncio.create_task(ev.fn(), name=ev_key)
                        new_tasks[ev_key] = new_task
                        new_task_to_ev[new_task] = ev_key
            task_to_be_canceled: List[asyncio.Task] = []
            for task in pending:
                if task in task_to_ev:
                    ev_key = task_to_ev[task]
                    if ev_key not in self._delete_events:
                        new_tasks[ev_key] = task
                        new_task_to_ev[task] = ev_key
                    else:
                        task_to_be_canceled.append(task)
            wait_tasks.append(wait_new_ev_task)
            wait_tasks.append(wait_del_ev_task)
            wait_tasks.extend(new_tasks.values())
            self._delete_events.clear()
            sending_tasks = []
            # we must cancel task AFTER clear _delete_events
            for task in task_to_be_canceled:
                # TODO better cancel, don't await here.
                await _cancel(task)
            for task in done:
                if task in task_to_ev:
                    exc = task.exception()
                    if exc is not None:
                        msg_type = core_io.SocketMsgType.EventError
                        res = self.service_core._remote_exception_json(exc)
                    else:
                        msg_type = core_io.SocketMsgType.Event
                        res = task.result()
                    ev_str = task_to_ev[task]
                    ev_clients = self.event_to_clients[ev_str]
                    for client in ev_clients:
                        sending_tasks.append(
                            client.send([res],
                                        service_key=ev_str,
                                        msg_type=msg_type))
            if sending_tasks:
                try:
                    # TODO if this function fail...
                    await asyncio.wait(sending_tasks)
                except ConnectionResetError:
                    print("Cannot write to closing transport")
            task_to_ev = new_task_to_ev


class HttpService:
    def __init__(self, service_core: ProtobufServiceCore):
        self.service_core = service_core

    async def remote_json_call_http(self, request: web.Request):
        try:
            data_bin = await request.read()
            pb_data = rpc_message_pb2.RemoteJsonCallRequest()
            pb_data.ParseFromString(data_bin)
            pb_data.flags = rpc_message_pb2.JsonArray
            res = await self.service_core.remote_json_call_async(pb_data)
        except Exception as e:
            data = self.service_core._remote_exception_json(e)
            res = rpc_message_pb2.RemoteCallReply(exception=data)
        byte = res.SerializeToString()
        # TODO better headers
        headers = {
            'Access-Control-Allow-Origin': '*',
            # 'Access-Control-Allow-Headers': '*',
            # 'Access-Control-Allow-Method': 'POST',
        }
        res = web.Response(body=byte, headers=headers)
        return res


async def _await_shutdown(shutdown_ev, loop):
    return await loop.run_in_executor(None, shutdown_ev.wait)


async def serve_app(app,
                    port,
                    shutdown_ev: threading.Event,
                    async_shutdown_ev: asyncio.Event,
                    is_sync: bool = False,
                    url=None):
    loop = asyncio.get_running_loop()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=url, port=port)
    await site.start()
    if not is_sync:
        await async_shutdown_ev.wait()
    else:
        await _await_shutdown(shutdown_ev, loop)
        async_shutdown_ev.set()
    await runner.cleanup()


async def serve_service_core_task(server_core: ProtobufServiceCore,
                                  port=50052,
                                  credentials=None,
                                  rpc_name="/api/rpc",
                                  ws_name="/api/ws",
                                  is_sync: bool = False):
    http_service = HttpService(server_core)
    server_core._init_async_members()
    ws_service = AllWebsocketHandler(server_core)
    app = web.Application()
    loop_task = asyncio.create_task(ws_service.event_provide_executor())
    app.router.add_post(rpc_name, http_service.remote_json_call_http)
    app.router.add_get(ws_name, ws_service.handle_new_connection)
    return await asyncio.gather(
        serve_app(app, port, server_core.shutdown_event,
                  server_core.async_shutdown_event, is_sync), loop_task)


def serve_service_core(server_core: ProtobufServiceCore,
                       port=50052,
                       credentials=None,
                       rpc_name="/api/rpc",
                       ws_name="/api/ws"):
    http_task = serve_service_core_task(server_core, port, None, is_sync=True)
    try:
        asyncio.run(http_task)
    except KeyboardInterrupt:
        print("shutdown by keyboard interrupt")
