import asyncio
import contextlib
from dataclasses import dataclass
import io
import json
import sys
import threading
import traceback
from functools import partial
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import aiohttp
from aiohttp import web

from tensorpc import compat
from tensorpc.core import core_io, defs
from tensorpc.core import serviceunit
from tensorpc.core.client import RemoteException, format_stdout
from tensorpc.core.serviceunit import ServiceType
import ssl
from tensorpc.core.server_core import ProtobufServiceCore, ServiceCore, ServerMeta
from pathlib import Path
from tensorpc.protos_export import remote_object_pb2
from tensorpc.protos_export import remote_object_pb2 as remote_object_pb2
from tensorpc.protos_export import rpc_message_pb2

from tensorpc.protos_export import wsdef_pb2
from tensorpc.utils import df_logging
from tensorpc.constants import TENSORPC_WEBSOCKET_MSG_SIZE
from contextlib import suppress
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

LOGGER = df_logging.get_logger()
JS_MAX_SAFE_INT = 2 ** 53 - 1


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
        self._ev_cnt = 0

    async def send_raw(self, data):
        return await self.ws.send_bytes(data)

    def get_client_id(self):
        return id(self.ws)

    def get_event_id(self):
        self._ev_cnt = (self._ev_cnt + 1) % JS_MAX_SAFE_INT
        return self._ev_cnt

    def __hash__(self):
        return self.get_client_id()

    async def send(self,
                   data,
                   msg_type: core_io.SocketMsgType,
                   service_key: str = "",
                   request_id: int = 0,
                   is_json: bool = False,
                   dynamic_key: str = ""):
        """data must not be encoded.
        """
        if self._uid is not None:
            request_id = self._uid
        sid = 0
        if service_key != "":
            sid = self._name_to_serv_id[service_key]
        if msg_type.value & core_io.SocketMsgType.ErrorMask.value:
            req = wsdef_pb2.Header(service_id=sid,
                                   data=json.dumps(data),
                                   rpc_id=request_id)
        else:
            req = wsdef_pb2.Header(service_id=sid,
                                   rpc_id=request_id,
                                   dynamic_key=dynamic_key)
        if is_json:
            return await self.ws.send_bytes(
                core_io.json_only_encode(data, msg_type, req))
        max_size = self.ws._max_msg_size - 128
        # max_size = 1024 * 1024
        # TODO reslove "8192"
        encoder = core_io.SocketMessageEncoder(data, skeleton_size_limit=max_size - 8192)
        tasks = []
        # max_size = TENSORPC_WEBSOCKET_MSG_SIZE
        # t = time.time()
        # chunks = list(encoder.get_message_chunks(msg_type, req, max_size))
        # print("ENCODE TEIM", len(chunks), time.time() - t)
        # if len(chunks) > 1:
        #     header_rec = core_io.TensoRPCHeader(chunks[0])
        #     rec = core_io.parse_message_chunks(header_rec, chunks[1:])
        # print("SEND CHUNKS", len(chunks))
        # if len(chunks) > 1:
        #     print("BEFORE SEND")
        try:
            for chunk in encoder.get_message_chunks(msg_type, req, max_size):
                assert len(chunk) <= max_size
                # tasks.append(self.ws.send_bytes(chunk))
                await self.ws.send_bytes(chunk)
        except ConnectionResetError:
            print("CLIENT SEND ERROR, RETURN")
            return
        # await tasks[0]
        # if len(tasks) > 1:
        #     tasks = [asyncio.create_task(t) for t in tasks[1:]]
        #     await asyncio.wait(tasks)

    async def send_exception(self, exc: BaseException,
                             type: core_io.SocketMsgType, request_id: int):
        return await self.send(core_io.get_exception_json(exc),
                               type,
                               request_id=request_id,
                               is_json=True)

    async def send_error_string(self, err: str, detail: str,
                                type: core_io.SocketMsgType, request_id: int):
        return await self.send(core_io.get_error_json(err, detail),
                               type,
                               request_id=request_id,
                               is_json=True)

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
        with service_core._enter_exec_context():
            service_core.service_units.init_service()
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
            await client.send(json.loads(res),
                              msg_type=msg_type,
                              request_id=req_id,
                              is_json=True)
            return
        else:
            msg_type = core_io.SocketMsgType.RPC
        if is_notification:
            return
        await client.send([res], msg_type=msg_type, request_id=req_id)

    async def handle_new_connection(self, request):
        print("NEW CONN", request)
        service_core = self.service_core
        ws = web.WebSocketResponse()
        conn_st_ev = asyncio.Event()
        # wait at most 100 rpcs
        conn_rpc_queue: "asyncio.Queue[asyncio.Task]" = asyncio.Queue(1000)
        await ws.prepare(request)
        client = WebsocketClient(
            ws, service_core.service_units.get_service_id_to_name())
        with service_core._enter_exec_context():
            try:
                service_core.service_units.websocket_onconnect(client)
            except Exception as e:
                await client.send_user_error(e, 0)
        # assert not self.event_to_clients and not self.client_to_events
        wait_task = asyncio.create_task(
            self.rpc_awaiter(conn_rpc_queue, conn_st_ev))
        try:
            # send serv ids first
            await client.send(self._name_to_serv_id,
                              core_io.SocketMsgType.QueryServiceIds,
                              request_id=0,
                              is_json=True)
            # TODO we should wait shutdown here
            # TODO handle send error
            async for ws_msg in ws:
                if ws_msg.type == aiohttp.WSMsgType.BINARY:
                    data = ws_msg.data
                    try:
                        header = core_io.TensoRPCHeader(data)
                        msg_type = header.type
                        req = header.req
                    except Exception as e:
                        await client.send_user_error(e, 0)
                        continue
                    if req.service_id < 0:
                        serv_key = req.service_key
                        if serv_key not in self._name_to_serv_id:
                            await client.send_user_error_string(
                                "ServiceNotFound",
                                f"can't find your service {req.service_key}",
                                req.rpc_id)
                            continue
                        req.service_id = self._name_to_serv_id[serv_key]
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
                        if client not in self.event_to_clients[event_key]:
                            self.event_to_clients[event_key].add(client)
                            if client not in self.client_to_events:
                                self.client_to_events[client] = set()
                            self.client_to_events[client].add(event_key)
                        # send OK
                        # TODO send error if this event is already subscribed
                        await client.send([],
                                          msg_type=msg_type,
                                          request_id=req.rpc_id)

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
                            # if event_key isn't exists, the client_to_events
                            # shouldn't have this event too.

                            client_evs = self.client_to_events[client]
                            client_evs.remove(event_key)
                            if not client_evs:
                                self.client_to_events.pop(client)
                        # TODO send error if this event is already subscribed
                        # send OK
                        await client.send([],
                                          msg_type=msg_type,
                                          request_id=req.rpc_id)

                    elif msg_type == core_io.SocketMsgType.RPC:
                        arg_data = core_io.parse_message_chunks(header, [data])
                        # TODO if full for some time, drop rpc (raise busy error)
                        await conn_rpc_queue.put(
                            asyncio.create_task(
                                self._handle_rpc(client, serv_key, arg_data,
                                                 req.rpc_id, False)))
                    elif msg_type == core_io.SocketMsgType.Notification:
                        arg_data = core_io.parse_message_chunks(header, [data])
                        # TODO if full for some time, drop rpc (raise busy error)
                        await conn_rpc_queue.put(
                            asyncio.create_task(
                                self._handle_rpc(client, serv_key, arg_data,
                                                 req.rpc_id, True)))
                    elif msg_type == core_io.SocketMsgType.QueryServiceIds:
                        await client.send(self._name_to_serv_id,
                                          msg_type,
                                          request_id=req.rpc_id,
                                          is_json=True)
                    else:
                        raise NotImplementedError
                elif ws_msg.type == aiohttp.WSMsgType.ERROR:
                    print("ERROR", ws_msg)
                    print("ERROR", ws_msg.data)

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
            conn_st_ev.set()
            await wait_task
            # cancel all rpc
            while True:
                try:
                    task = conn_rpc_queue.get_nowait()
                    await _cancel(task)
                except:
                    break
        print("CONN", request, "disconnected.")

    async def rpc_awaiter(self, rpc_queue: "asyncio.Queue[asyncio.Task]",
                          shutdown_ev: asyncio.Event):
        _shutdown_task = asyncio.create_task(shutdown_ev.wait())
        rpc_q_task = asyncio.create_task(rpc_queue.get())
        wait_tasks: List[asyncio.Task] = [
            rpc_q_task,
            _shutdown_task,
        ]
        while True:
            (_,
             pending) = await asyncio.wait(wait_tasks,
                                           return_when=asyncio.FIRST_COMPLETED)
            if shutdown_ev.is_set():
                for task in pending:
                    await _cancel(task)
                break
            await rpc_q_task.result()
            rpc_q_task = asyncio.create_task(rpc_queue.get())
            wait_tasks = [
                rpc_q_task,
                _shutdown_task,
            ]

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
            # t = time.time()
            if self.shutdown_ev.is_set():
                for task in pending:
                    await _cancel(task)
                break
            new_tasks: Dict[str, asyncio.Task] = {}
            new_task_to_ev: Dict[asyncio.Task, str] = {}
            wait_tasks = [
                self._shutdown_task,
            ]
            # cur_ev =""
            # determine events waited next.
            if self.new_event_ev.is_set():
                for new_ev in self._new_events:
                    # add new event to loop
                    ev = self.all_ev_providers[new_ev]
                    # self.service_core.service_units.
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
            # schedule new event tasks if they are done
            for task in done:
                if task in task_to_ev:
                    ev_key = task_to_ev[task]
                    if ev_key not in self._delete_events:
                        ev = self.all_ev_providers[ev_key]
                        new_task = asyncio.create_task(ev.fn(), name=ev_key)
                        new_tasks[ev_key] = new_task
                        new_task_to_ev[new_task] = ev_key
                        # cur_ev = ev_key
                    else:
                        # this done task is deleted, may due to unsubscribe or client error.
                        # just remove them.
                        task_to_ev.pop(task)

            task_to_be_canceled: List[asyncio.Task] = []
            # cancel event tasks if they are pending and deleted
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
            sending_tasks: List[Tuple[asyncio.Task, str]] = []
            for task in done:
                # done may contains deleted tasks. they will be removed in task_to_ev before.
                if task in task_to_ev:
                    exc = task.exception()

                    ev_str = task_to_ev[task]

                    if exc is not None:
                        msg_type = core_io.SocketMsgType.EventError
                        ss = io.StringIO()
                        task.print_stack(file=ss)
                        detail = ss.getvalue()
                        res = self.service_core._remote_exception_dict(
                            exc, detail)
                    else:
                        msg_type = core_io.SocketMsgType.Event
                        res = task.result()
                    if isinstance(res, defs.DynamicEvents):
                        # exc is None
                        for dykey, data in res.name_and_datas:
                            data_to_send = [data]
                            ev_clients = self.event_to_clients[ev_str]
                            # we need to generate a rpc id for event
                            for client in ev_clients:
                                rpc_id = client.get_event_id()
                                task = asyncio.create_task(
                                    client.send(data_to_send,
                                                service_key=ev_str,
                                                msg_type=msg_type,
                                                request_id=rpc_id,
                                                is_json=exc is not None,
                                                dynamic_key=dykey))
                                sending_tasks.append((task, ev_str))
                    else:

                        if isinstance(res, defs.DynamicEvent):
                            data = res.data
                            dynamic_key = res.name
                        else:
                            data = res
                            dynamic_key = ""
                        # this event may be deleted before.
                        if exc is None:
                            data_to_send = [data]
                        else:
                            data_to_send = data

                        ev_clients = self.event_to_clients[ev_str]
                        # we need to generate a rpc id for event
                        for client in ev_clients:
                            rpc_id = client.get_event_id()
                            task = asyncio.create_task(
                                client.send(data_to_send,
                                            service_key=ev_str,
                                            msg_type=msg_type,
                                            request_id=rpc_id,
                                            is_json=exc is not None,
                                            dynamic_key=dynamic_key))
                            sending_tasks.append((task, ev_str))
            # we must cancel task AFTER clear _delete_events
            for task in task_to_be_canceled:
                # TODO better cancel, don't await here.
                await _cancel(task)
            # t = time.time()
            if sending_tasks:
                try:
                    # TODO if this function fail...
                    await asyncio.wait([x[0] for x in sending_tasks])
                except ConnectionResetError:
                    print("Cannot write to closing transport")
            for task, ev_str in sending_tasks:
                exc = task.exception()
                if exc is not None:
                    msg_type = core_io.SocketMsgType.EventError
                    ss = io.StringIO()
                    task.print_stack(file=ss)
                    detail = ss.getvalue()
                    res = self.service_core._remote_exception_dict(exc, detail)
                    ev_clients = self.event_to_clients[ev_str]
                    # we need to generate a rpc id for event
                    for client in ev_clients:
                        rpc_id = client.get_event_id()
                        asyncio.create_task(
                            client.send(res,
                                        service_key=ev_str,
                                        msg_type=msg_type,
                                        request_id=rpc_id,
                                        is_json=True,
                                        dynamic_key=""))

            # print("SEND TIME", cur_ev, time.time() - t)
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

    async def file_upload_call(self, request: web.Request):
        reader = await request.multipart()
        # /!\ Don't forget to validate your inputs /!\
        # reader.next() will `yield` the fields of your form
        headers = {
            'Access-Control-Allow-Origin': '*',
            # 'Access-Control-Allow-Headers': '*',
            # 'Access-Control-Allow-Method': 'POST',
        }

        field = await reader.next()
        assert field is not None
        assert field.name == 'data'
        # TODO how to handle large file?
        data = await field.read()
        data = json.loads(data)
        serv_key = data["serv_key"]
        serv_data = data["serv_data"]
        file_size = data["file_size"]

        field = await reader.next()
        assert field is not None
        assert field.name == 'file'
        filename = field.filename
        content = await field.read()
        f = defs.File(filename, content, serv_data)
        # return web.Response(text='{} sized of {} successfully stored'
        #                             ''.format(filename, content), headers=headers)
        res, is_exc = await self.service_core.execute_async_service(
            serv_key, [f], {}, json_call=False)
        # You cannot rely on Content-Length if transfer is chunked.
        if not is_exc:
            return web.Response(text='{} sized of {} successfully stored'
                                ''.format(filename, content),
                                headers=headers)
        else:
            return web.Response(status=500, text=res, headers=headers)

    async def remote_pickle_call_http(self, request: web.Request):
        try:
            data_bin = await request.read()
            pb_data = rpc_message_pb2.RemoteCallRequest()
            pb_data.ParseFromString(data_bin)
            pb_data.flags = rpc_message_pb2.Pickle
            res = await self.service_core.remote_call_async(pb_data)
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
                    url=None,
                    ssl_context=None):
    loop = asyncio.get_running_loop()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=url, port=port, ssl_context=ssl_context)
    await site.start()
    if not is_sync:
        await async_shutdown_ev.wait()
    else:
        await _await_shutdown(shutdown_ev, loop)
        async_shutdown_ev.set()
    await runner.cleanup()


async def serve_service_core_task(server_core: ProtobufServiceCore,
                                  port=50052,
                                  rpc_name="/api/rpc",
                                  ws_name="/api/ws",
                                  is_sync: bool = False,
                                  rpc_pickle_name: str = "/api/rpc_pickle",
                                  client_max_size: int = 4 * 1024**2,
                                  standalone: bool = True,
                                  ssl_key_path: str = "",
                                  ssl_crt_path: str = ""):
    # client_max_size 4MB is enough for most image upload.
    http_service = HttpService(server_core)
    ctx = contextlib.nullcontext()
    if standalone:
        ctx = server_core.enter_global_context()
    with ctx:
        if standalone:
            await server_core._init_async_members()
        ws_service = AllWebsocketHandler(server_core)
        app = web.Application(client_max_size=client_max_size)
        # TODO should we create a global client session for all http call in server?
        loop_task = asyncio.create_task(ws_service.event_provide_executor())
        app.router.add_post(rpc_name, http_service.remote_json_call_http)
        app.router.add_post(rpc_pickle_name,
                            http_service.remote_pickle_call_http)
        app.router.add_post("/api/rpc_file", http_service.file_upload_call)
        app.router.add_get(ws_name, ws_service.handle_new_connection)
        ssl_context = None
        if ssl_key_path != "" and ssl_key_path != "":
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(ssl_crt_path, ssl_key_path)
        return await asyncio.gather(
            serve_app(app,
                      port,
                      server_core.shutdown_event,
                      server_core.async_shutdown_event,
                      is_sync,
                      ssl_context=ssl_context), loop_task)


def serve_service_core(server_core: ProtobufServiceCore,
                       port=50052,
                       rpc_name="/api/rpc",
                       ws_name="/api/ws"):
    http_task = serve_service_core_task(server_core,
                                        port,
                                        is_sync=True,
                                        standalone=True)
    try:
        asyncio.run(http_task)
    except KeyboardInterrupt:

        print("shutdown by keyboard interrupt")
