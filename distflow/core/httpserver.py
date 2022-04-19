import asyncio
import json
import sys
import threading
import traceback
from functools import partial
from typing import Awaitable, Callable, List, Optional, Type, Union

import aiohttp
from aiohttp import web

from distflow import compat
from distflow.core import core_io
from distflow.core.client import RemoteException, format_stdout
from distflow.core.serviceunit import ServiceType

from distflow.core.server_core import ProtobufServiceCore, ServiceCore

from distflow.protos import remote_object_pb2
from distflow.protos import remote_object_pb2 as remote_object_pb2
from distflow.utils import df_logging

LOGGER = df_logging.get_logger()


def chunkify(a, n):
    assert len(a) > 0 and n <= len(a)
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


class PeerExit(Exception):
    pass


def chunkify_binary(binary, max_size):
    chunks = list(chunkify(binary, (max_size + len(binary) - 1) // max_size))
    num_chunks = len(chunks)
    desp = [num_chunks, len(binary)]
    header_msg = remote_object_pb2.WebSocketStream(
        data=json.dumps(desp), type=remote_object_pb2.WebSocketType.Chunk)
    return header_msg, chunks


class WebsocketClientBase(object):
    # TODO peer client use a async queue instead of recv because
    # aiohttp don't allow parallel recv
    def __init__(self, ws: web.WebSocketResponse, uid: Optional[int] = None):
        self.ws = ws
        self._uid = uid  # type: Optional[int]

    def _check_remote_exception(self, exception_str: str):
        if exception_str == "":
            return
        exc_dict = json.loads(exception_str)
        raise RemoteException(exc_dict["detail"])

    async def send_raw(self, data):
        return await self.ws.send_bytes(data)

    async def send(self,
                   data,
                   service_key="",
                   msg_type=remote_object_pb2.WebSocketType.Peer,
                   exc_str="",
                   request_id=0):
        data = [data]
        arrays, decoupled = core_io.extract_arrays_from_data(data,
                                                             json_index=True)
        arrays = [core_io.data2pb(a) for a in arrays]
        if self._uid is not None:
            request_id = self._uid
        pb = remote_object_pb2.WebSocketStream(arrays=arrays,
                                               data=json.dumps(decoupled),
                                               service_key=service_key,
                                               exception=exc_str,
                                               type=msg_type,
                                               id=request_id)
        # if use list of bytes, browser raise "invalid frame buffer"
        # so we handle chunks by ourself.
        # TODO better chunk messages
        if compat.Python35:
            max_size = 1 << 20
        else:
            max_size = self.ws._max_msg_size
        binary = pb.SerializeToString()
        if len(binary) <= max_size:
            return await self.ws.send_bytes(binary)
        header, chunks = chunkify_binary(binary, max_size)
        await self.ws.send_bytes(header.SerializeToString())
        for c in chunks:
            await self.ws.send_bytes(c)

    def _parse_response(self, response):
        self._check_remote_exception(response.exception)
        arrays = [core_io.pb2data(b) for b in response.arrays]
        data_skeleton = json.loads(response.data)
        results = core_io.put_arrays_to_data(arrays,
                                             data_skeleton,
                                             json_index=True)
        results = results[0]
        return results

    async def recv(self):
        # if receive rpc request, run rpc and wait again.
        # return if peer message get.
        data = await self.ws.receive_bytes()
        response = remote_object_pb2.WebSocketStream()
        response.ParseFromString(data)
        if response.type == remote_object_pb2.WebSocketType.PeerExit:
            raise PeerExit()
        return self._parse_response(response)

    async def send_exception(self, exc, request_id):
        exception_json = {"error": str(exc), "detail": "unknown"}
        string = json.dumps(exception_json)
        return await self.send("", exc_str=string, request_id=request_id)

    async def send_exception_with_traceback(self, exc):
        detail = traceback.format_exc()
        exception_json = {"error": str(exc), "detail": detail}
        string = json.dumps(exception_json)
        return await self.send("", exc_str=string)

    async def send_peer_exit(self):
        return await self.send(
            "", msg_type=remote_object_pb2.WebSocketType.PeerExit)


class PeerWebsocketClientBase(WebsocketClientBase):
    def __init__(self, ws, uid):
        super().__init__(ws, uid)
        self._msg_q = asyncio.Queue()

    async def recv(self):
        # if receive rpc request, run rpc and wait again.
        # return if peer message get.
        response = await self._msg_q.get()
        if response.type == remote_object_pb2.WebSocketType.PeerExit:
            raise PeerExit()
        return self._parse_response(response)

    async def put(self, msg):
        return await self._msg_q.put(msg)


class ServerClient(WebsocketClientBase):
    def __init__(self, ws, service_core: ProtobufServiceCore):
        super().__init__(ws)
        self.__service_core = service_core
        self._id_to_peer_client = {}

    async def _handle_peer_msg(self, response):
        uid = response.id
        if uid in self._id_to_peer_client:
            client = self._id_to_peer_client[uid]
            return await client.put(response)
        else:
            raise RuntimeError("receive a peer msg, but the client not exists")

    async def _peer_task(self, client, response, uid):
        service_key = response.service_key
        func, serv_meta = self.__service_core.service_units.get_service_and_meta(
            service_key)
        arrays = [core_io.pb2data(b) for b in response.arrays]
        data_skeleton = json.loads(response.data)
        args, kwargs = core_io.put_arrays_to_data(arrays,
                                                  data_skeleton,
                                                  json_index=True)[0]
        # peer function return will be ignored, must be async func
        # and not generator.
        assert serv_meta.type == ServiceType.AsyncWebSocket
        # with self.__service_core._enter_exec_context():
        try:
            await func(client, *args, **kwargs)
        except PeerExit as e:
            # only raised by recv, not recv_raw.
            LOGGER.info("peer {} exit received".format(service_key))
        except Exception as e:
            exc = self.__service_core._remote_exception_json(e)
            await self.send(None, exc_str=exc)
        finally:
            self._id_to_peer_client.pop(uid)

    def _handle_peer_request(self, response):
        # TODO do we need to generate a server uid? the peer client is used
        # for only one websocket client.
        uid = response.id
        assert uid != 0, "you must provide a uid"
        assert uid not in self._id_to_peer_client
        peer_q_client = PeerWebsocketClientBase(self.ws, uid)
        self._id_to_peer_client[uid] = peer_q_client
        return self._peer_task(peer_q_client, response, uid)

    async def _handle_rpc(self, response):
        service_key = response.service_key
        func, serv_meta = self.__service_core.service_units.get_service_and_meta(
            service_key)
        arrays = [core_io.pb2data(b) for b in response.arrays]
        data_skeleton = json.loads(response.data)
        args, kwargs = core_io.put_arrays_to_data(arrays,
                                                  data_skeleton,
                                                  json_index=True)[0]

        exc = ""
        res = None
        if serv_meta.is_async:
            # we cant capture stdouts in async service
            assert not serv_meta.is_gen
            res, is_exception = await self.__service_core.execute_async_service(
                service_key, args, kwargs, json_call=True)
            if is_exception:
                exc = res
                res = None
        elif serv_meta.type == ServiceType.Normal:
            res, is_exception = self.__service_core.execute_service(
                service_key, args, kwargs, json_call=True)
            if is_exception:
                exc = res
                res = None
        else:
            exc = (
                "not implemented rpc service type, available: unary, async unary"
            )
        await self.send(res,
                        msg_type=remote_object_pb2.WebSocketType.RPCReply,
                        exc_str=exc,
                        request_id=response.id)


def create_task(coro):
    if compat.Python3_7AndLater:
        return asyncio.create_task(coro)
    else:
        return asyncio.ensure_future(coro)


class WebsocketHandler:
    def __init__(self, service_core: ProtobufServiceCore):
        self.service_core = service_core
        self._peer_tasks = []
        self._alock = asyncio.Lock()
        self._peer_task_q = asyncio.Queue()
        self._peer_status_q = asyncio.Queue()
        self._shutdown_ev = asyncio.Event()

    async def peer_task_executor(self):
        peer_tasks = set()
        wait_q_task = create_task(self._peer_task_q.get())
        wait_exit_task = create_task(self._shutdown_ev.wait())
        try:
            while True:
                (done, pending) = await asyncio.wait(
                    [*peer_tasks, wait_q_task, wait_exit_task],
                    return_when=asyncio.FIRST_COMPLETED)
                if wait_exit_task in done:
                    for t in peer_tasks:
                        t.cancel()
                    wait_q_task.cancel()
                    break
                new_peer_tasks = set()
                for t in peer_tasks:
                    if t not in done:
                        new_peer_tasks.add(t)
                if wait_q_task in done:
                    new_peer_coro = wait_q_task.result()
                    new_peer_tasks.add(create_task(new_peer_coro))
                    wait_q_task = create_task(self._peer_task_q.get())
                peer_tasks = new_peer_tasks
        except Exception as e:
            traceback.print_exc()
            raise e

    async def websocket_msg_handler(self, request):
        service_core = self.service_core
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        peer_client = ServerClient(ws, service_core)
        with service_core._enter_exec_context():
            try:
                service_core.service_units.websocket_onconnect(peer_client)
            except Exception as e:
                await peer_client.send_exception_with_traceback(e)
        try:
            # TODO dispatch messages for peer waiters
            async for ws_msg in ws:
                if ws_msg.type == aiohttp.WSMsgType.BINARY:
                    data = ws_msg.data
                    ws_stream = remote_object_pb2.WebSocketStream()
                    ws_stream.ParseFromString(data)
                    if ws_stream.type == remote_object_pb2.WebSocketType.PeerRequest:
                        # peer function return will be ignored, must be async func
                        # and not generator.
                        await self._peer_task_q.put(
                            peer_client._handle_peer_request(ws_stream))
                    elif ws_stream.type == remote_object_pb2.WebSocketType.Peer:
                        await peer_client._handle_peer_msg(ws_stream)
                    elif ws_stream.type == remote_object_pb2.WebSocketType.RPCRequest:
                        # TODO should we run rpc task in another loop? 
                        await peer_client._handle_rpc(ws_stream)
                    elif ws_stream.type == remote_object_pb2.WebSocketType.PeerExit:
                        # ok to ignore peer exit when service already exit
                        
                        continue
                    else:
                        raise NotImplementedError

                elif ws_msg.type == aiohttp.WSMsgType.ERROR:
                    print("closed")
        finally:
            # for task in self._peer_task_q:
            #     task.cancel()
            service_core.service_units.websocket_ondisconnect(peer_client)
            self._shutdown_ev.set()


class AllWebsocketHandler:
    def __init__(self, service_core: ProtobufServiceCore):
        self.service_core = service_core

    async def websocket_handler(self, request):
        client = WebsocketHandler(self.service_core)
        return await asyncio.gather(client.websocket_msg_handler(request),
                                    client.peer_task_executor())


class HttpService:
    def __init__(self, service_core: ProtobufServiceCore):
        self.service_core = service_core

    async def remote_json_call_http(self, request: web.Request):
        data_bin = await request.read()
        pb_data = remote_object_pb2.RemoteJsonCallRequest()
        pb_data.ParseFromString(data_bin)
        pb_data.flags = remote_object_pb2.EncodeMethod.JsonArray
        res = await self.service_core.remote_json_call_async(pb_data)
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


async def serve_app(app, port, shutdown_ev, url=None):
    loop = asyncio.get_event_loop()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=url, port=port)
    await site.start()
    await _await_shutdown(shutdown_ev, loop)
    await runner.cleanup()


async def serve_service_core_task(server_core: ProtobufServiceCore,
                                  port=50052,
                                  credentials=None,
                                  rpc_name="/api/jsonrpc",
                                  ws_name="/api/ws"):
    http_service = HttpService(server_core)
    ws_service = AllWebsocketHandler(server_core)
    shutdown_ev = server_core.shutdown_event
    app = web.Application()
    app.router.add_post(rpc_name, http_service.remote_json_call_http)
    app.router.add_get(ws_name, ws_service.websocket_handler)
    return await serve_app(app, port, shutdown_ev)


def serve_service_core(server_core: ProtobufServiceCore,
                       port=50052,
                       credentials=None,
                       rpc_name="/api/jsonrpc",
                       ws_name="/api/ws"):
    http_service = HttpService(server_core)
    ws_service = AllWebsocketHandler(server_core)
    shutdown_ev = server_core.shutdown_event
    loop = asyncio.get_event_loop()
    try:
        app = web.Application()
        app.router.add_post(rpc_name, http_service.remote_json_call_http)
        app.router.add_get(ws_name, ws_service.websocket_handler)
        if compat.Python35:
            web.run_app(app, port=port)
        else:
            loop.run_until_complete(serve_app(app, port, shutdown_ev))
        # web.run_app(app, port=port)
        # we just use shutdown event in service core.
        # TODO find a way to exit connection in service
        # loop.run_until_complete(_await_shutdown(shutdown_ev, loop))
    except KeyboardInterrupt:
        print("aiohttp server shutdown by keyboard interrupt")
