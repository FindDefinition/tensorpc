import asyncio
import contextlib
import json
import threading
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import aiohttp
from aiohttp import web

from tensorpc.core import core_io, defs
import ssl
from tensorpc.core.server_core import ProtobufServiceCore, ServiceCore, ServerMeta
from pathlib import Path
from tensorpc.protos_export import remote_object_pb2
from tensorpc.protos_export import remote_object_pb2 as remote_object_pb2
from tensorpc.protos_export import rpc_message_pb2

from .core import WebsocketClientBase, WebsocketMsg, WebsocketMsgType, WebsocketHandler

class AiohttpWebsocketClient(WebsocketClientBase):
    def __init__(self,
                 ws: web.WebSocketResponse,
                 serv_id_to_name: Dict[int, str],
                 uid: Optional[int] = None):
        super().__init__(serv_id_to_name, uid)
        self.ws = ws

    def get_msg_max_size(self) -> int: 
        return self.ws._max_msg_size

    async def send_bytes(self, data: bytes): 
        return await self.ws.send_bytes(data)


    def get_client_id(self) -> int: 
        return id(self.ws)

    async def binary_msg_generator(self) -> AsyncGenerator[WebsocketMsg, None]: 
        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                yield WebsocketMsg(msg.data, WebsocketMsgType.Binary)
            elif msg.type == aiohttp.WSMsgType.TEXT:
                yield WebsocketMsg(msg.data, WebsocketMsgType.Text)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                raise Exception("websocket connection closed with exception %s" %
                                self.ws.exception())

class AiohttpWebsocketHandler(WebsocketHandler):
    async def handle_new_connection_aiohttp(self, request):
        print("NEW CONN", request)
        service_core = self.service_core
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        client = AiohttpWebsocketClient(
            ws, service_core.service_units.get_service_id_to_name())
        return await self.handle_new_connection(client)


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
        ws_service = AiohttpWebsocketHandler(server_core)
        app = web.Application(client_max_size=client_max_size)
        # TODO should we create a global client session for all http call in server?
        loop_task = asyncio.create_task(ws_service.event_provide_executor())
        app.router.add_post(rpc_name, http_service.remote_json_call_http)
        app.router.add_post(rpc_pickle_name,
                            http_service.remote_pickle_call_http)
        app.router.add_post("/api/rpc_file", http_service.file_upload_call)
        app.router.add_get(ws_name, ws_service.handle_new_connection_aiohttp)
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
