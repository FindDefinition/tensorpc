import asyncio
import contextlib
import json
import threading
import traceback
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Type, Union

# import aiohttp
# from aiohttp import web

from tensorpc.core import core_io, defs
import ssl
from tensorpc.core.asynctools import cancel_task
from tensorpc.core.constants import TENSORPC_API_FILE_UPLOAD, TENSORPC_FETCH_STATUS
from tensorpc.core.server_core import ProtobufServiceCore, ServiceCore, ServerMeta
from pathlib import Path
from tensorpc.protos_export import remote_object_pb2
from tensorpc.protos_export import remote_object_pb2 as remote_object_pb2
from tensorpc.protos_export import rpc_message_pb2
import asyncio
import uvicorn
import blacksheep as web
from blacksheep import Application, WebSocket, Request, Response
from tensorpc.constants import TENSORPC_WEBSOCKET_MSG_SIZE

from .core import WebsocketClientBase, WebsocketMsg, WebsocketMsgType, WebsocketHandler


class BlacksheepWebsocketClient(WebsocketClientBase):

    def __init__(self,
                id: str,
                 ws: WebSocket,
                 serv_id_to_name: Dict[int, str],
                 uid: Optional[int] = None,
                 client_max_size: int = -1):
        super().__init__(id, serv_id_to_name, uid)
        self.ws = ws
        self.client_max_size = client_max_size

    async def close(self):
        return await self.ws.close()

    def get_msg_max_size(self) -> int:
        if self.client_max_size == -1:
            return TENSORPC_WEBSOCKET_MSG_SIZE
        else:
            return self.client_max_size
        
    async def send_bytes(self, data: bytes):
        return await self.ws.send_bytes(data)

    def get_client_id(self) -> int:
        return id(self.ws)

    async def binary_msg_generator(self, shutdown_ev: asyncio.Event) -> AsyncGenerator[WebsocketMsg, None]:
        while True:
            msg = await self.ws.receive_bytes()
            yield WebsocketMsg(msg, WebsocketMsgType.Binary)

            # recv_task = asyncio.create_task(self.ws.receive_bytes(), name="recv_task")
            # st_task = asyncio.create_task(shutdown_ev.wait(), name="shutdown_task")
            # done, pending = await asyncio.wait([recv_task, st_task], return_when=asyncio.FIRST_COMPLETED)
            # if st_task in done:
            #     # cancel recv task 
            #     await cancel_task(recv_task)
            #     break
            # else:
            #     msg = recv_task.result()
            #     yield WebsocketMsg(msg, WebsocketMsgType.Binary)
            #     recv_task = asyncio.create_task(self.ws.receive_bytes(), name="recv_task")


class BlacksheepWebsocketHandler(WebsocketHandler):
    def __init__(self, service_core: ProtobufServiceCore, client_max_size: int):
        super().__init__(service_core)
        self.client_max_size = client_max_size

    async def handle_new_connection_blacksheep(self, request: WebSocket, client_id: str):
        print("NEW CONN", client_id, request)
        service_core = self.service_core
        await request.accept()
        client = BlacksheepWebsocketClient(client_id, 
            request, service_core.service_units.get_service_id_to_name(), 
            client_max_size=self.client_max_size)
        return await self.handle_new_connection(client, client_id)

    async def handle_new_backup_connection_blacksheep(self, request: WebSocket, client_id: str):
        print("NEW CONN", client_id, request)
        service_core = self.service_core
        await request.accept()
        client = BlacksheepWebsocketClient(client_id, 
            request, service_core.service_units.get_service_id_to_name(), 
            client_max_size=self.client_max_size)
        return await self.handle_new_connection(client, client_id, True)


class HttpService:

    def __init__(self, service_core: ProtobufServiceCore):
        self.service_core = service_core

    async def fetch_status(self, request: web.Request):
        status = {
            "status": "ok",
        }
        res = web.Response(
            status=200,
            content=web.JSONContent(status),
        )
        res.add_header(b'Access-Control-Allow-Origin', b'*')
        return res
    

    async def remote_json_call_http(self,
                                    request: web.Request) -> web.Response:
        try:
            data_bin = await request.read()
            assert data_bin is not None
            pb_data = rpc_message_pb2.RemoteJsonCallRequest()
            pb_data.ParseFromString(data_bin)
            pb_data.flags = rpc_message_pb2.JsonArray
            res = await self.service_core.remote_json_call_async(pb_data)
        except Exception as e:
            data = self.service_core._remote_exception_json(e)
            res = rpc_message_pb2.RemoteCallReply(exception=data)
        byte = res.SerializeToString()
        # TODO better headers
        res = web.Response(
            status=200,
            content=web.Content(b"", byte),
        )
        res.add_header(b'Access-Control-Allow-Origin', b'*')
        return res

    async def remote_pickle_call_http(self, request: web.Request):
        try:
            data_bin = await request.read()
            assert data_bin is not None

            pb_data = rpc_message_pb2.RemoteCallRequest()
            pb_data.ParseFromString(data_bin)
            pb_data.flags = rpc_message_pb2.Pickle
            res = await self.service_core.remote_call_async(pb_data)
        except Exception as e:
            data = self.service_core._remote_exception_json(e)
            res = rpc_message_pb2.RemoteCallReply(exception=data)
        byte = res.SerializeToString()
        # TODO better headers
        res = web.Response(
            status=200,
            content=web.Content(b"", byte),
        )
        res.add_header(b'Access-Control-Allow-Origin', b'*')
        return res

    async def file_upload_call(self, request: web.Request):
        files = await request.multipart()
        metadata = None
        handled_fnames: List[str] = []
        for part in files:
            if part.name == b"data":
                metadata = json.loads(part.data)
            elif part.name == b"file":
                assert metadata is not None
                filename = part.file_name
                assert filename is not None 
                serv_key = metadata["serv_key"]
                serv_data = metadata["serv_data"]
                file_size = metadata["file_size"]
                handled_fnames.append(filename.decode())
                f = defs.File(filename.decode(), part.data, serv_data)
                res, is_exc = await self.service_core.execute_async_service(
                    serv_key, [f], {}, json_call=False)
                if is_exc:
                    return web.text(status=500, value=res)
        return web.text('{} successfully stored'.format(handled_fnames))

async def _await_shutdown(shutdown_ev, loop):
    return await loop.run_in_executor(None, shutdown_ev.wait)

async def serve_app(serv: uvicorn.Server,
                    shutdown_ev: threading.Event,
                    async_shutdown_ev: asyncio.Event,
                    is_sync: bool = False):
    loop = asyncio.get_running_loop()
    await serv.serve()
    async_shutdown_ev.set()
    if not is_sync:
        await async_shutdown_ev.wait()
    else:
        await _await_shutdown(shutdown_ev, loop)
        async_shutdown_ev.set()


async def serve_service_core_task(server_core: ProtobufServiceCore,
                                  port=50052,
                                  rpc_name="/api/rpc",
                                  ws_name="/api/ws/{client_id}",
                                  is_sync: bool = False,
                                  rpc_pickle_name: str = "/api/rpc_pickle",
                                  client_max_size: int = 4 * 1024**2,
                                  standalone: bool = True,
                                  ssl_key_path: str = "",
                                  ssl_crt_path: str = "",
                                  ws_backup_name="/api/ws_backup/{client_id}"):
    http_service = HttpService(server_core)
    ctx = contextlib.nullcontext()
    if standalone:
        ctx = server_core.enter_global_context()
    with ctx:
        if standalone:
            await server_core._init_async_members()
        ws_service = BlacksheepWebsocketHandler(server_core, client_max_size)
        app = web.Application()
        # TODO should we create a global client session for all http call in server?
        loop_task = asyncio.create_task(ws_service.event_provide_executor())
        @app.router.post(rpc_name)
        async def _handle_rpc(request):
            return await http_service.remote_json_call_http(request)

        @app.router.post(rpc_pickle_name)
        async def _handle_rpc_pkl(request):
            return await http_service.remote_pickle_call_http(request)

        @app.router.post(TENSORPC_API_FILE_UPLOAD)
        async def _handle_rpc_file(request):
            return await http_service.file_upload_call(request)
        
        @app.router.get(TENSORPC_FETCH_STATUS)
        async def _fetch_status(request):
            return await http_service.fetch_status(request)

        @app.router.ws(ws_name)
        async def _handle_new_connection(request, client_id):
            return await ws_service.handle_new_connection_blacksheep(request, client_id)

        @app.router.ws(ws_backup_name)
        async def _handle_new_backup_connection(request, client_id):
            return await ws_service.handle_new_backup_connection_blacksheep(request, client_id)

        config = uvicorn.Config(app,
                                port=port,
                                log_level="warning",
                                ws_max_size=client_max_size,
                                ssl_keyfile=ssl_key_path,
                                ssl_certfile=ssl_crt_path)
        server = uvicorn.Server(config)

        return await asyncio.gather(
            serve_app(server, server_core.shutdown_event,
                      server_core.async_shutdown_event, is_sync), loop_task)
