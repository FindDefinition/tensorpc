import asyncio
import contextlib
import io
import json
import threading
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import aiohttp
from aiohttp import web

from tensorpc.core import core_io, defs
import ssl
from tensorpc.core.asynctools import cancel_task
from tensorpc.core.constants import TENSORPC_API_FILE_DOWNLOAD, TENSORPC_API_FILE_UPLOAD, TENSORPC_FETCH_STATUS
from tensorpc.core.server_core import ProtobufServiceCore, ServiceCore, ServerMeta
from pathlib import Path
from tensorpc.protos_export import remote_object_pb2
from tensorpc.protos_export import remote_object_pb2 as remote_object_pb2
from tensorpc.protos_export import rpc_message_pb2
from contextlib import suppress
from aiohttp import streamer


@streamer
async def file_sender(writer, file_bytes: bytes, chunk_size=2**16):
    """
    This function will read large file chunk by chunk and send it through HTTP
    without reading them into memory
    """
    bio = io.BytesIO(file_bytes)
    chunk = bio.read(2 ** 16)
    while chunk:
        await writer.write(chunk)
        chunk = bio.read(2 ** 16)

@streamer
async def grpc_iter_file_sender(writer, reader):
    """
    This function will read large file chunk by chunk and send it through HTTP
    without reading them into memory
    """
    async for chunk, is_exc in reader:
        if is_exc:
            raise ValueError(chunk)
        await writer.write(chunk)


from .core import WebsocketClientBase, WebsocketMsg, WebsocketMsgType, WebsocketHandler
async def _cancel(task):
    # more info: https://stackoverflow.com/a/43810272/1113207
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task

class AiohttpWebsocketClient(WebsocketClientBase):
    def __init__(self,
                 id: str,
                 ws: web.WebSocketResponse,
                 serv_id_to_name: Dict[int, str],
                 uid: Optional[int] = None):
        super().__init__(id, serv_id_to_name, uid)
        self.ws = ws

    async def close(self):
        return await self.ws.close()

    def get_msg_max_size(self) -> int: 
        return self.ws._max_msg_size

    async def send_bytes(self, data: bytes): 
        return await self.ws.send_bytes(data)


    def get_client_id(self) -> int: 
        return id(self.ws)

    async def binary_msg_generator(self, shutdown_ev: asyncio.Event) -> AsyncGenerator[WebsocketMsg, None]: 
        # while True:
        #     recv_task = asyncio.create_task(self.ws.receive(), name="recv_task")
        #     st_task = asyncio.create_task(shutdown_ev.wait(), name="shutdown_task")
        #     done, pending = await asyncio.wait([recv_task, st_task], return_when=asyncio.FIRST_COMPLETED)
        #     if st_task in done:
        #         # cancel recv task 
        #         await cancel_task(recv_task)
        #         break
        #     else:
        #         msg = recv_task.result()
        #         if msg.type == aiohttp.WSMsgType.BINARY:
        #             yield WebsocketMsg(msg.data, WebsocketMsgType.Binary)
        #         elif msg.type == aiohttp.WSMsgType.TEXT:
        #             yield WebsocketMsg(msg.data, WebsocketMsgType.Text)
        #         elif msg.type == aiohttp.WSMsgType.ERROR:
        #             raise Exception("websocket connection closed with exception %s" %
        #                             self.ws.exception())
        #         recv_task = asyncio.create_task(self.ws.receive(), name="recv_task")
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
        client_id = request.match_info.get('client_id')
        print("NEW CONN", client_id, request)
        service_core = self.service_core
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        client = AiohttpWebsocketClient(client_id, 
            ws, service_core.service_units.get_service_id_to_name())
        await self.handle_new_connection(client, client_id)
        return ws

    async def handle_new_backup_connection_aiohttp(self, request):
        client_id = request.match_info.get('client_id')
        print("NEW BACKUP CONN", client_id, request)
        service_core = self.service_core
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        client = AiohttpWebsocketClient(client_id, 
            ws, service_core.service_units.get_service_id_to_name())
        await self.handle_new_connection(client, client_id, True)
        return ws

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
    
    async def fetch_status(self, request: web.Request):
        status = {
            "status": "ok",
        }
        # TODO better headers
        headers = {
            'Access-Control-Allow-Origin': '*',
            # 'Access-Control-Allow-Headers': '*',
            # 'Access-Control-Allow-Method': 'POST',
        }
        res = web.json_response(status, headers=headers)
        return res
    

    async def resource_download_call(self, request: web.Request):
        params = request.rel_url.query
        node_uid = params.get('nodeUid')
        resource_key = params.get('key')
        headers = {
            'Access-Control-Allow-Origin': '*',
            "Content-Disposition": f"Attachment;filename={resource_key}",
            # 'Access-Control-Allow-Headers': '*',
            # 'Access-Control-Allow-Method': 'POST',
        }
        if node_uid is not None and resource_key is not None:
            ait = self.service_core.execute_async_generator_service(
                "tensorpc.flow.serv.core::Flow.app_get_file", [node_uid, resource_key], {}, json_call=False)
            desp, is_exc = await ait.__anext__()
            if is_exc:
                return web.Response(status=500, text=desp, headers=headers)
            assert isinstance(desp, defs.FileDesp)
            headers["Content-Disposition"] = f"Attachment;filename={desp.name}"
            if desp.content_type is not None:
                headers["Content-Type"] = desp.content_type
            return web.Response(
                body=grpc_iter_file_sender(reader=ait),
                headers=headers
            )
        else:
            raise web.HTTPBadRequest(text="nodeUid or key is None")


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
                                  ws_name="/api/ws/{client_id}",
                                  is_sync: bool = False,
                                  rpc_pickle_name: str = "/api/rpc_pickle",
                                  client_max_size: int = 16 * 1024**2,
                                  standalone: bool = True,
                                  ssl_key_path: str = "",
                                  ssl_crt_path: str = "",
                                  ws_backup_name="/api/ws_backup/{client_id}"):
    # client_max_size 4MB is enough for most image upload.
    http_service = HttpService(server_core)
    ctx = contextlib.nullcontext()
    if standalone:
        ctx = server_core.enter_global_context()
    with ctx:
        if standalone:
            await server_core._init_async_members()
        ws_service = AiohttpWebsocketHandler(server_core)
        # print("???????", client_max_size)
        app = web.Application(client_max_size=client_max_size)
        # TODO should we create a global client session for all http call in server?
        loop_task = asyncio.create_task(ws_service.event_provide_executor())
        app.router.add_post(rpc_name, http_service.remote_json_call_http)
        app.router.add_post(rpc_pickle_name,
                            http_service.remote_pickle_call_http)
        app.router.add_post(TENSORPC_API_FILE_UPLOAD, http_service.file_upload_call)
        app.router.add_get(TENSORPC_API_FILE_DOWNLOAD, http_service.resource_download_call)
        app.router.add_get(TENSORPC_FETCH_STATUS, http_service.fetch_status)

        app.router.add_get(ws_name, ws_service.handle_new_connection_aiohttp)
        app.router.add_get(ws_backup_name, ws_service.handle_new_backup_connection_aiohttp)

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
