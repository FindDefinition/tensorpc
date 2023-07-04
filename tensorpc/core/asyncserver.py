# Copyright 2022 Yan Yan
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
"""The Python implementation of the GRPC RemoteCall.RemoteObject server."""

import asyncio
import json
import os
import time
from functools import partial
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import grpc
import grpc.aio
import numpy as np

from tensorpc import compat
from tensorpc.core.defs import ServiceDef
from tensorpc.core.server_core import ProtobufServiceCore, ServerMeta
from tensorpc.protos_export import remote_object_pb2 as remote_object_pb2
from tensorpc.protos_export import rpc_message_pb2
from tensorpc.protos_export import \
    remote_object_pb2_grpc as remote_object_pb2_grpc
from tensorpc.utils.df_logging import get_logger
from tensorpc.core.httpservers import aiohttp_impl as httpserver
# from tensorpc.core.httpservers import blacksheep_impl as httpserver
import aiohttp

LOGGER = get_logger()

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class AsyncRemoteObjectService(remote_object_pb2_grpc.RemoteObjectServicer):
    """Main service of codeai.distributed. Arbitrary python code execute service.
    """

    # TODO when too much stdout in server, logger may crash.
    # TODO add option to disable dynamic code add
    # TODO support regular load modules
    # TODO when nested RPC, logger crash
    def __init__(self, server_core: ProtobufServiceCore, is_local, length=-1):
        super().__init__()
        self.is_local = is_local
        self.length = length
        self.server_core = server_core

    async def QueryServerMeta(self, request, context):
        meta = {
            "is_local": self.is_local,
            "service_metas": self.server_core.get_service_meta(),
            "message_max_length": self.length,
        }
        return rpc_message_pb2.SimpleReply(data=json.dumps(meta))

    async def QueryServiceMeta(self, request, context):
        service_key = request.service_key
        _, meta = self.server_core.service_units.get_service_and_meta(
            service_key)
        return rpc_message_pb2.SimpleReply(data=json.dumps(meta.to_json()))

    async def RemoteJsonCall(self, request, context):
        res = await self.server_core.remote_json_call_async(request)
        return res

    async def RemoteCall(self, request, context):
        res = await self.server_core.remote_call_async(request)
        return res

    async def RemoteGenerator(self, request, context):
        async for res in self.server_core.remote_generator_async(request):
            yield res

    async def ChunkedRemoteCall(self, request_iter, context):
        async for res in self.server_core.chunked_remote_call_async(
                request_iter):
            yield res

    async def RemoteStreamCall(self, request_iter, context):
        async for res in self.server_core.remote_stream_call_async(
                request_iter):
            yield res

    async def ClientStreamRemoteCall(self, request_iter, context):
        return await self.server_core.client_stream_async(request_iter)

    async def BiStreamRemoteCall(self, request_iter, context):
        async for res in self.server_core.bi_stream_async(request_iter):
            yield res

    def ServerShutdown(self, request, context):
        print("Shutdown message received")
        self.server_core._reset_timeout()
        context.add_callback(
            lambda: self.server_core.async_shutdown_event.set())
        return rpc_message_pb2.SimpleReply()

    async def HealthCheck(self, request, context):
        self.server_core._reset_timeout()
        return rpc_message_pb2.SimpleReply(data="{}")

    async def SayHello(self, request, context):
        return rpc_message_pb2.HelloReply(data=request.data)


async def _await_thread_ev(ev, loop, timeout=None):
    waiter = partial(ev.wait, timeout=timeout)
    return await loop.run_in_executor(None, waiter)

# https://github.com/grpc/grpc/blob/master/examples/python/helloworld/async_greeter_server_with_graceful_shutdown.py
# Coroutines to be invoked when the event loop is shutting down.
_cleanup_coroutines = []


async def serve_service(service: AsyncRemoteObjectService,
                        wait_time=-1,
                        port=50051,
                        length=-1,
                        is_local=False,
                        max_threads=10,
                        process_id=-1,
                        ssl_key_path: str = "",
                        ssl_crt_path: str = ""):
    assert isinstance(service, AsyncRemoteObjectService)
    if is_local and process_id >= 0:
        if hasattr(os, "sched_setaffinity"):
            # lock process to cpu to increase performance.
            LOGGER.info("lock worker {} to core {}".format(
                process_id, process_id))
            os.sched_setaffinity(0, [process_id])
    wait_interval = _ONE_DAY_IN_SECONDS
    if wait_time > 0:
        wait_interval = wait_time
    options = None
    if length > 0:
        options = [('grpc.max_send_message_length', length * 1024 * 1024),
                   ('grpc.max_receive_message_length', length * 1024 * 1024)]
    server = grpc.aio.server(options=options)
    remote_object_pb2_grpc.add_RemoteObjectServicer_to_server(service, server)
    url = '[::]:{}'.format(port)
    credentials = None
    if ssl_key_path != "" and ssl_key_path != "":
        with open(ssl_key_path, "rb") as f:
            private_key = f.read()
        with open(ssl_crt_path, "rb") as f:
            certificate_chain = f.read()
        credentials = grpc.ssl_server_credentials([(private_key,
                                                    certificate_chain)])
    if credentials is not None:
        server.add_secure_port(url, credentials)
    else:
        server.add_insecure_port(url)

    await server.start()
    loop = asyncio.get_running_loop()
    server_core = service.server_core

    async def server_graceful_shutdown():
        # Shuts down the server with 5 seconds of grace period. During the
        # grace period, the server won't accept new connections and allow
        # existing RPCs to continue within the grace period.
        await server.stop(5)
    _cleanup_coroutines.append(server_graceful_shutdown())
    await server_core.async_shutdown_event.wait()

    # while True:
    #     looks like event return false instead of raise timeouterror
    #     if await _await_thread_ev(server_core.shutdown_event, loop,
    #                               wait_interval):
    #         break
    #     with server_core.reset_timeout_lock:
    #         interval = time.time() - server_core.latest_active_time
    #         if wait_time > 0 and interval > wait_time:
    #             break
    await server.stop(0)
    await server.wait_for_termination()
    # exec cleanup functions
    await server_core.exec_exit_funcs()


# def serve_with_http(service_def: ServiceDef,
#                     wait_time=-1,
#                     port=50051,
#                     http_port=50052,
#                     length=-1,
#                     is_local=False,
#                     max_threads=10,
#                     process_id=-1,
#                     credentials=None):
#     if not compat.Python3_7AndLater:
#         raise NotImplementedError
#     from distflow.core import httpserver
#     # run grpc server in background, and ws in main
#     url = '[::]:{}'.format(port)
#     # loop = asyncio.get_running_loop()
#     server_core = ProtobufServiceCore(url, service_def, loop=None)
#     service = AsyncRemoteObjectService(server_core, is_local, length)
#     grpc_task = serve_service(service, wait_time, port, length, is_local,
#                               max_threads, process_id, credentials)
#     http_task = httpserver.serve_service_core_task(server_core, http_port,
#                                                    None, is_sync=False)
#     coro = asyncio.gather(grpc_task, http_task)
#     try:
#         asyncio.run(coro)
#         # loop.run_until_complete(coro)
#     except KeyboardInterrupt:
#         server_core.shutdown_event.set()
#         server_core.async_shutdown_event.set()
#         # set shutdown ev and resume previous task.
#         # loop.run_until_complete(coro)
#         print("shutdown by keyboard interrupt")


async def serve_with_http_async(server_core: ProtobufServiceCore,
                                url: str,
                                wait_time=-1,
                                port=50051,
                                http_port=50052,
                                length=-1,
                                is_local=False,
                                max_threads=10,
                                process_id=-1,
                                ssl_key_path: str = "",
                                ssl_crt_path: str = ""):
    if not compat.Python3_7AndLater:
        raise NotImplementedError
    smeta = ServerMeta(port=port, http_port=http_port)

    # server_core = ProtobufServiceCore(url, service_def, False, smeta)
    async with aiohttp.ClientSession() as sess:
        server_core.init_http_client_session(sess)

        url = '[::]:{}'.format(port)
        with server_core.enter_global_context():
            await server_core._init_async_members()

            service = AsyncRemoteObjectService(server_core, is_local, length)
            grpc_task = serve_service(service, wait_time, port, length,
                                      is_local, max_threads, process_id,
                                      ssl_key_path, ssl_crt_path)
            http_task = httpserver.serve_service_core_task(
                server_core,
                http_port,
                is_sync=False,
                standalone=False,
                ssl_key_path=ssl_key_path,
                ssl_crt_path=ssl_crt_path)
            return await asyncio.gather(grpc_task, http_task)


async def run_exit_async(server_core: ProtobufServiceCore):
    async with aiohttp.ClientSession() as sess:
        server_core.init_http_client_session(sess)
        with server_core.enter_global_context():
            await server_core.exec_exit_funcs()


async def serve_async(sc: ProtobufServiceCore,
                      wait_time=-1,
                      port=50051,
                      length=-1,
                      is_local=False,
                      max_threads=10,
                      process_id=-1,
                      ssl_key_path: str = "",
                      ssl_crt_path: str = ""):
    if not compat.Python3_7AndLater:
        raise NotImplementedError

    server_core = sc
    with server_core.enter_global_context():
        await server_core._init_async_members()
        service = AsyncRemoteObjectService(server_core, is_local, length)
        grpc_task = serve_service(service, wait_time, port, length, is_local,
                                  max_threads, process_id, ssl_key_path,
                                  ssl_crt_path)
                                  
        return await grpc_task


def serve(service_def: ServiceDef,
          wait_time=-1,
          port=50051,
          length=-1,
          is_local=False,
          max_threads=10,
          process_id=-1,
          ssl_key_path: str = "",
          ssl_crt_path: str = ""):
    if not compat.Python3_7AndLater:
        raise NotImplementedError
    url = '[::]:{}'.format(port)
    smeta = ServerMeta(port=port, http_port=-1)
    server_core = ProtobufServiceCore(url, service_def, False, smeta)
    loop = asyncio.get_event_loop()

    try:
        loop.run_until_complete(
            serve_async(server_core,
                        port=port,
                        length=length,
                        is_local=is_local,
                        max_threads=max_threads,
                        process_id=process_id,
                        ssl_key_path=ssl_key_path,
                        ssl_crt_path=ssl_crt_path))
    except KeyboardInterrupt:
        loop.run_until_complete(run_exit_async(server_core))
        if _cleanup_coroutines:
            loop.run_until_complete(*_cleanup_coroutines)
        print("shutdown by keyboard interrupt")


# import uvloop
def serve_with_http(service_def: ServiceDef,
                    wait_time=-1,
                    port=50051,
                    http_port=50052,
                    length=-1,
                    is_local=False,
                    max_threads=10,
                    process_id=-1,
                    ssl_key_path: str = "",
                    ssl_crt_path: str = ""):
    if not compat.Python3_7AndLater:
        raise NotImplementedError
    url = '[::]:{}'.format(port)
    smeta = ServerMeta(port=port, http_port=http_port)
    server_core = ProtobufServiceCore(url, service_def, False, smeta)
    loop = asyncio.get_event_loop()
    try:
        # uvloop.install()
        # print("UVLOOP")
        loop.run_until_complete(
            serve_with_http_async(server_core,
                                  url,
                                  wait_time=wait_time,
                                  port=port,
                                  http_port=http_port,
                                  length=length,
                                  is_local=is_local,
                                  max_threads=max_threads,
                                  process_id=process_id,
                                  ssl_key_path=ssl_key_path,
                                  ssl_crt_path=ssl_crt_path))
    except KeyboardInterrupt:
        loop.run_until_complete(run_exit_async(server_core))
        if _cleanup_coroutines:
            loop.run_until_complete(*_cleanup_coroutines)
        print("shutdown by keyboard interrupt")
