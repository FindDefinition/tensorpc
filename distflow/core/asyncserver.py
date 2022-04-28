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
import numpy as np

from distflow import compat
from distflow.core.defs import ServiceDef
from distflow.core.server_core import ProtobufServiceCore
from distflow.protos import remote_object_pb2 as remote_object_pb2
from distflow.protos import rpc_message_pb2

from distflow.protos import \
    remote_object_pb2_grpc as remote_object_pb2_grpc
from distflow.utils.df_logging import get_logger

LOGGER = get_logger()

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class AsyncRemoteObjectService(remote_object_pb2_grpc.RemoteObjectServicer):
    """Main service of codeai.distributed. Arbitrary python code execute service.
    """

    # TODO: try C++ server (still wait for sub-interpreters)
    # TODO when too much stdout in server, logger may crash.
    # TODO make methods in dymodule pickleable
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

    def Shutdown(self, request, context):
        print("Shutdown message received")
        self.server_core._reset_timeout()
        context.add_callback(lambda: self.server_core.async_shutdown_event.set())
        return rpc_message_pb2.SimpleReply()

    async def HealthCheck(self, request, context):
        self.server_core._reset_timeout()
        return rpc_message_pb2.SimpleReply(data="{}")

    async def SayHello(self, request, context):
        return rpc_message_pb2.HelloReply(data=request.data)


async def _await_thread_ev(ev, loop, timeout=None):
    waiter = partial(ev.wait, timeout=timeout)
    return await loop.run_in_executor(None, waiter)


async def serve_service(service: AsyncRemoteObjectService,
                        wait_time=-1,
                        port=50051,
                        length=-1,
                        is_local=False,
                        max_threads=10,
                        process_id=-1,
                        credentials=None):
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
        options = [('grpc.max_message_length', length * 1024 * 1024),
                   ('grpc.max_receive_message_length', length * 1024 * 1024)]
    server = grpc.aio.server(options=options)
    remote_object_pb2_grpc.add_RemoteObjectServicer_to_server(service, server)
    url = '[::]:{}'.format(port)

    if credentials is not None:
        server.add_secure_port(url, credentials)
    else:
        server.add_insecure_port(url)

    await server.start()
    loop = asyncio.get_running_loop()
    server_core = service.server_core
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
    # exec cleanup functions
    server_core.exec_exit_funcs()



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

async def serve_with_http_async(service_def: ServiceDef,
                    url: str,
                    wait_time=-1,
                    port=50051,
                    http_port=50052,
                    length=-1,
                    is_local=False,
                    max_threads=10,
                    process_id=-1,
                    credentials=None):
    if not compat.Python3_7AndLater:
        raise NotImplementedError
    server_core = ProtobufServiceCore(url, service_def)

    from distflow.core import httpserver
    url = '[::]:{}'.format(port)
    server_core._init_async_members()
    service = AsyncRemoteObjectService(server_core, is_local, length)
    grpc_task = serve_service(service, wait_time, port, length, is_local,
                              max_threads, process_id, credentials)
    http_task = httpserver.serve_service_core_task(server_core, http_port,
                                                   None, is_sync=False)
    return await asyncio.gather(grpc_task, http_task)

async def serve_async(service_def: ServiceDef,
        wait_time=-1,
        port=50051,
        length=-1,
        is_local=False,
        max_threads=10,
        process_id=-1,
        credentials=None):
    if not compat.Python3_7AndLater:
        raise NotImplementedError

    from distflow.core import httpserver
    url = '[::]:{}'.format(port)
    server_core = ProtobufServiceCore(url, service_def)
    server_core._init_async_members()
    service = AsyncRemoteObjectService(server_core, is_local, length)
    grpc_task = serve_service(service, wait_time, port, length, is_local,
                              max_threads, process_id, credentials)
    return await grpc_task

def serve(service_def: ServiceDef,
          wait_time=-1,
          port=50051,
          length=-1,
          is_local=False,
          max_threads=10,
          process_id=-1,
          credentials=None):
    if not compat.Python3_7AndLater:
        raise NotImplementedError
    try:
        asyncio.run(serve_async(service_def, 
            port=port, length=length, is_local=is_local,
            max_threads=max_threads, process_id=process_id, credentials=credentials))
    except KeyboardInterrupt:
        print("shutdown by keyboard interrupt")

def serve_with_http(service_def: ServiceDef,
                    wait_time=-1,
                    port=50051,
                    http_port=50052,
                    length=-1,
                    is_local=False,
                    max_threads=10,
                    process_id=-1,
                    credentials=None):
    if not compat.Python3_7AndLater:
        raise NotImplementedError
    url = '[::]:{}'.format(port)
    try:
        asyncio.run(serve_with_http_async(service_def, url, wait_time=wait_time, 
            port=port, http_port=http_port, length=length, is_local=is_local,
            max_threads=max_threads, process_id=process_id, credentials=credentials))
    except KeyboardInterrupt:
        print("shutdown by keyboard interrupt")
