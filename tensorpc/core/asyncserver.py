# Copyright 2024 Yan Yan
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
import threading
import time
from functools import partial
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import grpc
import grpc.aio
import traceback
import numpy as np

from tensorpc import compat
from tensorpc.constants import TENSORPC_PORT_MAX_TRY
from tensorpc.core.defs import ServiceDef
from tensorpc.core.server_core import ProtobufServiceCore, ServerMeta
from tensorpc.core.serviceunit import ServiceEventType
from tensorpc.protos_export import remote_object_pb2 as remote_object_pb2
from tensorpc.protos_export import rpc_message_pb2
from tensorpc.protos_export import \
    remote_object_pb2_grpc as remote_object_pb2_grpc
from tensorpc.utils.rich_logging import get_logger
from tensorpc.core.httpservers import aiohttp_impl as httpserver
# from tensorpc.core.httpservers import blacksheep_impl as httpserver
import aiohttp

from tensorpc.utils.wait_tools import get_free_ports

LOGGER = get_logger("tensorpc.aioserver")

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
        try:
            service_key = request.service_key
            meta = self.server_core.service_units.get_service_meta_only(
                service_key)
        except:
            traceback.print_exc()
            raise
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

    async def ChunkedRemoteCall(self, request_iterator, context):
        async for res in self.server_core.chunked_remote_call_async(
                request_iterator):
            yield res

    async def RemoteStreamCall(self, request_iterator, context):
        async for res in self.server_core.remote_stream_call_async(
                request_iterator):
            yield res

    async def ClientStreamRemoteCall(self, request_iterator, context):
        return await self.server_core.client_stream_async(request_iterator)

    async def BiStreamRemoteCall(self, request_iterator, context):
        async for res in self.server_core.bi_stream_async(request_iterator):
            yield res

    async def ChunkedBiStreamRemoteCall(self, request_iterator, context):
        async for res in self.server_core.chunked_bi_stream_async(
                request_iterator):
            yield res

    async def ChunkedRemoteGenerator(self, request_iterator, context):
        async for res in self.server_core.chunked_remote_generator_async(
                request_iterator):
            yield res

    async def ChunkedClientStreamRemoteCall(self, request_iterator, context):
        async for res in self.server_core.chunked_client_stream_async(
                request_iterator):
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


async def serve_service(
        service: AsyncRemoteObjectService,
        wait_time=-1,
        port=50051,
        length=-1,
        is_local=False,
        max_threads=10,
        process_id=-1,
        ssl_key_path: str = "",
        ssl_crt_path: str = "",
        grpc_options: Optional[List[Tuple[str, Union[str, int]]]] = None,
        start_thread_ev: Optional[threading.Event] = None,
        max_port_retry: int = TENSORPC_PORT_MAX_TRY):
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
    options = []
    if length > 0:
        options = [('grpc.max_send_message_length', length * 1024 * 1024),
                   ('grpc.max_receive_message_length', length * 1024 * 1024)]
    options.append(('grpc.so_reuseport', 0))
    if grpc_options is not None:
        options = grpc_options  # override
    server = grpc.aio.server(options=options)
    remote_object_pb2_grpc.add_RemoteObjectServicer_to_server(service, server)
    credentials = None
    if ssl_key_path != "" and ssl_key_path != "":
        with open(ssl_key_path, "rb") as f:
            private_key = f.read()
        with open(ssl_crt_path, "rb") as f:
            certificate_chain = f.read()
        credentials = grpc.ssl_server_credentials([(private_key,
                                                    certificate_chain)])

    for i in range(max_port_retry):
        if port == -1:
            port = get_free_ports(1)[0]
        url = '[::]:{}'.format(port)
        try:
            if credentials is not None:
                server.add_secure_port(url, credentials)
            else:
                server.add_insecure_port(url)
            LOGGER.warning("server started at {}".format(url))
            break
        except:
            traceback.print_exc()
            port = -1
    if port == -1:
        raise RuntimeError("Cannot find free port")
    server_core = service.server_core
    server_core._set_port(port)
    await server_core.run_event_async(ServiceEventType.BeforeServerStart)
    if start_thread_ev is not None:
        start_thread_ev.set()
    await server.start()
    loop = asyncio.get_running_loop()

    async def server_graceful_shutdown():
        # Shuts down the server with 5 seconds of grace period. During the
        # grace period, the server won't accept new connections and allow
        # existing RPCs to continue within the grace period.
        await server.stop(5)

    _cleanup_coroutines.append(server_graceful_shutdown())
    await server_core.async_shutdown_event.wait()
    await server.stop(0)
    await server.wait_for_termination()
    # exec cleanup functions
    # this may be run in keyboard handler.
    async with server_core._shutdown_handler_lock:
        if not server_core._is_exit_async_run:
            await server_core.run_event_async(ServiceEventType.Exit)
            server_core._is_exit_async_run = True


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
                                ssl_crt_path: str = "",
                                max_port_retry: int = TENSORPC_PORT_MAX_TRY):
    smeta = ServerMeta(port=port, http_port=http_port)

    # server_core = ProtobufServiceCore(url, service_def, False, smeta)
    async with aiohttp.ClientSession() as sess:
        server_core.init_http_client_session(sess)

        url = '[::]:{}'.format(port)
        with server_core.enter_global_context():
            with server_core.enter_exec_context():
                await server_core._init_async_members()
                await server_core.run_event_async(ServiceEventType.Init)
            service = AsyncRemoteObjectService(server_core, is_local, length)
            grpc_task = serve_service(service, wait_time, port, length,
                                      is_local, max_threads, process_id,
                                      ssl_key_path, ssl_crt_path,
                                      max_port_retry=max_port_retry)
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
            with server_core.enter_exec_context():
                server_core.async_shutdown_event.set()
                async with server_core._shutdown_handler_lock:
                    if not server_core._is_exit_async_run:
                        await server_core.run_event_async(ServiceEventType.Exit)
                        server_core._is_exit_async_run = True

async def serve_async(sc: ProtobufServiceCore,
                      wait_time=-1,
                      port=50051,
                      length=-1,
                      is_local=False,
                      max_threads=10,
                      process_id=-1,
                      ssl_key_path: str = "",
                      ssl_crt_path: str = "",
                      start_thread_ev: Optional[threading.Event] = None,
                      max_port_retry: int = TENSORPC_PORT_MAX_TRY):
    server_core = sc
    with server_core.enter_global_context():
        with server_core.enter_exec_context():
            await server_core._init_async_members()
            await server_core.run_event_async(ServiceEventType.Init)
        service = AsyncRemoteObjectService(server_core, is_local, length)
        grpc_task = serve_service(service, wait_time, port, length, is_local,
                                  max_threads, process_id, ssl_key_path,
                                  ssl_crt_path, start_thread_ev=start_thread_ev,
                                  max_port_retry=max_port_retry)

        return await grpc_task


def serve(service_def: ServiceDef,
          wait_time=-1,
          port=50051,
          length=-1,
          is_local=False,
          max_threads=10,
          process_id=-1,
          ssl_key_path: str = "",
          ssl_crt_path: str = "",
          create_loop: bool = False,
          max_port_retry: int = TENSORPC_PORT_MAX_TRY):
    url = '[::]:{}'.format(port)
    smeta = ServerMeta(port=port, http_port=-1)
    server_core = ProtobufServiceCore(url, service_def, False, smeta)
    return serve_service_core(server_core, wait_time, length, is_local,
                              max_threads, process_id, ssl_key_path, ssl_crt_path,
                              max_port_retry=max_port_retry)

def serve_service_core(service_core: ProtobufServiceCore,
          wait_time=-1,
          length=-1,
          is_local=False,
          max_threads=10,
          process_id=-1,
          ssl_key_path: str = "",
          ssl_crt_path: str = "",
          create_loop: bool = False,
          start_thread_ev: Optional[threading.Event] = None,
          max_port_retry: int = TENSORPC_PORT_MAX_TRY):
    # url = '[::]:{}'.format(port)
    # smeta = ServerMeta(port=port, http_port=-1)
    # server_core = ProtobufServiceCore(url, service_def, False, smeta)
    if create_loop:
        loop = asyncio.new_event_loop()
    else:
        loop = asyncio.get_event_loop()
    try:
        service_core._loop = loop
        loop.run_until_complete(
            serve_async(service_core,
                        port=service_core.server_meta.port,
                        length=length,
                        is_local=is_local,
                        max_threads=max_threads,
                        process_id=process_id,
                        ssl_key_path=ssl_key_path,
                        ssl_crt_path=ssl_crt_path,
                        start_thread_ev=start_thread_ev,
                        max_port_retry=max_port_retry))
    except KeyboardInterrupt:
        loop.run_until_complete(run_exit_async(service_core))
        print("shutdown by keyboard interrupt")
    finally:
        if _cleanup_coroutines:
            loop.run_until_complete(*_cleanup_coroutines)
            _cleanup_coroutines.pop()
        service_core._loop = None
        if create_loop:
            loop.close()


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
                    ssl_crt_path: str = "",
                    max_port_retry: int = TENSORPC_PORT_MAX_TRY):
    url = '[::]:{}'.format(port)
    smeta = ServerMeta(port=port, http_port=http_port)
    server_core = ProtobufServiceCore(url, service_def, False, smeta)
    loop = asyncio.get_event_loop()
    try:
        # uvloop.install()
        # print("UVLOOP")
        server_core._loop = loop
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
                                  ssl_crt_path=ssl_crt_path,
                                  max_port_retry=max_port_retry))
    except KeyboardInterrupt:
        loop.run_until_complete(run_exit_async(server_core))
        print("shutdown by keyboard interrupt")
    finally:
        if _cleanup_coroutines:
            loop.run_until_complete(*_cleanup_coroutines)
        server_core._loop = None 
