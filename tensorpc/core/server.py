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
import threading
import time
import traceback
from concurrent import futures
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import grpc
import numpy as np
from tensorpc.core.defs import ServiceDef

from tensorpc.core.server_core import ProtobufServiceCore, ServerMeta

from tensorpc.protos_export import remote_object_pb2 as remote_object_pb2
from tensorpc.protos_export import rpc_message_pb2

from tensorpc.protos_export import \
    remote_object_pb2_grpc as remote_object_pb2_grpc
from tensorpc.utils.df_logging import get_logger

LOGGER = get_logger()

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class RemoteObjectService(remote_object_pb2_grpc.RemoteObjectServicer):
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

    def QueryServerMeta(self, request, context):
        meta = {
            "is_local": self.is_local,
            "service_metas": self.server_core.get_service_meta(),
            "message_max_length": self.length,
        }
        return rpc_message_pb2.SimpleReply(data=json.dumps(meta))

    def QueryServiceMeta(self, request, context):
        service_key = request.service_key
        _, meta = self.server_core.service_units.get_service_and_meta(
            service_key)
        return rpc_message_pb2.SimpleReply(data=json.dumps(meta.to_json()))

    def RemoteJsonCall(self, request, context):
        res = self.server_core.remote_json_call(request)
        if request.callback != "":
            callback = lambda: self.server_core.service_units.get_service(
                request.callback)()
            context.add_callback(callback)
        return res

    def RemoteCall(self, request, context):
        res = self.server_core.remote_call(request)
        if request.callback != "":
            callback = lambda: self.server_core.service_units.get_service(
                request.callback)()
            context.add_callback(callback)
        return res

    def RemoteGenerator(self, request, context):
        yield from self.server_core.remote_generator(request)
        if request.callback != "":
            callback = lambda: self.server_core.service_units.get_service(
                request.callback)()
            context.add_callback(callback)

    def ChunkedRemoteCall(self, request_iter, context):
        yield from self.server_core.chunked_remote_call(request_iter)

    def RemoteStreamCall(self, request_iter, context):
        yield from self.server_core.remote_stream_call(request_iter)

    def ClientStreamRemoteCall(self, request_iter, context):
        return self.server_core.client_stream(request_iter)

    def BiStreamRemoteCall(self, request_iter, context):
        yield from self.server_core.bi_stream(request_iter)

    def ServerShutdown(self, request, context):
        print("Shutdown message received")
        self.server_core._reset_timeout()

        def shutdown_cb():
            self.server_core.shutdown_event.set()

        context.add_callback(shutdown_cb)
        return rpc_message_pb2.SimpleReply()

    def HealthCheck(self, request, context):
        self.server_core._reset_timeout()
        return rpc_message_pb2.SimpleReply(data="{}")

    def SayHello(self, request, context):
        return rpc_message_pb2.HelloReply(data=request.data)


def serve_service(service: RemoteObjectService,
                  wait_time=-1,
                  port=50051,
                  length=-1,
                  is_local=False,
                  max_threads=10,
                  process_id=-1,
                  credentials=None):
    assert isinstance(service, RemoteObjectService)
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
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_threads),
                         options=options)
    url = '[::]:{}'.format(port)
    remote_object_pb2_grpc.add_RemoteObjectServicer_to_server(service, server)
    if credentials is not None:
        server.add_secure_port(url, credentials)
    else:
        server.add_insecure_port(url)
    server.start()
    server_core = service.server_core
    try:
        while True:
            # looks like event return false instead of raise timeouterror
            if server_core.shutdown_event.wait(wait_interval):
                break
            with server_core.reset_timeout_lock:
                interval = time.time() - server_core.latest_active_time
                if wait_time > 0 and interval > wait_time:
                    break
        server.stop(0)
        # exec cleanup functions
        server_core.exec_exit_funcs_sync()
        LOGGER.info("server closed")
    except KeyboardInterrupt:
        server_core.exec_exit_funcs_sync()
        server.stop(0)
        LOGGER.info("server shutdown by keyboard interrupt")


def serve(service_def: ServiceDef,
          wait_time=-1,
          port=50051,
          length=-1,
          is_local=False,
          max_threads=10,
          process_id=-1,
          credentials=None):
    url = '[::]:{}'.format(port)
    smeta = ServerMeta(port=port, http_port=-1)
    server_core = ProtobufServiceCore(url, service_def, True, smeta)
    service = RemoteObjectService(server_core, is_local, length)
    return serve_service(service, wait_time, port, length, is_local,
                         max_threads, process_id, credentials)


def serve_with_http(service_def: ServiceDef,
                    wait_time=-1,
                    port=50051,
                    http_port=50052,
                    length=-1,
                    is_local=False,
                    max_threads=10,
                    process_id=-1,
                    credentials=None):
    from tensorpc.core.httpservers import aiohttp_impl as httpserver

    # run grpc server in background, and ws in main
    url = '[::]:{}'.format(port)
    smeta = ServerMeta(port=port, http_port=http_port)

    server_core = ProtobufServiceCore(url, service_def, True, smeta)
    service = RemoteObjectService(server_core, is_local, length)

    kwargs = {
        "service": service,
        "wait_time": wait_time,
        "port": port,
        "length": length,
        "is_local": is_local,
        "max_threads": max_threads,
        "process_id": process_id,
        "credentials": credentials,
    }
    threads = []
    thread = threading.Thread(target=serve_service, kwargs=kwargs)
    thread.setDaemon(True)
    thread.start()
    threads.append(thread)
    try:
        httpserver.serve_service_core(server_core, http_port, None)
    finally:
        server_core.shutdown_event.set()
        # loop = asyncio.get_running_loop()
        for thread in threads:
            thread.join()
