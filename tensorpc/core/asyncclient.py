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

import atexit
import collections
import contextlib
import inspect
import json
import multiprocessing
import pickle
import time
from functools import wraps
import traceback
from typing import (Any, AsyncIterator, Dict, Generator, Iterator, List,
                    Optional, Tuple, Union, AsyncGenerator)

import grpc
import grpc.aio
import numpy as np

from tensorpc.core import core_io as core_io
from tensorpc.core.client import RemoteException, format_stdout
from tensorpc.protos_export import remote_object_pb2 as rpc_pb2
from tensorpc.protos_export import rpc_message_pb2

from tensorpc.protos_export import \
    remote_object_pb2_grpc as remote_object_pb2_grpc
from tensorpc.utils.wait_tools import wait_blocking_async, wait_until, wait_until_async
from tensorpc.utils.df_logging import get_logger

LOGGER = get_logger()


class _PlaceHolder:
    pass


class AsyncRemoteObject(object):
    """
    channel: grpc.Channel
    stub: remote_object_pb2_grpc.RemoteObjectStub
    func_dict: Dict[str, Any]
    name: str
    shared_mem: np.ndarray
    output_shared_mem: np.ndarray
    num_blocks: int
    """
    _stub: Optional[remote_object_pb2_grpc.RemoteObjectStub]
    def __init__(self,
                 channel: Optional[grpc.aio.Channel],
                 name="",
                 print_stdout=True):

        self._channel = channel
        if channel is not None:
            self._stub = remote_object_pb2_grpc.RemoteObjectStub(channel)
        else:
            self._stub = None

        self.func_dict = {}
        self.name = name
        self.print_stdout = print_stdout

    def enabled(self):
        return self._channel is not None

    @property
    def channel(self):
        assert self._channel is not None, "you need to provide a channel to enable rpc feature."
        return self._channel

    @property
    def stub(self):
        assert self._stub is not None, "you need to provide a channel to enable rpc feature."
        return self._stub

    async def query_server_meta(self):
        response = await self.stub.QueryServerMeta(
            rpc_message_pb2.RemoteCallRequest())
        return json.loads(response.data)

    async def query_service_meta(self, key):
        response = await self.stub.QueryServiceMeta(
            rpc_message_pb2.RemoteCallRequest(service_key=key))
        return json.loads(response.data)

    def _check_remote_exception(self, exception_bytes: bytes):
        if exception_bytes == "":
            return
        exc_dict = json.loads(exception_bytes)
        raise RemoteException(exc_dict["detail"])

    def _check_remote_exception_noexcept(self, exception_bytes: bytes):
        if exception_bytes == "":
            return None
        exc_dict = json.loads(exception_bytes)
        return RemoteException(exc_dict["detail"])

    async def say_hello(self, msg: str):
        response = await self.stub.SayHello(
            rpc_message_pb2.HelloRequest(data=msg))
        return response.data

    async def remote_call(self,
                          key: str,
                          *args,
                          timeout: Optional[int] = None,
                          rpc_callback="",
                          rpc_flags: int = rpc_message_pb2.PickleArray,
                          **kwargs) -> Any:
        data_to_be_send = core_io.data_to_pb((args, kwargs), rpc_flags)
        request = rpc_message_pb2.RemoteCallRequest(service_key=key,
                                                    arrays=data_to_be_send,
                                                    callback=rpc_callback,
                                                    flags=rpc_flags)
        return self.parse_remote_response(await self.stub.RemoteCall(request))

    async def remote_json_call(self,
                               key: str,
                               *args,
                               timeout: Optional[int] = None,
                               rpc_callback="",
                               rpc_flags: int = rpc_message_pb2.JsonArray,
                               **kwargs) -> Any:
        arrays, decoupled = core_io.data_to_json((args, kwargs), rpc_flags)
        request = rpc_message_pb2.RemoteJsonCallRequest(service_key=key,
                                                        arrays=arrays,
                                                        data=decoupled,
                                                        callback=rpc_callback,
                                                        flags=rpc_flags)

        return self.parse_remote_json_response(
            await self.stub.RemoteJsonCall(request))

    def parse_remote_json_response(self, response):
        self._check_remote_exception(response.exception)
        return core_io.data_from_json(response.arrays, response.data,
                                      response.flags)[0]

    def parse_remote_response_noexcept(self, response):
        exc = self._check_remote_exception_noexcept(response.exception)
        if exc is not None:
            return response, exc
        # TODO core_io.data_from_pb is slow (45us), try to optimize it.
        results = core_io.data_from_pb(response.arrays, response.flags)
        results = results[0]
        return results, exc

    def parse_remote_response(self, response):
        res, exc = self.parse_remote_response_noexcept(response)
        if exc is not None:
            raise exc
        return res

    async def remote_generator(self,
                               key: str,
                               *args,
                               timeout: Optional[int] = None,
                               rpc_callback="",
                               rpc_flags: int = rpc_message_pb2.PickleArray,
                               **kwargs) -> AsyncGenerator[Any, None]:
        data_to_be_send = core_io.data_to_pb((args, kwargs), rpc_flags)
        request = rpc_message_pb2.RemoteCallRequest(service_key=key,
                                                    arrays=data_to_be_send,
                                                    callback=rpc_callback,
                                                    flags=rpc_flags)
        async for response in self.stub.RemoteGenerator(request):
            yield self.parse_remote_response(response)

    async def client_stream(self,
                            key: str,
                            stream_iter,
                            *args,
                            timeout: Optional[int] = None,
                            rpc_flags: int = rpc_message_pb2.PickleArray,
                            **kwargs) -> Any:
        flags = rpc_flags

        def wrapped_generator():
            data_to_be_send = core_io.data_to_pb((args, kwargs), flags)
            request = rpc_message_pb2.RemoteCallRequest(service_key=key,
                                                        arrays=data_to_be_send,
                                                        flags=flags)
            yield request
            for data in stream_iter:
                data_to_be_send = core_io.data_to_pb(((data, ), {}), flags)
                request = rpc_message_pb2.RemoteCallRequest(
                    service_key=key, arrays=data_to_be_send, flags=flags)
                yield request

        response = await self.stub.ClientStreamRemoteCall(wrapped_generator())
        return self.parse_remote_response(response)

    async def bi_stream(self,
                        key: str,
                        stream_iter,
                        *args,
                        timeout: Optional[int] = None,
                        rpc_flags: int = rpc_message_pb2.PickleArray,
                        **kwargs) -> AsyncGenerator[Any, None]:
        flags = rpc_flags

        def wrapped_generator():
            data_to_be_send = core_io.data_to_pb((args, kwargs), flags)
            request = rpc_message_pb2.RemoteCallRequest(service_key=key,
                                                        arrays=data_to_be_send,
                                                        flags=flags)
            yield request
            for data in stream_iter:
                data_to_be_send = core_io.data_to_pb(((data, ), {}), flags)
                request = rpc_message_pb2.RemoteCallRequest(
                    service_key=key, arrays=data_to_be_send, flags=flags)
                yield request

        async for response in self.stub.BiStreamRemoteCall(
                wrapped_generator()):
            yield self.parse_remote_response(response)

    async def stream_remote_call(
        self,
        key: str,
        stream_iter: Iterator[Any],
        rpc_flags: int = rpc_message_pb2.PickleArray
    ) -> AsyncGenerator[Any, None]:
        # assert key in self.func_dict
        flags = rpc_flags

        def stream_generator():
            for data in stream_iter:
                # data must be (args, kwargs)
                data_to_be_send = core_io.data_to_pb(data, flags)
                yield rpc_message_pb2.RemoteCallRequest(service_key=key,
                                                        arrays=data_to_be_send,
                                                        flags=flags)

        async for response in self.stub.RemoteStreamCall(stream_generator()):
            yield self.parse_remote_response(response)

    async def shutdown(self) -> str:
        response = await self.stub.ServerShutdown(
            rpc_message_pb2.HealthCheckRequest())
        return response.data

    async def health_check(self,
                           wait_for_ready=False,
                           timeout=None) -> Dict[str, float]:
        t = time.time()
        response = await self.stub.HealthCheck(
            rpc_message_pb2.HealthCheckRequest(),
            wait_for_ready=wait_for_ready,
            timeout=timeout)
        # server_time = json.loads(response.data)
        return {
            "total": time.time() - t,
            # "to_server": server_time - t,
        }

    async def chunked_stream_remote_call(
            self,
            key: str,
            stream_iter,
            rpc_flags: int = rpc_message_pb2.PickleArray
    ) -> AsyncIterator[Any]:
        # assert key in self.func_dict
        flags = rpc_flags

        def stream_generator():
            for data in stream_iter:
                arrays, data_skeleton = core_io.extract_arrays_from_data(data)
                data_to_be_send = arrays + [
                    core_io.dumps_method(data_skeleton, flags)
                ]
                stream = core_io.to_protobuf_stream(data_to_be_send, key,
                                                    flags)
                for s in stream:
                    yield s

        from_stream = core_io.FromBufferStream()
        async for response in self.stub.ChunkedRemoteCall(stream_generator()):
            self._check_remote_exception(response.exception)
            res = from_stream(response)
            if res is not None:
                from_stream.clear()
                results_raw, _ = res
                results_array = results_raw[:-1]
                data_skeleton_bytes = results_raw[-1]
                data_skeleton = core_io.loads_method(data_skeleton_bytes,
                                                     flags)
                results = core_io.put_arrays_to_data(results_array,
                                                     data_skeleton)
                results = results[0]
                yield results

    async def chunked_remote_call(self,
                                  key,
                                  *args,
                                  rpc_flags: int = rpc_message_pb2.PickleArray,
                                  **kwargs) -> Any:

        def stream_generator():
            yield (args, kwargs)

        count = 0
        res: Optional[_PlaceHolder] = _PlaceHolder()
        async for res in self.chunked_stream_remote_call(key,
                                                         stream_generator(),
                                                         rpc_flags=rpc_flags):
            count += 1
        assert count == 1
        assert not isinstance(res, _PlaceHolder)
        return res

    async def _wait_func(self):
        try:
            await self.health_check()
            return True
        except grpc.RpcError:
            LOGGER.info("server still not ready")
            return False

    async def wait_for_remote_ready(self,
                                    timeout: float = 10,
                                    max_retries: int = 20):
        try:
            await wait_until_async(self._wait_func, max_retries,
                                   timeout / max_retries)
        except TimeoutError as e:
            LOGGER.error("server timeout.")
            raise e

    async def reconnect(self, timeout: float = 10, max_retries: int = 20):
        await self.wait_for_remote_ready(timeout / max_retries, max_retries)


class AsyncRemoteManager(AsyncRemoteObject):

    def __init__(self,
                 url,
                 name="",
                 channel_options=None,
                 credentials=None,
                 print_stdout=True,
                 enabled=True):
        if enabled:
            if credentials is not None:
                channel = grpc.aio.secure_channel(url,
                                                  credentials,
                                                  options=channel_options)
            else:
                channel = grpc.aio.insecure_channel(url,
                                                    options=channel_options)
        else:
            channel = None
        self.credentials = credentials
        self._channel_options = channel_options
        self.url = url
        if enabled:
            self._channel = channel
        atexit.register(self.close)
        super().__init__(channel, name, print_stdout)

    async def reconnect(self, timeout=10, max_retries=20):
        self.close()
        if self.credentials is not None:
            self._channel = grpc.aio.secure_channel(
                self.url, self.credentials, options=self._channel_options)
        else:
            self._channel = grpc.aio.insecure_channel(
                self.url, options=self._channel_options)
        self._stub = remote_object_pb2_grpc.RemoteObjectStub(self.channel)
        await self.wait_for_remote_ready(timeout, max_retries)

    async def wait_for_channel_ready(self,
                                     timeout: float = 10,
                                     max_retries=20):
        # https://github.com/grpc/grpc/blob/master/examples/python/wait_for_ready/asyncio_wait_for_ready_example.py
        assert self._channel is not None
        wait_for_ready = True
        try:
            await self.health_check(wait_for_ready)
        except grpc.aio.AioRpcError as rpc_error:
            traceback.print_exc()
            assert rpc_error.code() == grpc.StatusCode.UNAVAILABLE
            assert not wait_for_ready
        else:
            assert wait_for_ready

    async def wait_for_remote_ready(self, timeout: float = 10, max_retries=20):
        # await self.wait_for_channel_ready(timeout)
        await super().wait_for_remote_ready(timeout, max_retries)

    async def available(self, timeout=10, max_retries=20):
        try:
            await self.wait_for_remote_ready(timeout, max_retries)
            return True
        except TimeoutError:
            return False

    def close(self):
        if self._channel is not None:
            # if we shutdown remote and close channel,
            # will raise strange error.
            # self.channel.close()
            del self._channel
            self._channel = None

    async def shutdown(self):
        await super().shutdown()
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, exc_traceback):
        if self._channel is not None:
            await self._channel.__aexit__(exc_type, exc_value, exc_traceback)
        return self.close()


async def simple_remote_call_async(addr, key, *args, timeout=None, **kwargs):
    async with AsyncRemoteManager(addr) as robj:
        return await robj.remote_call(key, *args, timeout=timeout, **kwargs)


async def simple_chunk_call_async(addr, key, *args, **kwargs):
    async with AsyncRemoteManager(addr) as robj:
        return await robj.chunked_remote_call(key, *args, **kwargs)

async def shutdown_server_async(addr):
    async with AsyncRemoteManager(addr) as robj:
        return await robj.shutdown()
