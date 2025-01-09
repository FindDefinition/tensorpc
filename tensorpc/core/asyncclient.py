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
                          rpc_timeout: Optional[int] = None,
                          rpc_callback="",
                          rpc_flags: int = rpc_message_pb2.PickleArray,
                          **kwargs) -> Any:
        data_to_be_send = core_io.data_to_pb((args, kwargs), rpc_flags)
        request = rpc_message_pb2.RemoteCallRequest(service_key=key,
                                                    arrays=data_to_be_send,
                                                    callback=rpc_callback,
                                                    flags=rpc_flags)
        return self.parse_remote_response(await self.stub.RemoteCall(request, timeout=rpc_timeout))

    async def remote_json_call(self,
                               key: str,
                               *args,
                               rpc_timeout: Optional[int] = None,
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
            await self.stub.RemoteJsonCall(request, timeout=rpc_timeout))

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
                               rpc_timeout: Optional[int] = None,
                               rpc_callback="",
                               rpc_flags: int = rpc_message_pb2.PickleArray,
                               **kwargs) -> AsyncGenerator[Any, None]:
        data_to_be_send = core_io.data_to_pb((args, kwargs), rpc_flags)
        request = rpc_message_pb2.RemoteCallRequest(service_key=key,
                                                    arrays=data_to_be_send,
                                                    callback=rpc_callback,
                                                    flags=rpc_flags)
        async for response in self.stub.RemoteGenerator(request, timeout=rpc_timeout):
            yield self.parse_remote_response(response)

    async def client_stream(self,
                            key: str,
                            stream_iter,
                            *args,
                            timeout: Optional[int] = None,
                            rpc_flags: int = rpc_message_pb2.PickleArray,
                            **kwargs) -> Any:
        flags = rpc_flags
        if inspect.isasyncgen(stream_iter):
            async def wrapped_generator_async():
                data_to_be_send = core_io.data_to_pb((args, kwargs), flags)
                request = rpc_message_pb2.RemoteCallRequest(service_key=key,
                                                            arrays=data_to_be_send,
                                                            flags=flags)
                yield request
                async for data in stream_iter:
                    data_to_be_send = core_io.data_to_pb(((data, ), {}), flags)
                    request = rpc_message_pb2.RemoteCallRequest(
                        service_key=key, arrays=data_to_be_send, flags=flags)
                    yield request
            wrapped_func = wrapped_generator_async
        else:
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
            wrapped_func = wrapped_generator

        response = await self.stub.ClientStreamRemoteCall(wrapped_func())
        return self.parse_remote_response(response)

    async def bi_stream(self,
                        key: str,
                        stream_iter,
                        *args,
                        rpc_timeout: Optional[int] = None,
                        rpc_flags: int = rpc_message_pb2.PickleArray,
                        **kwargs) -> AsyncGenerator[Any, None]:
        flags = rpc_flags
        if inspect.isasyncgen(stream_iter):
            async def wrapped_generator_async():
                data_to_be_send = core_io.data_to_pb((args, kwargs), flags)
                request = rpc_message_pb2.RemoteCallRequest(service_key=key,
                                                            arrays=data_to_be_send,
                                                            flags=flags)
                yield request
                async for data in stream_iter:
                    data_to_be_send = core_io.data_to_pb(((data, ), {}), flags)
                    request = rpc_message_pb2.RemoteCallRequest(
                        service_key=key, arrays=data_to_be_send, flags=flags)
                    yield request
            wrapped_func = wrapped_generator_async
        else:
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
            wrapped_func = wrapped_generator
        async for response in self.stub.BiStreamRemoteCall(
                wrapped_func()):
            yield self.parse_remote_response(response)

    async def stream_remote_call(
        self,
        key: str,
        stream_iter: Iterator[Any],
        rpc_flags: int = rpc_message_pb2.PickleArray
    ) -> AsyncGenerator[Any, None]:
        """Call a remote function (not generator) with stream data:
        ```Python
        for (args, kwargs) in stream_iter:
            yield remote_func(*args, **kwargs)
        ```
        args and returns aren't chunked, their size is limited by grpc server
        (4MB by default).
        """
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
            rpc_flags: int = rpc_message_pb2.PickleArray,
            rpc_timeout: Optional[int] = None,
            rpc_chunk_size: int = 64 * 1024,
    ) -> AsyncIterator[Any]:
        """Call a remote function (not generator) with stream data:
        ```Python
        for (args, kwargs) in stream_iter:
            yield remote_func(*args, **kwargs)
        ```
        all args and returns are chunked to support arbitrary size of data.
        """
        flags = rpc_flags
        if inspect.isasyncgen(stream_iter):
            async def stream_generator_async():
                async for data in stream_iter:
                    try:
                        arrays, data_skeleton = core_io.extract_arrays_from_data(data)
                        data_to_be_send = arrays + [
                            core_io.dumps_method(data_skeleton, flags)
                        ]
                        stream = core_io.to_protobuf_stream(data_to_be_send, key,
                                                            flags, rpc_chunk_size)
                    except:
                        traceback.print_exc()
                        continue
                    for s in stream:
                        yield s
            stream_generator_func = stream_generator_async
        else:
            def stream_generator():
                for data in stream_iter:
                    try:
                        arrays, data_skeleton = core_io.extract_arrays_from_data(data)
                        data_to_be_send = arrays + [
                            core_io.dumps_method(data_skeleton, flags)
                        ]
                        stream = core_io.to_protobuf_stream(data_to_be_send, key,
                                                            flags, rpc_chunk_size)
                    except:
                        traceback.print_exc()
                        continue
                    for s in stream:
                        yield s
            stream_generator_func = stream_generator
        from_stream = core_io.FromBufferStream()
        async for response in self.stub.ChunkedRemoteCall(stream_generator_func(), timeout=rpc_timeout):
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
                                  rpc_timeout: Optional[int] = None,
                                  rpc_chunk_size: int = 64 * 1024,
                                  **kwargs) -> Any:

        def stream_generator():
            yield (args, kwargs)

        count = 0
        res: Optional[_PlaceHolder] = _PlaceHolder()
        async for res in self.chunked_stream_remote_call(key,
                                                         stream_generator(),
                                                         rpc_flags=rpc_flags,
                                                         rpc_timeout=rpc_timeout,
                                                         rpc_chunk_size=rpc_chunk_size):
            count += 1
        assert count == 1
        assert not isinstance(res, _PlaceHolder)
        return res

    def _arg_chunked_sender(self, key: str, data, rpc_chunk_size: int, rpc_flags: int = rpc_message_pb2.PickleArray):
        try:
            arrays, data_skeleton = core_io.extract_arrays_from_data(data)
            data_to_be_send = arrays + [
                core_io.dumps_method(data_skeleton, rpc_flags)
            ]
            stream = core_io.to_protobuf_stream(data_to_be_send, key,
                                                rpc_flags, rpc_chunk_size)
        except:
            traceback.print_exc()
            raise
        for s in stream:
            yield s

    async def chunked_bi_stream(self,
                        key: str,
                        stream_iter,
                        *args,
                        rpc_timeout: Optional[int] = None,
                        rpc_flags: int = rpc_message_pb2.PickleArray,
                        rpc_chunk_size: int = 64 * 1024,
                        **kwargs) -> AsyncGenerator[Any, None]:
        if inspect.isasyncgen(stream_iter):
            async def wrapped_generator_async():
                for chunk in self._arg_chunked_sender(key, (args, kwargs), rpc_chunk_size, rpc_flags):
                    yield chunk
                async for data in stream_iter:
                    for chunk in self._arg_chunked_sender(key, [data], rpc_chunk_size, rpc_flags):
                        yield chunk
            wrapped_func = wrapped_generator_async
        else:
            def wrapped_generator():
                for chunk in self._arg_chunked_sender(key, (args, kwargs), rpc_chunk_size, rpc_flags):
                    yield chunk
                for data in stream_iter:
                    for chunk in self._arg_chunked_sender(key, [data], rpc_chunk_size, rpc_flags):
                        yield chunk
            wrapped_func = wrapped_generator
        from_stream = core_io.FromBufferStream()
        async for req, data in from_stream.generator_async(self.stub.ChunkedBiStreamRemoteCall(
                wrapped_func())):
            yield data[0]

    async def chunked_client_stream(self,
                        key: str,
                        stream_iter,
                        *args,
                        rpc_timeout: Optional[int] = None,
                        rpc_flags: int = rpc_message_pb2.PickleArray,
                        rpc_chunk_size: int = 64 * 1024,
                        **kwargs) -> Any:
        if inspect.isasyncgen(stream_iter):
            async def wrapped_generator_async():
                for chunk in self._arg_chunked_sender(key, (args, kwargs), rpc_chunk_size, rpc_flags):
                    yield chunk
                async for data in stream_iter:
                    for chunk in self._arg_chunked_sender(key, [data], rpc_chunk_size, rpc_flags):
                        yield chunk
            wrapped_func = wrapped_generator_async
        else:
            def wrapped_generator():
                for chunk in self._arg_chunked_sender(key, (args, kwargs), rpc_chunk_size, rpc_flags):
                    yield chunk
                for data in stream_iter:
                    for chunk in self._arg_chunked_sender(key, [data], rpc_chunk_size, rpc_flags):
                        yield chunk
            wrapped_func = wrapped_generator
        from_stream = core_io.FromBufferStream()
        async for req, data in from_stream.generator_async(self.stub.ChunkedClientStreamRemoteCall(
                wrapped_func())):
            return data[0]

    async def chunked_remote_generator(self, key: str, *args, rpc_flags: int = rpc_message_pb2.PickleArray, rpc_timeout: Optional[int] = None, rpc_chunk_size: int = 64 * 1024, **kwargs) -> AsyncGenerator[Any, None]:
        def wrapped_generator():
            for chunk in self._arg_chunked_sender(key, (args, kwargs), rpc_chunk_size, rpc_flags):
                yield chunk
        from_stream = core_io.FromBufferStream()
        async for req, data in from_stream.generator_async(self.stub.ChunkedRemoteGenerator(
                wrapped_generator())):
            yield data[0]

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
        # atexit.register(self.close)
        super().__init__(channel, name, print_stdout)

    async def reconnect(self, timeout=10, max_retries=20):
        await self.close()
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
            await self.health_check(wait_for_ready, timeout)
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

    async def close(self, close_channel: bool = False):
        if self._channel is not None:
            # if we shutdown remote and close channel,
            # will raise strange error.
            if close_channel:
                await self._channel.close()
            del self._channel
            self._channel = None

    async def shutdown(self):
        res = await super().shutdown()
        await self.close()
        return res

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, exc_traceback):
        if self._channel is not None:
            await self._channel.__aexit__(exc_type, exc_value, exc_traceback)
        return await self.close()


async def simple_remote_call_async(addr, key, *args, rpc_timeout=None, **kwargs):
    async with AsyncRemoteManager(addr) as robj:
        return await robj.remote_call(key, *args, rpc_timeout=rpc_timeout, **kwargs)


async def simple_chunk_call_async(addr, key, *args, rpc_timeout=None, **kwargs):
    async with AsyncRemoteManager(addr) as robj:
        # await robj.wait_for_channel_ready()
        return await robj.chunked_remote_call(key, *args, rpc_timeout=rpc_timeout, **kwargs)


async def shutdown_server_async(addr):
    async with AsyncRemoteManager(addr) as robj:
        return await robj.shutdown()
