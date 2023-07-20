import asyncio
from concurrent.futures import Executor
import contextlib
import ctypes
import io
import json
import os
import pickle
import sys
import tempfile
import threading
import time
import traceback
from typing import (TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, Iterator, List,
                    Mapping, Optional, Sequence, Union)
import dataclasses

import aiohttp
from tensorpc.core.defs import Service, ServiceDef
from tensorpc import compat
from tensorpc.core import core_io, serviceunit
from tensorpc.protos_export import remote_object_pb2 as remote_object_pb2
from tensorpc.protos_export import rpc_message_pb2 as rpc_msg_pb2

from tensorpc.utils import df_logging
import contextvars
if TYPE_CHECKING:
    from tensorpc.core.httpservers.core import WebsocketClientPair
LOGGER = df_logging.get_logger()


@dataclasses.dataclass
class ServerMeta:
    port: int
    http_port: int


class _ExposedServerProps(object):
    """we save static methods/props of service to a object
    """

    def __init__(self, exec_lock, service_units, shutdown_event, local_url,
                 is_sync: bool, server_meta: ServerMeta):
        self.exec_lock = exec_lock
        self.service_units = service_units
        self.shutdown_event = shutdown_event
        self.local_url = local_url
        self.is_sync = is_sync
        self.server_meta = server_meta
        self.http_client_session: Optional[aiohttp.ClientSession] = None
        self._async_shutdown_event: Optional[asyncio.Event] = None
        self._executor: Optional[Executor] = None

    @property
    def async_shutdown_event(self):
        assert self._async_shutdown_event is not None
        return self._async_shutdown_event


class ServerContext(object):

    def __init__(self,
                 exposed_props: _ExposedServerProps,
                 service_key=None,
                 json_call=False):
        self.exposed_props = exposed_props
        self.service_key = service_key
        self.json_call = json_call


class ServerGlobalContext(object):

    def __init__(self, local_url: str, is_sync: bool, server_meta: ServerMeta):
        self.http_client_session: Optional[aiohttp.ClientSession] = None
        self.local_url = local_url
        self.is_sync = is_sync
        self.server_meta = server_meta


SERVER_RPC_CONTEXT = {}

CONTEXT_LOCK = threading.Lock()

# we need contextvars to support service context in asyncio
SERVER_RPC_CONTEXT_VAR = contextvars.ContextVar("service_rpc_context",
                                                default=None)
# we need contextvars to support service context in asyncio
SERVER_GLOBAL_CONTEXT_VAR = contextvars.ContextVar("service_context",
                                                   default=None)


def is_in_server_context() -> bool:
    if compat.Python3_7AndLater:
        assert SERVER_RPC_CONTEXT_VAR is not None
        return SERVER_RPC_CONTEXT_VAR.get() is not None
    tid = threading.get_ident()
    exist = True
    with CONTEXT_LOCK:
        if tid not in SERVER_RPC_CONTEXT:
            exist = False
    return exist


def get_server_context() -> ServerContext:
    ctx = None
    if compat.Python3_7AndLater:
        assert SERVER_RPC_CONTEXT_VAR is not None
        ctx = SERVER_RPC_CONTEXT_VAR.get()
        if ctx is None:
            raise ValueError(
                "you can't call primitives outside server context.")
        return ctx
    tid = threading.get_ident()
    notexist = False
    with CONTEXT_LOCK:
        if tid not in SERVER_RPC_CONTEXT:
            notexist = True
        else:
            ctx = SERVER_RPC_CONTEXT[tid]
    if notexist:
        raise ValueError("you can't call primitives outside server context.")
    assert ctx is not None
    return ctx


def is_in_global_context() -> bool:
    assert SERVER_GLOBAL_CONTEXT_VAR is not None
    return SERVER_GLOBAL_CONTEXT_VAR.get() is not None


def get_global_context() -> ServerGlobalContext:
    assert SERVER_GLOBAL_CONTEXT_VAR is not None
    ctx = SERVER_GLOBAL_CONTEXT_VAR.get()
    if ctx is None:
        raise ValueError(
            "you can't call primitives outside server global context.")
    return ctx


class ServiceCore(object):

    def __init__(self, local_url: str, service_def: ServiceDef, is_sync: bool,
                 server_meta: ServerMeta):
        self._exec_lock = threading.Lock()
        self.local_url = local_url
        self.shutdown_event = threading.Event()
        self.latest_active_time = time.time()
        self.reset_timeout_lock = threading.Lock()
        self.service_def = service_def
        self.service_units = serviceunit.ServiceUnits([
            serviceunit.ServiceUnit(d.module_name, d.config)
            for d in service_def.services
        ])
        self.is_sync = is_sync
        self._register_exit_lock = threading.Lock()
        self._exit_funcs = {}
        self._exposed_props = _ExposedServerProps(self._exec_lock,
                                                  self.service_units,
                                                  self.shutdown_event,
                                                  self.local_url, is_sync,
                                                  server_meta)

        self._global_context = ServerGlobalContext(self.local_url, is_sync,
                                                   server_meta)
        self._all_websockets: Dict[str, WebsocketClientPair] = {}

    async def _init_async_members(self):
        # in future python versions, asyncio event can't be created if no event loop running.
        self.async_shutdown_event = asyncio.Event()
        self._exposed_props._async_shutdown_event = self.async_shutdown_event
        await self.service_units.run_async_init()

    def init_http_client_session(self, sess: aiohttp.ClientSession):
        self._global_context.http_client_session = sess

    async def exec_exit_funcs(self):
        return await self.service_units.run_exit()

    def exec_exit_funcs_sync(self):
        return self.service_units.run_exit_sync()

    def _reset_timeout(self):
        with self.reset_timeout_lock:
            self.latest_active_time = time.time()

    def _remote_exception_json(self, e: BaseException):
        return json.dumps(self._remote_exception_dict(e))

    def _remote_exception_dict(self,
                               e: BaseException,
                               detail: Optional[Any] = None):
        if detail is None:
            detail = traceback.format_exc()
        exception_json = {"error": type(e).__qualname__, "detail": detail}
        return exception_json

    def get_service_meta(self):
        return self.service_units.get_all_service_metas_json()

    @contextlib.contextmanager
    def _enter_exec_context(self, service_key=None, json_call=False):
        ctx = ServerContext(self._exposed_props, service_key, json_call)
        if compat.Python3_7AndLater:
            # we need contextvars in async code. so we drop websocket
            # support before python 3.7.
            assert SERVER_RPC_CONTEXT_VAR is not None
            token = SERVER_RPC_CONTEXT_VAR.set(ctx)
            yield ctx
            SERVER_RPC_CONTEXT_VAR.reset(token)
        else:
            tid = threading.get_ident()
            with CONTEXT_LOCK:
                SERVER_RPC_CONTEXT[tid] = ctx
            yield ctx
            with CONTEXT_LOCK:
                SERVER_RPC_CONTEXT[tid] = None

    @contextlib.contextmanager
    def enter_global_context(self):
        assert SERVER_GLOBAL_CONTEXT_VAR is not None
        token = SERVER_GLOBAL_CONTEXT_VAR.set(self._global_context)
        yield self._global_context
        SERVER_GLOBAL_CONTEXT_VAR.reset(token)

    def execute_service(self,
                        service_key,
                        args,
                        kwargs,
                        service_type=serviceunit.ServiceType.Normal,
                        json_call=False):
        is_exception = False
        try:
            # no lock here, user must use 'get_exec_lock' to get global lock
            # or create lock by themselves.
            with self._enter_exec_context(service_key, json_call) as ctx:
                # all services are lazy-loaded,
                # so we need to put get_service in try block
                func, meta = self.service_units.get_service_and_meta(
                    service_key)
                assert service_type == meta.type, f"{service_type}, {meta.type}"
                assert not meta.is_async and not meta.is_gen
                # client code can call primitives to get server contents.
                res = func(*args, **kwargs)
        except Exception as e:
            res = self._remote_exception_json(e)
            is_exception = True
        return res, is_exception

    async def execute_async_service(
            self,
            service_key,
            args,
            kwargs,
            service_type=serviceunit.ServiceType.Normal,
            json_call=False):
        is_exception = False
        try:
            # no lock here, user must use 'get_exec_lock' to get global lock
            # or create lock by themselves.
            with self._enter_exec_context(service_key, json_call) as ctx:
                # all services are lazy-loaded,
                # so we need to put get_service in try block
                func, meta = self.service_units.get_service_and_meta(
                    service_key)
                assert service_type == meta.type
                # client code can call primitives to get server contents.
                assert not meta.is_gen
                if meta.is_async:
                    res = await func(*args, **kwargs)
                else:
                    res = func(*args, **kwargs)
        except Exception as e:

            res = self._remote_exception_json(e)
            is_exception = True
        return res, is_exception

    def execute_generator_service(self,
                                  service_key,
                                  args,
                                  kwargs,
                                  json_call=False,
                                  service_type=serviceunit.ServiceType.Normal):
        is_exception = False
        try:
            # no lock here, user must use 'get_exec_lock' to get lock
            # or create lock by themselves.
            with self._enter_exec_context(service_key,
                                          json_call=json_call) as ctx:
                # all services are lazy-loaded,
                # so we need to put get_service in try block
                func, meta = self.service_units.get_service_and_meta(
                    service_key)
                assert not meta.is_async and meta.is_gen
                assert meta.type == service_type
                # client code can call primitives to get server contents.
                for res in func(*args, **kwargs):
                    yield res, is_exception

        except Exception as e:
            res = self._remote_exception_json(e)
            yield res, True

    async def execute_async_generator_service(
            self,
            service_key,
            args,
            kwargs,
            json_call=False,
            service_type=serviceunit.ServiceType.Normal):
        is_exception = False
        try:
            # no lock here, user must use 'get_exec_lock' to get lock
            # or create lock by themselves.
            with self._enter_exec_context(service_key,
                                          json_call=json_call) as ctx:
                # all services are lazy-loaded,
                # so we need to put get_service in try block
                func, meta = self.service_units.get_service_and_meta(
                    service_key)
                assert meta.is_async and meta.is_gen, service_key
                assert meta.type == service_type
                # client code can call primitives to get server contents.
                async for res in func(*args, **kwargs):
                    yield res, is_exception

        except Exception as e:
            res = self._remote_exception_json(e)
            yield res, True


class ProtobufServiceCore(ServiceCore):
    """service with core io (protobuf)
    """

    def _process_data(self, arrays, method: int):
        return core_io.data_from_pb(arrays, method)

    def remote_call(self, request: rpc_msg_pb2.RemoteCallRequest):
        self._reset_timeout()
        args, kwargs = self._process_data(request.arrays, request.flags)
        res_func, is_exc = self.execute_service(request.service_key, args,
                                                kwargs)
        if is_exc:
            return rpc_msg_pb2.RemoteCallReply(exception=res_func)
        res = rpc_msg_pb2.RemoteCallReply(arrays=core_io.data_to_pb(
            [res_func], request.flags),
                                          flags=request.flags)
        del res_func
        return res

    async def remote_call_async(self, request: rpc_msg_pb2.RemoteCallRequest):
        self._reset_timeout()
        args, kwargs = self._process_data(request.arrays, request.flags)
        res_func, is_exc = await self.execute_async_service(
            request.service_key, args, kwargs)
        if is_exc:
            return rpc_msg_pb2.RemoteCallReply(exception=res_func)
        res = rpc_msg_pb2.RemoteCallReply(arrays=core_io.data_to_pb(
            [res_func], request.flags),
                                          flags=request.flags)
        del res_func
        return res

    def remote_generator(self, request: rpc_msg_pb2.RemoteCallRequest):
        self._reset_timeout()
        flags = request.flags
        args, kwargs = self._process_data(request.arrays, flags)
        for res, is_exc in self.execute_generator_service(
                request.service_key, args, kwargs, False):
            self._reset_timeout()
            if is_exc:  # exception
                yield rpc_msg_pb2.RemoteCallReply(exception=res)

                break
            res = [res]
            res = core_io.data_to_pb(res, flags)
            yield rpc_msg_pb2.RemoteCallReply(arrays=res, flags=flags)

    def remote_json_generator(self,
                              request: rpc_msg_pb2.RemoteJsonCallRequest):
        self._reset_timeout()
        flags = request.flags
        args, kwargs = core_io.data_from_json(request.arrays, request.data,
                                              flags)
        for res, is_exc in self.execute_generator_service(
                request.service_key, args, kwargs, False):
            self._reset_timeout()
            if is_exc:  # exception
                yield rpc_msg_pb2.RemoteJsonCallReply(exception=res)
                break
            res = [res]
            arrays, decoupled = core_io.data_to_json(res, flags)
            yield rpc_msg_pb2.RemoteJsonCallReply(arrays=arrays,
                                                  data=decoupled,
                                                  flags=flags)

    def remote_json_call(self, request: rpc_msg_pb2.RemoteJsonCallRequest):
        self._reset_timeout()
        flags = request.flags
        args, kwargs = core_io.data_from_json(request.arrays, request.data,
                                              flags)
        res, is_exc = self.execute_service(request.service_key,
                                           args,
                                           kwargs,
                                           json_call=True)
        if is_exc:
            return rpc_msg_pb2.RemoteJsonCallReply(exception=res)
        res = [res]
        arrays, decoupled = core_io.data_to_json(res, flags)
        return rpc_msg_pb2.RemoteJsonCallReply(arrays=arrays,
                                               data=decoupled,
                                               flags=flags)

    async def remote_json_call_async(
            self, request: rpc_msg_pb2.RemoteJsonCallRequest):
        self._reset_timeout()
        flags = request.flags
        args, kwargs = core_io.data_from_json(request.arrays, request.data,
                                              flags)
        res, is_exc = await self.execute_async_service(request.service_key,
                                                       args,
                                                       kwargs,
                                                       json_call=True)
        if is_exc:
            return rpc_msg_pb2.RemoteJsonCallReply(exception=res)
        res = [res]
        arrays, decoupled = core_io.data_to_json(res, flags)
        return rpc_msg_pb2.RemoteJsonCallReply(arrays=arrays,
                                               data=decoupled,
                                               flags=flags)

    def chunked_remote_call(
            self, request_iter: Iterator[rpc_msg_pb2.RemoteCallStream]):
        self._reset_timeout()
        from_stream = core_io.FromBufferStream()
        for request in request_iter:
            res = from_stream(request)
            if res is not None:
                from_stream.clear()
                incoming, func_key = res
                arrays = incoming[:-1]
                data_skeleton_bytes = incoming[-1]
                data_skeleton = core_io.loads_method(data_skeleton_bytes,
                                                     request.flags)
                args, kwargs = core_io.put_arrays_to_data(
                    arrays, data_skeleton)
                res, is_exc = self.execute_service(func_key, args, kwargs)
                if is_exc:
                    # exception
                    yield rpc_msg_pb2.RemoteCallStream(
                        exception=res,
                        chunked_data=b'',
                    )
                    break
                res = [res]
                arrays, data_skeleton = core_io.extract_arrays_from_data(res)
                data_skeleton_bytes = core_io.dumps_method(
                    data_skeleton, request.flags)
                res = arrays + [data_skeleton_bytes]
                res_streams = core_io.to_protobuf_stream(
                    res, func_key, request.flags)
                for chunk in res_streams:
                    yield chunk
        del from_stream

    def remote_stream_call(
            self, request_iter: Iterator[rpc_msg_pb2.RemoteCallRequest]):
        self._reset_timeout()
        for request in request_iter:
            yield self.remote_call(request)

    def client_stream(self,
                      request_iter: Iterator[rpc_msg_pb2.RemoteCallRequest]):
        self._reset_timeout()
        call_request = next(request_iter)
        args, kwargs = self._process_data(call_request.arrays,
                                          call_request.flags)
        key = call_request.service_key

        def generator():
            for request in request_iter:
                self._reset_timeout()
                args, _ = self._process_data(request.arrays,
                                             call_request.flags)
                data = args[0]
                yield data

        res, is_exc = self.execute_service(
            key, [generator(), *args],
            kwargs,
            service_type=serviceunit.ServiceType.ClientStream)
        if is_exc:
            return rpc_msg_pb2.RemoteCallReply(exception=res)
        res = [res]
        res = core_io.data_to_pb(res, call_request.flags)
        return rpc_msg_pb2.RemoteCallReply(arrays=res,
                                           flags=call_request.flags)

    def bi_stream(self, request_iter: Iterator[rpc_msg_pb2.RemoteCallRequest]):
        self._reset_timeout()
        call_request = next(request_iter)
        args, kwargs = self._process_data(call_request.arrays,
                                          call_request.flags)
        key = call_request.service_key

        def generator():
            for request in request_iter:
                args, _ = self._process_data(request.arrays,
                                             call_request.flags)
                data = args[0]
                yield data

        for res, is_exc in self.execute_generator_service(
                key, [generator(), *args],
                kwargs,
                False,
                service_type=serviceunit.ServiceType.BiStream):
            self._reset_timeout()
            if is_exc:  # exception
                yield rpc_msg_pb2.RemoteCallReply(exception=res)
                break
            res = [res]
            res = core_io.data_to_pb(res, call_request.flags)
            yield rpc_msg_pb2.RemoteCallReply(arrays=res,
                                              flags=call_request.flags)

    async def remote_generator_async(self,
                                     request: rpc_msg_pb2.RemoteCallRequest):
        self._reset_timeout()
        # TODO determine generator is async generator
        flags = request.flags
        args, kwargs = self._process_data(request.arrays, flags)
        _, meta = self.service_units.get_service_and_meta(request.service_key)
        if not meta.is_async and meta.is_gen:
            for res, is_exc in self.execute_generator_service(
                    request.service_key, args, kwargs, False):
                self._reset_timeout()
                if is_exc:  # exception
                    yield rpc_msg_pb2.RemoteCallReply(exception=res)

                    break
                res = [res]
                res = core_io.data_to_pb(res, flags)
                yield rpc_msg_pb2.RemoteCallReply(arrays=res, flags=flags)
        else:
            async for res, is_exc in self.execute_async_generator_service(
                    request.service_key, args, kwargs, False):
                self._reset_timeout()
                if is_exc:  # exception
                    yield rpc_msg_pb2.RemoteCallReply(exception=res)
                    break
                res = [res]
                res = core_io.data_to_pb(res, flags)
                yield rpc_msg_pb2.RemoteCallReply(arrays=res, flags=flags)

    async def remote_json_generator_async(
            self, request: rpc_msg_pb2.RemoteJsonCallRequest):
        self._reset_timeout()
        flags = request.flags
        args, kwargs = core_io.data_from_json(request.arrays, request.data,
                                              flags)
        _, meta = self.service_units.get_service_and_meta(request.service_key)
        if not meta.is_async and meta.is_gen:
            for res, is_exc in self.execute_generator_service(
                    request.service_key, args, kwargs, False):
                self._reset_timeout()
                if is_exc:  # exception
                    yield rpc_msg_pb2.RemoteJsonCallReply(exception=res)
                    break
                res = [res]
                arrays, decoupled = core_io.data_to_json(res, flags)
                yield rpc_msg_pb2.RemoteJsonCallReply(arrays=arrays,
                                                      data=decoupled,
                                                      flags=flags)
        else:
            async for res, is_exc in self.execute_async_generator_service(
                    request.service_key, args, kwargs, False):
                self._reset_timeout()
                if is_exc:  # exception
                    yield rpc_msg_pb2.RemoteJsonCallReply(exception=res)
                    break
                res = [res]
                arrays, decoupled = core_io.data_to_json(res, flags)
                yield rpc_msg_pb2.RemoteJsonCallReply(arrays=arrays,
                                                      data=decoupled,
                                                      flags=flags)

    async def chunked_remote_call_async(
            self, request_iter: AsyncIterator[rpc_msg_pb2.RemoteCallStream]):
        self._reset_timeout()
        from_stream = core_io.FromBufferStream()
        async for request in request_iter:
            res = from_stream(request)
            if res is not None:
                from_stream.clear()
                incoming, func_key = res
                arrays = incoming[:-1]
                data_skeleton_bytes = incoming[-1]
                data_skeleton = core_io.loads_method(data_skeleton_bytes,
                                                     request.flags)
                args, kwargs = core_io.put_arrays_to_data(
                    arrays, data_skeleton)
                res, is_exc = await self.execute_async_service(
                    func_key, args, kwargs)
                if is_exc:
                    # exception
                    yield rpc_msg_pb2.RemoteCallStream(
                        exception=res,
                        chunked_data=b'',
                    )
                    break
                res = [res]
                arrays, data_skeleton = core_io.extract_arrays_from_data(res)
                data_skeleton_bytes = core_io.dumps_method(
                    data_skeleton, request.flags)
                res = arrays + [data_skeleton_bytes]
                res_streams = core_io.to_protobuf_stream(
                    res, func_key, request.flags)
                for chunk in res_streams:
                    yield chunk
        del from_stream

    async def remote_stream_call_async(
            self, request_iter: AsyncIterator[rpc_msg_pb2.RemoteCallRequest]):
        self._reset_timeout()
        async for request in request_iter:
            yield await self.remote_call_async(request)

    async def client_stream_async(
            self, request_iter: AsyncIterator[rpc_msg_pb2.RemoteCallRequest]):
        self._reset_timeout()
        call_request = None
        async for call_request in request_iter:
            break
        assert call_request is not None
        # call_request = anext(request_iter)
        args, kwargs = self._process_data(call_request.arrays,
                                          call_request.flags)
        key = call_request.service_key

        async def generator():
            async for request in request_iter:
                self._reset_timeout()
                args, _ = self._process_data(request.arrays,
                                             call_request.flags)
                data = args[0]
                yield data

        res, is_exc = await self.execute_async_service(
            key, [generator(), *args],
            kwargs,
            service_type=serviceunit.ServiceType.ClientStream)
        if is_exc:
            return rpc_msg_pb2.RemoteCallReply(exception=res)
        res = [res]
        res = core_io.data_to_pb(res, call_request.flags)
        return rpc_msg_pb2.RemoteCallReply(arrays=res,
                                           flags=call_request.flags)

    async def bi_stream_async(
            self, request_iter: AsyncIterator[rpc_msg_pb2.RemoteCallRequest]):
        self._reset_timeout()
        call_request = None
        async for call_request in request_iter:
            break
        assert call_request is not None
        args, kwargs = self._process_data(call_request.arrays,
                                          call_request.flags)
        key = call_request.service_key

        async def generator():
            async for request in request_iter:
                args, _ = self._process_data(request.arrays,
                                             call_request.flags)
                data = args[0]
                yield data

        _, meta = self.service_units.get_service_and_meta(key)
        if not meta.is_async and meta.is_gen:
            for res, is_exc in self.execute_generator_service(
                    key, [generator(), *args],
                    kwargs,
                    False,
                    service_type=serviceunit.ServiceType.BiStream):
                self._reset_timeout()
                if is_exc:  # exception
                    yield rpc_msg_pb2.RemoteCallReply(exception=res)
                    break
                res = [res]
                res = core_io.data_to_pb(res, call_request.flags)
                yield rpc_msg_pb2.RemoteCallReply(arrays=res,
                                                  flags=call_request.flags)
        else:
            async for res, is_exc in self.execute_async_generator_service(
                    key, [generator(), *args],
                    kwargs,
                    False,
                    service_type=serviceunit.ServiceType.BiStream):
                self._reset_timeout()
                if is_exc:  # exception
                    yield rpc_msg_pb2.RemoteCallReply(exception=res)
                    break
                res = [res]
                res = core_io.data_to_pb(res, call_request.flags)
                yield rpc_msg_pb2.RemoteCallReply(arrays=res,
                                                  flags=call_request.flags)
