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
from typing import (TYPE_CHECKING, Any, AsyncIterator, Callable, Dict,
                    Iterator, List, Mapping, Optional, Sequence, Union)
import dataclasses

import aiohttp
from tensorpc.core.defs import Service, ServiceDef
from tensorpc import compat
from tensorpc.core import core_io, serviceunit
from tensorpc.protos_export import remote_object_pb2 as remote_object_pb2
from tensorpc.protos_export import rpc_message_pb2 as rpc_msg_pb2
from tensorpc.core.serviceunit import ServiceEventType
from tensorpc.utils import df_logging
import contextvars

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
                 json_call=False,
                 is_loopback_call: bool = False):
        self.exposed_props = exposed_props
        self.service_key = service_key
        self.json_call = json_call
        self.is_loopback_call = is_loopback_call


class ServerGlobalContext(object):

    def __init__(self, local_url: str, is_sync: bool, server_meta: ServerMeta):
        self.http_client_session: Optional[aiohttp.ClientSession] = None
        self.local_url = local_url
        self.is_sync = is_sync
        self.server_meta = server_meta


SERVER_RPC_CONTEXT = {}

CONTEXT_LOCK = threading.Lock()

# we need contextvars to support service context in asyncio
SERVER_RPC_CONTEXT_VAR: contextvars.ContextVar[Optional[ServerContext]] = contextvars.ContextVar("service_rpc_context",
                                                default=None)
# we need contextvars to support service context in asyncio
SERVER_GLOBAL_CONTEXT_VAR: contextvars.ContextVar[Optional[ServerGlobalContext]] = contextvars.ContextVar("service_context",
                                                   default=None)


def is_in_server_context() -> bool:
    assert SERVER_RPC_CONTEXT_VAR is not None
    return SERVER_RPC_CONTEXT_VAR.get() is not None


def get_server_context() -> ServerContext:
    assert SERVER_RPC_CONTEXT_VAR is not None
    ctx = SERVER_RPC_CONTEXT_VAR.get()
    if ctx is None:
        raise ValueError(
            "you can't call primitives outside server context.")
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
        self.server_meta = server_meta
        self._exposed_props = _ExposedServerProps(self._exec_lock,
                                                  self.service_units,
                                                  self.shutdown_event,
                                                  self.local_url, is_sync,
                                                  server_meta)

        self._global_context = ServerGlobalContext(self.local_url, is_sync,
                                                   server_meta)
        # protect run_event_async ServiceEventType.Exit
        self._shutdown_handler_lock = asyncio.Lock()
        self._is_exit_async_run = False

        self._loop: Optional[asyncio.AbstractEventLoop] = None


    async def _init_async_members(self):
        # in future python versions, asyncio event can't be created if no event loop running.
        self.async_shutdown_event = asyncio.Event()
        self._exposed_props._async_shutdown_event = self.async_shutdown_event

    def _set_port(self, port: int):
        self._exposed_props.server_meta.port = port

    def init_http_client_session(self, sess: aiohttp.ClientSession):
        self._global_context.http_client_session = sess

    def run_event(self, event: serviceunit.ServiceEventType, *args: Any):
        return self.service_units.run_event(event, *args)

    async def run_event_async(self, event: serviceunit.ServiceEventType, *args:
                              Any):
        return await self.service_units.run_event_async(event, *args)

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
    def enter_exec_context(self, service_key=None, json_call=False, is_loopback_call: bool = False):
        ctx = ServerContext(self._exposed_props, service_key, json_call, is_loopback_call)
        assert SERVER_RPC_CONTEXT_VAR is not None
        token = SERVER_RPC_CONTEXT_VAR.set(ctx)
        try:
            yield ctx
        finally:
            SERVER_RPC_CONTEXT_VAR.reset(token)

    @contextlib.contextmanager
    def enter_global_context(self):
        assert SERVER_GLOBAL_CONTEXT_VAR is not None
        token = SERVER_GLOBAL_CONTEXT_VAR.set(self._global_context)
        try:
            yield self._global_context
        finally:
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
            with self.enter_exec_context(service_key, json_call) as ctx:
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
            with self.enter_exec_context(service_key, json_call) as ctx:
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

    async def execute_async_service_locally(
            self,
            service_key,
            args,
            kwargs,
            service_type=serviceunit.ServiceType.Normal,
            json_call=False):
        # no lock here, user must use 'get_exec_lock' to get global lock
        # or create lock by themselves.
        with self.enter_global_context():
            with self.enter_exec_context(service_key, json_call, is_loopback_call=True) as ctx:
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
        return res


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
            with self.enter_exec_context(service_key,
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
            with self.enter_exec_context(service_key,
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

    def _return_chunked_sender(self, key: str, req: rpc_msg_pb2.RemoteCallStream, res: Any):
        res = [res]
        arrays, data_skeleton = core_io.extract_arrays_from_data(res)
        data_skeleton_bytes = core_io.dumps_method(
            data_skeleton, req.flags)
        res = arrays + [data_skeleton_bytes]
        res_streams = core_io.to_protobuf_stream(
            res, key, req.flags)
        for chunk in res_streams:
            yield chunk

    def _extract_chunked_data(self, request_iter: Iterator[rpc_msg_pb2.RemoteCallStream]):
        from_stream = core_io.FromBufferStream()
        call_data = None
        call_request = None
        key = None
        for call_request, call_data in from_stream.generator(request_iter):
            key = call_request.func_key
            break
        assert call_request is not None and key is not None and call_data is not None
        return key, call_request, call_data

    async def _extract_chunked_data_async(self, request_iter: AsyncIterator[rpc_msg_pb2.RemoteCallStream]):
        from_stream = core_io.FromBufferStream()
        call_data = None
        call_request = None
        key = None
        async for call_request, call_data in from_stream.generator_async(request_iter):
            key = call_request.func_key
            break

        assert call_request is not None and key is not None and call_data is not None
        return key, call_request, call_data

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
            self, request: rpc_msg_pb2.RemoteJsonCallRequest, json_only: bool = False):
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
        if json_only:
            return rpc_msg_pb2.RemoteJsonCallReply(arrays=[],
                                                data=json.dumps(res),
                                                flags=flags)
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

    def chunked_bi_stream(
            self, request_iter: Iterator[rpc_msg_pb2.RemoteCallStream]):
        self._reset_timeout()
        key, call_request, call_data = self._extract_chunked_data(request_iter)
        args, kwargs = call_data

        def generator():
            from_stream = core_io.FromBufferStream()
            for req, data in from_stream.generator(request_iter):
                yield data[0]

        for res, is_exc in self.execute_generator_service(
                key, [generator(), *args],
                kwargs,
                False,
                service_type=serviceunit.ServiceType.BiStream):
            self._reset_timeout()
            if is_exc:
                yield rpc_msg_pb2.RemoteCallStream(
                    exception=res,
                    chunked_data=b'',
                )
                break
            for chunk in self._return_chunked_sender(key, call_request, res):
                yield chunk

    def chunked_client_stream(self, request_iter: Iterator[rpc_msg_pb2.RemoteCallStream]):
        self._reset_timeout()
        key, call_request, call_data = self._extract_chunked_data(request_iter)
        args, kwargs = call_data

        def generator():
            from_stream = core_io.FromBufferStream()
            for req, data in from_stream.generator(request_iter):
                yield data[0]

        res, is_exc = self.execute_service(
            key, [generator(), *args],
            kwargs,
            service_type=serviceunit.ServiceType.ClientStream)
        if is_exc:
            return rpc_msg_pb2.RemoteCallStream(exception=res)
        for chunk in self._return_chunked_sender(key, call_request, res):
            yield chunk

    def chunked_remote_generator(
            self, request_iter: Iterator[rpc_msg_pb2.RemoteCallStream]):
        self._reset_timeout()
        key, call_request, call_data = self._extract_chunked_data(request_iter)
        args, kwargs = call_data

        for res, is_exc in self.execute_generator_service(
                key, args,
                kwargs,
                False,
                service_type=serviceunit.ServiceType.Normal):
            self._reset_timeout()
            if is_exc:
                # exception
                yield rpc_msg_pb2.RemoteCallStream(
                    exception=res,
                    chunked_data=b'',
                )
                break
            for chunk in self._return_chunked_sender(key, call_request, res):
                yield chunk

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
        async for req, (args, kwargs) in from_stream.generator_async(request_iter):
            func_key = req.func_key
            res, is_exc = await self.execute_async_service(
                func_key, args, kwargs)
            if is_exc:
                # exception
                yield rpc_msg_pb2.RemoteCallStream(
                    exception=res,
                    chunked_data=b'',
                )
                break
            for chunk in self._return_chunked_sender(func_key, req, res):
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


    async def chunked_bi_stream_async(
            self, request_iter: AsyncIterator[rpc_msg_pb2.RemoteCallStream]):
        self._reset_timeout()
        key, call_request, call_data = await self._extract_chunked_data_async(request_iter)
        args, kwargs = call_data
        async def generator():
            from_stream = core_io.FromBufferStream()
            async for req, data in from_stream.generator_async(request_iter):
                yield data[0]
        _, meta = self.service_units.get_service_and_meta(key)
        if not meta.is_async and meta.is_gen:
            for res, is_exc in self.execute_generator_service(
                    key, [generator(), *args],
                    kwargs,
                    False,
                    service_type=serviceunit.ServiceType.BiStream):
                self._reset_timeout()
                if is_exc:
                    # exception
                    yield rpc_msg_pb2.RemoteCallStream(
                        exception=res,
                        chunked_data=b'',
                    )
                    break
                for chunk in self._return_chunked_sender(key, call_request, res):
                    yield chunk
        else:
            async for res, is_exc in self.execute_async_generator_service(
                    key, [generator(), *args],
                    kwargs,
                    False,
                    service_type=serviceunit.ServiceType.BiStream):
                self._reset_timeout()
                if is_exc:
                    # exception
                    yield rpc_msg_pb2.RemoteCallStream(
                        exception=res,
                        chunked_data=b'',
                    )
                    break
                for chunk in self._return_chunked_sender(key, call_request, res):
                    yield chunk


    async def chunked_client_stream_async(
            self, request_iter: AsyncIterator[rpc_msg_pb2.RemoteCallStream]):
        self._reset_timeout()
        key, call_request, call_data = await self._extract_chunked_data_async(request_iter)
        args, kwargs = call_data

        async def generator():
            from_stream = core_io.FromBufferStream()
            async for req, data in from_stream.generator_async(request_iter):
                yield data[0]

        res, is_exc = await self.execute_async_service(
            key, [generator(), *args],
            kwargs,
            service_type=serviceunit.ServiceType.ClientStream)
        if is_exc:
            # exception
            yield rpc_msg_pb2.RemoteCallStream(
                exception=res,
                chunked_data=b'',
            )
        for chunk in self._return_chunked_sender(key, call_request, res):
            yield chunk

    async def chunked_remote_generator_async(
            self, request_iter: AsyncIterator[rpc_msg_pb2.RemoteCallStream]):
        self._reset_timeout()
        key, call_request, call_data = await self._extract_chunked_data_async(request_iter)
        args, kwargs = call_data
        _, meta = self.service_units.get_service_and_meta(key)
        if not meta.is_async and meta.is_gen:
            for res, is_exc in self.execute_generator_service(
                    key, args,
                    kwargs,
                    False,
                    service_type=serviceunit.ServiceType.Normal):
                self._reset_timeout()
                if is_exc:
                    # exception
                    yield rpc_msg_pb2.RemoteCallStream(
                        exception=res,
                        chunked_data=b'',
                    )
                    break
                for chunk in self._return_chunked_sender(key, call_request, res):
                    yield chunk
        else:
            async for res, is_exc in self.execute_async_generator_service(
                    key, args,
                    kwargs,
                    False,
                    service_type=serviceunit.ServiceType.BiStream):
                self._reset_timeout()
                if is_exc:
                    # exception
                    yield rpc_msg_pb2.RemoteCallStream(
                        exception=res,
                        chunked_data=b'',
                    )
                    break
                for chunk in self._return_chunked_sender(key, call_request, res):
                    yield chunk
