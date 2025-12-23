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

import dataclasses
from typing import Any, AsyncGenerator, List, Optional, Tuple, Generator

from tensorpc.core import BuiltinServiceProcType, prim
from tensorpc.core.client import simple_remote_call
from tensorpc.protos_export import rpc_message_pb2
from tensorpc.utils.proctitle import list_all_tensorpc_server_in_machine, set_tensorpc_server_process_title

from tensorpc.utils.address import get_url_port
from tensorpc.dock import constants
from tensorpc.dock.serv_names import serv_names
from tensorpc.core.httpclient import http_remote_call_request
import tensorpc

import os
import time
from tensorpc.dock.coretypes import Message, MessageItem, MessageLevel
import uuid

import psutil


class MasterMeta:

    def __init__(self) -> None:
        gid = os.getenv(constants.TENSORPC_FLOW_GRAPH_ID)
        nid = os.getenv(constants.TENSORPC_FLOW_NODE_ID)
        nrid = os.getenv(constants.TENSORPC_FLOW_NODE_READABLE_ID)

        port = os.getenv(constants.TENSORPC_FLOW_MASTER_HTTP_PORT)
        gport = os.getenv(constants.TENSORPC_FLOW_MASTER_GRPC_PORT)

        use_rf = os.getenv(constants.TENSORPC_FLOW_USE_REMOTE_FWD)
        fwd_http_port = os.getenv(constants.TENSORPC_FLOW_APP_HTTP_FWD_PORT)

        url = ""
        grpc_url = ""
        if (use_rf is not None and use_rf == "1"):
            if port is not None:
                url = f"http://localhost:{port}/api/rpc"
            if gport is not None:
                grpc_url = f"localhost:{gport}"
        else:
            # for direct connection
            ssh_server = os.getenv("SSH_CLIENT")
            if ssh_server is not None:
                ssh_server_ip = ssh_server.split(" ")[0]
                if ssh_server_ip == "::1":
                    # TODO mac os
                    ssh_server_ip = "localhost"
                if port is not None:
                    url = f"http://{ssh_server_ip}:{port}/api/rpc"
                if gport is not None:
                    grpc_url = f"{ssh_server_ip}:{gport}"
        self._node_readable_id = nrid
        self._graph_id = gid
        self._node_id = nid
        self.grpc_port = gport
        self.http_port = port
        self.grpc_url = grpc_url
        self.http_url = url
        self.fwd_http_port = fwd_http_port

        self.is_grpc_valid = grpc_url != ""
        self.is_http_valid = self.http_url != ""
        self.is_inside_devflow = gid is not None and nid is not None

    @property
    def graph_id(self):
        assert self._graph_id is not None
        return self._graph_id

    @property
    def node_readable_id(self):
        assert self._node_readable_id is not None
        return self._node_readable_id

    @property
    def node_id(self):
        assert self._node_id is not None
        return self._node_id

    def set_process_title(self, app_module_name: Optional[str] = None):
        gport = self.grpc_port if self.grpc_port else 0
        port = self.http_port if self.http_port else 0
        app_meta = AppLocalMeta()
        app_gport = app_meta.grpc_port if app_meta.grpc_port else 0
        app_port = app_meta.http_port if app_meta.http_port else 0
        readable_id = self.node_readable_id
        args = [str(gport), str(port), str(app_gport), str(app_port), readable_id]
        if app_module_name is not None and app_module_name != "":
            args.append(app_module_name)
        return set_tensorpc_server_process_title(BuiltinServiceProcType.DOCK_APP, 
            *args)


class AppLocalMeta:

    def __init__(self) -> None:
        gport = os.getenv(constants.TENSORPC_FLOW_APP_GRPC_PORT)
        port = os.getenv(constants.TENSORPC_FLOW_APP_HTTP_PORT)
        self.module_name = os.getenv(constants.TENSORPC_FLOW_APP_MODULE_NAME,
                                     "")
        url = ""
        grpc_url = ""
        if port is not None:
            url = f"http://localhost:{port}/api/rpc"
        if gport is not None:
            grpc_url = f"localhost:{gport}"
        self.grpc_port = gport
        self.http_port = port
        self.grpc_url = grpc_url
        self.http_url = url
        self.is_inside_devflow = gport is not None and port is not None and self.module_name != ""


@dataclasses.dataclass
class AppProcessMeta:
    name: str
    pid: int
    grpc_port: int
    port: int
    app_grpc_port: int
    app_port: int
    module_name: str = ""

    def create_async_client(self, url: str = "localhost"):
        return AsyncStandaloneAppClient(self, url)

    def create_client(self, url: str = "localhost"):
        return StandaloneAppClient(self, url)

@dataclasses.dataclass
class AppInRelayMeta:
    id: str
    readable_id: str 
    module_name: str
    relay_url: str
    master_url: str

    def create_async_client(self):
        return AsyncRelayAppClient(self.master_url, self.relay_url)

    def create_client(self):
        return RelayAppClient(self.master_url, self.relay_url)

def list_all_app_in_machine():
    res: List[AppProcessMeta] = []
    proc_metas = list_all_tensorpc_server_in_machine(BuiltinServiceProcType.DOCK_APP)
    for meta in proc_metas:
        gport = int(meta.args[0])
        port = int(meta.args[1])
        app_gport = int(meta.args[2])
        app_port = int(meta.args[3])
        readable_id = meta.args[4]
        module_name = ""
        if len(meta.args) > 5:
            module_name = meta.args[5]
        meta = AppProcessMeta(readable_id, meta.pid, gport, port, app_gport, app_port, module_name)
        res.append(meta)
    return res

def list_all_running_apps_in_relay():
    meta = MasterMeta()
    assert meta.is_inside_devflow
    master_url = meta.grpc_url

    all_running_app_infos = simple_remote_call(master_url,
                                               serv_names.FLOW_QUERY_ALL_RUNNING_APPS,
                                               meta.graph_id)
    res: list[AppInRelayMeta] = []
    for info in all_running_app_infos:
        info["master_url"] = master_url
        res.append(AppInRelayMeta(**info))
    return res


def is_inside_devflow():
    meta = MasterMeta()
    return meta.is_inside_devflow


def is_inside_app_session():
    meta = AppLocalMeta()
    return meta.is_inside_devflow


def _get_ids_and_url():
    gid = os.getenv(constants.TENSORPC_FLOW_GRAPH_ID)
    nid = os.getenv(constants.TENSORPC_FLOW_NODE_ID)
    nrid = os.getenv(constants.TENSORPC_FLOW_NODE_READABLE_ID)

    port = os.getenv(constants.TENSORPC_FLOW_MASTER_HTTP_PORT)
    gport = os.getenv(constants.TENSORPC_FLOW_MASTER_GRPC_PORT)

    use_rf = os.getenv(constants.TENSORPC_FLOW_USE_REMOTE_FWD)
    if (use_rf is not None and use_rf == "1"):
        url = f"http://localhost:{port}/api/rpc"
        grpc_url = f"localhost:{gport}"
    else:
        # for direct connection
        ssh_server = os.getenv("SSH_CLIENT")
        if (gid is None or nid is None or ssh_server is None or port is None):
            raise ValueError(
                "this function can only be called via devflow frontend")
        ssh_server_ip = ssh_server.split(" ")[0]
        url = f"http://{ssh_server_ip}:{port}/api/rpc"
        grpc_url = f"{ssh_server_ip}:{gport}"

    return gid, nid, nrid, url, grpc_url


def update_node_status(content: str):
    meta = MasterMeta()
    if meta.is_inside_devflow and meta.is_http_valid:
        # TODO add try catch, if not found, just ignore error.
        http_remote_call_request(meta.http_url,
                                    serv_names.FLOW_UPDATE_NODE_STATUS,
                                    meta.graph_id, meta.node_id, content)
        return True
    return False


def add_message(title: str, level: MessageLevel, items: List[MessageItem]):
    timestamp = time.time_ns()
    gid, nid, nrid, url, grpc_url = _get_ids_and_url()
    if (gid is None or nid is None):
        raise ValueError(
            "this function can only be called via devflow frontend")
    msg = Message(str(uuid.uuid4()), level, timestamp, gid, nid,
                  f"{title} ({nrid})", items)
    tensorpc.simple_remote_call(grpc_url, serv_names.FLOW_ADD_MESSAGE,
                                [msg.to_dict_with_detail()])


def add_info_message(title: str, items: List[MessageItem]):
    return add_message(title, MessageLevel.Info, items)


def add_warning_message(title: str, items: List[MessageItem]):
    return add_message(title, MessageLevel.Warning, items)


def add_error_message(title: str, items: List[MessageItem]):
    return add_message(title, MessageLevel.Error, items)


def query_app_urls(master_url: str, graph_id: str,
                   node_id: str) -> Tuple[Tuple[str, str], str]:
    master_addr, _ = get_url_port(master_url)
    res = tensorpc.simple_remote_call(master_url,
                                      serv_names.FLOW_QUERY_APP_NODE_URLS,
                                      graph_id, node_id)
    assert res is not None
    grpc_url = res["grpc_url"]
    http_url = res["http_url"]
    module_name = res["module_name"]

    res_urls = []
    for url in [grpc_url, http_url]:
        if "localhost" in url:
            url = url.replace("localhost", master_addr)
        res_urls.append(url)
    return tuple(res_urls), module_name


class AppClientBase(tensorpc.RemoteManager):

    def __init__(self,
                 master_url: str,
                 graph_id: str,
                 node_id: str,
                 name="",
                 channel_options=None,
                 credentials=None,
                 print_stdout=True,
                 enabled: bool = True):
        if enabled:
            if master_url == "":
                local_meta = AppLocalMeta()
                module_key = local_meta.module_name
                assert local_meta.is_inside_devflow, "you can only use this in devflow."
                app_urls = (local_meta.grpc_url, local_meta.http_url)
            else:
                app_urls, module_key = query_app_urls(
                    master_url, graph_id, node_id)
        else:
            app_urls = ("", "")
            module_key = ""
        super().__init__(app_urls[0],
                         name,
                         channel_options,
                         credentials,
                         print_stdout,
                         enabled=enabled)
        self.graph_id = graph_id
        self.node_id = node_id
        self.module_key = module_key
        self.remote_key = "tensorpc.dock.serv.flowapp::FlowApp.run_app_service"
        self.async_gen_key = "tensorpc.dock.serv.flowapp::FlowApp.run_app_async_gen_service"
        self.graph_args = []

    def app_remote_call(self,
                        key: str,
                        *args,
                        timeout: Optional[int] = None,
                        rpc_callback="",
                        rpc_flags: int = rpc_message_pb2.PickleArray,
                        **kwargs):
        return self.remote_call(self.remote_key,
                                self.module_key + "." + key,
                                *self.graph_args,
                                *args,
                                rpc_timeout=timeout,
                                rpc_callback=rpc_callback,
                                rpc_flags=rpc_flags,
                                **kwargs)

    def app_chunked_remote_call(self,
                                key: str,
                                *args,
                                timeout: Optional[int] = None,
                                rpc_callback="",
                                rpc_flags: int = rpc_message_pb2.PickleArray,
                                **kwargs):
        return self.chunked_remote_call(self.remote_key,
                                        self.module_key + "." + key,
                                        *self.graph_args,
                                        *args,
                                        rpc_flags=rpc_flags,
                                        **kwargs)

    def app_remote_generator(self,
                             key: str,
                             *args,
                             timeout: Optional[int] = None,
                             rpc_callback="",
                             rpc_flags: int = rpc_message_pb2.PickleArray,
                             **kwargs) -> Generator[Any, None, None]:
        for data in self.remote_generator(self.async_gen_key,
                                          self.module_key + "." + key,
                                          *self.graph_args,
                                          *args,
                                          rpc_flags=rpc_flags,
                                          rpc_callback=rpc_callback,
                                          **kwargs):
            yield data


class AppClient(AppClientBase):

    def __init__(self,
                 master_url: str,
                 graph_id: str,
                 node_id: str,
                 name="",
                 channel_options=None,
                 credentials=None,
                 print_stdout=True,
                 enabled: bool = True):
        assert master_url != "" and graph_id != "" and node_id != ""
        super().__init__(master_url, graph_id, node_id, name, channel_options,
                         credentials, print_stdout, enabled)


class AppLocalClient(AppClientBase):

    def __init__(self,
                 name="",
                 channel_options=None,
                 credentials=None,
                 print_stdout=True,
                 enabled: bool = True):
        super().__init__("", "", "", name, channel_options, credentials,
                         print_stdout, enabled)


class AsyncAppClientBase(tensorpc.AsyncRemoteManager):

    def __init__(self,
                 master_url: str,
                 graph_id: str,
                 node_id: str,
                 name="",
                 channel_options=None,
                 credentials=None,
                 print_stdout=True,
                 enabled: bool = True):
        if enabled:
            if master_url == "":
                local_meta = AppLocalMeta()
                module_key = local_meta.module_name
                assert local_meta.is_inside_devflow, "you can only use this in devflow."
                app_urls = (local_meta.grpc_url, local_meta.http_url)
            else:
                app_urls, module_key = query_app_urls(
                    master_url, graph_id, node_id)
        else:
            app_urls = ("", "")
            module_key = ""

        super().__init__(app_urls[0],
                         name,
                         channel_options,
                         credentials,
                         print_stdout,
                         enabled=enabled)
        self.graph_id = graph_id
        self.node_id = node_id
        self.remote_key = "tensorpc.dock.serv.flowapp::FlowApp.run_app_service"
        self.async_gen_key = "tensorpc.dock.serv.flowapp::FlowApp.run_app_async_gen_service"

        self.graph_args = []
        self.module_key = module_key

    async def app_remote_call(self,
                              key: str,
                              *args,
                              timeout: Optional[int] = None,
                              rpc_callback="",
                              rpc_flags: int = rpc_message_pb2.PickleArray,
                              **kwargs):

        return await self.remote_call(self.remote_key,
                                      self.module_key + "." + key,
                                      *self.graph_args,
                                      *args,
                                      rpc_timeout=timeout,
                                      rpc_callback=rpc_callback,
                                      rpc_flags=rpc_flags,
                                      **kwargs)

    async def app_chunked_remote_call(
            self,
            key: str,
            *args,
            timeout: Optional[int] = None,
            rpc_callback="",
            rpc_flags: int = rpc_message_pb2.PickleArray,
            **kwargs):
        return await self.chunked_remote_call(self.remote_key,
                                              self.module_key + "." + key,
                                              *self.graph_args,
                                              *args,
                                              rpc_flags=rpc_flags,
                                              **kwargs)

    async def app_remote_generator(
            self,
            key: str,
            *args,
            timeout: Optional[int] = None,
            rpc_callback="",
            rpc_flags: int = rpc_message_pb2.PickleArray,
            **kwargs) -> AsyncGenerator[Any, None]:
        async for data in self.remote_generator(self.async_gen_key,
                                                self.module_key + "." + key,
                                                *self.graph_args,
                                                *args,
                                                rpc_flags=rpc_flags,
                                                rpc_callback=rpc_callback,
                                                **kwargs):
            yield data


class AsyncAppClient(AsyncAppClientBase):

    def __init__(self,
                 master_url: str,
                 graph_id: str,
                 node_id: str,
                 name="",
                 channel_options=None,
                 credentials=None,
                 print_stdout=True,
                 enabled: bool = True):
        assert master_url != "" and graph_id != "" and node_id != ""
        super().__init__(master_url, graph_id, node_id, name, channel_options,
                         credentials, print_stdout, enabled)


class AsyncAppLocalClient(AsyncAppClientBase):

    def __init__(self,
                 name="",
                 channel_options=None,
                 credentials=None,
                 print_stdout=True,
                 enabled: bool = True):
        super().__init__("", "", "", name, channel_options, credentials,
                         print_stdout, enabled)


class RemoteCompRpcManager(tensorpc.AsyncRemoteManager):

    def __init__(self,
                 url: str,
                 name="",
                 channel_options=None,
                 credentials=None,
                 print_stdout=True,
                 enabled: bool = True):
        super().__init__(url,
                         name,
                         channel_options,
                         credentials,
                         print_stdout,
                         enabled=enabled)
        self.serv_key = serv_names.REMOTE_COMP_SIMPLE_RPC

    async def comp_remote_call(self,
                            app_key: str,
                            *args,
                            timeout: Optional[int] = None,
                            rpc_callback="",
                            rpc_flags: int = rpc_message_pb2.PickleArray,
                            **kwargs):

        return await self.remote_call(self.serv_key,
                                      app_key,
                                      *args,
                                      timeout=timeout,
                                      rpc_callback=rpc_callback,
                                      rpc_flags=rpc_flags,
                                      **kwargs)

class AsyncStandaloneAppClient(tensorpc.AsyncRemoteManager):

    def __init__(self,
                 meta: AppProcessMeta,
                 url: str = "localhost",
                 name="",
                 channel_options=None,
                 credentials=None,
                 print_stdout=True):
        super().__init__(f"{url}:{meta.app_grpc_port}",
                         name,
                         channel_options,
                         credentials,
                         print_stdout)
        self.remote_key = "tensorpc.dock.serv.flowapp::FlowApp.run_app_service"
        self.async_gen_key = "tensorpc.dock.serv.flowapp::FlowApp.run_app_async_gen_service"


    async def app_remote_call(self,
                              key: str,
                              *args,
                              timeout: Optional[int] = None,
                              rpc_callback="",
                              rpc_flags: int = rpc_message_pb2.PickleArray,
                              **kwargs):

        return await self.remote_call(self.remote_key,
                                      key,
                                      *args,
                                      rpc_timeout=timeout,
                                      rpc_callback=rpc_callback,
                                      rpc_flags=rpc_flags,
                                      **kwargs)

    async def app_chunked_remote_call(
            self,
            key: str,
            *args,
            timeout: Optional[int] = None,
            rpc_callback="",
            rpc_flags: int = rpc_message_pb2.PickleArray,
            **kwargs):
        return await self.chunked_remote_call(self.remote_key,
                                              key,
                                              *args,
                                              rpc_timeout=timeout,
                                              rpc_flags=rpc_flags,
                                              **kwargs)

    async def app_remote_generator(
            self,
            key: str,
            *args,
            timeout: Optional[int] = None,
            rpc_callback="",
            rpc_flags: int = rpc_message_pb2.PickleArray,
            **kwargs) -> AsyncGenerator[Any, None]:
        async for data in self.remote_generator(self.async_gen_key,
                                                key,
                                                *args,
                                                rpc_flags=rpc_flags,
                                                rpc_timeout=timeout,
                                                rpc_callback=rpc_callback,
                                                **kwargs):
            yield data

class StandaloneAppClient(tensorpc.RemoteManager):

    def __init__(self,
                 meta: AppProcessMeta,
                 url: str = "localhost",
                 name="",
                 channel_options=None,
                 credentials=None,
                 print_stdout=True):
        super().__init__(f"{url}:{meta.app_grpc_port}",
                         name,
                         channel_options,
                         credentials,
                         print_stdout)
        self.remote_key = "tensorpc.dock.serv.flowapp::FlowApp.run_app_service"
        self.async_gen_key = "tensorpc.dock.serv.flowapp::FlowApp.run_app_async_gen_service"


    def app_remote_call(self,
                              key: str,
                              *args,
                              timeout: Optional[int] = None,
                              rpc_callback="",
                              rpc_flags: int = rpc_message_pb2.PickleArray,
                              **kwargs):

        return self.remote_call(self.remote_key,
                                      key,
                                      *args,
                                      rpc_timeout=timeout,
                                      rpc_callback=rpc_callback,
                                      rpc_flags=rpc_flags,
                                      **kwargs)

    def app_chunked_remote_call(
            self,
            key: str,
            *args,
            timeout: Optional[int] = None,
            rpc_callback="",
            rpc_flags: int = rpc_message_pb2.PickleArray,
            **kwargs):
        return self.chunked_remote_call(self.remote_key,
                                              key,
                                              *args,
                                              rpc_timeout=timeout,
                                              rpc_flags=rpc_flags,
                                              **kwargs)

    def app_remote_generator(
            self,
            key: str,
            *args,
            timeout: Optional[int] = None,
            rpc_callback="",
            rpc_flags: int = rpc_message_pb2.PickleArray,
            **kwargs) -> Generator[Any, None, None]:
        for data in self.remote_generator(self.async_gen_key,
                                                key,
                                                *args,
                                                rpc_flags=rpc_flags,
                                                rpc_timeout=timeout,
                                                rpc_callback=rpc_callback,
                                                **kwargs):
            yield data

class AsyncRelayAppClient(tensorpc.AsyncRemoteManager):

    def __init__(self,
                 master_url: str,
                 app_relay_url: str,
                 name="",
                 channel_options=None,
                 credentials=None,
                 print_stdout=True):
        
        super().__init__(master_url,
                         name,
                         channel_options,
                         credentials,
                         print_stdout)
        self.remote_key = "tensorpc.dock.serv.flowapp::FlowApp.run_app_service"
        self.async_gen_key = "tensorpc.dock.serv.flowapp::FlowApp.run_app_async_gen_service"
        self._app_relay_url = app_relay_url

    async def app_remote_call(self,
                              key: str,
                              *args,
                              timeout: Optional[int] = None,
                              rpc_callback="",
                              rpc_flags: int = rpc_message_pb2.PickleArray,
                              **kwargs):

        return await self.chunked_remote_call(self.remote_key,
                                      key,
                                      *args,
                                      rpc_timeout=timeout,
                                      rpc_callback=rpc_callback,
                                      rpc_flags=rpc_flags,
                                      rpc_relay_urls=[self._app_relay_url],
                                      **kwargs)

    async def app_chunked_remote_call(
            self,
            key: str,
            *args,
            timeout: Optional[int] = None,
            rpc_callback="",
            rpc_flags: int = rpc_message_pb2.PickleArray,
            **kwargs):
        return await self.chunked_remote_call(self.remote_key,
                                              key,
                                              *args,
                                              rpc_timeout=timeout,
                                              rpc_flags=rpc_flags,
                                              rpc_relay_urls=[self._app_relay_url],
                                              **kwargs)

    async def app_remote_generator(
            self,
            key: str,
            *args,
            timeout: Optional[int] = None,
            rpc_callback="",
            rpc_flags: int = rpc_message_pb2.PickleArray,
            **kwargs) -> AsyncGenerator[Any, None]:
        async for data in self.remote_generator(self.async_gen_key,
                                                key,
                                                *args,
                                                rpc_flags=rpc_flags,
                                                rpc_timeout=timeout,
                                                rpc_callback=rpc_callback,
                                                rpc_relay_urls=[self._app_relay_url],
                                                **kwargs):
            yield data

class RelayAppClient(tensorpc.RemoteManager):

    def __init__(self,
                 master_url: str,
                 app_relay_url: str,
                 name="",
                 channel_options=None,
                 credentials=None,
                 print_stdout=True):
        
        super().__init__(master_url,
                         name,
                         channel_options,
                         credentials,
                         print_stdout)
        self.remote_key = "tensorpc.dock.serv.flowapp::FlowApp.run_app_service"
        self.async_gen_key = "tensorpc.dock.serv.flowapp::FlowApp.run_app_async_gen_service"
        self._app_relay_url = app_relay_url

    def app_remote_call(self,
                              key: str,
                              *args,
                              timeout: Optional[int] = None,
                              rpc_callback="",
                              rpc_flags: int = rpc_message_pb2.PickleArray,
                              **kwargs):

        return self.chunked_remote_call(self.remote_key,
                                      key,
                                      *args,
                                      rpc_timeout=timeout,
                                      rpc_callback=rpc_callback,
                                      rpc_flags=rpc_flags,
                                      rpc_relay_urls=[self._app_relay_url],
                                      **kwargs)

    def app_chunked_remote_call(
            self,
            key: str,
            *args,
            timeout: Optional[int] = None,
            rpc_callback="",
            rpc_flags: int = rpc_message_pb2.PickleArray,
            **kwargs):
        return self.chunked_remote_call(self.remote_key,
                                              key,
                                              *args,
                                              rpc_timeout=timeout,
                                              rpc_flags=rpc_flags,
                                              rpc_relay_urls=[self._app_relay_url],
                                              **kwargs)

    def app_remote_generator(
            self,
            key: str,
            *args,
            timeout: Optional[int] = None,
            rpc_callback="",
            rpc_flags: int = rpc_message_pb2.PickleArray,
            **kwargs) -> Generator[Any, None, None]:
        for data in self.remote_generator(self.async_gen_key,
                                                key,
                                                *args,
                                                rpc_flags=rpc_flags,
                                                rpc_timeout=timeout,
                                                rpc_callback=rpc_callback,
                                                rpc_relay_urls=[self._app_relay_url],
                                                **kwargs):
            yield data


if __name__ == "__main__":
    print(list_all_app_in_machine())
    if is_inside_devflow():
        print(list_all_running_apps_in_relay())

