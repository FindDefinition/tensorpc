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

from typing import List
from . import constants
from .serv_names import serv_names
from tensorpc.core.httpclient import http_remote_call_request
import tensorpc

import os 
import time 
from tensorpc.apps.flow.coretypes import Message, MessageItem, MessageLevel, RelayUpdateNodeEvent
import uuid 

class MasterMeta:
    def __init__(self) -> None:
        gid = os.getenv(constants.TENSORPC_FLOW_GRAPH_ID)
        nid = os.getenv(constants.TENSORPC_FLOW_NODE_ID)
        nrid = os.getenv(constants.TENSORPC_FLOW_NODE_READABLE_ID)

        port = os.getenv(constants.TENSORPC_FLOW_MASTER_HTTP_PORT)
        gport = os.getenv(constants.TENSORPC_FLOW_MASTER_GRPC_PORT)

        use_rf = os.getenv(constants.TENSORPC_FLOW_USE_REMOTE_FWD)
        is_worker_env = os.getenv(constants.TENSORPC_FLOW_IS_WORKER)
        is_worker = is_worker_env is not None and is_worker_env == "1"
        url = ""
        grpc_url = ""
        if (use_rf is not None and use_rf == "1") or is_worker:
            if port is not None:
                url = f"http://localhost:{port}/api/rpc"
            if gport is not None:
                grpc_url = f"localhost:{gport}"
        else:
            # for direct connection
            ssh_server = os.getenv("SSH_CLIENT")
            if ssh_server is not None:
                ssh_server_ip = ssh_server.split(" ")[0]
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
        self.is_worker = is_worker

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


def _get_ids_and_url():
    gid = os.getenv(constants.TENSORPC_FLOW_GRAPH_ID)
    nid = os.getenv(constants.TENSORPC_FLOW_NODE_ID)
    nrid = os.getenv(constants.TENSORPC_FLOW_NODE_READABLE_ID)

    port = os.getenv(constants.TENSORPC_FLOW_MASTER_HTTP_PORT)
    gport = os.getenv(constants.TENSORPC_FLOW_MASTER_GRPC_PORT)

    use_rf = os.getenv(constants.TENSORPC_FLOW_USE_REMOTE_FWD)
    is_worker_env = os.getenv(constants.TENSORPC_FLOW_IS_WORKER)
    is_worker = is_worker_env is not None and is_worker_env == "1"
    if (use_rf is not None and use_rf == "1") or is_worker:
        url = f"http://localhost:{port}/api/rpc"
        grpc_url = f"localhost:{gport}"
    else:
        # for direct connection
        ssh_server = os.getenv("SSH_CLIENT")
        if (gid is None or nid is None or ssh_server is None or port is None):
            raise ValueError("this function can only be called via devflow frontend")
        ssh_server_ip = ssh_server.split(" ")[0]
        url = f"http://{ssh_server_ip}:{port}/api/rpc"
        grpc_url = f"{ssh_server_ip}:{gport}"

    return gid, nid, nrid, is_worker, url, grpc_url

def update_node_status(content: str):
    meta = MasterMeta()
    if meta.is_inside_devflow and meta.is_http_valid:
        # TODO add try catch, if not found, just ignore error.
        if not meta.is_worker:
            http_remote_call_request(meta.http_url, serv_names.FLOW_UPDATE_NODE_STATUS, meta.graph_id, meta.node_id, content)
        else:
            # TODO remove this assert
            assert meta.graph_id is not None 
            assert meta.node_id is not None 
            ev = RelayUpdateNodeEvent(meta.graph_id, meta.node_id, content)
            http_remote_call_request(meta.http_url, serv_names.FLOWWORKER_PUT_WORKER_EVENT_JSON, meta.graph_id, ev.to_dict())
        return True 
    return False 

def add_message(title: str, level: MessageLevel, items: List[MessageItem]):
    timestamp = time.time_ns()
    gid, nid, nrid, is_worker, url, grpc_url = _get_ids_and_url()
    if (gid is None or nid is None):
        raise ValueError("this function can only be called via devflow frontend")
    msg = Message(str(uuid.uuid4()), level, timestamp, gid, nid, f"{title} ({nrid})", items)
    if not is_worker:
        tensorpc.simple_remote_call(grpc_url, serv_names.FLOW_ADD_MESSAGE, [msg.to_dict_with_detail()])
    else:
        tensorpc.simple_remote_call(grpc_url, serv_names.FLOWWORKER_ADD_MESSAGE, gid, [msg.to_dict_with_detail()])
