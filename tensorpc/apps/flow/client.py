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
import os 
from tensorpc.apps.flow.coretypes import MessageItem, MessageLevel, RelayUpdateNodeEvent

def _get_ids_and_url():
    gid = os.getenv(constants.TENSORPC_FLOW_GRAPH_ID)
    nid = os.getenv(constants.TENSORPC_FLOW_NODE_ID)
    port = os.getenv(constants.TENSORPC_FLOW_MASTER_HTTP_PORT)
    use_rf = os.getenv(constants.TENSORPC_FLOW_USE_REMOTE_FWD)
    is_worker_env = os.getenv(constants.TENSORPC_FLOW_IS_WORKER)
    is_worker = is_worker_env is not None and is_worker_env == "1"
    if (use_rf is not None and use_rf == "1") or is_worker:
        url = f"http://localhost:{port}/api/rpc"
    else:
        # for direct connection
        ssh_server = os.getenv("SSH_CLIENT")
        if (gid is None or nid is None or ssh_server is None or port is None):
            raise ValueError("this function can only be called via devflow")
        ssh_server_ip = ssh_server.split(" ")[0]
        url = f"http://{ssh_server_ip}:{port}/api/rpc"
    return gid, nid, is_worker, url

def update_node_status(content: str):
    gid, nid, is_worker, url = _get_ids_and_url()
    if not is_worker:
        http_remote_call_request(url, serv_names.FLOW_UPDATE_NODE_STATUS, gid, nid, content)
    else:
        # TODO remove this assert
        assert gid is not None and nid is not None 
        ev = RelayUpdateNodeEvent(gid, nid, content)
        http_remote_call_request(url, serv_names.FLOWWORKER_PUT_WORKER_EVENT_JSON, gid, ev.to_dict())

def add_message(title: str, timestamp: int, level: MessageLevel, items: List[MessageItem]):
    gid, nid, is_worker, url = _get_ids_and_url()
    if not is_worker:
        http_remote_call_request(url, serv_names.FLOW_UPDATE_NODE_STATUS, gid, nid, content)
    else:
        # TODO remove this assert
        assert gid is not None and nid is not None 
        ev = RelayUpdateNodeEvent(gid, nid, content)
        http_remote_call_request(url, serv_names.FLOWWORKER_PUT_WORKER_EVENT_JSON, gid, ev.to_dict())
