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
""" worker that running inside tmux and manage ssh tasks
"""
import asyncio
import enum
import os
import traceback
from typing import Any, Dict, List, Optional
import tensorpc
from tensorpc.apps.flow import constants as flowconstants
from tensorpc.apps.flow.serv_names import serv_names
from tensorpc import prim
import grpc
from tensorpc.apps.flow.coretypes import RelayEvent, RelayEventType, RelaySSHEvent, RelayUpdateNodeEvent, relay_event_from_dict
from tensorpc.autossh.core import (CommandEvent, CommandEventType, EofEvent,
                                   Event, ExceptionEvent, LineEvent, SSHClient)
from .core import CommandNode, _get_uid


class FlowClient:
    def __init__(self) -> None:
        self.previous_connection_url = ""
        self._send_loop: "asyncio.Queue[RelayEvent]" = asyncio.Queue()
        self._send_loop_task: Optional[asyncio.Task] = None
        self.shutdown_ev = asyncio.Event()
        self._cached_nodes: Dict[str, CommandNode] = {}
        self._need_to_send_env: Optional[RelayEvent] = None

    async def _send_event(self, ev: RelayEvent, robj: tensorpc.AsyncRemoteManager):
        if isinstance(ev, RelayUpdateNodeEvent):
            await robj.remote_call(
                serv_names.FLOW_UPDATE_NODE_STATUS, ev.graph_id,
                ev.node_id, ev.content)
        elif isinstance(ev, RelaySSHEvent):
            if isinstance(ev.event, (EofEvent, ExceptionEvent)):
                node = self._cached_nodes[ev.uid]
                print(node.readable_id, "DISCONNECTING...",
                        type(ev.event))
                if isinstance(ev.event, ExceptionEvent):
                    print(ev.event.traceback_str)
                await node.shutdown()
                print(node.readable_id, "DISCONNECTED.")
            await robj.remote_call(
                serv_names.FLOW_PUT_WORKER_EVENT, ev.event)
        else:
            raise NotImplementedError

    async def _grpc_send_loop(self, url: str):
        shut_task = asyncio.create_task(self.shutdown_ev.wait())
        async with tensorpc.AsyncRemoteManager(url) as robj:
            if self._need_to_send_env is not None:
                await self._send_event(self._need_to_send_env, robj)
                self._need_to_send_env = None
            wait_tasks: List[asyncio.Task] = [
                shut_task, asyncio.create_task(self._send_loop.get())
            ]
            while True:
                # TODO if send fail, save this ev and send after reconnection
                ev = await self._send_loop.get()
                (done,
                pending) = await asyncio.wait(wait_tasks,
                                            return_when=asyncio.FIRST_COMPLETED)
                if shut_task in done:
                    break
                wait_tasks: List[asyncio.Task] = [
                    shut_task, asyncio.create_task(self._send_loop.get())
                ]
                try:
                    await self._send_event(ev, robj)
                except Exception as e:
                    # remote call may fail by connection broken
                    # TODO retry for reconnection
                    traceback.print_exc()
                    self._send_loop_task = None
                    self._need_to_send_env = ev
                    break
        self._send_loop_task = None

    async def create_connection(self, url: str, timeout: float):
        async with tensorpc.AsyncRemoteManager(url) as robj:
            await robj.wait_for_remote_ready(timeout)
        self.previous_connection_url = url
        self.shutdown_ev.clear()
        self._send_loop_task = asyncio.create_task(self._grpc_send_loop(url))

    async def check_and_reconnect(self, master_url: str, timeout: float = 10):
        if self.connected():
            return
        return await self.create_connection(master_url,
                                            timeout)

    def connected(self):
        return self._send_loop_task is not None

    def _get_node_envs(self, graph_id: str, node_id: str):
        uid = _get_uid(graph_id, node_id)
        node = self._cached_nodes[uid]
        envs: Dict[str, str] = {}
        if isinstance(node, CommandNode):
            envs[flowconstants.TENSORPC_FLOW_GRAPH_ID] = graph_id
            envs[flowconstants.TENSORPC_FLOW_NODE_ID] = node_id
            envs[flowconstants.TENSORPC_FLOW_NODE_UID] = node.get_uid()
            envs[flowconstants.TENSORPC_FLOW_MASTER_GRPC_PORT] = str(
                prim.get_server_meta().port)
            envs[flowconstants.TENSORPC_FLOW_MASTER_HTTP_PORT] = str(
                prim.get_server_meta().http_port)
            envs[flowconstants.TENSORPC_FLOW_IS_WORKER] = "1"

        return envs

    def query_nodes_last_event(self, graph_id: str, node_ids: List[str]):
        res = []
        for nid in node_ids:
            uid = _get_uid(graph_id, nid)
            if uid in self._cached_nodes:
                res.append({
                    "id": nid,
                    "last_event": self._cached_nodes[uid].last_event.value,
                    "stdout": self._cached_nodes[uid].stdout,
                })
            else:
                res.append({
                    "id": nid,
                    "last_event": CommandEventType.PROMPT_END,
                    "stdout": "",
                })
        return res 

    async def create_ssh_session(self, flow_data: Dict[str,
                                                       Any], graph_id: str,
                                 url: str, username: str, password: str, init_cmds: str,
                                 master_url: str):
        # check connection, if not available, try to reconnect
        await self.check_and_reconnect(master_url)
        assert self._send_loop_task is not None
        uid = _get_uid(graph_id, flow_data["id"])

        if uid in self._cached_nodes:
            node = self._cached_nodes[uid]
            if node.last_event == CommandEventType.COMMAND_OUTPUT_START:
                # TODO tell master still running
                return
            node.update_data(graph_id, flow_data)
        else:
            node = CommandNode(flow_data, graph_id)
            self._cached_nodes[uid] = node

        if not node.is_session_started():
            async def callback(ev: Event):
                await self._send_loop.put(RelaySSHEvent(ev, uid))
            envs = self._get_node_envs(graph_id, node.id)
            node.start_session(callback, url, username, password, envs=envs)
            if init_cmds:
                await node.input_queue.put(init_cmds)
        await node.run_command()

    async def stop(self, graph_id: str, node_id: str):
        node = self._cached_nodes[_get_uid(graph_id, node_id)]
        if node.is_session_started():
            await node.send_ctrl_c()
        print("STOP", graph_id, node_id, node.is_session_started())

    def close_grpc_connection(self):
        self.shutdown_ev.set()

    async def shutdown_node_session(self, graph_id: str, node_id: str):
        uid = _get_uid(graph_id, node_id)
        if uid not in self._cached_nodes:
            return 
        node = self._cached_nodes[uid]
        if node.is_session_started():
            await node.shutdown()

    async def remove_node(self, graph_id: str, node_id: str):
        uid = _get_uid(graph_id, node_id)
        if uid not in self._cached_nodes:
            return 
        await self.shutdown_node_session(graph_id, node_id)
        self._cached_nodes.pop(uid)

class FlowWorker:
    def __init__(self) -> None:
        self.worker_port = prim.get_server_grpc_port()
        self._clients: Dict[str, FlowClient] = {}

    def _get_client(self, graph_id: str):
        # graph_id: 
        if graph_id not in self._clients:
            self._clients[graph_id] = FlowClient()
        return self._clients[graph_id]

    async def create_connection(self, graph_id: str, url: str,
                                timeout: float):
        return await self._get_client(graph_id).create_connection(
            url, timeout)

    async def create_ssh_session(self, flow_data: Dict[str, Any], graph_id: str,
                                 url: str, username: str, password: str, init_cmds: str,
                                 master_url: str):
        return await self._get_client(graph_id).create_ssh_session(
            flow_data, graph_id, url, username, password, init_cmds,
            master_url)

    async def stop(self, graph_id: str, node_id: str):
        return await self._get_client(graph_id).stop(graph_id, node_id)

    async def put_relay_event(self, graph_id: str, ev: RelayEvent):
        return await self._get_client(graph_id)._send_loop.put(ev)

    async def put_relay_event_json(self, graph_id: str, ev_data: dict):
        return await self._get_client(graph_id)._send_loop.put(relay_event_from_dict(ev_data))
    
    def query_nodes_last_event(self, graph_id: str, node_ids: List[str]):
        return self._get_client(graph_id).query_nodes_last_event(graph_id, node_ids)

    def close_grpc_connection(self, graph_id: str):
        return self._get_client(graph_id).close_grpc_connection()
