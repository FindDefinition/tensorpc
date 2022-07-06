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
from typing import Any, Dict, List, Optional, Union
import tensorpc
from tensorpc.apps.flow import constants as flowconstants
from tensorpc.apps.flow.flowapp import AppEvent, app_event_from_data
from tensorpc.apps.flow.serv_names import serv_names
from tensorpc import prim
import grpc
from tensorpc.apps.flow.coretypes import MessageEvent, MessageEventType, RelayEvent, RelayEventType, RelaySSHEvent, RelayUpdateNodeEvent, relay_event_from_dict, Message
from tensorpc.autossh.core import (CommandEvent, CommandEventType, EofEvent,
                                   Event, ExceptionEvent, LineEvent, RawEvent,
                                   SSHClient, SSHRequest, SSHRequestType)
from tensorpc.core.httpclient import http_remote_call
from tensorpc.utils.address import convert_url_to_local, get_url_port
from .core import AppNode, CommandNode, Node, NodeWithSSHBase, _get_uid, node_from_data
import time
from tensorpc.utils.wait_tools import get_free_ports

ALL_EVENT_TYPES = Union[RelayEvent, MessageEvent, AppEvent]


async def _get_free_port(count: int):
    return get_free_ports(count)


class FlowClient:
    def __init__(self) -> None:
        self.previous_connection_url = ""
        self._send_loop_queue: "asyncio.Queue[ALL_EVENT_TYPES]" = asyncio.Queue(
        )
        self._send_loop_task: Optional[asyncio.Task] = None
        self.shutdown_ev = asyncio.Event()
        self._cached_nodes: Dict[str, CommandNode] = {}
        self._need_to_send_env: Optional[ALL_EVENT_TYPES] = None
        self.selected_node_uid = ""
        self.lock = asyncio.Lock()
        self._app_q: "asyncio.Queue[AppEvent]" = asyncio.Queue()

    async def delete_message(self, graph_id: str, node_id: str,
                             message_id: str):
        node = self._get_node(graph_id, node_id)
        node.messages.pop(message_id)

    async def query_message(self):
        msgs = []
        for node in self._cached_nodes.values():
            msgs.extend([v.to_dict() for v in node.messages.values()])
        return msgs

    async def query_single_message_detail(self, graph_id: str, node_id: str,
                                          message_id: str):
        node = self._get_node(graph_id, node_id)
        res = node.messages[message_id].to_dict_with_detail()
        return res

    async def put_app_event(self, ev_dict: Dict[str, Any]):
        #
        await self._app_q.put(app_event_from_data(ev_dict))

    async def _send_event(self, ev: ALL_EVENT_TYPES,
                          robj: tensorpc.AsyncRemoteManager):
        if isinstance(ev, RelayUpdateNodeEvent):
            await robj.remote_call(serv_names.FLOW_UPDATE_NODE_STATUS,
                                   ev.graph_id, ev.node_id, ev.content)
        elif isinstance(ev, AppEvent):
            await robj.remote_call(serv_names.FLOW_PUT_APP_EVENT, ev.to_dict())

        elif isinstance(ev, RelaySSHEvent):
            if isinstance(ev.event, (EofEvent, ExceptionEvent)):
                node = self._cached_nodes[ev.uid]
                print(node.readable_id, "DISCONNECTING...", type(ev.event))
                if isinstance(ev.event, ExceptionEvent):
                    print(ev.event.traceback_str)
                await node.shutdown()
                print(node.readable_id, "DISCONNECTED.")
            # print("SEND", ev.event)

            await robj.remote_call(serv_names.FLOW_PUT_WORKER_EVENT, ev.event)
        elif isinstance(ev, MessageEvent):
            await robj.remote_call(serv_names.FLOW_ADD_MESSAGE, ev.rawmsgs)
        else:
            raise NotImplementedError

    async def _grpc_send_loop(self, url: str):
        shut_task = asyncio.create_task(self.shutdown_ev.wait())
        async with tensorpc.AsyncRemoteManager(url) as robj:
            if self._need_to_send_env is not None:
                await self._send_event(self._need_to_send_env, robj)
                self._need_to_send_env = None
            send_task = asyncio.create_task(self._send_loop_queue.get())
            wait_tasks: List[asyncio.Task] = [shut_task, send_task]
            while True:
                # TODO if send fail, save this ev and send after reconnection
                # ev = await self._send_loop_queue.get()
                (done, pending) = await asyncio.wait(
                    wait_tasks, return_when=asyncio.FIRST_COMPLETED)
                if shut_task in done:
                    break
                ev: ALL_EVENT_TYPES = send_task.result()
                send_task = asyncio.create_task(self._send_loop_queue.get())
                wait_tasks: List[asyncio.Task] = [shut_task, send_task]
                try:
                    await self._send_event(ev, robj)
                except Exception as e:
                    # remote call may fail by connection broken
                    # TODO retry for reconnection
                    traceback.print_exc()
                    self._send_loop_task = None
                    self._need_to_send_env = ev
                    # when disconnect to master, enter slient mode
                    for n in self._cached_nodes.items():
                        if isinstance(n, NodeWithSSHBase):
                            n.terminal_close_ts = time.time_ns()
                    self.selected_node_uid = ""
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
        return await self.create_connection(master_url, timeout)

    def connected(self):
        return self._send_loop_task is not None

    def _get_node(self, graph_id: str, node_id: str):
        return self._cached_nodes[_get_uid(graph_id, node_id)]

    def _has_node(self, graph_id: str, node_id: str):
        return _get_uid(graph_id, node_id) in self._cached_nodes

    async def run_ui_event(self, graph_id: str, node_id: str,
                           ui_ev_dict: Dict[str, Any]):
        node = self._get_node(graph_id, node_id)
        assert isinstance(node, AppNode)
        sess = prim.get_http_client_session()
        http_port = node.http_port
        app_url = f"localhost:{http_port}"
        print("RUN APP UI EVENT", app_url)
        return await http_remote_call(sess, app_url,
                                      serv_names.APP_RUN_UI_EVENT, ui_ev_dict)

    async def add_message(self, raw_msgs: List[Any]):
        await self._send_loop_queue.put(
            MessageEvent(MessageEventType.Update, raw_msgs))
        for m in raw_msgs:
            msg = Message.from_dict(m)
            node = self._get_node(msg.graph_id, msg.node_id)
            node.messages[msg.uid] = msg

    async def select_node(self,
                          graph_id: str,
                          node_id: str,
                          width: int = -1,
                          height: int = -1):
        node = self._get_node(graph_id, node_id)
        assert isinstance(node, (NodeWithSSHBase))
        self.selected_node_uid = node.get_uid()
        # here we can't use saved stdout because it contains
        # input string and cause problem.
        # we must use state from xterm.js in frontend.
        # if that terminal closed, we assume no destructive input
        # (have special input charactors) exists
        node.terminal_close_ts = -1
        # print("SELECT NODE", len(node.terminal_state))
        if width >= 0 and height >= 0:
            await self.ssh_change_size(graph_id, node_id, width, height)

        return node.terminal_state

    async def sync_graph(self, graph_id: str, node_datas: List[Dict[str,
                                                                    Any]]):
        new_nodes = [node_from_data(d) for d in node_datas]
        # print("EXIST", self._cached_nodes)
        # print("SYNCED NODES", [n.id for n in new_nodes])
        new_node_dict: Dict[str, CommandNode] = {}
        for new_node in new_nodes:
            uid = new_node.get_uid()
            if uid in self._cached_nodes:
                old_node = self._cached_nodes[uid]
                if old_node.remote_driver_id != new_node.remote_driver_id:
                    # remote driver changed. stop this node.
                    await old_node.shutdown()
            assert isinstance(new_node, CommandNode)
            new_node_dict[uid] = new_node
        for k, v in self._cached_nodes.items():
            if k not in new_node_dict:
                # node removed.
                print("NODE SHUTDOWN???")
                await v.shutdown()
            else:
                # we need to keep local state such as terminal state
                # so we update here instead of replace.
                v.update_data(graph_id, new_node_dict[k]._flow_data)
                new_node_dict[k] = v
        async with self.lock:
            self._cached_nodes = new_node_dict
        res = []
        for node in new_node_dict.values():
            msgs = node.messages
            res.append({
                "id": node.id,
                "last_event": node.last_event.value,
                "stdout": node.stdout,
                "msgs": [m.to_dict() for m in msgs.values()],
            })
        # print("GRAPH", self._cached_nodes)
        return res

    def save_terminal_state(self, graph_id: str, node_id: str, state,
                            timestamp_ms: int):
        if len(state) > 0:
            node = self._get_node(graph_id, node_id)
            print("SAVE STATE", len(state))
            assert isinstance(node, (NodeWithSSHBase))
            node.terminal_state = state
            node.terminal_close_ts = timestamp_ms * 1000000
        self.selected_node_uid = ""

    def _get_node_envs(self, graph_id: str, node_id: str):
        uid = _get_uid(graph_id, node_id)
        node = self._cached_nodes[uid]
        envs: Dict[str, str] = {}
        if isinstance(node, CommandNode):
            envs[flowconstants.TENSORPC_FLOW_GRAPH_ID] = graph_id
            envs[flowconstants.TENSORPC_FLOW_NODE_ID] = node_id
            envs[flowconstants.TENSORPC_FLOW_NODE_UID] = node.get_uid()
            envs[flowconstants.
                 TENSORPC_FLOW_NODE_READABLE_ID] = node.readable_id
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
                msgs = self._cached_nodes[uid].messages
                res.append({
                    "id": nid,
                    "last_event": self._cached_nodes[uid].last_event.value,
                    "stdout": self._cached_nodes[uid].stdout,
                    "msgs": [m.to_dict() for m in msgs.values()],
                })
            else:
                res.append({
                    "id": nid,
                    "last_event": CommandEventType.PROMPT_END.value,
                    "stdout": "",
                    "msgs": [],
                })
        return res

    async def create_ssh_session(self, flow_data: Dict[str,
                                                       Any], graph_id: str,
                                 url: str, username: str, password: str,
                                 init_cmds: str, master_url: str):
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
                if isinstance(ev, RawEvent):
                    node.stdout += ev.raw
                    node.push_raw_event(ev)
                    # we assume node never produce special input strings during
                    # terminal frontend closing.
                    if node.terminal_close_ts >= 0:
                        if ev.timestamp > node.terminal_close_ts:
                            evs = node.collect_raw_event_after_ts(ev.timestamp)
                            print("NODE APPEND STATE")
                            node.terminal_state += "".join(ev.raw
                                                           for ev in evs)
                            node.terminal_close_ts = ev.timestamp
                    if uid != self.selected_node_uid:
                        return
                await self._send_loop_queue.put(RelaySSHEvent(ev, uid))

            envs = self._get_node_envs(graph_id, node.id)
            await node.start_session(callback,
                                     convert_url_to_local(url),
                                     username,
                                     password,
                                     envs=envs)
            if init_cmds:
                await node.input_queue.put(init_cmds)
        await node.run_command(_get_free_port)

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

    async def command_node_input(self, graph_id: str, node_id: str, data: str):
        node = self._get_node(graph_id, node_id)
        # print("INPUT", data.encode("utf-8"))
        if (isinstance(node, (NodeWithSSHBase))):
            if node.is_session_started():
                await node.input_queue.put(data)

    async def ssh_change_size(self, graph_id: str, node_id: str, width: int,
                              height: int):
        # TODO handle remote node
        node = self._get_node(graph_id, node_id)
        if isinstance(node, (NodeWithSSHBase)):
            if node.is_session_started():
                # print("CHANGE SIZE")
                req = SSHRequest(SSHRequestType.ChangeSize, (width, height))
                await node.input_queue.put(req)
            else:
                node.init_terminal_size = (width, height)


class FlowWorker:
    def __init__(self) -> None:
        self.worker_port = prim.get_server_grpc_port()
        self._clients: Dict[str, FlowClient] = {}

    def _get_client(self, graph_id: str):
        # graph_id:
        if graph_id not in self._clients:
            self._clients[graph_id] = FlowClient()
        return self._clients[graph_id]

    async def sync_graph(self, graph_id: str, node_datas: List[Dict[str,
                                                                    Any]]):
        return await self._get_client(graph_id).sync_graph(
            graph_id, node_datas)

    async def create_connection(self, graph_id: str, url: str, timeout: float):
        return await self._get_client(graph_id).create_connection(url, timeout)

    async def create_ssh_session(self, flow_data: Dict[str,
                                                       Any], graph_id: str,
                                 url: str, username: str, password: str,
                                 init_cmds: str, master_url: str):
        return await self._get_client(graph_id).create_ssh_session(
            flow_data, graph_id, url, username, password, init_cmds,
            master_url)

    async def stop(self, graph_id: str, node_id: str):
        return await self._get_client(graph_id).stop(graph_id, node_id)

    async def put_relay_event(self, graph_id: str, ev: RelayEvent):
        return await self._get_client(graph_id)._send_loop_queue.put(ev)

    async def put_relay_event_json(self, graph_id: str, ev_data: dict):
        return await self._get_client(graph_id)._send_loop_queue.put(
            relay_event_from_dict(ev_data))

    def query_nodes_last_event(self, graph_id: str, node_ids: List[str]):
        return self._get_client(graph_id).query_nodes_last_event(
            graph_id, node_ids)

    def close_grpc_connection(self, graph_id: str):
        return self._get_client(graph_id).close_grpc_connection()

    async def select_node(self,
                          graph_id: str,
                          node_id: str,
                          width: int = -1,
                          height: int = -1):
        return await self._get_client(graph_id).select_node(
            graph_id, node_id, width, height)

    def save_terminal_state(self, graph_id: str, node_id: str, state,
                            timestamp_ms: int):
        return self._get_client(graph_id).save_terminal_state(
            graph_id, node_id, state, timestamp_ms)

    async def command_node_input(self, graph_id: str, node_id: str, data: str):
        return await self._get_client(graph_id).command_node_input(
            graph_id, node_id, data)

    async def ssh_change_size(self, graph_id: str, node_id: str, width: int,
                              height: int):
        return await self._get_client(graph_id).ssh_change_size(
            graph_id, node_id, width, height)

    async def delete_message(self, graph_id: str, node_id: str,
                             message_id: str):
        return await self._get_client(graph_id).delete_message(
            graph_id, node_id, message_id)

    async def query_message(self, graph_id: str):
        return await self._get_client(graph_id).query_message()

    async def add_message(self, graph_id: str, raw_msgs: List[Any]):
        return await self._get_client(graph_id).add_message(raw_msgs)

    async def query_single_message_detail(self, graph_id: str, node_id: str,
                                          message_id: str):
        return await self._get_client(graph_id).query_single_message_detail(
            graph_id, node_id, message_id)

    async def put_app_event(self, graph_id: str, ev_dict: Dict[str, Any]):
        return await self._get_client(graph_id).put_app_event(ev_dict)

    async def run_ui_event(self, graph_id: str, node_id: str,
                           ui_ev_dict: Dict[str, Any]):
        return await self._get_client(graph_id).run_ui_event(
            graph_id, node_id, ui_ev_dict)
