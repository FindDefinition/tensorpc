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
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import grpc
import tensorpc
from tensorpc import prim
from tensorpc.flow import constants as flowconstants
from tensorpc.flow.coretypes import (Message, MessageEvent, MessageEventType,
                                     RelayEvent, RelayEventType, RelaySSHEvent,
                                     RelayUpdateNodeEvent, ScheduleEvent,
                                     relay_event_from_dict)
from tensorpc.flow.flowapp.core import AppEvent, AppEventType, ScheduleNextForApp, app_event_from_data
from tensorpc.flow.serv_names import serv_names
from tensorpc.autossh.core import (CommandEvent, CommandEventType, EofEvent,
                                   Event, ExceptionEvent, LineEvent, RawEvent,
                                   SSHClient, SSHRequest, SSHRequestType)
from tensorpc.core import get_grpc_url, get_http_url
from ...core.client import simple_chunk_call
from tensorpc.core.httpclient import http_remote_call
from tensorpc.utils.address import convert_url_to_local, get_url_port
from tensorpc.utils.wait_tools import get_free_ports

from .core import (AppNode, CommandNode, HandleTypes, Node, NodeWithSSHBase,
                   RemoteSSHNode, SessionStatus, _get_uid, node_from_data,
                   JINJA2_VARIABLE_ENV, _extract_graph_node_id)

ALL_EVENT_TYPES = Union[RelayEvent, MessageEvent, AppEvent]


class FlowClient:

    def __init__(self) -> None:
        self.previous_connection_url = ""
        self._send_loop_queue: "asyncio.Queue[ALL_EVENT_TYPES]" = asyncio.Queue(
        )
        self._send_loop_task: Optional[asyncio.Task] = None
        self.shutdown_ev = asyncio.Event()
        self._cached_nodes: Dict[str, CommandNode] = {}
        self._readable_node_map: Dict[str, str] = {}
        self._need_to_send_env: Optional[ALL_EVENT_TYPES] = None
        self.selected_node_uid = ""
        self.lock = asyncio.Lock()
        self.var_dict: Dict[str, str] = {}

        self._driver_node: Optional[RemoteSSHNode] = None

    def get_driver_node(self):
        assert self._driver_node is not None, "this shouldn't happen"
        return self._driver_node

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
        ev = app_event_from_data(ev_dict)
        new_t2e = {}
        for k, v in ev.type_to_event.items():
            if k == AppEventType.ScheduleNext:
                assert isinstance(v, ScheduleNextForApp)
                gid, nid = _extract_graph_node_id(ev.uid)
                await self.schedule_next(gid, nid, v.data)
            else:
                new_t2e[k] = v
        ev.type_to_event = new_t2e
        if new_t2e:
            await self._send_loop_queue.put(app_event_from_data(ev_dict))

    def node_schedule_next(self, graph_id: str, node: Node, ev: ScheduleEvent):
        out_handles = node.get_output_handles(HandleTypes.Output.value)
        next_nodes = [
            self._get_node(graph_id, h.target_node_id) for h in out_handles
        ]
        res: Dict[str, Tuple[Node, ScheduleEvent]] = {}
        res_not_here: Dict[str, ScheduleEvent] = {}

        for h in out_handles:
            if self._has_node(graph_id, h.target_node_id):
                n = self._get_node(graph_id, h.target_node_id)
                if n.schedulable:
                    res[n.id] = (n, ev)
            else:
                res_not_here[h.target_node_id] = ev
        return res, res_not_here

    async def schedule_next(self, graph_id: str, node_id: str,
                            sche_ev_data: Dict[str, Any]):
        # schedule next node(s) of this node with data.
        # in worker, if node exists in remote worker, schedule it.
        # else send schedule event to master.
        cur_sche_ev = ScheduleEvent.from_dict(sche_ev_data)
        node = self._get_node(graph_id, node_id)
        assert node.schedulable, "only command node and scheduler node can be scheduled."
        next_schedule, next_schedule_not_here = self.node_schedule_next(
            graph_id, node, cur_sche_ev)
        for (sche_node, sche_ev) in next_schedule.values():

            if isinstance(sche_node,
                          CommandNode) and not isinstance(sche_node, AppNode):
                if not sche_node.is_session_started():
                    assert isinstance(sche_driv, DirectSSHNode)
                    await self._start_session_direct(graph_id, sche_node,
                                                     sche_driv)
                if sche_node.is_running():
                    sche_node.queued_commands.append(sche_ev)
                else:
                    # TODO if two schedule events come rapidly
                    await sche_node.run_schedule_event(sche_ev)
            elif isinstance(sche_node, AppNode):
                if sche_node.is_running():
                    sess = prim.get_http_client_session()
                    http_port = sche_node.http_port
                    durl, _ = get_url_port(sche_driv.url)
                    if sche_driv.enable_port_forward:
                        app_url = get_http_url("localhost",
                                               sche_node.fwd_http_port)
                    else:
                        app_url = get_http_url(durl, http_port)
                    await http_remote_call(sess, app_url,
                                           serv_names.APP_RUN_SCHEDULE_EVENT,
                                           sche_ev.to_dict())

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
        # TODO support app event merge
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

    def render_command(self, cmd: str):
        template = JINJA2_VARIABLE_ENV.from_string(cmd)
        return template.render(**self.var_dict)

    async def schedule_node(self, graph_id: str, node_id: str, sche_ev_dict):
        sche_node = self._get_node(graph_id, node_id)
        sche_ev = ScheduleEvent.from_dict(sche_ev_dict)
        if isinstance(sche_node,
                      CommandNode) and not isinstance(sche_node, AppNode):
            if sche_node.is_running():
                sche_node.queued_commands.append(sche_ev)
            else:
                # TODO if two schedule events come rapidly
                await sche_node.run_schedule_event(sche_ev)
        elif isinstance(sche_node, AppNode):
            if sche_node.is_running():
                sess = prim.get_http_client_session()
                http_port = sche_node.http_port
                app_url = get_http_url("localhost", http_port)
                await http_remote_call(sess, app_url,
                                       serv_names.APP_RUN_SCHEDULE_EVENT,
                                       sche_ev.to_dict())

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
        uid = _get_uid(graph_id, node_id)
        if uid in self._readable_node_map:
            uid = self._readable_node_map[uid]
        return self._cached_nodes[uid]

    def _has_node(self, graph_id: str, node_id: str):
        uid = _get_uid(graph_id, node_id)
        if uid in self._readable_node_map:
            uid = self._readable_node_map[uid]
        return uid in self._cached_nodes

    async def run_single_event(self, graph_id: str, node_id: str, type: int,
                               ui_ev_dict: Dict[str, Any]):
        node = self._get_node(graph_id, node_id)
        assert isinstance(node, AppNode)
        sess = prim.get_http_client_session()
        http_port = node.http_port
        app_url = get_http_url("localhost", http_port)
        return await http_remote_call(sess, app_url,
                                      serv_names.APP_RUN_SINGLE_EVENT, type,
                                      ui_ev_dict)

    async def run_app_service(self, graph_id: str, node_id: str, key: str,
                              *args, **kwargs):
        node = self._get_node(graph_id, node_id)
        assert isinstance(node, AppNode)
        port = node.grpc_port
        app_url = get_grpc_url("localhost", port)
        return await tensorpc.simple_chunk_call_async(app_url, key, *args,
                                                      **kwargs)

    async def get_layout(self,
                         graph_id: str,
                         node_id: str,
                         editor_only: bool = False):
        node = self._get_node(graph_id, node_id)
        assert isinstance(node, AppNode)
        sess = prim.get_http_client_session()
        http_port = node.http_port
        app_url = get_http_url("localhost", http_port)
        print("GET LAYOUT", app_url)
        return await http_remote_call(sess, app_url, serv_names.APP_GET_LAYOUT,
                                      editor_only)

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

    async def sync_graph(self, graph_id: str, driver_data: Dict[str, Any],
                         node_datas: List[Dict[str, Any]],
                         var_dict: Dict[str, str]):
        new_nodes = [node_from_data(d) for d in node_datas]
        # print("EXIST", self._cached_nodes)
        # print("SYNCED NODES", [n.id for n in new_nodes])
        new_node_dict: Dict[str, CommandNode] = {}
        new_readable_node_dict: Dict[str, str] = {}

        for new_node in new_nodes:
            uid = new_node.get_uid()
            if uid in self._cached_nodes:
                old_node = self._cached_nodes[uid]

                if old_node.driver_id != new_node.driver_id:
                    # remote driver changed. stop this node.
                    await old_node.shutdown()
            assert isinstance(new_node, CommandNode)
            new_node_dict[uid] = new_node
            new_readable_node_dict[new_node.get_readable_uid()] = uid

        driver_node = RemoteSSHNode.from_dict(driver_data)
        self._driver_node = driver_node
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
            self._readable_node_map = new_readable_node_dict
        self.var_dict = var_dict
        res = []
        for node in new_node_dict.values():
            msgs = node.messages
            res.append({
                "id": node.id,
                "last_event": node.last_event.value,
                "session_status": node.get_session_status().value,
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

    def query_nodes_status(self, graph_id: str, node_ids: List[str]):
        res = []
        for nid in node_ids:
            uid = _get_uid(graph_id, nid)
            if uid in self._cached_nodes:
                msgs = self._cached_nodes[uid].messages
                res.append({
                    "id":
                    nid,
                    "last_event":
                    self._cached_nodes[uid].last_event.value,
                    "session_status":
                    self._cached_nodes[uid].get_session_status().value,
                    "stdout":
                    self._cached_nodes[uid].stdout,
                    "msgs": [m.to_dict() for m in msgs.values()],
                })
            else:
                res.append({
                    "id": nid,
                    "last_event": CommandEventType.PROMPT_END.value,
                    "session_status": SessionStatus.Stop.value,
                    "stdout": "",
                    "msgs": [],
                })
        return res

    async def _start_node_ssh_session(self, graph_id: str, node: CommandNode):
        uid = node.get_uid()
        driver_node = self.get_driver_node()
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
            # we don't need any port fwd for remote worker.
            await node.start_session(callback,
                                     convert_url_to_local("localhost:22"),
                                     driver_node.username,
                                     driver_node.password,
                                     envs=envs,
                                     enable_port_forward=False,
                                     is_worker=True)
            if driver_node.remote_init_commands:
                await node.input_queue.put(
                    self.render_command(driver_node.remote_init_commands) +
                    "\n")
        await node.run_command(cmd_renderer=self.render_command)

    async def create_ssh_session(self, flow_data: Dict[str, Any],
                                 graph_id: str):
        # check connection, if not available, try to reconnect
        driver_node = self.get_driver_node()
        await self.check_and_reconnect(driver_node.master_grpc_url)
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
        return await self._start_node_ssh_session(graph_id, node)

    async def stop(self, graph_id: str, node_id: str):
        node = self._cached_nodes[_get_uid(graph_id, node_id)]
        if node.is_session_started():
            await node.send_ctrl_c()
        print("STOP", graph_id, node_id, node.is_session_started())

    async def stop_session(self, graph_id: str, node_id: str):
        node = self._cached_nodes[_get_uid(graph_id, node_id)]
        if node.is_session_started():
            await node.soft_shutdown()

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

    async def sync_graph(self, graph_id: str, driver_data: Dict[str, Any],
                         node_datas: List[Dict[str, Any]],
                         var_dict: Dict[str, str]):
        return await self._get_client(graph_id).sync_graph(
            graph_id, driver_data, node_datas, var_dict)

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

    def query_nodes_status(self, graph_id: str, node_ids: List[str]):
        return self._get_client(graph_id).query_nodes_status(
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

    async def run_app_service(self, graph_id: str, node_id: str, key: str,
                              *args, **kwargs):
        return await self._get_client(graph_id).run_app_service(
            graph_id, node_id, key, *args, **kwargs)

    async def run_single_event(self, graph_id: str, node_id: str, type: int,
                               ui_ev_dict: Dict[str, Any]):
        return await self._get_client(graph_id).run_single_event(
            graph_id, node_id, type, ui_ev_dict)

    async def get_layout(self,
                         graph_id: str,
                         node_id: str,
                         editor_only: bool = False):
        return await self._get_client(graph_id).get_layout(
            graph_id, node_id, editor_only)

    async def stop_session(self, graph_id: str, node_id: str):
        return await self._get_client(graph_id).stop_session(graph_id, node_id)

    async def exit(self):
        for k, v in self._clients.items():
            for n in v._cached_nodes.values():
                if isinstance(n, NodeWithSSHBase):
                    if n.is_session_started():
                        await n.soft_shutdown()
                        await n.exit_event.wait()
        prim.get_async_shutdown_event().set()

    async def schedule_node(self, graph_id: str, node_id: str, sche_ev_dict):
        return await self._get_client(graph_id).schedule_node(
            graph_id, node_id, sche_ev_dict)
