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

import asyncio
import enum
import json
from pathlib import Path
import traceback
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
import time
import uuid
import aiohttp
import asyncssh
import tensorpc
from tensorpc import marker, prim, http_remote_call, get_http_url
from tensorpc.apps.flow.coretypes import MessageEventType, MessageLevel, UserContentEvent, UserEvent, UserStatusEvent, Message, MessageEvent, get_uid
from tensorpc.apps.flow.serv_names import serv_names
from tensorpc.apps.flow.constants import (
    FLOW_DEFAULT_GRAPH_ID, FLOW_FOLDER_PATH, TENSORPC_FLOW_GRAPH_ID,
    TENSORPC_FLOW_MASTER_GRPC_PORT, TENSORPC_FLOW_MASTER_HTTP_PORT,
    TENSORPC_FLOW_NODE_ID, TENSORPC_FLOW_NODE_UID,
    TENSORPC_FLOW_USE_REMOTE_FWD, TENSORPC_FLOW_DEFAULT_TMUX_NAME)
from tensorpc.autossh.core import (CommandEvent, CommandEventType, EofEvent,
                                   Event, ExceptionEvent, LineEvent, SSHClient,
                                   RawEvent, SSHRequest, SSHRequestType,
                                   remove_ansi_seq)
from tensorpc.core import get_grpc_url
from tensorpc.core.asynctools import cancel_task
from collections import deque
import bisect
import itertools

from tensorpc.utils.registry import HashableRegistry

ALL_NODES = HashableRegistry()

class HandleTypes(enum.Enum):
    Driver = "driver"

def _extract_graph_node_id(uid: str):
    parts = uid.split("@")
    return parts[0], parts[1]


def _get_uid(graph_id: str, node_id: str):
    return get_uid(graph_id, node_id)


def _get_status_from_last_event(ev: CommandEventType):
    if ev == CommandEventType.COMMAND_OUTPUT_START:
        return "running"
    elif ev == CommandEventType.COMMAND_COMPLETE:
        return "success"
    else:
        return "idle"

class Handle:
    def __init__(self, target_node_id: str, type: str, edge_id: str) -> None:
        self.target_node_id = target_node_id
        self.type = type 
        self.edge_id = edge_id

    def to_dict(self):
        return {
            "target_node_id": self.target_node_id,
            "type": self.type,
            "edge_id": self.edge_id,
        }
    
    def __repr__(self):
        return f"{self.type}@{self.target_node_id}"

    @classmethod
    def from_dict(cls, data: Dict[str, Any ]):
        return cls(data["target_node_id"], data["type"], data["edge_id"])

@ALL_NODES.register
class Node:
    def __init__(self, flow_data: Dict[str, Any], graph_id: str = "") -> None:
        self._flow_data = flow_data
        self.id: str = flow_data["id"]
        self.graph_id: str = graph_id
        self.inputs: Dict[str, List[Handle]] = {}
        self.outputs: Dict[str, List[Handle]] = {}
        self.remote_driver_id: str = ""
        self.messages: Dict[str, Message] = {}

    def to_dict(self):
        # currently this method is only used for remote worker.
        raw = self._flow_data.copy()
        raw.update({
            "tensorpc_flow": {
                "graph_id": self.graph_id,
                "type": type(self).__name__,
                "remote_driver_id": self.remote_driver_id,
                "inputs": {n: [vv.to_dict() for vv in v] for n, v in self.inputs.items()},
                "outputs": {n: [vv.to_dict() for vv in v] for n, v in self.outputs.items()},
            }
        })
        return raw 

    def get_messages_dict(self):
        return [v.to_dict() for v in self.messages.values()]

    def get_local_state(self):
        return {}

    def set_local_state(self, state: Dict[str, Any]):
        return

    @classmethod
    def from_dict(cls, data: Dict[str, Any ]):
        extra_data = data["tensorpc_flow"]
        node = cls(data, extra_data["graph_id"])
        node.remote_driver_id = extra_data['remote_driver_id']
        node.inputs = {n: [Handle.from_dict(vv) for vv in v] for n, v in extra_data["inputs"].items()}
        node.outputs = {n: [Handle.from_dict(vv) for vv in v] for n, v in extra_data["outputs"].items()}
        return node 

    def get_input_handles(self, type: str):
        if type not in self.inputs:
            return []
        return self.inputs[type]

    def get_output_handles(self, type: str):
        if type not in self.outputs:
            return []
        return self.outputs[type]

    @property
    def position(self) -> Tuple[float, float]:
        pos = self._flow_data["position"]
        return (pos["x"], pos["y"])

    @property
    def type(self) -> str:
        return self._flow_data["type"]

    @property
    def node_data(self) -> Dict[str, Any]:
        return self._flow_data["data"]

    @property
    def readable_id(self) -> str:
        return self._flow_data["data"]["readableNodeId"]

    @property
    def raw_data(self) -> Dict[str, Any]:
        return self._flow_data

    def update_data(self, graph_id: str, flow_data: Dict[str, Any]):
        self._flow_data = flow_data
        # graph id may change due to rename
        self.graph_id = graph_id
        self.inputs: Dict[str, List[Handle]] = {}
        self.outputs: Dict[str, List[Handle]] = {}
        if "tensorpc_flow" in flow_data:
            # for remote workers
            extra_data = flow_data["tensorpc_flow"]
            self.remote_driver_id = extra_data['remote_driver_id']
            self.inputs = {n: [Handle.from_dict(vv) for vv in v] for n, v in extra_data["inputs"].items()}
            self.outputs = {n: [Handle.from_dict(vv) for vv in v] for n, v in extra_data["outputs"].items()}

    def get_uid(self):
        return _get_uid(self.graph_id, self.id)

    async def shutdown(self):
        return

    def clear_connections(self):
        self.inputs.clear()
        self.outputs.clear()

def node_from_data(data: Dict[str, Any]) -> Node:
    for k, v in ALL_NODES.items():
        if k == data["tensorpc_flow"]["type"]:
            return v.from_dict(data)
    raise ValueError("not found", data["tensorpc_flow"]["type"])

@ALL_NODES.register
class DirectSSHNode(Node):
    @property
    def url(self) -> str:
        return self.node_data["url"]

    @property
    def username(self) -> str:
        return self.node_data["username"]

    @property
    def password(self) -> str:
        return self.node_data["password"]

    @property
    def enable_port_forward(self) -> bool:
        return self.node_data["enablePortForward"]

    @property
    def init_commands(self) -> str:
        cmds = self.node_data["initCommands"].strip() + "\n"
        return cmds


class NodeWithSSHBase(Node):
    def __init__(self, flow_data: Dict[str, Any], graph_id: str = "") -> None:
        super().__init__(flow_data, graph_id)
        self.shutdown_ev = asyncio.Event()
        self.task: Optional[asyncio.Task] = None
        self.input_queue = asyncio.Queue()
        self.last_event: CommandEventType = CommandEventType.PROMPT_END
        self.stdout = ""
        self.init_terminal_size: Tuple[int, int] = (34, 16)
        self.terminal_state = ""
        self.terminal_close_ts: int = -1
        self._raw_event_history: "deque[RawEvent]" = deque(maxlen=10000)


    def push_raw_event(self, ev: RawEvent):
        self._raw_event_history.append(ev)

    def collect_raw_event_after_ts(self, ts: int):
        left = bisect.bisect_left(self._raw_event_history, ts, 0,
                                  len(self._raw_event_history))
        return itertools.islice(self._raw_event_history, left)

    async def send_ctrl_c(self):
        # https://github.com/ronf/asyncssh/issues/112#issuecomment-343318916
        return await self.input_queue.put("\x03")

    async def shutdown(self):
        print("NODE", self.id, "SHUTDOWN")
        if self.task is not None:
            self.shutdown_ev.set()
            await cancel_task(self.task)
            self.task = None
            self.shutdown_ev.clear()

    def is_session_started(self):
        return self.task is not None

@ALL_NODES.register
class RemoteSSHNode(NodeWithSSHBase):
    def __init__(self, flow_data: Dict[str, Any], graph_id: str = "") -> None:
        super().__init__(flow_data, graph_id)
        self.worker_port: int = -1
        self.worker_http_port: int = -1
        self.remote_master_port: int = -1
        self.remote_master_http_port: int = -1
        self.remote_port = 54051
        self.remote_http_port = 54052
        self._remote_self_ip = ""

    @property
    def worker_http_url(self) -> str:
        if self.enable_port_forward:
            assert self.worker_http_port >= 0

            return get_http_url("localhost", self.worker_http_port)

        return get_http_url(self.url, self.remote_http_port)
    
    @property
    def worker_grpc_url(self) -> str:
        if self.enable_port_forward:
            assert self.worker_port >= 0

            return get_grpc_url("localhost", self.worker_port)

        return get_grpc_url(self.url, self.remote_port)

    @property
    def master_grpc_url(self) -> str:
        if self.enable_port_forward:
            assert self.remote_master_port >= 0
            return get_grpc_url("localhost", self.remote_master_port)
        assert self._remote_self_ip != ""
        return get_grpc_url(self._remote_self_ip, prim.get_server_grpc_port())

    async def http_remote_call(self, key: str, *args, **kwargs):
        return await http_remote_call(prim.get_http_client_session(),
                                      self.worker_http_url, key, *args,
                                      **kwargs)

    @property
    def url(self) -> str:
        return self.node_data["url"]

    @property
    def username(self) -> str:
        return self.node_data["username"]

    @property
    def password(self) -> str:
        return self.node_data["password"]

    @property
    def enable_port_forward(self) -> bool:
        return self.node_data["enablePortForward"]

    @property
    def init_commands(self) -> str:
        cmds = self.node_data["initCommands"].strip() + "\n"
        return cmds

    @property
    def remote_init_commands(self) -> str:
        cmds = self.node_data["remoteInitCommands"].strip() + "\n"
        return cmds

    def _env_port_modifier(self, fports: List[int], rfports: List[int],
                           env: Dict[str, str]):
        if fports:
            self.worker_port = fports[0]
            self.worker_http_port = fports[1]
        self.remote_master_port = rfports[0]
        if len(rfports) > 1:
            self.remote_master_http_port = rfports[1]

    async def start_session(self,
                            callback: Callable[[Event], Awaitable[None]],
                            url: str,
                            username: str,
                            password: str,
                            rfports: Optional[List[int]] = None):
        assert self.task is None
        new_sess_name = TENSORPC_FLOW_DEFAULT_TMUX_NAME
        client = SSHClient(url, username, password, None, self.get_uid())
        # firstly we check if the tmux worker exists
        # if exists, we need to query some node state from remote.
        worker_exists: bool = False
        async with client.simple_connect() as conn:
            try:
                result = await conn.run('tmux ls', check=True)
                stdout = result.stdout
                if stdout is not None:
                    if isinstance(stdout, bytes):
                        stdout = stdout.decode("utf-8")
                    sess_lines = stdout.strip().split("\n")
                    sess_names = [s.split(":")[0] for s in sess_lines]
                    worker_exists = new_sess_name in sess_names
            except asyncssh.process.ProcessError:
                worker_exists = False
        print("CONNECT", url)

        # TODO sync graph, close removed node when we reconnect
        # a remote worker.
        # we assume that if this session is started, we
        # can connect to remote worker successfully.
        async def callback2(ev: Event):
            # if isinstance(ev, LineEvent):
            #     self.stdout += ev.line
            # elif isinstance(ev, CommandEvent):
            #     self.last_event = ev.type
            await callback(ev)

        async def exit_callback():
            self.task = None
            http_url = self.worker_http_url
            self.last_event = CommandEventType.PROMPT_END
            self.worker_port = -1
            self.worker_http_port = -1
            self.remote_master_port = -1
            self.remote_master_http_port = -1
            self._remote_self_ip = ""
            print("SESSION EXIT!!!")
            # TODO send message to disable grpc client in
            # remote worker.
            try:
                await http_remote_call(prim.get_http_client_session(),
                                       http_url,
                                       serv_names.FLOWWORKER_CLOSE_CONNECTION,
                                       self.graph_id, self.id)
            except:
                traceback.print_exc()

        def client_ip_callback(cip: str):
            self._remote_self_ip = cip.strip()

        sd_task = asyncio.create_task(self.shutdown_ev.wait())
        forward_ports = []
        if self.enable_port_forward:
            forward_ports = [self.remote_port, self.remote_http_port]
        self.task = asyncio.create_task(
            client.connect_queue(self.input_queue,
                                 callback2,
                                 sd_task,
                                 forward_ports=forward_ports,
                                 r_forward_ports=rfports,
                                 env_port_modifier=self._env_port_modifier,
                                 exit_callback=exit_callback,
                                 client_ip_callback=client_ip_callback))
        await self.input_queue.put(
            SSHRequest(SSHRequestType.ChangeSize, self.init_terminal_size))
        return worker_exists

    async def run_command(self):
        if self.init_commands:
            await self.input_queue.put(self.init_commands)
        await self.input_queue.put((
            f"python -m tensorpc.cli.start_worker --name={TENSORPC_FLOW_DEFAULT_TMUX_NAME} "
            f"--port={self.remote_port} "
            # f"--http_port={self.remote_http_port} && while :; do sleep 2073600; done\n"
            f"--http_port={self.remote_http_port}\n"
        ))

@ALL_NODES.register
class EnvNode(Node):
    pass

@ALL_NODES.register
class CommandNode(NodeWithSSHBase):
    def __init__(self, flow_data: Dict[str, Any], graph_id: str = "") -> None:
        super().__init__(flow_data, graph_id)

    @property
    def commands(self):
        args = self.node_data["args"]
        return [x["value"] for x in filter(lambda x: x["enabled"], args)]

    async def run_command(self):
        await self.input_queue.put(" ".join(self.commands) + "\n")

    async def start_session(
        self,
        callback: Callable[[Event], Awaitable[None]],
        #   msg_q: asyncio.Queue,
        url: str,
        username: str,
        password: str,
        envs: Dict[str, str],
        rfports: Optional[List[int]] = None,
        env_port_modifier: Optional[Callable[
            [List[int], List[int], Dict[str, str]], None]] = None):
        assert self.task is None
        client = SSHClient(url, username, password, None, self.get_uid())

        # async def callback(ev: Event):
        #     await msg_q.put(ev)
        async def exit_callback():
            self.task = None
            self.last_event = CommandEventType.PROMPT_END

        sd_task = asyncio.create_task(self.shutdown_ev.wait())
        self.task = asyncio.create_task(
            client.connect_queue(self.input_queue,
                                 callback,
                                 sd_task,
                                 env=envs,
                                 r_forward_ports=rfports,
                                 env_port_modifier=env_port_modifier,
                                 exit_callback=exit_callback))
        await self.input_queue.put(
            SSHRequest(SSHRequestType.ChangeSize, self.init_terminal_size))


_TYPE_TO_NODE_CLS: Dict[str, Type[Node]] = {
    "command": CommandNode,
    "env": EnvNode,
    "directssh": DirectSSHNode,
    "input": Node,
    "remotessh": RemoteSSHNode,
}


class Edge:
    def __init__(self, flow_data: Dict[str, Any], graph_id: str = "") -> None:
        self._flow_data = flow_data
        self.id: str = flow_data["id"]
        self.graph_id: str = graph_id

    @property
    def raw_data(self) -> Dict[str, Any]:
        return self._flow_data

    def get_uid(self):
        return _get_uid(self.graph_id, self.id)

    def update_data(self, graph_id: str, flow_data: Dict[str, Any]):
        self._flow_data = flow_data
        self.graph_id = graph_id

    @property
    def source_id(self):
        return self._flow_data["source"]

    @property
    def target_id(self):
        return self._flow_data["target"]

    @property
    def source_handle(self):
        return self._flow_data["sourceHandle"]

    @property
    def target_handle(self):
        return self._flow_data["targetHandle"]


class FlowGraph:
    def __init__(self, flow_data: Dict[str, Any], graph_id: str = "") -> None:
        graph_data = flow_data

        nodes = [
            _TYPE_TO_NODE_CLS[d["type"]](d, graph_id)
            for d in graph_data["nodes"]
        ]
        edges = [Edge(d, graph_id) for d in graph_data["edges"]]
        self.viewport = graph_data["viewport"]

        self._node_id_to_node = {n.id: n for n in nodes}
        self._node_rid_to_node = {n.readable_id: n for n in nodes}

        self._edge_id_to_edge = {n.id: n for n in edges}
        self._update_connection(edges)
        self._update_driver()

        self.graph_id = graph_id
        self.ssh_data = flow_data["ssh"]

        self.messages: Dict[str, Message] = {}

    def _update_connection(self, edges: List[Edge]):
        for k, v in self._node_id_to_node.items():
            v.clear_connections()
        for edge in edges:
            source = edge.source_id
            target = edge.target_id
            src_handle = Handle(source, edge.source_handle, edge.id)
            tgt_handle = Handle(target, edge.target_handle, edge.id)
            source_outs = self._node_id_to_node[source].outputs
            if tgt_handle.type not in source_outs:
                source_outs[tgt_handle.type] = []
            source_outs[tgt_handle.type].append(tgt_handle)
            target_outs = self._node_id_to_node[target].inputs
            if src_handle.type not in target_outs:
                target_outs[src_handle.type] = []
            target_outs[src_handle.type].append(src_handle)

    def get_input_nodes_of_handle_type(self, node: Node, type: HandleTypes):
        out_handles = node.get_input_handles(type.value)
        out_nodes = [self.get_node_by_id(h.target_node_id) for h in out_handles]
        return out_nodes

    def get_output_nodes_of_handle_type(self, node: Node, type: HandleTypes):
        out_handles = node.get_output_handles(type.value)
        out_nodes = [self.get_node_by_id(h.target_node_id) for h in out_handles]
        return out_nodes

    def _update_driver(self, ):
        for k, v in self._node_id_to_node.items():
            v.remote_driver_id = ""
        for node in self.nodes:
            if isinstance(node, RemoteSSHNode):
                out_nodes = self.get_output_nodes_of_handle_type(node, HandleTypes.Driver)
                for n in out_nodes:
                    n.remote_driver_id = node.id
                    # print("WTF", node.readable_id, n.readable_id)
    def update_nodes(self, nodes: Iterable[Node]):
        self._node_id_to_node = {n.id: n for n in nodes}
        self._node_rid_to_node = {n.readable_id: n for n in nodes}

    def get_node_by_id(self, node_id: str):
        if node_id in self._node_id_to_node:
            return self._node_id_to_node[node_id]
        else:
            return self._node_rid_to_node[node_id]

    def node_exists(self, node_id: str):
        if node_id in self._node_id_to_node:
            return True
        else:
            return node_id in self._node_rid_to_node

    def get_edge_by_id(self, edge_id: str):
        return self._edge_id_to_edge[edge_id]

    @property
    def nodes(self):
        return self._node_id_to_node.values()

    @property
    def edges(self):
        return self._edge_id_to_edge.values()

    def to_dict(self):
        return {
            "viewport": self.viewport,
            "nodes": [n.raw_data for n in self.nodes],
            "edges": [n.raw_data for n in self.edges],
            "ssh": self.ssh_data,
            "id": self.graph_id,
        }

    async def update_graph(self, graph_id: str, new_flow_data):
        """TODO delete message when node is deleted.
        """
        # we may need to shutdown node, so use async function
        new_graph_data = new_flow_data
        self.ssh_data = new_flow_data["ssh"]
        self.viewport = new_graph_data["viewport"]
        self.graph_id = graph_id
        nodes = [
            _TYPE_TO_NODE_CLS[d["type"]](d, graph_id)
            for d in new_graph_data["nodes"]
        ]
        edges = [Edge(d, graph_id) for d in new_graph_data["edges"]]
        new_node_id_to_node: Dict[str, Node] = {}
        # update unchanged node data
        for node in nodes:
            if node.id in self._node_id_to_node:
                self._node_id_to_node[node.id].update_data(
                    graph_id, node.raw_data)
                new_node_id_to_node[node.id] = self._node_id_to_node[node.id]
            else:
                # new node. just append to node
                new_node_id_to_node[node.id] = node
        # handle deleted node
        for node in self._node_id_to_node.values():
            if node.id not in new_node_id_to_node:
                if node.remote_driver_id == "":
                    # shutdown this node
                    await node.shutdown()
                else:
                    # if node is a remote node,
                    # shutdown process will be handled
                    # in sync_graph RPC.
                    pass

        self.update_nodes(new_node_id_to_node.values())
        # we assume edges don't contain any state, so just update them.
        # we may need to handle this in future.
        self._edge_id_to_edge = {n.id: n for n in edges}
        self._update_connection(edges)
        self._update_driver()
        return


def _empty_flow_graph(graph_id: str = ""):
    data = {
        "nodes": [],
        "edges": [],
        "viewport": {
            "x": 0,
            "y": 0,
            "zoom": 1,
        },
        "ssh": {
            "url": "",
            "username": "",
            "password": "",
        },
        "id": graph_id,
    }
    return FlowGraph(data, graph_id)


class Flow:
    def __init__(self, root: Optional[str] = None) -> None:
        self._user_ev_q: "asyncio.Queue[Tuple[str, UserEvent]]" = asyncio.Queue(
        )
        self._ssh_q: "asyncio.Queue[Event]" = asyncio.Queue()
        self._msg_q: "asyncio.Queue[MessageEvent]" = asyncio.Queue()

        # self._ssh_stdout_q: "asyncio.Queue[Tuple[str, Event]]" = asyncio.Queue()
        self.selected_node_uid: str = ""
        if root is None or root == "":
            root = str(FLOW_FOLDER_PATH)
        self.root = Path(root)
        if not self.root.exists():
            self.root.mkdir(0o755, True, True)
        self.default_flow_path = self.root / f"{FLOW_DEFAULT_GRAPH_ID}.json"
        if not self.default_flow_path.exists():
            with self.default_flow_path.open("w") as f:
                json.dump(
                    _empty_flow_graph(FLOW_DEFAULT_GRAPH_ID).to_dict(), f)
        self.flow_dict: Dict[str, FlowGraph] = {}
        for flow_path in self.root.glob("*.json"):
            with flow_path.open("r") as f:
                flow_data = json.load(f)
            self.flow_dict[flow_path.stem] = FlowGraph(flow_data,
                                                       flow_path.stem)

    def _get_node(self, graph_id: str, node_id: str):
        return self.flow_dict[graph_id].get_node_by_id(node_id)

    @marker.mark_websocket_ondisconnect
    def _on_client_disconnect(self, cl):
        # TODO when all client closed instead of one client, close all terminal
        for g in self.flow_dict.values():
            for n in g.nodes:
                if isinstance(n, NodeWithSSHBase):
                    if n.terminal_close_ts < 0:
                        n.terminal_close_ts = time.time_ns()

    def _node_exists(self, graph_id: str, node_id: str):
        if graph_id not in self.flow_dict:
            return False
        return self.flow_dict[graph_id].node_exists(node_id)

    @marker.mark_websocket_event
    async def node_user_event(self):
        # ws client wait for this event to get new node update msg
        (uid, userev) = await self._user_ev_q.get()
        return prim.DynamicEvent(uid, userev.to_dict())

    @marker.mark_websocket_event
    async def message_event(self):
        # ws client wait for this event to get new node update msg
        ev = await self._msg_q.get()
        return ev.to_dict()

    async def put_event_from_worker(self, ev: Event):
        await self._ssh_q.put(ev)
        
    async def add_message(self, raw_msgs: List[Any]):
        await self._msg_q.put(MessageEvent(MessageEventType.Update, raw_msgs))
        for m in raw_msgs:
            msg = Message.from_dict(m)
            node = self._get_node(msg.graph_id, msg.node_id)
            node.messages[msg.uid] = msg 

    async def delete_message(self, graph_id: str, node_id: str, message_id: str):
        node = self._get_node(graph_id, node_id)
        if message_id in node.messages:
            node.messages.pop(message_id)

    async def query_single_message_detail(self, graph_id: str, node_id: str, message_id: str):
        node = self._get_node(graph_id, node_id)
        return node.messages[message_id].to_dict_with_detail()

    async def query_message(self, graph_id: str):
        print("QUERY INIT MESSAGE")
        graph = self.flow_dict[graph_id]
        all_msgs = []
        for node in graph.nodes:
            if node.remote_driver_id != "":
                # handle remote node in remote driver
                continue 
            if isinstance(node, RemoteSSHNode):
                if node.is_session_started():
                    msgs = await node.http_remote_call(
                        serv_names.FLOWWORKER_QUERY_MESSAGE, graph_id)
                    all_msgs.extend(msgs)
            all_msgs.extend(node.get_messages_dict())
        # sort in frontend
        return {m["uid"]: m for m in all_msgs}

    @marker.mark_websocket_event
    async def command_node_event(self):
        # uid: {graph_id}@{node_id}
        while True:
            event = await self._ssh_q.get()
            uid = event.uid
            # print(event, f"uid={uid}", self.selected_node_uid)
            graph_id, node_id = _extract_graph_node_id(uid)
            node = self._get_node(graph_id, node_id)
            assert isinstance(node, (CommandNode, RemoteSSHNode))
            # if isinstance(event, LineEvent):
            #     # print(node.id, self.selected_node_uid == uid, event.line, end="")
            #     node.stdout += event.line
            #     if uid != self.selected_node_uid:
            #         continue
            if isinstance(event, RawEvent):
                # print(node.id, self.selected_node_uid == uid, event.line, end="")
                node.stdout += event.raw
                node.push_raw_event(event)
                # we assume node never produce special input strings during
                # terminal frontend closing.
                if node.terminal_close_ts >= 0:
                    if event.timestamp > node.terminal_close_ts:
                        evs = node.collect_raw_event_after_ts(event.timestamp)
                        node.terminal_state += "".join(ev.raw for ev in evs)
                        node.terminal_close_ts = event.timestamp
                        # print("NODE APPEND STATE")

                if uid != self.selected_node_uid:
                    continue

            elif isinstance(event, (CommandEvent)):
                node.last_event = event.type
                # if event.type == CommandEventType.PROMPT_END:
                #     node.stdout += str(event.arg)

            elif isinstance(event, (EofEvent, ExceptionEvent)):
                print(node.readable_id, "DISCONNECTING...", type(event))
                if isinstance(event, ExceptionEvent):
                    print(event.traceback_str)
                await node.shutdown()
                print(node.readable_id, "DISCONNECTED.")
            return prim.DynamicEvent(uid, event.to_dict())

    def update_node_status(self, graph_id: str, node_id: str, content: Any):
        # user client call this rpc to send message to frontend.
        loop = asyncio.get_running_loop()
        uid = _get_uid(graph_id, node_id)
        ev = UserContentEvent(content)
        asyncio.run_coroutine_threadsafe(self._user_ev_q.put((uid, ev)), loop)

    def query_node_last_event(self, graph_id: str, node_id: str):
        if not self._node_exists(graph_id, node_id):
            return CommandEventType.PROMPT_END.value
        node = self._get_node(graph_id, node_id)
        if isinstance(node, (NodeWithSSHBase)):
            return node.last_event.value
        return CommandEventType.PROMPT_END.value

    async def save_terminal_state(self, graph_id: str, node_id: str, state,
                                  timestamp_ms: int):
        if len(state) > 0:
            node = self._get_node(graph_id, node_id)
            assert isinstance(node, (NodeWithSSHBase))
            print("node.remote_driver_id", node.remote_driver_id)
            if node.remote_driver_id != "":
                driver = self._get_node(graph_id, node.remote_driver_id)
                assert isinstance(driver, RemoteSSHNode)
                self.selected_node_uid = ""
                return await driver.http_remote_call(
                    serv_names.FLOWWORKER_SET_TERMINAL_STATE, graph_id,
                    node_id, state, timestamp_ms)

            node.terminal_state = state
            node.terminal_close_ts = timestamp_ms * 1000000
        self.selected_node_uid = ""

    async def select_node(self, graph_id: str, node_id: str, width: int = -1, height: int = -1):
        node = self._get_node(graph_id, node_id)
        assert isinstance(node, (NodeWithSSHBase))
        drv_nodes = self.flow_dict[graph_id].get_input_nodes_of_handle_type(node, HandleTypes.Driver)
        if node.remote_driver_id != "":
            driver = self._get_node(graph_id, node.remote_driver_id)
            assert isinstance(driver, RemoteSSHNode)
            if driver.is_session_started():
                res = await driver.http_remote_call(
                    serv_names.FLOWWORKER_SELECT_NODE, graph_id,
                    node_id, width, height)
            else:
                res = ""
            self.selected_node_uid = node.get_uid()
            return res 
        self.selected_node_uid = node.get_uid()
        # here we can't use saved stdout because it contains
        # input string and cause problem.
        # we must use state from xterm.js in frontend.
        # if that terminal closed, we assume no destructive input
        # (have special input charactors) exists
        node.terminal_close_ts = -1
        if width >= 0 and height >= 0:
            await self.ssh_change_size(graph_id, node_id, width, height)
        return node.terminal_state

    async def command_node_input(self, graph_id: str, node_id: str, data: str):
        node = self._get_node(graph_id, node_id)
        # print("INPUT", data.encode("utf-8"))
        if (isinstance(node, (NodeWithSSHBase))):
            if node.remote_driver_id != "":
                driver = self._get_node(graph_id, node.remote_driver_id)
                assert isinstance(driver, RemoteSSHNode)
                if driver.is_session_started():
                    return await driver.http_remote_call(
                        serv_names.FLOWWORKER_COMMAND_NODE_INPUT, graph_id,
                        node_id, data)

            if node.is_session_started():
                await node.input_queue.put(data)

    async def ssh_change_size(self, graph_id: str, node_id: str, width: int,
                              height: int):
        # TODO handle remote node
        node = self._get_node(graph_id, node_id)
        if isinstance(node, (NodeWithSSHBase)):
            if node.remote_driver_id != "":
                driver = self._get_node(graph_id, node.remote_driver_id)
                assert isinstance(driver, RemoteSSHNode)
                if driver.is_session_started():
                    return await driver.http_remote_call(
                        serv_names.FLOWWORKER_SSH_CHANGE_SIZE, graph_id,
                        node_id, width, height)
            if node.is_session_started():
                req = SSHRequest(SSHRequestType.ChangeSize, (width, height))
                await node.input_queue.put(req)
            else:
                node.init_terminal_size = (width, height)

    async def save_graph(self, graph_id: str, flow_data):
        # TODO do we need a async lock here?
        # TODO we should sync graph to remote worker in every graph update
        
        ssh_data = flow_data["ssh"]
        flow_data = flow_data["graph"]
        flow_data["ssh"] = ssh_data
        flow_data["id"] = graph_id
        if graph_id in self.flow_dict:
            await self.flow_dict[graph_id].update_graph(graph_id, flow_data)
        else:
            self.flow_dict[graph_id] = FlowGraph(flow_data, graph_id)
        # print ("SAVE GRAPH", [n.id for n in self.flow_dict[graph_id].nodes])
        graph = self.flow_dict[graph_id]
        for n in flow_data["nodes"]:
            n["selected"] = False
        flow_path = self.root / f"{graph_id}.json"
        with flow_path.open("w") as f:
            json.dump(flow_data, f)
        for node in graph.nodes:
            if isinstance(node, RemoteSSHNode):
                # sync graph node to remote 
                if node.is_session_started():
                    driv_nodes = graph.get_output_nodes_of_handle_type(node, HandleTypes.Driver)
                    await node.http_remote_call(serv_names.FLOWWORKER_SYNC_GRAPH, 
                        graph_id, [n.to_dict() for n in driv_nodes])

    async def load_default_graph(self):
        return await self.load_graph(FLOW_DEFAULT_GRAPH_ID, force_reload=False)

    async def load_graph(self, graph_id: str, force_reload: bool = False):
        flow_path = self.root / f"{graph_id}.json"
        with flow_path.open("r") as f:
            flow_data = json.load(f)
        for n in flow_data["nodes"]:
            n["selected"] = False
            if "width" in n:
                n.pop("width")
            if "height" in n:

                n.pop("height")

            if "handleBounds" in n:
                n.pop("handleBounds")
        # print(json.dumps(flow_data, indent=2))
        if force_reload:
            reload = True
        else:
            reload = graph_id not in self.flow_dict
        if graph_id in self.flow_dict:
            graph = self.flow_dict[graph_id]
            # update node status
            for n in graph.nodes:
                if isinstance(n, (CommandNode, RemoteSSHNode)):
                    uid = _get_uid(graph_id, n.id)

                    await self._user_ev_q.put(
                        (uid,
                         UserStatusEvent(
                             _get_status_from_last_event(n.last_event))))
        if reload:
            # TODO we should shutdown all session first.
            self.flow_dict[graph_id] = FlowGraph(flow_data, graph_id)

        return flow_data

    def _get_node_envs(self, graph_id: str, node_id: str):
        node = self.flow_dict[graph_id].get_node_by_id(node_id)
        envs: Dict[str, str] = {}
        if isinstance(node, CommandNode):
            envs[TENSORPC_FLOW_NODE_ID] = node_id
            envs[TENSORPC_FLOW_GRAPH_ID] = graph_id
            envs[TENSORPC_FLOW_NODE_ID] = node_id
            envs[TENSORPC_FLOW_NODE_UID] = node.get_uid()
            envs[TENSORPC_FLOW_MASTER_GRPC_PORT] = str(
                prim.get_server_meta().port)
            envs[TENSORPC_FLOW_MASTER_HTTP_PORT] = str(
                prim.get_server_meta().http_port)
        return envs

    @staticmethod
    def _env_modifier(fwd_ports: List[int], rfwd_ports: List[int],
                      env: Dict[str, str]):
        env[TENSORPC_FLOW_MASTER_GRPC_PORT] = str(rfwd_ports[0])
        env[TENSORPC_FLOW_MASTER_HTTP_PORT] = str(rfwd_ports[1])
        env[TENSORPC_FLOW_USE_REMOTE_FWD] = "1"

    async def _cmd_node_callback(self, ev: Event):
        await self._ssh_q.put(ev)

    async def _start_remote_worker(self, graph_id: str, node_id: str):
        node = self.flow_dict[graph_id].get_node_by_id(node_id)
        if isinstance(node, RemoteSSHNode):
            worker_exists: bool = False
            if not node.is_session_started():
                print("START, RemoteSSHNode", graph_id, node_id,
                      node.is_session_started())
                rfports = []
                if node.enable_port_forward:
                    rfports = [prim.get_server_meta().port]
                    if prim.get_server_meta().http_port >= 0:
                        rfports.append(prim.get_server_meta().http_port)

                worker_exists = await node.start_session(
                    self._cmd_node_callback,
                    node.url,
                    node.username,
                    node.password,
                    rfports=rfports)
            await node.run_command()
            if not worker_exists:
                # wait for channel ready, then sync graph
                async with tensorpc.AsyncRemoteManager(node.worker_grpc_url) as robj:
                    await robj.wait_for_remote_ready()
            print("QUERY STATUS!!!")
            # sync graph and query node status.
            driv_nodes = self.flow_dict[graph_id].get_output_nodes_of_handle_type(node, HandleTypes.Driver)
            res = await node.http_remote_call(serv_names.FLOWWORKER_SYNC_GRAPH, 
                graph_id, [n.to_dict() for n in driv_nodes])
            for ev, node in zip(res, driv_nodes):
                le = CommandEventType(ev["last_event"])
                raw_msgs = ev["msgs"]
                await self._msg_q.put(MessageEvent(MessageEventType.Update, raw_msgs))
                # node.last_event = le
                await self._user_ev_q.put(
                    (node.get_uid(),
                        UserStatusEvent(_get_status_from_last_event(le))))

    async def start(self, graph_id: str, node_id: str):
        node = self.flow_dict[graph_id].get_node_by_id(node_id)
        if isinstance(node, RemoteSSHNode):
            return await self._start_remote_worker(graph_id, node_id)
        if isinstance(node, CommandNode):
            print("START", graph_id, node_id, node.commands,
                  node.is_session_started())
            if not node.inputs:
                print("ERRROROROR")
                return
            driver = self._get_node(graph_id, node.get_input_handles(HandleTypes.Driver.value)[0].target_node_id)
            if isinstance(driver, DirectSSHNode):
                if not node.is_session_started():
                    # ssh_data = self.flow_dict[graph_id].ssh_data

                    assert (driver.url != "" and driver.username != ""
                            and driver.password != "")

                    # TODO if graph name changed, session must be restart
                    # we need to find a way to avoid this.
                    envs = self._get_node_envs(graph_id, node_id)
                    # node.start_session(self._ssh_q, ssh_data["url"],
                    #                 ssh_data["username"], ssh_data["password"],
                    #                 envs=envs)
                    rfports = []
                    if driver.enable_port_forward:
                        rfports = [prim.get_server_meta().port]
                        if prim.get_server_meta().http_port >= 0:
                            rfports.append(prim.get_server_meta().http_port)
                    await node.start_session(
                        self._cmd_node_callback,
                        driver.url,
                        driver.username,
                        driver.password,
                        envs=envs,
                        rfports=rfports,
                        env_port_modifier=self._env_modifier)
                    if driver.init_commands != "":
                        await node.input_queue.put(driver.init_commands)
                await node.run_command()
            elif isinstance(driver, RemoteSSHNode):
                assert (driver.url != "" and driver.username != ""
                        and driver.password != "")
                if not driver.is_session_started():
                    print(
                        f"DRIVER {driver.readable_id} not running. run it first."
                    )
                    return
                # if driver started, we can send start msg to
                # this driver
                driver_http_url = driver.worker_http_url
                # we send (may be new) node data to remote worker, then run it.
                # TODO if node deleted, shutdown them in remote worker
                print("START REMOTE")
                await http_remote_call(
                    prim.get_http_client_session(), driver_http_url,
                    serv_names.FLOWWORKER_CREATE_SESSION, node.raw_data,
                    graph_id, driver.url, driver.username, driver.password,
                    driver.remote_init_commands, driver.master_grpc_url)
            else:
                print("ERROROROROR")

    def pause(self, graph_id: str, node_id: str):
        # currently not supported.
        print("PAUSE", graph_id, node_id)

    async def stop(self, graph_id: str, node_id: str):
        print("STOP", graph_id, node_id)
        node = self.flow_dict[graph_id].get_node_by_id(node_id)
        if isinstance(node, RemoteSSHNode):
            # TODO if command nodes driven by this node still running
            # raise error that stop them first.
            if node.is_session_started():
                await node.send_ctrl_c()
            # should we stop tmux session?
            return
        driver_node_id = node.get_input_handles(HandleTypes.Driver.value)[0].target_node_id
        driver = self._get_node(graph_id, driver_node_id)
        if isinstance(driver, RemoteSSHNode):
            if driver.is_session_started():
                driver_http_url = driver.worker_http_url
                await http_remote_call(prim.get_http_client_session(),
                                       driver_http_url,
                                       serv_names.FLOWWORKER_STOP, graph_id,
                                       node_id)
            # TODO raise a exception to front end if driver not start
            return
        if isinstance(node, CommandNode):
            if node.is_session_started():
                await node.send_ctrl_c()

    @marker.mark_exit
    async def _on_exit(self):
        # send exit message to all remote workers
        for g in self.flow_dict.values():
            for n in g.nodes:
                if isinstance(n, RemoteSSHNode):
                    print("?????")
                    # send close-connection message to remote worker
                    if n.worker_http_port >= 0:
                        await n.shutdown()