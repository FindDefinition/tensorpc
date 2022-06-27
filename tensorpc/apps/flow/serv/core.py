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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

from tensorpc import marker, prim
from tensorpc.apps.flow.constants import (
    FLOW_DEFAULT_GRAPH_ID, FLOW_FOLDER_PATH, TENSORPC_FLOW_GRAPH_ID,
    TENSORPC_FLOW_MASTER_GRPC_PORT, TENSORPC_FLOW_MASTER_HTTP_PORT,
    TENSORPC_FLOW_NODE_ID, TENSORPC_FLOW_NODE_UID, TENSORPC_FLOW_USE_REMOTE_FWD)
from tensorpc.autossh.core import (CommandEvent, CommandEventType, EofEvent,
                                   Event, ExceptionEvent, LineEvent, SSHClient)
from tensorpc.core.asynctools import cancel_task


def _extract_graph_node_id(uid: str):
    parts = uid.split("@")
    return parts[0], parts[1]


def _get_uid(graph_id: str, node_id: str):
    return f"{graph_id}@{node_id}"


class Node:
    def __init__(self, flow_data: Dict[str, Any], graph_id: str = "") -> None:
        self._flow_data = flow_data
        self.id: str = flow_data["id"]
        self.graph_id: str = graph_id
        self.inputs: List[str] = []
        self.outputs: List[str] = []

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

    def get_uid(self):
        return _get_uid(self.graph_id, self.id)

    async def shutdown(self):
        return

    def clear_connections(self):
        self.inputs.clear()
        self.outputs.clear()


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
    def enable_remote_forward(self) -> bool:
        return self.node_data["enableRemoteForward"]


class EnvNode(Node):
    pass


class CommandNode(Node):
    def __init__(self, flow_data: Dict[str, Any], graph_id: str = "") -> None:
        super().__init__(flow_data, graph_id)
        # print(json.dumps(flow_data, indent=2))
        self.task: Optional[asyncio.Task] = None
        self.shutdown_ev = asyncio.Event()
        self.input_queue = asyncio.Queue()
        self.stdout = ""

    @property
    def commands(self):
        args = self.node_data["args"]
        return [x["value"] for x in filter(lambda x: x["enabled"], args)]

    async def shutdown(self):
        print("NODE", self.id, "SHUTDOWN")
        if self.task is not None:
            self.shutdown_ev.set()
            await cancel_task(self.task)
            self.task = None
            self.shutdown_ev.clear()

    async def run_command(self):
        await self.input_queue.put(" ".join(self.commands) + "\n")

    async def send_ctrl_c(self):
        # https://github.com/ronf/asyncssh/issues/112#issuecomment-343318916
        return await self.input_queue.put("\x03")

    def start_session(self,
                      msg_q: asyncio.Queue,
                      url: str,
                      username: str,
                      password: str,
                      envs: Dict[str, str],
                      rfports: Optional[List[int]] = None,
                      env_port_modifier: Optional[Callable[[List[int], Dict[str, str]], None]] = None):
        assert self.task is None
        client = SSHClient(url, username, password, None, self.get_uid())

        async def callback(ev: Event):
            await msg_q.put(ev)

        sd_task = asyncio.create_task(self.shutdown_ev.wait())
        self.task = asyncio.create_task(
            client.connect_queue(self.input_queue,
                                 callback,
                                 sd_task,
                                 env=envs,
                                 r_forward_ports=rfports,
                                 env_port_modifier=env_port_modifier))

    def is_started(self):
        return self.task is not None


_TYPE_TO_NODE_CLS: Dict[str, Type[Node]] = {
    "command": CommandNode,
    "env": EnvNode,
    "directssh": DirectSSHNode,
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

        self.graph_id = graph_id
        self.ssh_data = flow_data["ssh"]

    def _update_connection(self, edges: List[Edge]):
        for k, v in self._node_id_to_node.items():
            v.clear_connections()
        for edge in edges:
            source = edge.source_id
            target = edge.target_id
            self._node_id_to_node[source].outputs.append(target)
            self._node_id_to_node[target].inputs.append(source)

    def update_nodes(self, nodes: Iterable[Node]):
        self._node_id_to_node = {n.id: n for n in nodes}
        self._node_rid_to_node = {n.readable_id: n for n in nodes}

    def get_node_by_id(self, node_id: str):
        if node_id in self._node_id_to_node:
            return self._node_id_to_node[node_id]
        else:
            return self._node_rid_to_node[node_id]

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
                # shutdown this node
                await node.shutdown()
        self.update_nodes(new_node_id_to_node.values())
        # we assume edges don't contain any state, so just update them.
        # we may need to handle this in future.
        self._edge_id_to_edge = {n.id: n for n in edges}
        self._update_connection(edges)
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
        self._q = asyncio.Queue()
        self._ssh_q: "asyncio.Queue[Event]" = asyncio.Queue()
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

    @marker.mark_websocket_event
    async def node_status_change(self):
        # ws client wait for this event to get new node update msg
        (uid, content) = await self._q.get()
        return prim.DynamicEvent(uid, content)

    @marker.mark_websocket_event
    async def command_node_event(self):
        # uid: {graph_id}@{node_id}
        while True:
            event = await self._ssh_q.get()
            # print(event)
            uid = event.uid
            graph_id, node_id = _extract_graph_node_id(uid)
            node = self._get_node(graph_id, node_id)
            assert isinstance(node, CommandNode)
            if isinstance(event, LineEvent):
                # print(node.id, self.selected_node_uid == uid, event.line, end="")
                node.stdout += event.line
                if uid != self.selected_node_uid:
                    continue
            elif isinstance(event, CommandEvent):
                if event.type == CommandEventType.PROMPT_END:
                    node.stdout += str(event.arg)

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
        asyncio.run_coroutine_threadsafe(self._q.put((uid, content)), loop)

    def select_node(self, graph_id: str, node_id: str):
        node = self._get_node(graph_id, node_id)
        assert isinstance(node, CommandNode)
        self.selected_node_uid = node.get_uid()
        # print("STDOUT", node.stdout, node.id)
        return node.stdout

    async def command_node_input(self, graph_id: str, node_id: str, data: str):
        node = self._get_node(graph_id, node_id)
        if (isinstance(node, CommandNode)):
            if node.is_started():
                await node.input_queue.put(data)

    def remove_node(self):
        self.selected_node_uid = ""

    async def save_graph(self, graph_id: str, flow_data):
        # TODO do we need a async lock here?
        ssh_data = flow_data["ssh"]
        flow_data = flow_data["graph"]
        flow_data["ssh"] = ssh_data
        flow_data["id"] = graph_id
        if graph_id in self.flow_dict:
            await self.flow_dict[graph_id].update_graph(graph_id, flow_data)
        else:
            self.flow_dict[graph_id] = FlowGraph(flow_data, graph_id)
        for n in flow_data["nodes"]:
            n["selected"] = False
        flow_path = self.root / f"{graph_id}.json"
        with flow_path.open("w") as f:
            json.dump(flow_data, f)

    def load_default_graph(self):
        return self.load_graph(FLOW_DEFAULT_GRAPH_ID, force_reload=False)

    def load_graph(self, graph_id: str, force_reload: bool = True):
        flow_path = self.root / f"{graph_id}.json"
        with flow_path.open("r") as f:
            flow_data = json.load(f)
        for n in flow_data["nodes"]:
            n["selected"] = False
            n.pop("handleBounds")
        if force_reload:
            reload = True
        else:
            reload = graph_id not in self.flow_dict
        if reload:
            self.flow_dict[graph_id] = FlowGraph(flow_data, graph_id)
        return flow_data

    def _get_node_envs(self, graph_id: str, node_id: str):
        node = self.flow_dict[graph_id].get_node_by_id(node_id)
        envs: Dict[str, str] = {}
        if isinstance(node, CommandNode):
            envs[TENSORPC_FLOW_GRAPH_ID] = graph_id
            envs[TENSORPC_FLOW_NODE_ID] = node_id
            envs[TENSORPC_FLOW_NODE_UID] = node.get_uid()
            envs[TENSORPC_FLOW_MASTER_GRPC_PORT] = str(
                prim.get_server_meta().port)
            envs[TENSORPC_FLOW_MASTER_HTTP_PORT] = str(
                prim.get_server_meta().http_port)
        return envs

    @staticmethod
    def _env_modifier(fwd_ports: List[int], env: Dict[str, str]):
        env[TENSORPC_FLOW_MASTER_GRPC_PORT] = str(
            fwd_ports[0])
        env[TENSORPC_FLOW_MASTER_HTTP_PORT] = str(
            fwd_ports[1])
        env[TENSORPC_FLOW_USE_REMOTE_FWD] = "1"

    async def start(self, graph_id: str, node_id: str):
        node = self.flow_dict[graph_id].get_node_by_id(node_id)
        if isinstance(node, CommandNode):
            print("START", graph_id, node_id, node.commands, node.is_started())
            if not node.is_started():
                if not node.inputs:
                    print("ERRROROROR")
                    return
                driver = self._get_node(graph_id, node.inputs[0])
                if isinstance(driver, DirectSSHNode):
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
                    if driver.enable_remote_forward:
                        rfports = [prim.get_server_meta().port]
                        if prim.get_server_meta().http_port >= 0:
                            rfports.append(prim.get_server_meta().http_port)
                    node.start_session(self._ssh_q,
                                       driver.url,
                                       driver.username,
                                       driver.password,
                                       envs=envs,
                                       rfports=rfports,
                                       env_port_modifier=self._env_modifier)
                else:
                    print("ERROROROROR")
            await node.run_command()

    def pause(self, graph_id: str, node_id: str):
        print("PAUSE", graph_id, node_id)

    async def stop(self, graph_id: str, node_id: str):
        print("STOP", graph_id, node_id)
        node = self.flow_dict[graph_id].get_node_by_id(node_id)
        if isinstance(node, CommandNode):
            if node.is_started():
                await node.send_ctrl_c()
