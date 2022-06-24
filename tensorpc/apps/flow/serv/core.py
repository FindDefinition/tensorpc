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

from pathlib import Path
import asyncio
from typing import Any, Dict, Iterable, Optional, Tuple, Type
import json
from tensorpc import marker, prim
from tensorpc.apps.flow.constants import FLOW_FOLDER_PATH, FLOW_DEFAULT_GRAPH_ID
import enum
from tensorpc.autossh.core import EofEvent, ExceptionEvent, SSHClient, Event, LineEvent, CommandEvent, CommandEventType
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

    def start_session(self, msg_q: asyncio.Queue, url: str, username: str,
                      password: str):
        assert self.task is None
        client = SSHClient(url, username, password, None, self.get_uid())

        async def callback(ev: Event):
            await msg_q.put(ev)

        sd_task = asyncio.create_task(self.shutdown_ev.wait())
        self.task = asyncio.create_task(
            client.connect_queue(self.input_queue, callback, sd_task))

    def is_started(self):
        return self.task is not None


_TYPE_TO_NODE_CLS: Dict[str, Type[Node]] = {
    "command": CommandNode,
    "env": Node,
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

        self.graph_id = graph_id
        self.ssh_data = flow_data["ssh"]

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
        self._ssh_q: "asyncio.Queue[Tuple[str, Event]]" = asyncio.Queue()
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
            (uid, event) = await self._ssh_q.get()
            print(uid, event)
            graph_id, node_id = _extract_graph_node_id(uid)
            node = self._get_node(graph_id, node_id)
            assert isinstance(node, CommandNode)
            if isinstance(event, LineEvent):
                print(node.id, event.line, end="")
                node.stdout += event.line
                if uid != self.selected_node_uid:
                    continue
            elif isinstance(event, (EofEvent, ExceptionEvent)):
                await node.shutdown()
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
        print("STDOUT", node.stdout, node.id)
        return node.stdout

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
        flow_path = self.root / f"{graph_id}.json"
        with flow_path.open("w") as f:
            json.dump(flow_data, f)

    def load_default_graph(self):
        return self.load_graph(FLOW_DEFAULT_GRAPH_ID)

    def load_graph(self, graph_id: str):
        flow_path = self.root / f"{graph_id}.json"
        with flow_path.open("r") as f:
            flow_data = json.load(f)
        self.flow_dict[graph_id] = FlowGraph(flow_data, graph_id)
        return flow_data

    async def start(self, graph_id: str, node_id: str):
        print("START", graph_id, node_id)
        node = self.flow_dict[graph_id].get_node_by_id(node_id)
        if isinstance(node, CommandNode):
            if not node.is_started():
                ssh_data = self.flow_dict[graph_id].ssh_data
                assert ssh_data["url"] != ""
                node.start_session(self._ssh_q, ssh_data["url"],
                                   ssh_data["username"], ssh_data["password"])
            await node.run_command()

    def pause(self, graph_id: str, node_id: str):
        print("PAUSE", graph_id, node_id)

    async def stop(self, graph_id: str, node_id: str):
        print("STOP", graph_id, node_id)
        node = self.flow_dict[graph_id].get_node_by_id(node_id)
        if isinstance(node, CommandNode):
            if node.is_started():
                await node.send_ctrl_c()

