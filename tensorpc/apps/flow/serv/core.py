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
from typing import Any, Dict, Optional, Tuple 
import json 
from tensorpc import marker, prim
from tensorpc.apps.flow.constants import FLOW_FOLDER_PATH, FLOW_DEFAULT_GRAPH_NAME
import enum 

class Node:
    def __init__(self, flow_data: Dict[str, Any]) -> None:
        self._flow_data = flow_data
        self.id = flow_data["id"]

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
    def raw_data(self) -> Dict[str, Any]:
        return self._flow_data


class CommandNode(Node):
    def __init__(self, flow_data: Dict[str, Any]) -> None:
        super().__init__(flow_data)
        self.session = None


class Edge:
    def __init__(self, flow_data: Dict[str, Any]) -> None:
        self._flow_data = flow_data
        self.id = flow_data["id"]

    @property 
    def raw_data(self) -> Dict[str, Any]:
        return self._flow_data

class FlowGraph:
    def __init__(self, graph_data: Dict[str, Any]) -> None:
        self.nodes = [Node(d) for d in graph_data["nodes"]] 
        self.edges = [Edge(d) for d in graph_data["edges"]]  
        self.viewport = graph_data["viewport"]

    def to_dict(self):
        return {
            "viewport": self.viewport,
            "nodes": [n.raw_data for n in self.nodes],
            "edges": [n.raw_data for n in self.edges],
        }

def _empty_flow_graph():
    return FlowGraph({
        "nodes": [],
        "edges": [],
        "viewport": {
            "x": 0,
            "y": 0,
            "zoom": 1,
        }
    })

class Flow:
    def __init__(self, root: Optional[str] = None) -> None:
        self._q = asyncio.Queue()
        if root is None or root == "":
            root = str(FLOW_FOLDER_PATH)
        self.root = Path(root)
        if not self.root.exists():
            self.root.mkdir(0o755, True, True)
        self.default_graph_path = self.root / f"{FLOW_DEFAULT_GRAPH_NAME}.json"
        if not self.default_graph_path.exists():
            with self.default_graph_path.open("w") as f:
                json.dump(_empty_flow_graph().to_dict(), f)
        self.graphs_dict: Dict[str, Any] = {}
        for flow_path in self.root.glob("*.json"):
            with flow_path.open("r") as f:
                graph = json.load(f)
            self.graphs_dict[flow_path.stem] = graph

    @marker.mark_websocket_event
    async def node_status_change(self):
        # ws client wait for this event to get new node update msg
        (name, content) = await self._q.get()
        return prim.DynamicEvent(name, content) 

    def update_node_status(self, name: str, content: Any):
        # user client call this rpc to send message to frontend.
        loop = asyncio.get_running_loop()
        asyncio.run_coroutine_threadsafe(self._q.put((name, content)), loop)

    def query_node_stdout(self, name: str):
        return f"Hello World {name}!!!\\n"

    def save_graph(self, graph_name: str, graph):
        self.graphs_dict[graph_name] = FlowGraph(graph)
        flow_path = self.root / f"{graph_name}.json"
        with flow_path.open("w") as f:
            json.dump(graph, f)

    def load_default_graph(self):
        return self.load_graph(FLOW_DEFAULT_GRAPH_NAME)

    def load_graph(self, graph_name: str):
        flow_path = self.root / f"{graph_name}.json"
        with flow_path.open("r") as f:
            graph = json.load(f)
        self.graphs_dict[graph_name] = FlowGraph(graph)
        return graph

    def start(self, name: str):
        print("START", name)

    def pause(self, name: str):
        print("PAUSE", name)

    def stop(self, name: str):
        print("STOP", name)