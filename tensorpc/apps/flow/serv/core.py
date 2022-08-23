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
import base64
import bisect
import enum
import itertools
import json
import time
import traceback
import uuid
from collections import deque
from functools import partial
from pathlib import Path
from typing import (Any, Awaitable, Callable, Coroutine, Dict, Iterable, List,
                    Optional, Set, Tuple, Type, Union)

import aiohttp
import asyncssh
import bcrypt
import tensorpc
from tensorpc.constants import TENSORPC_SPLIT
from tensorpc import get_http_url, http_remote_call, marker, prim
from tensorpc.apps.flow.constants import (
    FLOW_DEFAULT_GRAPH_ID, FLOW_FOLDER_PATH, TENSORPC_FLOW_DEFAULT_TMUX_NAME,
    TENSORPC_FLOW_GRAPH_ID, TENSORPC_FLOW_MASTER_GRPC_PORT,
    TENSORPC_FLOW_MASTER_HTTP_PORT, TENSORPC_FLOW_NODE_ID,
    TENSORPC_FLOW_NODE_READABLE_ID, TENSORPC_FLOW_NODE_UID,
    TENSORPC_FLOW_USE_REMOTE_FWD)
from tensorpc.apps.flow.coretypes import (Message, MessageEvent,
                                          MessageEventType, MessageLevel,
                                          ScheduleEvent, SessionStatus,
                                          UserContentEvent, UserEvent,
                                          UserStatusEvent, get_uid)
from tensorpc.apps.flow.flowapp.core import AppEvent, AppEventType, NotifyEvent, NotifyType, ScheduleNextForApp, UISaveStateEvent, app_event_from_data
from tensorpc.apps.flow.serv_names import serv_names
from tensorpc.autossh.core import (CommandEvent, CommandEventType, EofEvent,
                                   Event, ExceptionEvent, LineEvent, RawEvent,
                                   SSHClient, SSHRequest, SSHRequestType,
                                   remove_ansi_seq)
from tensorpc.core import get_grpc_url
from tensorpc.core.asynctools import cancel_task
from tensorpc.utils.address import get_url_port
from tensorpc.utils.registry import HashableRegistry
from tensorpc.utils.wait_tools import get_free_ports
from jinja2 import BaseLoader, Environment, Template

JINJA2_VARIABLE_ENV = Environment(loader=BaseLoader,
                                  variable_start_string="{{",
                                  variable_end_string="}}")

ALL_NODES = HashableRegistry()


class HandleTypes(enum.Enum):
    Driver = "driver"
    Input = "inputs"
    Output = "outputs"


class NodeStatus:

    def __init__(self, cmd_status: CommandEventType,
                 session_status: SessionStatus) -> None:
        self.cmd_status = cmd_status
        self.session_status = session_status

    def to_dict(self):
        return {
            "cmdStatus": self.cmd_status.value,
            "sessionStatus": self.session_status.value,
        }

    @staticmethod
    def empty():
        return NodeStatus(CommandEventType.PROMPT_END, SessionStatus.Stop)

ENCODING = "utf-8"
ENCODING = None

def _extract_graph_node_id(uid: str):
    parts = uid.split("@")
    return parts[0], parts[1]


def _get_uid(graph_id: str, node_id: str):
    return get_uid(graph_id, node_id)


def _get_status_from_last_event(ev: CommandEventType):
    if ev == CommandEventType.CURRENT_COMMAND:
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
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data["target_node_id"], data["type"], data["edge_id"])


@ALL_NODES.register
class Node:

    def __init__(self,
                 flow_data: Dict[str, Any],
                 graph_id: str = "",
                 schedulable: bool = False) -> None:
        self._flow_data = flow_data
        self.id: str = flow_data["id"]
        self.graph_id: str = graph_id
        self.inputs: Dict[str, List[Handle]] = {}
        self.outputs: Dict[str, List[Handle]] = {}
        # self.remote_driver_id: str = ""
        self.messages: Dict[str, Message] = {}

        self._schedulable = schedulable

    @property
    def schedulable(self):
        return self._schedulable

    # @property
    # def is_remote(self):
    #     return self.remote_driver_id != ""

    def schedule_next(self, ev: ScheduleEvent,
                      graph: "FlowGraph") -> Dict[str, ScheduleEvent]:
        return {}

    def to_dict(self):
        # currently this method is only used for remote worker.
        raw = self._flow_data.copy()
        raw.update({
            "tensorpc_flow": {
                "graph_id": self.graph_id,
                "type": type(self).__name__,
                # "remote_driver_id": self.remote_driver_id,
                "inputs": {
                    n: [vv.to_dict() for vv in v]
                    for n, v in self.inputs.items()
                },
                "outputs": {
                    n: [vv.to_dict() for vv in v]
                    for n, v in self.outputs.items()
                },
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
    def from_dict(cls, data: Dict[str, Any]):
        extra_data = data["tensorpc_flow"]
        node = cls(data, extra_data["graph_id"])
        # node.remote_driver_id = extra_data['remote_driver_id']
        node.inputs = {
            n: [Handle.from_dict(vv) for vv in v]
            for n, v in extra_data["inputs"].items()
        }
        node.outputs = {
            n: [Handle.from_dict(vv) for vv in v]
            for n, v in extra_data["outputs"].items()
        }
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
            # self.remote_driver_id = extra_data['remote_driver_id']
            self.inputs = {
                n: [Handle.from_dict(vv) for vv in v]
                for n, v in extra_data["inputs"].items()
            }
            self.outputs = {
                n: [Handle.from_dict(vv) for vv in v]
                for n, v in extra_data["outputs"].items()
            }

    def get_uid(self):
        return _get_uid(self.graph_id, self.id)

    def get_readable_uid(self):
        return _get_uid(self.graph_id, self.readable_id)

    async def shutdown(self):
        return

    def clear_connections(self):
        self.inputs.clear()
        self.outputs.clear()

    @property 
    def driver_id(self) -> str:
        return self._flow_data["data"].get("driver", "")

class RunnableNodeBase(Node):
    pass
    

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
        cmds = self.node_data["initCommands"].strip()
        return cmds


class NodeWithSSHBase(RunnableNodeBase):

    def __init__(self,
                 flow_data: Dict[str, Any],
                 graph_id: str = "",
                 schedulable: bool = False) -> None:
        super().__init__(flow_data, graph_id, schedulable)
        self.shutdown_ev = asyncio.Event()
        self.task: Optional[asyncio.Task] = None
        self.input_queue = asyncio.Queue()
        self.last_event: CommandEventType = CommandEventType.PROMPT_END
        if ENCODING is None:
            self.stdout = b""
        else:
            self.stdout = ""

        self.init_terminal_size: Tuple[int, int] = (34, 16)
        if ENCODING is None:
            self.terminal_state = b""
        else:
            self.terminal_state = ""

        self.terminal_close_ts: int = -1
        self._raw_event_history: "deque[RawEvent]" = deque(maxlen=10000)
        self.session_status: SessionStatus = SessionStatus.Stop

        self.exit_event = asyncio.Event()

        self.queued_commands: List[ScheduleEvent] = []
        self.running_driver_id = ""

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
            self.set_stop_status()
            self.shutdown_ev.clear()

    async def soft_shutdown(self):
        """only set shutdown event.
        ssh client will produce a ExitEvent to tell
        frontend node is stopped.
        """
        self.shutdown_ev.set()

    def set_start_status(self):
        self.session_status = SessionStatus.Running

    def set_stop_status(self):
        self.session_status = SessionStatus.Stop

    def is_session_started(self):
        return self.session_status == SessionStatus.Running

    def is_running(self):
        return self.last_event != CommandEventType.PROMPT_END

    def get_session_status(self):
        if self.is_session_started():
            return SessionStatus.Running
        else:
            return SessionStatus.Stop

    def get_node_status(self):
        return UserStatusEvent(_get_status_from_last_event(self.last_event),
                               self.get_session_status())

    @staticmethod
    def _env_port_modifier(fwd_ports: List[int], rfwd_ports: List[int],
                           env: Dict[str, str]):
        if (len(rfwd_ports) > 0):
            env[TENSORPC_FLOW_MASTER_GRPC_PORT] = str(rfwd_ports[0])
            env[TENSORPC_FLOW_MASTER_HTTP_PORT] = str(rfwd_ports[1])
        env[TENSORPC_FLOW_USE_REMOTE_FWD] = "1"


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

    def to_dict(self):
        res = super().to_dict()
        res["worker_port"] = self.worker_port 
        res["worker_http_port"] = self.worker_http_port 
        res["remote_master_port"] = self.remote_master_port 
        res["remote_master_http_port"] = self.remote_master_http_port 
        res["remote_port"] = self.remote_port 
        res["remote_http_port"] = self.remote_http_port 
        res["_remote_self_ip"] = self._remote_self_ip 
        return res 

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        extra_data = data["tensorpc_flow"]
        node = cls(data, extra_data["graph_id"])
        # node.remote_driver_id = extra_data['remote_driver_id']
        node.inputs = {
            n: [Handle.from_dict(vv) for vv in v]
            for n, v in extra_data["inputs"].items()
        }
        node.outputs = {
            n: [Handle.from_dict(vv) for vv in v]
            for n, v in extra_data["outputs"].items()
        }
        node.worker_port = data["worker_port"]
        node.worker_http_port = data["worker_http_port"]
        node.remote_master_port = data["remote_master_port"]
        node.remote_master_http_port = data["remote_master_http_port"]
        node.remote_port = data["remote_port"]
        node.remote_http_port = data["remote_http_port"]
        node._remote_self_ip = data["_remote_self_ip"]
        return node

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
        cmds = self.node_data["initCommands"].strip()
        return cmds

    @property
    def remote_init_commands(self) -> str:
        cmds = self.node_data["remoteInitCommands"].strip()
        return cmds

    @property
    def rfport_pairs(self) -> str:
        ports_str = self.node_data.get("remoteForwardPorts", "").strip()
        return ports_str

    def _env_port_modifier(self, fports: List[int], rfports: List[int],
                           env: Dict[str, str]):
        if fports:
            self.worker_port = fports[0]
            self.worker_http_port = fports[1]
        self.remote_master_port = rfports[0]
        if len(rfports) > 1:
            self.remote_master_http_port = rfports[1]
        super()._env_port_modifier(fports, rfports, env)

    async def start_session(self,
                            callback: Callable[[Event], Awaitable[None]],
                            url: str,
                            username: str,
                            password: str,
                            rfports: Optional[List[int]] = None):
        assert self.task is None
        new_sess_name = TENSORPC_FLOW_DEFAULT_TMUX_NAME
        client = SSHClient(url, username, password, None, self.get_uid(), ENCODING)
        self.shutdown_ev.clear()
        self.exit_event.clear()
        # firstly we check if the tmux worker exists
        # if exists, we need to query some node state from remote.
        worker_exists: bool = False
        init_event = asyncio.Event()
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
        async def exit_callback():
            self.task = None
            self.set_stop_status()
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
            # try:
            #     await http_remote_call(prim.get_http_client_session(),
            #                            http_url,
            #                            serv_names.FLOWWORKER_CLOSE_CONNECTION,
            #                            self.graph_id, self.id)
            # except:
            #     traceback.print_exc()

        def client_ip_callback(cip: str):
            self._remote_self_ip = cip.strip()

        sd_task = asyncio.create_task(self.shutdown_ev.wait())
        forward_ports = []
        if self.enable_port_forward:
            forward_ports = [self.remote_port, self.remote_http_port]
        new_rfports: Optional[List[Union[Tuple[int, int], int]]] = None
        if rfports is not None:
            new_rfports = [*rfports]

            ports_str = self.rfport_pairs
            if ports_str:
                pair_strs = ports_str.split(",")
                pair_ports = [(int(p.split(":")[0]), int(p.split(":")[1])) for p in pair_strs]
                new_rfports.extend(pair_ports)
        self.task = asyncio.create_task(
            client.connect_queue(self.input_queue,
                                 callback,
                                 sd_task,
                                 forward_ports=forward_ports,
                                 r_forward_ports=new_rfports,
                                 env_port_modifier=self._env_port_modifier,
                                 exit_callback=exit_callback,
                                 client_ip_callback=client_ip_callback,
                                 init_event=init_event,
                                 exit_event=self.exit_event))
        self.set_start_status()

        await self.input_queue.put(
            SSHRequest(SSHRequestType.ChangeSize, self.init_terminal_size))
        return worker_exists, init_event

    async def run_command(self,
                          newenvs: Optional[Dict[str, Any]] = None,
                          cmd_renderer: Optional[Callable[[str], str]] = None):
        init_cmd = self.init_commands
        if cmd_renderer is not None and init_cmd:
            init_cmd = cmd_renderer(init_cmd)
        if init_cmd:
            await self.input_queue.put(init_cmd + "\n")
        await self.input_queue.put((
            f"python -m tensorpc.cli.start_worker --name={TENSORPC_FLOW_DEFAULT_TMUX_NAME} "
            f"--port={self.remote_port} "
            f"--http_port={self.remote_http_port} && while :; do sleep 2073600; done\n"
            # f"--http_port={self.remote_http_port}\n"
        ))


@ALL_NODES.register
class EnvNode(Node):

    @property
    def key(self):
        return self.node_data["key"]

    @property
    def value(self):
        return self.node_data["value"]

@ALL_NODES.register
class DataStorageNode(Node):
    def __init__(self, flow_data: Dict[str, Any], graph_id: str = "") -> None:
        super().__init__(flow_data, graph_id, True)

        self.stored_data: Optional[Any] = None 

    @property
    def full(self):
        return self.stored_data is not None


class CommandResult:
    def __init__(self, cmd: str, stdouts: List[bytes], return_code: int) -> None:
        self.cmd = cmd 
        self.stdouts = stdouts
        self.return_code = return_code


    def to_dict(self):
        return {
            "cmd": self.cmd ,
            "stdouts": self.stdouts,
            "return_code": self.return_code,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data["cmd"], data["stdouts"], data["return_code"])
        

@ALL_NODES.register
class CommandNode(NodeWithSSHBase):

    def __init__(self, flow_data: Dict[str, Any], graph_id: str = "") -> None:
        super().__init__(flow_data, graph_id, schedulable=True)
        self._start_record_stdout: bool = False
        self._previous_cmd = ""
        self._previous_ret_code = -1
        self._current_line_events: List[LineEvent] = []

    def push_command(self, ev: LineEvent):
        if self._start_record_stdout:
            self._current_line_events.append(ev)

    def get_previous_cmd_result(self):
        return CommandResult(self._previous_cmd, [l.line for l in self._current_line_events], self._previous_ret_code)

    def clear_previous_cmd(self):
        self._start_record_stdout = False 
        self._previous_cmd = ""
        self._previous_ret_code = -1
        self._current_line_events.clear()

    @property
    def commands(self):
        args = self.node_data["args"]
        return [x["value"] for x in filter(lambda x: x["enabled"], args)]

    async def run_command(self,
                          newenvs: Optional[Dict[str, Any]] = None,
                          cmd_renderer: Optional[Callable[[str], str]] = None):
        cmd = " ".join(self.commands)
        if cmd_renderer:
            cmd = cmd_renderer(cmd)
        # self._start_record_stdout = True
        if newenvs:
            envs_stmt = [f"export {k}={v}" for k, v in newenvs.items()]
            cmd = " && ".join(envs_stmt +
                                                   [cmd])
        self._previous_cmd = cmd
        await self.input_queue.put(cmd + "\n")

    # async def push_new_envs(self, envs: Dict[str, Any]):
    #     envs_stmt = [f"export {k}={v}" for k, v in envs.items()]
    #     await self.input_queue.put(" && ".join(envs_stmt) + "\n")

    def schedule_next(self, ev: ScheduleEvent,
                      graph: "FlowGraph") -> Dict[str, ScheduleEvent]:
        next_nodes = graph.get_output_nodes_of_handle_type(
            self, HandleTypes.Output)
        # print("NEXT NODES", next_nodes, self.outputs, self.inputs)
        res: Dict[str, ScheduleEvent] = {}
        for n in next_nodes:
            if n.schedulable:
                res[n.id] = ev
        return res

    async def run_schedule_event(self, sche_ev: ScheduleEvent):
        await self.run_command(sche_ev.envs)

    async def start_session(self,
                            callback: Callable[[Event], Awaitable[None]],
                            url: str,
                            username: str,
                            password: str,
                            envs: Dict[str, str],
                            is_worker: bool,
                            enable_port_forward: bool,
                            rfports: Optional[List[Union[int, Tuple[int, int]]]] = None,
                            init_cmds: str = "",
                            running_driver_id: str = ""):
        assert self.task is None
        init_event = asyncio.Event()
        self.shutdown_ev.clear()
        self.exit_event.clear()
        client = SSHClient(url, username, password, None, self.get_uid(), ENCODING)

        # async def callback(ev: Event):
        #     await msg_q.put(ev)
        self.running_driver_id = running_driver_id
        async def exit_callback():
            self.task = None
            self.last_event = CommandEventType.PROMPT_END
            self.set_stop_status()
            self.running_driver_id = ""

        sd_task = asyncio.create_task(self.shutdown_ev.wait())
        self.task = asyncio.create_task(
            client.connect_queue(self.input_queue,
                                 callback,
                                 sd_task,
                                 env=envs,
                                 r_forward_ports=rfports,
                                 env_port_modifier=self._env_port_modifier,
                                 exit_callback=exit_callback,
                                 init_event=init_event,
                                 exit_event=self.exit_event))
        self.set_start_status()

        await self.input_queue.put(
            SSHRequest(SSHRequestType.ChangeSize, self.init_terminal_size))
        return True, init_event


@ALL_NODES.register
class AppNode(CommandNode):

    def __init__(self, flow_data: Dict[str, Any], graph_id: str = "") -> None:
        super().__init__(flow_data, graph_id)
        self.grpc_port = -1
        self.http_port = -1
        self.fwd_grpc_port = -1
        self.fwd_http_port = -1

        self.state: Optional[dict] = None

    @property
    def module_name(self):
        return self.node_data["module"]

    @property
    def init_config(self):
        if self.node_data["initConfig"] == "":
            return {}
        return json.loads(self.node_data["initConfig"])

    def _app_env_port_modifier(self, fports: List[int], rfports: List[int],
                               env: Dict[str, str]):
        if fports:
            self.fwd_grpc_port = fports[0]
            self.fwd_http_port = fports[1]
        super()._env_port_modifier(fports, rfports, env)

    async def start_session(self,
                            callback: Callable[[Event], Awaitable[None]],
                            url: str,
                            username: str,
                            password: str,
                            envs: Dict[str, str],
                            is_worker: bool,
                            enable_port_forward: bool,
                            rfports: Optional[List[Union[int, Tuple[int, int]]]] = None,
                            init_cmds: str = "",
                            running_driver_id: str = ""):
        assert self.task is None
        init_event = asyncio.Event()
        self.shutdown_ev.clear()
        self.exit_event.clear()
        client = SSHClient(url, username, password, None, self.get_uid(), ENCODING)
        # print("APP", url, client.url_no_port, client.port)
        if not is_worker:
            # query two free port in target via ssh, then use them as app ports
            ports = await _get_free_port(2, url, username, password, init_cmds)
        else:
            # query two local ports in flow remote worker, then use them as app ports
            ports = get_free_ports(2)
        # print("APP PORTS", ports)
        if len(ports) != 2:
            raise ValueError("get free port failed. exit.")

        self.grpc_port = ports[0]
        self.http_port = ports[1]
        fwd_ports = []
        self.fwd_grpc_port = self.grpc_port
        self.fwd_http_port = self.http_port
        if enable_port_forward:
            fwd_ports = ports
        # async def callback(ev: Event):
        #     await msg_q.put(ev)
        self.running_driver_id = running_driver_id
        async def exit_callback():
            self.task = None
            self.last_event = CommandEventType.PROMPT_END
            self.set_stop_status()
            self.running_driver_id = ""

        sd_task = asyncio.create_task(self.shutdown_ev.wait())
        self.task = asyncio.create_task(
            client.connect_queue(self.input_queue,
                                 callback,
                                 sd_task,
                                 env=envs,
                                 forward_ports=fwd_ports,
                                 r_forward_ports=rfports,
                                 env_port_modifier=self._app_env_port_modifier,
                                 exit_callback=exit_callback,
                                 init_event=init_event,
                                 exit_event=self.exit_event))
        self.set_start_status()
        await self.input_queue.put(
            SSHRequest(SSHRequestType.ChangeSize, self.init_terminal_size))
        return True, init_event

    async def run_command(self,
                          newenvs: Optional[Dict[str, Any]] = None,
                          cmd_renderer: Optional[Callable[[str], str]] = None):
        serv_name = f"tensorpc.apps.flow.serv.flowapp{TENSORPC_SPLIT}FlowApp"
        cfg = {
            serv_name: {
                "module_name": self.module_name,
                "config": self.init_config,
            }
        }
        cfg_encoded = base64.b64encode(
            json.dumps(cfg).encode("utf-8")).decode("utf-8")
        # TODO only use http port
        cmd = (f"python -m tensorpc.serve {serv_name} "
               f"--port={self.grpc_port} --http_port={self.http_port} "
               f"--serv_config_b64 {cfg_encoded}")
        await self.input_queue.put(cmd + "\n")


_TYPE_TO_NODE_CLS: Dict[str, Type[Node]] = {
    "command": CommandNode,
    "env": EnvNode,
    "directssh": DirectSSHNode,
    "input": Node,
    "remotessh": RemoteSSHNode,
    "app": AppNode,
    "datastorage": DataStorageNode,
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

        self.messages: Dict[str, Message] = {}

        # in-memory per-graph salt that will be
        # created when user firstly login devflow frontend.
        # passwords will be encrypted during save-graph
        # and saved.
        self.salt = ""

        self.variable_dict = self._get_jinja_variable_dict(nodes)

    def render_command(self, cmd: str):
        template = JINJA2_VARIABLE_ENV.from_string(cmd)
        return template.render(**self.variable_dict)

    def _get_jinja_variable_dict(self, nodes: List[Node]):
        variable_dict: Dict[str, str] = {}
        for n in nodes:
            if isinstance(n, EnvNode):
                variable_dict[n.key] = n.value
        return variable_dict

    def _update_connection(self, edges: List[Edge]):
        for k, v in self._node_id_to_node.items():
            v.clear_connections()
        for edge in edges:
            source = edge.source_id
            target = edge.target_id
            src_handle = Handle(source, edge.source_handle, edge.id)
            tgt_handle = Handle(target, edge.target_handle, edge.id)
            source_outs = self._node_id_to_node[source].outputs
            if src_handle.type not in source_outs:
                source_outs[src_handle.type] = []
            source_outs[src_handle.type].append(tgt_handle)
            target_outs = self._node_id_to_node[target].inputs
            if tgt_handle.type not in target_outs:
                target_outs[tgt_handle.type] = []
            target_outs[tgt_handle.type].append(src_handle)

    def get_input_nodes_of_handle_type(self, node: Node, type: HandleTypes):
        out_handles = node.get_input_handles(type.value)
        out_nodes = [
            self.get_node_by_id(h.target_node_id) for h in out_handles
        ]
        return out_nodes

    def get_output_nodes_of_handle_type(self, node: Node, type: HandleTypes):
        out_handles = node.get_output_handles(type.value)
        out_nodes = [
            self.get_node_by_id(h.target_node_id) for h in out_handles
        ]
        return out_nodes
                    
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

    def get_driver_nodes(self, driver: Node):
        return list(filter(lambda x: x.driver_id == driver.id, self.nodes))

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
                prev_node = self._node_id_to_node[node.id]
                if isinstance(prev_node, CommandNode) and prev_node.driver_id != "":
                    if self.node_exists(prev_node.driver_id):
                        driver = self.get_node_by_id(prev_node.driver_id)
                        if isinstance(driver, DirectSSHNode):
                            if prev_node.running_driver_id and prev_node.running_driver_id != driver.id:
                                await prev_node.shutdown()
            else:
                # new node. just append to node
                new_node_id_to_node[node.id] = node
        # handle deleted node
        for node in self._node_id_to_node.values():
            if node.id not in new_node_id_to_node:
                if isinstance(node, RunnableNodeBase):
                    if node.driver_id == "":
                        # shutdown this node
                        await node.shutdown()
                    else:
                        # if node is a remote node,
                        # shutdown process will be handled
                        # in sync_graph RPC.
                        if self.node_exists(node.driver_id):
                            driver = self.get_node_by_id(node.driver_id)
                            if isinstance(driver, DirectSSHNode):
                                await node.shutdown()


        self.update_nodes(new_node_id_to_node.values())
        # we assume edges don't contain any state, so just update them.
        # we may need to handle this in future.
        self._edge_id_to_edge = {n.id: n for n in edges}
        self._update_connection(edges)
        # self._update_driver()
        self.variable_dict = self._get_jinja_variable_dict(nodes)
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


async def _get_free_port(count: int, url: str, username: str, password: str, init_cmds: str = ""):
    client = SSHClient(url, username, password, None, "", "utf-8")
    ports = []
    # res = await client.simple_run_command(f"python -m tensorpc.cli.free_port {count}")
    # print(res)
    async with client.simple_connect() as conn:
        try:
            if init_cmds:
                cmd = (f"bash -i -c "
                    f'"{init_cmds} && python -m tensorpc.cli.free_port {count}"')
            else:
                cmd = (f"bash -i -c "
                    f'"python -m tensorpc.cli.free_port {count}"')
            result = await conn.run(cmd, check=True)
            stdout = result.stdout
            if stdout is not None:
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8")
                port_strs = stdout.strip().split(",")
                ports = list(map(int, port_strs))
        except asyncssh.process.ProcessError as e:
            traceback.print_exc()
            print(e.stdout)
            print(e.stderr)
    return ports

class NodeDesp:
    def __init__(self, node: Node, graph: FlowGraph, driver: Optional[Node]) -> None:
        self.node = node
        self.driver = driver
        self.graph = graph
        self.has_remote_driver = isinstance(driver, RemoteSSHNode)

    def get_remote_driver(self):
        assert self.driver is not None and isinstance(self.driver, RemoteSSHNode)
        return self.driver

class Flow:

    def __init__(self, root: Optional[str] = None) -> None:
        self._user_ev_q: "asyncio.Queue[Tuple[str, UserEvent]]" = asyncio.Queue(
        )
        self._ssh_q: "asyncio.Queue[Event]" = asyncio.Queue()
        self._msg_q: "asyncio.Queue[MessageEvent]" = asyncio.Queue()
        self._app_q: "asyncio.Queue[AppEvent]" = asyncio.Queue(10)

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

    def _get_node_desp(self, graph_id: str, node_id: str):
        assert graph_id in self.flow_dict, f"can't find graph {graph_id}"
        gh = self.flow_dict[graph_id]
        assert gh.node_exists(node_id), f"can't find node {node_id}"
        node = gh.get_node_by_id(node_id)
        driver: Optional[Node] = None
        if node.driver_id != "":
            if gh.node_exists(node.driver_id):
                driver = gh.get_node_by_id(node.driver_id)
        return NodeDesp(node, gh, driver)

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
    async def app_event(self):
        # ws client wait for this event to get new app event
        appev = await self._app_q.get()
        return prim.DynamicEvent(appev.uid, appev.to_dict())

    @marker.mark_websocket_event
    async def message_event(self):
        # ws client wait for this event to get new node update msg
        ev = await self._msg_q.get()
        return ev.to_dict()

    async def put_app_event(self, ev_dict: Dict[str, Any]):
        # print("APP EVENT RECEIVED")
        ev = app_event_from_data(ev_dict)
        new_t2e = {}
        gid, nid = _extract_graph_node_id(ev.uid)
        app_node, _ = self._get_app_node_and_driver(gid, nid)
        for k, v in ev.type_to_event.items():
            if k == AppEventType.ScheduleNext:
                assert isinstance(v, ScheduleNextForApp)
                gid, nid = _extract_graph_node_id(ev.uid)
                await self.schedule_next(gid, nid, v.data)
            elif k == AppEventType.UISaveStateEvent:
                assert isinstance(v, UISaveStateEvent)
                app_node.state = v.uid_to_data
            elif k == AppEventType.Notify:
                assert isinstance(v, NotifyEvent)
                if v.type == NotifyType.AppStart and app_node.state is not None:
                    save_ev = UISaveStateEvent(app_node.state)
                    print(save_ev.to_dict())
                    await self.run_single_event(gid, nid, AppEventType.UISaveStateEvent.value, save_ev.to_dict())
            else:
                new_t2e[k] = v
        ev.type_to_event = new_t2e
        if new_t2e:
            await self._app_q.put(ev)

    async def schedule_next(self, graph_id: str, node_id: str,
                            sche_ev_data: Dict[str, Any]):
        # schedule next node(s) of this node with data.
        cur_sche_ev = ScheduleEvent.from_dict(sche_ev_data)
        node_desp = self._get_node_desp(graph_id, node_id)
        node = node_desp.node
        
        # TODO if node is remote, run schedule_next in remote worker
        assert not node_desp.has_remote_driver, "TODO"
        assert node.schedulable, "only command node and scheduler node can be scheduled."
        next_schedule = node.schedule_next(cur_sche_ev, node_desp.graph)
        # print("ENTER SCHEDULE", next_schedule)

        for node_id, sche_ev in next_schedule.items():
            # print(node_id)
            sche_node = node_desp.graph.get_node_by_id(node_id)
            sche_node_remote = False
            sche_driv: Optional[Union[RemoteSSHNode, DirectSSHNode]] = None
            if node_desp.graph.node_exists(sche_node.driver_id):
                drv = node_desp.graph.get_node_by_id(sche_node.driver_id)
                assert isinstance(drv, (RemoteSSHNode, DirectSSHNode))
                sche_driv = drv
                sche_node_remote = isinstance(sche_driv, RemoteSSHNode)
            if sche_driv is None:
                print(f"node {sche_node.readable_id} don't have driver.")
                continue
            if sche_node_remote:
                # TODO
                raise NotImplementedError
            else:
                if isinstance(sche_node, CommandNode) and not isinstance(sche_node, AppNode):
                    if not sche_node.is_session_started():
                        assert isinstance(sche_driv, DirectSSHNode)
                        await self._start_session_direct(
                            graph_id, sche_node, sche_driv)
                    if sche_node.is_running():
                        sche_node.queued_commands.append(sche_ev)
                    else:
                        # TODO if two schedule events come rapidly
                        await sche_node.run_schedule_event(sche_ev)
                elif isinstance(sche_node, AppNode):
                    if sche_node.is_running():
                        await self.run_single_event(graph_id, node_id, AppEventType.ScheduleNext.value, sche_ev.to_dict())


    def _get_app_node_and_driver(self, graph_id: str, node_id: str):
        node_desp = self._get_node_desp(graph_id, node_id)
        assert isinstance(node_desp.node, AppNode)
        assert node_desp.driver is not None, f"you must select a driver for app node first"
        assert isinstance(node_desp.driver, (RemoteSSHNode, DirectSSHNode))
        return node_desp.node, node_desp.driver

        # driver = self._get_node_desp(
        #     graph_id,
        #     node.get_input_handles(HandleTypes.Driver.value)[0].target_node_id)
        # assert isinstance(driver, DirectSSHNode)
        # return node, driver

    async def run_single_event(self, graph_id: str, node_id: str,
                           type: int, ui_ev_dict: Dict[str, Any]):
        worker_key = serv_names.FLOWWORKER_RUN_APP_SINGLE_EVENT
        app_key = serv_names.APP_RUN_SINGLE_EVENT

        node, driver = self._get_app_node_and_driver(graph_id, node_id)

        if isinstance(driver, RemoteSSHNode):
            return await driver.http_remote_call(
                worker_key, graph_id, node_id,
                type, ui_ev_dict)
        else:
            sess = prim.get_http_client_session()
            http_port = node.http_port
            durl, _ = get_url_port(driver.url)
            if driver.enable_port_forward:
                app_url = get_http_url("localhost", node.fwd_http_port)
            else:
                app_url = get_http_url(durl, http_port)
            return await http_remote_call(sess, app_url,
                                          app_key,
                                          type, ui_ev_dict)

    async def run_ui_event(self, graph_id: str, node_id: str,
                           ui_ev_dict: Dict[str, Any]):
        return await self.run_single_event(graph_id, node_id, AppEventType.UIEvent.value, ui_ev_dict)

    async def run_app_editor_event(self, graph_id: str, node_id: str,
                           ui_ev_dict: Dict[str, Any]):
        return await self.run_single_event(graph_id, node_id, AppEventType.AppEditor.value, ui_ev_dict)

    async def query_app_state(self, graph_id: str, node_id: str):
        node, driver = self._get_app_node_and_driver(graph_id, node_id)
        print(node.last_event)
        if not node.is_session_started():
            return None
        if node.last_event != CommandEventType.CURRENT_COMMAND:
            return None
        if isinstance(driver, RemoteSSHNode):
            return await driver.http_remote_call(
                serv_names.FLOWWORKER_APP_GET_LAYOUT, graph_id, node_id)
        else:
            sess = prim.get_http_client_session()
            http_port = node.http_port
            durl, _ = get_url_port(driver.url)
            app_url = get_http_url(durl, http_port)
            return await http_remote_call(sess, app_url,
                                          serv_names.APP_GET_LAYOUT)

    def query_app_node_urls(self, graph_id: str, node_id: str):
        node, driver = self._get_app_node_and_driver(graph_id, node_id)
        if not node.is_session_started():
            return None
        if node.last_event != CommandEventType.CURRENT_COMMAND:
            return None
        if isinstance(driver, RemoteSSHNode):
            return {
                "grpc_url": driver.worker_grpc_url,
                "http_url": driver.worker_http_url,
                "is_remote": True,
                "module_name": node.module_name,
            }
            return (driver.worker_grpc_url, driver.worker_http_url, True, node.module_name)
        else:
            http_port = node.http_port
            grpc_port = node.grpc_port
            durl, _ = get_url_port(driver.url)
            app_url = get_http_url(durl, http_port)
            app_grpc_url = get_grpc_url(durl, grpc_port)
            return {
                "grpc_url": app_grpc_url,
                "http_url": app_url,
                "is_remote": False,
                "module_name": node.module_name,
            }

    async def put_event_from_worker(self, ev: Event):
        await self._ssh_q.put(ev)

    def query_salt(self):
        # TODO finish salt part
        return self.salt

    def set_salt(self, salt: str):
        self.salt = salt

    async def add_message(self, raw_msgs: List[Any]):
        await self._msg_q.put(MessageEvent(MessageEventType.Update, raw_msgs))
        for m in raw_msgs:
            msg = Message.from_dict(m)
            node_desp = self._get_node_desp(msg.graph_id, msg.node_id)
            node_desp.node.messages[msg.uid] = msg

    async def delete_message(self, graph_id: str, node_id: str,
                             message_id: str):
        node_desp = self._get_node_desp(graph_id, node_id)
        if message_id in node_desp.node.messages:
            node_desp.node.messages.pop(message_id)

    async def query_single_message_detail(self, graph_id: str, node_id: str,
                                          message_id: str):
        node_desp = self._get_node_desp(graph_id, node_id)
        if node_desp.has_remote_driver:
            res = await node_desp.get_remote_driver().http_remote_call(
                serv_names.FLOWWORKER_QUERY_MESSAGE_DETAIL, graph_id, node_id,
                message_id)
        else:
            res = node_desp.node.messages[message_id].to_dict_with_detail()
        return res

    async def query_message(self, graph_id: str):
        if graph_id == "":
            return {}
        print("QUERY INIT MESSAGE")
        graph = self.flow_dict[graph_id]
        all_msgs = []
        for node in graph.nodes:
            if node.driver_id != "":
                if self._node_exists(graph_id, node.driver_id):
                    driver_desp = self._get_node_desp(graph_id, node.driver_id)
                    if isinstance(driver_desp.node, RemoteSSHNode):
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
            node_desp = self._get_node_desp(graph_id, node_id)
            node = node_desp.node
            assert isinstance(node, (CommandNode, RemoteSSHNode))
            # if isinstance(event, LineEvent):
            #     # print(node.id, self.selected_node_uid == uid, event.line, end="")
            #     node.stdout += event.line
            #     if uid != self.selected_node_uid:
            #         continue
            if isinstance(event, LineEvent):
                # line event is useless for frontend.
                # we gather it as node output
                if isinstance(node, CommandNode):
                    if node._start_record_stdout:
                        node._current_line_events.append(event)
                continue

            if isinstance(event, RawEvent):
                # print(node.id, self.selected_node_uid == uid, event.line, end="")
                node.stdout += event.raw
                node.push_raw_event(event)
                # we assume node never produce special input strings during
                # terminal frontend closing.
                if node.terminal_close_ts >= 0:
                    if event.timestamp > node.terminal_close_ts:
                        evs = node.collect_raw_event_after_ts(event.timestamp)
                        if isinstance(node.terminal_state, bytes):
                            node.terminal_state += b"".join(ev.raw for ev in evs)
                        else:
                            node.terminal_state += "".join(ev.raw for ev in evs)
                        node.terminal_close_ts = event.timestamp
                        # print("NODE APPEND STATE")

                if uid != self.selected_node_uid:
                    continue

            elif isinstance(event, (CommandEvent)):
                node.last_event = event.type
                if event.type == CommandEventType.CURRENT_COMMAND:
                    if isinstance(node, CommandNode):
                        current_cmd = event.arg.decode("utf-8")
                        if node._previous_cmd.strip() == current_cmd.strip():
                            node._start_record_stdout = True 
                if event.type == CommandEventType.COMMAND_COMPLETE:
                    if isinstance(node, CommandNode):
                        if node._start_record_stdout:
                            res = node.get_previous_cmd_result()
                            if event.arg is not None:
                                res.return_code = int(event.arg)
                            print(res.cmd, res.return_code)
                            sch_ev = ScheduleEvent(time.time_ns(), res.to_dict(), {})
                            node.clear_previous_cmd()
                            await self.schedule_next(graph_id, node_id, sch_ev.to_dict())
                if event.type == CommandEventType.PROMPT_END:
                    # schedule queued tasks here.
                    if isinstance(node, CommandNode) and node.queued_commands:
                        await node.run_schedule_event(
                            node.queued_commands.pop())
                        # TODO automatic schedule next node if no manually schedule event exists.
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

    def query_node_status(self, graph_id: str, node_id: str):
        # TODO query status in remote
        if not self._node_exists(graph_id, node_id):
            return UserStatusEvent.empty().to_dict()
        node_desp = self._get_node_desp(graph_id, node_id)
        if isinstance(node_desp.node, (NodeWithSSHBase)):
            return node_desp.node.get_node_status().to_dict()
        return UserStatusEvent.empty().to_dict()


    async def save_terminal_state(self, graph_id: str, node_id: str, state,
                                  timestamp_ms: int):
        if len(state) > 0:
            node_desp = self._get_node_desp(graph_id, node_id)
            node = node_desp.node
            assert isinstance(node, (NodeWithSSHBase))
            if node_desp.has_remote_driver:
                self.selected_node_uid = ""
                return await node_desp.get_remote_driver().http_remote_call(
                    serv_names.FLOWWORKER_SET_TERMINAL_STATE, graph_id,
                    node_id, state, timestamp_ms)

            node.terminal_state = state
            node.terminal_close_ts = timestamp_ms * 1000000
        self.selected_node_uid = ""

    async def select_node(self,
                          graph_id: str,
                          node_id: str,
                          width: int = -1,
                          height: int = -1):
        node_desp = self._get_node_desp(graph_id, node_id)
        node = node_desp.node 
        assert isinstance(node, (NodeWithSSHBase))
        if node_desp.has_remote_driver:
            driver = node_desp.get_remote_driver()
            if driver.is_session_started():
                res = await driver.http_remote_call(
                    serv_names.FLOWWORKER_SELECT_NODE, graph_id, node_id,
                    width, height)
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
        node_desp = self._get_node_desp(graph_id, node_id)
        # print("INPUT", data.encode("utf-8"))
        node = node_desp.node 
        if (isinstance(node, (NodeWithSSHBase))):
            if node_desp.has_remote_driver:
                driver = node_desp.get_remote_driver()
                if driver.is_session_started():
                    return await driver.http_remote_call(
                        serv_names.FLOWWORKER_COMMAND_NODE_INPUT, graph_id,
                        node_id, data)
            if node.is_session_started():
                await node.input_queue.put(data)

    async def ssh_change_size(self, graph_id: str, node_id: str, width: int,
                              height: int):
        node_desp = self._get_node_desp(graph_id, node_id)
        node = node_desp.node 

        if isinstance(node, (NodeWithSSHBase)):
            if node_desp.has_remote_driver:
                driver = node_desp.get_remote_driver()
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
                    driv_nodes = graph.get_driver_nodes(node)

                    await node.http_remote_call(
                        serv_names.FLOWWORKER_SYNC_GRAPH, graph_id, node.to_dict(),
                        [n.to_dict() for n in driv_nodes], graph.variable_dict)

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
        if "viewport" in flow_data:
            flow_data["viewport"]["zoom"] = 1

        if graph_id in self.flow_dict:
            graph = self.flow_dict[graph_id]
            # update node status
            for n in graph.nodes:
                if isinstance(n, (CommandNode, RemoteSSHNode)):
                    uid = _get_uid(graph_id, n.id)
                    await self._user_ev_q.put((uid, n.get_node_status()))
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
            envs[TENSORPC_FLOW_NODE_READABLE_ID] = node.readable_id

            envs[TENSORPC_FLOW_MASTER_GRPC_PORT] = str(
                prim.get_server_meta().port)
            envs[TENSORPC_FLOW_MASTER_HTTP_PORT] = str(
                prim.get_server_meta().http_port)
        return envs

    async def _cmd_node_callback(self, ev: Event):
        await self._ssh_q.put(ev)

    async def _start_remote_worker(self, graph_id: str, node_id: str):
        node_desp = self._get_node_desp(graph_id, node_id)
        node = node_desp.node 
        graph = node_desp.graph
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
                # ssh_init_event ensure port forward enabled, but don't ensure server
                # starts.
                worker_exists, ssh_init_event = await node.start_session(
                    self._cmd_node_callback,
                    node.url,
                    node.username,
                    node.password,
                    rfports=rfports)
                await ssh_init_event.wait()
            await node.run_command(cmd_renderer=graph.render_command)
            # wait for ssh session init
            if not worker_exists:
                # wait for channel ready, then sync graph
                async with tensorpc.AsyncRemoteManager(
                        node.worker_grpc_url) as robj:
                    await robj.wait_for_remote_ready()
            print("QUERY STATUS!!!")
            # sync graph and query node status.
            driv_nodes = self.flow_dict[
                graph_id].get_output_nodes_of_handle_type(
                    node, HandleTypes.Driver)
            res = await node.http_remote_call(
                serv_names.FLOWWORKER_SYNC_GRAPH, graph_id,
                [n.to_dict() for n in driv_nodes], graph.variable_dict)
            for ev, node in zip(res, driv_nodes):
                le = CommandEventType(ev["last_event"])
                sess_st = SessionStatus(ev["session_status"])
                raw_msgs = ev["msgs"]
                await self._msg_q.put(
                    MessageEvent(MessageEventType.Update, raw_msgs))
                # node.last_event = le
                ue = UserStatusEvent(_get_status_from_last_event(le), sess_st)
                await self._user_ev_q.put((node.get_uid(), ue))

    async def _start_session_direct(self, graph_id: str, node: Node,
                                    driver: Union[RemoteSSHNode,
                                                  DirectSSHNode]):
        assert isinstance(node, CommandNode)
        assert isinstance(driver, DirectSSHNode)
        assert (driver.url != "" and driver.username != ""
                and driver.password != "")
        envs = self._get_node_envs(graph_id, node.id)
        rfports: List[Union[int, Tuple[int, int]]] = []
        if driver.enable_port_forward:
            rfports = [prim.get_server_meta().port]
            if prim.get_server_meta().http_port >= 0:
                rfports.append(prim.get_server_meta().http_port)
        # TODO render init commands
        await node.start_session(
            self._cmd_node_callback,
            driver.url,
            driver.username,
            driver.password,
            is_worker=False,
            enable_port_forward=driver.enable_port_forward,
            envs=envs,
            rfports=rfports,
            init_cmds=driver.init_commands,
            running_driver_id=driver.id)
        if driver.init_commands != "":
            await node.input_queue.put(driver.init_commands + "\n")

    async def start(self, graph_id: str, node_id: str):
        node_desp = self._get_node_desp(graph_id, node_id)
        node = node_desp.node 
        graph = node_desp.graph
        if isinstance(node, RemoteSSHNode):
            return await self._start_remote_worker(graph_id, node_id)
        if isinstance(node, CommandNode):
            if node_desp.driver is None:
                raise ValueError("you need to assign a driver to node first", node.readable_id)
            print("START", graph_id, node_id, node.is_session_started(),
                  type(node))
            driver = node_desp.driver
            if isinstance(driver, DirectSSHNode):
                if not node.is_session_started():
                    await self._start_session_direct(graph_id, node, driver)
                await node.run_command(cmd_renderer=graph.render_command)
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
                print("START REMOTE", driver_http_url)
                # TODO add this to remote node setting
                remote_ssh_url = "localhost:22"
                await http_remote_call(
                    prim.get_http_client_session(), driver_http_url,
                    serv_names.FLOWWORKER_CREATE_SESSION, node.raw_data,
                    graph_id)
            else:
                raise NotImplementedError

    def pause(self, graph_id: str, node_id: str):
        # currently not supported.
        print("PAUSE", graph_id, node_id)

    async def stop(self, graph_id: str, node_id: str):
        print("STOP", graph_id, node_id)
        node_desp = self._get_node_desp(graph_id, node_id)
        node = node_desp.node
        if isinstance(node, RemoteSSHNode):
            # TODO if command nodes driven by this node still running
            # raise error that stop them first.
            if node.is_session_started():
                await node.send_ctrl_c()
            # should we stop tmux session?
            return
        if node_desp.has_remote_driver:
            driver = node_desp.get_remote_driver()
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
                if isinstance(node, AppNode):
                    # query simple app state and save on 
                    # master memory (inputs, switchs, etc)
                    pass
                await node.send_ctrl_c()
            return
        else:
            raise NotImplementedError

    async def stop_session(self, graph_id: str, node_id: str):
        print("STOP SESSION", graph_id, node_id)
        node_desp = self._get_node_desp(graph_id, node_id)
        node = node_desp.node
        if isinstance(node, RemoteSSHNode):
            # TODO if command nodes driven by this node still running
            # raise error that stop them first.
            driver_http_url = node.worker_http_url
            # TODO use grpc exit RPC for better exit
            # stop tmux server
            await http_remote_call(prim.get_http_client_session(),
                                   driver_http_url, serv_names.FLOWWORKER_EXIT)
            if node.is_session_started():
                await node.soft_shutdown()
            # should we wait exit event?
            return
        if node_desp.has_remote_driver:
            driver = node_desp.get_remote_driver()
            if driver.is_session_started():
                driver_http_url = driver.worker_http_url
                await http_remote_call(prim.get_http_client_session(),
                                       driver_http_url,
                                       serv_names.FLOWWORKER_STOP_SESSION,
                                       graph_id, node_id)
            # TODO raise a exception to front end if driver not start
            return
        if isinstance(node, CommandNode):
            if node.is_session_started():
                await node.soft_shutdown()
        else:
            raise NotImplementedError

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
