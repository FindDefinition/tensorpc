# Copyright 2024 Yan Yan
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

import enum
from tensorpc.core.defs import Service, ServiceDef, from_yaml_path
from tensorpc.constants import TENSORPC_SPLIT

class BuiltinServiceKeys(enum.Enum):
    FileOps = f"tensorpc.services.collection{TENSORPC_SPLIT}FileOps"
    ArgServer = f"tensorpc.services.collection{TENSORPC_SPLIT}ArgServer"
    SpeedTestServer = f"tensorpc.services.collection{TENSORPC_SPLIT}SpeedTestServer"
    Flow = f"tensorpc.flow.serv.core{TENSORPC_SPLIT}Flow"
    RemoteComponentService = f"tensorpc.flow.serv.remote_comp{TENSORPC_SPLIT}RemoteComponentService"
    BackgroundDebugTools = f"tensorpc.services.dbg.tools{TENSORPC_SPLIT}BackgroundDebugTools"
    RTTraceStorageService = f"tensorpc.services.dbg.rttrace{TENSORPC_SPLIT}RTTraceStorageService"
    Simple = f"tensorpc.services.collection{TENSORPC_SPLIT}Simple"
    Scheduler = f"tensorpc.autossh.services.scheduler{TENSORPC_SPLIT}Scheduler"
    TaskWrapper = f"tensorpc.autossh.services.taskwrapper{TENSORPC_SPLIT}TaskWrapper"
    TaskManager = f"tensorpc.autossh.services.taskwrapper{TENSORPC_SPLIT}TaskManager"

BUILTIN_SERVICES = [
    Service(BuiltinServiceKeys.FileOps.value, {}),
    Service(BuiltinServiceKeys.ArgServer.value, {}),
    Service(BuiltinServiceKeys.SpeedTestServer.value, {}),
    Service(BuiltinServiceKeys.Flow.value, {}),
    Service(BuiltinServiceKeys.RemoteComponentService.value, {}),
    Service(BuiltinServiceKeys.Simple.value, {}),
    Service(BuiltinServiceKeys.BackgroundDebugTools.value, {}),
    Service(BuiltinServiceKeys.RTTraceStorageService.value, {}),
    Service(BuiltinServiceKeys.Scheduler.value, {}),
    Service(BuiltinServiceKeys.TaskWrapper.value, {}),
    Service(BuiltinServiceKeys.TaskManager.value, {}),
]


def get_http_url(url: str, port: int):
    return f"http://{url}:{port}/api/rpc"


def get_grpc_url(url: str, port: int):
    return f"{url}:{port}"


def get_websocket_url(url: str, port: int):
    return f"http://{url}:{port}/api/ws"
