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

from tensorpc.core.defs import Service, ServiceDef, from_yaml_path

BUILTIN_SERVICES = [ 
    Service("tensorpc.services.vis:VisService", {}),
    Service("tensorpc.services.collection:FileOps", {}),
    Service("tensorpc.apps.flow.serv.core:Flow", {}),
    Service("tensorpc.apps.flow.serv.worker:FlowWorker", {}),
    Service("tensorpc.services.collection:Simple", {}),
]

def get_http_url(url: str, port: int):
    return f"http://{url}:{port}/api/rpc"

def get_grpc_url(url: str, port: int):
    return f"{url}:{port}"

def get_websocket_url(url: str, port: int):
    return f"http://{url}:{port}/api/ws"