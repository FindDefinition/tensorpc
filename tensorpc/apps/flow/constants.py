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

FLOW_FOLDER_PATH = Path.home() / ".tensorpc" / "flow"

FLOW_DEFAULT_TIMEOUT = 10

FLOW_DEFAULT_GRAPH_ID = "default_flow"

TENSORPC_FLOW_GRAPH_ID = "TENSORPC_FLOW_GRAPH_ID"
TENSORPC_FLOW_NODE_ID = "TENSORPC_FLOW_NODE_ID"
TENSORPC_FLOW_NODE_UID = "TENSORPC_FLOW_NODE_UID"
TENSORPC_FLOW_MASTER_GRPC_PORT = "TENSORPC_FLOW_MASTER_GRPC_PORT"
TENSORPC_FLOW_MASTER_HTTP_PORT = "TENSORPC_FLOW_MASTER_HTTP_PORT"
TENSORPC_FLOW_USE_REMOTE_FWD = "TENSORPC_FLOW_USE_REMOTE_FWD"
TENSORPC_FLOW_IS_WORKER = "TENSORPC_FLOW_IS_WORKER"
TENSORPC_FLOW_DEFAULT_TMUX_NAME = "tensorpc_flow"

