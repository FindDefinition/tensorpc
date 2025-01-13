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

import os

TENSORPC_ASYNCSSH_PROXY = os.getenv("TENSORPC_ASYNCSSH_PROXY", None)

TENSORPC_ASYNCSSH_INIT_SUCCESS = "__TENSORPC_ASYNCSSH_INIT_SUCCESS__"

TENSORPC_ASYNCSSH_TASK_PORT = "TENSORPC_ASYNCSSH_TASK_PORT"

TENSORPC_ASYNCSSH_TASK_HTTP_PORT = "TENSORPC_ASYNCSSH_TASK_HTTP_PORT"

TENSORPC_ASYNCSSH_TASK_KEY = "TENSORPC_ASYNCSSH_TASK_KEY"

TENSORPC_SSH_TASK_NAME_PREFIX = "__tensorpc_ssh_task"

TENSORPC_ASYNCSSH_ENV_INIT_INDICATE = "__TENSORPC_ASYNCSSH_ENV_INIT_INDICATE__"