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

import tensorpc 
import time 
from tensorpc.apps.flow.constants import FLOW_UPDATE_NODE_STATUS
def update_status():
    tensorpc.simple_remote_call("localhost:51051", FLOW_UPDATE_NODE_STATUS, "WTF", f"NEWS {time.time()}")

if __name__ == "__main__":
    print(FLOW_UPDATE_NODE_STATUS)
    update_status()