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
from typing import Any 

from tensorpc import marker, prim

class Flow:
    def __init__(self) -> None:
        self._q = asyncio.Queue()

    @marker.mark_websocket_event
    async def node_status_change(self):
        # ws client wait for this event to get new node update msg
        (name, content) = await self._q.get()
        print("?")
        return prim.DynamicEvent(name, content) 

    def update_node_status(self, name: str, content: Any):
        # user client call this rpc to send message to frontend.
        loop = asyncio.get_running_loop()
        asyncio.run_coroutine_threadsafe(self._q.put((name, content)), loop)

    def query_node_stdout(self, name: str):
        return f"Hello World {name}!!!\\n"

    def start(self, name: str):
        print("START", name)

    def pause(self, name: str):
        print("PAUSE", name)

    def stop(self, name: str):
        print("STOP", name)