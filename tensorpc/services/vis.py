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

from tensorpc import marker
from tensorpc.core import asyncclient, marker, prim
from typing import Any, Dict, List
import asyncio

_ALOCK = asyncio.Lock()

_CACHED_CLIENTS = {}  # type: Dict[str, asyncclient.AsyncRemoteManager]


class VisService:

    def __init__(self) -> None:
        self._q = asyncio.Queue()

    @marker.mark_websocket_event
    async def new_vis_message(self):
        # ws client wait for this event to get new vis msg
        msg = await self._q.get()
        return msg

    def send_vis_message(self, msg):
        # user client call this rpc to send message to frontend.
        loop = asyncio.get_running_loop()
        asyncio.run_coroutine_threadsafe(self._q.put(msg), loop)

    async def grpc_call(self, url, service_id, args, kwargs):
        async with _ALOCK:
            if url not in _CACHED_CLIENTS:
                _CACHED_CLIENTS[url] = asyncclient.AsyncRemoteManager(url)
                await _CACHED_CLIENTS[url].wait_for_remote_ready()
            client = _CACHED_CLIENTS[url]
        return await client.remote_json_call(service_id, *args, **kwargs)
