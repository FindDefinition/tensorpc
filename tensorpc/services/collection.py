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

from pathlib import Path
import queue
import threading
from typing import Optional, Union, List, Dict
import multiprocessing
import sys
from tensorpc import marker, prim, AsyncRemoteManager
import traceback
import asyncio
import numpy as np
import time

from tensorpc.core.serviceunit import ServiceEventType


class Simple:

    def __init__(self) -> None:
        print("???????????")

    def echo(self, x):
        return x

    def sleep(self, interval: float):
        time.sleep(interval)


class SpeedTestServer:

    def recv_data(self, x):
        return

    def send_data(self, size_mb: int):
        return np.zeros([size_mb * 1024 * 1024], dtype=np.uint8)


class FileOps:

    def print_in_server(self, content):
        print(content)

    def get_file(self, path, start_chunk=0, chunk_size=65536):
        """service that get a large file from server.
        you need to use remote_generator instead of remote_call.
        If error occurs in client, you can use chunk index to
        recover transfer.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError("{} not found.".format(path))
        with path.open("rb") as f:
            f.seek(start_chunk * chunk_size)
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                yield data

    def get_file_size(self, path) -> int:
        path = Path(path)
        if not path.exists():
            return -1
        return path.stat().st_size

    def path_exists(self, path) -> bool:
        path = Path(path)
        return path.exists()

    def glob(self, folder, pattern):
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError("{} not found.".format(folder))
        res = list(folder.glob(pattern))
        if prim.is_json_call():
            return list(map(str, res))
        # for python client, we can send Path objects which have more information.
        return res

    def rglob(self, folder, pattern):
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError("{} not found.".format(folder))
        res = list(folder.glob(pattern))
        if prim.is_json_call():
            return list(map(str, res))
        return res

    @marker.mark_client_stream
    async def upload_file(self,
                          gen_iter,
                          path,
                          exist_ok: bool = False,
                          parents: bool = False):
        """service that upload a large file to server.
        you need to use client_stream instead of remote_call.
        for transfer recovery, we need to save states to server
        which isn't covered in this example.
        """
        path = Path(path)
        if path.exists() and not exist_ok:
            raise FileExistsError("{} exists.".format(path))
        if not path.parent.exists():
            if parents:
                path.parent.mkdir(mode=0o755, parents=parents)
            else:
                raise ValueError("{} parent not exist.".format(path))
        try:
            with path.open("wb") as f:
                async for chunk in gen_iter:
                    f.write(chunk)
        except Exception as e:
            path.unlink()
            raise e


class ProcessObserver:

    def __init__(
            self,
            q: Optional[Union[queue.Queue,
                              multiprocessing.Queue]] = None) -> None:
        self.q = q

    @marker.mark_server_event(event_type=ServiceEventType.BeforeServerStart)
    def server_start(self):
        if self.q is not None:
            port = prim.get_server_grpc_port()
            self.q.put(port)

    def get_threads_current_status(self):
        this_tid = threading.get_ident()
        res = []
        threading_path = Path(threading.__file__)
        for threadId, frame in sys._current_frames().items():
            if threadId == this_tid:
                continue
            if threading_path == Path(frame.f_code.co_filename):
                continue
            res.append({
                "thread_id": threadId,
                "filename": frame.f_code.co_filename,
                "lineno": frame.f_lineno,
            })
        return res


class ProcessObserveManager:

    def __init__(self) -> None:
        is_sync_server = prim.get_server_exposed_props().is_sync
        self._lock = asyncio.Lock()
        self.clients: Dict[str, AsyncRemoteManager] = {}
        self.is_sync_server = is_sync_server

    @marker.mark_server_event(event_type=marker.ServiceEventType.Init)
    async def init(self):
        if self.is_sync_server:
            return
        self._check_loop_task = asyncio.create_task(self._check_client_loop())

    async def register_client(self, url: str, identifier: str):
        if self.is_sync_server:
            raise ValueError(
                "register_client can only be called in async server.")
        async with self._lock:
            self.clients[identifier] = AsyncRemoteManager(url)

    async def register_local_client(self, port: int, identifier: str):
        return await self.register_client(f"localhost:{port}", identifier)

    async def _check_client_loop(self):
        stdn_ev = prim.get_async_shutdown_event()
        shut_task = asyncio.create_task(stdn_ev.wait())

        wait_tasks: List[asyncio.Task] = [
            shut_task, asyncio.create_task(asyncio.sleep(1))
        ]

        while True:
            (done,
             pending) = await asyncio.wait(wait_tasks,
                                           return_when=asyncio.FIRST_COMPLETED)
            if shut_task in done:
                break
            wait_tasks: List[asyncio.Task] = [
                shut_task, asyncio.create_task(asyncio.sleep(1))
            ]
            for identifier, client in self.clients.items():
                try:
                    res = await client.health_check(timeout=1)
                except Exception as e:
                    async with self._lock:
                        self.clients.pop(identifier)
