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

import dataclasses
import inspect
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import queue
import threading
from typing import Any, Callable, Optional, Union, List, Dict
import multiprocessing
import sys
from tensorpc import marker, prim, AsyncRemoteManager
import traceback
import asyncio
import numpy as np
import time
from tensorpc.core.event_emitter.call_server import SimpleRPCHandler
from tensorpc.core.serviceunit import ServiceEventType
from tensorpc.compat import Python3_13AndLater
from multiprocessing.resource_tracker import unregister

class Simple:

    def __init__(self) -> None:
        pass

    def echo(self, x):
        return x

    def sleep(self, interval: float):
        time.sleep(interval)


class SpeedTestServer:
    def __init__(self):
        self._cached_data = None
        self._cached_size = 0

    def recv_data(self, x):
        return

    def send_data(self, size_mb: int):
        if size_mb == self._cached_size:
            return self._cached_data
        self._cached_size = size_mb
        np.random.seed(5)
        self._cached_data = np.random.uniform(size=[size_mb * 1024 * 1024 // 4]).astype(np.float32)
        return self._cached_data


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

    @marker.mark_server_event(event_type=ServiceEventType.Exit)
    async def on_exit(self):
        print("EXIT!!!")

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

@dataclasses.dataclass
class HandlerItem:
    type: str
    handler: Callable
    once: bool = False
    loop: Optional[asyncio.AbstractEventLoop] = None


class ArgServer(SimpleRPCHandler):
    """usually used when you want to run some python function inside
    subprocess/SSH (can't pass python objects directly).
    User can run a client in subprocess, load args from this service,
    run python code and store results to this service.
    """

    def on(self, event: str, f: Callable, force_replace: bool = False, once: bool = False, loop: Optional[asyncio.AbstractEventLoop] = None):
        return super().on(event, f, force_replace, once, loop)

    def once(self, event: str, f: Callable, force_replace: bool = False, loop: Optional[asyncio.AbstractEventLoop] = None):
        return self.on(event, f, force_replace, True, loop)

    def off(self, event: str):
        return super().off(event)

    async def call_event(self, event: str, *args, **kwargs):
        return await super().call_event(event, *args, **kwargs)

class KVStore:
    def __init__(self):
        self._store = {}

    def set_item(self, key: str, value):
        self._store[key] = value

    def has_item(self, key: str) -> bool:
        return key in self._store

    def get_item(self, key: str):
        return self._store[key]

    def get_all_keys(self) -> List[str]:
        return list(self._store.keys())

    def remove_item(self, key: str):
        if key in self._store:
            del self._store[key]

    def remove_items(self, keys: List[str]):
        for key in keys:
            self.remove_item(key)

    def clear(self):
        self._store.clear()

@dataclasses.dataclass
class SharedArraySegment:
    shm: SharedMemory
    shape: list[int]
    dtype: np.dtype

    def get_array_view(self):
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def get_desc(self):
        return SharedArraySegmentDesc(self.shm.name, self.shape, self.dtype)

    def close_in_remote(self):
        self.shm.close()

@dataclasses.dataclass
class SharedArraySegmentDesc:
    shm_name: str
    shape: list[int]
    dtype: np.dtype

    def get_byte_size(self):
        return int(np.prod(self.shape) * np.dtype(self.dtype).itemsize)

    def get_segment(self):
        shm = SharedMemory(name=self.shm_name, create=False, size=self.get_byte_size())
        if Python3_13AndLater:
            # when we use this shared mem in other process, we don't need
            # to track it in resource tracker.
            shm = SharedMemory(name=self.shm_name, create=False, size=self.get_byte_size(), track=False) # type: ignore
        else:
            shm = SharedMemory(name=self.shm_name, create=False, size=self.get_byte_size())
            # if not Python3_13AndLater:
            unregister(shm._name, "shared_memory") # type: ignore

        return SharedArraySegment(shm, self.shape, self.dtype)

class ShmKVStore:
    def __init__(self):
        self._store = {}
        self._store_shared_mgrs: dict[str, SharedMemoryManager] = {}
        self._store_shared_segments: dict[str, list[SharedArraySegment]] = {}

    @marker.mark_server_event(event_type=marker.ServiceEventType.Exit)
    def _exit(self):
        print("SHUTDOWN shared memory managers")
        for key, segments in self._store_shared_segments.items():
            for seg in segments:
                seg.shm.close()
        for key, mgr in self._store_shared_mgrs.items():
            mgr.shutdown()
        self._store_shared_mgrs.clear()

    def _validate_arr_desps(self, key: str, arr_desps: list[tuple[list[int], np.dtype]]):
        segments = self._store_shared_segments[key]
        if len(arr_desps) == len(segments):
            for i in range(len(arr_desps)):
                shape, dtype = arr_desps[i]
                seg = segments[i]
                seg_byte_size = np.prod(shape) * np.dtype(dtype).itemsize
                target_byte_size = seg.shm.size
                if seg_byte_size != target_byte_size:
                    raise ValueError(f"{key} already allocated with different byte length.")
        else:
            raise ValueError(f"{key} already allocated with different number of segments.")

    def get_or_create_shared_array_segments(self, key: str, arr_desps: list[tuple[list[int], np.dtype]]):
        if key in self._store_shared_segments:
            # validate existed segments. if same, just return them.
            segments = self._store_shared_segments[key]
            self._validate_arr_desps(key, arr_desps)
            return segments
        mgr = SharedMemoryManager()
        mgr.start()
        self._store_shared_mgrs[key] = mgr
        segments: list[SharedArraySegment] = []
        for shape, dtype in arr_desps:
            # segments.append(mgr.SharedMemory(size=size))
            size = np.prod(shape) * np.dtype(dtype).itemsize
            shm = mgr.SharedMemory(size=int(size))
            segments.append(SharedArraySegment(shm, shape, dtype))

        self._store_shared_segments[key] = segments
        return segments

    def set_item_treespec(self, key: str, treespec: Any, arr_desps: list[tuple[list[int], np.dtype]]):
        # validate value_arr_desps
        if key not in self._store_shared_segments:
            raise ValueError(f"{key} not allocated. call `get_or_create_shared_array_segments` first.")
        self._validate_arr_desps(key, arr_desps)
        # we assume client already copy data to shared memory. so we only
        # need to store treespec.
        self._store[key] = treespec

    def has_item(self, key: str) -> bool:
        return key in self._store

    def get_item_treespec(self, key: str):
        return self._store[key]

    def get_item_segment_descs(self, key: str):
        return [seg.get_desc() for seg in self._store_shared_segments[key]]

    def get_item_shm_size(self, key: str):
        if key not in self._store_shared_segments:
            raise ValueError(f"{key} not allocated.")
        segments = self._store_shared_segments[key]
        size = 0
        for seg in segments:
            size += seg.shm.size
        return size

    def get_all_keys(self) -> List[str]:
        return list(self._store.keys())

    def remove_item(self, key: str):
        if key in self._store:
            del self._store[key]
            assert key in self._store_shared_mgrs
            mgr = self._store_shared_mgrs[key]
            mgr.shutdown()
            del self._store_shared_mgrs[key]
            # segments always created from mgr, so no need to close
            # manually.
            del self._store_shared_segments[key]

    def remove_items(self, keys: List[str]):
        for key in keys:
            self.remove_item(key)

    def clear(self):
        for key in self._store.keys():
            self.remove_item(key)
        