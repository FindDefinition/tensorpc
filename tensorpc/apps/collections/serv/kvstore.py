
import dataclasses
import enum
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
import time
from typing import Any, Callable, Optional, Union, List, Dict
from tensorpc import marker, prim, AsyncRemoteManager
import numpy as np
from tensorpc.core.event_emitter.aio import AsyncIOEventEmitter
from tensorpc.compat import Python3_13AndLater
from multiprocessing.resource_tracker import unregister

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

class KVStoreEventType(enum.IntEnum):
    ITEM_CHANGE = 0

@dataclasses.dataclass
class KVStoreItem:
    type: Union[int, str]
    mtime: float
    data: Any

@dataclasses.dataclass
class KVStoreChangeEvent:
    event_type: KVStoreEventType
    store: dict[str, KVStoreItem]

class KVStore:
    def __init__(self):
        self._store: dict[str, KVStoreItem] = {}

    @marker.mark_server_event(event_type=marker.ServiceEventType.Init)
    def _init(self):
        self._event_emitter: AsyncIOEventEmitter[KVStoreEventType, dict[str, KVStoreItem]] = AsyncIOEventEmitter()
        
    def backend_get_event_emitter(self):
        return self._event_emitter

    async def set_item(self, key: str, value: Any, type: Union[int, str] = "unknown"):
        self._store[key] = KVStoreItem(type=type, mtime=time.time(), data=value)
        await self._event_emitter.emit_async(KVStoreEventType.ITEM_CHANGE, self._store)

    def has_item(self, key: str) -> bool:
        return key in self._store

    def get_item(self, key: str):
        return self._store[key].data

    def get_all_keys(self) -> List[str]:
        return list(self._store.keys())

    async def remove_item(self, key: str, emit_event: bool = True):
        if key in self._store:
            del self._store[key]
            if emit_event:
                await self._event_emitter.emit_async(KVStoreEventType.ITEM_CHANGE, self._store)

    async def remove_items(self, keys: List[str]):
        for key in keys:
            await self.remove_item(key, emit_event=False)
        await self._event_emitter.emit_async(KVStoreEventType.ITEM_CHANGE, self._store)

    async def clear(self):
        self._store.clear()
        await self._event_emitter.emit_async(KVStoreEventType.ITEM_CHANGE, self._store)

class ShmKVStore:
    def __init__(self):
        self._store: dict[str, KVStoreItem] = {}
        self._store_shared_mgrs: dict[str, SharedMemoryManager] = {}
        self._store_shared_segments: dict[str, list[SharedArraySegment]] = {}

    @marker.mark_server_event(event_type=marker.ServiceEventType.Init)
    def _init(self):
        self._event_emitter: AsyncIOEventEmitter[KVStoreEventType, dict[str, KVStoreItem]] = AsyncIOEventEmitter()

    @marker.mark_server_event(event_type=marker.ServiceEventType.Exit)
    def _exit(self):
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
        return [s.get_desc() for s in segments]

    def backend_get_event_emitter(self):
        return self._event_emitter

    async def set_item_treespec(self, key: str, treespec: Any, arr_desps: list[tuple[list[int], np.dtype]], type: Union[int, str] = "unknown"):
        # validate value_arr_desps
        if key not in self._store_shared_segments:
            raise ValueError(f"{key} not allocated. call `get_or_create_shared_array_segments` first.")
        self._validate_arr_desps(key, arr_desps)
        # we assume client already copy data to shared memory. so we only
        # need to store treespec.
        self._store[key] = KVStoreItem(type=type, mtime=time.time(), data=treespec)
        await self._event_emitter.emit_async(KVStoreEventType.ITEM_CHANGE, self._store)

    def has_item(self, key: str) -> bool:
        return key in self._store

    def get_item_treespec(self, key: str):
        return self._store[key].data

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

    async def remove_item(self, key: str, emit_event: bool = True):
        if key in self._store:
            del self._store[key]
            assert key in self._store_shared_mgrs
            mgr = self._store_shared_mgrs[key]
            mgr.shutdown()
            del self._store_shared_mgrs[key]
            # segments always created from mgr, so no need to close
            # manually.
            del self._store_shared_segments[key]
            if emit_event:
                await self._event_emitter.emit_async(KVStoreEventType.ITEM_CHANGE, self._store)

    async def remove_items(self, keys: List[str]):
        for key in keys:
            await self.remove_item(key, emit_event=False)
        await self._event_emitter.emit_async(KVStoreEventType.ITEM_CHANGE, self._store)

    async def clear(self):
        for key in self._store.keys():
            await self.remove_item(key, emit_event=False)
        await self._event_emitter.emit_async(KVStoreEventType.ITEM_CHANGE, self._store)
