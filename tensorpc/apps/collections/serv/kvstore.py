
import asyncio
import dataclasses
import enum
# from multiprocessing import shared_memory
# from multiprocessing.managers import SharedMemoryManager
# from multiprocessing.shared_memory import SharedMemory
from tensorpc.apps.collections.serv.shm_util import SharedMemory
import time
from typing import Any, Callable, Optional, Union, List, Dict
from tensorpc import marker, prim, AsyncRemoteManager
import numpy as np
from tensorpc.core.event_emitter.aio import AsyncIOEventEmitter
from tensorpc.compat import Python3_13AndLater
# from multiprocessing.resource_tracker import unregister

ALIGN_SIZE = 128

def _align_up(size: int, align: int) -> int:
    return (size + align - 1) // align * align

@dataclasses.dataclass
class TensorInfo:
    shape: list[int]
    dtype: np.dtype
    meta: Optional[Any]

@dataclasses.dataclass
class SharedArraySegmentDesc(TensorInfo):
    shm_offset: int = 0

    def get_byte_size(self):
        return int(np.prod(self.shape) * np.dtype(self.dtype).itemsize)

    def get_aligned_byte_size(self):
        return _align_up(self.get_byte_size(), ALIGN_SIZE)

@dataclasses.dataclass
class SharedArraySegments:
    shm: SharedMemory
    descs: list[SharedArraySegmentDesc]

    def __len__(self):
        return len(self.descs)

    def get_aligned_byte_size(self):
        return sum(seg.get_aligned_byte_size() for seg in self.descs)

    def get_array_view(self, index: int):
        desc = self.descs[index]
        byte_size = desc.get_byte_size()
        memview = self.shm.buf[desc.shm_offset:desc.shm_offset + byte_size]
        return np.ndarray(desc.shape, dtype=desc.dtype, buffer=memview)

    def close_in_remote(self):
        self.shm.close()

    def get_segments_desc(self):
        return SharedArraySegmentsDesc(self.shm.name, self.descs)

def create_untracked_shm():
    pass 

@dataclasses.dataclass
class SharedArraySegmentsDesc:
    shm_name: str
    descs: list[SharedArraySegmentDesc]

    def get_aligned_byte_size(self):
        return sum(seg.get_aligned_byte_size() for seg in self.descs)

    def get_segments(self):
        if Python3_13AndLater:
            # when we use this shared mem in other process, we don't need
            # to track it in resource tracker.
            shm = SharedMemory(name=self.shm_name, create=False, size=self.get_aligned_byte_size(), track=False) # type: ignore
        else:
            shm = SharedMemory(name=self.shm_name, create=False, size=self.get_aligned_byte_size(), track=False)
            # if not Python3_13AndLater:
            # unregister(shm._name, "shared_memory") # type: ignore
        return SharedArraySegments(shm, self.descs)


class KVStoreEventType(enum.IntEnum):
    ITEM_CHANGE = 0

@dataclasses.dataclass
class KVStoreItem:
    mtime: float
    data: Any
    metadata: Optional[Any] = None

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

    async def set_item(self, key: str, value: Any, metadata: Optional[Any] = None):
        self._store[key] = KVStoreItem(metadata=metadata, mtime=time.time(), data=value)
        await self._event_emitter.emit_async(KVStoreEventType.ITEM_CHANGE, self._store)

    def has_item(self, key: str) -> bool:
        return key in self._store

    def get_item(self, key: str):
        return self._store[key].data

    def get_all_keys(self) -> List[str]:
        return list(self._store.keys())

    def get_all_key_to_meta(self) -> dict[str, Any]:
        return {key: item.metadata for key, item in self._store.items()}

    def get_item_metadata(self, key: str):
        return self._store[key].metadata

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
        # self._store_shared_mgrs: dict[str, SharedMemoryManager] = {}
        self._store_shared_segments: dict[str, SharedArraySegments] = {}
        self._event_emitter: AsyncIOEventEmitter[KVStoreEventType, dict[str, KVStoreItem]] = AsyncIOEventEmitter()

        self._lock = asyncio.Lock()

    @marker.mark_server_event(event_type=marker.ServiceEventType.Exit)
    def _exit(self):
        for key, segments in self._store_shared_segments.items():
            # mgr = self._store_shared_mgrs[key]
            segments.shm.close()
            segments.shm.unlink()
            # mgr.shutdown()
        self._store_shared_segments.clear()
        # self._store_shared_mgrs.clear()

    def _validate_arr_desps(self, key: str, arr_desps: list[TensorInfo], raise_exc: bool = True):
        segment_descs = self._store_shared_segments[key].descs
        if len(arr_desps) == len(segment_descs):
            for i in range(len(arr_desps)):
                info = arr_desps[i]
                shape, dtype = info.shape, info.dtype
                seg_desc = segment_descs[i]
                seg_byte_size = np.prod(shape) * np.dtype(dtype).itemsize
                target_byte_size = seg_desc.get_byte_size()
                if seg_byte_size != target_byte_size:
                    if raise_exc:
                        raise ValueError(f"{key} already allocated with different byte length.")
                    else:
                        return False
        else:
            if raise_exc:
                raise ValueError(f"{key} already allocated with different number of segments.")
            else:
                return False
        return True

    def _rename_key(self, old_key: str, new_key: str):
        if old_key == new_key:
            return 
        if old_key in self._store:
            self._store[new_key] = self._store[old_key]
            del self._store[old_key]
        if old_key in self._store_shared_segments:
            self._store_shared_segments[new_key] = self._store_shared_segments[old_key]
            del self._store_shared_segments[old_key]
        # if old_key in self._store_shared_mgrs:
        #     self._store_shared_mgrs[new_key] = self._store_shared_mgrs[old_key]
        #     del self._store_shared_mgrs[old_key]

    async def get_or_create_shared_array_segments(self, key: str, arr_desps: list[TensorInfo], removed_keys: Optional[set[str]] = None):
        async with self._lock:
            if removed_keys is not None:
                # try to reuse removed shared memory
                reuse_found = False
                removed_keys_copy = removed_keys.copy()
                for reuse_key in removed_keys:
                    if reuse_key in self._store_shared_segments:
                        if self._validate_arr_desps(reuse_key, arr_desps, raise_exc=False):
                            # reuse this key
                            self._rename_key(reuse_key, key)
                            removed_keys_copy.remove(reuse_key)
                            reuse_found = True
                            break
                if removed_keys_copy:
                    for key in removed_keys_copy:
                        await self.remove_item(key, emit_event=False)
                if reuse_found or removed_keys_copy:
                    await self._event_emitter.emit_async(KVStoreEventType.ITEM_CHANGE, self._store)
            if key in self._store_shared_segments:
                # validate existed segments. if same, just return them.
                segments = self._store_shared_segments[key]
                self._validate_arr_desps(key, arr_desps)
                return segments.get_segments_desc()
            # mgr = SharedMemoryManager()
            # mgr.start()
            # self._store_shared_mgrs[key] = mgr
            segment_descs: list[SharedArraySegmentDesc] = []
            offset = 0
            for info in arr_desps:
                shape, dtype, meta = info.shape, info.dtype, info.meta
                # segments.append(mgr.SharedMemory(size=size))
                desc = SharedArraySegmentDesc(shape, dtype, meta, offset)
                aligned_size = desc.get_aligned_byte_size()
                offset += aligned_size
                segment_descs.append(desc)
            total_size = sum(seg.get_aligned_byte_size() for seg in segment_descs)
            # FIXME currently we don't track shm in shm server because it wll generate too much zombie
            # forked processes.
            mem = SharedMemory(create=True, size=total_size, track=False)
            self._store_shared_segments[key] = SharedArraySegments(mem, segment_descs)
            print("create new shared memory", key, total_size)
            return self._store_shared_segments[key].get_segments_desc()

    def backend_get_event_emitter(self):
        return self._event_emitter

    def backend_get_store(self):
        return self._store

    async def set_item_treespec(self, key: str, treespec: Any, arr_desps: list[TensorInfo], metadata: Optional[Any] = None):
        async with self._lock:
            # validate value_arr_desps
            if key not in self._store_shared_segments:
                raise ValueError(f"{key} not allocated. call `get_or_create_shared_array_segments` first.")
            self._validate_arr_desps(key, arr_desps)
            # we assume client already copy data to shared memory. so we only
            # need to store treespec.
            self._store[key] = KVStoreItem(metadata=metadata, mtime=time.time(), data=treespec)
            await self._event_emitter.emit_async(KVStoreEventType.ITEM_CHANGE, self._store)

    def has_item(self, key: str) -> bool:
        return key in self._store

    def get_item_treespec(self, key: str):
        return self._store[key].data

    def get_item_metadata(self, key: str):
        return self._store[key].metadata

    def get_item_segment_descs(self, key: str):
        return self._store_shared_segments[key].get_segments_desc()

    def get_item_shm_size(self, key: str):
        if key not in self._store_shared_segments:
            raise ValueError(f"{key} not allocated.")
        segments = self._store_shared_segments[key]
        return segments.shm.size

    def get_all_keys(self) -> List[str]:
        return list(self._store.keys())

    def get_all_key_to_meta(self) -> dict[str, Any]:
        return {key: item.metadata for key, item in self._store.items()}

    async def remove_item(self, key: str, emit_event: bool = True):
        async with self._lock:
            if key in self._store:
                del self._store[key]
                # assert key in self._store_shared_mgrs
                # mgr = self._store_shared_mgrs.pop(key)
                seg = self._store_shared_segments.pop(key)
                seg.shm.close()
                seg.shm.unlink()
                # mgr.shutdown()
                # segments always created from mgr, so no need to close
                # manually.
                if emit_event:
                    await self._event_emitter.emit_async(KVStoreEventType.ITEM_CHANGE, self._store)

    async def remove_items(self, keys: List[str]):
        # async with self._lock:
        for key in keys:
            await self.remove_item(key, emit_event=False)
        await self._event_emitter.emit_async(KVStoreEventType.ITEM_CHANGE, self._store)

    async def clear(self):
        # async with self._lock:
        for key in list(self._store.keys()):
            await self.remove_item(key, emit_event=False)
        await self._event_emitter.emit_async(KVStoreEventType.ITEM_CHANGE, self._store)
