import asyncio
import contextlib
import dataclasses
import enum
import traceback
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

    def get_byte_size(self):
        return int(np.prod(self.shape) * np.dtype(self.dtype).itemsize)

    def get_aligned_byte_size(self):
        return _align_up(self.get_byte_size(), ALIGN_SIZE)


@dataclasses.dataclass
class PartialArrayInfo:
    start: int
    length: int

    def get_aligned_byte_size(self):
        return _align_up(self.length, ALIGN_SIZE)


@dataclasses.dataclass
class SharedArraySegmentDesc(TensorInfo):
    shm_offset: int = 0
    var_idx: int = -1
    partial_info: Optional[PartialArrayInfo] = None

    def get_partial_info_checked(self):
        assert self.partial_info is not None
        return self.partial_info

    def is_partial(self):
        if self.partial_info is None:
            return False
        return self.partial_info.length < self.get_aligned_byte_size()

    def get_partial_size(self):
        if self.partial_info is None:
            return self.get_byte_size()
        return self.partial_info.length

    def get_partial_size_aligned(self):
        if self.partial_info is None:
            return self.get_aligned_byte_size()
        return self.partial_info.get_aligned_byte_size()

    def get_shm_offset_with_partial(self):
        if self.partial_info is None:
            return self.shm_offset
        return self.shm_offset + self.partial_info.start

    def is_last_partial(self):
        if self.partial_info is None:
            return True
        return self.partial_info.start + self.partial_info.length == self.get_aligned_byte_size(
        )


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
        assert not desc.is_partial()
        byte_size = desc.get_byte_size()
        memview = self.shm.buf[desc.shm_offset:desc.shm_offset + byte_size]
        return np.ndarray(desc.shape, dtype=desc.dtype, buffer=memview)

    def get_array_view_raw(self, index: int):
        desc = self.descs[index]
        byte_size = desc.get_partial_size()
        memview = self.shm.buf[desc.shm_offset:desc.shm_offset + byte_size]
        return np.ndarray([byte_size], dtype=np.uint8, buffer=memview)

    def close_in_remote(self):
        self.shm.close()

    def get_segments_desc(self):
        return SharedArraySegmentsDesc(self.shm.name, self.descs)

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
            shm = SharedMemory(name=self.shm_name,
                               create=False,
                               size=self.get_aligned_byte_size(),
                               track=False)  # type: ignore
        else:
            shm = SharedMemory(name=self.shm_name,
                               create=False,
                               size=self.get_aligned_byte_size(),
                               track=False)
            # if not Python3_13AndLater:
            # unregister(shm._name, "shared_memory") # type: ignore
        return SharedArraySegments(shm, self.descs)


@dataclasses.dataclass
class SharedArraySegmentsLimitedDesc(SharedArraySegmentsDesc):
    desc_start_idx: int


class KVStoreEventType(enum.IntEnum):
    ITEM_CHANGE = 0


@dataclasses.dataclass
class KVStoreItem:
    mtime: float
    data: Any
    metadata: Optional[Any]


@dataclasses.dataclass
class KVStoreItemWithArrDesc(KVStoreItem):
    buffer: np.ndarray
    descs: list[SharedArraySegmentDesc]


@dataclasses.dataclass
class KVStoreChangeEvent:
    event_type: KVStoreEventType
    store: dict[str, KVStoreItem]


class KVStore:

    def __init__(self):
        self._store: dict[str, KVStoreItem] = {}

    @marker.mark_server_event(event_type=marker.ServiceEventType.Init)
    def _init(self):
        self._event_emitter: AsyncIOEventEmitter[KVStoreEventType, dict[
            str, KVStoreItem]] = AsyncIOEventEmitter()

    def backend_get_event_emitter(self):
        return self._event_emitter

    async def set_item(self,
                       key: str,
                       value: Any,
                       metadata: Optional[Any] = None):
        self._store[key] = KVStoreItem(metadata=metadata,
                                       mtime=time.time(),
                                       data=value)
        await self._event_emitter.emit_async(KVStoreEventType.ITEM_CHANGE,
                                             self._store)

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
                await self._event_emitter.emit_async(
                    KVStoreEventType.ITEM_CHANGE, self._store)

    async def remove_items(self, keys: List[str]):
        for key in keys:
            await self.remove_item(key, emit_event=False)
        await self._event_emitter.emit_async(KVStoreEventType.ITEM_CHANGE,
                                             self._store)

    async def clear(self):
        self._store.clear()
        await self._event_emitter.emit_async(KVStoreEventType.ITEM_CHANGE,
                                             self._store)


class ShmKVStore:

    def __init__(self):
        self._store: dict[str, KVStoreItem] = {}
        # self._store_shared_mgrs: dict[str, SharedMemoryManager] = {}
        self._store_shared_segments: dict[str, SharedArraySegments] = {}
        self._event_emitter: AsyncIOEventEmitter[KVStoreEventType, dict[
            str, KVStoreItem]] = AsyncIOEventEmitter()

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

    def _validate_arr_descs(self,
                            key: str,
                            arr_descs: list[TensorInfo],
                            raise_exc: bool = True):
        segment_descs = self._store_shared_segments[key].descs
        if len(arr_descs) == len(segment_descs):
            for i in range(len(arr_descs)):
                info = arr_descs[i]
                shape, dtype = info.shape, info.dtype
                seg_desc = segment_descs[i]
                seg_byte_size = np.prod(shape) * np.dtype(dtype).itemsize
                target_byte_size = seg_desc.get_byte_size()
                if seg_byte_size != target_byte_size:
                    if raise_exc:
                        raise ValueError(
                            f"{key} already allocated with different byte length."
                        )
                    else:
                        return False
        else:
            if raise_exc:
                raise ValueError(
                    f"{key} already allocated with different number of segments."
                )
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
            self._store_shared_segments[new_key] = self._store_shared_segments[
                old_key]
            del self._store_shared_segments[old_key]
        # if old_key in self._store_shared_mgrs:
        #     self._store_shared_mgrs[new_key] = self._store_shared_mgrs[old_key]
        #     del self._store_shared_mgrs[old_key]

    async def get_or_create_shared_array_segments(
            self,
            key: str,
            arr_descs: list[TensorInfo],
            removed_keys: Optional[set[str]] = None):
        async with self._lock:
            if removed_keys is not None:
                # try to reuse removed shared memory
                reuse_found = False
                removed_keys_copy = removed_keys.copy()
                for reuse_key in removed_keys:
                    if reuse_key in self._store_shared_segments:
                        if self._validate_arr_descs(reuse_key,
                                                    arr_descs,
                                                    raise_exc=False):
                            # reuse this key
                            self._rename_key(reuse_key, key)
                            removed_keys_copy.remove(reuse_key)
                            reuse_found = True
                            break
                if removed_keys_copy:
                    for key in removed_keys_copy:
                        await self.remove_item(key, emit_event=False)
                if reuse_found or removed_keys_copy:
                    await self._event_emitter.emit_async(
                        KVStoreEventType.ITEM_CHANGE, self._store)
            if key in self._store_shared_segments:
                # validate existed segments. if same, just return them.
                segments = self._store_shared_segments[key]
                self._validate_arr_descs(key, arr_descs)
                return segments.get_segments_desc()
            # mgr = SharedMemoryManager()
            # mgr.start()
            # self._store_shared_mgrs[key] = mgr
            segment_descs: list[SharedArraySegmentDesc] = []
            offset = 0
            for info in arr_descs:
                shape, dtype, meta = info.shape, info.dtype, info.meta
                # segments.append(mgr.SharedMemory(size=size))
                desc = SharedArraySegmentDesc(shape, dtype, meta, offset)
                aligned_size = desc.get_aligned_byte_size()
                offset += aligned_size
                segment_descs.append(desc)
            total_size = sum(seg.get_aligned_byte_size()
                             for seg in segment_descs)
            # FIXME currently we don't track shm in shm server because it wll generate too much zombie
            # forked processes.
            mem = SharedMemory(create=True, size=total_size, track=False)
            self._store_shared_segments[key] = SharedArraySegments(
                mem, segment_descs)
            print("create new shared memory", key, total_size)
            return self._store_shared_segments[key].get_segments_desc()

    def backend_get_event_emitter(self):
        return self._event_emitter

    def backend_get_store(self):
        return self._store

    async def set_item_treespec(self,
                                key: str,
                                treespec: Any,
                                arr_descs: list[TensorInfo],
                                metadata: Optional[Any] = None):
        async with self._lock:
            # validate value_arr_descs
            if key not in self._store_shared_segments:
                raise ValueError(
                    f"{key} not allocated. call `get_or_create_shared_array_segments` first."
                )
            self._validate_arr_descs(key, arr_descs)
            # we assume client already copy data to shared memory. so we only
            # need to store treespec.
            self._store[key] = KVStoreItem(metadata=metadata,
                                           mtime=time.time(),
                                           data=treespec)
            await self._event_emitter.emit_async(KVStoreEventType.ITEM_CHANGE,
                                                 self._store)

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
                    await self._event_emitter.emit_async(
                        KVStoreEventType.ITEM_CHANGE, self._store)

    async def remove_items(self, keys: List[str]):
        # async with self._lock:
        for key in keys:
            await self.remove_item(key, emit_event=False)
        await self._event_emitter.emit_async(KVStoreEventType.ITEM_CHANGE,
                                             self._store)

    async def clear(self):
        # async with self._lock:
        for key in list(self._store.keys()):
            await self.remove_item(key, emit_event=False)
        await self._event_emitter.emit_async(KVStoreEventType.ITEM_CHANGE,
                                             self._store)


class ShmTrOnlyKVStore:
    """use double buffer to send/receive data to/from client by size limited shm.
    """

    def __init__(self):
        self._max_concurrent_clients = 8  # assume 8 clients (e.g. 8 GPUs per node)
        self._shm_size_per_client = 1 * 1024 * 1024 * 1024  # 1GB (double buffer)

        self._semaphore = asyncio.Semaphore(self._max_concurrent_clients)
        self._shm_state_is_idle = [True] * self._max_concurrent_clients

        self._shm_buffers: list[tuple[SharedMemory, SharedMemory]] = []

        self._store: dict[str, KVStoreItemWithArrDesc] = {}
        # self._store_shared_mgrs: dict[str, SharedMemoryManager] = {}
        # self._store_shared_segments: dict[str, SharedArraySegments] = {}
        self._event_emitter: AsyncIOEventEmitter[KVStoreEventType, dict[
            str, KVStoreItemWithArrDesc]] = AsyncIOEventEmitter()

        self._lock = asyncio.Lock()

    def _lazy_init_buffer(self):
        if not self._shm_buffers:
            for i in range(self._max_concurrent_clients):
                # double buffer
                shm_1 = SharedMemory(create=True,
                                     size=self._shm_size_per_client,
                                     track=False)
                shm_2 = SharedMemory(create=True,
                                     size=self._shm_size_per_client,
                                     track=False)
                print("alloc new shared memory", shm_1.name, shm_2.name,
                      self._shm_size_per_client)
                self._shm_buffers.append((shm_1, shm_2))

    @marker.mark_server_event(event_type=marker.ServiceEventType.Exit)
    def _exit(self):
        for shm_1, shm_2 in self._shm_buffers:
            shm_1.close()
            shm_1.unlink()
            shm_2.close()
            shm_2.unlink()

    @contextlib.asynccontextmanager
    async def _acquire_shm_double_buffer(self):
        await self._semaphore.acquire()
        aquire_idx = -1
        for i in range(self._max_concurrent_clients):
            if self._shm_state_is_idle[i]:
                self._shm_state_is_idle[i] = False
                aquire_idx = i
                break
        shm_double_buffer = self._shm_buffers[aquire_idx]
        try:
            yield shm_double_buffer
        finally:
            self._shm_state_is_idle[aquire_idx] = True
            self._semaphore.release()

    def _validate_arr_descs(self,
                            key: str,
                            segment_descs: list[SharedArraySegmentDesc],
                            arr_descs: list[TensorInfo],
                            raise_exc: bool = True):
        if len(arr_descs) == len(segment_descs):
            for i in range(len(arr_descs)):
                info = arr_descs[i]
                shape, dtype = info.shape, info.dtype
                seg_desc = segment_descs[i]
                seg_byte_size = np.prod(shape) * np.dtype(dtype).itemsize
                target_byte_size = seg_desc.get_byte_size()
                if seg_byte_size != target_byte_size:
                    if raise_exc:
                        raise ValueError(
                            f"{key} already allocated with different byte length."
                        )
                    else:
                        return False
        else:
            if raise_exc:
                raise ValueError(
                    f"{key} already allocated with different number of segments."
                )
            else:
                return False
        return True

    def _rename_key(self, old_key: str, new_key: str):
        if old_key == new_key:
            return
        if old_key in self._store:
            self._store[new_key] = self._store[old_key]
            del self._store[old_key]
        # if old_key in self._store_shared_segments:
        #     self._store_shared_segments[new_key] = self._store_shared_segments[old_key]
        #     del self._store_shared_segments[old_key]
        # if old_key in self._store_shared_mgrs:
        #     self._store_shared_mgrs[new_key] = self._store_shared_mgrs[old_key]
        #     del self._store_shared_mgrs[old_key]

    def _chunkize_arr_descs(self, arr_descs: list[SharedArraySegmentDesc],
                            max_size: int):
        chunks: list[list[SharedArraySegmentDesc]] = []
        cur_chunk: list[SharedArraySegmentDesc] = []
        cur_size = 0
        for info in arr_descs:
            size = info.get_byte_size()
            offset = 0
            while size > 0:
                remain_size = max_size - cur_size
                if remain_size > 0:
                    stored_size = min(remain_size, size)
                    partial_info = PartialArrayInfo(offset, stored_size)
                    stored_size_aligned = partial_info.get_aligned_byte_size()
                    assert stored_size_aligned <= remain_size
                    cur_chunk.append(
                        dataclasses.replace(info, partial_info=partial_info))
                    cur_size += stored_size_aligned
                    size -= stored_size_aligned
                    offset += stored_size_aligned
                else:
                    assert remain_size == 0
                    chunks.append(cur_chunk)
                    cur_chunk = []
                    cur_size = 0
        if cur_chunk:
            chunks.append(cur_chunk)
        return chunks

    def _save_chunk_to_shm(self, shm: SharedMemory,
                           chunk: list[SharedArraySegmentDesc],
                           item: KVStoreItemWithArrDesc):
        t = time.time()
        chunk_total_aligned_size = sum(info.get_partial_size_aligned()
                                       for info in chunk)
        first_offset = chunk[0].get_shm_offset_with_partial()
        shm_np_array = np.ndarray([len(shm.buf)],
                                  dtype=np.uint8,
                                  buffer=shm.buf)
        shm_np_array[:chunk_total_aligned_size] = item.buffer[
            first_offset:first_offset + chunk_total_aligned_size]
        # print("save chunk to shm", chunk_total_aligned_size, time.time() - t)
        return time.time() - t

    def _load_chunk_from_shm(self, shm: SharedMemory,
                             chunk: list[SharedArraySegmentDesc],
                             item: KVStoreItemWithArrDesc):
        chunk_total_aligned_size = sum(info.get_partial_size_aligned()
                                       for info in chunk)
        first_offset = chunk[0].get_shm_offset_with_partial()
        shm_np_array = np.ndarray([len(shm.buf)],
                                  dtype=np.uint8,
                                  buffer=shm.buf)
        item.buffer[
            first_offset:first_offset +
            chunk_total_aligned_size] = shm_np_array[:chunk_total_aligned_size]

    async def _save_chunk_to_shm_async(self, shm: SharedMemory,
                                       chunk: list[SharedArraySegmentDesc],
                                       item: KVStoreItemWithArrDesc):
        # use executor here to make sure slow operation won't block async server.
        return await asyncio.get_running_loop().run_in_executor(
            None, self._save_chunk_to_shm, shm, chunk, item)

    async def _load_chunk_from_shm_async(self, shm: SharedMemory,
                                     chunk: list[SharedArraySegmentDesc],
                                     item: KVStoreItemWithArrDesc):
        # use executor here to make sure slow operation won't block async server.
        return await asyncio.get_running_loop().run_in_executor(
            None, self._load_chunk_from_shm, shm, chunk, item)

    @marker.mark_bidirectional_stream
    async def send_stream(self, iter, key: str):
        self._lazy_init_buffer()
        item = self._store[key]
        arr_descs = item.descs
        chunks = self._chunkize_arr_descs(arr_descs, self._shm_size_per_client)
        async with self._acquire_shm_double_buffer() as shm_double_buffer:
            chunk_idx = 0
            send_counts = 0
            recv_cnt = 0
            total_time = 0
            async for client_ping in iter:
                recv_cnt += 1
                cur_send_client_buffer = shm_double_buffer[chunk_idx % 2]
                cur_send_prepare_buffer = shm_double_buffer[(chunk_idx + 1) %
                                                            2]
                if chunk_idx == 0:
                    total_time += await self._save_chunk_to_shm_async(
                        cur_send_client_buffer, chunks[chunk_idx], item)
                chunk_to_send = chunks[chunk_idx]
                chunk_to_send = [
                    dataclasses.replace(
                        info,
                        shm_offset=info.get_shm_offset_with_partial() -
                        chunk_to_send[0].get_shm_offset_with_partial())
                    for info in chunk_to_send
                ]
                send_data = SharedArraySegmentsLimitedDesc(
                    cur_send_client_buffer.name, chunk_to_send, send_counts)
                yield send_data
                send_counts += len(chunks[chunk_idx])
                chunk_idx += 1
                if chunk_idx < len(chunks):
                    total_time += await self._save_chunk_to_shm_async(
                        cur_send_prepare_buffer, chunks[chunk_idx], item)
                else:
                    break
            # print("recv_cnt", recv_cnt)
            yield None
        # print("send stream total copy time", total_time)

    @marker.mark_bidirectional_stream
    async def receive_stream(self,
                             iter,
                             key: str,
                             treespec: Any,
                             client_arr_descs: list[TensorInfo],
                             metadata: Optional[Any] = None,
                             removed_keys: Optional[set[str]] = None):
        try:
            self._lazy_init_buffer()
            need_update = False
            if removed_keys is not None:
                # try to reuse removed shared memory
                reuse_found = False
                removed_keys_copy = removed_keys.copy()
                for reuse_key in removed_keys:
                    if reuse_key in self._store:
                        if self._validate_arr_descs(
                                reuse_key,
                                self._store[reuse_key].descs,
                                client_arr_descs,
                                raise_exc=False):
                            # reuse this key
                            self._rename_key(reuse_key, key)
                            removed_keys_copy.remove(reuse_key)
                            reuse_found = True
                            break
                if removed_keys_copy:
                    for key in removed_keys_copy:
                        self._remove_item(key, )
                if reuse_found or removed_keys_copy:
                    need_update = True

            if key in self._store:
                self._validate_arr_descs(key,
                                         self._store[key].descs,
                                         client_arr_descs,
                                         raise_exc=True)
                item = self._store[key]
            else:
                # create new item
                segment_descs: list[SharedArraySegmentDesc] = []
                offset = 0
                for var_idx, info in enumerate(client_arr_descs):
                    shape, dtype, meta = info.shape, info.dtype, info.meta
                    desc = SharedArraySegmentDesc(shape,
                                                  dtype,
                                                  meta,
                                                  offset,
                                                  var_idx=var_idx)
                    aligned_size = desc.get_aligned_byte_size()
                    offset += aligned_size
                    segment_descs.append(desc)
                total_size = sum(seg.get_aligned_byte_size()
                                 for seg in segment_descs)
                buffer = np.empty([total_size], dtype=np.uint8)
                item = KVStoreItemWithArrDesc(metadata=metadata,
                                              mtime=time.time(),
                                              data=treespec,
                                              buffer=buffer,
                                              descs=segment_descs)
                need_update = True
            arr_descs = item.descs
            self._validate_arr_descs(key,
                                     item.descs,
                                     client_arr_descs,
                                     raise_exc=True)
            async with self._acquire_shm_double_buffer() as shm_double_buffer:
                chunk_idx = 0
                recv_counts = 0
                chunks = self._chunkize_arr_descs(arr_descs,
                                                  self._shm_size_per_client)
                chunk_to_send = chunks[chunk_idx]
                chunk_to_send = [
                    dataclasses.replace(info,
                                        shm_offset=info.shm_offset -
                                        chunk_to_send[0].shm_offset)
                    for info in chunk_to_send
                ]

                send_data = SharedArraySegmentsLimitedDesc(
                    shm_double_buffer[0].name, chunk_to_send, recv_counts)
                recv_counts += len(chunks[chunk_idx])
                chunk_idx += 1
                yield send_data
                async for client_ping in iter:
                    cur_recv_client_buffer = shm_double_buffer[chunk_idx % 2]
                    cur_recv_prepare_buffer = shm_double_buffer[(chunk_idx + 1)
                                                                % 2]
                    if chunk_idx < len(chunks):
                        chunk_to_send = chunks[chunk_idx]
                        chunk_to_send = [
                            dataclasses.replace(
                                info,
                                shm_offset=info.get_shm_offset_with_partial() -
                                chunk_to_send[0].get_shm_offset_with_partial())
                            for info in chunk_to_send
                        ]
                        send_data = SharedArraySegmentsLimitedDesc(
                            cur_recv_client_buffer.name, chunk_to_send,
                            recv_counts)
                        yield send_data
                    await self._load_chunk_from_shm_async(cur_recv_prepare_buffer,
                                              chunks[chunk_idx - 1], item)
                    if chunk_idx >= len(chunks):
                        break
                    recv_counts += len(chunks[chunk_idx])
                    chunk_idx += 1
                yield None
        except:
            traceback.print_exc()
            raise
        # print("STORE ITEM SUCCEED", key, need_update)
        self._store[key] = item
        item.metadata = metadata
        item.mtime = time.time()
        item.data = treespec
        if need_update:
            async with self._lock:
                await self._event_emitter.emit_async(
                    KVStoreEventType.ITEM_CHANGE, self._store)

    def backend_get_event_emitter(self):
        return self._event_emitter

    def backend_get_store(self):
        return self._store

    def has_item(self, key: str) -> bool:
        return key in self._store

    def get_item_treespec(self, key: str):
        return self._store[key].data

    def get_item_arr_desc(self, key: str):
        return self._store[key].descs

    async def set_item_treespec(self, key: str, treespec: Any,
                                arr_descs: list[TensorInfo]):
        # validate value_arr_descs
        if key not in self._store:
            raise ValueError(
                f"{key} not allocated. call `get_or_create_shared_array_segments` first."
            )
        self._validate_arr_descs(key, self._store[key].descs, arr_descs)
        # we assume client already copy data to shared memory. so we only
        # need to store treespec.
        self._store[key].data = treespec

    def get_item_metadata(self, key: str):
        return self._store[key].metadata

    def get_item_shm_size(self, key: str):
        if key not in self._store:
            raise ValueError(f"{key} not allocated.")
        segments = self._store[key]
        return segments.buffer.size

    def get_all_keys(self) -> List[str]:
        return list(self._store.keys())

    def get_all_key_to_meta(self) -> dict[str, Any]:
        return {key: item.metadata for key, item in self._store.items()}

    def _remove_item(self, key: str):
        if key in self._store:
            del self._store[key]

    async def remove_item(self, key: str):
        async with self._lock:
            if key in self._store:
                del self._store[key]
                await self._event_emitter.emit_async(
                    KVStoreEventType.ITEM_CHANGE, self._store)

    async def remove_items(self, keys: List[str]):
        # async with self._lock:
        for key in keys:
            self._remove_item(key)
        await self._event_emitter.emit_async(KVStoreEventType.ITEM_CHANGE,
                                             self._store)

    async def clear(self):
        # async with self._lock:
        for key in list(self._store.keys()):
            self._remove_item(key)
        await self._event_emitter.emit_async(KVStoreEventType.ITEM_CHANGE,
                                             self._store)
