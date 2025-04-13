"""Client for tensorpc builtin service `ShmKVStore`."""
from typing import Any
from tensorpc.core.asyncclient import AsyncRemoteObject
from tensorpc.core.client import RemoteObject
from tensorpc.core import BuiltinServiceKeys, core_io
import numpy as np 
from tensorpc.services.collection import SharedArraySegment, SharedArraySegmentDesc

class ShmKVStoreClient:
    def __init__(self, robj: RemoteObject):
        self._robj = robj
        self._serv_key = BuiltinServiceKeys.ShmKVStore.value

    def store_array_tree(self, key: str, arr_tree: Any):
        variables, treespec = core_io.extract_arrays_from_data(arr_tree, (np.ndarray,))
        arr_desps = [(a.shape, a.dtype) for a in variables] # type: ignore
        segments: list[SharedArraySegment] = self._robj.remote_call(f"{self._serv_key}.get_or_create_shared_array_segments", key, arr_desps)
        # copy to shm
        for i, a in enumerate(variables):
            segments[i].get_array_view()[:] = a
        # send tree spec
        self._robj.remote_call(f"{self._serv_key}.set_item_treespec", key, treespec, arr_desps)

    def get_array_tree(self, key: str, copy: bool = True):
        treespec = self._robj.remote_call(f"{self._serv_key}.get_item_treespec", key)
        segment_descs: list[SharedArraySegmentDesc] = self._robj.remote_call(f"{self._serv_key}.get_item_segment_descs", key)
        if copy:
            segments = [s.get_segment() for s in segment_descs]
            variables = [s.get_array_view().copy() for s in segments]
            for segment in segments:
                segment.close_in_remote()
        else:
            segments = [s.get_segment() for s in segment_descs]
            variables = [s.get_array_view() for s in segments]
        res = core_io.put_arrays_to_data(variables, treespec)
        return res 

    def remove_array_tree(self, key: str):
        return self.remove_items([key])

    def get_all_keys(self):
        return self._robj.remote_call(f"{self._serv_key}.get_all_keys")

    def remove_items(self, keys: list[str]):
        self._robj.remote_call(f"{self._serv_key}.remove_items", keys)

    def get_shm_size(self, key: str):
        return self._robj.remote_call(f"{self._serv_key}.get_item_shm_size", key)

class ShmKVStoreAsyncClient:
    def __init__(self, robj: AsyncRemoteObject):
        self._robj = robj
        self._serv_key = BuiltinServiceKeys.ShmKVStore.value

    async def store_array_tree(self, key: str, arr_tree: Any):
        variables, treespec = core_io.extract_arrays_from_data(arr_tree, (np.ndarray,))
        arr_desps = [(a.shape, a.dtype) for a in variables] # type: ignore
        segments: list[SharedArraySegment] = await self._robj.remote_call(f"{self._serv_key}.get_or_create_shared_array_segments", key, arr_desps)
        # copy to shm
        for i, a in enumerate(variables):
            segments[i].get_array_view()[:] = a
        # send tree spec
        await self._robj.remote_call(f"{self._serv_key}.set_item_treespec", key, treespec, arr_desps)

    async def get_array_tree(self, key: str, copy: bool = True):
        treespec = await self._robj.remote_call(f"{self._serv_key}.get_item_treespec", key)
        segment_descs: list[SharedArraySegmentDesc] = await self._robj.remote_call(f"{self._serv_key}.get_item_segment_descs", key)
        if copy:
            segments = [s.get_segment() for s in segment_descs]
            variables = [s.get_array_view().copy() for s in segments]
            for segment in segments:
                segment.close_in_remote()
        else:
            segments = [s.get_segment() for s in segment_descs]
            variables = [s.get_array_view() for s in segments]
        res = core_io.put_arrays_to_data(variables, treespec)
        return res 

    async def remove_array_tree(self, key: str):
        return await self.remove_items([key])

    async def get_all_keys(self):
        return await self._robj.remote_call(f"{self._serv_key}.get_all_keys")

    async def remove_items(self, keys: list[str]):
        await self._robj.remote_call(f"{self._serv_key}.remove_items", keys)

    async def get_shm_size(self, key: str):
        return await self._robj.remote_call(f"{self._serv_key}.get_item_shm_size", key)