"""Client for tensorpc builtin service `ShmKVStore`."""
from contextlib import nullcontext
import time
from typing import Any, Optional, Union
from tensorpc.core.asyncclient import AsyncRemoteObject
from tensorpc.core.client import RemoteObject
from tensorpc.core import BuiltinServiceKeys, core_io
import numpy as np 
from tensorpc.apps.collections.serv.kvstore import SharedArraySegmentsDesc, SharedArraySegmentDesc, TensorInfo
from tensorpc.protos_export import rpc_message_pb2 as rpc_msg_pb2

class ShmKVStoreClientBase:
    def __init__(self, robj: RemoteObject):
        self._robj = robj
        self._serv_key = BuiltinServiceKeys.ShmKVStore.value

    def remove_array_tree(self, key: str):
        return self.remove_items([key])

    def get_all_keys(self):
        return self._robj.remote_call(f"{self._serv_key}.get_all_keys")
    
    def get_all_key_to_meta(self):
        return self._robj.remote_call(f"{self._serv_key}.get_all_key_to_meta")

    def remove_items(self, keys: list[str]):
        self._robj.remote_call(f"{self._serv_key}.remove_items", keys)

    def get_shm_size(self, key: str):
        return self._robj.remote_call(f"{self._serv_key}.get_item_shm_size", key)

    def has_item(self, key: str) -> bool:
        return self._robj.remote_call(f"{self._serv_key}.has_item", key)
    
    def get_item_type(self, key: str):
        return self._robj.remote_call(f"{self._serv_key}.get_item_type", key)

    def get_item_tree_spec(self, key: str):
        return self._robj.remote_call(f"{self._serv_key}.get_item_treespec", key, rpc_flags=rpc_msg_pb2.Pickle)

class ShmKVStoreClient(ShmKVStoreClientBase):

    def store_array_tree(self, key: str, arr_tree: Any, metadata: Optional[Any] = None):
        variables, treespec = core_io.extract_arrays_from_data(arr_tree, (np.ndarray,))
        arr_desps = [TensorInfo(a.shape, a.dtype, None) for a in variables] # type: ignore
        segments_desc: SharedArraySegmentsDesc = self._robj.remote_call(f"{self._serv_key}.get_or_create_shared_array_segments", key, arr_desps)
        # copy to shm
        segments = segments_desc.get_segments()
        for i, a in enumerate(variables):
            segments.get_array_view(i)[:] = a
        segments.close_in_remote()
        # send tree spec
        self._robj.remote_call(f"{self._serv_key}.set_item_treespec", key, treespec, arr_desps, metadata, rpc_flags=rpc_msg_pb2.Pickle)

    def get_array_tree(self, key: str, copy: bool = True):
        treespec = self._robj.remote_call(f"{self._serv_key}.get_item_treespec", key, rpc_flags=rpc_msg_pb2.Pickle)
        segments_desc: SharedArraySegmentsDesc = self._robj.remote_call(f"{self._serv_key}.get_item_segment_descs", key)
        if copy:
            segments = segments_desc.get_segments()
            variables = [segments.get_array_view(i).copy() for i in range(len(segments))]
            segments.close_in_remote()
        else:
            segments = segments_desc.get_segments()
            variables = [segments.get_array_view(i) for i in range(len(segments))]
            segments.close_in_remote()
        res = core_io.put_arrays_to_data(variables, treespec)
        return res 

_ITEMSIZE_TO_NP_DTYPE = {
    1: np.dtype(np.uint8),
    2: np.dtype(np.uint16),
    4: np.dtype(np.uint32),
    8: np.dtype(np.uint64),
}

def _torch_dtype_to_np_dtype_size_equal(th_dtype):
    # when we store torch tensor, we only need dtype with same item size, since some
    # torch dtype isn't supported by numpy.
    return _ITEMSIZE_TO_NP_DTYPE[th_dtype.itemsize]

class ShmKVStoreTensorClient(ShmKVStoreClientBase):
    def _extract_tensor_desps(self, arr_tree: Any):
        import torch
        from torch.distributed.tensor import DTensor
        variables, treespec = core_io.extract_arrays_from_data(arr_tree, (torch.Tensor,), json_index=True)
        arr_desps: list[TensorInfo] = []
        new_variables: list[torch.Tensor] = []
        with torch.no_grad():
            for v in variables:
                assert isinstance(v, torch.Tensor)
                if isinstance(v, DTensor):
                    v = v.to_local()
                new_variables.append(v)
                np_dtype = _torch_dtype_to_np_dtype_size_equal(v.dtype)
                th_meta = v.dtype 
                arr_desps.append(TensorInfo(list(v.shape), np_dtype, th_meta))
        return new_variables, arr_desps, treespec

    def store_tensor_tree(self, key: str, arr_tree: Any, metadata: Optional[Any] = None, removed_keys: Optional[set[str]] = None, stream: Optional[Any] = None):
        import torch
        from torch.distributed.tensor import DTensor
        variables, arr_desps, treespec = self._extract_tensor_desps(arr_tree)
        segments_desc: SharedArraySegmentsDesc = self._robj.remote_call(f"{self._serv_key}.get_or_create_shared_array_segments", key, arr_desps, removed_keys)
        segments = segments_desc.get_segments()
        # copy to shm
        try:
            with torch.no_grad():
                if stream is not None:
                    stream_ctx = torch.cuda.stream(stream)
                else:
                    stream_ctx = nullcontext()
                with stream_ctx:
                    # assume user will wait this stream before next optim step
                    for i, a in enumerate(variables):
                        assert isinstance(a, torch.Tensor)
                        if isinstance(a, DTensor):
                            a = a.to_local()
                        meta = arr_desps[i].meta
                        assert meta is not None 
                        s_np = segments.get_array_view(i)
                        s_th = torch.from_numpy(s_np).view(meta)
                        s_th.copy_(a)
            # TODO replace sync with better option
            if stream is None:
                torch.cuda.synchronize()
        finally:
            segments.close_in_remote()
        # send tree spec
        self._robj.remote_call(f"{self._serv_key}.set_item_treespec", key, treespec, arr_desps, metadata, rpc_flags=rpc_msg_pb2.Pickle)

    def get_tensor_tree(self, key: str, device: Optional[Any] = None):
        import torch
        treespec = self._robj.remote_call(f"{self._serv_key}.get_item_treespec", key, rpc_flags=rpc_msg_pb2.Pickle)
        segments_desc: SharedArraySegmentsDesc = self._robj.remote_call(f"{self._serv_key}.get_item_segment_descs", key)
        variables = []
        total_byte_size = 0
        segments = segments_desc.get_segments()
        try:
            for i, segment_desc in enumerate(segments.descs):
                s_np = segments.get_array_view(i)
                assert segment_desc.meta is not None 
                if device is not None:
                    s_th = torch.from_numpy(s_np).view(segment_desc.meta).to(device)
                else:
                    s_th = torch.from_numpy(s_np).view(segment_desc.meta).clone()
                total_byte_size += s_th.numel() * s_th.element_size()
                variables.append(s_th)
        finally:
            torch.cuda.synchronize()
            segments.close_in_remote()
        res = core_io.put_arrays_to_data(variables, treespec)
        return res 

    def _validate_load_tensor_tree(self, arr_desps: list[TensorInfo], segment_descs: list[SharedArraySegmentDesc]):
        if len(arr_desps) != len(segment_descs):
            raise ValueError(f"arr_desps and segment_descs length not match.")
        for i, info in enumerate(arr_desps):
            shape, dtype, meta = info.shape, info.dtype, info.meta            
            seg = segment_descs[i]
            assert seg.meta == meta
            seg_byte_size = np.prod(shape) * np.dtype(dtype).itemsize
            target_byte_size = seg.get_byte_size()
            if seg_byte_size != target_byte_size:
                raise ValueError(f"{info} already allocated with different byte length.")

    def load_tensor_tree(self, key: str, arr_tree: Any):
        import torch
        variables, arr_desps, treespec = self._extract_tensor_desps(arr_tree)
        segments_desc: SharedArraySegmentsDesc = self._robj.remote_call(f"{self._serv_key}.get_item_segment_descs", key)
        self._validate_load_tensor_tree(arr_desps, segments_desc.descs)
        segments = segments_desc.get_segments()
        try:
            with torch.no_grad():
                for i, segment_desc in enumerate(segments.descs):
                    s_np = segments.get_array_view(i)
                    assert segment_desc.meta is not None 
                    s_th = torch.from_numpy(s_np).view(segment_desc.meta)
                    v = variables[i]
                    assert isinstance(v, torch.Tensor)
                    v.copy_(s_th)
            # TODO replace sync with better option
            torch.cuda.synchronize()
        finally:
            segments.close_in_remote()
        return self.get_item_tree_spec(key)

class ShmKVStoreAsyncClient:
    def __init__(self, robj: AsyncRemoteObject):
        self._robj = robj
        self._serv_key = BuiltinServiceKeys.ShmKVStore.value

    async def get_item_type(self, key: str):
        return await self._robj.remote_call(f"{self._serv_key}.get_item_type", key)

    async def store_array_tree(self, key: str, arr_tree: Any, metadata: Optional[Any] = None):
        variables, treespec = core_io.extract_arrays_from_data(arr_tree, (np.ndarray,))
        arr_desps = [TensorInfo(a.shape, a.dtype, None) for a in variables] # type: ignore
        segments_desc: SharedArraySegmentsDesc = await self._robj.remote_call(f"{self._serv_key}.get_or_create_shared_array_segments", key, arr_desps)
        # copy to shm
        segments = segments_desc.get_segments()
        for i, a in enumerate(variables):
            segments.get_array_view(i)[:] = a
        segments.close_in_remote()
        # send tree spec
        await self._robj.remote_call(f"{self._serv_key}.set_item_treespec", key, treespec, arr_desps, metadata, rpc_flags=rpc_msg_pb2.Pickle)

    async def get_array_tree(self, key: str, copy: bool = True):
        treespec = self._robj.remote_call(f"{self._serv_key}.get_item_treespec", key, rpc_flags=rpc_msg_pb2.Pickle)
        segments_desc: SharedArraySegmentsDesc = await self._robj.remote_call(f"{self._serv_key}.get_item_segment_descs", key)
        if copy:
            segments = segments_desc.get_segments()
            variables = [segments.get_array_view(i).copy() for i in range(len(segments))]
            segments.close_in_remote()
        else:
            segments = segments_desc.get_segments()
            variables = [segments.get_array_view(i) for i in range(len(segments))]
            segments.close_in_remote()
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

    async def has_item(self, key: str) -> bool:
        return await self._robj.remote_call(f"{self._serv_key}.has_item", key)

