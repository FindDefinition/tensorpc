"""Client for tensorpc builtin service `ShmKVStore`."""
from collections.abc import Sequence
from contextlib import nullcontext
import queue
import time
from typing import Any, Optional, Union
from tensorpc.core.asyncclient import AsyncRemoteObject
from tensorpc.core.client import RemoteObject
from tensorpc.core import BuiltinServiceKeys, core_io
import numpy as np 
from tensorpc.apps.collections.serv.kvstore import SharedArraySegmentsDesc, SharedArraySegmentDesc, SharedArraySegmentsLimitedDesc, TensorInfo
from tensorpc.protos_export import rpc_message_pb2 as rpc_msg_pb2

class ShmKVStoreClientBase:
    def __init__(self, robj: RemoteObject, serv_key: str = BuiltinServiceKeys.ShmKVStore.value):
        self._robj = robj
        self._serv_key = serv_key

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
        arr_descs = [TensorInfo(a.shape, a.dtype, None) for a in variables] # type: ignore
        segments_desc: SharedArraySegmentsDesc = self._robj.remote_call(f"{self._serv_key}.get_or_create_shared_array_segments", key, arr_descs)
        # copy to shm
        segments = segments_desc.get_segments()
        for i, a in enumerate(variables):
            segments.get_array_view(i)[:] = a
        segments.close_in_remote()
        # send tree spec
        self._robj.remote_call(f"{self._serv_key}.set_item_treespec", key, treespec, arr_descs, metadata, rpc_flags=rpc_msg_pb2.Pickle)

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

_JSON_INDEX_KEY = "__shm_jidx__"

def _extract_tensor_descs(arr_tree: Any):
    import torch
    variables, treespec = core_io.extract_arrays_from_data(arr_tree, (torch.Tensor,), json_index=_JSON_INDEX_KEY)
    new_variables, arr_descs = _extract_tensor_descs_from_variables(variables)
    return new_variables, arr_descs, treespec

def _get_tensor_info_from_tensor(tensor: Any):
    from torch.distributed.tensor import DTensor
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    np_dtype = _torch_dtype_to_np_dtype_size_equal(tensor.dtype)
    th_meta = tensor.dtype 
    return tensor, TensorInfo(list(tensor.shape), np_dtype, th_meta)

def _extract_tensor_descs_from_variables(variables: Any):
    import torch
    from torch.distributed.tensor import DTensor
    arr_descs: list[TensorInfo] = []
    new_variables: list[torch.Tensor] = []
    with torch.no_grad():
        for v in variables:
            assert isinstance(v, torch.Tensor)
            ten, info = _get_tensor_info_from_tensor(v)
            new_variables.append(ten)
            arr_descs.append(info)
    return new_variables, arr_descs

def _validate_load_tensor_tree(arr_descs: list[TensorInfo], segment_descs: Sequence[TensorInfo]):
    if len(arr_descs) != len(segment_descs):
        raise ValueError(f"arr_descs and segment_descs length not match.")
    for i, info in enumerate(arr_descs):
        shape, dtype, meta = info.shape, info.dtype, info.meta            
        seg = segment_descs[i]
        assert seg.meta == meta
        seg_byte_size = np.prod(shape) * np.dtype(dtype).itemsize
        target_byte_size = seg.get_byte_size()
        if seg_byte_size != target_byte_size:
            raise ValueError(f"{info} already allocated with different byte length.")

class ShmKVStoreTensorClient(ShmKVStoreClientBase):
    def store_tensor_tree(self, key: str, arr_tree: Any, metadata: Optional[Any] = None, removed_keys: Optional[set[str]] = None, stream: Optional[Any] = None):
        import torch
        from torch.distributed.tensor import DTensor
        variables, arr_descs, treespec = _extract_tensor_descs(arr_tree)
        segments_desc: SharedArraySegmentsDesc = self._robj.remote_call(f"{self._serv_key}.get_or_create_shared_array_segments", key, arr_descs, removed_keys)
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
                        meta = arr_descs[i].meta
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
        self._robj.remote_call(f"{self._serv_key}.set_item_treespec", key, treespec, arr_descs, metadata, rpc_flags=rpc_msg_pb2.Pickle)

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

    def load_tensor_tree(self, key: str, arr_tree: Any):
        import torch
        variables, arr_descs, treespec = _extract_tensor_descs(arr_tree)
        segments_desc: SharedArraySegmentsDesc = self._robj.remote_call(f"{self._serv_key}.get_item_segment_descs", key)
        _validate_load_tensor_tree(arr_descs, segments_desc.descs)
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


class ShmTrOnlyKVStoreTensorClient(ShmKVStoreClientBase):
    """Shm KVStore client for tensor only. 

    This client only use shm for transfer tensor data, weights are stored in regular memory.
    """
    def __init__(self, robj: RemoteObject, serv_key: str = BuiltinServiceKeys.ShmTrOnlyKVStore.value):
        super().__init__(robj, serv_key)

    def _bistream_q(self, q: queue.Queue, timeout: int = 40):
        while True:
            item = q.get(timeout=timeout)
            if item is None:
                break
            yield item

    def store_tensor_tree(self, key: str, arr_tree: Any, metadata: Optional[Any] = None, removed_keys: Optional[set[str]] = None, stream: Optional[Any] = None):
        import torch
        from torch.distributed.tensor import DTensor
        variables, arr_descs, treespec = _extract_tensor_descs(arr_tree)

        q = queue.Queue()
        with torch.no_grad():
            for data in self._robj.bi_stream(f"{self._serv_key}.receive_stream", self._bistream_q(q), key, treespec, arr_descs, metadata, removed_keys):
                if data is None:
                    q.put(None)
                    continue 
                assert isinstance(data, SharedArraySegmentsLimitedDesc)
                segments = data.get_segments()
                for i, desc in enumerate(segments.descs):
                    a = variables[desc.var_idx]
                    assert isinstance(a, torch.Tensor)
                    if isinstance(a, DTensor):
                        a = a.to_local()
                    partial_info = desc.partial_info
                    # TODO support non-contiguous tensor
                    assert a.is_contiguous()
                    assert partial_info is not None 
                    a_partial = a.view(-1).view(torch.uint8)[partial_info.start:partial_info.start + partial_info.length]
                    arr_view_raw = segments.get_array_view_raw(i)
                    s_th = torch.from_numpy(arr_view_raw)
                    s_th.copy_(a_partial)
                torch.cuda.synchronize()
                segments.close_in_remote()
                q.put(0)

    def get_tensor_tree(self, key: str, device: Optional[Any] = None):
        import torch
        treespec = self._robj.remote_call(f"{self._serv_key}.get_item_treespec", key, rpc_flags=rpc_msg_pb2.Pickle)
        variables = []
        q = queue.Queue()
        q.put(0)
        q_put_cnt = 1
        with torch.no_grad():
            cur_partial_tensor: Optional[torch.Tensor] = None
            cur_partial_tensor_view: Optional[torch.Tensor] = None
            total_time = 0
            for data in self._robj.bi_stream(f"{self._serv_key}.send_stream", self._bistream_q(q), key):
                if data is None:
                    q.put(None)
                    continue 
                    # break
                t = time.time()
                assert isinstance(data, SharedArraySegmentsLimitedDesc)
                segments = data.get_segments()
                for i, segment_desc in enumerate(segments.descs):
                    assert segment_desc.meta is not None 
                    data_raw = segments.get_array_view_raw(i)
                    partial_info = segment_desc.get_partial_info_checked()
                    if partial_info.start != 0:
                        assert cur_partial_tensor_view is not None
                    else:
                        cur_partial_tensor = torch.empty(segment_desc.shape, dtype=segment_desc.meta, device=device)
                        cur_partial_tensor_view = cur_partial_tensor.view(-1).view(torch.uint8)
                    a_partial = cur_partial_tensor_view[partial_info.start:partial_info.start + partial_info.length]
                    a_partial.copy_(torch.from_numpy(data_raw))
                    if segment_desc.is_last_partial():
                        variables.append(cur_partial_tensor)
                        cur_partial_tensor = None
                        cur_partial_tensor_view = None
                torch.cuda.synchronize()
                segments.close_in_remote()
                q.put(0)
                q_put_cnt += 1
                total_time += time.time() - t
        # print("client total time", total_time)
        # print("q_put_cnt", q_put_cnt)
        res = core_io.put_arrays_to_data(variables, treespec, _JSON_INDEX_KEY)
        return res 

    def _path_to_str(self, path: tuple[Any, ...]):
        from torch.utils import _pytree as pytree
        path_str = []
        for p in path:
            if isinstance(p, pytree.MappingKey):
                path_str.append(f"{p.key}")
            elif isinstance(p, pytree.SequenceKey):
                path_str.append(f"{p.idx}")
            else:
                raise ValueError(f"Unknown path type {type(p)}")
        return ".".join(path_str)

    def load_tensor_tree(self, key: str, arr_tree: Any, strict=True, is_rank0: bool = True):
        import torch
        from torch.utils import _pytree as pytree
        variables: list[Any] = []
        stored_index_need_load_to_tensor = {}
        stored_index_need_load_to_path = {}
        tensorpc_treespec = self.get_item_tree_spec(key)

        if strict:
            variables, arr_descs, treespec = _extract_tensor_descs(arr_tree)
            store_arr_descs: list[SharedArraySegmentDesc] = self._robj.remote_call(f"{self._serv_key}.get_item_arr_desc", key)
            _validate_load_tensor_tree(arr_descs, store_arr_descs)
        else:
            # we use {_JSON_INDEX_KEY: ...} as leaf node, convert to pytree spec
            path_with_var, _ = pytree.tree_flatten_with_path(tensorpc_treespec, is_leaf=lambda x: isinstance(x, dict) and _JSON_INDEX_KEY in x)
            path_with_var_to_load, _ = pytree.tree_flatten_with_path(arr_tree, is_leaf=lambda x: isinstance(x, torch.Tensor))
            path_with_var_dict = dict(path_with_var)
            saved_tensor_paths: set[Any] = set()
            unexpected_paths: set[Any] = set()
            for path, info in path_with_var_dict.items():
                if isinstance(info, dict) and _JSON_INDEX_KEY in info:
                    saved_tensor_paths.add(path)
            path_with_var_to_load_dict = dict(path_with_var_to_load)
            tensor_count = 0
            for path, ten in path_with_var_to_load_dict.items():
                if isinstance(ten, torch.Tensor):
                    tensor_count += 1
            load_arr_descs: list[TensorInfo] = []
            store_arr_descs_need_load: list[TensorInfo] = []
            store_arr_descs: list[SharedArraySegmentDesc] = self._robj.remote_call(f"{self._serv_key}.get_item_arr_desc", key)
            for path, ten in path_with_var_to_load_dict.items():
                if isinstance(ten, torch.Tensor):
                    if path in path_with_var_dict:
                        index = path_with_var_dict[path][_JSON_INDEX_KEY][0]
                        store_arr_descs_need_load.append(store_arr_descs[index])
                        ten, info = _get_tensor_info_from_tensor(ten)
                        load_arr_descs.append(info)
                        stored_index_need_load_to_tensor[index] = ten
                        stored_index_need_load_to_path[index] = path
                        saved_tensor_paths.remove(path)
                    else:
                        unexpected_paths.add(path)
            _validate_load_tensor_tree(load_arr_descs, store_arr_descs_need_load)
            if is_rank0:
                if saved_tensor_paths:
                    print("------ Missing Keys ------")
                    for path in saved_tensor_paths:
                        print(self._path_to_str(path))
                if unexpected_paths:
                    print("------ Unexpected Keys ------")
                    for path in unexpected_paths:
                        print(self._path_to_str(path))

        q = queue.Queue()
        q.put(0)
        with torch.no_grad():
            for data in self._robj.bi_stream(f"{self._serv_key}.send_stream", self._bistream_q(q), key):
                if data is None:
                    q.put(None)
                    continue 
                assert isinstance(data, SharedArraySegmentsLimitedDesc)
                segments = data.get_segments()

                for i, segment_desc in enumerate(segments.descs):
                    if strict:
                        v = variables[segment_desc.var_idx]
                    else:
                        stored_var_idx = segment_desc.var_idx
                        if stored_var_idx not in stored_index_need_load_to_tensor:
                            continue
                        v = stored_index_need_load_to_tensor[segment_desc.var_idx]
                    arr_view_raw = segments.get_array_view_raw(i)
                    assert isinstance(v, torch.Tensor)
                    partial_info = segment_desc.partial_info
                    # TODO support non-contiguous tensor
                    assert v.is_contiguous()
                    assert partial_info is not None 
                    v_partial = v.view(-1).view(torch.uint8)[partial_info.start:partial_info.start + partial_info.length]
                    s_th = torch.from_numpy(arr_view_raw)
                    v_partial.copy_(s_th)
                torch.cuda.synchronize()
                segments.close_in_remote()
                q.put(0)
        return tensorpc_treespec


class ShmKVStoreAsyncClient:
    def __init__(self, robj: AsyncRemoteObject):
        self._robj = robj
        self._serv_key = BuiltinServiceKeys.ShmKVStore.value

    async def get_item_type(self, key: str):
        return await self._robj.remote_call(f"{self._serv_key}.get_item_type", key)

    async def store_array_tree(self, key: str, arr_tree: Any, metadata: Optional[Any] = None):
        variables, treespec = core_io.extract_arrays_from_data(arr_tree, (np.ndarray,))
        arr_descs = [TensorInfo(a.shape, a.dtype, None) for a in variables] # type: ignore
        segments_desc: SharedArraySegmentsDesc = await self._robj.remote_call(f"{self._serv_key}.get_or_create_shared_array_segments", key, arr_descs)
        # copy to shm
        segments = segments_desc.get_segments()
        for i, a in enumerate(variables):
            segments.get_array_view(i)[:] = a
        segments.close_in_remote()
        # send tree spec
        await self._robj.remote_call(f"{self._serv_key}.set_item_treespec", key, treespec, arr_descs, metadata, rpc_flags=rpc_msg_pb2.Pickle)

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

