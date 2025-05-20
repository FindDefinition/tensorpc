import contextlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Optional, Union
from tensorpc.apps.collections.shm_kvstore import ShmKVStoreTensorClient, ShmTrOnlyKVStoreTensorClient

from tensorpc.apps.distssh.constants import TENSORPC_ENV_DISTSSH_URL_WITH_PORT
from tensorpc.apps.distssh.typedefs import CheckpointMetadata, CheckpointType
from tensorpc.core.client import RemoteObject, simple_chunk_call
from tensorpc import simple_remote_call
import os 
from tensorpc.core import BuiltinServiceKeys
from tensorpc.core.tree_id import UniqueTreeId
import traceback
from tensorpc.apps.dbg.bkpt import breakpoint, init, force_stop_trace
from tensorpc.core.bgserver import BACKGROUND_SERVER

_DISTSSH_URL = os.getenv(TENSORPC_ENV_DISTSSH_URL_WITH_PORT)

def _get_rank_may_distributed():
    import torch.distributed as dist
    return dist.get_rank() if dist.is_initialized() else 0

class TorchDistributedCkptClient(ShmTrOnlyKVStoreTensorClient):
    def __init__(self, robj: RemoteObject, max_major_ckpt: int, max_minor_ckpt: int, replicate_size: int = -1):
        super().__init__(robj)
        self._max_major_ckpt = max_major_ckpt
        self._max_minor_ckpt = max_minor_ckpt
        self._replicate_size = replicate_size

    def _get_key_to_ckpt_meta(self):
        """
        Get all checkpoint metadata from the remote object.
        """
        key_to_meta: dict[str, Any] = self.get_all_key_to_meta()
        
        key_to_ckpt_meta: dict[str, CheckpointMetadata] = {}
        for k, v in key_to_meta.items():
            if isinstance(v, CheckpointMetadata):
                key_to_ckpt_meta[k] = v
        return key_to_ckpt_meta

    def has_fixed_checkpoint(self, key: str):
        key_to_ckpt_meta = self._get_key_to_ckpt_meta()
        if key in key_to_ckpt_meta:
            meta = key_to_ckpt_meta[key]
            if meta.type == CheckpointType.FIXED:
                return True
        return False

    def has_train_checkpoint(self, key: str, step: int):
        key_to_ckpt_meta = self._get_key_to_ckpt_meta()
        store_key, rank = self._encode_train_key(key, step)
        if store_key in key_to_ckpt_meta:
            meta = key_to_ckpt_meta[store_key]
            if meta.type != CheckpointType.FIXED and meta.step == step:
                return True
        return False

    def _encode_train_key(self, key: str, step: int):
        rank = _get_rank_may_distributed()
        new_store_key = UniqueTreeId.from_parts([key, str(step), str(rank)]).uid_encoded
        return new_store_key, rank

    def _store_train_checkpoint(self, is_major: bool, key: str, step: int, state_dict: dict[str, Any], stream: Optional[Any] = None):
        key_to_ckpt_meta = self._get_key_to_ckpt_meta()
        ckpt_type = CheckpointType.TRAIN_MAJOR if is_major else CheckpointType.TRAIN_MINOR
        store_key, rank = self._encode_train_key(key, step)
        if store_key in key_to_ckpt_meta:
            meta = key_to_ckpt_meta[store_key]
            if meta.type == CheckpointType.FIXED:
                raise ValueError(
                    f"Checkpoint {store_key} is fixed, not train, use another key."
                )
        all_ckpts: dict[int, list[tuple[str, CheckpointMetadata]]] = {}
        for k, v in key_to_ckpt_meta.items():
            if v.key == key and v.type == ckpt_type and v.rank == rank:
                if v.step is not None:
                    cur_step = v.step
                else:
                    cur_step = -1
                if cur_step not in all_ckpts:
                    all_ckpts[cur_step] = []
                all_ckpts[cur_step].append((k, v))
        all_ckpts_list = list(all_ckpts.items())
        all_ckpts_list.sort(key=lambda x: x[0])
        num_ckpt_limit = self._max_major_ckpt if is_major else self._max_minor_ckpt
        store_keys_to_remove: list[str] = []
        while len(all_ckpts_list) >= num_ckpt_limit:
            poped_item = all_ckpts_list.pop(0)
            all_keys_to_remove = [x[0] for x in poped_item[1]]
            store_keys_to_remove.extend(all_keys_to_remove)
        new_meta = CheckpointMetadata(ckpt_type, key, step, rank)
        return self.store_tensor_tree(store_key, state_dict, new_meta, removed_keys=set(store_keys_to_remove), stream=stream)
    
    def store_major_checkpoint(self, key: str, step: int, state_dict: dict[str, Any]):
        return self._store_train_checkpoint(True, key, step, state_dict)

    def store_minor_checkpoint(self, key: str, step: int, state_dict: dict[str, Any], stream: Optional[Any] = None):
        # stream only available for minor (flash) checkpoint.
        return self._store_train_checkpoint(False, key, step, state_dict, stream=stream)

    def store_fixed_checkpoint(self, key: str, state_dict: dict):
        return self.store_tensor_tree(key, state_dict, CheckpointMetadata(CheckpointType.FIXED, key))

    def get_fixed_checkpoint(self, key: str, device: Optional[Any] = None):
        if not self.has_fixed_checkpoint(key):
            raise ValueError(f"Fixed checkpoint {key} not found.")
        return self.get_tensor_tree(key, device=device)

    def get_train_checkpoint(self, key: str, step: int, device: Optional[Any] = None):
        if not self.has_train_checkpoint(key, step):
            raise ValueError(f"train checkpoint {key}-{step} not found.")
        store_key = self._encode_train_key(key, step)[0]
        return self.get_tensor_tree(store_key, device=device)

    def load_train_checkpoint(self, key: str, step: int, state_dict: dict[str, Any], strict: bool = True):
        if not self.has_train_checkpoint(key, step):
            raise ValueError(f"train checkpoint {key}-{step} not found.")
        store_key = self._encode_train_key(key, step)[0]
        rank = _get_rank_may_distributed()

        return self.load_tensor_tree(store_key, state_dict,  strict=strict, is_rank0=rank == 0)

    def get_all_train_checkpoint_metas(self,  key: str):
        rank = _get_rank_may_distributed()
        key_to_ckpt_meta = self._get_key_to_ckpt_meta()
        all_ckpts: dict[int, list[tuple[str, CheckpointMetadata]]] = {}
        for k, v in key_to_ckpt_meta.items():
            if v.key == key and v.type != CheckpointType.FIXED and v.rank == rank:
                if v.step is not None:
                    cur_step = v.step
                else:
                    cur_step = -1
                if cur_step not in all_ckpts:
                    all_ckpts[cur_step] = []
                all_ckpts[cur_step].append((k, v))
        return all_ckpts

def start_distssh_logging(logdir: str):
    """
    Start logger of distssh.
    """
    assert _DISTSSH_URL is not None, "you must run this in distssh server"
    simple_remote_call(_DISTSSH_URL, f"{BuiltinServiceKeys.FaultToleranceSSHServer.value}.start_logging", logdir)

def pth_control_point(*, _frame_cnt: int = 2):
    import torch.distributed as dist
    url_with_port = os.environ.get(TENSORPC_ENV_DISTSSH_URL_WITH_PORT)
    if url_with_port is None:
        raise ValueError("You must use pth_control_point inside distssh.")
    if not dist.is_initialized():
        raise RuntimeError(
            "You must use pth_control_point inside a pytorch distributed process group."
        )
    global_rank = dist.get_rank()
    should_enter_breakpoint = False 
    if global_rank == 0:
        try:
            should_enter_breakpoint = simple_remote_call(
                url_with_port, BuiltinServiceKeys.FaultToleranceSSHServer.value + ".is_user_control_enabled"
            )
        except:
            # server may not prepared yet, ignore this control.
            traceback.print_exc()
            should_enter_breakpoint = False
    # broadcast should_enter_breakpoint to all rank
    world_size = dist.get_world_size()
    obj_list = [should_enter_breakpoint] * world_size
    dist.broadcast_object_list(obj_list, src=0)
    should_enter_breakpoint = obj_list[global_rank]
    if not should_enter_breakpoint:
        # tell dbg server disable all running traces.
        # trace result won't be saved.
        if not BACKGROUND_SERVER.is_started:
            return 
        force_stop_trace()
        return 
    init()
    res = breakpoint(_frame_cnt=_frame_cnt)
    # we must sync bkpt res from rank 0 to avoid inconsistent state.
    obj_list = [res] * world_size
    dist.broadcast_object_list(obj_list, src=0)
    return res 

def is_inside_distssh():
    """
    Check if the current process is inside distssh.
    """
    return _DISTSSH_URL is not None


class PerfMonitorClient:
    def __init__(self):
        self._buffer = []

        self._base_ts = 0
        self._cur_ts = time.time_ns()

    def _check_valid_name(self, name: str):
        # name must be unique in buffer
        return 
        # for item in self._buffer:
        #     if item["name"] == name:
        #         raise ValueError(f"PerfMonitorClient: {name} already exists.")

    def record(self, name: str, enable: bool = True):
        if not enable:
            return 
        self._check_valid_name(name)
        cur_ts = time.time_ns() - self._base_ts
        res = {
            "name": name,
            "pid": 0,
            "tid": 0,
            "ph": "X",
            "ts": self._cur_ts,
            "dur": cur_ts - self._cur_ts,
        }
        cur_ts = time.time_ns() - self._base_ts
        self._buffer.append(res)
        self._cur_ts = cur_ts
        return 

    def extend_external_events(self, events: list[dict]):
        for event in events:
            self._check_valid_name(event["name"])
            self._buffer.append(event)
        self._cur_ts = time.time_ns()
        # self._base_ts = time.time_ns()
        self._base_ts = 0
        return

    @contextlib.contextmanager
    def duration(self, name: str):
        self._check_valid_name(name)
        self._cur_ts = time.time_ns() - self._base_ts
        try:
            yield 
        finally:
            self.record(name)

    def flush_allgather(self, step: int, enable: bool = True, metadata: Any = None, scale: Optional[float] = None):
        if enable:
            if metadata is not None:
                json.dumps(metadata) # validate metadata (must be a json serializable object)
            self.allgather_set_perf_monitor_data(step, self._buffer, scale, metadata)
        self._buffer = []
        self._cur_ts = time.time_ns()
        # self._base_ts = time.time_ns()
        self._base_ts = 0
        return

    def allgather_set_perf_monitor_data(self, step: int, data: list[dict], scale: Optional[float] = None, metadata: Any = None):
        url_with_port = os.environ.get(TENSORPC_ENV_DISTSSH_URL_WITH_PORT)
        if url_with_port is None:
            return False
        import torch.distributed as dist
        if not dist.is_initialized():
            raise RuntimeError(
                "You must use pth_control_point inside a pytorch distributed process group."
            )
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        obj_list = [(data, metadata)] * world_size
        dist.all_gather_object(obj_list, (data, metadata))
        data_list = [x[0] for x in obj_list]
        metadata_list = [x[1] for x in obj_list]
        if global_rank == 0:
            try:
                simple_chunk_call(
                    url_with_port, BuiltinServiceKeys.FaultToleranceSSHServer.value + ".set_perf_data", step, data_list,
                    metadata_list, scale
                )
            except:
                traceback.print_exc()
                return 
