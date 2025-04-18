from pathlib import Path
from typing import Any, Optional
from tensorpc.apps.collections.shm_kvstore import ShmKVStoreTensorClient

from tensorpc.apps.distssh.constants import TENSORPC_ENV_DISTSSH_URL_WITH_PORT
from tensorpc.apps.distssh.typedefs import CheckpointMetadata, CheckpointType
from tensorpc.core.client import RemoteObject
from tensorpc import simple_remote_call
import os 
from tensorpc.core import BuiltinServiceKeys
from tensorpc.core.tree_id import UniqueTreeId

_DISTSSH_URL = os.getenv(TENSORPC_ENV_DISTSSH_URL_WITH_PORT)

def _get_rank_may_distributed():
    import torch.distributed as dist
    return dist.get_rank() if dist.is_initialized() else 0

class TorchDistributedCkptClient(ShmKVStoreTensorClient):
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

    def _store_train_checkpoint(self, is_major: bool, key: str, step: int, state_dict: dict[str, Any]):
        key_to_ckpt_meta = self._get_key_to_ckpt_meta()
        ckpt_type = CheckpointType.TRAIN_MAJOR if is_major else CheckpointType.TRAIN_MINOR
        store_key, rank = self._encode_train_key(key, step)
        if store_key in key_to_ckpt_meta:
            meta = key_to_ckpt_meta[key]
            if meta.type == CheckpointType.FIXED:
                raise ValueError(
                    f"Checkpoint {key} is fixed, not train, use another key."
                )
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
        all_ckpts_list = list(all_ckpts.items())
        all_ckpts_list.sort(key=lambda x: x[0])
        num_ckpt_limit = self._max_major_ckpt if is_major else self._max_minor_ckpt
        store_keys_to_remove: list[str] = []
        while len(all_ckpts_list) >= num_ckpt_limit:
            poped_item = all_ckpts_list.pop(0)
            all_keys_to_remove = [x[0] for x in poped_item[1]]
            store_keys_to_remove.extend(all_keys_to_remove)
        self.remove_items(store_keys_to_remove)
        new_meta = CheckpointMetadata(ckpt_type, key, step, rank)
        print(len(all_ckpts), len(all_ckpts_list), store_key)
        return self.store_tensor_tree(store_key, state_dict, new_meta)
    
    def store_major_checkpoint(self, key: str, step: int, state_dict: dict[str, Any]):
        return self._store_train_checkpoint(True, key, step, state_dict)

    def store_minor_checkpoint(self, key: str, step: int, state_dict: dict[str, Any]):
        return self._store_train_checkpoint(False, key, step, state_dict)

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

    def load_train_checkpoint(self, key: str, step: int, state_dict: dict[str, Any]):
        if not self.has_train_checkpoint(key, step):
            raise ValueError(f"train checkpoint {key}-{step} not found.")
        store_key = self._encode_train_key(key, step)[0]
        return self.load_tensor_tree(store_key, state_dict)

def start_distssh_logging(logdir: str):
    """
    Start logger of distssh.
    """
    assert _DISTSSH_URL is not None, "you must run this in distssh server"
    simple_remote_call(_DISTSSH_URL, f"{BuiltinServiceKeys.FaultToleranceSSHServer.value}.start_logging", logdir)