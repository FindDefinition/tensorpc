from typing import Any, Optional, TypeVar, Union
from typing_extensions import overload
import uuid
from .base import DataHandle, ExecutorRemoteDesc

_T = TypeVar("_T")

class DataHandleManager:
    """Manage current handle states during flow scheduling.
    """
    def __init__(self, node_inputs: dict[str, dict[str, Any]]):
        self._node_id_to_input_handles: dict[str, dict[str, DataHandle]] = self._node_inputs_to_handle(node_inputs)

    def get_current_node_inputs(self):
        return self._node_id_to_input_handles

    def get_current_node_ids(self):
        return list(self._node_id_to_input_handles.keys())

    def has_node_inputs(self, node_id: str):
        return node_id in self._node_id_to_input_handles

    @overload
    def get_node_inputs(self, node_id: str) -> dict[str, DataHandle]: ...

    @overload
    def get_node_inputs(self, node_id: str, default: _T) -> Union[dict[str, DataHandle], _T]: ...

    def get_node_inputs(self, node_id: str, default=None):
        return self._node_id_to_input_handles.get(node_id, default)

    def _node_inputs_to_handle(self, node_inputs: dict[str, dict[str, Any]]):
        res: dict[str, dict[str, DataHandle]] = {}
        for node_id, inp in node_inputs.items():
            inp_handles = {}
            for k, v in inp.items():
                if not isinstance(v, DataHandle):
                    uid = uuid.uuid4().hex
                    inp_handles[k] = DataHandle(f"{node_id}-{uid}", ExecutorRemoteDesc.get_empty(), data=v)
                else:
                    inp_handles[k] = v
            res[node_id] = inp_handles
        return res

    async def merge_new_inputs_shallow(self, node_inputs: dict[str, dict[str, Any]]):
        node_handles = self._node_inputs_to_handle(node_inputs)
        handle_to_release: list[DataHandle] = []
        handle_id_to_add: set[str] = set()

        for node_id, node_inp_dict in node_handles.items():
            if node_id in self._node_id_to_input_handles:
                prev_inputs = self._node_id_to_input_handles[node_id]
                for prev_v in prev_inputs.values():
                    handle_to_release.append(prev_v)
            for v in node_inp_dict.values():
                handle_id_to_add.add(v.id)
            self._node_id_to_input_handles[node_id] = node_inp_dict
        await self._release_handles(handle_to_release)

    async def _release_handles(self, handle_to_release: list[DataHandle]):
        all_handle_ids = set()
        for node_id, node_inp_dict in self._node_id_to_input_handles.items():
            for v in node_inp_dict.values():
                all_handle_ids.add(v.id)
        handle_released: set[str] = set()
        for h in handle_to_release:
            if h.id not in all_handle_ids and h.id not in handle_released:
                await h.release()

    async def set_new_inputs(self, node_inputs: dict[str, dict[str, Any]]):
        node_handles = self._node_inputs_to_handle(node_inputs)
        handle_to_release: list[DataHandle] = []
        handle_id_to_add: set[str] = set()

        for node_id, node_inp_dict in node_handles.items():
            for v in node_inp_dict.values():
                handle_id_to_add.add(v.id)
        for node_id, node_inp_dict in self._node_id_to_input_handles.items():
            for v in node_inp_dict.values():
                handle_to_release.append(v)
        self._node_id_to_input_handles = node_handles
        await self._release_handles(handle_to_release)

    async def merge_new_inputs(self, node_inputs: dict[str, dict[str, Any]], handle_to_release: Optional[list[DataHandle]] = None):
        node_handles = self._node_inputs_to_handle(node_inputs)
        if handle_to_release is None:
            handle_to_release = []
        handle_id_to_add: set[str] = set()

        for node_id, node_inp_dict in node_handles.items():
            if node_id in self._node_id_to_input_handles:
                prev_inputs = self._node_id_to_input_handles[node_id]
                for cur_k, cur_v in node_inp_dict.items():
                    if cur_k in prev_inputs:
                        handle_to_release.append(prev_inputs[cur_k])
                prev_inputs = prev_inputs.copy()
                prev_inputs.update(node_inp_dict)
                self._node_id_to_input_handles[node_id] = prev_inputs
            else:
                for v in node_inp_dict.values():
                    handle_id_to_add.add(v.id)
                self._node_id_to_input_handles[node_id] = node_inp_dict
        await self._release_handles(handle_to_release)
              

    async def remove_and_merge(self, node_id_to_remove: list[str], node_inputs: dict[str, dict[str, Any]]):
        # remove finished node inputs first, then merge new inputs.
        handle_to_release: list[DataHandle] = []
        for node_id in node_id_to_remove:
            assert node_id in self._node_id_to_input_handles
            node_inp_dict = self._node_id_to_input_handles[node_id]
            for v in node_inp_dict.values():
                handle_to_release.append(v)
            self._node_id_to_input_handles.pop(node_id)
        return await self.merge_new_inputs(node_inputs, handle_to_release)
            
    def force_add_new_inputs(self, node_inputs: dict[str, dict[str, Any]]):
        node_handles = self._node_inputs_to_handle(node_inputs)
        for node_id, node_inp_dict in node_handles.items():
            assert node_id not in self._node_id_to_input_handles, f"node_id {node_id} already exists."
            self._node_id_to_input_handles[node_id] = node_inp_dict

    async def release_all_handles(self):
        handle_to_release: list[DataHandle] = []
        for node_id, node_inp_dict in self._node_id_to_input_handles.items():
            for v in node_inp_dict.values():
                handle_to_release.append(v)
        await self._release_handles(handle_to_release)
        self._node_id_to_input_handles = {}