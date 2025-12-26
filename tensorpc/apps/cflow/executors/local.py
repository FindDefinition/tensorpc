from typing import Any, Optional
import uuid
from tensorpc.apps.cflow.model import ComputeNodeModel
from tensorpc.core.annolib import Undefined
from tensorpc.dock.components import mui
from .base import NodeExecutorBase, DataHandle, ExecutorRemoteDesc
import inspect

from typing_extensions import override

class LocalNodeExecutor(NodeExecutorBase):
    def get_executor_remote_desp(self) -> ExecutorRemoteDesc: 
        return ExecutorRemoteDesc.get_empty()

    # each scheduler should only have one local executor.
    @override
    async def run_node(self, node: ComputeNodeModel, inputs: dict[str, DataHandle]) -> Optional[dict[str, DataHandle]]:
        assert node.runtime is not None 
        node_inp_handles = node.runtime.inp_handles
        node_inp_handles_dict = {h.name: h for h in node_inp_handles}
        cnode = node.runtime.cnode
        compute_func = cnode.get_compute_func()
        inputs_val = {}
        for k, inp in inputs.items():
            handle = node_inp_handles_dict[k]
            if handle.type == "handledictsource":
                assert not isinstance(inp.data, Undefined)
                inp_special_dict = {}
                for k2, v2 in inp.data.items():
                    assert isinstance(v2, DataHandle)
                    if v2.has_data():
                        inp_special_dict[k2] = v2.data
                    else:
                        inp_special_dict[k2] = await v2.get_data_from_remote()
                inputs_val[k] = inp_special_dict
            else:
                if inp.has_data():
                    inputs_val[k] = inp.data
                else:
                    inputs_val[k] = await inp.get_data_from_remote()
        data = compute_func(**inputs_val)
        if inspect.iscoroutine(data):
            data = await data
        # TODO currently local data handle will be sent to remote executors directly instead of sending remote handle.
        if isinstance(data, dict):
            data_handle_dict: dict[str, DataHandle] = {}
            for k, v in data.items():
                uuid_str = uuid.uuid4().hex
                uid = f"{self.get_id()}-{uuid_str}-{k}"
                data_handle_dict[k] = DataHandle(id=uid, executor_desp=ExecutorRemoteDesc.get_empty(), data=v)
            return data_handle_dict
        else:
            assert data is None, f"compute_func {compute_func} should return None or dict."
        return data

    async def close(self):
        return None

    def is_local(self) -> bool: 
        return True

    def get_bottom_layout(self) -> Optional[mui.FlexBox]:
        return mui.VBox([
            mui.Markdown("## Hello Local Executor!")
        ]).prop(width="100%", height="100%", overflow="auto")

    @override
    async def setup_node(self, node: ComputeNodeModel) -> None:
        return 