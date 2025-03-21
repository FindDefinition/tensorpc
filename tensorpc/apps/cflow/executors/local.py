from typing import Any
from tensorpc.apps.cflow.model import ComputeFlowNodeModel
from .base import NodeExecutorBase, DataHandle, ExecutorRemoteDesp
import inspect

from typing_extensions import override

class LocalNodeExecutor(NodeExecutorBase):
    # each scheduler should only have one local executor.
    @override
    async def run_node(self, node: ComputeFlowNodeModel, inputs: dict[str, DataHandle]) -> DataHandle:
        assert node.runtime is not None 
        cnode = node.runtime.cnode
        compute_func = cnode.get_compute_func()
        inputs_val = {}
        for k, inp in inputs.items():
            if inp.has_data():
                inputs_val[k] = inp.data
            else:
                inputs_val[k] = await inp.get_data_from_remote()
        data = compute_func(**inputs_val)
        if inspect.iscoroutine(data):
            data = await data
        # local data handle will be sent to remote executors directly instead of sending remote handle.
        return DataHandle(id="", executor_desp=ExecutorRemoteDesp.get_empty(), data=data)
