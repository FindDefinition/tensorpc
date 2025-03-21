from typing import Any
from tensorpc.apps.cflow.model import ComputeFlowNodeModel
from tensorpc.core.asyncclient import AsyncRemoteManager
from .base import NodeExecutorBase, DataHandle, ExecutorRemoteDesp
import inspect

from typing_extensions import override


class SingleProcGrpcNodeExecutor(NodeExecutorBase):
    def __init__(self, robj: AsyncRemoteManager):
        self.robj = robj 

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
        return DataHandle(id="", executor_desp=ExecutorRemoteDesp.get_empty(), data=data)
