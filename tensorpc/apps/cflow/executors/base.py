import contextlib
import enum
from typing import Any, Optional, Union
from tensorpc.core.annolib import Undefined, undefined
from tensorpc.core.datamodel.draft import DraftUpdateOp
from tensorpc.dock import mui
import abc 
from tensorpc.apps.cflow.model import ComputeFlowNodeModel, ResourceDesp
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.core import BuiltinServiceKeys

class ExecutorType(enum.IntEnum):
    SINGLE_PROC = 0
    TORCH_DIST = 1
    # if handle data come from local executor, we send data directly (saved on data field) instead of remote handle.
    LOCAL = 2

def get_serv_key_from_exec_type(type: ExecutorType) -> str:
    if type == ExecutorType.SINGLE_PROC:
        return str(BuiltinServiceKeys.ComputeFlowSingleProcExec)
    elif type == ExecutorType.TORCH_DIST:
        return str(BuiltinServiceKeys.ComputeFlowTorchDistExec)
    else:
        raise NotImplementedError(f"ExecutorType {type} not supported.")

@dataclasses.dataclass
class ExecutorRemoteDesp:
    id: str # global unique id
    type: ExecutorType
    url: str
    rc: ResourceDesp
    
    @staticmethod
    def get_empty():
        return ExecutorRemoteDesp(id="", url="", type=ExecutorType.LOCAL, rc=ResourceDesp())

    def is_empty(self):
        return self.id == "" and self.url == ""

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class DataHandle:
    # for distributed task, we won't send node run result back to scheduler, instead we send back the data handle.
    # other executors can use this handle to get the data directly.
    id: str 
    executor_desp: ExecutorRemoteDesp
    data: Union[Undefined, Any] = undefined
    update_ops: Optional[list[DraftUpdateOp]] = None

    def has_data(self):
        return not isinstance(self.data, Undefined)

    async def get_data_from_remote(self):
        if self.has_data():
            return self.data
        raise NotImplementedError("you need to inherit and provide rpc call to get data from remote.")

    async def release(self):
        if self.has_data():
            self.data = undefined

    def __hash__(self):
        return hash(self.id)
        
# @dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
# class RemoteDataHandle(DataHandle):
#     # for distributed task, we won't send node run result back to scheduler, instead we send back the data handle.
#     # other executors can use this handle to get the data directly.
#     remote_obj: Optional[]

#     def has_data(self):
#         return not isinstance(self.data, Undefined)

#     async def get_data_from_remote(self):
#         if self.has_data():
#             return self.data
#         raise NotImplementedError("you need to inherit and provide rpc call to get data from remote.")

class NodeExecutorBase(abc.ABC):
    def __init__(self, id: str, desp: ResourceDesp):
        self._current_resource_desp = desp
        self._resource_desp = desp
        self._id = id

    @contextlib.contextmanager
    def request_resource(self, desp: ResourceDesp):
        assert self._current_resource_desp.is_request_sufficient(desp)
        try:
            self._current_resource_desp = self._current_resource_desp.get_request_remain_rc(desp)
            yield
        finally:
            self._current_resource_desp = self._current_resource_desp.add_request_rc(desp)

    def get_id(self) -> str:
        return self._id

    def get_current_resource_desp(self) -> ResourceDesp:
        return self._current_resource_desp

    def get_bottom_layout(self) -> Optional[mui.FlexBox]:
        return None 

    def get_right_layout(self) -> Optional[mui.FlexBox]:
        return None

    async def get_remote_node_preview_layout(self, node_id: str) -> Optional[Union[mui.FlexBox, mui.RemoteComponentBase]]:
        """Executor can use a remote component to connect a remote node preview layout.
        for local executor, it should return None because we control it directly.
        """
        return None

    async def get_remote_node_detail_layout(self, node_id: str) -> Optional[Union[mui.FlexBox, mui.RemoteComponentBase]]:
        """Executor can use a remote component to connect a remote node detail layout.
        for local executor, it should return None because we control it directly.
        """
        return None

    @abc.abstractmethod
    async def run_node(self, node: ComputeFlowNodeModel, inputs: dict[str, DataHandle]) -> Optional[dict[str, DataHandle]]: ...

    @abc.abstractmethod
    async def close(self):
        return None