import contextlib
import enum
from typing import TYPE_CHECKING, Any, Optional, Union
from tensorpc.core.annolib import Undefined, undefined
from tensorpc.core.asyncclient import AsyncRemoteManager
from tensorpc.core.datamodel.draft import DraftUpdateOp
from tensorpc.dock import mui
import abc 
from tensorpc.apps.cflow.coremodel import ResourceDesc
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.core import BuiltinServiceKeys
from tensorpc.dock.components.terminal import AsyncSSHTerminal, TerminalBuffer

if TYPE_CHECKING:
    from tensorpc.apps.cflow.model import ComputeNodeModel


class ExecutorType(enum.IntEnum):
    SINGLE_PROC = 0
    TORCH_DIST = 1
    # if handle data come from local executor, we send data directly (saved on data field) instead of remote handle.
    LOCAL = 2

NODE_EXEC_SERVICE = f"tensorpc.apps.cflow.services.executors::NodeExecutorService"
RELAY_SERVICE = f"tensorpc.apps.dbg.services.relay::RelayMonitor"


class RemoteExecutorServiceKeys(enum.Enum):
    GET_DATA = f"{NODE_EXEC_SERVICE}.get_cached_data"
    RELEASE_DATA = f"{NODE_EXEC_SERVICE}.remove_cached_data"
    GET_DESP = f"{NODE_EXEC_SERVICE}.get_desp"
    RUN_NODE = f"{NODE_EXEC_SERVICE}.run_node"
    IMPORT_REGISTRY_MODULES = f"{NODE_EXEC_SERVICE}.import_registry_modules"
    SETUP_NODE = f"{NODE_EXEC_SERVICE}.setup_node"

@dataclasses.dataclass
class ExecutorRemoteDesc:
    id: str # global unique id
    type: ExecutorType
    url: str
    rc: ResourceDesc
    
    @staticmethod
    def get_empty():
        return ExecutorRemoteDesc(id="", url="", type=ExecutorType.LOCAL, rc=ResourceDesc())

    def is_empty(self):
        return self.id == "" and self.url == ""

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class DataHandle:
    # for distributed task, we won't send node run result back to scheduler, instead we send back the data handle.
    # other executors can use this handle to get the data directly.
    id: str 
    executor_desp: ExecutorRemoteDesc
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

@dataclasses.dataclass(config=dataclasses.PyDanticConfigForAnyObject)
class RemoteGrpcDataHandle(DataHandle):
    remote_obj: Optional[AsyncRemoteManager] = None
    async def get_data_from_remote(self):
        if self.has_data():
            return self.data
        assert self.remote_obj is not None
        return await self.remote_obj.chunked_remote_call(RemoteExecutorServiceKeys.GET_DATA.value, self.id)

    async def release(self):
        if self.has_data():
            self.data = undefined
            return
        assert self.remote_obj is not None
        await self.remote_obj.chunked_remote_call(RemoteExecutorServiceKeys.RELEASE_DATA.value, self.id)

class NodeExecutorBase(abc.ABC):
    def __init__(self, id: str, desc: ResourceDesc):
        self._current_resource_desp = desc
        self._resource_desp = desc
        self._id = id
        self._terminal_buffer = TerminalBuffer()

    @contextlib.contextmanager
    def request_resource(self, desc: ResourceDesc):
        assert self._current_resource_desp.is_request_sufficient(desc)
        try:
            self._current_resource_desp = self._current_resource_desp.get_request_remain_rc(desc)
            yield
        finally:
            self._current_resource_desp = self._current_resource_desp.add_request_rc(desc)

    def get_id(self) -> str:
        return self._id

    def get_terminal_buffer(self):
        return self._terminal_buffer

    def get_ssh_terminal(self) -> Optional[AsyncSSHTerminal]:
        return None 

    def get_current_resource_desp(self) -> ResourceDesc:
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
    async def run_node(self, node: "ComputeNodeModel", inputs: dict[str, DataHandle]) -> Optional[dict[str, DataHandle]]: ...

    @abc.abstractmethod
    async def close(self):
        return None

    def is_local(self) -> bool: 
        return True

    @abc.abstractmethod
    async def setup_node(self, node: "ComputeNodeModel") -> None: ...
