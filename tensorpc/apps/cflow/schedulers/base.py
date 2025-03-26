import abc
import asyncio
from typing import Any, Callable, Optional 
from tensorpc.apps.cflow.model import ComputeFlowDrafts, ComputeFlowModel, ComputeFlowModelRoot, ComputeFlowNodeModel
from tensorpc.apps.cflow.executors.base import NodeExecutorBase
from tensorpc.dock.components import mui

class SchedulerBase(abc.ABC):
    @abc.abstractmethod
    def get_datamodel_component(self) -> mui.DataModel[ComputeFlowModelRoot]:
        ...

    @abc.abstractmethod
    def get_compute_flow_drafts(self) -> ComputeFlowDrafts:
        ...

    @abc.abstractmethod
    def assign_node_executor(self, nodes: list[ComputeFlowNodeModel], executors: list[NodeExecutorBase]) -> dict[str, NodeExecutorBase]:
        ...

    @abc.abstractmethod
    async def schedule(self, flow: ComputeFlowModel, 
                        node_inputs: dict[str, dict[str, Any]], executors: list[NodeExecutorBase],
                        shutdown_ev: asyncio.Event) -> Optional[asyncio.Task]:
        ...

    @abc.abstractmethod
    async def close(self): ...
