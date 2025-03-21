import abc
import asyncio
from typing import Any, Callable 
from tensorpc.apps.cflow.model import ComputeFlowModel, ComputeFlowNodeModel
from tensorpc.apps.cflow.executors.base import NodeExecutorBase

class SchedulerBase(abc.ABC):
    @abc.abstractmethod
    async def schedule(self, flow: ComputeFlowModel, node_ids: list[str],
                        node_inputs: dict[str, dict[str, Any]], executors: list[NodeExecutorBase],
                        temp_executor_creator: Callable[[], NodeExecutorBase],
                        shutdown_ev: asyncio.Event) -> None:
        ...

