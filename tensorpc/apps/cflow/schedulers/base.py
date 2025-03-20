import abc
from typing import Any 
from tensorpc.dock.components.flowplus.model import ComputeFlowModel, ComputeFlowNodeModel

class SchedulerBase(abc.ABC):
    @abc.abstractmethod
    async def schedule(self, flow: ComputeFlowModel, node_ids: list[str],
                        node_inputs: dict[str, dict[str, Any]]) -> None:
        raise NotImplementedError