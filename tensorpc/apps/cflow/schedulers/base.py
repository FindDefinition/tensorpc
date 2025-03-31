import abc
import asyncio
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union 
from tensorpc.apps.cflow.model import ComputeFlowDrafts, ComputeFlowModel, ComputeFlowModelRoot, ComputeNodeModel, ComputeNodeRuntime
from tensorpc.apps.cflow.executors.base import NodeExecutorBase
from tensorpc.dock.components import mui
from tensorpc.dock.components.terminal import TerminalBuffer

class SchedulerBase(abc.ABC):
    @abc.abstractmethod
    def get_datamodel_component(self) -> mui.DataModel[ComputeFlowModelRoot]:
        ...

    @abc.abstractmethod
    def get_compute_flow_drafts(self) -> ComputeFlowDrafts:
        ...

    @abc.abstractmethod
    def assign_node_executor(self, nodes: Sequence[ComputeNodeModel], executors: Sequence[NodeExecutorBase]) -> dict[str, Union[NodeExecutorBase, Callable[[ComputeNodeModel], NodeExecutorBase]]]:
        ...

    @abc.abstractmethod
    async def schedule(self, flow: ComputeFlowModel, 
                        node_inputs: dict[str, dict[str, Any]], 
                        executors: Sequence[NodeExecutorBase],
                        executor_term_buffers: dict[str, TerminalBuffer],
                        shutdown_ev: asyncio.Event) -> Optional[asyncio.Task]:
        """Schedule nodes with its args.

        Args:
            flow: The compute flow model.
            node_inputs: The inputs for each node.
            executors: The executors to run the nodes.
            executor_term_buffers: The terminal buffers for executor with SSH Terminal.
                the Main UI will create buffers in init. temp executors need 
                to add its buffer to this dict temporarily.
            shutdown_ev: An event to signal shutdown.
        """
        ...

    @abc.abstractmethod
    async def close(self): ...

    @abc.abstractmethod
    async def run_sub_graph(self, flow: ComputeFlowModel, node_id: str,
                            executors: Sequence[NodeExecutorBase],
                            executor_term_buffers: dict[str, TerminalBuffer]) -> Any:
        ...
