import asyncio
from .base import SchedulerBase

import abc
from typing import Any, Callable, Optional 
from tensorpc.apps.cflow.model import ComputeFlowModel, ComputeFlowNodeModel
from tensorpc.apps.cflow.executors.base import NodeExecutorBase
from typing_extensions import override

import dataclasses as dataclasses_plain

@dataclasses_plain.dataclass
class SimpleSchedulerState:
    wait_node_inputs: dict[str, dict[str, Any]]

class SimpleScheduler(SchedulerBase):
    def __init__(self):
        self._state: Optional[SimpleSchedulerState] = None

    @override
    async def schedule(self, flow: ComputeFlowModel, node_ids: list[str],
                        node_inputs: dict[str, dict[str, Any]], executors: list[NodeExecutorBase],
                        temp_executor_creator: Callable[[], NodeExecutorBase],
                        shutdown_ev: asyncio.Event) -> None:
        cur_node_inputs = node_inputs.copy()
        # cur_anode_iters: dict[str, AsyncIterator] = {}
        shutdown_task = asyncio.create_task(shutdown_ev.wait())
        nodes = [flow.nodes[node_id] for node_id in node_ids]
        nodes_to_schedule: list[ComputeFlowNodeModel] = nodes
        state = self._state
        if state is not None:
            wait_nodes, wait_inputs = state.wait_node_inputs
            nodes_to_schedule = nodes_to_schedule + wait_nodes
            node_inputs = {**wait_inputs, **node_inputs}

