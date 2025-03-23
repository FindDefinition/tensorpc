import asyncio
import traceback
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
            wait_inputs = state.wait_node_inputs
            wait_nodes = [flow.nodes[node_id] for node_id in wait_inputs.keys()]
            nodes_to_schedule = nodes_to_schedule + wait_nodes
            node_inputs = {**wait_inputs, **node_inputs}
        cur_node_inputs = node_inputs.copy()
        shutdown_task = asyncio.create_task(shutdown_ev.wait())

        try:
            while (nodes_to_schedule
                   or (state is not None and state.wait_node_inputs)):
                new_nodes_to_schedule: list[ComputeFlowNodeModel] = []
                new_node_inputs: dict[str, dict[str, Any]] = {}
                # 1. validate node inputs, all input handle (not optional) must be
                #    provided in node inputs
                if state is not None:
                    wait_inputs = state.wait_node_inputs
                    wait_nodes = [flow.nodes[node_id] for node_id in wait_inputs.keys()]
                    nodes_to_schedule = nodes_to_schedule + wait_nodes
                    cur_node_inputs = {**cur_node_inputs, **wait_inputs}
                    valid_nodes, node_inputs_in_future = self._filter_node_cant_schedule(
                    nodes_to_schedule, cur_node_inputs, cur_anode_iters)

        except Exception as exc:
            # await self.send_exception(exc)
            traceback.print_exc()
            raise exc
        print("Done")
        self._schedule_task = None
