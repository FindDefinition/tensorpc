import asyncio
import time
import traceback

from tensorpc.apps.cflow.nodes.cnode.ctx import enter_flow_ui_node_context, enter_flow_ui_node_context_object
from tensorpc.dock.components.flowui import FlowInternals
from tensorpc.dock.components.models.flow import BaseEdgeModel
from .base import SchedulerBase

import abc
from typing import Any, Callable, Optional 
from tensorpc.apps.cflow.model import ComputeFlowDrafts, ComputeFlowModel, ComputeFlowModelRoot, ComputeFlowNodeModel, get_compute_flow_drafts
from tensorpc.apps.cflow.executors.base import NodeExecutorBase
from typing_extensions import override
from tensorpc.dock.components import mui

import dataclasses as dataclasses_plain

@dataclasses_plain.dataclass
class SimpleSchedulerState:
    wait_node_inputs: dict[str, dict[str, Any]]


def filter_node_cant_schedule(flow_runtime: FlowInternals[ComputeFlowNodeModel, BaseEdgeModel], nodes: list[ComputeFlowNodeModel],
                                node_inputs: dict[str, dict[str, Any]], init_nodes_cant_schedule: list[str]):
    new_nodes: list[ComputeFlowNodeModel] = []
    nodes_dont_have_enough_inp: list[ComputeFlowNodeModel] = []
    init_nodes_cant_schedule_set = set(init_nodes_cant_schedule)
    
    for n in nodes:
        assert n.runtime is not None 
        if n.id in init_nodes_cant_schedule_set:
            nodes_dont_have_enough_inp.append(n)
            continue
        if n.id in node_inputs:
            node_inp = node_inputs[n.id]
        else:
            node_inp = {}
        not_found = False
        for handle in n.runtime.inp_handles:
            if not handle.is_optional and handle.name not in node_inp:
                not_found = True
                break
        if not_found:
            nodes_dont_have_enough_inp.append(n)
            continue
        new_nodes.append(n)
    new_nodes_ids = set(n.id for n in new_nodes)
    node_inputs_sched_in_future: dict[str, dict[str, Any]] = {}
    for node in nodes_dont_have_enough_inp:
        all_parents = flow_runtime.get_all_parent_nodes(node.id)
        for parent in all_parents:
            if parent.id in new_nodes_ids:
                if node.id not in node_inputs:
                    node_inputs_sched_in_future[node.id] = {}
                else:
                    node_inputs_sched_in_future[node.id] = node_inputs[
                        node.id]
                break
    return new_nodes, node_inputs_sched_in_future


class SimpleScheduler(SchedulerBase):
    def __init__(self, dm_comp: mui.DataModel[ComputeFlowModelRoot]):
        self._state: Optional[SimpleSchedulerState] = None
        self._dm_comp = dm_comp
        self._drafts = get_compute_flow_drafts(dm_comp.get_draft_type_only()
        )

    def get_datamodel_component(self) -> mui.DataModel[ComputeFlowModelRoot]:
        return self._dm_comp

    def get_compute_flow_drafts(self) -> ComputeFlowDrafts:
        return self._drafts

    def assign_node_executor(self, nodes: list[ComputeFlowNodeModel], executors: list[NodeExecutorBase]) -> dict[str, NodeExecutorBase]:
        res: dict[str, NodeExecutorBase] = {}
        # 1. group nodes by executor id
        ex_id_to_nodes_sortkey: dict[str, list[tuple[ComputeFlowNodeModel, tuple[int, ...]]]] = {}
        for n in nodes:
            ex_id = n.vExecId
            if ex_id not in ex_id_to_nodes_sortkey:
                ex_id_to_nodes_sortkey[ex_id] = []
            ex_id_to_nodes_sortkey[ex_id].append((n, (n.vGPU, n.vGPUMem, n.vCPU, n.vMem)))
        # 2. assign executor to nodes
        for ex_id, node_group in ex_id_to_nodes_sortkey.items():
            node_group.sort(key=lambda x: x[1], reverse=True)
            # find first executor that can handle node with largest resource
            for ex in executors:
                ex_rc = ex.get_resource_desp()
                max_node_rc_req = node_group[0][0].get_request_resource_desp()
                if ex_rc.is_request_sufficient(max_node_rc_req):
                    # try to assign all node to this executor
                    for n, _ in node_group:
                        node_rc_req = n.get_request_resource_desp()
                        if ex_rc.is_request_sufficient(max_node_rc_req):
                            res[n.id] = ex
                            ex_rc = ex_rc.get_request_remain_rc(node_rc_req)
                    break
        assert res, "no executor can handle nodes"
        return res

    @override
    async def schedule(self, flow: ComputeFlowModel, node_ids: list[str],
                        node_inputs: dict[str, dict[str, Any]], executors: list[NodeExecutorBase],
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
                valid_nodes, node_inputs_in_future = filter_node_cant_schedule(
                    flow.runtime, nodes_to_schedule, cur_node_inputs)
                if not valid_nodes and not node_inputs_in_future:
                    break
                # 2. remove duplicate nodes
                valid_nodes_id_set = set(n.id for n in valid_nodes)
                new_valid_nodes: list[ComputeFlowNodeModel] = []
                for n in valid_nodes:
                    if n.id in valid_nodes_id_set:
                        new_valid_nodes.append(n)
                        valid_nodes_id_set.remove(n.id)
                tasks: list[asyncio.Task] = []
                task_to_noded: dict[asyncio.Task, ComputeFlowNodeModel] = {}
                node_outputs: dict[str, dict[str, Any]] = {}
                for n in valid_nodes:
                    node_inp = cur_node_inputs.get(n.id, {})
                    node_rt = n.runtime 
                    assert node_rt is not None
                    compute_func = node_rt.cnode.get_compute_func()
                    node_state = flow.create_or_convert_node_state(n.id)
                    node_drafts = drafts.get_node_drafts(n.id)
                    with enter_flow_ui_node_context(n.id, node_state, node_drafts.node_state):
                        try:
                            t1 = time.time()
                            data = compute_func(**node_inp)
                        except Exception as exc:
                            await wrapper.update_status(NodeStatus.Error)
                            traceback.print_exc()
                            await self.send_exception(exc)
                            continue

                        pass 

        except Exception as exc:
            # await self.send_exception(exc)
            traceback.print_exc()
            raise exc
        print("Done")
        self._schedule_task = None
