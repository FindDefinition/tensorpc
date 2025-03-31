import asyncio
from collections.abc import Sequence
from contextlib import ExitStack
import datetime
import time
import traceback
import uuid

import rich

from tensorpc.apps.cflow.executors.handlemgr import DataHandleManager
from tensorpc.apps.cflow.executors.local import LocalNodeExecutor
from tensorpc.apps.cflow.logger import CFLOW_LOGGER
from tensorpc.apps.cflow.nodes.cnode.ctx import enter_flow_ui_node_context
from tensorpc.apps.cflow.nodes.cnode.registry import ComputeNodeFlags, ComputeNodeRuntime
from tensorpc.dock.components.flowui import FlowInternals
from tensorpc.dock.components.models.flow import BaseEdgeModel
from tensorpc.dock.components.terminal import TerminalBuffer
from .base import SchedulerBase

import abc
from typing import Any, Callable, Optional, Union
from tensorpc.apps.cflow.model import ComputeFlowDrafts, ComputeFlowModel, ComputeFlowModelRoot, ComputeFlowNodeDrafts, ComputeNodeModel, ComputeNodeStatus, ComputeNodeType, ResourceDesc, get_compute_flow_drafts
from tensorpc.apps.cflow.executors.base import DataHandle, ExecutorRemoteDesc, NodeExecutorBase
from typing_extensions import override
from tensorpc.dock.components import mui
import tensorpc.core.dataclass_dispatch as dataclasses
import dataclasses as dataclasses_plain
import humanize
from tensorpc.core.asynctools import cancel_task


def _short_precise_delta(dt: datetime.timedelta, minimum_unit: str = "seconds") -> str:
    res = humanize.precisedelta(dt, minimum_unit=minimum_unit)

    # replace milliseconds with ms
    res = res.replace("milliseconds", "ms")
    # replace seconds with s
    res = res.replace("seconds", "s")
    # replace minutes with m
    res = res.replace("minutes", "m")
    # replace hours with h
    res = res.replace("hours", "h")
    return res

@dataclasses_plain.dataclass
class SimpleSchedulerState:
    wait_node_inputs: dict[str, dict[str, Any]]
    task: Optional[asyncio.Task] = None

    def comsume_wait_node_inputs(self):
        res = self.wait_node_inputs
        self.wait_node_inputs = {}
        return res

    def update_node_inputs(self, node_inputs: dict[str, dict[str, Any]]):
        self.wait_node_inputs.update(node_inputs)


def filter_node_cant_schedule(nodes: list[ComputeNodeModel],
                              node_inputs: dict[str, dict[str, Any]]):
    new_nodes: list[ComputeNodeModel] = []
    nodes_dont_have_enough_inp: list[ComputeNodeModel] = []

    for n in nodes:
        assert n.runtime is not None
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
    return new_nodes, nodes_dont_have_enough_inp


def get_node_inputs_sched_in_future(
        flow_runtime: FlowInternals[ComputeNodeModel,
                                    BaseEdgeModel], next_node_ids: list[str],
        node_wont_schedule: list[ComputeNodeModel],
        node_inputs: dict[str, dict[str, Any]]):
    next_node_ids_set = set(n for n in next_node_ids)

    node_inputs_sched_in_future: dict[str, dict[str, Any]] = {}
    for node in node_wont_schedule:
        all_parents = flow_runtime.get_all_parent_nodes(node.id)
        for parent in all_parents:
            if parent.id in next_node_ids_set:
                if node.id not in node_inputs:
                    node_inputs_sched_in_future[node.id] = {}
                else:
                    node_inputs_sched_in_future[node.id] = node_inputs[node.id]
                break
    return node_inputs_sched_in_future


def _get_next_node_inputs(root: ComputeFlowModelRoot,
                          flow_runtime: FlowInternals[ComputeNodeModel,
                                                      BaseEdgeModel],
                          node_id_to_outputs: dict[str, dict[str, Any]]):
    new_node_inputs: dict[str, dict[str, Any]] = {}
    for node_id, output in node_id_to_outputs.items():
        # TODO handle array/dict handle
        node_target_and_handles = flow_runtime.get_target_node_and_handles(
            node_id)
        for target_node, source_handle, target_handle in node_target_and_handles:
            target_node_runtime = root.get_or_create_node_runtime(target_node)
            assert source_handle is not None and target_handle is not None
            source_handle_name = source_handle.split("-")[1]
            target_handle_name = target_handle.split("-")[1]
            if source_handle_name in output:
                handle_name_to_inp_handle = {
                    h.name: h
                    for h in target_node_runtime.inp_handles
                }
                if target_node.id not in new_node_inputs:
                    new_node_inputs[target_node.id] = {}
                # for SpecialHandleDict, we need to accumulate the dict
                if handle_name_to_inp_handle[
                        target_handle_name].type == "handledictsource":
                    if target_handle_name not in new_node_inputs[
                            target_node.id]:
                        new_node_inputs[
                            target_node.id][target_handle_name] = {}
                    new_node_inputs[target_node.id][target_handle_name][
                        source_handle_name] = output[source_handle_name]
                else:
                    new_node_inputs[target_node.id][
                        target_handle_name] = output[source_handle_name]
    return new_node_inputs


class SimpleScheduler(SchedulerBase):

    def __init__(self, dm_comp: mui.DataModel[ComputeFlowModelRoot],
                 shutdown_ev: asyncio.Event):
        self._shutdown_ev = shutdown_ev
        self._state: Optional[SimpleSchedulerState] = None
        self._dm_comp = dm_comp
        self._drafts = get_compute_flow_drafts(dm_comp.get_draft_type_only())

    def get_datamodel_component(self) -> mui.DataModel[ComputeFlowModelRoot]:
        return self._dm_comp

    def get_compute_flow_drafts(self) -> ComputeFlowDrafts:
        return self._drafts

    def assign_node_executor(
            self, nodes: Sequence[ComputeNodeModel],
            executors: Sequence[NodeExecutorBase]) -> dict[str, Union[NodeExecutorBase, Callable[[ComputeNodeModel], NodeExecutorBase]]]:
        res: dict[str, Union[NodeExecutorBase, Callable[[ComputeNodeModel], NodeExecutorBase]]] = {}
        if not nodes:
            return res
        local_ex: Optional[NodeExecutorBase] = None
        # 1. group nodes by executor id
        ex_id_to_nodes_sortkey: dict[str, list[tuple[ComputeNodeModel,
                                                    ResourceDesc,
                                                     tuple[int, ...]]]] = {}
        for n in nodes:
            ex_id = n.vExecId
            assert n.runtime is not None 
            if n.runtime.cfg.temp_executor_creator is not None:
                res[n.id] = n.runtime.cfg.temp_executor_creator
                continue
            if n.runtime.cfg.flags & ComputeNodeFlags.EXEC_ALWAYS_LOCAL:
                if local_ex is None:
                    for ex in executors:
                        if ex.is_local():
                            local_ex = ex
                            break 
                    assert local_ex is not None, "Local executor is required for observer nodes"
                res[n.id] = local_ex
            else:
                if ex_id not in ex_id_to_nodes_sortkey:
                    ex_id_to_nodes_sortkey[ex_id] = []
                nrc = n.vResource
                if n.runtime.cfg.resource_desp is not None:
                    nrc = n.runtime.cfg.resource_desp 
                ex_id_to_nodes_sortkey[ex_id].append(
                    (n, nrc, (nrc.GPU, nrc.GPUMem, nrc.CPU, nrc.Mem)))
        # 2. assign executor to nodes
        for ex_id, node_group in ex_id_to_nodes_sortkey.items():
            node_group.sort(key=lambda x: x[2], reverse=True)
            # 1. if user exec id is same as executor id, try to assign to this executor
            exactly_match = False
            for ex in executors:
                if ex.get_id() == ex_id:
                    ex_rc = ex.get_current_resource_desp()
                    max_node_rc_req = node_group[0][1]
                    if ex_rc.is_request_sufficient(max_node_rc_req):
                        exactly_match = True
                        # try to assign all node to this executor
                        for n, nrc, _ in node_group:
                            if ex_rc.is_request_sufficient(max_node_rc_req):
                                res[n.id] = ex
                                ex_rc = ex_rc.get_request_remain_rc(nrc)
                    else:
                        CFLOW_LOGGER.warning("Executor %s(%s) can't handle node %s(%s)", ex.get_id(), str(ex_rc), node_group[0][0].id, str(max_node_rc_req))
                    break
            if exactly_match:
                continue
            # find first executor that can handle node with largest resource
            for ex in executors:
                ex_rc = ex.get_current_resource_desp()
                max_node_rc_req = node_group[0][1]
                if ex_rc.is_request_sufficient(max_node_rc_req):
                    # try to assign all node to this executor
                    for n, nrc, _ in node_group:
                        if ex_rc.is_request_sufficient(max_node_rc_req):
                            res[n.id] = ex
                            ex_rc = ex_rc.get_request_remain_rc(nrc)
                    break
        assert res, "no executor can handle nodes"
        return res

    async def _schedule_node(self, node: ComputeNodeModel, node_inp,
                             node_ex: Union[NodeExecutorBase, Callable[[ComputeNodeModel], NodeExecutorBase]],
                             node_drafts: ComputeFlowNodeDrafts,
                             dm_comp: mui.DataModel[ComputeFlowModelRoot],
                             ex_term_bufs: dict[str, TerminalBuffer]):
        temp_ex: Optional[NodeExecutorBase] = None
        try:
            with ExitStack() as stack:
                assert node.runtime is not None 
                if isinstance(node_ex, NodeExecutorBase):
                    stack.enter_context(node_ex.request_resource(node.get_request_resource_desp())) 
                else:
                    node_ex = node_ex(node)
                    temp_ex = node_ex
                    term = node_ex.get_ssh_terminal()
                    if term is not None:
                        assert node_ex.get_id() not in ex_term_bufs
                        # to share multiple terminal backends in one frontend terminal
                        # we must use a global dict to store each backend state.
                        ex_term_bufs[node_ex.get_id()] = TerminalBuffer()
                        term.set_state_buffers(ex_term_bufs)

                async with dm_comp.draft_update():
                    node_drafts.node.runtime.executor = node_ex # type: ignore
                    node_drafts.node.status = ComputeNodeStatus.Running
                    node_drafts.node.msg = "running"
                t = time.time()
                if isinstance(node_ex, NodeExecutorBase):
                    res = await node_ex.run_node(node, node_inp)
                assert res is None or isinstance(res, dict)
                async with dm_comp.draft_update():
                    # TODO should we clear executor here?
                    # this will cause executor layout to be unmount-remount.
                    node_drafts.node.runtime.executor = None # type: ignore
                    node_drafts.node.status = ComputeNodeStatus.Ready
                    dt = datetime.timedelta(seconds=time.time() - t)
                    node_drafts.node.msg = _short_precise_delta(
                        dt, minimum_unit="milliseconds")
            return res, True
        except BaseException as exc:
            async with dm_comp.draft_update():
                node_drafts.node.runtime.executor = None # type: ignore
                node_drafts.node.status = ComputeNodeStatus.Error
                node_drafts.node.msg = "error"
            traceback.print_exc()
            await dm_comp.send_exception(exc)
            return None, False
        finally:
            try:
                if temp_ex is not None:
                    if temp_ex.get_id() in ex_term_bufs:
                        del ex_term_bufs[temp_ex.get_id()]
                    term = temp_ex.get_ssh_terminal()
                    if term is not None:
                        term.reset_state_buffers()
                    await temp_ex.close()
            except BaseException as e:
                traceback.print_exc()
                CFLOW_LOGGER.error("close temp executor %s failed: %s", str(temp_ex), str(e))

    def _get_nodes_can_schedule(self, nodes: list[ComputeNodeModel],
                                node_inputs: dict[str, dict[str, Any]],
                                executors: Sequence[NodeExecutorBase]):
        valid_nodes, inp_not_enough_nodes = filter_node_cant_schedule(
            nodes, node_inputs)
        assigned_execs = self.assign_node_executor(valid_nodes, executors)
        valid_nodes_dict: dict[str, ComputeNodeModel] = {}
        for n in valid_nodes:
            if n.id not in assigned_execs:
                inp_not_enough_nodes.append(n)
            else:
                valid_nodes_dict[n.id] = n
        return valid_nodes_dict, assigned_execs

    def _get_node_scheduled_task(self, flow: ComputeFlowModel,
                                 node: ComputeNodeModel,
                                 assigned_execs: dict[str, Union[NodeExecutorBase, Callable[[ComputeNodeModel], NodeExecutorBase]]],
                                 ex_term_bufs: dict[str, TerminalBuffer],
                                 node_inputs_state: DataHandleManager):
        node_ex = assigned_execs[node.id]
        drafts = self.get_compute_flow_drafts()
        dm_comp = self.get_datamodel_component()
        node_inp = node_inputs_state.get_node_inputs(node.id, {})
        node_rt = node.runtime
        if node.runtime is None:
            node.runtime = node.get_node_runtime(dm_comp.get_model())
        assert node_rt is not None
        node_state = flow.create_or_convert_node_state(node.id)
        node_drafts = drafts.get_node_drafts(node.id)
        with enter_flow_ui_node_context(node.id, node_state,
                                        node_drafts.node_state):
            return asyncio.create_task(self._schedule_node(
                node, node_inp, node_ex, node_drafts, dm_comp, ex_term_bufs),
                                       name=node.id)

    async def run_sub_graph(self, flow: ComputeFlowModel, node_id: str,
                            executors: Sequence[NodeExecutorBase],
                            executor_term_buffers: dict[str, TerminalBuffer]):
        node = flow.nodes[node_id]
        assert node.nType == ComputeNodeType.COMPUTE, f"node {node_id} is not a compute node"
        all_nodes = flow.runtime.get_all_nodes_in_connected_graph(node)
        root_node_inputs = {}
        for node in all_nodes:
            if flow.runtime.get_source_node_and_handles(node.id):
                continue
            root_node_inputs[node.id] = {}
        return await self.schedule(flow, root_node_inputs, executors,
                                    executor_term_buffers,
                                   self._shutdown_ev)

    async def schedule_nodes(self, flow: ComputeFlowModel,
                             node_id_to_inputs: dict[str, dict[str, Any]],
                             executors: Sequence[NodeExecutorBase],
                             executor_term_buffers: dict[str, TerminalBuffer]):
        for n in node_id_to_inputs.keys():
            assert n in flow.nodes, f"node {n} not in flow"
            assert flow.nodes[
                n].nType == ComputeNodeType.COMPUTE, f"node {n} is not a compute node"
        if self._state is not None:
            self._state.update_node_inputs(node_id_to_inputs)
        else:
            await self.schedule(flow, node_id_to_inputs, executors,
                                executor_term_buffers,
                                self._shutdown_ev)

    async def schedule_node(self, flow: ComputeFlowModel, node_id: str,
                            node_inputs: dict[str, Any],
                            executors: list[NodeExecutorBase],
                            executor_term_buffers: dict[str, TerminalBuffer]):
        return await self.schedule_nodes(flow, {node_id: node_inputs},
                                         executors, executor_term_buffers)

    async def schedule_next(self, flow: ComputeFlowModel, node_id: str,
                            node_outputs: dict[str, Any],
                            executors: list[NodeExecutorBase],
                            executor_term_buffers: dict[str, TerminalBuffer]):
        node = flow.nodes[node_id]
        assert node.nType == ComputeNodeType.COMPUTE, f"node {node_id} is not a compute node"
        dm_comp = self.get_datamodel_component()
        node_inputs = _get_next_node_inputs(dm_comp.get_model(), flow.runtime,
                                            node_outputs)
        return await self.schedule(flow, node_inputs, executors,
                                    executor_term_buffers,
                                   self._shutdown_ev)

    async def _schedule(self, flow: ComputeFlowModel,
                        node_inputs: dict[str, dict[str, Any]],
                        executors: Sequence[NodeExecutorBase],
                        executor_term_buffers: dict[str, TerminalBuffer],
                        shutdown_ev: asyncio.Event) -> None:
        try:
            nodes = [flow.nodes[node_id] for node_id in node_inputs.keys()]
            nodes_to_schedule: list[ComputeNodeModel] = nodes
            # cur_anode_iters: dict[str, AsyncIterator] = {}
            assert self._state is not None
            shutdown_task = asyncio.create_task(shutdown_ev.wait())
            state = self._state
            # cur_node_inputs = node_inputs.copy()
            node_inputs_state: DataHandleManager = DataHandleManager(
                node_inputs.copy())
            shutdown_task = asyncio.create_task(shutdown_ev.wait())
            dm_comp = self.get_datamodel_component()

            for node in nodes_to_schedule:
                if node.runtime is None:
                    node.runtime = node.get_node_runtime(dm_comp.get_model())

            nodes_can_schedule, assigned_execs = self._get_nodes_can_schedule(
                nodes_to_schedule, node_inputs, executors)
            if not nodes_can_schedule:
                CFLOW_LOGGER.warning("No node can be scheduled")
                return
            node_tasks: list[asyncio.Task] = []
            for node in nodes_can_schedule.values():
                task = self._get_node_scheduled_task(flow, node, assigned_execs,
                                                    executor_term_buffers, node_inputs_state)
                node_tasks.append(task)
            wait_tasks = node_tasks + [shutdown_task]
        except:
            traceback.print_exc()
            raise 
        finally:
            self._state = None

        try:
            while wait_tasks:
                (done, pending) = await asyncio.wait(
                    wait_tasks, return_when=asyncio.FIRST_COMPLETED)
                if shutdown_task in done:
                    await dm_comp.send_error("Flow shutdown", "")
                    is_shutdown = True
                    for task in pending:
                        await cancel_task(task)
                    break
                node_outputs: dict[str, dict[str, DataHandle]] = {}
                for task in done:
                    task_exc = task.exception()
                    if task_exc is not None:
                        raise task_exc  # this shouldn't happen since we already handle user exc in task.
                    res, success = task.result()
                    if not success:
                        continue
                    node_id = task.get_name()
                    if res is None:
                        res = {}  # for node without output, we support None.
                    node = nodes_can_schedule[node_id]
                    assert node.runtime is not None
                    handles = node.runtime.out_handles
                    node_out_valid = True
                    for handle in handles:
                        if not handle.is_optional:
                            if handle.name not in res:
                                await dm_comp.send_error(
                                    f"Node {node_id} compute return dict missing {handle.name}",
                                    "")
                                node_out_valid = False
                                break
                    if node_out_valid:
                        node_outputs[node_id] = res
                done_node_ids = list(task.get_name() for task in done)
                new_node_inputs = _get_next_node_inputs(
                    dm_comp.get_model(), flow.runtime, node_outputs)
                # rich.print("NODE OUTPUTS", node_outputs, )
                # rich.print("NEXT NODE INPUTS", new_node_inputs)

                await node_inputs_state.remove_and_merge(
                    done_node_ids, new_node_inputs)
                # rich.print("MERGED", node_inputs_state.get_current_node_inputs())

                pending_node_tasks = set(task for task in pending
                                         if task is not shutdown_task)
                pending_node_ids = set(task.get_name() for task in pending
                                       if task is not shutdown_task)

                # schedule next
                wait_inputs = state.comsume_wait_node_inputs()
                for n in wait_inputs:
                    if node_inputs_state.has_node_inputs(n):
                        CFLOW_LOGGER.warning(
                            "Input data of node %s is in both wait and schedule list, ignored.",
                            n)
                        wait_inputs.pop(n)
                node_inputs_state.force_add_new_inputs(wait_inputs)
                nodes_to_schedule: list[ComputeNodeModel] = [
                    flow.nodes[nid]
                    for nid in node_inputs_state.get_current_node_ids()
                    if nid not in pending_node_ids
                ]
                # print(done_node_ids, new_node_inputs, [n.id for n in nodes_to_schedule])
                for node in nodes_to_schedule:
                    if node.runtime is None:
                        node.runtime = node.get_node_runtime(dm_comp.get_model())
                nodes_can_schedule, assigned_execs = self._get_nodes_can_schedule(
                    nodes_to_schedule,
                    node_inputs_state.get_current_node_inputs(), executors)
                if not nodes_can_schedule:
                    CFLOW_LOGGER.warning("No node can be scheduled")
                    break
                node_tasks: list[asyncio.Task] = list(pending_node_tasks)
                for node in nodes_can_schedule.values():
                    task = self._get_node_scheduled_task(
                        flow, node, assigned_execs, executor_term_buffers, node_inputs_state)
                    node_tasks.append(task)
                if not node_tasks:
                    CFLOW_LOGGER.warning("No node can be scheduled")
                    break
                wait_tasks = node_tasks + [shutdown_task]
            print("Done")
            await node_inputs_state.release_all_handles()

        except Exception as exc:
            # await self.send_exception(exc)
            traceback.print_exc()
            raise exc
        finally:
            self._state = None

    @override
    async def schedule(self, flow: ComputeFlowModel,
                       node_inputs: dict[str, dict[str, Any]],
                       executors: Sequence[NodeExecutorBase],
                       executor_term_buffers: dict[str, TerminalBuffer],
                       shutdown_ev: asyncio.Event) -> Optional[asyncio.Task]:
        if self._state is not None:
            self._state.wait_node_inputs.update(node_inputs)
            return
        task = asyncio.create_task(
            self._schedule(flow, node_inputs, executors, executor_term_buffers, shutdown_ev))
        self._state = SimpleSchedulerState(wait_node_inputs={}, task=task)
        return task

    async def close(self):
        self._shutdown_ev.set()
        if self._state is not None and self._state.task is not None:
            await self._state.task
            self._state = None