import traceback
from typing import Any, Optional
from tensorpc.apps.cflow.nodes.cnode.ctx import ComputeFlowNodeContext, enter_flow_ui_node_context_object
from tensorpc.core.asyncclient import AsyncRemoteManager
from tensorpc.apps.cflow.model import ComputeFlowNodeModel, ComputeNodeRuntime
from tensorpc.apps.cflow.executors.base import NodeExecutorBase, DataHandle, ExecutorRemoteDesp, get_serv_key_from_exec_type
import inspect 
import uuid

from tensorpc.core.datamodel.draft import capture_draft_update, draft_from_node_and_type
from tensorpc.core.datamodel.draftast import DraftASTNode 

class SingleProcNodeExecutor:
    def __init__(self):
        self._cached_data: dict[str, Any] = {}
        self._id_to_robjs: dict[str, AsyncRemoteManager] = {}
        self._node_id_to_node_rt: dict[str, ComputeNodeRuntime] = {}
        self._node_id_to_code: dict[str, str] = {}

        self._desp = ExecutorRemoteDesp.get_empty()

    def clear(self):
        self._cached_data.clear()
        self._id_to_robjs.clear()
        self._node_id_to_node_rt.clear()
        self._node_id_to_code.clear()
        self._desp = ExecutorRemoteDesp.get_empty()

    async def init_set_desp_and_remote_clients(self, desp: ExecutorRemoteDesp, id_to_desps: dict[str, ExecutorRemoteDesp]):
        self.clear()
        self._desp = desp
        for id, desp in id_to_desps.items():
            self._id_to_robjs[id] = AsyncRemoteManager(desp.url)

    def get_cached_data(self, data_id: str) -> Any:
        return self._cached_data[data_id]

    def remove_cached_datas(self, data_ids: set[str]):
        for data_id in data_ids:
            self._cached_data.pop(data_id, None)

    async def run_node(self, node: ComputeFlowNodeModel, 
                node_state_dict: dict[str, Any],
                node_state_ast: DraftASTNode, 
                inputs: dict[str, DataHandle], 
                removed_data_ids: Optional[set[str]] = None) -> DataHandle:
        if removed_data_ids is not None:
            self.remove_cached_datas(removed_data_ids)
        node_id = node.id 
        if node_id in self._node_id_to_code:
            prev_code = self._node_id_to_code[node_id]
            cur_code = node.impl.code
            if prev_code != cur_code:
                self._node_id_to_node_rt.pop(node_id, None)
        if node_id not in self._node_id_to_node_rt:
            self._node_id_to_node_rt[node_id] = node.get_node_runtime_from_remote()
            self._node_id_to_code[node_id] = node.impl.code
        runtime = self._node_id_to_node_rt[node_id] 
        cnode = runtime.cnode
        compute_func = cnode.get_compute_func()
        inputs_val = {}
        # TODO group data if they come froms same executor.
        for k, inp in inputs.items():
            if inp.has_data():
                inputs_val[k] = inp.data
            else:
                if inp.executor_desp.id == self._desp.id:
                    inputs_val[k] = self.get_cached_data(inp.id)
                else:
                    key = get_serv_key_from_exec_type(inp.executor_desp.type)
                    robj = self._id_to_robjs[inp.executor_desp.id]
                    inputs_val[k] = await robj.chunked_remote_call(f"{key}.get_cached_data", inp.id)
        with capture_draft_update() as ctx:
            if runtime.cfg.state_dcls is not None:
                try:
                    state_obj = runtime.cfg.state_dcls(**node_state_dict)
                except Exception as e:
                    traceback.print_exc()
                    state_obj = runtime.cfg.state_dcls()
                draft = draft_from_node_and_type(node_state_ast, runtime.cfg.state_dcls)
                with enter_flow_ui_node_context_object(ComputeFlowNodeContext(node_id, state_obj, draft)):
                    data = compute_func(**inputs_val)
                    if inspect.iscoroutine(data):
                        data = await data
            else:
                data = compute_func(**inputs_val)
                if inspect.iscoroutine(data):
                    data = await data
        data_uid = str(uuid.uuid4())
        self._cached_data[data_uid] = data
        return DataHandle(id=data_uid, executor_desp=self._desp, update_ops=ctx._ops)

class TorchDistNodeExecutor:
    # TODO
    def __init__(self):
        pass 
