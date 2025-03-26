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
from tensorpc.dock.components import mui 

class _NodeStateManager:
    def __init__(self):
        self._node_id_to_node_rt: dict[str, ComputeNodeRuntime] = {}
        self._node_id_to_impl_key: dict[str, str] = {}
        self._node_id_to_preview_layout: dict[str, Optional[mui.FlexBox]] = {}
        self._node_id_to_detail_layout: dict[str, Optional[mui.FlexBox]] = {}

    def process_node(self, node: ComputeFlowNodeModel, 
                node_impl_code: str):
        node_id = node.id 
        cur_impl_key = node.impl.code if node.key == "" else node.key
        if node_id in self._node_id_to_impl_key:
            prev_impl_key = self._node_id_to_impl_key[node_id]
            if prev_impl_key != cur_impl_key:
                self._node_id_to_node_rt.pop(node_id, None)
        if node_id not in self._node_id_to_node_rt:
            self._node_id_to_node_rt[node_id] = node.get_node_runtime_from_remote(node_impl_code)
            self._node_id_to_impl_key[node_id] = cur_impl_key
            runtime = self._node_id_to_node_rt[node_id]
            detail_layout = runtime.cnode.get_node_detail_layout(None)
            preview_layout = runtime.cnode.get_node_preview_layout(None)
            self._node_id_to_preview_layout[node_id] = preview_layout
            self._node_id_to_detail_layout[node_id] = detail_layout
        runtime = self._node_id_to_node_rt[node_id] 
        return runtime

    def clear(self):
        self._node_id_to_node_rt.clear()
        self._node_id_to_impl_key.clear()
        self._node_id_to_preview_layout.clear()
        self._node_id_to_detail_layout.clear()

class SingleProcNodeExecutor:
    def __init__(self):
        self._cached_data: dict[str, Any] = {}
        self._id_to_robjs: dict[str, AsyncRemoteManager] = {}

        self._node_state_mgr = _NodeStateManager()

        self._desp = ExecutorRemoteDesp.get_empty()

    def clear(self):
        self._cached_data.clear()
        self._id_to_robjs.clear()
        self._node_state_mgr.clear()
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
                node_impl_code: str,
                node_state_dict: dict[str, Any],
                node_state_ast: DraftASTNode, 
                inputs: dict[str, DataHandle], 
                removed_data_ids: Optional[set[str]] = None) -> Optional[dict[str, DataHandle]]:
        if removed_data_ids is not None:
            self.remove_cached_datas(removed_data_ids)
        node_id = node.id 
        runtime = self._node_state_mgr.process_node(node, node_impl_code)
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

        if isinstance(data, dict):
            data_handle_dict: dict[str, DataHandle] = {}
            for k, v in data.items():
                uuid_str = uuid.uuid4().hex
                uid = f"{self._desp.id}-{uuid_str}-{k}"
                self._cached_data[uid] = v
                data_handle_dict[k] = DataHandle(id=uid, executor_desp=self._desp)
            return data_handle_dict
        else:
            assert data is None, f"compute_func {compute_func} should return None or dict."
        return data

class TorchDistNodeExecutor:
    # TODO
    def __init__(self):
        pass 
    def clear(self):
        pass