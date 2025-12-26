import importlib
import sys
import traceback
from types import FrameType
from typing import Any, Optional, Union
from tensorpc.apps.cflow.nodes.cnode.ctx import ComputeFlowNodeContext, enter_flow_ui_node_context_object
from tensorpc.core.asyncclient import AsyncRemoteManager
from tensorpc.apps.cflow.model import ComputeNodeModel, ComputeNodeRuntime
from tensorpc.apps.cflow.executors.base import ExecutorType, NodeExecutorBase, DataHandle, ExecutorRemoteDesc, RemoteExecutorServiceKeys
import inspect 
import uuid
from tensorpc import prim
from tensorpc.core import BuiltinServiceProcType, inspecttools, marker

from tensorpc.core.datamodel.draft import DraftUpdateOp, capture_draft_update, draft_from_node_and_type
from tensorpc.core.datamodel.draftast import DraftASTNode
from tensorpc.core.serviceunit import ServiceEventType
from tensorpc.core.tree_id import UniqueTreeId
from tensorpc.dock.components import mui 
from tensorpc.dock.serv_names import serv_names as app_serv_names
from tensorpc.apps.dbg.components.bkptpanel import BreakpointDebugPanel
from tensorpc.apps.dbg.components.traceview import TraceView
from tensorpc.apps.dbg.bkpt import _try_get_distributed_meta
from tensorpc.apps.dbg.constants import (DebugServerProcessInfo, TENSORPC_DBG_FRAME_INSPECTOR_KEY,
                                    TENSORPC_DBG_TRACE_VIEW_KEY)
from tensorpc.utils.proctitle import set_tensorpc_server_process_title
from tensorpc.apps.dbg.serv_names import serv_names as dbg_serv_names

class _NodeStateManager:
    def __init__(self):
        self._node_id_to_node_rt: dict[str, ComputeNodeRuntime] = {}
        self._node_id_to_impl_key: dict[str, str] = {}
        self._node_id_to_preview_layout: dict[str, Optional[mui.FlexBox]] = {}
        self._node_id_to_detail_layout: dict[str, Optional[mui.FlexBox]] = {}

    async def process_node(self, node: ComputeNodeModel, 
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
            if preview_layout is not None:
                await prim.get_service(app_serv_names.REMOTE_COMP_SET_LAYOUT_OBJECT)(UniqueTreeId.from_parts([node_id, "preview"]).uid_encoded, preview_layout)
            if detail_layout is not None:
                await prim.get_service(app_serv_names.REMOTE_COMP_SET_LAYOUT_OBJECT)(UniqueTreeId.from_parts([node_id, "detail"]).uid_encoded, detail_layout)
        runtime = self._node_id_to_node_rt[node_id] 
        return runtime

    def clear(self):
        self._node_id_to_node_rt.clear()
        self._node_id_to_impl_key.clear()
        self._node_id_to_preview_layout.clear()
        self._node_id_to_detail_layout.clear()

class _RemoteObjectStateManager:
    def __init__(self):
        self._exec_id_to_robj: dict[str, AsyncRemoteManager] = {}

    async def clear(self):
        for robj in self._exec_id_to_robj.values():
            try:
                await robj.close()
            except:
                traceback.print_exc()
        self._exec_id_to_robj.clear()

    async def get_or_create_remote_obj(self, exec_desp: ExecutorRemoteDesc) -> AsyncRemoteManager:
        if exec_desp.id not in self._exec_id_to_robj:
            self._exec_id_to_robj[exec_desp.id] = AsyncRemoteManager(exec_desp.url)
        else:
            prev_robj = self._exec_id_to_robj[exec_desp.id]
            if prev_robj.url != exec_desp.url:
                try:
                    await prev_robj.close()
                except:
                    traceback.print_exc()
                self._exec_id_to_robj[exec_desp.id] = AsyncRemoteManager(exec_desp.url)
        return self._exec_id_to_robj[exec_desp.id]

class SingleProcNodeExecutor:
    def __init__(self, desc: ExecutorRemoteDesc):
        self._cached_data: dict[str, Any] = {}
        self._node_state_mgr = _NodeStateManager()
        self._remote_state_mgr = _RemoteObjectStateManager()
        self._desp = desc
        self._has_exc: bool = False

    def get_executor_remote_desp(self) -> ExecutorRemoteDesc: 
        return self._desp

    async def clear(self):
        self._cached_data.clear()
        self._node_state_mgr.clear()
        self._desp = ExecutorRemoteDesc.get_empty()
        await self._remote_state_mgr.clear()

    def get_cached_data(self, data_id: str) -> Any:
        return self._cached_data[data_id]

    def remove_cached_data(self, data_id: str):
        return self.remove_cached_datas({data_id})

    def remove_cached_datas(self, data_ids: set[str]):
        for data_id in data_ids:
            self._cached_data.pop(data_id, None)

    async def setup_node(self, node: ComputeNodeModel, 
                node_impl_code: str):
        await self._node_state_mgr.process_node(node, node_impl_code)

    async def run_node(self, node: ComputeNodeModel, 
                node_impl_code: str,
                node_state_dict: dict[str, Any],
                node_state_ast: DraftASTNode, 
                inputs: dict[str, DataHandle], 
                removed_data_ids: Optional[set[str]] = None) -> tuple[Optional[dict[str, DataHandle]], list[DraftUpdateOp]]:
        if removed_data_ids is not None:
            self.remove_cached_datas(removed_data_ids)
        # 
        if self._has_exc:
            dbg_serv = prim.get_service(dbg_serv_names.DBG_SET_EXTERNAL_FRAME)
            await dbg_serv(None)
            self._has_exc = False

        node_id = node.id 
        runtime = await self._node_state_mgr.process_node(node, node_impl_code)
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
                    robj = await self._remote_state_mgr.get_or_create_remote_obj(inp.executor_desp)
                    inputs_val[k] = await robj.chunked_remote_call(RemoteExecutorServiceKeys.GET_DATA, inp.id)
        with capture_draft_update() as ctx:
            if runtime.cfg.state_dcls is not None:
                try:
                    state_obj = runtime.cfg.state_dcls(**node_state_dict)
                except Exception as e:
                    traceback.print_exc()
                    state_obj = runtime.cfg.state_dcls()
                draft = draft_from_node_and_type(node_state_ast, Any)
                # print(runtime.cfg.state_dcls, draft, type(draft))

                with enter_flow_ui_node_context_object(ComputeFlowNodeContext(node_id, state_obj, draft)):
                    try:
                        data = compute_func(**inputs_val)
                        if inspect.iscoroutine(data):
                            data = await data
                    except:
                        _, _, exc_traceback = sys.exc_info()
                        dbg_serv = prim.get_service(dbg_serv_names.DBG_SET_EXTERNAL_FRAME)
                        frame: Optional[FrameType] = None
                        # walk to the innermost frame
                        for frame, _ in traceback.walk_tb(exc_traceback):
                            pass
                        if frame is None:
                            raise
                        self._has_exc = True
                        await dbg_serv(frame)
                        raise 
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
            return data_handle_dict, ctx._ops
        else:
            assert data is None, f"compute_func {compute_func} should return None or dict."
        return data, ctx._ops

class TorchDistNodeExecutor:
    # TODO
    def __init__(self, desc: ExecutorRemoteDesc):
        pass 

    async def clear(self):
        pass

    def get_cached_data(self, data_id: str) -> Any:
        raise NotImplementedError 

    def remove_cached_data(self, data_id: str):
        raise NotImplementedError 

    def remove_cached_datas(self, data_ids: set[str]):
        raise NotImplementedError 

    async def run_node(self, node: ComputeNodeModel, 
                node_impl_code: str,
                node_state_dict: dict[str, Any],
                node_state_ast: DraftASTNode, 
                inputs: dict[str, DataHandle], 
                removed_data_ids: Optional[set[str]] = None) -> tuple[Optional[dict[str, DataHandle]], list[DraftUpdateOp]]:
        raise NotImplementedError 

    async def setup_node(self, node: ComputeNodeModel, 
                node_impl_code: str):
        raise NotImplementedError
        
class NodeExecutorService:
    def __init__(self, desc: dict[str, Any]):
        desp_obj = ExecutorRemoteDesc(**desc)
        self._desp = desp_obj
        if desp_obj.type == ExecutorType.SINGLE_PROC:
            self._executor = SingleProcNodeExecutor(desp_obj)
        elif desp_obj.type == ExecutorType.TORCH_DIST:
            self._executor = TorchDistNodeExecutor(desp_obj)
        else:
            raise ValueError(f"unsupported executor type {desp_obj.type}")

    @marker.mark_server_event(event_type=marker.ServiceEventType.Init)
    async def _init(self):
        port = prim.get_server_grpc_port()
        set_tensorpc_server_process_title(
            BuiltinServiceProcType.SERVER_WITH_DEBUG, self._desp.id, str(port))
        panel = BreakpointDebugPanel().prop(flex=1)
        userdata = _try_get_distributed_meta()
        trace_view = TraceView(userdata).prop(flex=1)
        set_layout_service = prim.get_service(
            app_serv_names.REMOTE_COMP_SET_LAYOUT_OBJECT)
        await set_layout_service(TENSORPC_DBG_FRAME_INSPECTOR_KEY, panel)
        await set_layout_service(TENSORPC_DBG_TRACE_VIEW_KEY, trace_view)

    def import_registry_modules(self, registry_module_ids: list[str]):
        # import all modules in registry_module_ids
        for module_id in registry_module_ids:
            module_key = module_id.split("::")[0]
            importlib.import_module(module_key)

    def get_executor_remote_desp(self) -> ExecutorRemoteDesc: 
        return self._desp

    @marker.mark_server_event(event_type=ServiceEventType.Exit)
    async def close(self):
        await self.clear()

    async def clear(self):
        return await self._executor.clear()

    def get_cached_data(self, data_id: str) -> Any:
        return self._executor.get_cached_data(data_id)

    def remove_cached_data(self, data_id: str):
        return self._executor.remove_cached_data(data_id)

    def remove_cached_datas(self, data_ids: set[str]):
        return self._executor.remove_cached_datas(data_ids)

    async def run_node(self, node: ComputeNodeModel, 
                node_impl_code: str,
                node_state_dict: dict[str, Any],
                node_state_ast: DraftASTNode, 
                inputs: dict[str, DataHandle], 
                removed_data_ids: Optional[set[str]] = None) -> tuple[Optional[dict[str, DataHandle]], list[DraftUpdateOp]]:
        return await self._executor.run_node(node, node_impl_code, node_state_dict, node_state_ast, inputs, removed_data_ids)

    async def setup_node(self, node: ComputeNodeModel, 
                node_impl_code: str):
        await self._executor.setup_node(node, node_impl_code)
