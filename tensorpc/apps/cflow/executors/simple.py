from typing import Any, Optional
from tensorpc.apps.cflow.model import ComputeNodeModel
from tensorpc.apps.cflow.nodes.cnode.ctx import get_compute_flow_node_context
from tensorpc.core.annolib import Undefined, undefined
from tensorpc.core.asyncclient import AsyncRemoteManager
from tensorpc.core.datamodel.asdict import as_dict_no_undefined
from tensorpc.core.datamodel.draft import get_draft_ast_node
from .base import NodeExecutorBase, DataHandle, ExecutorRemoteDesp, RemoteExecutorServiceKeys
import inspect
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.dock.components.terminal import AsyncSSHTerminal

from typing_extensions import override


class FixedNodeExecutor(NodeExecutorBase):
    def __init__(self, robj: AsyncRemoteManager):
        self.robj = robj 
        self._desp: Optional[ExecutorRemoteDesp] = None

    @override
    async def run_node(self, node: ComputeNodeModel, inputs: dict[str, DataHandle]) -> Optional[dict[str, DataHandle]]:
        if self._desp is None:
            self._desp = await self.robj.remote_call(RemoteExecutorServiceKeys.GET_DESP.value)
        run_ctx = get_compute_flow_node_context()
        assert run_ctx is not None 
        state = run_ctx.state
        state_dict = as_dict_no_undefined(state)
        state_draft = run_ctx.state_draft
        state_draft_ast = get_draft_ast_node(state_draft)
        assert node.runtime is not None 
        node_impl_code = node.runtime.impl_code
        node_no_runtime = node.get_node_without_runtime()
        res = await self.robj.chunked_remote_call(RemoteExecutorServiceKeys.RUN_NODE.value, 
            node_no_runtime, node_impl_code, state_dict, state_draft_ast, inputs)
        return res

class _SSHManagedRemoteObject:
    def __init__(self):
        self._terminal: Optional[AsyncSSHTerminal] = None

    async def _handle_ssh_close(self):
        assert self._terminal is not None
        pass

    async def get_or_create_remote_object(self, url: str, user: str, password: str, init_cmds: Optional[list[str]] = None):
        if self._terminal is None:
            # other cluster-based executors can override this method to request resource first.
            self._terminal = AsyncSSHTerminal(url, user, password, manual_connect=True, manual_disconnect=True, init_size=(80, 24))
            self._terminal.event_ssh_conn_close
        
        return self._terminal


class SSHCreationNodeExecutor(NodeExecutorBase):
    """Lazy create new ssh session for real executor service
    TODO: currently we require the server support remote forward
    """
    def __init__(self, url: str, user: str, password: str, init_cmds: Optional[list[str]] = None):
        self._url = url
        self._user = user
        self._password = password

        self._terminal: Optional[AsyncSSHTerminal] = None
        self._init_cmds = init_cmds

    async def get_or_create_ssh_terminal(self):
        if self._terminal is None:
            # other cluster-based executors can override this method to request resource first.
            self._terminal = AsyncSSHTerminal(self._url, self._user, self._password, manual_connect=True, manual_disconnect=True)
        return self._terminal

    @override
    async def run_node(self, node: ComputeNodeModel, inputs: dict[str, DataHandle]) -> Optional[dict[str, DataHandle]]:

        ssh_terminal = get_or_create_ssh_terminal()
        if self._desp is None:
            self._desp = await self.robj.remote_call(RemoteExecutorServiceKeys.GET_DESP.value)
        run_ctx = get_compute_flow_node_context()
        assert run_ctx is not None 
        state = run_ctx.state
        state_dict = as_dict_no_undefined(state)
        state_draft = run_ctx.state_draft
        state_draft_ast = get_draft_ast_node(state_draft)
        assert node.runtime is not None 
        node_impl_code = node.runtime.impl_code
        node_no_runtime = node.get_node_without_runtime()
        res = await self.robj.chunked_remote_call(RemoteExecutorServiceKeys.RUN_NODE.value, 
            node_no_runtime, node_impl_code, state_dict, state_draft_ast, inputs)
        return res
