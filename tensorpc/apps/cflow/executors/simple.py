import abc
import asyncio
import base64
import enum
import json
from re import L
import traceback
from typing import Any, Optional
from tensorpc.apps.cflow.coremodel import ResourceDesc
from tensorpc.apps.cflow.model import ComputeNodeModel
from tensorpc.apps.cflow.nodes.cnode.ctx import get_compute_flow_node_context
from tensorpc.apps.cflow.nodes.cnode.registry import get_registry_func_modules_for_remote
from tensorpc.autossh.core import SSHConnDesc
from tensorpc.core.annolib import Undefined, undefined
from tensorpc.core.asyncclient import AsyncRemoteManager
from tensorpc.core.datamodel.asdict import as_dict_no_undefined
from tensorpc.core.datamodel.draft import get_draft_ast_node
from tensorpc.dock.components import mui
from .base import NODE_EXEC_SERVICE, ExecutorType, NodeExecutorBase, DataHandle, ExecutorRemoteDesc, RemoteExecutorServiceKeys, RemoteGrpcDataHandle
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.dock.components.terminal import AsyncSSHTerminal
import dataclasses as dataclasses_plain
from typing_extensions import override




async def run_node_with_robj(
        robj: AsyncRemoteManager, node: ComputeNodeModel,
        inputs: dict[str, DataHandle]) -> Optional[dict[str, DataHandle]]:
    run_ctx = get_compute_flow_node_context()
    assert run_ctx is not None
    state = run_ctx.state
    state_dict = as_dict_no_undefined(state)
    state_draft = run_ctx.state_draft
    state_draft_ast = get_draft_ast_node(state_draft)
    assert node.runtime is not None
    node_impl_code = node.runtime.impl_code
    node_no_runtime = node.get_node_without_runtime()
    res, draft_update_ops = await robj.chunked_remote_call(
        RemoteExecutorServiceKeys.RUN_NODE.value, node_no_runtime,
        node_impl_code, state_dict, state_draft_ast, inputs)
    if res is None:
        return None 
    assert isinstance(res, dict)
    for k, handle in res.items():
        assert isinstance(handle, DataHandle)
        # construct remote handle
        new_handle = RemoteGrpcDataHandle(handle.id, handle.executor_desp, handle.data, handle.update_ops, robj)
        res[k] = new_handle
    return res


class FixedNodeExecutor(NodeExecutorBase):

    def __init__(self, robj: AsyncRemoteManager):
        self.robj = robj
        self._desp: Optional[ExecutorRemoteDesc] = None

    @override
    async def run_node(
            self, node: ComputeNodeModel,
            inputs: dict[str, DataHandle]) -> Optional[dict[str, DataHandle]]:
        if self._desp is None:
            self._desp = await self.robj.remote_call(
                RemoteExecutorServiceKeys.GET_DESP.value)
        return await run_node_with_robj(self.robj, node, inputs)


@dataclasses_plain.dataclass
class _SSHManagedRemoteObjectState:
    task: asyncio.Task
    robj: AsyncRemoteManager
    desc: ExecutorRemoteDesc


class _SSHManagedRemoteObject:

    def __init__(self, terminal: AsyncSSHTerminal, exec_id: str,
                 resource: ResourceDesc):
        self._exec_id = exec_id
        self._exec_rc = resource
        self._terminal: AsyncSSHTerminal = terminal

        self._state: Optional[_SSHManagedRemoteObjectState] = None

    async def close(self):
        if self._state is not None:
            await self._state.robj.shutdown()
            await self._state.robj.close()
            self._terminal._shutdown_ev.set()
            await self._state.task
        self._state = None
        await self._terminal.disconnect()

    async def _exec_rpc_waiter(self, term: AsyncSSHTerminal, cmd: str):
        try:
            res = await term.ssh_command_rpc(cmd)
        except:
            traceback.print_exc()
        finally:
            self._state = None

    def _get_cfg_encoded(self, desc: ExecutorRemoteDesc):
        serv_name = NODE_EXEC_SERVICE
        cfg = {
            serv_name: {
                "desc": dataclasses.asdict(desc),
            }
        }
        cfg_encoded = base64.b64encode(
            json.dumps(cfg).encode("utf-8")).decode("utf-8")
        return cfg_encoded


    async def get_or_create_remote_object_and_desp(
            self,
            ssh_desc: SSHConnDesc,
            init_cmds: Optional[list[str]] = None):
        if not self._terminal.is_connected():
            await self._terminal.connect_with_new_desc(ssh_desc,
                                                       init_cmds=init_cmds)
        if self._state is None:
            # other cluster-based executors can override this method to request resource first.
            url = ssh_desc.url_with_port.split(":")[0]
            port = -1
            for j in range(3):
                # allocate a port
                res = await self._terminal.ssh_command_rpc(
                    "python -m tensorpc.cli.free_port 1\n")
                if res.return_code == 0:
                    stdout = res.get_output().decode("utf-8").strip()
                    port = int(stdout)
                    break
            assert port != -1, f"failed to allocate port for ssh terminal {ssh_desc.url_with_port}"
            exec_url_with_port = f"{url}:{port}"
            desc = ExecutorRemoteDesc(self._exec_id, ExecutorType.SINGLE_PROC,
                                      exec_url_with_port, self._exec_rc)
            # launch executor service
            cfg_encoded = self._get_cfg_encoded(desc)
            # TODO only use http port
            cmd = (f" python -m tensorpc.serve {NODE_EXEC_SERVICE} "
                   f"--port={port} "
                   f"--serv_config_b64 '{cfg_encoded}'\n")
            robj = AsyncRemoteManager(exec_url_with_port)
            task = asyncio.create_task(
                self._exec_rpc_waiter(self._terminal, cmd))
            await robj.wait_for_channel_ready()
            await robj.remote_call(RemoteExecutorServiceKeys.IMPORT_REGISTRY_MODULES.value, get_registry_func_modules_for_remote())

            self._state = _SSHManagedRemoteObjectState(task, robj, desc)
        return self._state.robj, self._state.desc


class SSHCreationNodeExecutor(NodeExecutorBase):
    """Lazy create new ssh session for real executor service
    TODO: currently we require the server support remote forward
    """

    def __init__(self,
                 id: str,
                 resource: ResourceDesc,
                 ssh_desc: SSHConnDesc,
                 init_cmds: Optional[list[str]] = None,
                 enable_ssh_stdin: bool = False):
        """
        Args:
            id (str): executor id
            resource (ResourceDesc): resource desc
            url (str): ssh url
            user (str): ssh user
            password (str): ssh password
            init_cmds (Optional[list[str]]): commands to run after ssh connected
            enable_ssh_stdin (bool): whether to enable stdin for ssh terminal
                WARNING: should only be used for temp executor
        """
        super().__init__(id, resource)
        if init_cmds is not None:
            for cmd in init_cmds:
                assert cmd.endswith("\n"), f"command {cmd} should end with \\n"
        self._ssh_desc = ssh_desc
        # read-only terminal for user
        self._terminal: AsyncSSHTerminal = AsyncSSHTerminal(
            manual_connect=True,
            manual_disconnect=True,
            terminalId=id).prop(disableStdin=not enable_ssh_stdin)
        self._serv_rpc_state = _SSHManagedRemoteObject(self._terminal, id,
                                                       resource)
        self._init_cmds = init_cmds

    @override
    def get_ssh_terminal(self) -> AsyncSSHTerminal | None:
        return self._terminal

    async def close(self):
        await self._serv_rpc_state.close()
        return None

    def get_bottom_layout(self) -> Optional[mui.FlexBox]:
        return mui.VBox([self._terminal]).prop(width="100%",
                                               height="100%",
                                               overflow="auto")

    @override
    async def run_node(
            self, node: ComputeNodeModel,
            inputs: dict[str, DataHandle]) -> Optional[dict[str, DataHandle]]:
        robj, desc = await self._serv_rpc_state.get_or_create_remote_object_and_desp(
            self._ssh_desc, self._init_cmds)
        return await run_node_with_robj(robj, node, inputs)


class SSHTempExecutorBase(SSHCreationNodeExecutor):
    def __init__(self,
                 id: str,
                 init_cmds: Optional[list[str]] = None):
        super().__init__(id, ResourceDesc(), SSHConnDesc("", "", ""),
                         init_cmds=init_cmds, enable_ssh_stdin=True)

    @abc.abstractmethod
    def get_ssh_info_from_node_state(self, node_state: Any) -> SSHConnDesc:
        ...

    @override
    async def run_node(
            self, node: ComputeNodeModel,
            inputs: dict[str, DataHandle]) -> Optional[dict[str, DataHandle]]:
        run_ctx = get_compute_flow_node_context()
        assert run_ctx is not None
        state = run_ctx.state

        ssh_desc = self.get_ssh_info_from_node_state(state)
        exit_ev = asyncio.Event()
        await self._terminal.connect_with_new_desc(ssh_desc,
                                               init_cmds=self._init_cmds,
                                               exit_event=exit_ev)
        await exit_ev.wait()
        return None

    async def close(self):
        await self._terminal.disconnect()
        return None
