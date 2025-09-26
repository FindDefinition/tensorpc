import abc
import asyncio
import base64
import contextlib
import enum
import json
from re import L
import traceback
from typing import Any, Awaitable, Callable, Optional
from tensorpc.apps.cflow.coremodel import ResourceDesc
from tensorpc.apps.cflow.model import ComputeNodeModel
from tensorpc.apps.cflow.nodes.cnode.ctx import get_compute_flow_node_context
from tensorpc.apps.cflow.nodes.cnode.registry import get_registry_func_modules_for_remote
from tensorpc.apps.dbg.components.dbgpanel import MasterDebugPanel
from tensorpc.apps.dbg.services.relay import RelayMonitorConfig
from tensorpc.autossh.core import SSHConnDesc
from tensorpc.core.annolib import Undefined, undefined
from tensorpc.core.asyncclient import AsyncRemoteManager
from tensorpc.core.asynctools import cancel_task
from tensorpc.core.datamodel.asdict import as_dict_no_undefined
from tensorpc.core.datamodel.draft import get_draft_ast_node
from tensorpc.dock.components import mui
from tensorpc.dock.components.plus.styles import get_tight_tab_theme_horizontal
from .base import NODE_EXEC_SERVICE, RELAY_SERVICE, ExecutorType, NodeExecutorBase, DataHandle, ExecutorRemoteDesc, RemoteExecutorServiceKeys, RemoteGrpcDataHandle
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.dock.components.terminal import AsyncSSHTerminal, SSHCloseError
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
    wait_task: asyncio.Task
    robj: AsyncRemoteManager
    relay_robj: AsyncRemoteManager
    desc: ExecutorRemoteDesc
    shutdown_ev: asyncio.Event


class _SSHManagedRemoteObject:

    def __init__(self, terminal: AsyncSSHTerminal, relay_terminal: AsyncSSHTerminal, exec_id: str,
                 resource: ResourceDesc):
        self._exec_id = exec_id
        self._exec_rc = resource
        self._terminal: AsyncSSHTerminal = terminal
        self._relay_terminal = relay_terminal

        self._state: Optional[_SSHManagedRemoteObjectState] = None

        self._wait_timeout = 20

    async def close_rpc_tasks(self):
        if self._state is not None:
            try:
                await self._state.robj.shutdown()
                await self._state.robj.close()
            except:
                traceback.print_exc()
            try:
                await self._state.relay_robj.shutdown()
                await self._state.relay_robj.close()
            except:
                traceback.print_exc()

            self._terminal._shutdown_ev.set()
            self._relay_terminal._shutdown_ev.set()
            await self._terminal.disconnect()
            await self._relay_terminal.disconnect()
            self._state = None

    async def close(self):
        if self._state is not None:
            if self._state.wait_task is not None:
                self._state.shutdown_ev.set()
                await self._state.wait_task
        self._state = None
        await self._terminal.disconnect()
        await self._relay_terminal.disconnect()

    async def _exec_rpc_waiter(self, shutdown_ev: asyncio.Event, term: AsyncSSHTerminal, relay_term: AsyncSSHTerminal, cmd: str, relay_cmd: str, relay_end_callback: Optional[Callable[[], Awaitable[None]]] = None):
        shutdown_task = asyncio.create_task(shutdown_ev.wait())
        term_cmd_task = asyncio.create_task(term.ssh_command_rpc(cmd))
        relay_cmd_task = asyncio.create_task(relay_term.ssh_command_rpc(relay_cmd))
        while True:
            try:
                done, pending = await asyncio.wait(
                    [shutdown_task, term_cmd_task, relay_cmd_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except asyncio.CancelledError:
                await cancel_task(shutdown_task)
                await cancel_task(term_cmd_task)
                await cancel_task(relay_cmd_task)
                break
            if shutdown_task in done:
                for t in pending:
                    await cancel_task(t)
                break
            if term_cmd_task in done:
                if term_cmd_task.exception() is not None:
                    traceback.print_exception(term_cmd_task.exception())
            if relay_cmd_task in done:
                if relay_cmd_task.exception() is not None:
                    traceback.print_exception(relay_cmd_task.exception())
            break 
        if relay_end_callback is not None:
            await relay_end_callback()
        await self.close_rpc_tasks()

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

    def _get_relay_cfg_encoded(self, pid: int, desc: RelayMonitorConfig):
        serv_name = RELAY_SERVICE
        cfg = {
            serv_name: {
                "observed_pid": pid,
                "config_dict": dataclasses.asdict(desc),
            }
        }
        cfg_encoded = base64.b64encode(
            json.dumps(cfg).encode("utf-8")).decode("utf-8")
        return cfg_encoded

    async def _setup_ssh_terminal_port(self, ssh_desc: SSHConnDesc, term: AsyncSSHTerminal):
        assert self._state is None 
        # other cluster-based executors can override this method to request resource first.
        port = -1
        for j in range(3):
            # allocate a port
            res = await term.ssh_command_rpc(
                "python -m tensorpc.cli.free_port 1\n")
            if res.return_code == 0:
                stdout = res.get_output().decode("utf-8").strip()
                port = int(stdout)
                break
            else:
                print(res.get_output().decode("utf-8"))

        assert port != -1, f"failed to allocate port for ssh terminal {ssh_desc.url_with_port}"
        return port

    def _get_remote_exec_cmd_and_desc(self, ssh_desc: SSHConnDesc, port: int):
        url = ssh_desc.url_with_port.split(":")[0]
        exec_url_with_port = f"{url}:{port}"
        desc = ExecutorRemoteDesc(self._exec_id, ExecutorType.SINGLE_PROC,
                                  exec_url_with_port, self._exec_rc)
        # launch executor service
        cfg_encoded = self._get_cfg_encoded(desc)
        # TODO only use http port
        cmd = (f" python -m tensorpc.serve {NODE_EXEC_SERVICE} "
               f"--port={port} "
               f"--serv_config_b64 '{cfg_encoded}'\n")
        return cmd, desc

    def _get_relay_cmd(self, port: int, remote_exec_pid: int):
        desc = RelayMonitorConfig()
        cfg_encoded = self._get_relay_cfg_encoded(remote_exec_pid, desc)
        # TODO only use http port
        cmd = (f" python -m tensorpc.serve {RELAY_SERVICE} "
               f"--port={port} "
               f"--serv_config_b64 '{cfg_encoded}'\n")
        return cmd

    @contextlib.asynccontextmanager
    async def _run_relay_with_ssh_directly(self, ssh_desc: SSHConnDesc,
            init_cmds: Optional[list[str]] = None):
        assert self._state is None
        assert not self._relay_terminal.is_connected()
        assert not self._terminal.is_connected()
        exit_ev = asyncio.Event()
        if init_cmds is None and ssh_desc.init_cmd != "":
            init_cmds = [ssh_desc.init_cmd]
        await self._terminal.connect_with_new_desc(ssh_desc,
                                                init_cmds=init_cmds,
                                                exit_event=exit_ev)

        await self._relay_terminal.connect_with_new_desc(ssh_desc,
                                                        init_cmds=init_cmds)
        relay_port = await self._setup_ssh_terminal_port(ssh_desc, self._relay_terminal)
        cur_term_state = self._terminal.get_current_state()
        url = ssh_desc.url_with_port.split(":")[0]
        if url.strip() != 'localhost':
            relay_port_local = await self._relay_terminal.forward_local_port(relay_port)
            relay_url_with_port = f"localhost:{relay_port_local}"
        else:
            relay_url_with_port = f"{url}:{relay_port}"

        assert cur_term_state is not None
        remote_exec_pid = cur_term_state.pid

        relay_cmd = self._get_relay_cmd(relay_port, remote_exec_pid)
        relay_task = asyncio.create_task(self._relay_terminal.ssh_command_rpc(relay_cmd))
        async with AsyncRemoteManager(relay_url_with_port) as relay_robj:
            await relay_robj.wait_for_channel_ready(timeout=self._wait_timeout)
            try:
                yield relay_robj, relay_task, exit_ev
            finally:
                await self._relay_terminal.disconnect()
                await self._terminal.disconnect()

    async def get_or_create_remote_object_and_desp(
            self,
            ssh_desc: SSHConnDesc,
            init_cmds: Optional[list[str]] = None,
            relay_start_callback: Optional[Callable[[AsyncRemoteManager], Awaitable[None]]] = None,
            relay_end_callback: Optional[Callable[[], Awaitable[None]]] = None):
        if init_cmds is None and ssh_desc.init_cmd != "":
            init_cmds = [ssh_desc.init_cmd]
        if self._state is None:
            if not self._terminal.is_connected():
                await self._terminal.connect_with_new_desc(ssh_desc,
                                                        init_cmds=init_cmds)
            if not self._relay_terminal.is_connected():
                await self._relay_terminal.connect_with_new_desc(ssh_desc,
                                                                init_cmds=init_cmds)

            # other cluster-based executors can override this method to request resource first.
            url = ssh_desc.url_with_port.split(":")[0]
            port = await self._setup_ssh_terminal_port(ssh_desc, self._terminal)
            relay_port = await self._setup_ssh_terminal_port(ssh_desc, self._relay_terminal)
            if url.strip() != 'localhost':
                port_local = await self._terminal.forward_local_port(port)
                relay_port_local = await self._relay_terminal.forward_local_port(relay_port)
                exec_url_with_port = f"localhost:{port_local}"
                relay_url_with_port = f"localhost:{relay_port_local}"
            else:
                exec_url_with_port = f"{url}:{port}"
                relay_url_with_port = f"{url}:{relay_port}"
            cur_term_state = self._terminal.get_current_state()
            assert cur_term_state is not None
            remote_exec_pid = cur_term_state.pid

            cmd, desc = self._get_remote_exec_cmd_and_desc(ssh_desc, port)
            relay_cmd = self._get_relay_cmd(relay_port, remote_exec_pid)
            robj = AsyncRemoteManager(exec_url_with_port)
            relay_robj = AsyncRemoteManager(relay_url_with_port)
            shutdown_ev = asyncio.Event()
            task = asyncio.create_task(
                self._exec_rpc_waiter(shutdown_ev, self._terminal, self._relay_terminal, cmd, relay_cmd, relay_end_callback))
            await robj.wait_for_channel_ready(timeout=self._wait_timeout)
            await robj.remote_call(RemoteExecutorServiceKeys.IMPORT_REGISTRY_MODULES.value, get_registry_func_modules_for_remote())
            await relay_robj.wait_for_channel_ready(timeout=self._wait_timeout)
            if relay_start_callback is not None:
                await relay_start_callback(relay_robj)
            self._state = _SSHManagedRemoteObjectState(task, robj, relay_robj, desc, shutdown_ev)
        return self._state.robj, self._state.relay_robj, self._state.desc


class SSHCreationNodeExecutor(NodeExecutorBase):
    """Lazy create new ssh session for real executor service
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
        self._ssh_desc = ssh_desc
        # read-only terminal for user
        self._terminal: AsyncSSHTerminal = AsyncSSHTerminal(
            manual_connect=True,
            manual_disconnect=True,
            terminalId=id,
            log_to_stdout=True).prop(disableStdin=not enable_ssh_stdin)
        self._relay_terminal: AsyncSSHTerminal = AsyncSSHTerminal(
            manual_connect=True,
            manual_disconnect=True,
            terminalId=id + "relay",
            log_to_stdout=True).prop(disableStdin=True)

        self._serv_rpc_state = _SSHManagedRemoteObject(self._terminal, self._relay_terminal, id,
                                                       resource)
        self._init_cmds = init_cmds
        self._debug_panel_box = mui.HBox([])
        self._debug_panel: Optional[MasterDebugPanel] = None
        self._debug_panel_box.event_after_mount.on(self._exec_layout_mount)
        # self._debug_panel_box.event_before_mount.on(self._exec_layout_unmount)

        self._ui_container_box = mui.HBox({
            "terminal": self._get_ui_terminals().prop(flex=1, overflow="hidden"),
            "debug": self._debug_panel_box.prop(flex=2),
        }).prop(width="100%", height="100%", overflow="hidden")

    async def _exec_layout_mount(self):
        if self._debug_panel is not None:
            await self._debug_panel_box.set_new_layout([
                self._debug_panel
            ])

    async def _exec_layout_unmount(self):
        await self._debug_panel_box.set_new_layout([])

    def _get_ui_terminals(self):
        tab_theme = get_tight_tab_theme_horizontal()
        tabdefs = [
            mui.TabDef("terminal",
                       "terminal",
                       self._terminal,
                       tooltip="terminal"),
            mui.TabDef("relay",
                       "relay",
                       self._relay_terminal,
                       tooltip="relay terminal (read only)"),
        ]
        tabs = mui.Tabs(tabdefs, init_value="terminal").prop(
                    panelProps=mui.FlexBoxProps(flex=1, padding=0, overflow="hidden"),
                    borderBottom=1,
                    borderColor='divider',
                    tooltipPlacement="bottom")
        res = mui.VBox([
            mui.ThemeProvider([
                tabs
            ], tab_theme)
        ])
        return res

    @override
    def get_ssh_terminal(self) -> AsyncSSHTerminal | None:
        return self._terminal

    async def close(self):
        await self._serv_rpc_state.close()
        return None

    def get_bottom_layout(self) -> Optional[mui.FlexBox]:
        return self._ui_container_box

    async def _relay_service_start(self, robj: AsyncRemoteManager):
        if self._debug_panel_box.is_mounted():
            await self._debug_panel_box.set_new_layout([
                MasterDebugPanel(relay_robj=robj).prop(flex=1)
            ])
        else:
            self._debug_panel = MasterDebugPanel(relay_robj=robj).prop(flex=1)

    async def _relay_service_end(self):
        if self._debug_panel_box.is_mounted():
            await self._debug_panel_box.set_new_layout([])
        else:
            self._debug_panel = None

    @override
    async def run_node(
            self, node: ComputeNodeModel,
            inputs: dict[str, DataHandle]) -> Optional[dict[str, DataHandle]]:
        robj, _, desc = await self._serv_rpc_state.get_or_create_remote_object_and_desp(
            self._ssh_desc, self._init_cmds, self._relay_service_start,
            self._relay_service_end)
        return await run_node_with_robj(robj, node, inputs)

    @override
    async def setup_node(self, node: ComputeNodeModel) -> None:
        robj, _, desc = await self._serv_rpc_state.get_or_create_remote_object_and_desp(
            self._ssh_desc, self._init_cmds, self._relay_service_start,
            self._relay_service_end)
        assert node.runtime is not None
        node_impl_code = node.runtime.impl_code
        node_no_runtime = node.get_node_without_runtime()

        await robj.remote_call(
            RemoteExecutorServiceKeys.SETUP_NODE.value, node_no_runtime,
                node_impl_code)


    def is_local(self) -> bool: 
        return False

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

        async with self._serv_rpc_state._run_relay_with_ssh_directly(
                ssh_desc, self._init_cmds) as (robj, relay_task, exit_ev):
            # wait for relay task to finish
            await self._relay_service_start(robj)
            await exit_ev.wait()
            await self._relay_service_end()
        try:
            await relay_task
        except SSHCloseError:
            pass 
        return None

    # async def close(self):
    #     await self._terminal.disconnect()
    #     return None
