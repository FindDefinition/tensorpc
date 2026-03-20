import asyncio
import contextlib
from functools import partial
from typing import Any, Callable, Literal, Optional

import asyncssh
import grpc

from tensorpc.apps.cm.manager import ClusterProviderBase, ClusterSpec, NodeSpec
from tensorpc.apps.cm.coretypes import CM_LOGGER, ClusterBaseInfo, ClusterInfo, GroupCoarseStatus, GroupSSHStatus, ResourceInfo, TaskGroupInfo, WorkerInfo, WorkerUIType
from tensorpc.apps.cm.node_master import GroupSpec
from tensorpc.autossh.core import SSHConnDesc, enter_ssh_jumps
from tensorpc.core.annolib import Undefined
from tensorpc.core.asyncclient import AsyncRemoteManager
from tensorpc.core.distributed.raft import PeerInfo
from tensorpc.core.tree_id import UniqueTreeId
from tensorpc.dock.components import mui
import tensorpc.core.datamodel as D
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.apps.cm.serv_names import master_serv_names
import dataclasses as dataclasses_plain

from tensorpc.dock.components.plus.styles import get_tight_icon_tab_theme_horizontal

@dataclasses.dataclass
class ClusterRuntimeInfo(ClusterInfo):
    num_cpu_remaining: int 
    num_gpu_remaining: int 

@dataclasses.dataclass
class ClusterPanelState:
    cur_cluster: Optional[ClusterRuntimeInfo]
    clusters: list[ClusterRuntimeInfo]
    task_groups: list[TaskGroupInfo]

@dataclasses.dataclass
class GroupCreateModel:
    num_nodes: int 
    num_cpus_per_node: int
    num_gpus_per_node: int


class GroupCreateDialog(mui.Dialog):
    def __init__(self, draft: ClusterPanelState, callback: Optional[Callable[[mui.DialogCloseEvent], mui.CORO_NONE]] = None):
        cluster = mui.Autocomplete("Clusters", []).prop(textFieldProps=mui.TextFieldProps(muiMargin="dense"),
                                                           size="small", labelKey="id")
        self._cluster_select = cluster
        cluster_info_draft = (draft.cur_cluster.num_cpu_remaining, draft.cur_cluster.num_cpu, draft.cur_cluster.num_gpu_remaining, draft.cur_cluster.num_gpu)

        cluster.bind_draft_change(draft.cur_cluster)
        cluster.bind_fields(options=draft.clusters)
        # cluster_name_draft = D.literal_val("Create Group in %s (%s)") % (draft.cur_cluster.name, draft.cur_cluster.provider)
        # cluster_name = mui.Typography("").prop(variant="h6", alignSelf="center", value=cluster_name_draft)
        num_nodes_field = mui.NumberField(1, 2, 1).prop(label="Num Nodes", size="small", max=draft.cur_cluster.num_nodes)
        num_gpu_per_node = mui.NumberField(0, 8, 1, init_value=0).prop(label="Num GPUs", size="small")
        group_name = mui.TextField("Group Name").prop(size="small")
        workdir = mui.TextField("Work Dir").prop(size="small")

        self.num_nodes_field = num_nodes_field
        self.num_gpu_per_node = num_gpu_per_node
        self.group_name = group_name

        super().__init__([
            mui.VBox([
                mui.Markdown(":red[*]Select Cluster"),
                self._cluster_select.prop(flex=1),
                mui.Typography("").prop(variant="body2", value=D.not_null(draft.cur_cluster.provider, "Select one cluster"), muiColor=D.where(draft.cur_cluster.provider == None, "error", "inherit")),
                mui.Markdown("").prop(value=D.literal_val("`%d/%d` CPUs, `%d/%d` GPUs") % cluster_info_draft),
            

            ]),
            mui.VBox([
                mui.Markdown(":red[*]Group Name"),
                group_name,
            ]),

            mui.VBox([
                mui.Markdown(":red[*]Resource Configuration"),
                mui.HBox([
                    num_nodes_field.prop(flex=1),
                    num_gpu_per_node.prop(flex=1),
                ]).prop(margin="10px 0")

            ]),
            mui.VBox([
                mui.HBox([
                    mui.Markdown("Work Dir"),
                    mui.Icon(mui.IconType.Info).prop(tooltip="used for persisting app states.", iconSize="small", tooltipPlacement="top")
                ]),
                mui.HBox([
                    workdir.prop(flex=1, muiMargin="dense"),
                ])
            ]),

        ], callback)
        self.prop(dialogMaxWidth=False, fullWidth=False,
            width="75vw", height="80vh", gap=2,
            display="flex", flexDirection="column", overflow="hidden",)


class NodeCreateDialog(mui.Dialog):
    def __init__(self, providers: dict[str, ClusterProviderBase],
         callback: Optional[Callable[[mui.DialogCloseEvent], mui.CORO_NONE]] = None):
        self._providers = providers
        providers_keys = list(providers.keys())
        options = [(k, k) for k in providers_keys if providers[k].support_creation()]
        init_value = options[0][0] if options else None
        self.provider_select = mui.Select("Provider", 
            options, init_value=init_value).prop(flex=1)
        self.provider_ui_box = mui.VBox([
            
        ]).prop(flex=1, overflow="auto")
        super().__init__([
            self.provider_select,
            self.provider_ui_box,
        ], callback)
        self.prop(dialogMaxWidth=False, fullWidth=False,
            width="75vw", height="75vh",
            display="flex", flexDirection="column", overflow="hidden")

    async def _select_provider(self, provider_key: str):
        provider = self._providers[provider_key]
        provider_ui = await provider.get_create_ui()
        await self.provider_ui_box.set_new_layout([
            provider_ui,
        ])

    async def pop_provider_ui(self):
        childs = list(self.provider_ui_box._child_comps.values())
        assert len(childs) == 1
        provider_ui = childs[0]
        await self.provider_ui_box.set_new_layout([])
        return provider_ui

class GroupCard(mui.Paper):
    def __init__(self, group_info_draft: TaskGroupInfo, on_delete_group: Callable[[mui.Event], mui.CORO_ANY]):
        header_draft = D.literal_val("%s @ %s") % (group_info_draft.group_name, group_info_draft.cluster_name)
        self.header = mui.HBox([
            mui.FlexBox([
                mui.Typography().prop(variant="body1", value=header_draft, textOverflow="ellipsis", overflow="hidden", whiteSpace="nowrap", ),
            ]).prop(flex=1, minWidth=0),
            mui.Typography().prop(variant="caption", value=group_info_draft.status, muiColor=group_info_draft.color),
        ]).prop(takeDragRef=True, cursor="move", alignItems="baseline", minWidth=0)
        cluster_info_draft = (group_info_draft.num_nodes, group_info_draft.num_cpu, group_info_draft.num_gpu)
        # card get bigger when hover
        tags_chip = mui.Chip("tag").prop(muiColor="success",
                                        size="small",
                                        clickable=False,
                                        items=group_info_draft.tags)
        delete_btn = mui.IconButton(mui.IconType.Delete)
        delete_btn.prop(size="small", tooltip="Delete Task Group", 
                confirmTitle="Are you sure to delete this task group?", confirmMessage="This action cannot be undone.")
        delete_btn.event_click.on_standard(on_delete_group)
        self.dragable_box = mui.HBox([
            mui.VBox([
                self.header,
                mui.HDivider(),
                mui.Markdown("").prop(value=D.literal_val("`%s` Nodes, `%d` CPUs, `%d` GPUs") % cluster_info_draft),
                mui.HDivider(),
                tags_chip,
            ]).prop(flex=1, minWidth=0),
            mui.VDivider(),
            delete_btn,
        ])
        self.dragable_box.prop(draggable=True, flex=1, dragInChild=True, dragType="ClusterPanelTaskGroup", 
            dragData=group_info_draft.dragData, minWidth=0)
        super().__init__([
            self.dragable_box,
        ])
        self.prop(display="flex", flexDirection="column", padding="5px", margin="5px", elevation=4, minWidth=0)

class GroupRemoteLayout(mui.DockViewLayout):

    def __init__(self) -> None:
        super().__init__()
        self.event_drop.on(self._on_drop)
        self.event_close_tab.on(self._on_tab_close)
        # self.prop(font=mui.FlexLayoutFontProps(size="14px"), 
        #     allowedDndTypes=["ClusterPanelTaskGroup"],
        #     tabNameKey="name")
        self.prop(
            allowedDndTypes=["ClusterPanelTaskGroup"],
            tabNameKey="name")


    async def _on_drop(self, ev: Any):
        tab_id = ev["complexLayoutTabNodeId"]
        url_with_port = ev["dragData"]["url_with_port"]
        tensorpc_jumps = ev["dragData"].get("tensorpc_jump_urls", None)
        ssh_jumps = ev["dragData"].get("ssh_jumps", None)
        ext_url_with_port = ev["dragData"].get("external_url_with_port", None)
        if ssh_jumps is not None:
            ssh_jumps = [SSHConnDesc(**desc) for desc in ssh_jumps]
        if ext_url_with_port is not None:
            url_with_port = ext_url_with_port
        url = url_with_port.split(":")[0]
        port = int(url_with_port.split(":")[1])
        group_id = ev["dragData"]["group_id"]
        remote_box = mui.RemoteBoxGrpc(url, port, 
            UniqueTreeId.from_parts([group_id, WorkerUIType.TERMINAL.value]).uid_encoded,
            relay_urls=tensorpc_jumps,
            ssh_jumps=ssh_jumps)
        remote_box.prop(width="100%", height="100%", overflow="hidden")
        await self.update_childs({tab_id: remote_box})

    async def _on_tab_close(self, data):
        name = data["id"]
        if name in self._child_comps:
            await self.remove_childs_by_keys([name])

@dataclasses.dataclass
class GroupLocalResourceInfo:
    group_id: str
    info: ResourceInfo


@dataclasses.dataclass
class NodeRuntimeInfo:
    node_spec: NodeSpec
    # which group is using this node. empty means idle.
    group_infos: dict[str, GroupLocalResourceInfo]

    def get_worker_info(self, rank: int, resource_info: Optional[ResourceInfo] = None) -> WorkerInfo:
        peer_info = PeerInfo(
            uid=self.node_spec.id,
            url=self.node_spec.local_url_with_port,
        )
        return WorkerInfo(
            peer_info=peer_info,
            rank=rank,
            resource=resource_info,
        )

    def get_remain_resource(self) -> ResourceInfo:
        used_cpu = 0
        used_gpu = 0
        used_mem_gb = 0
        for group in self.group_infos.values():
            used_cpu += group.info.num_cpu
            used_gpu += group.info.num_gpu
            used_mem_gb += group.info.num_mem_gb
        remain_cpu = self.node_spec.resource_spec.num_cpu - used_cpu
        remain_gpu = self.node_spec.resource_spec.num_gpu - used_gpu
        remain_mem_gb = self.node_spec.resource_spec.num_mem_gb - used_mem_gb
        return ResourceInfo(num_cpu=remain_cpu, num_gpu=remain_gpu, num_mem_gb=remain_mem_gb, 
            gpu_type=self.node_spec.resource_spec.gpu_type)

@dataclasses_plain.dataclass
class ClusterSSHJumpConn:
    conn: Optional[asyncssh.SSHClientConnection]
    conn_ctx: contextlib.AbstractAsyncContextManager

@dataclasses_plain.dataclass
class ClusterBackendState:
    info: ClusterBaseInfo
    nodes: dict[str, NodeRuntimeInfo]
    groups: dict[str, GroupSpec]
    cluster_spec: ClusterSpec
    ssh_jump_conn: Optional[ClusterSSHJumpConn] = None
    lock: asyncio.Lock = dataclasses_plain.field(default_factory=asyncio.Lock)

    async def close(self):
        if self.ssh_jump_conn is not None:
            await self.ssh_jump_conn.conn_ctx.__aexit__(None, None, None)
            self.ssh_jump_conn = None

    async def cached_chunk_call_async(self,
                                    node: NodeSpec,
                                    key,
                                    *args,
                                    rpc_timeout=None,
                                    **kwargs):
        async with self.lock:
            url_with_port = node.external_url_with_port or node.local_url_with_port
            url_port = url_with_port.split(":")
            assert len(url_port) == 2
            url, port = url_port[0], int(url_port[1])
            if self.ssh_jump_conn is None:
                ctx_jump = enter_ssh_jumps(self.cluster_spec.ssh_jumps)
                conn = await ctx_jump.__aenter__()
                self.ssh_jump_conn = ClusterSSHJumpConn(conn=conn, conn_ctx=ctx_jump)
            conn = self.ssh_jump_conn.conn
            if conn is None:
                ctx = contextlib.nullcontext()
            else:
                ctx = conn.forward_local_port('', 0, url, port)
            async with ctx as listener:
                if listener is not None:
                    fwd_port = listener.get_port()
                    addr = f"localhost:{fwd_port}"
                else:
                    addr = url_with_port
                async with AsyncRemoteManager(addr) as robj:
                    res = await robj.chunked_remote_call(key,
                                                        *args,
                                                        rpc_timeout=rpc_timeout,
                                                        rpc_relay_urls=node.tensorpc_jump_urls,
                                                        **kwargs)
        return res


    def get_cluster_runtime_info(self):
        num_cpus = 0
        num_gpus = 0
        for node in self.nodes.values():
            num_cpus += node.node_spec.resource_spec.num_cpu
            num_gpus += node.node_spec.resource_spec.num_gpu
        num_cpus_remaining = num_cpus
        num_gpus_remaining = num_gpus
        for node in self.nodes.values():
            for group_info in node.group_infos.values():
                num_cpus_remaining -= group_info.info.num_cpu
                num_gpus_remaining -= group_info.info.num_gpu
        return ClusterRuntimeInfo(
            id=self.info.id,
            provider=self.info.provider,
            name=self.info.name,
            num_nodes=len(self.nodes),
            num_cpu=num_cpus,
            num_gpu=num_gpus,
            num_cpu_remaining=num_cpus_remaining,
            num_gpu_remaining=num_gpus_remaining,
            tags=[],
        )


def group_ssh_status_to_ui_repr(status: GroupSSHStatus) -> tuple[str, mui.StdColorNoDefault]:
    if status == GroupSSHStatus.HAS_DISCONNECTED:
        return ("disconnected", "error")
    elif status == GroupSSHStatus.HAS_RUNNING:
        return ("running", "success")
    elif status == GroupSSHStatus.HAS_PARTIAL_RUNNING:
        return ("running (partial)", "warning")
    elif status == GroupSSHStatus.HAS_PAUSED_PROCESS:
        return ("paused", "secondary")
    elif status == GroupSSHStatus.ALL_IDLE_WITH_LAST_ERROR:
        return ("idle (error)", "error")
    elif status == GroupSSHStatus.ALL_IDLE_WITHOUT_ERROR:
        return ("idle", "inherit")
    else:
        return ("unknown", "inherit")

class ClusterManagePanel(mui.FlexBox):

    def __init__(self, providers: dict[str, ClusterProviderBase]):
        self.providers = providers

        disable_create = all(not p.support_creation() for p in providers.values())

        self._create_dialog = NodeCreateDialog(providers, self._handle_create_dialog_close)
        self._group_dialog_container = mui.Fragment([])
        state = ClusterPanelState(
            cur_cluster=None,
            clusters=[],
            task_groups=[],
        )
        self._scan_timeout = 5
        self._group_query_timeout = 5
        self.dm = mui.DataModel(state, [])
        draft = self.dm.get_draft()
        draft_for_group = mui.DataModel.get_draft_external_type(TaskGroupInfo)
        self._remote_layout = GroupRemoteLayout()
        group_panel = mui.VBox([
            self._group_dialog_container,
            self._create_dialog,
            mui.DataFlexBox(GroupCard(draft_for_group, self._handle_delete_task_group))
                .prop(flex=1, overflowY="auto", minWidth=0, dataList=draft.task_groups, flexDirection="column")
        ]).prop(width="100%", height="100%", overflow="hidden", minWidth=0)
        cluster_info_panel = mui.VBox([
        ]).prop(width="100%", height="100%", overflow="hidden")
        tab_defs = [
            mui.TabDef("",
                       "group",
                        group_panel,
                       icon=mui.IconType.Task,
                       tooltip="Group Panel"),
            mui.TabDef("",
                       "clusters",
                        cluster_info_panel,
                       icon=mui.IconType.Lan,
                       tooltip="Clusters"),

        ]

        self._tabs = mui.Tabs(tab_defs, init_value="group").prop(panelProps=mui.FlexBoxProps(
                                  height="100%", padding=0, minWidth=0,),
                                                  orientation="horizontal",
                                                  borderBottom=1,
                                                  flex=1,
                                                  borderColor='divider',
                                                  # overflow="hidden",
                                                  tooltipPlacement="top")
        ctrl_panel = mui.ThemeProvider([
            mui.VBox([

                mui.HBox([
                    mui.Typography("Cluster Manager").prop(variant="body2", flex=1),
                    # self._cluster_select.prop(flex=1),
                    mui.IconButton(mui.IconType.Add, self._handle_create_nodes).prop(tooltip="Create Nodes", size="small", disabled=disable_create),
                    mui.IconButton(mui.IconType.AddTask, self._handle_create_task_group).prop(tooltip="Create Task Group", size="small"),

                ]).prop(alignItems="center"),
                mui.HDivider(),

                self._tabs.prop(flex=1)
            ]).prop(flex=1),
        ], get_tight_icon_tab_theme_horizontal())

        global_panel = mui.Allotment(mui.Allotment.ChildDef([
            mui.Allotment.Pane(ctrl_panel),
            mui.Allotment.Pane(mui.HBox([
                self._remote_layout,
            ]).prop(width="100%", height="100%", overflow="hidden")
            ),
        ])).prop(vertical=False, defaultSizes=[100, 300])

        self.dm.init_add_layout([
            # ctrl_panel,
            # mui.VDivider(),
            # mui.HBox([
            #     self._remote_layout,
            # ]).prop(flex=2, overflow="hidden")
            global_panel,
            # drop_box.prop(flex=2),
        ])
        draft = self.dm.get_draft()
        self._provider_to_clusters: dict[str, list[ClusterSpec]] = {
            k: [] for k in providers.keys()
        }
        self._cluster_states: dict[str, ClusterBackendState] = {}

        super().__init__([
            self.dm,
        ])

        self.prop(flexDirection="row", overflow="hidden")

        self.event_before_mount.on(self._handle_before_mount)
        self.event_before_unmount.on(self._handle_before_unmount)

        self._group_scan_task: Optional[asyncio.Task] = None
        self._group_query_task: Optional[asyncio.Task] = None
        self._shutdown_ev: Optional[asyncio.Event] = None
        self._scan_lock = asyncio.Lock()

    async def _handle_before_mount(self):
        self._shutdown_ev = asyncio.Event()
        self._group_scan_task = asyncio.create_task(self._scan_loop(self._shutdown_ev)) 
        self._group_query_task = asyncio.create_task(self._group_query_loop(self._shutdown_ev))

    async def _handle_before_unmount(self):
        if self._shutdown_ev is not None:
            self._shutdown_ev.set()
            self._shutdown_ev = None 
            self._group_scan_task = None 
            self._group_query_task = None
        for state in self._cluster_states.values():
            await state.close()

    async def _scan_loop(self, shutdown_ev: asyncio.Event):
        # do group scan.
        # currently user should call discover by themselves.
        shutdown_task = asyncio.create_task(shutdown_ev.wait())
        is_start = False
        # first scan should be fast.
        scan_timeout = 0.1
        while True:
            if is_start:
                scan_timeout = self._scan_timeout
            wait_task = asyncio.create_task(asyncio.sleep(scan_timeout))
            done, pending = await asyncio.wait([shutdown_task, wait_task], return_when=asyncio.FIRST_COMPLETED)
            if shutdown_task in done:
                break
            async with self._scan_lock:
                try:
                    await self._all_provider_discover()
                    cluster_states = self._create_cluster_states()
                    cluster_groups = await self._sync_scan_groups(cluster_states)
                    for cluster_id, state in cluster_states.items():
                        state.groups = cluster_groups.get(cluster_id, {})
                    await self._merge_cluster_states(cluster_states)
                    await self._sync_to_frontend()
                except grpc.aio.AioRpcError as e:
                    CM_LOGGER.error(f"gRPC error in scan loop: {e}")
                    continue
                except Exception as e:
                    CM_LOGGER.exception(f"Unknown Error in scan loop: {e}")
                    continue
            is_start = True

    async def _group_query_loop(self, shutdown_ev: asyncio.Event):
        # do group query on existed groups.
        shutdown_task = asyncio.create_task(shutdown_ev.wait())
        while True:
            wait_task = asyncio.create_task(asyncio.sleep(self._group_query_timeout))
            done, pending = await asyncio.wait([shutdown_task, wait_task], return_when=asyncio.FIRST_COMPLETED)
            if shutdown_task in done:
                break
            async with self._scan_lock:
                async with self.dm.draft_update() as draft:
                    for cluster_id, state in self._cluster_states.items():
                        groups = state.groups
                        for group_id, group_spec in groups.items():
                            leader_id = group_spec.leader_id
                            if leader_id is None:
                                continue
                            leader_node = state.nodes.get(leader_id, None)
                            if leader_node is None:
                                CM_LOGGER.warning(f"Leader node {leader_id} for group {group_id} not found in cluster {cluster_id}")
                                continue 
                            try:
                                group_status: GroupCoarseStatus = await state.cached_chunk_call_async(
                                    leader_node.node_spec,
                                    master_serv_names.GROUP_QUERY_COARSE_STATUS, 
                                    group_id=group_id, 
                                    rpc_timeout=5)
                            except grpc.aio.AioRpcError as e:
                                CM_LOGGER.error(f"Failed to query group status for group {group_id} in cluster {cluster_id}: {e}")
                                continue
                            if not group_status.success:
                                if not group_status.is_raft_node:
                                    CM_LOGGER.warning(f"Group {group_id} in cluster {cluster_id} has no raft node.")
                                    continue
                                elif group_status.leader_info is not None:
                                    group_spec.leader_id = group_status.leader_info.uid
                                    leader_node = state.nodes.get(group_spec.leader_id, None)
                                    if leader_node is None:
                                        CM_LOGGER.warning(f"Leader node {group_spec.leader_id} for group {group_id} not found in cluster {cluster_id}")
                                        continue
                                    try:
                                        group_status: GroupCoarseStatus = await state.cached_chunk_call_async(
                                            leader_node.node_spec,
                                            master_serv_names.GROUP_QUERY_COARSE_STATUS, 
                                            group_id=group_id, 
                                            rpc_timeout=5)
                                    except grpc.aio.AioRpcError as e:
                                        CM_LOGGER.error(f"Failed to query group status for group {group_id} in cluster {cluster_id} after getting leader info: {e}")
                                        continue
                            if group_status.success:
                                status, color = group_ssh_status_to_ui_repr(GroupSSHStatus(group_status.group_ssh_status))
                                draft.task_groups[group_spec.index].status = status
                                draft.task_groups[group_spec.index].color = color


    async def _sync_scan_groups_and_update(self):
        cluster_states = self._create_cluster_states()
        cluster_groups = await self._sync_scan_groups(cluster_states)
        for cluster_id, state in cluster_states.items():
            state.groups = cluster_groups.get(cluster_id, {})
        await self._sync_to_frontend()


    def _create_cluster_states(self):
        cluster_states: dict[str, ClusterBackendState] = {}
        for provider_key, cspecs in self._provider_to_clusters.items():
            for cspec in cspecs:
                state = ClusterBackendState(cspec.cluster_info, nodes={}, groups={}, cluster_spec=cspec)
                for node in cspec.nodes:
                    state.nodes[node.id] = NodeRuntimeInfo(node_spec=node, group_infos={})
                cluster_states[cspec.id] = state
        return cluster_states

    async def _merge_cluster_states(self, new_states: dict[str, ClusterBackendState]):
        # merge clusters
        cluster_states = self._cluster_states.copy()
        self._cluster_states.clear()
        removed_old_cluster_ids = set(cluster_states.keys()) - set(new_states.keys())
        for cluster_id in removed_old_cluster_ids:
            old_state = cluster_states[cluster_id]
            await old_state.close()
        for cluster_id, new_state in new_states.items():
            if cluster_id in cluster_states:
                old_state = cluster_states[cluster_id]
                # only update nodes, other info is fixed after creation.
                if not old_state.cluster_spec.is_ssh_jump_equal(new_state.cluster_spec):
                    await old_state.close()
            self._cluster_states[cluster_id] = new_state

    async def _update_provider_nodes(self, provider_key: str, clusters: list[ClusterSpec]):
        self._provider_to_clusters[provider_key] = clusters

    async def _handle_create_dialog_close(self, ev: mui.DialogCloseEvent):
        if not ev.ok:
            return
        assert isinstance(ev.userData, str)
        provider_key = ev.userData
        provider = self.providers[provider_key]

        provider_ui = await self._create_dialog.pop_provider_ui()
        assert isinstance(provider_ui, mui.FlexBox)
        await provider.create_nodes(provider_ui)
        cur_provider_nodes = await provider.discover()
        await self._update_provider_nodes(provider_key, cur_provider_nodes)

    async def _handle_create_nodes(self):
        pass 

    async def _all_provider_discover(self):
        for provider_key, provider in self.providers.items():
            nodes = await provider.discover()
            await self._update_provider_nodes(provider_key, nodes)

    async def _sync_to_frontend(self):
        model = self.dm.model
        async with self.dm.draft_update() as draft:
            draft.clusters = [state.get_cluster_runtime_info() for state in self._cluster_states.values()]
            if model.cur_cluster is not None:
                # if current cluster is deleted, set it to None.
                if model.cur_cluster.id not in self._cluster_states:
                    draft.cur_cluster = None 
                else:
                    draft.cur_cluster = self._cluster_states[model.cur_cluster.id].get_cluster_runtime_info()
            task_groups: list[TaskGroupInfo] = []
            for state in self._cluster_states.values():
                node_id_to_nodespec = {node.node_spec.id: node.node_spec for node in state.nodes.values()}
                for group_id, group_spec in state.groups.items():
                    num_cpu = 0
                    num_gpu = 0
                    tags: list[mui.ChipGroupItem] = []
                    num_gpu_types = set()
                    for cnode in group_spec.compute_node_specs:
                        if cnode.resource is not None:
                            num_cpu += cnode.resource.num_cpu
                            num_gpu += cnode.resource.num_gpu
                            if cnode.resource.gpu_type is not None:
                                num_gpu_types.add(cnode.resource.gpu_type)
                    if not num_gpu_types:
                        tags.append(mui.ChipGroupItem(label="CPU Only"))
                    else:
                        for gpu_type in num_gpu_types:
                            tags.append(mui.ChipGroupItem(label=gpu_type))
                    if group_spec.leader_id is not None:
                        raft_node_spec = node_id_to_nodespec[group_spec.leader_id]
                    else:
                        raft_node_spec = node_id_to_nodespec[group_spec.raft_node_specs[0].peer_info.uid]
                    group_spec.index = len(task_groups)
                    status, color = group_ssh_status_to_ui_repr(GroupSSHStatus(group_spec.status))
                    task_groups.append(TaskGroupInfo(
                        id=UniqueTreeId.from_parts([state.info.id, group_id]).uid_encoded,
                        group_name=group_id,
                        cluster_id=state.info.id,
                        cluster_name=state.info.name,
                        num_nodes=len(group_spec.compute_node_specs),
                        num_cpu=num_cpu,
                        num_gpu=num_gpu,
                        status=status,
                        color=color,
                        cur_cmd="",
                        tags=tags,
                        dragData={
                            "name": f"{group_id} @ {state.info.name}",
                            "group_id": group_id,
                            "url_with_port": raft_node_spec.local_url_with_port,
                            "ssh_jumps": state.cluster_spec.ssh_jumps,
                            "tensorpc_jump_urls": raft_node_spec.tensorpc_jump_urls,
                            "external_url_with_port": raft_node_spec.external_url_with_port,
                        }
                    ))
            draft.task_groups = task_groups 


    async def _scan_group_update(self, state: ClusterBackendState, nodes_dict: dict[str, NodeRuntimeInfo]):
        all_node_urls = [node.node_spec.local_url_with_port for node in nodes_dict.values()]
        first_node_spec = list(nodes_dict.values())[0].node_spec
        self_url = all_node_urls[0]
        kwargs = {
            "self_url": self_url,
            "all_node_urls": all_node_urls,
            "num_partition": 8,
        }
        group_specs: dict[str, GroupSpec] = await state.cached_chunk_call_async(
            first_node_spec, master_serv_names.GROUP_TREE_SCAN_GROUPS,  
            rpc_timeout=30, **kwargs)
        for group_id, group_spec in group_specs.items():
            for cnode in group_spec.compute_node_specs:
                node_id = cnode.peer_info.uid
                if node_id not in nodes_dict:
                    CM_LOGGER.warning(f"node {node_id} in group {group_id} not found in cluster nodes.")
                    continue
                rt_info = nodes_dict[node_id]
                if cnode.resource is not None:
                    resource = cnode.resource
                else:
                    resource = ResourceInfo(num_cpu=0, num_mem_gb=0, num_gpu=0)
                ginfo = GroupLocalResourceInfo(
                    group_id=group_id, info=resource)
                rt_info.group_infos[group_id] = ginfo
        return group_specs

    async def _delete_group(self, cluster_id: str, group_id: str):
        cluster_state = self._cluster_states[cluster_id]
        group_spec = cluster_state.groups[group_id]
        all_node_urls: list[str] = []
        entry_node: Optional[NodeRuntimeInfo] = None
        for cnode in group_spec.raft_node_specs + group_spec.compute_node_specs:
            node_id = cnode.peer_info.uid
            if node_id not in cluster_state.nodes:
                CM_LOGGER.warning(f"node {node_id} in group {group_id} not found in cluster nodes.")
                continue
            rt_info = cluster_state.nodes[node_id]
            if entry_node is None:
                entry_node = rt_info
            all_node_urls.append(rt_info.node_spec.local_url_with_port)
        assert entry_node is not None
        self_url = entry_node.node_spec.local_url_with_port
        kwargs = {
            "self_url": self_url,
            "all_node_urls": all_node_urls,
            "num_partition": 8,
            "group_id": group_id,
            "remove_stopped_group": False,
        }
        await cluster_state.cached_chunk_call_async(
            entry_node.node_spec, master_serv_names.GROUP_TREE_REMOVE_GROUP, 
            rpc_timeout=30, **kwargs)
        kwargs["remove_stopped_group"] = True 
        await cluster_state.cached_chunk_call_async(
            entry_node.node_spec, master_serv_names.GROUP_TREE_REMOVE_GROUP, 
            rpc_timeout=30, **kwargs)

        # remove group info 
        for cnode in group_spec.raft_node_specs + group_spec.compute_node_specs:
            node_id = cnode.peer_info.uid
            if node_id not in cluster_state.nodes:
                continue
            rt_info = cluster_state.nodes[node_id]
            if group_id in rt_info.group_infos:
                del rt_info.group_infos[group_id]
        del cluster_state.groups[group_id]

    async def _sync_scan_groups(self, cluster_states: dict[str, ClusterBackendState]):
        cluster_groups: dict[str, dict[str, GroupSpec]] = {}
        for cid, state in cluster_states.items():
            if not state.nodes:
                continue 
            group_specs = await self._scan_group_update(state, state.nodes)
            cluster_groups[cid] = group_specs
        return cluster_groups

    async def _handle_create_task_group(self):
        model = self.dm.model 
        group_create_dialog = GroupCreateDialog(self.dm.get_draft())
        group_create_dialog.event_modal_close.on(partial(self._handle_group_create_dialog_close, dialog=group_create_dialog))
        await self._group_dialog_container.set_new_layout([
            group_create_dialog,
        ])
        await group_create_dialog.set_open(True)

    def _get_num_raft_nodes(self, num_workers: int):
        if num_workers <= 4:
            return 1
        elif num_workers <= 64:
            return 3 
        else:
            return 5

    async def _handle_group_create_dialog_close(self, ev: mui.DialogCloseEvent, dialog: GroupCreateDialog):
        await self._group_dialog_container.set_new_layout([])
        if not ev.ok:
            return 
        assert self.dm.model.cur_cluster is not None, "should select a cluster first"

        num_nodes = dialog.num_nodes_field.value 
        num_gpus_per_node = dialog.num_gpu_per_node.value
        assert not isinstance(num_nodes, Undefined) and not isinstance(num_gpus_per_node, Undefined)
        cur_rc = ResourceInfo(num_cpu=0, num_gpu=int(num_gpus_per_node), num_mem_gb=0)

        compute_node_infos: list[WorkerInfo] = []
        cur_cluster = self.dm.model.cur_cluster
        num_nodes = int(num_nodes)
        cur_cluster_state = self._cluster_states[cur_cluster.id]
        group_name = dialog.group_name.value
        assert group_name not in cur_cluster_state.groups, f"group name {group_name} already exists in cluster {cur_cluster.name}"
        assert num_nodes <= len(cur_cluster_state.nodes), f"num nodes exceed cluster capacity {len(cur_cluster_state.nodes)}"
        rank_cnt = 0
        num_raft_node = self._get_num_raft_nodes(num_nodes)
        # TODO currently we use compute nodes as raft nodes.
        first_raft_info: Optional[NodeRuntimeInfo] = None
        raft_peer_infos: list[PeerInfo] = []
        for node_info in cur_cluster_state.nodes.values():
            if len(compute_node_infos) >= num_nodes:
                break 
            remain_rc = node_info.get_remain_resource()
            if not remain_rc.is_sufficient_for(cur_rc):
                continue
            worker_info = node_info.get_worker_info(rank_cnt, cur_rc)
            compute_node_infos.append(worker_info)
            rank_cnt += 1
            if len(raft_peer_infos) < num_raft_node:
                if not raft_peer_infos:
                    first_raft_info = node_info
                raft_peer_infos.append(worker_info.peer_info)
        assert first_raft_info is not None
        if len(compute_node_infos) < num_nodes:
            CM_LOGGER.error(f"not enough resource to create group, required {num_nodes} nodes with {cur_rc}, but only found {len(compute_node_infos)}")
            return
        self_uid = raft_peer_infos[0].uid
        kwargs = {
            "group_id": group_name,
            "self_uid": self_uid,
            "compute_node_infos": compute_node_infos,
            "raft_node_infos": raft_peer_infos,
            "num_partition": 8,
            "worker_cfg": None
        }
        await cur_cluster_state.cached_chunk_call_async(
            first_raft_info.node_spec,
            master_serv_names.GROUP_MASTER_CREATE_GROUP, 
            rpc_timeout=30, **kwargs)
        await self._sync_scan_groups_and_update()

    async def _handle_delete_task_group(self, ev: mui.Event):
        group_uid = ev.get_keys_checked()[0] 
        parts = UniqueTreeId(group_uid).parts 
        cluster_id = parts[0]
        group_id = parts[1]
        cluster_state = self._cluster_states.get(cluster_id, None)
        if cluster_state is None:
            CM_LOGGER.error(f"Cluster {cluster_id} not found for deleting group {group_id}")
            return
        group_spec = cluster_state.groups.get(group_id, None)
        if group_spec is None:
            CM_LOGGER.error(f"Group {group_id} not found in cluster {cluster_id} for deleting")
            return
        await self._delete_group(cluster_id, group_id)
        await self._sync_scan_groups_and_update()
        