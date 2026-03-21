

import asyncio
import contextlib
import os
import traceback
import yaml
from tensorpc.apps.cm.coretypes import ClusterBaseInfo, ResourceInfo, WorkerInfo, WorkerUIType
from tensorpc.apps.cm.manager import ClusterSpec, FixedClusterProvider, NodeSpec
from tensorpc.constants import TENSORPC_DEV_SECRET_PATH
from tensorpc.core import asyncclient
from tensorpc.core.distributed.raft import PeerInfo
from tensorpc.core.tree_id import UniqueTreeId
from tensorpc.utils.wait_tools import get_free_ports
import subprocess
from tensorpc.apps.cm.serv_names import master_serv_names
from tensorpc.dock import mui, three, plus, mark_create_layout, appctx


from tensorpc.apps.cm.components.cluster_panel import ClusterManagePanel

class LocalClusterBase:
    def __init__(self):
        self._procs: list[subprocess.Popen] = []
        self._peer_infos: list[PeerInfo] = []


    async def _create_processes(self, num_node: int):
        with open(TENSORPC_DEV_SECRET_PATH, "r") as f:
            secret = yaml.safe_load(f)
        cm_debug = secret["cm_debug"]
        username = cm_debug["username"]
        password = cm_debug["password"]
        ports = get_free_ports(num_node)
        procs: list[subprocess.Popen] = []
        peer_infos: list[PeerInfo] = []
        for i in range(num_node):
            port = ports[i]
            uid = f"node-{i}"
            url = f"localhost:{port}"
            proc = subprocess.Popen(
                f"python -m tensorpc.apps.cm.cli --uid=\"{uid}\" --port {port} --username {username} --password {password}",
                shell=True)
            peer_info = PeerInfo(uid=uid, url=url)
            procs.append(proc)
            peer_infos.append(peer_info)
        self._procs = procs
        self._peer_infos = peer_infos
        all_node_urls = [peer_info.url for peer_info in peer_infos]
        async with asyncclient.AsyncRemoteManager(all_node_urls[0]) as robj:
            await robj.wait_for_channel_ready(timeout=20)
            kwargs = {
                "self_url": all_node_urls[0],
                "all_node_urls": all_node_urls,
                "num_partition": 2,
            }

            await robj.remote_call(
                master_serv_names.GROUP_TREE_SCAN_GROUPS,
                **kwargs
            )
            group_id = "test-group"
            num_raft = 1
            rc = ResourceInfo(num_cpu=32, num_mem_gb=512, num_gpu=8)
            compute_node_infos = [WorkerInfo(peer_info=peer_info, rank=i, resource=rc) for i, peer_info in enumerate(peer_infos)]
            raft_node_infos = [peer_infos[i] for i in range(num_raft)]
            
            kwargs = {
                "group_id": group_id,
                "self_uid": peer_infos[0].uid,
                "compute_node_infos": compute_node_infos,
                "raft_node_infos": raft_node_infos,
                "num_partition": 2,
            }


            await robj.remote_call(
                master_serv_names.GROUP_MASTER_CREATE_GROUP,
                **kwargs
            )

        return group_id, ports, peer_infos

    async def _handle_before_unmount(self):
        # for peer_info in self._peer_infos:
        #     async with asyncclient.AsyncRemoteManager(peer_info.url) as robj:
        #         try:
        #             # await robj.wait_for_channel_ready()
        #             await robj.shutdown()
        #         except:

        #             traceback.print_exc()
        for proc in self._procs:
            proc.wait()

class App(LocalClusterBase):

    @mark_create_layout
    def my_layout(self):

        self._root_box = mui.VBox([
        ]).prop(width="100%", height="100%", overflow="hidden")
        # remote_box = mui.RemoteBoxGrpc("localhost", TENSORPC_APPS_DISTSSH_DEFAULT_PORT, TENSORPC_DISTSSH_UI_KEY)
        self._root_box.event_before_mount.on(self._handle_before_mount)
        self._root_box.event_before_unmount.on(self._handle_before_unmount)
        self._procs: list[subprocess.Popen] = []
        self._peer_infos: list[PeerInfo] = []
        return self._root_box

    async def _handle_before_mount(self):
        group_id, ports, peer_infos = await self._create_processes(6)
        self._peer_infos = peer_infos
        all_node_urls = [peer_info.url for peer_info in peer_infos]

        async with asyncclient.AsyncRemoteManager(all_node_urls[0]) as robj:
            await robj.wait_for_channel_ready()
            remote_box = mui.RemoteBoxGrpc("localhost", ports[0], 
                UniqueTreeId.from_parts([group_id, WorkerUIType.TERMINAL.value]).uid_encoded)
            await self._root_box.set_new_layout([
                remote_box.prop(flex=1)
            ])


class ManagerApp(LocalClusterBase):

    @mark_create_layout
    def my_layout(self):
        self._provider = FixedClusterProvider([])
        self._panel = ClusterManagePanel({
            "local": self._provider,
        })
        self._root_box = mui.VBox([
            self._panel.prop(flex=1)
        ]).prop(width="100%", height="100%", overflow="hidden")
        # remote_box = mui.RemoteBoxGrpc("localhost", TENSORPC_APPS_DISTSSH_DEFAULT_PORT, TENSORPC_DISTSSH_UI_KEY)
        self._root_box.event_before_mount.on(self._handle_before_mount)
        self._root_box.event_before_unmount.on(self._handle_before_unmount)
        # self._procs: list[subprocess.Popen] = []
        # self._peer_infos: list[PeerInfo] = []
        return self._root_box

    async def _handle_before_mount(self):
        group_id, ports, peer_infos = await self._create_processes(4)
        self._peer_infos = peer_infos
        node_specs: list[NodeSpec] = []
        for peer_info in peer_infos:
            node_spec = NodeSpec(
                id=peer_info.uid,
                tags=[],
                resource_spec=ResourceInfo(num_cpu=32, num_mem_gb=512, num_gpu=8),
                local_url_with_port=peer_info.url,
            )
            node_specs.append(node_spec)
        cluster_spec = ClusterSpec(
            id="local-cluster",
            name="Local Cluster",
            nodes=node_specs,
        )
        self._provider._cluster_specs = [cluster_spec]

        # async with asyncclient.AsyncRemoteManager(all_node_urls[0]) as robj:
        #     await robj.wait_for_channel_ready()
        #     remote_box = mui.RemoteBoxGrpc("localhost", ports[0], 
        #         UniqueTreeId.from_parts([group_id, WorkerUIType.TERMINAL.value]).uid_encoded)
        #     await self._root_box.set_new_layout([
        #         remote_box.prop(flex=1)
        #     ])
