# agent manage local SSH connections and compute resources.
import asyncio
from functools import partial
from typing import TYPE_CHECKING, Any, Coroutine, Generic, Optional, TypeVar, Union, cast

from tensorpc.apps.cm.worker import A2AStateMachine, NodeFlags, GroupNodeSpec, SSHA2AWorker, SSHWorkerConfig
from tensorpc.apps.dbg.components.dbgpanel import list_all_dbg_server_in_machine
from tensorpc.core import dataclass_dispatch as dataclasses, marker, prim
from tensorpc.core.distributed.comm.grpcimpl import AsyncGRPCComm
from tensorpc.core.distributed.raft import (
    AppendEntriesRequest,
    AppendEntriesResponse,
    InstallSnapshotRequest,
    InstallSnapshotResponse,
    LeaderQueryResultBase,
    PeerInfo,
    RaftRole,
    RequestVoteRequest,
    RequestVoteResponse,
)
from tensorpc.core.serviceunit import ServiceEventType
from tensorpc.core.tree_id import UniqueTreeId
from tensorpc.dock.serv_names import serv_names as app_serv_names
from tensorpc.utils.pyspyutil import PyspyTraceMode
from tensorpc.utils.wait_tools import get_primary_ip
from .coretypes import CMNodeManagerArgs, GroupCoarseStatus, GroupSSHStatus, LeaderUIStateResult, ResourceInfo, UserCmd, WorkerInfo, CM_LOGGER, WorkerSSHStatus
from .serv_names import master_serv_names
from tensorpc.utils.pyspyutil import fetch_pyspy_info
from tensorpc.utils.json_utils import json_load_from_bytes

@dataclasses.dataclass
class GroupSpec:
    id: str
    leader_id: Optional[str]
    raft_node_specs: list[GroupNodeSpec]
    compute_node_specs: list[GroupNodeSpec]
    # for ClusterManagePanel only, store frontend info index.
    index: int = -1
    status: int = int(GroupSSHStatus.UNKNOWN)
    worker_last_activity: str = ""

@dataclasses.dataclass
class ScanGroupResult:
    group_specs: dict[str, GroupSpec]
    user_url_to_sever_ids: dict[str, str]

class GroupInstance:

    def __init__(self, group_id: str, worker: SSHA2AWorker):
        self.group_id = group_id
        self.worker = worker

@dataclasses.dataclass
class NodeMasterConfig:
    resource: ResourceInfo
    worker_cfg: SSHWorkerConfig

@dataclasses.dataclass
class NodeMasterUIState:
    groups: list[Any]
    pass 

class NodeMaster:

    def __init__(self, uid: str, config_dict: dict):
        self._uid = uid
        self._cfg = CMNodeManagerArgs(
            **config_dict
        )
        self._groups: dict[str, GroupInstance] = {}
        self._lock = asyncio.Lock()

    @marker.mark_server_event(event_type=ServiceEventType.Init)
    async def _on_init(self):
        self_ip = get_primary_ip()
        self_port = prim.get_server_grpc_port()
        # TODO better unique id generation, e.g. mac address?
        self._uid = f"{self._uid}@{self_ip}:{self_port}"
        CM_LOGGER.warning(f"NodeMaster {self._uid}({self_ip}:{self_port}) initialized.")

    async def internal_create_group(self, group_id: str, rank: int, world_size: int, peer_info: PeerInfo, raft_node_infos: list[PeerInfo], worker_cfg: Optional[SSHWorkerConfig] = None,
            resource_info: Optional[ResourceInfo] = None,):
        # TODO if group exists, check its newly created.
        assert group_id not in self._groups, f"group {group_id} already exists"
        comm = AsyncGRPCComm(self._cfg.master_comm_cfg)
        if worker_cfg is None:
            worker_cfg = self._cfg.get_worker_cfg()
        assert peer_info.uid == self._uid, f"peer_info.uid {peer_info.uid} must be same with self._uid {self._uid}"
        worker = SSHA2AWorker(
            group_id,
            rank=rank,
            world_size=world_size,
            peer_info=peer_info,
            init_raft_infos=raft_node_infos,
            comm=comm,
            cfg=worker_cfg,
            resource_info=resource_info,
            propose_fn=partial(self.propose, group_id),
            debug_panel_fn=partial(self.master_debug_panel_action, group_id),
            fetch_pyspy_info_fn=partial(self.master_fetch_pyspy_info, group_id),
        )
        async with self._lock:
            await worker.start()
            set_layout_service = prim.get_service(
                app_serv_names.REMOTE_COMP_SET_LAYOUT_OBJECT)
            try:
                for comp_name, comp in worker.get_components_dict().items():
                    comp_uid = UniqueTreeId.from_parts([group_id, comp_name]).uid_encoded
                    await set_layout_service(comp_uid, comp)
            except:
                CM_LOGGER.exception(f"Failed to set layout object for group {group_id}")
                raise

            group = GroupInstance(
                group_id=group_id,
                worker=worker,
            )
            self._groups[group_id] = group

    def _next_scan_partition(self, self_worker_info: Optional[WorkerInfo], all_worker_infos: list[WorkerInfo], num_partition: int) -> tuple[list[WorkerInfo], list[list[WorkerInfo]]]:
        if self_worker_info is not None:
            all_worker_infos = [w for w in all_worker_infos if w.peer_info.uid != self_worker_info.peer_info.uid]
        if not all_worker_infos:
            return [], []
        # remove self from all_node_urls
        # pick num_partition childs, call tree_scan on each child, and aggregate results.
        next_scan_masters = all_worker_infos[:num_partition]
        next_scan_partitions: list[list[WorkerInfo]] = [[] for _ in range(num_partition)]
        for i, url in enumerate(all_worker_infos[num_partition:]):
            next_scan_partitions[i % num_partition].append(url)
        return next_scan_masters, next_scan_partitions

    async def internal_tree_create_group(
        self, group_id: str, world_size: int, self_worker_info: Optional[WorkerInfo], 
        compute_node_infos: list[WorkerInfo], raft_node_infos: list[PeerInfo],
        num_partition: int, worker_cfg: Optional[SSHWorkerConfig] = None, 
        scan_comm: Optional[AsyncGRPCComm] = None,
    ):
        is_ext_comm = scan_comm is not None
        if scan_comm is None:
            scan_comm = AsyncGRPCComm(self._cfg.master_comm_cfg)
        try:
            if self_worker_info is not None:
                await self.internal_create_group(
                    group_id, self_worker_info.rank, world_size, 
                    self_worker_info.peer_info, raft_node_infos, worker_cfg, 
                    self_worker_info.resource,
                )
            next_scan_masters, next_scan_partitions = self._next_scan_partition(self_worker_info, compute_node_infos, num_partition)
            if not next_scan_masters:
                return
            remote_call_coros: list[Coroutine[None, None, Any]] = []
            for scan_master, scan_partition in zip(next_scan_masters, next_scan_partitions):
                kwargs = {
                    "group_id": group_id,
                    "world_size": world_size,
                    "self_worker_info": scan_master,
                    "compute_node_infos": scan_partition,
                    "num_partition": num_partition,
                    "worker_cfg": worker_cfg,
                    "raft_node_infos": raft_node_infos,
                }
                remote_call_coros.append(scan_comm.remote_call(scan_master.peer_info.url, 
                    master_serv_names.INTERNAL_GROUP_TREE_CREATE_GROUP, **kwargs))
            await asyncio.gather(*remote_call_coros)
        finally:
            if not is_ext_comm:
                await scan_comm.close()

    async def master_create_group(
        self, group_id: str, self_uid: str, compute_node_infos: list[WorkerInfo], 
        raft_node_infos: list[PeerInfo], num_partition: int, worker_cfg: Optional[SSHWorkerConfig] = None, 
    ):
        # WARNING: self_uid and uids in infos are **client uid** from provider, which may different from self._uid,
        # we only trust urls in infos.
        # so we do a map after scan group.
        world_size = len(compute_node_infos)
        # TODO validate uids is unique
        # 1. run a scan to make sure all nodes are alive and get their peer info and resource info, and then we can decide how to partition the group.
        cnode_urls = [node_info.peer_info.url for node_info in compute_node_infos if node_info.peer_info.uid != self_uid]
        raft_node_urls = [node_info.url for node_info in raft_node_infos if node_info.uid != self_uid]
        # extract possible self_url from all infos
        self_url: Optional[str] = None
        for node_info in compute_node_infos:
            if node_info.peer_info.uid == self_uid:
                self_url = node_info.peer_info.url
                break
        if self_url is None:
            for node_info in raft_node_infos:
                if node_info.uid == self_uid:
                    self_url = node_info.url
                    break
        all_node_urls = list(set(cnode_urls) | set(raft_node_urls))
        scan_comm = AsyncGRPCComm(self._cfg.master_comm_cfg)
        try:
            scan_res = await self.tree_scan_groups(
                self_url, all_node_urls, num_partition, scan_comm
            )
            del all_node_urls, cnode_urls, raft_node_urls
            assert group_id not in scan_res.group_specs, f"group {group_id} already exists"
            user_url_to_server_ids = scan_res.user_url_to_sever_ids
            # map self_uid, uid in compute_node_infos and raft_node_infos to server uids.
            if self_url is not None:
                self_uid = user_url_to_server_ids[self_url]
            compute_node_infos = [x.replace_uid(user_url_to_server_ids[x.peer_info.url]) for x in compute_node_infos]
            raft_node_infos = [x.replace_uid(user_url_to_server_ids[x.url]) for x in raft_node_infos]
            # TODO validate available resources here.
            # 2. create raft groups
            raft_node_uids = set(node_info.uid for node_info in raft_node_infos)
            cnode_id_to_info = {node_info.peer_info.uid: node_info for node_info in compute_node_infos}
            for raft_info in raft_node_infos:
                rank = -1
                rc: Optional[ResourceInfo] = None
                if raft_info.uid in cnode_id_to_info:
                    rank = cnode_id_to_info[raft_info.uid].rank
                    rc = cnode_id_to_info[raft_info.uid].resource
                kwargs = {
                    "group_id": group_id,
                    "rank": rank,
                    "peer_info": raft_info,
                    "raft_node_infos": raft_node_infos,
                    "worker_cfg": worker_cfg,
                    "world_size": world_size,
                    "resource_info": rc,
                }

                if raft_info.uid == self_uid:
                    await self.internal_create_group(**kwargs)
                else: 
                    await scan_comm.remote_call(raft_info.url, master_serv_names.INTERNAL_GROUP_CREATE_GROUP, **kwargs)

            # 3. create remain groups
            self_info: Optional[WorkerInfo] = None
            remain_worker_infos: list[WorkerInfo] = []
            for node_info in compute_node_infos:
                if node_info.peer_info.uid not in raft_node_uids:
                    if node_info.peer_info.uid != self_uid:
                        remain_worker_infos.append(node_info)
                    else:
                        self_info = node_info
            if not remain_worker_infos:
                return 
            await self.internal_tree_create_group(
                group_id, world_size, self_info, remain_worker_infos, raft_node_infos, num_partition, worker_cfg, 
                scan_comm
            )
        finally:
            await scan_comm.close()
    
    async def _remove_group(self, group_id: str, remove_stopped_group: bool = True):
        async with self._lock:
            if group_id in self._groups:
                if remove_stopped_group:
                    group = self._groups.pop(group_id)
                else:
                    group = self._groups[group_id]
                if group.worker.is_started():
                    await group.worker.stop()
                    remove_layout_service = prim.get_service(
                        app_serv_names.REMOTE_COMP_REMOVE_LAYOUT_OBJECT)
                    try:
                        for comp_name, comp in group.worker.get_components_dict().items():
                            comp_uid = UniqueTreeId.from_parts([group_id, comp_name]).uid_encoded
                            await remove_layout_service(comp_uid)
                    except:
                        CM_LOGGER.exception(f"Failed to remove layout object for group {group_id}")

    async def query_group_coarse_status(self, group_id: str) -> GroupCoarseStatus:
        group = self._groups[group_id]
        if group.worker._raft_node is None:
            return GroupCoarseStatus(
                id=group_id,
                success=False,
                is_raft_node=False,
            )
        if group.worker._raft_node.role != RaftRole.LEADER:
            return GroupCoarseStatus(
                id=group_id,
                success=False,
                is_raft_node=True,
                leader_info=group.worker._raft_node.get_leader_peer_info(),
            )
        assert group.worker._raft_node.state_machine is not None
        state_machine = group.worker._raft_node.state_machine
        assert isinstance(state_machine, A2AStateMachine)
        last_cmd = state_machine.raft_state["last_cmd"]
        group_ssh_status = group.worker.ssh_dm.model.group_ssh_status
        return GroupCoarseStatus(
            id=group_id,
            success=True,
            is_raft_node=True,
            group_ssh_status=group_ssh_status,
            last_cmd=last_cmd,
            worker_last_activity=group.worker.ssh_dm.model.worker_last_activity,
        )

    async def tree_scan_groups(
        self, self_url: Optional[str], all_node_urls: list[str], num_partition: int,
        scan_comm: Optional[AsyncGRPCComm] = None,
    ) -> ScanGroupResult:
        # TODO support retry in comm
        # we don't want to keep grpcs in whole cluster, so we create temp comm here.
        is_ext_comm = scan_comm is not None
        if scan_comm is None:
            scan_comm = AsyncGRPCComm(self._cfg.master_comm_cfg)
        res: ScanGroupResult = ScanGroupResult(group_specs={}, user_url_to_sever_ids={})
        # add local results
        async with self._lock:
            try:
                if self_url is not None:
                    res.user_url_to_sever_ids[self_url] = self._uid
                for group_id, group_inst in self._groups.items():
                    cur_spec = group_inst.worker.get_node_spec()
                    res.group_specs[group_id] = GroupSpec(group_id, None, [], [])
                    if cur_spec.flags & NodeFlags.IS_RAFT_NODE:
                        res.group_specs[group_id].raft_node_specs.append(cur_spec)
                    if cur_spec.flags & NodeFlags.IS_COMPUTE_NODE:
                        res.group_specs[group_id].compute_node_specs.append(cur_spec)
                    if cur_spec.flags & NodeFlags.IS_RAFT_LEADER:
                        res.group_specs[group_id].leader_id = cur_spec.peer_info.uid
                        res.group_specs[group_id].status = group_inst.worker.ssh_dm.model.group_ssh_status
                        res.group_specs[group_id].worker_last_activity = group_inst.worker.ssh_dm.model.worker_last_activity    
                if not all_node_urls:
                    return res
                all_node_urls = [url for url in all_node_urls if url != self_url]
                # remove self from all_node_urls
                # pick num_partition childs, call tree_scan on each child, and aggregate results.
                next_scan_masters = all_node_urls[:num_partition]
                next_scan_partitions = [[] for _ in range(num_partition)]
                for i, url in enumerate(all_node_urls[num_partition:]):
                    next_scan_partitions[i % num_partition].append(url)
                remote_call_coros: list[Coroutine[None, None, ScanGroupResult]] = []
                for scan_master, scan_partition in zip(next_scan_masters, next_scan_partitions):
                    remote_call_coros.append(scan_comm.remote_call(scan_master, 
                        master_serv_names.GROUP_TREE_SCAN_GROUPS, scan_master, scan_partition,
                        num_partition))
                remote_call_results = await asyncio.gather(*remote_call_coros)
                for res_next in remote_call_results:
                    for group_id, group_spec in res_next.group_specs.items():
                        if group_id not in res.group_specs:
                            res.group_specs[group_id] = group_spec
                        else:
                            # merge group spec
                            master_spec = res.group_specs[group_id]
                            if master_spec.leader_id == "":
                                master_spec.leader_id = group_spec.leader_id
                            master_spec.raft_node_specs.extend(group_spec.raft_node_specs)
                            master_spec.compute_node_specs.extend(group_spec.compute_node_specs)
                    res.user_url_to_sever_ids.update(res_next.user_url_to_sever_ids)
            finally:
                if not is_ext_comm:
                    await scan_comm.close()
        return res

    async def tree_remove_group(
        self, group_id: str, self_url: Optional[str], all_node_urls: list[str], num_partition: int,
        scan_comm: Optional[AsyncGRPCComm] = None,
        remove_stopped_group: bool = True,
    ) -> None:
        # TODO support retry in comm
        # we don't want to keep grpcs in whole cluster, so we create temp comm here.
        is_ext_comm = scan_comm is not None
        if scan_comm is None:
            scan_comm = AsyncGRPCComm(self._cfg.master_comm_cfg)
        try:
            if group_id in self._groups:
                await self._remove_group(group_id, remove_stopped_group)
            if not all_node_urls:
                return
            all_node_urls = [url for url in all_node_urls if url != self_url]
            # remove self from all_node_urls
            # pick num_partition childs, call tree_scan on each child, and aggregate results.
            next_scan_masters = all_node_urls[:num_partition]
            next_scan_partitions = [[] for _ in range(num_partition)]
            for i, url in enumerate(all_node_urls[num_partition:]):
                next_scan_partitions[i % num_partition].append(url)
            remote_call_coros: list[Coroutine[None, None, dict[str, GroupSpec]]] = []
            for scan_master, scan_partition in zip(next_scan_masters, next_scan_partitions):
                remote_call_coros.append(scan_comm.remote_call(scan_master, 
                    master_serv_names.GROUP_TREE_REMOVE_GROUP,
                    group_id, 
                    scan_master, scan_partition,
                    num_partition,
                    remove_stopped_group=remove_stopped_group))
            await asyncio.gather(*remote_call_coros)
        finally:
            if not is_ext_comm:
                await scan_comm.close()
        return

    async def tree_awake_worker(
        self, group_id: str, self_worker_info: WorkerInfo, 
        worker_infos: list[WorkerInfo], num_partition: int,
        scan_comm: Optional[AsyncGRPCComm] = None,
    ) -> None:
        """tell all workers to awake and check if master changed."""
        is_ext_comm = scan_comm is not None
        if scan_comm is None:
            scan_comm = AsyncGRPCComm(self._cfg.master_comm_cfg)
        group = self._groups[group_id]
        try:
            group.worker.awake_leader_observe_loop()
            next_scan_masters, next_scan_partitions = self._next_scan_partition(self_worker_info, worker_infos, num_partition)
            if not next_scan_masters:
                return

            remote_call_coros: list[Coroutine[None, None, dict[str, GroupSpec]]] = []
            for scan_master, scan_partition in zip(next_scan_masters, next_scan_partitions):
                remote_call_coros.append(scan_comm.remote_call(scan_master.peer_info.url, 
                    master_serv_names.GROUP_TREE_AWAKE_WORKER, group_id, 
                    scan_master, scan_partition, num_partition))
            await asyncio.gather(*remote_call_coros)
        finally:
            if not is_ext_comm:
                await scan_comm.close()

    async def request_vote(
        self, group_id: str, req: RequestVoteRequest
    ) -> RequestVoteResponse:
        group = self._groups[group_id]
        assert group.worker._raft_node is not None
        return await group.worker._raft_node.handle_request_vote(req)

    async def append_entries(
        self, group_id: str, req: AppendEntriesRequest
    ) -> AppendEntriesResponse:
        group = self._groups[group_id]
        assert group.worker._raft_node is not None
        return await group.worker._raft_node.handle_append_entries(req)

    async def install_snapshot(
        self, group_id: str, req: InstallSnapshotRequest
    ) -> InstallSnapshotResponse:
        group = self._groups[group_id]
        assert group.worker._raft_node is not None
        return await group.worker._raft_node.handle_install_snapshot(req)

    async def worker_heartbeat(
        self, group_id: str, worker_peer_info: PeerInfo, rank: int, ssh_status: WorkerSSHStatus
    ):
        group = self._groups[group_id]
        return await group.worker.handle_worker_heartbeat(worker_peer_info, rank, ssh_status)

    def _get_group_compute_worker_infos(self, group_id: str) -> list[WorkerInfo]:
        group = self._groups[group_id]
        worker_states = group.worker.get_all_worker_states()
        return [w["worker_info"] for w in worker_states]

    def _get_group_raft_worker_infos(self, group_id: str) -> list[WorkerInfo]:
        group = self._groups[group_id]
        return [WorkerInfo(p, -1) for p in group.worker.get_all_raft_infos()]

    async def propose(
        self, group_id: str, cmd: UserCmd, sync_all_workers: bool = False,
        run_iff_num_worker: Optional[int] = None,
    ):
        group = self._groups[group_id]
        assert group.worker._raft_node is not None
        worker_states = group.worker.get_all_worker_states()
        connected_cnt = 0 
        for worker_state in worker_states:
            if worker_state["is_connected"]:
                connected_cnt += 1
        if run_iff_num_worker is not None and connected_cnt != run_iff_num_worker:
            raise RuntimeError(f"Current connected worker num {connected_cnt} doesn't match required num {run_iff_num_worker}, won't run cmd {cmd}")
        res = await group.worker._raft_node.propose(cmd)
        if not res.success:
            return res 
        if sync_all_workers:
            assert group.worker._raft_node.config.apply_sync, "sync_all_workers is only supported when apply_sync is True"
            all_worker_infos = [w["worker_info"] for w in worker_states]
            await self.tree_awake_worker(group_id, group.worker.get_worker_info(), all_worker_infos, 8)
        return res

    @marker.mark_server_event(event_type=ServiceEventType.Exit)
    async def _on_exit(self):
        for group in self._groups.values():
            await group.worker.stop()
    
    async def tree_fetch_pyspy_info(self, group_id: str, mode: PyspyTraceMode, 
        self_worker_info: Optional[WorkerInfo], 
        worker_infos: list[WorkerInfo], num_partition: int,
        scan_comm: Optional[AsyncGRPCComm] = None):
        is_ext_comm = scan_comm is not None
        if scan_comm is None:
            scan_comm = AsyncGRPCComm(self._cfg.master_comm_cfg)
        try:
            final_res: dict[tuple[int, str], Any] = {}
            if self_worker_info is not None:
                self_res = await self._fetch_pyspy_info(group_id, mode)
                final_res[(self_worker_info.rank, self_worker_info.peer_info.uid)] = self_res
            next_scan_masters, next_scan_partitions = self._next_scan_partition(self_worker_info, worker_infos, num_partition)
            if not next_scan_masters:
                return final_res

            remote_call_coros: list[Coroutine[None, None, dict[tuple[int, str], Any]]] = []
            for scan_master, scan_partition in zip(next_scan_masters, next_scan_partitions):
                remote_call_coros.append(scan_comm.chunked_remote_call(scan_master.peer_info.url, 
                    master_serv_names.GROUP_TREE_FETCH_PYSPY_INFO, group_id, mode,
                    scan_master, scan_partition, num_partition))
            remote_call_results = await asyncio.gather(*remote_call_coros)
            for r in remote_call_results:
                final_res.update(r)
            return final_res
        finally:
            if not is_ext_comm:
                await scan_comm.close()
        
    async def _fetch_pyspy_info(self, group_id: str, mode: PyspyTraceMode):
        if mode == PyspyTraceMode.SERVER_PROCESS or mode == PyspyTraceMode.LOCAL_AIO_TASKS:
            pid = None 
        else:
            group = self._groups[group_id]
            state = group.worker.get_terminal().get_current_state()
            assert state is not None 
            pid = state.pid
        try:
            return await fetch_pyspy_info(mode, parent_pid=pid)
        except:
            CM_LOGGER.exception("get torchrun traceback failed", exc_info=True)
            return {}

    async def master_fetch_pyspy_info(self, group_id: str, mode: PyspyTraceMode, is_compute: bool):
        group = self._groups[group_id]
        res: dict[tuple[int, int], Any] = {}
        if mode == PyspyTraceMode.PYTORCH_DISTRIBUTED:
            if is_compute:
                worker_infos = self._get_group_compute_worker_infos(group_id)
            else:
                worker_infos = self._get_group_raft_worker_infos(group_id)
            return await self.tree_fetch_pyspy_info(group_id, mode, group.worker.get_worker_info(), worker_infos, num_partition=8)
        root_pyspy_info = await self._fetch_pyspy_info(group_id, mode)
        for v in root_pyspy_info.values():
            for pid, info in v.items():
                res[(group.worker.get_worker_info().rank, pid)] = info
        return res 

    async def _debug_panel_actions(self, group_id: str, key: str, *args: Any, **kwargs: Any):
        # pass
        if group_id not in self._groups:
            return
        group = self._groups[group_id]
        master_panel = group.worker._raft_mgr_panel.debug_panel
        if master_panel is not None:
            await master_panel.run_rpc_on_current_processes(key, *args, **kwargs)

    async def master_debug_panel_action(self, group_id: str, key: str, *args: Any, **kwargs: Any):
        if group_id not in self._groups:
            return
        group = self._groups[group_id]
        self_worker_info = group.worker.get_worker_info()
        worker_infos = self._get_group_compute_worker_infos(group_id)
        await self.tree_do_debug_panel_action(group_id, key, args, kwargs, self_worker_info, worker_infos, num_partition=8)

    async def tree_do_debug_panel_action(self, group_id: str, 
            key: str, args: Any, kwargs: Any,
            self_worker_info: Optional[WorkerInfo], 
            worker_infos: list[WorkerInfo], num_partition: int,
            scan_comm: Optional[AsyncGRPCComm] = None):
        is_ext_comm = scan_comm is not None
        if scan_comm is None:
            scan_comm = AsyncGRPCComm(self._cfg.master_comm_cfg)
        try:
            if self_worker_info is not None:
                await self._debug_panel_actions(group_id, key, *args, **kwargs)
            next_scan_masters, next_scan_partitions = self._next_scan_partition(self_worker_info, worker_infos, num_partition)
            if not next_scan_masters:
                return 
            remote_call_coros: list[Coroutine[None, None, dict[tuple[int, str], Any]]] = []
            for scan_master, scan_partition in zip(next_scan_masters, next_scan_partitions):
                remote_call_coros.append(scan_comm.chunked_remote_call(scan_master.peer_info.url, 
                    master_serv_names.GROUP_TREE_DO_DEBUG_PANEL_ACTION, group_id, key, args, kwargs,
                    scan_master, scan_partition, num_partition))
            await asyncio.gather(*remote_call_coros)
            return
        finally:
            if not is_ext_comm:
                await scan_comm.close()

    async def append_perfetto_data(self, group_id: str, 
            key: str, args: Any, kwargs: Any,
            self_worker_info: Optional[WorkerInfo], 
            worker_infos: list[WorkerInfo], num_partition: int,
            scan_comm: Optional[AsyncGRPCComm] = None):
        is_ext_comm = scan_comm is not None
        if scan_comm is None:
            scan_comm = AsyncGRPCComm(self._cfg.master_comm_cfg)
        try:
            if self_worker_info is not None:
                await self._debug_panel_actions(group_id, key, *args, **kwargs)
            next_scan_masters, next_scan_partitions = self._next_scan_partition(self_worker_info, worker_infos, num_partition)
            if not next_scan_masters:
                return 
            remote_call_coros: list[Coroutine[None, None, dict[tuple[int, str], Any]]] = []
            for scan_master, scan_partition in zip(next_scan_masters, next_scan_partitions):
                remote_call_coros.append(scan_comm.chunked_remote_call(scan_master.peer_info.url, 
                    master_serv_names.GROUP_TREE_DO_DEBUG_PANEL_ACTION, group_id, key, args, kwargs,
                    scan_master, scan_partition, num_partition))
            await asyncio.gather(*remote_call_coros)
            return
        finally:
            if not is_ext_comm:
                await scan_comm.close()

    async def set_perfetto_data(self, group_id: str, data: bytes, all_timestamps: list[int], key: str):
        group = self._groups[group_id]
        assert group.worker._raft_mgr_panel.debug_panel is not None, "debug panel is not available"
        await group.worker._raft_mgr_panel.debug_panel.external_set_perfetto_data(data, all_timestamps, key)

    async def query_leader_ui_state(self, group_id: str) -> LeaderUIStateResult:
        """pth control point will access this value and 
        enter breakpoint when set.
        """
        group = self._groups[group_id]
        if group.worker.is_raft_leader():
            return LeaderUIStateResult(
                success=True,
                leader_info=None,
                is_user_control_enabled=group.worker.ssh_dm.model.is_user_control_enabled,
            )
        leader = group.worker.get_leader_info()
        fail_res = LeaderUIStateResult(
            success=False,
            leader_info=group.worker.get_leader_info(),
        )
        if leader is None:
            return fail_res
        try:
            query_res: LeaderUIStateResult = await group.worker.comm.remote_call(
                leader.url,
                master_serv_names.GROUP_QUERY_LEADER_UI_STATE,
                group_id,
            )
            return query_res
        except Exception as e:
            CM_LOGGER.exception(
                f"Failed to perform query_leader_ui_state from {leader.uid}: {e}"
            )
            return fail_res

    def list_all_debug_servers(self):
        return list_all_dbg_server_in_machine()

    async def _set_perf_data(self, group: GroupInstance, rpc_done_ev: Optional[asyncio.Event], step: int, data: Union[list[list[dict]], bytes], metadata: list[Any], scale: Optional[float] = None):
        if rpc_done_ev is not None:
            await rpc_done_ev.wait()
        if isinstance(data, bytes):
            data = json_load_from_bytes(data)
        return await group.worker._raft_mgr_panel.debug_panel.perf_monitor.append_perf_data(step, cast(Any, data), metadata, scale)

    async def set_fast_perf_data(self, group_id: str, step: int, data: Union[list[list[dict]], bytes], 
            metadata: list[Any], scale: Optional[float] = None) -> LeaderQueryResultBase:
        group = self._groups[group_id]
        if group.worker.is_raft_leader():
            asyncio.create_task(self._set_perf_data(group, prim.get_async_rpc_done_event(), step, data, metadata, scale))
            return LeaderQueryResultBase(success=True, leader_info=None)

        leader = group.worker.get_leader_info()
        fail_res = LeaderQueryResultBase(
            success=False,
            leader_info=group.worker.get_leader_info(),
        )
        if leader is None:
            return fail_res
        try:
            query_res: LeaderQueryResultBase = await group.worker.comm.remote_call(
                leader.url,
                master_serv_names.DEBUG_SET_FAST_PERF_DATA,
                group_id,
            )
            return query_res
        except Exception as e:
            CM_LOGGER.exception(
                f"Failed to perform query_leader_ui_state from {leader.uid}: {e}"
            )
            return fail_res
