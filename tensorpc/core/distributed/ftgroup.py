import asyncio
from pathlib import Path
import traceback
from typing import Any, Callable, Coroutine, Optional
import grpc
from typing_extensions import Annotated
from tensorpc.core import prim
from tensorpc.core.asyncclient import AsyncRemoteManager, shutdown_server_async, simple_remote_call_async
from tensorpc.utils.wait_tools import get_primary_ip 
from tensorpc.utils import get_service_key_by_type, rich_logging
from tensorpc.core.datamodel.draft import DraftFieldMeta
from tensorpc.core.moduleid import import_dynamic_func
from tensorpc.core.event_emitter.single import SingleAsyncEventEmitter
import tensorpc.core.dataclass_dispatch as dataclasses
import enum 
import json 
import uuid
import humanize
from tensorpc.core.asynctools import cancel_task

LOGGER = rich_logging.get_logger("tensorpc.distributed")

class FTStatus(enum.IntEnum):
    OK = 0
    MASTER_DISCONNECTED = 1
    WORKER_DISCONNECTED = 2
    UNKNOWN = 3

@dataclasses.dataclass
class FTStateBase:
    rank: int
    ip: str
    port: int
    is_master: bool 
    cur_cmd: Annotated[Optional[str], DraftFieldMeta(is_external=True)] = None
    status: FTStatus = FTStatus.OK
    # backend only states don't need to send to frontend
    uuid: Annotated[str, DraftFieldMeta(is_external=True)] = ""
    master_uuid: Annotated[str, DraftFieldMeta(is_external=True)] = ""
    master_ip: Annotated[str, DraftFieldMeta(is_external=True)] = ""

@dataclasses.dataclass
class FTGroupConfig:
    sync_folder: Optional[str] = None
    master_discovery_fn: Optional[str] = None

    disconnect_rpc_check_timeout: int = 10
    heartbeat_interval: int = 5
    # when some worker or master disconnected, we assume
    # your cluster manager will restart it. so we 
    # wait for 5 min to check if the worker is really.
    disconnect_total_retry: int = 120
    disconnect_rpc_check_timeout: int = 2

_SYNC_FILE_NAME = "ftgroup-rank-{}.json"

class FaultToleranceRPCGroup:
    def __init__(self, global_shutdown_event: asyncio.Event, 
            rank: int, world_size: int, url: str, port: int, 
            set_worker_state_serv_key: str,
            dist_url_with_ports: Optional[list[str]] = None, 
            cfg: Optional[FTGroupConfig] = None):
        self._rank = rank
        self._world_size = world_size
        self._master_rank = 0
        self._dist_url_with_ports = dist_url_with_ports
        self._url = url
        self._port = port
        if cfg is None:
            cfg = FTGroupConfig()
        if cfg.sync_folder is None:
            assert dist_url_with_ports is not None 
            assert len(dist_url_with_ports) == world_size, (f"when sync_folder is None, dist_url_with_ports length "
                "{len(dist_url_with_ports)} must be equal to world_size {world_size}.")
        self._is_master = (rank == self._master_rank)
        self._client_robjs: dict[int, AsyncRemoteManager] = {}
        self._master_robj: Optional[AsyncRemoteManager] = None
        state = FTStateBase(
            rank=rank,
            uuid=uuid.uuid4().hex,
            ip=url,
            port=port,
            is_master=self._is_master,
            master_uuid="",
            master_ip=url,
        )
        self._global_shutdown_event = global_shutdown_event
        self.state = state
        self.client_states: list[FTStateBase] = []
        for j in range(self._world_size):
            if j == self._master_rank:
                self.client_states.append(state)
            else:
                init_client_state = FTStateBase(
                    rank=j,
                    uuid="",
                    ip="",
                    port=0,
                    is_master=False,
                    status=FTStatus.WORKER_DISCONNECTED,
                )
                self.client_states.append(init_client_state)
        if self._is_master:
            state.master_uuid = state.uuid
        self._cfg = cfg
        self._set_worker_state_serv_key = set_worker_state_serv_key
        if cfg.sync_folder is not None:
            assert Path(cfg.sync_folder).exists(), f"sync_folder {cfg.sync_folder} not exists."
        self._sync_folder = cfg.sync_folder
        self._state_lock = asyncio.Lock()
        self._fast_wakeup_event = asyncio.Event()
        self._master_discovery_fn: Optional[Callable[[], Optional[tuple[str, str]]]] = None
        if cfg.master_discovery_fn is not None:
            self._master_discovery_fn = import_dynamic_func(
                cfg.master_discovery_fn, is_func_id=True)
        self._has_discovery = cfg.master_discovery_fn is not None or cfg.sync_folder is not None
        self._client_ranks = set(range(world_size))
        self._client_ranks.remove(self._master_rank)
        self.event_heartbeat_ok_to_disconnect = SingleAsyncEventEmitter[()]()
        self.event_heartbeat_disconnect_to_ok = SingleAsyncEventEmitter[()]()
        self.event_heartbeat_step_end = SingleAsyncEventEmitter[()]()
        self.event_heartbeat_start = SingleAsyncEventEmitter[()]()
        self.event_restart = SingleAsyncEventEmitter[()]()
        self.event_set_state = SingleAsyncEventEmitter[FTStateBase]()
        self.event_exit = SingleAsyncEventEmitter[()]()
        self.event_master_update_client_state = SingleAsyncEventEmitter[FTStateBase]()

    @property 
    def rank(self) -> int:
        return self._rank

    @property 
    def is_master(self) -> bool:
        return self._is_master

    @property 
    def is_sync_mode(self):
        return self._dist_url_with_ports is not None 

    def init(self):
        if not self.is_sync_mode:
            if self._sync_folder is not None:
                file_name = Path(self._sync_folder) / _SYNC_FILE_NAME.format(self._rank)
                with file_name.open("w") as f:
                    json.dump({
                        "ip": self.state.ip,
                        "port": self._port,
                        "uuid": self.state.uuid,
                    }, f, indent=4)
            elif not self._has_discovery:
                assert self._dist_url_with_ports is not None
                # always use provided urls
                if self._is_master:
                    for i, url in enumerate(self._dist_url_with_ports):
                        if i == self._master_rank:
                            continue
                        self._client_robjs[i] = AsyncRemoteManager(url)
        self._heartbeat_loop_task = asyncio.create_task(self._heartbeat_loop(), name="ftgroup_heartbeat")

    def _num_client_robj_is_valid(self):
        if self._is_master:
            return len(self._client_robjs) == self._world_size - 1
        else:
            return self._master_robj is not None

    async def close(self):
        if self.state.is_master:
            for rank, robj in self._client_robjs.items():
                try:
                    await robj.close()
                except:
                    traceback.print_exc()
        else:
            if self._master_robj is not None:
                try:
                    await self._master_robj.close()
                except:
                    traceback.print_exc()

    async def master_call_all_client(self, key: str, exclude_ranks: set[int], *args, rpc_timeout: Optional[int] = None, rpc_is_health_check: bool = False, **kwargs):
        res_dict: dict[int, Any] = {}
        exc_dict: dict[int, BaseException] = {}

        assert self._is_master, "only master can call all client"
        if not rpc_is_health_check and self._dist_url_with_ports is not None:
            # for sync mode (sync distributed task such as pytorch), 
            # if some rank fail with disconnect, 
            # we must retry immediately instead of use heartbeat loop.
            assert self._num_client_robj_is_valid(), "all worker must be connected in sync mode."
            assert not exclude_ranks, "exclude_ranks not supported in sync mode."
            tasks: list[asyncio.Task] = []
            ranks: list[int] = []
            for rank, robj in self._client_robjs.items():
                tasks.append(asyncio.create_task(robj.remote_call(key, *args, rpc_timeout=rpc_timeout, **kwargs)))
                ranks.append(rank)
            ranks = ranks.copy()
            retry_cnt = max(1, self._cfg.disconnect_total_retry)
            failed = False
            while ranks:
                res = await asyncio.gather(*tasks, return_exceptions=True)
                new_tasks = []
                new_ranks = []
                exc_dict.clear()
                for rank, r in zip(ranks, res):
                    if isinstance(r, BaseException):
                        LOGGER.exception(f"master call worker {rank} failed with exception", exc_info=r)
                        if isinstance(r, grpc.aio.AioRpcError):
                            # exit without retry.
                            retry_cnt = 0
                        else:
                            self.client_states[rank].status = FTStatus.WORKER_DISCONNECTED
                            poped_robj = self._client_robjs[rank]
                            LOGGER.warning(f"worker {rank}({poped_robj.url}) disconnected")
                            new_tasks.append(asyncio.create_task(poped_robj.remote_call(key, *args, rpc_timeout=rpc_timeout, **kwargs)))
                            new_ranks.append(rank)
                        exc_dict[rank] = r
                    else:
                        self.client_states[rank].status = FTStatus.OK
                        res_dict[rank] = r
                tasks = new_tasks
                ranks = new_ranks
                if retry_cnt <= 0 and exc_dict:
                    LOGGER.warning(f"master call all client retry timeout, failed ranks: {ranks}. all rank will try to exit.")
                    for task in tasks:
                        await cancel_task(task)
                    # exit servers
                    prim.get_async_shutdown_event().set()
                    for rank, url in enumerate(self._dist_url_with_ports):
                        if rank == self._master_rank:
                            continue
                        try:
                            await shutdown_server_async(url)
                        except grpc.aio.AioRpcError:
                            traceback.print_exc()
                            continue
                    failed = True
                    break
                retry_cnt -= 1
            return res_dict, exc_dict, failed
        
        is_all_worker_conn = self._num_client_robj_is_valid()
        tasks = []
        ranks: list[int] = []
        for rank, robj in self._client_robjs.items():
            if rank in exclude_ranks:
                continue
            if rpc_is_health_check:
                tasks.append(asyncio.create_task(robj.health_check(timeout=rpc_timeout)))
            else:
                tasks.append(asyncio.create_task(robj.remote_call(key, *args, rpc_timeout=rpc_timeout, **kwargs)))
            ranks.append(rank)
        res = await asyncio.gather(*tasks, return_exceptions=True)
        for rank, r in zip(ranks, res):
            if isinstance(r, BaseException):
                self.client_states[rank].status = FTStatus.WORKER_DISCONNECTED
                poped_robj = self._client_robjs.pop(rank)
                try:
                    await poped_robj.close()
                except:
                    traceback.print_exc()
                LOGGER.warning(f"worker {rank}({poped_robj.url}) disconnected")
                exc_dict[rank] = r
            else:
                self.client_states[rank].status = FTStatus.OK
                res_dict[rank] = r
        if not is_all_worker_conn or exc_dict:
            self.state.status = FTStatus.WORKER_DISCONNECTED
        return res_dict, exc_dict, is_all_worker_conn and not exc_dict

    async def _query_master_robj(self):
        port = self._port
        robj: Optional[AsyncRemoteManager] = None
        if self._master_discovery_fn is not None:
            master_ip_port = self._master_discovery_fn()
            if master_ip_port is not None:
                master_ip, port = master_ip_port
                robj = AsyncRemoteManager(f"{master_ip}:{port}")
                try:
                    await robj.health_check(timeout=self._cfg.disconnect_rpc_check_timeout, wait_for_ready=True)
                except grpc.aio.AioRpcError as e:
                    if e.code() == grpc.StatusCode.UNAVAILABLE:
                        robj = None
                    elif e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                        robj = None
                    else:
                        traceback.print_exc()
                        robj = None
                return robj
        elif self._cfg.sync_folder is not None:
            folder_p = Path(self._cfg.sync_folder)
            master_files = list(folder_p.glob(_SYNC_FILE_NAME.format(self._master_rank)))
            if len(master_files) > 0:
                with master_files[0].open("r") as f:
                    master_info = json.load(f)
                    master_ip = master_info["ip"]
                    uuid = master_info["uuid"]
                    if "port" in master_info:
                        port = master_info["port"]
                    robj = AsyncRemoteManager(f"{master_ip}:{port}")
            else:
                robj = None
                LOGGER.warning("Can't find Master info file, please check the workdir.")
        else:
            assert self._dist_url_with_ports is not None
            robj = AsyncRemoteManager(f"{self._dist_url_with_ports[self._master_rank]}")
        if robj is not None:
            robj_url = robj.url
            try:
                await robj.health_check(timeout=self._cfg.disconnect_rpc_check_timeout, wait_for_ready=True)
            except grpc.aio.AioRpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    robj = None
                elif e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    robj = None
                else:
                    traceback.print_exc()
                    robj = None
                LOGGER.warning(f"Master {robj_url} disconnected with grpc code {e.code()}")
        return robj

    async def _client_set_worker_state(self, handle_restart: bool = True):
        assert self._master_robj is not None 
        try:
            master_ftstate = await self._master_robj.remote_call(self._set_worker_state_serv_key, self.state)
            if handle_restart:
                if self.state.master_uuid != "" and self.state.master_uuid != master_ftstate.master_uuid:
                    # master restart. if cmd is running, shutdown it.
                    await self.event_restart.emit_async()
            self.state.master_uuid = master_ftstate.master_uuid
            self.state.master_ip = master_ftstate.master_ip
        except:
            traceback.print_exc()
            self._master_robj = None 
            LOGGER.warning(f"Master disconnected")
            self.state.status = FTStatus.MASTER_DISCONNECTED
            await self.event_heartbeat_ok_to_disconnect.emit_async()


    async def _master_state_check_is_ok(self):
        if len(self._client_robjs) == self._world_size - 1:
            self.client_states[self._master_rank].status = FTStatus.OK
            await self.event_heartbeat_disconnect_to_ok.emit_async()
            self._disconnect_retry_count = 0

    async def set_worker_state(self, state: FTStateBase) -> FTStateBase:
        async with self._state_lock:
            assert state.rank != self._master_rank, "master rank should not be set"
            assert self.state.is_master, "only master can set worker state"
            prev_client_state = self.client_states[state.rank]
            client_is_restart = (prev_client_state.uuid != state.uuid and prev_client_state.uuid != "")
            master_is_restart = (self.state.uuid != state.master_uuid and state.master_uuid != "")
            if client_is_restart:
                LOGGER.error(f"client uuid changed ({prev_client_state.uuid} -> {state.uuid}), may be restarted.")
            if master_is_restart:
                LOGGER.error(f"master uuid changed ({state.master_uuid} -> {self.state.uuid}), may be restarted.")
            if client_is_restart or master_is_restart:
                await self.event_restart.emit_async()
            await self.event_master_update_client_state.emit_async(state)
            self.client_states[state.rank] = state
            if state.rank in self._client_robjs:
                prev_robj = self._client_robjs[state.rank]
                if client_is_restart:
                    try:
                        await prev_robj.close()
                    except:
                        traceback.print_exc()
                    self._client_robjs.pop(state.rank)
                    LOGGER.warning(f"Worker {state.rank}({state.ip}:{state.port}) reconnected.")
                    robj = AsyncRemoteManager(f"{state.ip}:{state.port}")
                    self._client_robjs[state.rank] = robj
            else:
                LOGGER.warning(f"Worker {state.rank}({state.ip}:{state.port}) connected.")
                robj = AsyncRemoteManager(f"{state.ip}:{state.port}")
                self._client_robjs[state.rank] = robj
            await self._master_state_check_is_ok()
            return self.state

    async def _master_check_workers(self):
        if self.state.status == FTStatus.OK:
            res, exc_res, all_client_ok = await self.master_call_all_client("", set(), rpc_timeout=self._cfg.disconnect_rpc_check_timeout, rpc_is_health_check=True)
            if not all_client_ok:
                await self.event_heartbeat_ok_to_disconnect.emit_async()
        else:
            disconnected_ranks = self._client_ranks - set(self._client_robjs.keys())
            if not self._has_discovery:
                assert self._dist_url_with_ports is not None 

                for rank in disconnected_ranks:
                    robj_url = self._dist_url_with_ports[rank]
                    new_robj = AsyncRemoteManager(robj_url)
                    try:
                        await new_robj.health_check(timeout=self._cfg.disconnect_rpc_check_timeout, wait_for_ready=True)
                    except grpc.aio.AioRpcError as e:
                        if e.code() == grpc.StatusCode.UNAVAILABLE:
                            new_robj = None
                        elif e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                            new_robj = None
                        else:
                            traceback.print_exc()
                            new_robj = None
                        LOGGER.warning(f"Worker {robj_url} still disconnected with grpc code {e.code()}")
                    if new_robj is not None:
                        self._client_robjs[rank] = new_robj
                        self.client_states[rank].status = FTStatus.OK
                        LOGGER.warning(f"Worker {rank}({robj_url}) reconnected.")
                disconnected_ranks = self._client_ranks - set(self._client_robjs.keys())
            if len(self._client_robjs) != self._world_size - 1:
                # get current disconnected worker rank
                self._disconnect_retry_count += 1
                LOGGER.warning("master wait for all worker retry: %d/%d, disconnected ranks: %s", 
                    self._disconnect_retry_count, self._cfg.disconnect_total_retry, str(disconnected_ranks))
                if self._disconnect_retry_count > self._cfg.disconnect_total_retry:
                    LOGGER.warning("master wait for all worker timeout, exit.")
                    return True
            else:
                self._disconnect_retry_count = 0
                self.state.status = FTStatus.OK
                await self.event_heartbeat_disconnect_to_ok.emit_async()
        return False

    async def _client_check_master(self):
        if self.state.status == FTStatus.OK:
            if self._master_robj is None:
                robj = await self._query_master_robj()
                if robj is None:
                    LOGGER.warning(f"Master disconnected")
                    self.state.status = FTStatus.MASTER_DISCONNECTED
                    await self.event_heartbeat_ok_to_disconnect.emit_async()
                    return False 
                else:
                    LOGGER.warning(f"Master {robj.url} connected")
                    self._master_robj = robj
            else:
                robj = self._master_robj
            await self._client_set_worker_state(handle_restart=True)
        else:
            robj = await self._query_master_robj()
            if robj is None:
                self._disconnect_retry_count += 1
                LOGGER.warning("worker wait for master retry: %d/%d", self._disconnect_retry_count,
                    self._cfg.disconnect_total_retry)
                if self._disconnect_retry_count > self._cfg.disconnect_total_retry:
                    return True 
            else:
                LOGGER.warning(f"Master {robj.url} connected")

                self._master_robj = robj
                self._disconnect_retry_count = 0
                self.state.status = FTStatus.OK
                await self.event_heartbeat_disconnect_to_ok.emit_async()
        return False 

    async def check_connection(self):
        async with self._state_lock:
            if self._is_master:
                return await self._master_check_workers()
            else:
                return await self._client_check_master()


    async def _heartbeat_loop(self):
        shutdown_ev = self._global_shutdown_event
        shutdown_ev_task = asyncio.create_task(shutdown_ev.wait(), name="heartbeat_shutdown")
        await self.event_heartbeat_start.emit_async()
        try:
            if self.state.is_master:
                fast_wakeup_task = asyncio.create_task(
                    self._fast_wakeup_event.wait(), name="heartbeat_fast_wakeup")
                while True:
                    sleep_task = asyncio.create_task(
                        asyncio.sleep(self._cfg.heartbeat_interval), name="heartbeat_sleep")
                    done, _ = await asyncio.wait(
                        [shutdown_ev_task, sleep_task, fast_wakeup_task],
                        return_when=asyncio.FIRST_COMPLETED)
                    if shutdown_ev_task in done:
                        await cancel_task(sleep_task)
                        await cancel_task(fast_wakeup_task)
                        break 
                    if fast_wakeup_task in done:
                        self._fast_wakeup_event.clear()
                        fast_wakeup_task = asyncio.create_task(
                            self._fast_wakeup_event.wait(), name="heartbeat_fast_wakeup")
                    # LOGGER.warning("Master Heartbeat|status: %s.", self.state.status.name)
                    async with self._state_lock:
                        should_exit = await self._master_check_workers()
                        if should_exit:
                            LOGGER.warning("master wait for all worker timeout, exit.")
                            shutdown_ev.set()
                            break
                        await self.event_heartbeat_step_end.emit_async()
            else:
                while True:
                    sleep_task = asyncio.create_task(
                        asyncio.sleep(self._cfg.heartbeat_interval), name="heartbeat_sleep")
                    done, _ = await asyncio.wait(
                        [shutdown_ev_task, sleep_task],
                        return_when=asyncio.FIRST_COMPLETED)
                    if shutdown_ev_task in done:
                        await cancel_task(sleep_task)
                        break 
                    LOGGER.info("Worker Heartbeat|status: %s.", self.state.status.name)
                    async with self._state_lock:
                        should_exit = await self._client_check_master()
                        if should_exit:
                            LOGGER.warning("worker wait for master timeout, exit.")
                            await self.event_exit.emit_async()
                            break
                        await self.event_heartbeat_step_end.emit_async()

        except:
            LOGGER.warning("heartbeat loop exception", exc_info=True)
            raise 
        finally:
            LOGGER.warning("heartbeat loop exit")
