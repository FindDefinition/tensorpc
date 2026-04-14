import asyncio
import copy
import enum
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, Literal, Optional

from pydantic import Field, TypeAdapter
from pydantic.dataclasses import dataclass as pydantic_dataclass
from tensorpc.core.asynctools import cancel_task
from tensorpc.core.event_emitter.aio import AsyncIOEventEmitter
from tensorpc.core.distributed.logger import LOGGER

@dataclass(frozen=True)
class PeerInfo:
    uid: str
    url: str
    info: Any = None

    def __hash__(self) -> int:
        return hash(self.url)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, PeerInfo) and self.url == other.url

    def replace_uid(self, new_uid: str):
        return replace(self, uid=new_uid)

def _create_peer_info_map(self_peer: PeerInfo, peers: list[PeerInfo]) -> dict[str, PeerInfo]:
    by_uid: dict[str, PeerInfo] = {}
    for peer in peers:
        by_uid[peer.uid] = peer
    by_uid[self_peer.uid] = self_peer
    return by_uid


@dataclass
class LogEntry:
    term: int
    command: Any


class RaftCommandType(enum.IntEnum):
    APP = 1
    JOINT_CONSENSUS = 2
    FINALIZE_CONFIG = 3

class RaftEventType(enum.IntEnum):
    COMMIT_APPLIED = 1
    ROLE_CHANGED = 2

class RaftStateChangeFlag(enum.IntFlag):
    COMMIT_APPLIED = enum.auto()
    ROLE_CHANGED = enum.auto()

@pydantic_dataclass
class AppCommand:
    type: Literal[RaftCommandType.APP] = RaftCommandType.APP
    data: Any = None


@pydantic_dataclass
class JointConsensusCommand:
    type: Literal[RaftCommandType.JOINT_CONSENSUS] = RaftCommandType.JOINT_CONSENSUS
    old_voters: list["PeerInfo"] = field(default_factory=list)
    new_voters: list["PeerInfo"] = field(default_factory=list)


@pydantic_dataclass
class FinalizeConfigCommand:
    type: Literal[RaftCommandType.FINALIZE_CONFIG] = RaftCommandType.FINALIZE_CONFIG
    new_voters: list["PeerInfo"] = field(default_factory=list)


RaftCommand = Annotated[
    AppCommand | JointConsensusCommand | FinalizeConfigCommand,
    Field(discriminator="type"),
]

RAFT_COMMAND_ADAPTER = TypeAdapter(RaftCommand)


@dataclass
class RequestVoteRequest:
    term: int
    candidate_id: str
    last_log_index: int
    last_log_term: int


@dataclass
class RequestVoteResponse:
    term: int
    vote_granted: bool


@dataclass
class AppendEntriesRequest:
    term: int
    leader_id: str
    prev_log_index: int
    prev_log_term: int
    entries: list[LogEntry]
    leader_commit: int


@dataclass
class AppendEntriesResponse:
    term: int
    success: bool
    match_index: int


@dataclass
class RaftSnapshot:
    last_included_index: int
    last_included_term: int
    state_machine: Any
    peers: list[PeerInfo]
    joint_new_peers: Optional[list[PeerInfo]] = None


@dataclass
class InstallSnapshotRequest:
    term: int
    leader_id: str
    snapshot: RaftSnapshot


InstallSnapshotResponse = AppendEntriesResponse


class RaftRole(enum.Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class NotLeaderError(RuntimeError):
    pass

@dataclass
class LeaderQueryResultBase:
    success: bool
    leader_info: Optional[PeerInfo]

@dataclass
class ProposeResult(LeaderQueryResultBase):
    index: Optional[int] = None


@dataclass
class ChangeConfigurationResult(LeaderQueryResultBase):
    joint_index: Optional[int] = None
    final_index: Optional[int] = None


class StateMachine(ABC):
    @abstractmethod
    def apply(self, command: Any, log_index: int):
        raise NotImplementedError

    @abstractmethod
    def serialize(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def deserialize(self, data: Any):
        raise NotImplementedError


class ListStateMachine(StateMachine):
    def __init__(self):
        self._items: list[Any] = []

    def apply(self, command: Any, log_index: int):
        self._items.append(command)

    def serialize(self) -> Any:
        return list(self._items)

    def deserialize(self, data: Any):
        if isinstance(data, list):
            self._items = list(data)
        else:
            self._items = []

    def __contains__(self, value: Any) -> bool:
        return value in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ListStateMachine):
            return self._items == other._items
        if isinstance(other, list):
            return self._items == other
        return False


@dataclass
class RaftConfig:
    election_timeout_min: float = 0.600
    election_timeout_max: float = 1.200
    heartbeat_interval: float = 0.300
    compact_min_entries: int = 1
    apply_sync: bool = False


class AsyncRaftComm(ABC):
    @abstractmethod
    async def request_vote(self, peer_id: str, req: RequestVoteRequest) -> RequestVoteResponse:
        raise NotImplementedError

    @abstractmethod
    async def append_entries(self, peer_id: str, req: AppendEntriesRequest) -> AppendEntriesResponse:
        raise NotImplementedError

    @abstractmethod
    async def install_snapshot(self, peer_id: str, req: InstallSnapshotRequest) -> InstallSnapshotResponse:
        raise NotImplementedError


@dataclass
class RaftNode:
    self_peer: PeerInfo
    peers: list[PeerInfo]
    comm: AsyncRaftComm
    config: RaftConfig = field(default_factory=RaftConfig)
    state_machine: StateMachine = field(default_factory=ListStateMachine)

    current_term: int = 0
    voted_for: Optional[str] = None
    log: list[LogEntry] = field(default_factory=list)

    commit_index: int = -1
    last_applied: int = -1

    role: RaftRole = RaftRole.FOLLOWER
    leader_id: Optional[str] = None
    events: AsyncIOEventEmitter[RaftEventType] = field(default_factory=AsyncIOEventEmitter)
    def __post_init__(self):
        self._state_lock = asyncio.Lock()
        self._new_commit_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._run_task: Optional[asyncio.Task] = None
        self._apply_task: Optional[asyncio.Task] = None

        self._last_contact_ts = time.monotonic()
        self._election_deadline = self._next_election_deadline()

        if not isinstance(self.self_peer, PeerInfo):
            raise TypeError("RaftNode.self_peer must be PeerInfo")
        if any(not isinstance(peer, PeerInfo) for peer in self.peers):
            raise TypeError("RaftNode.peers must be list[PeerInfo]")
        self._self_peer = self.self_peer
        self.peers = self._dedupe_peers_by_ip(self.peers)
        self._peer_info_by_uid = _create_peer_info_map(self._self_peer, self.peers)

        self._next_index: dict[str, int] = {}
        self._match_index: dict[str, int] = {}
        self._voters: set[PeerInfo] = set([self._self_peer, *self.peers])
        self._joint_new_voters: Optional[set[PeerInfo]] = None
        self._log_offset: int = 0
        self._empty_state_machine_snapshot = copy.deepcopy(self.state_machine.serialize())
        self._snapshot = RaftSnapshot(
            last_included_index=-1,
            last_included_term=0,
            state_machine=copy.deepcopy(self._empty_state_machine_snapshot),
            peers=self._peer_infos_sorted(self._voters),
            joint_new_peers=None,
        )

    def _dedupe_peers_by_ip(self, peers: list[PeerInfo]) -> list[PeerInfo]:
        by_ip: dict[str, PeerInfo] = {}
        for peer in peers:
            by_ip[peer.url] = peer
        return [by_ip[url] for url in sorted(by_ip.keys())]

    def _peer_uids(self, peers: list[PeerInfo]) -> list[str]:
        return [peer.uid for peer in peers]

    def _peer_info_from_uid(self, uid: str) -> PeerInfo:
        return self._peer_info_by_uid.get(uid, PeerInfo(uid=uid, url=uid))

    def _rebuild_peer_info_maps(self, peers: set[PeerInfo]):
        self._peer_info_by_uid = _create_peer_info_map(
            self._self_peer,
            self._peer_infos_sorted(peers),
        )

    def _peer_infos_sorted(self, peers: set[PeerInfo]) -> list[PeerInfo]:
        return sorted(peers, key=lambda peer: peer.url)

    def _majority_count(self, voters: set[PeerInfo]) -> int:
        return len(voters) // 2 + 1

    def _make_app_command(self, payload: Any) -> AppCommand:
        return AppCommand(data=payload)

    def _make_joint_consensus_command(self, old_voters: set[PeerInfo], new_voters: set[PeerInfo]) -> JointConsensusCommand:
        return JointConsensusCommand(
            old_voters=sorted(old_voters, key=lambda peer: peer.url),
            new_voters=sorted(new_voters, key=lambda peer: peer.url),
        )

    def _make_finalize_config_command(self, new_voters: set[PeerInfo]) -> FinalizeConfigCommand:
        return FinalizeConfigCommand(new_voters=sorted(new_voters, key=lambda peer: peer.url))

    def _normalize_command(self, command: Any) -> RaftCommand:
        try:
            return RAFT_COMMAND_ADAPTER.validate_python(command)
        except Exception:
            return AppCommand(data=command)

    def _has_majority(self, voters: set[PeerInfo], acknowledgers: set[PeerInfo]) -> bool:
        return len(voters & acknowledgers) >= self._majority_count(voters)

    def _local_index(self, index: int) -> int:
        return index - self._log_offset

    def _entry_at(self, index: int) -> LogEntry:
        return self.log[self._local_index(index)]

    def _term_at(self, index: int) -> Optional[int]:
        if index == self._log_offset - 1:
            return self._snapshot.last_included_term
        local_index = self._local_index(index)
        if 0 <= local_index < len(self.log):
            return self.log[local_index].term
        return None

    def _replication_peer_ids_locked(self) -> set[str]:
        peers = set(self._voters)
        if self._joint_new_voters is not None:
            peers |= self._joint_new_voters
        return {peer.uid for peer in peers if peer.uid != self._self_peer.uid}

    def _ensure_replication_trackers_locked(self):
        peer_ids = self._replication_peer_ids_locked()
        next_idx = self.last_log_index + 1
        for peer in peer_ids:
            self._next_index.setdefault(peer, next_idx)
            self._match_index.setdefault(peer, -1)
            if self._next_index[peer] < self._log_offset:
                self._next_index[peer] = self._log_offset

    def _prune_replication_trackers_locked(self):
        active_peers = self._replication_peer_ids_locked()
        for peer in list(self._next_index.keys()):
            if peer not in active_peers:
                self._next_index.pop(peer, None)
        for peer in list(self._match_index.keys()):
            if peer not in active_peers:
                self._match_index.pop(peer, None)

    def _set_active_voters_locked(self, voters: set[PeerInfo]):
        self._voters = {peer for peer in voters if peer.uid != self._self_peer.uid}
        self._voters.add(self._self_peer)
        self._rebuild_peer_info_maps(self._voters)
        self.peers = sorted(
            [peer for peer in self._voters if peer.uid != self._self_peer.uid],
            key=lambda peer: peer.url,
        )
        self._prune_replication_trackers_locked()

    def _drop_snapshot_locked(self):
        self._log_offset = 0
        self._snapshot = RaftSnapshot(
            last_included_index=-1,
            last_included_term=0,
            state_machine=copy.deepcopy(self._empty_state_machine_snapshot),
            peers=self._peer_infos_sorted(self._voters),
            joint_new_peers=None,
        )
        self._joint_new_voters = None
        self.log = []
        self.state_machine.deserialize(copy.deepcopy(self._empty_state_machine_snapshot))
        self.commit_index = -1
        self.last_applied = -1
        self._new_commit_event.clear()

    def _build_snapshot_locked(self) -> RaftSnapshot:
        return copy.deepcopy(self._snapshot)

    def _config_state_at_index(self, index: int) -> tuple[set[PeerInfo], Optional[set[PeerInfo]]]:
        voters = set(self._voters)
        joint_new: Optional[set[PeerInfo]] = (
            set(self._snapshot.joint_new_peers)
            if self._snapshot.joint_new_peers is not None
            else None
        )
        for i in range(max(self._log_offset, 0), index + 1):
            cmd = self._normalize_command(self._entry_at(i).command)
            cmd_type = cmd.type
            if cmd_type == RaftCommandType.JOINT_CONSENSUS:
                assert isinstance(cmd, JointConsensusCommand)
                voters = set(cmd.old_voters)
                joint_new = set(cmd.new_voters)
            elif cmd_type == RaftCommandType.FINALIZE_CONFIG:
                assert isinstance(cmd, FinalizeConfigCommand)
                voters = set(cmd.new_voters)
                joint_new = None
        return voters, joint_new

    def _next_election_deadline(self) -> float:
        timeout = random.uniform(
            self.config.election_timeout_min,
            self.config.election_timeout_max,
        )
        return time.monotonic() + timeout

    @property
    def last_log_index(self) -> int:
        return self._log_offset + len(self.log) - 1

    @property
    def last_log_term(self) -> int:
        if not self.log:
            return self._snapshot.last_included_term
        return self.log[-1].term

    async def start(self):
        if self._run_task is not None:
            return
        self._stop_event.clear()
        change_flag = RaftStateChangeFlag(0)
        async with self._state_lock:
            # Trivial single-node cluster can become leader immediately without waiting election timeout.
            if self._joint_new_voters is None and len(self._voters) == 1 and self.role != RaftRole.LEADER:
                self.current_term += 1
                self.voted_for = self._self_peer.uid
                change_flag |= self._become_leader_locked()
        await self._run_state_change_cb(change_flag)
        self._run_task = asyncio.create_task(self._run_loop(), name=f"raft-main-{self._self_peer.uid}")
        if self.config.apply_sync:
            self._apply_task = None
        else:
            self._apply_task = asyncio.create_task(self._apply_loop(), name=f"raft-apply-{self._self_peer.uid}")

    async def stop(self):
        self._stop_event.set()
        tasks = [task for task in [self._run_task, self._apply_task] if task is not None]
        for task in tasks:
            await cancel_task(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._run_task = None
        self._apply_task = None

    def get_leader_peer_info(self) -> Optional[PeerInfo]:
        if self.leader_id is None:
            return None
        return self._peer_info_from_uid(self.leader_id)

    async def _run_loop(self):
        while not self._stop_event.is_set():
            async with self._state_lock:
                role = self.role
                deadline = self._election_deadline
            now = time.monotonic()
            if role == RaftRole.LEADER:
                await self._leader_heartbeat_once()
                await asyncio.sleep(self.config.heartbeat_interval)
                continue
            if now >= deadline:
                await self._start_election()
                continue
            await asyncio.sleep(min(0.02, max(0.0, deadline - now)))

    async def _apply_loop(self):
        while not self._stop_event.is_set():
            await self._new_commit_event.wait()
            self._new_commit_event.clear()
            async with self._state_lock:
                change_flag = self._apply_committed_entries_locked()
            await self._run_state_change_cb(change_flag)

    async def _run_state_change_cb(self, change_flag: RaftStateChangeFlag):
        if change_flag & RaftStateChangeFlag.COMMIT_APPLIED:
            await self.events.emit_async(RaftEventType.COMMIT_APPLIED)
        if change_flag & RaftStateChangeFlag.ROLE_CHANGED:
            await self.events.emit_async(RaftEventType.ROLE_CHANGED)

    def _apply_committed_entries_locked(self) -> RaftStateChangeFlag:
        change_flag = RaftStateChangeFlag(0)
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self._entry_at(self.last_applied)
            command = self._normalize_command(entry.command)
            cmd_type = command.type
            if cmd_type == RaftCommandType.JOINT_CONSENSUS:
                assert isinstance(command, JointConsensusCommand)
                self._joint_new_voters = set(command.new_voters)
                self._set_active_voters_locked(set(command.old_voters))
                continue
            if cmd_type == RaftCommandType.FINALIZE_CONFIG:
                self._joint_new_voters = None
                assert isinstance(command, FinalizeConfigCommand)
                self._set_active_voters_locked(set(command.new_voters))
                continue
            if cmd_type == RaftCommandType.APP:
                assert isinstance(command, AppCommand)
                self.state_machine.apply(command.data, self._log_offset + self.last_applied)
                change_flag |= RaftStateChangeFlag.COMMIT_APPLIED
            self._compact_log_locked()
        return change_flag

    async def _start_election_internal(self) -> RaftStateChangeFlag:
        change_flag = RaftStateChangeFlag(0)
        async with self._state_lock:
            if self.role != RaftRole.CANDIDATE:
                change_flag |= RaftStateChangeFlag.ROLE_CHANGED
            self.role = RaftRole.CANDIDATE
            self.current_term += 1
            term = self.current_term
            self.voted_for = self._self_peer.uid
            self.leader_id = None
            self._election_deadline = self._next_election_deadline()
            req = RequestVoteRequest(
                term=term,
                candidate_id=self._self_peer.uid,
                last_log_index=self.last_log_index,
                last_log_term=self.last_log_term,
            )
            vote_targets = self._replication_peer_ids_locked()
            joint_new_voters = set(self._joint_new_voters) if self._joint_new_voters is not None else None
            old_voters = set(self._voters)

        votes = 1
        grant_voters: set[PeerInfo] = {self._self_peer}
        tasks = [asyncio.create_task(self.comm.request_vote(self._peer_info_from_uid(peer).url, req)) for peer in vote_targets]
        if tasks:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            responses = []
        async with self._state_lock:
            if self.role != RaftRole.CANDIDATE or self.current_term != term:
                return change_flag
            for resp in responses:
                if isinstance(resp, Exception):
                    LOGGER.warning(
                        "request_vote RPC exception on node %s term %s: %s",
                        self._self_peer.uid,
                        term,
                        resp,
                    )
                    continue
                if not isinstance(resp, RequestVoteResponse):
                    continue
                if resp.term > self.current_term:
                    change_flag |= self._become_follower(resp.term, leader_id=None)
                    return change_flag
                if resp.vote_granted:
                    votes += 1
                    # responses are in the same order as vote_targets
            for peer, resp in zip(vote_targets, responses):
                if isinstance(resp, RequestVoteResponse) and resp.vote_granted:
                    grant_voters.add(self._peer_info_from_uid(peer))

            if joint_new_voters is None:
                won = self._has_majority(old_voters, grant_voters)
            else:
                won = self._has_majority(old_voters, grant_voters) and self._has_majority(joint_new_voters, grant_voters)
            if won:
                change_flag |= self._become_leader_locked()
        return change_flag

    async def _start_election(self):
        change_flag = await self._start_election_internal()
        await self._run_state_change_cb(change_flag)

    def _become_leader_locked(self):
        prev_role = self.role
        self.role = RaftRole.LEADER
        self.leader_id = self._self_peer.uid
        self._prune_replication_trackers_locked()
        self._ensure_replication_trackers_locked()
        if prev_role != RaftRole.LEADER:
            return RaftStateChangeFlag.ROLE_CHANGED
        return RaftStateChangeFlag(0)

    def _become_follower(self, term: int, leader_id: Optional[str]):
        prev_role = self.role
        if term > self.current_term:
            self.current_term = term
            self.voted_for = None
        self.role = RaftRole.FOLLOWER
        self.leader_id = leader_id
        self._election_deadline = self._next_election_deadline()
        self._last_contact_ts = time.monotonic()
        if prev_role != RaftRole.FOLLOWER:
            return RaftStateChangeFlag.ROLE_CHANGED
        return RaftStateChangeFlag(0)

    def _leader_rpc_preamble_locked(self, term: int, leader_id: str) -> tuple[Optional[AppendEntriesResponse], RaftStateChangeFlag]:
        change_flag = RaftStateChangeFlag(0)
        if term < self.current_term:
            return AppendEntriesResponse(
                term=self.current_term,
                success=False,
                match_index=self.last_log_index,
            ), change_flag
        if term > self.current_term or self.role != RaftRole.FOLLOWER:
            change_flag = self._become_follower(term, leader_id=leader_id)
        else:
            self.leader_id = leader_id
            self._election_deadline = self._next_election_deadline()
            self._last_contact_ts = time.monotonic()
        return None, change_flag

    async def _leader_heartbeat_once_internal(self) -> RaftStateChangeFlag:
        change_flag = RaftStateChangeFlag(0)

        async with self._state_lock:
            if self.role != RaftRole.LEADER:
                return change_flag
            term = self.current_term
            leader_commit = self.commit_index
            peer_reqs: dict[str, AppendEntriesRequest] = {}
            peer_snapshot_reqs: dict[str, InstallSnapshotRequest] = {}
            self._ensure_replication_trackers_locked()
            snapshot = self._build_snapshot_locked()
            for peer in self._replication_peer_ids_locked():
                next_idx = self._next_index.get(peer, 0)
                if next_idx < self._log_offset:
                    peer_snapshot_reqs[peer] = InstallSnapshotRequest(
                        term=term,
                        leader_id=self._self_peer.uid,
                        snapshot=snapshot,
                    )
                    continue
                next_idx = max(self._log_offset, next_idx)
                prev_idx = next_idx - 1
                prev_term = self._term_at(prev_idx)
                if prev_term is None:
                    prev_term = 0
                entries = self.log[self._local_index(next_idx):]
                peer_reqs[peer] = AppendEntriesRequest(
                    term=term,
                    leader_id=self._self_peer.uid,
                    prev_log_index=prev_idx,
                    prev_log_term=prev_term,
                    entries=list(entries),
                    leader_commit=leader_commit,
                )

        tasks = {
            peer: asyncio.create_task(self.comm.append_entries(self._peer_info_from_uid(peer).url, req))
            for peer, req in peer_reqs.items()
        }
        snapshot_tasks = {
            peer: asyncio.create_task(self.comm.install_snapshot(self._peer_info_from_uid(peer).url, req))
            for peer, req in peer_snapshot_reqs.items()
        }
        if not tasks and not snapshot_tasks:
            await self._advance_commit_index()
            return change_flag
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        snapshot_results = await asyncio.gather(*snapshot_tasks.values(), return_exceptions=True)
        async with self._state_lock:
            if self.role != RaftRole.LEADER:
                return change_flag
            for peer, result in zip(tasks.keys(), results):
                if isinstance(result, Exception):
                    LOGGER.warning(
                        "append_entries RPC exception from leader %s to peer %s(url=%s): %s",
                        self._self_peer.uid,
                        peer,
                        self._peer_info_from_uid(peer).url,
                        result,
                    )
                    continue
                if not isinstance(result, AppendEntriesResponse):
                    continue
                if result.term > self.current_term:
                    change_flag |= self._become_follower(result.term, leader_id=None)
                    return change_flag
                if result.success:
                    self._match_index[peer] = result.match_index
                    self._next_index[peer] = result.match_index + 1
                else:
                    # decrement nextIndex and retry later (leader heartbeat loop will call _leader_heartbeat_once again).
                    self._next_index[peer] = max(0, self._next_index.get(peer, 0) - 1)
            for peer, result in zip(snapshot_tasks.keys(), snapshot_results):
                if isinstance(result, Exception):
                    LOGGER.warning(
                        "install_snapshot RPC exception from leader %s to peer %s(url=%s): %s",
                        self._self_peer.uid,
                        peer,
                        self._peer_info_from_uid(peer).url,
                        result,
                    )
                    continue
                if not isinstance(result, InstallSnapshotResponse):
                    continue
                if result.term > self.current_term:
                    change_flag |= self._become_follower(result.term, leader_id=None)
                    return change_flag
                if result.success:
                    self._match_index[peer] = max(self._match_index.get(peer, -1), result.match_index)
                    self._next_index[peer] = result.match_index + 1
        await self._advance_commit_index()
        return change_flag

    async def _leader_heartbeat_once(self):
        change_flag = await self._leader_heartbeat_once_internal()
        await self._run_state_change_cb(change_flag)

    async def _advance_commit_index(self):
        change_flag = RaftStateChangeFlag(0)
        async with self._state_lock:
            if self.role != RaftRole.LEADER:
                return
            
            for index in range(self.last_log_index, self.commit_index, -1):
                entry_term = self._term_at(index)
                if entry_term != self.current_term:
                    continue
                acks: set[PeerInfo] = {self._self_peer}
                for peer, match_idx in self._match_index.items():
                    if match_idx >= index:
                        acks.add(self._peer_info_from_uid(peer))
                old_voters, joint_new_voters = self._config_state_at_index(index)
                if joint_new_voters is None:
                    committable = self._has_majority(old_voters, acks)
                else:
                    committable = self._has_majority(old_voters, acks) and self._has_majority(joint_new_voters, acks)
                if committable:
                    self.commit_index = index
                    if self.config.apply_sync:
                        change_flag |= self._apply_committed_entries_locked()
                    else:
                        self._new_commit_event.set()
                    break
        await self._run_state_change_cb(change_flag)

    async def propose(self, command: Any) -> ProposeResult:
        async with self._state_lock:
            if self.role != RaftRole.LEADER:
                return ProposeResult(success=False, leader_info=self.get_leader_peer_info(), index=None)
            self.log.append(LogEntry(term=self.current_term, command=self._make_app_command(command)))
            index = self.last_log_index
        await self._leader_heartbeat_once()
        return ProposeResult(success=True, leader_info=self._self_peer, index=index)

    async def change_configuration(self, new_voters: list[PeerInfo]) -> ChangeConfigurationResult:
        new_voter_set = set(new_voters)
        if all(peer.uid != self._self_peer.uid for peer in new_voter_set):
            raise ValueError("leader must be part of new voters")

        async with self._state_lock:
            if self.role != RaftRole.LEADER:
                return ChangeConfigurationResult(success=False, leader_info=self.get_leader_peer_info())
            old_voters = set(self._voters)
            joint_cmd = self._make_joint_consensus_command(old_voters, new_voter_set)
            self.log.append(LogEntry(term=self.current_term, command=joint_cmd))
            joint_index = self.last_log_index
            self._joint_new_voters = set(new_voter_set)
            self._rebuild_peer_info_maps(old_voters | new_voter_set)
            self._ensure_replication_trackers_locked()

        await self._leader_heartbeat_once()
        await self._wait_commit_and_apply(joint_index)

        async with self._state_lock:
            final_cmd = self._make_finalize_config_command(new_voter_set)
            self.log.append(LogEntry(term=self.current_term, command=final_cmd))
            final_index = self.last_log_index

        await self._leader_heartbeat_once()
        await self._wait_commit_and_apply(final_index)
        return ChangeConfigurationResult(
            success=True,
            leader_info=self._self_peer,
            joint_index=joint_index,
            final_index=final_index,
        )

    async def _wait_commit_and_apply(self, index: int, timeout: float = 3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            async with self._state_lock:
                if self.config.apply_sync:
                    if self.commit_index >= index:
                        return
                elif self.commit_index >= index and self.last_applied >= index:
                    return
            await asyncio.sleep(0.02)
        raise TimeoutError(f"log index {index} not committed/applied before timeout")

    async def compact_log(self):
        async with self._state_lock:
            self._compact_log_locked()

    def _compact_log_locked(self):
            compact_upto = self.last_applied
            if compact_upto < self._log_offset:
                return
            min_entries = max(1, self.config.compact_min_entries)
            if compact_upto - self._log_offset + 1 < min_entries:
                return

            compact_term = self._term_at(compact_upto)
            if compact_term is None:
                return
            snapshot_voters, snapshot_joint_new = self._config_state_at_index(compact_upto)
            cut_count = compact_upto - self._log_offset + 1
            compacted_state_machine = self.state_machine.serialize()
            self.log = self.log[cut_count:]
            self._log_offset = compact_upto + 1
            self._snapshot = RaftSnapshot(
                last_included_index=compact_upto,
                last_included_term=compact_term,
                state_machine=copy.deepcopy(compacted_state_machine),
                peers=self._peer_infos_sorted(snapshot_voters),
                joint_new_peers=(
                    sorted(snapshot_joint_new, key=lambda peer: peer.url)
                    if snapshot_joint_new is not None
                    else None
                ),
            )

            for peer in list(self._next_index.keys()):
                if self._next_index[peer] < self._log_offset:
                    self._next_index[peer] = self._log_offset
            for peer in list(self._match_index.keys()):
                if self._match_index[peer] < self._log_offset - 1:
                    self._match_index[peer] = self._log_offset - 1

    async def handle_request_vote(self, req: RequestVoteRequest) -> RequestVoteResponse:
        change_flag = RaftStateChangeFlag(0)
        async with self._state_lock:
            if req.term < self.current_term:
                return RequestVoteResponse(term=self.current_term, vote_granted=False)

            if req.term > self.current_term:
                change_flag |= self._become_follower(req.term, leader_id=None)

            up_to_date = (
                req.last_log_term > self.last_log_term
                or (
                    req.last_log_term == self.last_log_term
                    and req.last_log_index >= self.last_log_index
                )
            )
            can_vote = self.voted_for in (None, req.candidate_id)
            grant = can_vote and up_to_date
            if grant:
                self.voted_for = req.candidate_id
                self._election_deadline = self._next_election_deadline()
                self._last_contact_ts = time.monotonic()
        await self._run_state_change_cb(change_flag)
        return RequestVoteResponse(term=self.current_term, vote_granted=grant)

    async def _handle_append_entries_internal(self, req: AppendEntriesRequest) -> tuple[AppendEntriesResponse, RaftStateChangeFlag]:
        change_flag = RaftStateChangeFlag(0)
        async with self._state_lock:
            preamble_response, change_flag_cur = self._leader_rpc_preamble_locked(req.term, req.leader_id)
            change_flag |= change_flag_cur
            if preamble_response is not None:
                return preamble_response, change_flag

            if req.prev_log_index >= 0:
                # tell leader send more.
                if req.prev_log_index < self._log_offset - 1:
                    self._drop_snapshot_locked()
                if req.prev_log_index > self.last_log_index:
                    return AppendEntriesResponse(
                        term=self.current_term,
                        success=False,
                        match_index=self.last_log_index,
                    ), change_flag
                prev_term = self._term_at(req.prev_log_index)
                if prev_term != req.prev_log_term:
                    # term mismatch, delete logs one by one (leader decrement nextIndex).
                    truncate_to = self._local_index(req.prev_log_index)
                    self.log = self.log[:truncate_to]
                    assert self.commit_index <= self.last_log_index
                    assert self.last_applied <= self.last_log_index
                    return AppendEntriesResponse(
                        term=self.current_term,
                        success=False,
                        match_index=self.last_log_index,
                    ), change_flag
            # now logs up to prev_log_index are consistent,
            # but follower may have extra logs after prev_log_index,
            # which may conflict with leader's logs.
            # append new entries and delete conflicts.
            idx = req.prev_log_index + 1
            for entry in req.entries:
                if idx < self._log_offset:
                    idx += 1
                    continue
                if idx <= self.last_log_index:
                    local_idx = self._local_index(idx)
                    if self.log[local_idx].term != entry.term:
                        self.log = self.log[:local_idx]
                        self.log.append(entry)
                else:
                    self.log.append(entry)
                idx += 1

            if req.leader_commit > self.commit_index:
                self.commit_index = min(req.leader_commit, self.last_log_index)
                if self.config.apply_sync:
                    change_flag |= self._apply_committed_entries_locked()
                else:
                    self._new_commit_event.set()

            response = AppendEntriesResponse(
                term=self.current_term,
                success=True,
                match_index=self.last_log_index,
            )
        return response, change_flag

    async def handle_append_entries(self, req: AppendEntriesRequest) -> AppendEntriesResponse:
        response, change_flag = await self._handle_append_entries_internal(req)
        await self._run_state_change_cb(change_flag)
        return response

    async def _handle_install_snapshot_internal(self, req: InstallSnapshotRequest) -> tuple[InstallSnapshotResponse, RaftStateChangeFlag]:
        change_flag = RaftStateChangeFlag(0)
        async with self._state_lock:
            preamble_response, change_flag_cur = self._leader_rpc_preamble_locked(req.term, req.leader_id)
            change_flag |= change_flag_cur
            if preamble_response is not None:
                return preamble_response, change_flag

            snapshot = req.snapshot
            if snapshot.last_included_index <= self._log_offset - 1:
                return InstallSnapshotResponse(
                    term=self.current_term,
                    success=True,
                    match_index=self.last_log_index,
                ), change_flag

            prev_last_applied = self.last_applied
            prev_last_log_index = self.last_log_index
            retain_suffix = (
                snapshot.last_included_index <= prev_last_log_index
                and self._term_at(snapshot.last_included_index) == snapshot.last_included_term
            )
            if retain_suffix:
                suffix_start = snapshot.last_included_index + 1
                if suffix_start <= prev_last_log_index:
                    self.log = self.log[self._local_index(suffix_start):]
                else:
                    self.log = []
            else:
                self.log = []

            self._log_offset = snapshot.last_included_index + 1
            self._snapshot = copy.deepcopy(snapshot)
            self.state_machine.deserialize(copy.deepcopy(snapshot.state_machine))
            if any(not isinstance(peer, PeerInfo) for peer in snapshot.peers):
                raise TypeError("RaftSnapshot.peers must be list[PeerInfo]")
            if snapshot.joint_new_peers is not None and any(
                not isinstance(peer, PeerInfo) for peer in snapshot.joint_new_peers
            ):
                raise TypeError("RaftSnapshot.joint_new_peers must be list[PeerInfo]")
            snapshot_peer_infos = list(snapshot.peers)
            self._snapshot.peers = snapshot_peer_infos
            voters = set(snapshot_peer_infos)
            voters.add(self._self_peer)
            self._set_active_voters_locked(voters)
            self._joint_new_voters = (
                set(self._snapshot.joint_new_peers)
                if self._snapshot.joint_new_peers is not None
                else None
            )
            if self._joint_new_voters is not None:
                self._rebuild_peer_info_maps(self._voters | self._joint_new_voters)

            replay_upto = min(prev_last_applied, self.last_log_index)
            for replay_idx in range(self._log_offset, replay_upto + 1):
                replay_cmd = self._normalize_command(self._entry_at(replay_idx).command)
                if replay_cmd.type == RaftCommandType.APP:
                    assert isinstance(replay_cmd, AppCommand)
                    self.state_machine.apply(replay_cmd.data, replay_idx)
                    change_flag |= RaftStateChangeFlag.COMMIT_APPLIED

            self.commit_index = max(self.commit_index, snapshot.last_included_index)
            self.last_applied = max(self.last_applied, snapshot.last_included_index)

            self._prune_replication_trackers_locked()

            response = InstallSnapshotResponse(
                term=self.current_term,
                success=True,
                match_index=self.last_log_index,
            )
        return response, change_flag

    async def handle_install_snapshot(self, req: InstallSnapshotRequest) -> InstallSnapshotResponse:
        response, change_flag = await self._handle_install_snapshot_internal(req)
        await self._run_state_change_cb(change_flag)
        return response

class InMemoryRaftNetwork:
    def __init__(self):
        self._nodes: dict[str, RaftNode] = {}

    def register(self, node: RaftNode):
        self._nodes[node.self_peer.uid] = node

    def unregister(self, node_id: str):
        self._nodes.pop(node_id, None)

    async def request_vote(self, peer_id: str, req: RequestVoteRequest) -> RequestVoteResponse:
        peer = self._nodes.get(peer_id)
        if peer is None:
            raise ConnectionError(f"peer {peer_id} not available")
        return await peer.handle_request_vote(req)

    async def append_entries(self, peer_id: str, req: AppendEntriesRequest) -> AppendEntriesResponse:
        peer = self._nodes.get(peer_id)
        if peer is None:
            raise ConnectionError(f"peer {peer_id} not available")
        return await peer.handle_append_entries(req)

    async def install_snapshot(self, peer_id: str, req: InstallSnapshotRequest) -> InstallSnapshotResponse:
        peer = self._nodes.get(peer_id)
        if peer is None:
            raise ConnectionError(f"peer {peer_id} not available")
        return await peer.handle_install_snapshot(req)


class InMemoryRaftComm(AsyncRaftComm):
    def __init__(self, network: InMemoryRaftNetwork):
        self._network = network

    async def request_vote(self, peer_id: str, req: RequestVoteRequest) -> RequestVoteResponse:
        return await self._network.request_vote(peer_id, req)

    async def append_entries(self, peer_id: str, req: AppendEntriesRequest) -> AppendEntriesResponse:
        return await self._network.append_entries(peer_id, req)

    async def install_snapshot(self, peer_id: str, req: InstallSnapshotRequest) -> InstallSnapshotResponse:
        return await self._network.install_snapshot(peer_id, req)


__all__ = [
    "AppendEntriesRequest",
    "AppendEntriesResponse",
    "AppCommand",
    "AsyncRaftComm",
    "ChangeConfigurationResult",
    "FinalizeConfigCommand",
    "InMemoryRaftComm",
    "InMemoryRaftNetwork",
    "InstallSnapshotRequest",
    "InstallSnapshotResponse",
    "JointConsensusCommand",
    "LogEntry",
    "NotLeaderError",
    "ProposeResult",
    "PeerInfo",
    "RaftConfig",
    "RaftCommandType",
    "RaftEventType",
    "RaftSnapshot",
    "RaftNode",
    "RaftRole",
    "RequestVoteRequest",
    "RequestVoteResponse",
    "StateMachine",
    "ListStateMachine",
]
