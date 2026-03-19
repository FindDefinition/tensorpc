import asyncio
from typing import Any, Iterable, Optional

import pytest

from tensorpc.core.distributed.raft import (
    AppCommand,
    AppendEntriesRequest,
    ChangeConfigurationResult,
    InMemoryRaftComm,
    InMemoryRaftNetwork,
    InstallSnapshotRequest,
    LogEntry,
    RaftConfig,
    RaftEventType,
    RaftNode,
    RaftRole,
    RaftSnapshot,
    ProposeResult,
    PeerInfo,
)
from tensorpc.core.event_emitter.aio import AsyncIOEventEmitter


async def _wait_for_leader(nodes: list[RaftNode], timeout: float = 3.0) -> RaftNode:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        leaders = [node for node in nodes if node.role == RaftRole.LEADER]
        if len(leaders) == 1:
            return leaders[0]
        await asyncio.sleep(0.02)
    raise TimeoutError("leader election timed out")


async def _wait_until_all_applied(nodes: list[RaftNode], value: object, timeout: float = 3.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if all(value in node.state_machine.serialize() for node in nodes):
            return
        await asyncio.sleep(0.02)
    raise TimeoutError("value replication/apply timed out")


async def _wait_until_applied_count(node: RaftNode, count: int, timeout: float = 3.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if len(node.state_machine.serialize()) >= count:
            return
        await asyncio.sleep(0.02)
    raise TimeoutError("state machine apply count timed out")


async def _wait_until(predicate, timeout: float = 3.0, interval: float = 0.02):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise TimeoutError("wait condition timed out")


def _peer_uids(node: RaftNode) -> set[str]:
    return {peer.uid if hasattr(peer, "uid") else str(peer) for peer in node.peers}


def _peer(uid: str) -> PeerInfo:
    return PeerInfo(uid=uid, url=uid)


def _peer_ips(peers: Optional[Iterable[Any]]) -> set[str]:
    if peers is None:
        return set()
    return {peer.url if hasattr(peer, "url") else str(peer) for peer in peers}


@pytest.mark.asyncio
async def test_simple_raft_in_memory_replication():
    network = InMemoryRaftNetwork()
    node_ids = ["n1", "n2", "n3"]
    config = RaftConfig(
        election_timeout_min=0.08,
        election_timeout_max=0.16,
        heartbeat_interval=0.03,
    )

    nodes: list[RaftNode] = []
    for node_id in node_ids:
        peers = [_peer(peer) for peer in node_ids if peer != node_id]
        node = RaftNode(self_peer=_peer(node_id), peers=peers, comm=InMemoryRaftComm(network), config=config)
        network.register(node)
        nodes.append(node)

    try:
        await asyncio.gather(*(node.start() for node in nodes))

        leader = await _wait_for_leader(nodes)
        command = {"op": "set", "k": "x", "v": 1}
        result = await leader.propose(command)

        assert result.success is True
        assert result.index == 0
        await _wait_until_all_applied(nodes, command)
        assert all(node.commit_index >= 0 for node in nodes)
    finally:
        await asyncio.gather(*(node.stop() for node in nodes), return_exceptions=True)
        for node_id in node_ids:
            network.unregister(node_id)


@pytest.mark.asyncio
async def test_propose_return_status_on_leader_and_follower():
    network = InMemoryRaftNetwork()
    node_ids = ["n1", "n2", "n3"]
    config = RaftConfig(
        election_timeout_min=0.08,
        election_timeout_max=0.16,
        heartbeat_interval=0.03,
    )

    nodes: list[RaftNode] = []
    for node_id in node_ids:
        peers = [_peer(peer) for peer in node_ids if peer != node_id]
        node = RaftNode(self_peer=_peer(node_id), peers=peers, comm=InMemoryRaftComm(network), config=config)
        network.register(node)
        nodes.append(node)

    try:
        await asyncio.gather(*(node.start() for node in nodes))
        leader = await _wait_for_leader(nodes)
        follower = next(node for node in nodes if node.self_peer.uid != leader.self_peer.uid)

        await _wait_until(lambda: follower.leader_id is not None)

        leader_payload = {"mode": "status", "target": "leader"}
        leader_result = await leader.propose(leader_payload)
        assert isinstance(leader_result, ProposeResult)
        assert leader_result.success is True
        assert leader_result.leader_info == leader.self_peer
        assert leader_result.index is not None

        follower_result = await follower.propose({"mode": "status", "target": "follower"})
        assert isinstance(follower_result, ProposeResult)
        assert follower_result.success is False
        assert follower_result.leader_info == leader.self_peer
        assert follower_result.index is None
    finally:
        await asyncio.gather(*(node.stop() for node in nodes), return_exceptions=True)
        for node_id in node_ids:
            network.unregister(node_id)


@pytest.mark.asyncio
async def test_raft_joint_consensus_config_change():
    network = InMemoryRaftNetwork()
    node_ids = ["n1", "n2", "n3", "n4"]
    config = RaftConfig(
        election_timeout_min=0.08,
        election_timeout_max=0.16,
        heartbeat_interval=0.03,
    )

    nodes: dict[str, RaftNode] = {}
    for node_id in node_ids:
        initial_members = ["n1", "n2", "n3"]
        peers = [_peer(peer) for peer in initial_members if peer != node_id]
        node = RaftNode(self_peer=_peer(node_id), peers=peers, comm=InMemoryRaftComm(network), config=config)
        network.register(node)
        nodes[node_id] = node

    try:
        await asyncio.gather(*(node.start() for node in nodes.values()))

        leader = await _wait_for_leader(list(nodes.values()))

        cfg_result = await leader.change_configuration([_peer("n1"), _peer("n2"), _peer("n3"), _peer("n4")])
        assert isinstance(cfg_result, ChangeConfigurationResult)
        assert cfg_result.success is True
        assert cfg_result.leader_info == leader.self_peer
        assert cfg_result.joint_index is not None
        assert cfg_result.final_index is not None
        assert cfg_result.final_index == cfg_result.joint_index + 1

        command = {"op": "set", "k": "reconfig", "v": 1}
        propose_result = await leader.propose(command)
        assert propose_result.success is True
        await _wait_until_all_applied(list(nodes.values()), command)

        assert all(node.commit_index >= 2 for node in nodes.values())
        assert all("n4" in _peer_uids(node) for node in [nodes["n1"], nodes["n2"], nodes["n3"]])
        assert _peer_uids(nodes["n4"]) == {"n1", "n2", "n3"}
    finally:
        await asyncio.gather(*(node.stop() for node in nodes.values()), return_exceptions=True)
        for node_id in node_ids:
            network.unregister(node_id)


@pytest.mark.asyncio
async def test_raft_reconfig_prune_stale_replication_trackers():
    network = InMemoryRaftNetwork()
    node_ids = ["n1", "n2", "n3", "n4"]
    config = RaftConfig(
        election_timeout_min=0.08,
        election_timeout_max=0.16,
        heartbeat_interval=0.03,
    )

    nodes: dict[str, RaftNode] = {}
    for node_id in node_ids:
        peers = [_peer(peer) for peer in node_ids if peer != node_id]
        node = RaftNode(self_peer=_peer(node_id), peers=peers, comm=InMemoryRaftComm(network), config=config)
        network.register(node)
        nodes[node_id] = node

    try:
        await asyncio.gather(*(node.start() for node in nodes.values()))
        leader = await _wait_for_leader(list(nodes.values()))

        removed_node_id = next(node_id for node_id in node_ids if node_id != leader.self_peer.uid)
        shrink_voters = [_peer(node_id) for node_id in node_ids if node_id != removed_node_id]

        shrink_result = await leader.change_configuration(shrink_voters)
        assert shrink_result.success is True
        await _wait_until(lambda: removed_node_id not in leader._next_index)
        await _wait_until(lambda: removed_node_id not in leader._match_index)

        expand_result = await leader.change_configuration([_peer(node_id) for node_id in node_ids])
        assert expand_result.success is True
        assert removed_node_id in leader._next_index
        assert removed_node_id in leader._match_index
    finally:
        await asyncio.gather(*(node.stop() for node in nodes.values()), return_exceptions=True)
        for node_id in node_ids:
            network.unregister(node_id)


@pytest.mark.asyncio
async def test_raft_log_compaction_and_user_compact_hook():
    network = InMemoryRaftNetwork()
    node = RaftNode(
        self_peer=_peer("n1"),
        peers=[],
        comm=InMemoryRaftComm(network),
        config=RaftConfig(
            election_timeout_min=0.08,
            election_timeout_max=0.16,
            heartbeat_interval=0.03,
        ),
    )
    network.register(node)

    try:
        await node.start()
        leader = await _wait_for_leader([node])
        assert leader is node

        for i in range(8):
            propose_result = await node.propose({"v": i})
            assert propose_result.success is True

        await _wait_until_applied_count(node, 8)

        assert node._log_offset > 0
        assert len(node.log) == 0
    finally:
        await node.stop()
        network.unregister("n1")


@pytest.mark.asyncio
async def test_raft_compact_min_entries_threshold():
    network = InMemoryRaftNetwork()
    node = RaftNode(
        self_peer=_peer("n1"),
        peers=[],
        comm=InMemoryRaftComm(network),
        config=RaftConfig(
            election_timeout_min=0.08,
            election_timeout_max=0.16,
            heartbeat_interval=0.03,
            compact_min_entries=3,
        ),
    )
    network.register(node)

    try:
        await node.start()
        await _wait_for_leader([node])

        propose_result = await node.propose({"v": 0})
        assert propose_result.success is True
        await _wait_until_applied_count(node, 1)
        assert node._log_offset == 0

        propose_result = await node.propose({"v": 1})
        assert propose_result.success is True
        await _wait_until_applied_count(node, 2)
        assert node._log_offset == 0

        propose_result = await node.propose({"v": 2})
        assert propose_result.success is True
        await _wait_until_applied_count(node, 3)
        await _wait_until(lambda: node._log_offset >= 3)
    finally:
        await node.stop()
        network.unregister("n1")


@pytest.mark.asyncio
async def test_raft_apply_sync_mode():
    network = InMemoryRaftNetwork()
    node = RaftNode(
        self_peer=_peer("n1"),
        peers=[],
        comm=InMemoryRaftComm(network),
        config=RaftConfig(
            election_timeout_min=0.08,
            election_timeout_max=0.16,
            heartbeat_interval=0.03,
            compact_min_entries=100,
            apply_sync=True,
        ),
    )
    network.register(node)

    try:
        await node.start()
        assert node._apply_task is None
        await _wait_for_leader([node])

        payload = {"v": "sync"}
        propose_result = await node.propose(payload)
        assert propose_result.success is True
        assert payload in node.state_machine.serialize()
        assert node.last_applied >= 0
    finally:
        await node.stop()
        network.unregister("n1")


@pytest.mark.asyncio
async def test_raft_commit_applied_event_emitted_async_mode():
    network = InMemoryRaftNetwork()
    event_count = 0

    async def _on_commit_applied(*_args: Any):
        nonlocal event_count
        event_count += 1

    events = AsyncIOEventEmitter()
    events.on(RaftEventType.COMMIT_APPLIED, _on_commit_applied)

    node = RaftNode(
        self_peer=_peer("n1"),
        peers=[],
        comm=InMemoryRaftComm(network),
        events=events,
        config=RaftConfig(
            election_timeout_min=0.08,
            election_timeout_max=0.16,
            heartbeat_interval=0.03,
            compact_min_entries=100,
            apply_sync=False,
        ),
    )
    network.register(node)

    try:
        await node.start()
        await _wait_for_leader([node])

        payload = {"v": "cb-async"}
        propose_result = await node.propose(payload)
        assert propose_result.success is True
        await _wait_until(lambda: event_count >= 1 and payload in node.state_machine.serialize())
    finally:
        await node.stop()
        network.unregister("n1")


@pytest.mark.asyncio
async def test_raft_commit_applied_event_emitted_sync_mode():
    network = InMemoryRaftNetwork()
    event_count = 0

    async def _on_commit_applied(*_args: Any):
        nonlocal event_count
        event_count += 1

    events = AsyncIOEventEmitter()
    events.on(RaftEventType.COMMIT_APPLIED, _on_commit_applied)

    node = RaftNode(
        self_peer=_peer("n1"),
        peers=[],
        comm=InMemoryRaftComm(network),
        events=events,
        config=RaftConfig(
            election_timeout_min=0.08,
            election_timeout_max=0.16,
            heartbeat_interval=0.03,
            compact_min_entries=100,
            apply_sync=True,
        ),
    )
    network.register(node)

    try:
        await node.start()
        await _wait_for_leader([node])

        payload = {"v": "cb-sync"}
        propose_result = await node.propose(payload)
        assert propose_result.success is True
        assert event_count >= 1
        assert payload in node.state_machine.serialize()
    finally:
        await node.stop()
        network.unregister("n1")


@pytest.mark.asyncio
async def test_raft_role_changed_event_emitted_single_node_startup():
    network = InMemoryRaftNetwork()
    event_count = 0

    async def _on_role_changed(*_args: Any):
        nonlocal event_count
        event_count += 1

    events = AsyncIOEventEmitter()
    events.on(RaftEventType.ROLE_CHANGED, _on_role_changed)

    node = RaftNode(
        self_peer=_peer("n1"),
        peers=[],
        comm=InMemoryRaftComm(network),
        events=events,
        config=RaftConfig(
            election_timeout_min=0.5,
            election_timeout_max=1.0,
            heartbeat_interval=0.03,
            compact_min_entries=100,
        ),
    )
    network.register(node)

    try:
        await node.start()
        assert node.role == RaftRole.LEADER
        assert event_count == 1
    finally:
        await node.stop()
        network.unregister("n1")


@pytest.mark.asyncio
async def test_raft_role_changed_event_emitted_when_becoming_candidate():
    network = InMemoryRaftNetwork()
    event_count = 0

    async def _on_role_changed(*_args: Any):
        nonlocal event_count
        event_count += 1

    events = AsyncIOEventEmitter()
    events.on(RaftEventType.ROLE_CHANGED, _on_role_changed)

    node = RaftNode(
        self_peer=_peer("n1"),
        peers=[_peer("n2")],
        comm=InMemoryRaftComm(network),
        events=events,
        config=RaftConfig(
            election_timeout_min=0.08,
            election_timeout_max=0.12,
            heartbeat_interval=0.03,
            compact_min_entries=100,
        ),
    )
    network.register(node)

    try:
        await node.start()
        await _wait_until(lambda: node.role == RaftRole.CANDIDATE and event_count >= 1)
    finally:
        await node.stop()
        network.unregister("n1")


@pytest.mark.asyncio
async def test_raft_role_changed_event_emitted_when_snapshot_forces_follower():
    network = InMemoryRaftNetwork()
    event_count = 0

    async def _on_role_changed(*_args: Any):
        nonlocal event_count
        event_count += 1

    events = AsyncIOEventEmitter()
    events.on(RaftEventType.ROLE_CHANGED, _on_role_changed)

    node = RaftNode(
        self_peer=_peer("n1"),
        peers=[],
        comm=InMemoryRaftComm(network),
        events=events,
        config=RaftConfig(
            election_timeout_min=0.5,
            election_timeout_max=1.0,
            heartbeat_interval=0.03,
            compact_min_entries=100,
        ),
    )
    network.register(node)

    try:
        await node.start()
        assert node.role == RaftRole.LEADER
        assert event_count == 1

        response = await node.handle_install_snapshot(
            InstallSnapshotRequest(
                term=node.current_term + 1,
                leader_id="n2",
                snapshot=RaftSnapshot(
                    last_included_index=-1,
                    last_included_term=0,
                    state_machine=[],
                    peers=[_peer("n2")],
                    joint_new_peers=None,
                ),
            )
        )

        assert response.success is True
        assert node.role == RaftRole.FOLLOWER
        assert node.leader_id == "n2"
        assert event_count == 2
    finally:
        await node.stop()
        network.unregister("n1")


@pytest.mark.asyncio
async def test_raft_single_node_starts_leader_without_election_delay():
    network = InMemoryRaftNetwork()
    node = RaftNode(
        self_peer=_peer("n1"),
        peers=[],
        comm=InMemoryRaftComm(network),
        config=RaftConfig(
            election_timeout_min=0.5,
            election_timeout_max=1.0,
            heartbeat_interval=0.03,
            compact_min_entries=100,
        ),
    )
    network.register(node)

    try:
        await node.start()
        assert node.role == RaftRole.LEADER
        assert node.leader_id == node.self_peer.uid

        payload = {"v": "instant"}
        propose_result = await node.propose(payload)
        assert propose_result.success is True
        assert propose_result.index == 0
        await _wait_until(lambda: node.last_applied >= 0 and payload in node.state_machine.serialize())
    finally:
        await node.stop()
        network.unregister("n1")


@pytest.mark.asyncio
async def test_raft_snapshot_contains_reconfig_peers_for_lagging_node():
    network = InMemoryRaftNetwork()
    node_ids = ["n1", "n2", "n3", "n4"]
    config = RaftConfig(
        election_timeout_min=0.08,
        election_timeout_max=0.16,
        heartbeat_interval=0.03,
    )

    nodes: dict[str, RaftNode] = {}
    initial_members = ["n1", "n2", "n3"]
    for node_id in node_ids:
        peers = [_peer(peer) for peer in initial_members if peer != node_id]
        node = RaftNode(self_peer=_peer(node_id), peers=peers, comm=InMemoryRaftComm(network), config=config)
        network.register(node)
        nodes[node_id] = node

    lagging_id = "n4"
    try:
        await asyncio.gather(*(nodes[node_id].start() for node_id in initial_members))
        leader = await _wait_for_leader([nodes[node_id] for node_id in initial_members])

        network.unregister(lagging_id)

        cfg_result = await leader.change_configuration([_peer(node_id) for node_id in node_ids])
        assert cfg_result.success is True
        for i in range(10):
            propose_result = await leader.propose({"k": "snap", "v": i})
            assert propose_result.success is True

        await _wait_until(lambda: leader._log_offset > 0, timeout=4.0)

        snapshot_req = InstallSnapshotRequest(
            term=leader.current_term,
            leader_id=leader.self_peer.uid,
            snapshot=leader._build_snapshot_locked(),
        )
        await nodes[lagging_id].handle_install_snapshot(snapshot_req)

        assert _peer_uids(nodes[lagging_id]) == {"n1", "n2", "n3"}
        assert _peer_ips(nodes[lagging_id]._voters) == set(node_ids)
    finally:
        await asyncio.gather(*(node.stop() for node in nodes.values()), return_exceptions=True)
        for node_id in node_ids:
            network.unregister(node_id)


@pytest.mark.asyncio
async def test_raft_snapshot_persists_joint_configuration():
    network = InMemoryRaftNetwork()
    source = RaftNode(self_peer=_peer("n1"), peers=[_peer("n2"), _peer("n3")], comm=InMemoryRaftComm(network))
    target = RaftNode(self_peer=_peer("n2"), peers=[_peer("n1"), _peer("n3")], comm=InMemoryRaftComm(network))

    source._log_offset = 5
    source._snapshot = RaftSnapshot(
        last_included_index=4,
        last_included_term=2,
        state_machine=[{"k": "v"}],
        peers=[_peer("n1"), _peer("n2"), _peer("n3"), _peer("n4")],
        joint_new_peers=[_peer("n1"), _peer("n2"), _peer("n3"), _peer("n4")],
    )

    snapshot_req = InstallSnapshotRequest(
        term=1,
        leader_id="n1",
        snapshot=source._build_snapshot_locked(),
    )
    await target.handle_install_snapshot(snapshot_req)

    assert _peer_ips(target._joint_new_voters) == {"n1", "n2", "n3", "n4"}
    assert _peer_ips(target._voters) == {"n1", "n2", "n3", "n4"}
    assert _peer_uids(target) == {"n1", "n3", "n4"}


@pytest.mark.asyncio
async def test_raft_follower_drops_snapshot_on_old_append_entries():
    node = RaftNode(self_peer=_peer("n1"), peers=[_peer("n2")], comm=InMemoryRaftComm(InMemoryRaftNetwork()))

    node._log_offset = 8
    node._snapshot = RaftSnapshot(
        last_included_index=7,
        last_included_term=3,
        state_machine=[{"k": "old"}],
        peers=[_peer("n1"), _peer("n2")],
        joint_new_peers=None,
    )
    node.state_machine.deserialize([{"k": "old"}])
    node.commit_index = 10
    node.last_applied = 10
    node.current_term = 1

    req = AppendEntriesRequest(
        term=1,
        leader_id="n2",
        prev_log_index=2,
        prev_log_term=1,
        entries=[],
        leader_commit=0,
    )
    resp = await node.handle_append_entries(req)

    assert resp.success is False
    assert node._log_offset == 0
    assert node._snapshot.last_included_term == 0
    assert node.state_machine.serialize() == []
    assert node.commit_index == -1
    assert node.last_applied == -1


@pytest.mark.asyncio
async def test_raft_compact_log_only_committed_entries():
    node = RaftNode(
        self_peer=_peer("n1"),
        peers=[],
        comm=InMemoryRaftComm(InMemoryRaftNetwork()),
        config=RaftConfig(),
    )

    node.log = [LogEntry(term=1, command=AppCommand(data=i)) for i in range(6)]
    node.last_applied = 5
    node.commit_index = 2
    node.state_machine.deserialize([0, 1, 2])
    await node.compact_log()

    assert node._log_offset == 6
    assert len(node.log) == 0


@pytest.mark.asyncio
async def test_raft_snapshot_state_only_up_to_compact_upto():
    network = InMemoryRaftNetwork()
    leader = RaftNode(
        self_peer=_peer("n1"),
        peers=[],
        comm=InMemoryRaftComm(network),
        config=RaftConfig(
            election_timeout_min=0.08,
            election_timeout_max=0.16,
            heartbeat_interval=0.03,
        ),
    )
    network.register(leader)

    try:
        await leader.start()
        await _wait_for_leader([leader])
        for i in range(5):
            propose_result = await leader.propose({"v": i})
            assert propose_result.success is True
        await _wait_until_applied_count(leader, 5)
    finally:
        await leader.stop()
        network.unregister("n1")

    assert leader._log_offset == 5
    assert leader._snapshot.state_machine == [{"v": 0}, {"v": 1}, {"v": 2}, {"v": 3}, {"v": 4}]
    assert leader.state_machine.serialize() == [{"v": 0}, {"v": 1}, {"v": 2}, {"v": 3}, {"v": 4}]


@pytest.mark.asyncio
async def test_raft_install_snapshot_retains_suffix_and_replays_applied():
    node = RaftNode(self_peer=_peer("n1"), peers=[_peer("n2")], comm=InMemoryRaftComm(InMemoryRaftNetwork()))
    node.current_term = 1
    node.log = [
        LogEntry(term=1, command=AppCommand(data={"v": 0})),
        LogEntry(term=1, command=AppCommand(data={"v": 1})),
        LogEntry(term=2, command=AppCommand(data={"v": 2})),
        LogEntry(term=2, command=AppCommand(data={"v": 3})),
        LogEntry(term=2, command=AppCommand(data={"v": 4})),
    ]
    node.commit_index = 3
    node.last_applied = 3
    node.state_machine.deserialize([{"v": 0}, {"v": 1}, {"v": 2}, {"v": 3}])

    snapshot_req = InstallSnapshotRequest(
        term=1,
        leader_id="n2",
        snapshot=RaftSnapshot(
            last_included_index=2,
            last_included_term=2,
            state_machine=[{"v": 0}, {"v": 1}, {"v": 2}],
            peers=[_peer("n1"), _peer("n2")],
        ),
    )
    await node.handle_install_snapshot(snapshot_req)

    assert node._log_offset == 3
    assert len(node.log) == 2
    assert node.log[0].command.data == {"v": 3}
    assert node.log[1].command.data == {"v": 4}
    assert node.state_machine.serialize() == [{"v": 0}, {"v": 1}, {"v": 2}, {"v": 3}]
    assert node.last_applied == 3
    assert node.commit_index == 3
