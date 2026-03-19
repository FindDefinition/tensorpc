

import asyncio
import contextlib
import os
import traceback
import pytest_asyncio
import rich
from tensorpc.apps.cm.coretypes import WorkerInfo
from tensorpc.core import asyncclient
from tensorpc.core.distributed.raft import PeerInfo
from tensorpc.utils.wait_tools import get_free_ports
import subprocess
from tensorpc.apps.cm.serv_names import master_serv_names

TEST_LOCAL_SSH_USERNAME = os.getenv("TEST_LOCAL_SSH_USERNAME")
TEST_LOCAL_SSH_PASSWORD = os.getenv("TEST_LOCAL_SSH_PASSWORD")

assert TEST_LOCAL_SSH_USERNAME is not None, "TEST_LOCAL_SSH_USERNAME env var must be set for the test to run"
assert TEST_LOCAL_SSH_PASSWORD is not None, "TEST_LOCAL_SSH_PASSWORD env var must be set for the test to run"

@contextlib.asynccontextmanager
async def create_master_nodes(num_node: int):
    ports = get_free_ports(num_node)
    procs: list[subprocess.Popen] = []
    peer_infos: list[PeerInfo] = []
    for i in range(num_node):
        port = ports[i]
        uid = f"node-{i}"
        url = f"localhost:{port}"
        proc = subprocess.Popen(
            f"python -m tensorpc.apps.cm.cli --uid=\"{uid}\" --port {port} --username {TEST_LOCAL_SSH_USERNAME} --password {TEST_LOCAL_SSH_PASSWORD}",
            shell=True)
        peer_info = PeerInfo(uid=uid, url=url)
        procs.append(proc)
        peer_infos.append(peer_info)
    try:
        yield peer_infos
    finally:
        print("try shutdown")
        for peer_info in peer_infos:
            async with asyncclient.AsyncRemoteManager(peer_info.url) as robj:
                await robj.wait_for_channel_ready()
                await robj.shutdown()
        for proc in procs:
            proc.wait()



@pytest_asyncio.fixture
async def master_nodes_single():
    async with create_master_nodes(1) as peer_infos:
        yield peer_infos

async def _main():

    async with create_master_nodes(8) as peer_infos:
        print("Started master nodes with peer infos:")
        for peer_info in peer_infos:
            print(peer_info)


        all_node_urls = [peer_info.url for peer_info in peer_infos]
        try:
            async with asyncclient.AsyncRemoteManager(all_node_urls[0]) as robj:
                await robj.wait_for_channel_ready()
                kwargs = {
                    "self_url": all_node_urls[0],
                    "all_node_urls": all_node_urls,
                    "num_partition": 2,
                }

                await robj.remote_call(
                    master_serv_names.GROUP_TREE_SCAN_GROUPS,
                    **kwargs
                )
                num_raft = 1
                compute_node_infos = [WorkerInfo(peer_info=peer_info, rank=i) for i, peer_info in enumerate(peer_infos)]
                raft_node_infos = [peer_infos[i] for i in range(num_raft)]
                
                kwargs = {
                    "group_id": "test-group",
                    "self_uid": peer_infos[0].uid,
                    "compute_node_infos": compute_node_infos,
                    "raft_node_infos": raft_node_infos,
                    "num_partition": 2,
                }


                await robj.remote_call(
                    master_serv_names.GROUP_MASTER_CREATE_GROUP,
                    **kwargs
                )

                await asyncio.sleep(15)
                kwargs = {
                    "self_url": all_node_urls[0],
                    "all_node_urls": all_node_urls,
                    "num_partition": 2,
                }

                group_infos = await robj.remote_call(
                    master_serv_names.GROUP_TREE_SCAN_GROUPS,
                    **kwargs
                )

                rich.print("FINISH SLEEP", group_infos)

        except:
            traceback.print_exc()

        print("WTFWTF")



if __name__ == "__main__":
    asyncio.run(_main())