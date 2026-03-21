import abc 
import contextlib
from typing import TYPE_CHECKING, Any, AsyncGenerator, Generic, Optional, Self, TypeVar

from tensorpc.autossh.core import SSHConnDesc, enter_ssh_jumped_addr
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.core.asyncclient import AsyncRemoteManager
from tensorpc.dock.components import mui
from tensorpc.apps.cm.coretypes import ClusterBaseInfo, ResourceInfo

@dataclasses.dataclass
class NodeSpec:
    id: str
    tags: list[mui.ChipGroupItem]
    resource_spec: ResourceInfo
    # url inside cluster. we assume all nodes in same cluster can access each other 
    # through this url. it can be ip:port or domain:port.
    # it can also be global url if we can access agent through localhost.
    local_url_with_port: str
    # if local_url_with_port/external_url_with_port don't work, 
    # we first try tensorpc grpc servers to access it.
    # remote jump call (chunk) is enabled for all tensorpc grpc servers by default.
    tensorpc_jump_urls: Optional[list[str]] = None
    # use external_url_with_port in app client (not in cluster) if set.
    external_url_with_port: Optional[str] = None  
    # you assign a unique id when you start NodeMaster server, this may
    # different with node id from provider. this will be set after scan_group.
    server_id: Optional[str] = None

    @property 
    def client_url_with_port(self):
        if self.external_url_with_port is not None:
            return self.external_url_with_port
        return self.local_url_with_port

@dataclasses.dataclass
class ClusterSpec:
    id: str 
    name: str 
    nodes: list[NodeSpec]
    # ssh jumps, we will create ssh tunnels before apply remote object
    # and tensorpc_jump_urls
    # all nodes in same cluster must share same ssh jumps.
    ssh_jumps: Optional[list[SSHConnDesc]] = None

    def is_ssh_jump_equal(self, other: Self) -> bool:
        if self.ssh_jumps is None and other.ssh_jumps is None:
            return True
        if (self.ssh_jumps is None) != (other.ssh_jumps is None):
            return False
        # both not None
        assert self.ssh_jumps is not None and other.ssh_jumps is not None
        if len(self.ssh_jumps) != len(other.ssh_jumps):
            return False
        for jump1, jump2 in zip(self.ssh_jumps, other.ssh_jumps):
            if jump1 != jump2:
                return False
        return True

    async def simple_chunk_call_async(self,
                                    node: NodeSpec,
                                    key,
                                    *args,
                                    rpc_timeout=None,
                                    **kwargs):
        url = node.external_url_with_port or node.local_url_with_port
        async with enter_ssh_jumped_addr(url, self.ssh_jumps or []) as (addr, _):

            async with AsyncRemoteManager(addr) as robj:
                res = await robj.chunked_remote_call(key,
                                                    *args,
                                                    rpc_timeout=rpc_timeout,
                                                    rpc_relay_urls=node.tensorpc_jump_urls,
                                                    **kwargs)
        return res


class ClusterProviderBase(abc.ABC):
    """Create/Discover/Delete cluster resources.
    """
    @abc.abstractmethod
    def support_creation(self) -> bool:
        # if we use fixed nodes, this function should return False
        # this should be fixed during the lifecycle of the App.
        return True  

    @abc.abstractmethod
    async def get_create_ui(self) -> mui.FlexBox:
        pass 

    async def get_tags(self) -> list[dict[str, Any]]:
        return [] 


    @abc.abstractmethod
    async def create_nodes(self, ui: mui.FlexBox):
        """Create nodes with the given specification.
        Note that you can only find these nodes through discover after they are created.
        this function don't return NodeSpec.
        """
        pass

    @abc.abstractmethod
    async def discover(self) -> list[ClusterSpec]:
        """Discover nodes.
        """
        pass

    @abc.abstractmethod
    async def delete_nodes(self, node_specs: list[NodeSpec]):
        """Delete nodes with the given spec.
        """
        pass

    async def discover_with_validation(self) -> list[ClusterSpec]:
        cluster_specs = await self.discover()
        # validate url and uid uniqueness for all nodes in each cluster.
        for cluster_spec in cluster_specs:
            url_set = set()
            uid_set = set()
            for node_spec in cluster_spec.nodes:
                if node_spec.client_url_with_port in url_set:
                    raise ValueError(f"Duplicate url {node_spec.client_url_with_port} in cluster {cluster_spec.name}")
                url_set.add(node_spec.client_url_with_port)
                if node_spec.id in uid_set:
                    raise ValueError(f"Duplicate id {node_spec.id} in cluster {cluster_spec.name}")
                uid_set.add(node_spec.id)
        return cluster_specs

class FixedClusterProvider(ClusterProviderBase):
    """A cluster provider with fixed nodes. It will only discover nodes and doesn't support creation and deletion.
    """
    def __init__(self, cluster_specs: list[ClusterSpec]):
        self._cluster_specs = cluster_specs

    def support_creation(self) -> bool:
        return False 

    async def get_create_ui(self) -> mui.FlexBox:
        raise NotImplementedError("FixedClusterProvider doesn't support creation.")

    async def create_nodes(self, ui: mui.FlexBox):
        raise NotImplementedError("FixedClusterProvider doesn't support creation.")

    async def discover(self) -> list[ClusterSpec]:
        return self._cluster_specs

    async def delete_nodes(self, node_specs: list[NodeSpec]):
        raise NotImplementedError("FixedClusterProvider doesn't support deletion.")