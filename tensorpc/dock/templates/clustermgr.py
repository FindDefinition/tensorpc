from tensorpc.apps.cm.constants import TENSORPC_CM_NODEMGR_DEFAULT_PORT
from tensorpc.apps.cm.coretypes import ResourceInfo
from tensorpc.apps.cm.manager import ClusterSpec, FixedClusterProvider, NodeSpec
from tensorpc.autossh.core import SSHConnDesc
from tensorpc.dock import mui, mark_create_layout


from tensorpc.apps.cm.components.cluster_panel import ClusterManagePanel


NAIVE_CLUSTER = [
    ClusterSpec(
        "local-cluster-id",
        "Local Cluster",
        [
            NodeSpec(
                "node-0",
                [],
                ResourceInfo(4, 16, 8, gpu_type="RTX4090"),
                # local url inside cluster.
                # port default to 
                local_url_with_port=f"10.0.0.1:{TENSORPC_CM_NODEMGR_DEFAULT_PORT}",
            )
        ],
        # if nodes in the cluster are not directly accessible, 
        # you can specify ssh jumps to access them.
        # manager will create ssh tunnels before RPC calls.
        # WARNING: make sure you can access ssh jumps by ssh cmd,
        # otherwise you will get long hang.
        ssh_jumps=[
            SSHConnDesc(
                "8.8.4.4:2222", # ssh host and port
                username="root",
                password="", # if you already have ssh key setup, you can leave password empty.
            )
        ]
    )
]

class App:

    @mark_create_layout
    def my_layout(self):
        # we use a fixed cluster here, you can also use your 
        # cluster apis to implement a dynamic provider.
        self._provider = FixedClusterProvider(NAIVE_CLUSTER)
        self._panel = ClusterManagePanel({
            "local": self._provider,
        })
        self._root_box = mui.VBox([
            self._panel.prop(flex=1)
        ]).prop(width="100%", height="100%", overflow="hidden")
        return self._root_box

