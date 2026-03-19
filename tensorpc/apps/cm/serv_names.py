from tensorpc.utils import get_service_key_by_type


class AgentServNames:

    @property
    def RAFT_REQUEST_VOTE(self):
        from tensorpc.apps.cm.node_master import NodeMaster
        return get_service_key_by_type(NodeMaster, NodeMaster.request_vote.__name__)

    @property
    def RAFT_APPEND_ENTRIES(self):
        from tensorpc.apps.cm.node_master import NodeMaster
        return get_service_key_by_type(NodeMaster, NodeMaster.append_entries.__name__)
    
    @property
    def RAFT_INSTALL_SNAPSHOT(self):
        from tensorpc.apps.cm.node_master import NodeMaster
        return get_service_key_by_type(NodeMaster, NodeMaster.install_snapshot.__name__)

    @property
    def RAFT_WORKER_HEARTBEAT(self):
        from tensorpc.apps.cm.node_master import NodeMaster
        return get_service_key_by_type(NodeMaster, NodeMaster.worker_heartbeat.__name__)

    @property
    def GROUP_TREE_SCAN_GROUPS(self):
        from tensorpc.apps.cm.node_master import NodeMaster
        return get_service_key_by_type(NodeMaster, NodeMaster.tree_scan_groups.__name__)


    @property
    def GROUP_TREE_REMOVE_GROUP(self):
        from tensorpc.apps.cm.node_master import NodeMaster
        return get_service_key_by_type(NodeMaster, NodeMaster.tree_remove_group.__name__)

    @property
    def GROUP_TREE_AWAKE_WORKER(self):
        from tensorpc.apps.cm.node_master import NodeMaster
        return get_service_key_by_type(NodeMaster, NodeMaster.tree_awake_worker.__name__)

    @property
    def GROUP_PROPOSE_CMD(self):
        from tensorpc.apps.cm.node_master import NodeMaster
        return get_service_key_by_type(NodeMaster, NodeMaster.propose.__name__)

    @property
    def GROUP_QUERY_COARSE_STATUS(self):
        from tensorpc.apps.cm.node_master import NodeMaster
        return get_service_key_by_type(NodeMaster, NodeMaster.query_group_coarse_status.__name__)

    @property
    def GROUP_MASTER_CREATE_GROUP(self):
        from tensorpc.apps.cm.node_master import NodeMaster
        return get_service_key_by_type(NodeMaster, NodeMaster.master_create_group.__name__)
    
    @property
    def INTERNAL_GROUP_TREE_CREATE_GROUP(self):
        from tensorpc.apps.cm.node_master import NodeMaster
        return get_service_key_by_type(NodeMaster, NodeMaster.internal_tree_create_group.__name__)
    
    @property
    def INTERNAL_GROUP_CREATE_GROUP(self):
        from tensorpc.apps.cm.node_master import NodeMaster
        return get_service_key_by_type(NodeMaster, NodeMaster.internal_create_group.__name__)
    

master_serv_names = AgentServNames()