from tensorpc.utils import get_service_key_by_type

class _ServiceNames:
    @property 
    def DISTSSH_LIST_DEBUG_SERVERS(self):
        from tensorpc.apps.distssh.services.core import FaultToleranceSSHServer
        return get_service_key_by_type(FaultToleranceSSHServer, FaultToleranceSSHServer.list_all_debug_servers.__name__)

serv_names = _ServiceNames()