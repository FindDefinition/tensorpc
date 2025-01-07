from tensorpc.core.client import RemoteObject
from tensorpc.autossh.serv_names import serv_names
class TaskWrapperClient:
    def __init__(self, robj: RemoteObject):
        self.robj = robj

    def log_event(self, event: str):
        return self.robj.remote_call(serv_names.TASK_WRAPPER_LOG_EVENT, event)