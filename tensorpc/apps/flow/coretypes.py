import enum 

from typing import Any, Dict

from tensorpc.autossh.core import Event , event_from_dict

class RelayEventType(enum.Enum):
    UpdateNodeStatus = "UpdateNodeStatus"
    SSHEvent = "SSHEvent"

class RelayEvent:
    def __init__(self, type: RelayEventType):
        self.type = type

    def to_dict(self):
        return {
            "type": self.type.value,
        }

class RelaySSHEvent(RelayEvent):
    def __init__(self, ev: Event, uid: str):
        super().__init__(RelayEventType.SSHEvent)
        self.uid = uid
        self.event = ev

    def to_dict(self):
        res = super().to_dict()
        res["uid"] = self.uid 
        res["event"] = self.event.to_dict()
        return res  

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        assert RelayEventType.SSHEvent.value == data["type"]
        ev = event_from_dict(data["event"])
        return cls(ev, data["uid"])

class RelayUpdateNodeEvent(RelayEvent):
    def __init__(self, graph_id: str, node_id: str, content: Any):
        super().__init__(RelayEventType.UpdateNodeStatus)
        self.graph_id = graph_id
        self.node_id = node_id
        self.content = content

    def to_dict(self):
        res = super().to_dict()
        res["graph_id"] = self.graph_id 
        res["node_id"] = self.node_id 
        res["content"] = self.content 
        return res  

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        assert RelayEventType.UpdateNodeStatus.value == data["type"]
        return cls(data["graph_id"], data["node_id"], data["content"])

def relay_event_from_dict(data: Dict[str, Any]):
    if data["type"] == RelayEventType.SSHEvent.value:
        return RelaySSHEvent.from_dict(data)
    elif data["type"] == RelayEventType.UpdateNodeStatus.value:
        return RelayUpdateNodeEvent.from_dict(data)
    raise NotImplementedError

class UserEventType(enum.Enum):
    """user event: event come from user code instead of
    ssh.
    for example:
    1. node call api to update content
        of a command node.
    2. node submit a message
    3. node submit a new status (currently
        only come from master server)
    """
    Status = "Status"
    Content = "Content"
    Message = "Message"

class UserEvent:
    def __init__(self, type: UserEventType):
        self.type = type

    def to_dict(self):
        return {
            "type": self.type.value,
        }

class UserStatusEvent(UserEvent):
    ALL_STATUS = set(["idle", "running", "error", "success"])
    def __init__(self, status: str):
        super().__init__(UserEventType.Status)
        assert status in self.ALL_STATUS
        self.status = status

    def to_dict(self):
        res = super().to_dict()
        res["status"] = self.status 
        return res  

class UserContentEvent(UserEvent):
    def __init__(self, content: Any):
        super().__init__(UserEventType.Content)
        self.content = content

    def to_dict(self):
        res = super().to_dict()
        res["content"] = self.content 
        return res  
