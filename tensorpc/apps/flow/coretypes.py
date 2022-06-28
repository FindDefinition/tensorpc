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
