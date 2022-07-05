import enum

from typing import Any, Dict, List

from tensorpc.autossh.core import Event, event_from_dict

def get_uid(graph_id: str, node_id: str):
    return f"{graph_id}@{node_id}"

class MessageItemType(enum.Enum):
    Text = 0
    Image = 1


class MessageLevel(enum.Enum):
    Info = 0
    Warning = 1
    Error = 2


class MessageItem:
    def __init__(self, type: MessageItemType, data: Any) -> None:
        self.type = type
        self.data = data

    def to_dict(self):
        return {
            "type": self.type.value,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(MessageItemType(data["type"]), data["data"])

class Message:
    def __init__(self, uid: str, level: MessageLevel, timestamp: int,
                 graph_id: str, node_id: str, title: str,
                 items: List[MessageItem]) -> None:
        self.uid = uid
        self.title = title
        self.items = items
        self.timestamp = timestamp
        self.graph_id = graph_id
        self.node_id = node_id
        self.level = level

    def get_node_uid(self):
        return get_uid(self.graph_id, self.node_id)

    def __hash__(self) -> int:
        return hash(self.uid)

    def to_dict(self, with_detail: bool = False):
        res = {
            "uid": self.uid,
            "level": self.level.value,
            "nodeId": self.node_id,
            "graphId": self.graph_id,
            "ts": self.timestamp,
            "title": self.title,
            "items": []
        }
        if with_detail:
            res["items"] = [n.to_dict() for n in self.items]
        return res

    def to_dict_with_detail(self):
        return self.to_dict(True)


    @classmethod
    def from_dict(cls, data):
        return cls(data["uid"], MessageLevel(data["level"]), data["ts"], data["graphId"],
            data["nodeId"], data["title"], [MessageItem.from_dict(it) for it in data["items"]])

class MessageEventType(enum.Enum):
    Update = "Update"
    Replace = "Replace"

class MessageEvent:
    def __init__(self, type: MessageEventType, rawmsgs: List[Any]) -> None:
        self.type = type 
        self.rawmsgs = rawmsgs

    def to_dict(self):
        return {
            "type": self.type.value,
            "msgs": self.rawmsgs,
        }
        

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


