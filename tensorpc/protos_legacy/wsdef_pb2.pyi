from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Header(_message.Message):
    __slots__ = ["chunk_index", "data", "dynamic_key", "rpc_id", "service_id", "service_key"]
    CHUNK_INDEX_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    DYNAMIC_KEY_FIELD_NUMBER: _ClassVar[int]
    RPC_ID_FIELD_NUMBER: _ClassVar[int]
    SERVICE_ID_FIELD_NUMBER: _ClassVar[int]
    SERVICE_KEY_FIELD_NUMBER: _ClassVar[int]
    chunk_index: int
    data: str
    dynamic_key: str
    rpc_id: int
    service_id: int
    service_key: str
    def __init__(self, service_id: _Optional[int] = ..., chunk_index: _Optional[int] = ..., rpc_id: _Optional[int] = ..., data: _Optional[str] = ..., service_key: _Optional[str] = ..., dynamic_key: _Optional[str] = ...) -> None: ...
