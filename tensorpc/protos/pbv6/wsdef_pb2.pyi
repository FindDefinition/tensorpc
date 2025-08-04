from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Header(_message.Message):
    __slots__ = ("service_id", "chunk_index", "rpc_id", "data", "service_key", "dynamic_key")
    SERVICE_ID_FIELD_NUMBER: _ClassVar[int]
    CHUNK_INDEX_FIELD_NUMBER: _ClassVar[int]
    RPC_ID_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    SERVICE_KEY_FIELD_NUMBER: _ClassVar[int]
    DYNAMIC_KEY_FIELD_NUMBER: _ClassVar[int]
    service_id: int
    chunk_index: int
    rpc_id: int
    data: str
    service_key: str
    dynamic_key: str
    def __init__(self, service_id: _Optional[int] = ..., chunk_index: _Optional[int] = ..., rpc_id: _Optional[int] = ..., data: _Optional[str] = ..., service_key: _Optional[str] = ..., dynamic_key: _Optional[str] = ...) -> None: ...
