from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar, Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Header(_message.Message):
    __slots__ = ["chunk_index", "data", "dynamic_key", "rpc_id", "service_id", "service_key"]
    CHUNK_INDEX_FIELD_NUMBER: ClassVar[int]
    DATA_FIELD_NUMBER: ClassVar[int]
    DYNAMIC_KEY_FIELD_NUMBER: ClassVar[int]
    RPC_ID_FIELD_NUMBER: ClassVar[int]
    SERVICE_ID_FIELD_NUMBER: ClassVar[int]
    SERVICE_KEY_FIELD_NUMBER: ClassVar[int]
    chunk_index: int
    data: str
    dynamic_key: str
    rpc_id: int
    service_id: int
    service_key: str
    def __init__(self, service_id: Optional[int] = ..., chunk_index: Optional[int] = ..., rpc_id: Optional[int] = ..., data: Optional[str] = ..., service_key: Optional[str] = ..., dynamic_key: Optional[str] = ...) -> None: ...