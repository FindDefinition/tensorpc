import arraybuf_pb2 as _arraybuf_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

ArrayMask: EncodeMethod
DESCRIPTOR: _descriptor.FileDescriptor
Json: EncodeMethod
JsonArray: EncodeMethod
Mask: EncodeMethod
MessagePack: EncodeMethod
MessagePackArray: EncodeMethod
NoArrayMask: EncodeMethod
Pickle: EncodeMethod
PickleArray: EncodeMethod
Unknown: EncodeMethod

class HealthCheckReply(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: str
    def __init__(self, data: _Optional[str] = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ["service"]
    SERVICE_FIELD_NUMBER: _ClassVar[int]
    service: str
    def __init__(self, service: _Optional[str] = ...) -> None: ...

class HelloReply(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: str
    def __init__(self, data: _Optional[str] = ...) -> None: ...

class HelloRequest(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: str
    def __init__(self, data: _Optional[str] = ...) -> None: ...

class RemoteCallReply(_message.Message):
    __slots__ = ["arrays", "block_id", "exception", "flags"]
    ARRAYS_FIELD_NUMBER: _ClassVar[int]
    BLOCK_ID_FIELD_NUMBER: _ClassVar[int]
    EXCEPTION_FIELD_NUMBER: _ClassVar[int]
    FLAGS_FIELD_NUMBER: _ClassVar[int]
    arrays: _containers.RepeatedCompositeFieldContainer[_arraybuf_pb2.ndarray]
    block_id: int
    exception: str
    flags: int
    def __init__(self, arrays: _Optional[_Iterable[_Union[_arraybuf_pb2.ndarray, _Mapping]]] = ..., block_id: _Optional[int] = ..., flags: _Optional[int] = ..., exception: _Optional[str] = ...) -> None: ...

class RemoteCallRequest(_message.Message):
    __slots__ = ["arrays", "block_id", "callback", "flags", "service_key"]
    ARRAYS_FIELD_NUMBER: _ClassVar[int]
    BLOCK_ID_FIELD_NUMBER: _ClassVar[int]
    CALLBACK_FIELD_NUMBER: _ClassVar[int]
    FLAGS_FIELD_NUMBER: _ClassVar[int]
    SERVICE_KEY_FIELD_NUMBER: _ClassVar[int]
    arrays: _containers.RepeatedCompositeFieldContainer[_arraybuf_pb2.ndarray]
    block_id: int
    callback: str
    flags: int
    service_key: str
    def __init__(self, arrays: _Optional[_Iterable[_Union[_arraybuf_pb2.ndarray, _Mapping]]] = ..., block_id: _Optional[int] = ..., flags: _Optional[int] = ..., callback: _Optional[str] = ..., service_key: _Optional[str] = ...) -> None: ...

class RemoteCallStream(_message.Message):
    __slots__ = ["arg_id", "chunk_id", "chunked_data", "dtype", "exception", "flags", "func_key", "num_args", "num_chunk", "shape"]
    ARG_ID_FIELD_NUMBER: _ClassVar[int]
    CHUNKED_DATA_FIELD_NUMBER: _ClassVar[int]
    CHUNK_ID_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    EXCEPTION_FIELD_NUMBER: _ClassVar[int]
    FLAGS_FIELD_NUMBER: _ClassVar[int]
    FUNC_KEY_FIELD_NUMBER: _ClassVar[int]
    NUM_ARGS_FIELD_NUMBER: _ClassVar[int]
    NUM_CHUNK_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    arg_id: int
    chunk_id: int
    chunked_data: bytes
    dtype: _arraybuf_pb2.dtype
    exception: str
    flags: int
    func_key: str
    num_args: int
    num_chunk: int
    shape: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, num_chunk: _Optional[int] = ..., chunk_id: _Optional[int] = ..., num_args: _Optional[int] = ..., arg_id: _Optional[int] = ..., flags: _Optional[int] = ..., dtype: _Optional[_Union[_arraybuf_pb2.dtype, _Mapping]] = ..., chunked_data: _Optional[bytes] = ..., func_key: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., exception: _Optional[str] = ...) -> None: ...

class RemoteJsonCallReply(_message.Message):
    __slots__ = ["arrays", "data", "exception", "flags"]
    ARRAYS_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    EXCEPTION_FIELD_NUMBER: _ClassVar[int]
    FLAGS_FIELD_NUMBER: _ClassVar[int]
    arrays: _containers.RepeatedCompositeFieldContainer[_arraybuf_pb2.ndarray]
    data: str
    exception: str
    flags: int
    def __init__(self, arrays: _Optional[_Iterable[_Union[_arraybuf_pb2.ndarray, _Mapping]]] = ..., data: _Optional[str] = ..., flags: _Optional[int] = ..., exception: _Optional[str] = ...) -> None: ...

class RemoteJsonCallRequest(_message.Message):
    __slots__ = ["arrays", "callback", "data", "flags", "service_key"]
    ARRAYS_FIELD_NUMBER: _ClassVar[int]
    CALLBACK_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    FLAGS_FIELD_NUMBER: _ClassVar[int]
    SERVICE_KEY_FIELD_NUMBER: _ClassVar[int]
    arrays: _containers.RepeatedCompositeFieldContainer[_arraybuf_pb2.ndarray]
    callback: str
    data: str
    flags: int
    service_key: str
    def __init__(self, arrays: _Optional[_Iterable[_Union[_arraybuf_pb2.ndarray, _Mapping]]] = ..., flags: _Optional[int] = ..., data: _Optional[str] = ..., service_key: _Optional[str] = ..., callback: _Optional[str] = ...) -> None: ...

class SimpleReply(_message.Message):
    __slots__ = ["data", "exception"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    EXCEPTION_FIELD_NUMBER: _ClassVar[int]
    data: str
    exception: str
    def __init__(self, data: _Optional[str] = ..., exception: _Optional[str] = ...) -> None: ...

class EncodeMethod(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
