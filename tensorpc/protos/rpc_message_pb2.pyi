import arraybuf_pb2 as _arraybuf_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class EncodeMethod(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    Unknown: _ClassVar[EncodeMethod]
    Json: _ClassVar[EncodeMethod]
    Pickle: _ClassVar[EncodeMethod]
    MessagePack: _ClassVar[EncodeMethod]
    JsonArray: _ClassVar[EncodeMethod]
    PickleArray: _ClassVar[EncodeMethod]
    MessagePackArray: _ClassVar[EncodeMethod]
    NoArrayMask: _ClassVar[EncodeMethod]
    ArrayMask: _ClassVar[EncodeMethod]
    Mask: _ClassVar[EncodeMethod]
Unknown: EncodeMethod
Json: EncodeMethod
Pickle: EncodeMethod
MessagePack: EncodeMethod
JsonArray: EncodeMethod
PickleArray: EncodeMethod
MessagePackArray: EncodeMethod
NoArrayMask: EncodeMethod
ArrayMask: EncodeMethod
Mask: EncodeMethod

class SimpleReply(_message.Message):
    __slots__ = ("data", "exception")
    DATA_FIELD_NUMBER: _ClassVar[int]
    EXCEPTION_FIELD_NUMBER: _ClassVar[int]
    data: str
    exception: str
    def __init__(self, data: _Optional[str] = ..., exception: _Optional[str] = ...) -> None: ...

class RemoteCallRequest(_message.Message):
    __slots__ = ("arrays", "block_id", "flags", "callback", "service_key")
    ARRAYS_FIELD_NUMBER: _ClassVar[int]
    BLOCK_ID_FIELD_NUMBER: _ClassVar[int]
    FLAGS_FIELD_NUMBER: _ClassVar[int]
    CALLBACK_FIELD_NUMBER: _ClassVar[int]
    SERVICE_KEY_FIELD_NUMBER: _ClassVar[int]
    arrays: _containers.RepeatedCompositeFieldContainer[_arraybuf_pb2.ndarray]
    block_id: int
    flags: int
    callback: str
    service_key: str
    def __init__(self, arrays: _Optional[_Iterable[_Union[_arraybuf_pb2.ndarray, _Mapping]]] = ..., block_id: _Optional[int] = ..., flags: _Optional[int] = ..., callback: _Optional[str] = ..., service_key: _Optional[str] = ...) -> None: ...

class RemoteJsonCallRequest(_message.Message):
    __slots__ = ("arrays", "flags", "data", "service_key", "callback")
    ARRAYS_FIELD_NUMBER: _ClassVar[int]
    FLAGS_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    SERVICE_KEY_FIELD_NUMBER: _ClassVar[int]
    CALLBACK_FIELD_NUMBER: _ClassVar[int]
    arrays: _containers.RepeatedCompositeFieldContainer[_arraybuf_pb2.ndarray]
    flags: int
    data: str
    service_key: str
    callback: str
    def __init__(self, arrays: _Optional[_Iterable[_Union[_arraybuf_pb2.ndarray, _Mapping]]] = ..., flags: _Optional[int] = ..., data: _Optional[str] = ..., service_key: _Optional[str] = ..., callback: _Optional[str] = ...) -> None: ...

class RemoteCallReply(_message.Message):
    __slots__ = ("arrays", "block_id", "flags", "exception")
    ARRAYS_FIELD_NUMBER: _ClassVar[int]
    BLOCK_ID_FIELD_NUMBER: _ClassVar[int]
    FLAGS_FIELD_NUMBER: _ClassVar[int]
    EXCEPTION_FIELD_NUMBER: _ClassVar[int]
    arrays: _containers.RepeatedCompositeFieldContainer[_arraybuf_pb2.ndarray]
    block_id: int
    flags: int
    exception: str
    def __init__(self, arrays: _Optional[_Iterable[_Union[_arraybuf_pb2.ndarray, _Mapping]]] = ..., block_id: _Optional[int] = ..., flags: _Optional[int] = ..., exception: _Optional[str] = ...) -> None: ...

class RemoteJsonCallReply(_message.Message):
    __slots__ = ("arrays", "data", "flags", "exception")
    ARRAYS_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    FLAGS_FIELD_NUMBER: _ClassVar[int]
    EXCEPTION_FIELD_NUMBER: _ClassVar[int]
    arrays: _containers.RepeatedCompositeFieldContainer[_arraybuf_pb2.ndarray]
    data: str
    flags: int
    exception: str
    def __init__(self, arrays: _Optional[_Iterable[_Union[_arraybuf_pb2.ndarray, _Mapping]]] = ..., data: _Optional[str] = ..., flags: _Optional[int] = ..., exception: _Optional[str] = ...) -> None: ...

class RemoteCallStream(_message.Message):
    __slots__ = ("num_chunk", "chunk_id", "num_args", "arg_id", "flags", "dtype", "chunked_data", "func_key", "shape", "exception")
    NUM_CHUNK_FIELD_NUMBER: _ClassVar[int]
    CHUNK_ID_FIELD_NUMBER: _ClassVar[int]
    NUM_ARGS_FIELD_NUMBER: _ClassVar[int]
    ARG_ID_FIELD_NUMBER: _ClassVar[int]
    FLAGS_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    CHUNKED_DATA_FIELD_NUMBER: _ClassVar[int]
    FUNC_KEY_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    EXCEPTION_FIELD_NUMBER: _ClassVar[int]
    num_chunk: int
    chunk_id: int
    num_args: int
    arg_id: int
    flags: int
    dtype: _arraybuf_pb2.dtype
    chunked_data: bytes
    func_key: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    exception: str
    def __init__(self, num_chunk: _Optional[int] = ..., chunk_id: _Optional[int] = ..., num_args: _Optional[int] = ..., arg_id: _Optional[int] = ..., flags: _Optional[int] = ..., dtype: _Optional[_Union[_arraybuf_pb2.dtype, _Mapping]] = ..., chunked_data: _Optional[bytes] = ..., func_key: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., exception: _Optional[str] = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ("service",)
    SERVICE_FIELD_NUMBER: _ClassVar[int]
    service: str
    def __init__(self, service: _Optional[str] = ...) -> None: ...

class HealthCheckReply(_message.Message):
    __slots__ = ("data",)
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: str
    def __init__(self, data: _Optional[str] = ...) -> None: ...

class HelloRequest(_message.Message):
    __slots__ = ("data",)
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: str
    def __init__(self, data: _Optional[str] = ...) -> None: ...

class HelloReply(_message.Message):
    __slots__ = ("data",)
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: str
    def __init__(self, data: _Optional[str] = ...) -> None: ...
