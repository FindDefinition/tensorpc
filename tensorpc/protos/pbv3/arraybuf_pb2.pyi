from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class arrayjson(_message.Message):
    __slots__ = ["arrays", "data"]
    ARRAYS_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    arrays: _containers.RepeatedCompositeFieldContainer[ndarray]
    data: str
    def __init__(self, arrays: _Optional[_Iterable[_Union[ndarray, _Mapping]]] = ..., data: _Optional[str] = ...) -> None: ...

class dtype(_message.Message):
    __slots__ = ["byte_order", "type"]
    class ByteOrder(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    BYTE_ORDER_FIELD_NUMBER: _ClassVar[int]
    Base64: dtype.DataType
    CustomBytes: dtype.DataType
    TYPE_FIELD_NUMBER: _ClassVar[int]
    bigEndian: dtype.ByteOrder
    bool_: dtype.DataType
    byte_order: dtype.ByteOrder
    float16: dtype.DataType
    float32: dtype.DataType
    float64: dtype.DataType
    int16: dtype.DataType
    int32: dtype.DataType
    int64: dtype.DataType
    int8: dtype.DataType
    littleEndian: dtype.ByteOrder
    na: dtype.ByteOrder
    native: dtype.ByteOrder
    type: dtype.DataType
    uint16: dtype.DataType
    uint32: dtype.DataType
    uint64: dtype.DataType
    uint8: dtype.DataType
    def __init__(self, type: _Optional[_Union[dtype.DataType, str]] = ..., byte_order: _Optional[_Union[dtype.ByteOrder, str]] = ...) -> None: ...

class ndarray(_message.Message):
    __slots__ = ["data", "dtype", "shape"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    dtype: dtype
    shape: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[_Union[dtype, _Mapping]] = ..., data: _Optional[bytes] = ...) -> None: ...
