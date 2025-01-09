from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class dtype(_message.Message):
    __slots__ = ("type", "byte_order")
    class ByteOrder(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        littleEndian: _ClassVar[dtype.ByteOrder]
        bigEndian: _ClassVar[dtype.ByteOrder]
        native: _ClassVar[dtype.ByteOrder]
        na: _ClassVar[dtype.ByteOrder]
    littleEndian: dtype.ByteOrder
    bigEndian: dtype.ByteOrder
    native: dtype.ByteOrder
    na: dtype.ByteOrder
    class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        float64: _ClassVar[dtype.DataType]
        float32: _ClassVar[dtype.DataType]
        float16: _ClassVar[dtype.DataType]
        uint64: _ClassVar[dtype.DataType]
        uint32: _ClassVar[dtype.DataType]
        uint16: _ClassVar[dtype.DataType]
        uint8: _ClassVar[dtype.DataType]
        int64: _ClassVar[dtype.DataType]
        int32: _ClassVar[dtype.DataType]
        int16: _ClassVar[dtype.DataType]
        int8: _ClassVar[dtype.DataType]
        bool_: _ClassVar[dtype.DataType]
        CustomBytes: _ClassVar[dtype.DataType]
        Base64: _ClassVar[dtype.DataType]
    float64: dtype.DataType
    float32: dtype.DataType
    float16: dtype.DataType
    uint64: dtype.DataType
    uint32: dtype.DataType
    uint16: dtype.DataType
    uint8: dtype.DataType
    int64: dtype.DataType
    int32: dtype.DataType
    int16: dtype.DataType
    int8: dtype.DataType
    bool_: dtype.DataType
    CustomBytes: dtype.DataType
    Base64: dtype.DataType
    TYPE_FIELD_NUMBER: _ClassVar[int]
    BYTE_ORDER_FIELD_NUMBER: _ClassVar[int]
    type: dtype.DataType
    byte_order: dtype.ByteOrder
    def __init__(self, type: _Optional[_Union[dtype.DataType, str]] = ..., byte_order: _Optional[_Union[dtype.ByteOrder, str]] = ...) -> None: ...

class ndarray(_message.Message):
    __slots__ = ("shape", "dtype", "data")
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: dtype
    data: bytes
    def __init__(self, shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[_Union[dtype, _Mapping]] = ..., data: _Optional[bytes] = ...) -> None: ...

class arrayjson(_message.Message):
    __slots__ = ("arrays", "data")
    ARRAYS_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    arrays: _containers.RepeatedCompositeFieldContainer[ndarray]
    data: str
    def __init__(self, arrays: _Optional[_Iterable[_Union[ndarray, _Mapping]]] = ..., data: _Optional[str] = ...) -> None: ...
