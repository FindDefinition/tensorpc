# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: arraybuf.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0e\x61rraybuf.proto\x12\x0f\x64istflow.protos\"\xe4\x02\n\x05\x64type\x12-\n\x04type\x18\x01 \x01(\x0e\x32\x1f.distflow.protos.dtype.DataType\x12\x34\n\nbyte_order\x18\x02 \x01(\x0e\x32 .distflow.protos.dtype.ByteOrder\"@\n\tByteOrder\x12\x10\n\x0clittleEndian\x10\x00\x12\r\n\tbigEndian\x10\x01\x12\n\n\x06native\x10\x02\x12\x06\n\x02na\x10\x03\"\xb3\x01\n\x08\x44\x61taType\x12\x0b\n\x07\x66loat64\x10\x00\x12\x0b\n\x07\x66loat32\x10\x01\x12\x0b\n\x07\x66loat16\x10\x02\x12\n\n\x06uint64\x10\x03\x12\n\n\x06uint32\x10\x04\x12\n\n\x06uint16\x10\x05\x12\t\n\x05uint8\x10\x06\x12\t\n\x05int64\x10\x07\x12\t\n\x05int32\x10\x08\x12\t\n\x05int16\x10\t\x12\x08\n\x04int8\x10\n\x12\t\n\x05\x62ool_\x10\x0b\x12\x0f\n\x0b\x43ustomBytes\x10\x0c\x12\n\n\x06\x42\x61se64\x10\r\"M\n\x07ndarray\x12\r\n\x05shape\x18\x01 \x03(\x03\x12%\n\x05\x64type\x18\x02 \x01(\x0b\x32\x16.distflow.protos.dtype\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\x0c\"C\n\tarrayjson\x12(\n\x06\x61rrays\x18\x01 \x03(\x0b\x32\x18.distflow.protos.ndarray\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\tb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'arraybuf_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _DTYPE._serialized_start=36
  _DTYPE._serialized_end=392
  _DTYPE_BYTEORDER._serialized_start=146
  _DTYPE_BYTEORDER._serialized_end=210
  _DTYPE_DATATYPE._serialized_start=213
  _DTYPE_DATATYPE._serialized_end=392
  _NDARRAY._serialized_start=394
  _NDARRAY._serialized_end=471
  _ARRAYJSON._serialized_start=473
  _ARRAYJSON._serialized_end=540
# @@protoc_insertion_point(module_scope)
