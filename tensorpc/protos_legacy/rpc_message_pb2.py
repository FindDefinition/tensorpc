# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: rpc_message.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import arraybuf_pb2 as arraybuf__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x11rpc_message.proto\x12\x0ftensorpc.protos\x1a\x0e\x61rraybuf.proto\".\n\x0bSimpleReply\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\t\x12\x11\n\texception\x18\x02 \x01(\t\"\x85\x01\n\x11RemoteCallRequest\x12(\n\x06\x61rrays\x18\x01 \x03(\x0b\x32\x18.tensorpc.protos.ndarray\x12\x10\n\x08\x62lock_id\x18\x02 \x01(\x03\x12\r\n\x05\x66lags\x18\x03 \x01(\x03\x12\x10\n\x08\x63\x61llback\x18\x04 \x01(\t\x12\x13\n\x0bservice_key\x18\x05 \x01(\t\"\x85\x01\n\x15RemoteJsonCallRequest\x12(\n\x06\x61rrays\x18\x01 \x03(\x0b\x32\x18.tensorpc.protos.ndarray\x12\r\n\x05\x66lags\x18\x02 \x01(\x03\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\t\x12\x13\n\x0bservice_key\x18\x04 \x01(\t\x12\x10\n\x08\x63\x61llback\x18\x05 \x01(\t\"o\n\x0fRemoteCallReply\x12(\n\x06\x61rrays\x18\x01 \x03(\x0b\x32\x18.tensorpc.protos.ndarray\x12\x10\n\x08\x62lock_id\x18\x02 \x01(\x03\x12\r\n\x05\x66lags\x18\x03 \x01(\x03\x12\x11\n\texception\x18\x04 \x01(\t\"o\n\x13RemoteJsonCallReply\x12(\n\x06\x61rrays\x18\x01 \x03(\x0b\x32\x18.tensorpc.protos.ndarray\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\t\x12\r\n\x05\x66lags\x18\x03 \x01(\x03\x12\x11\n\texception\x18\x04 \x01(\t\"\xd9\x01\n\x10RemoteCallStream\x12\x11\n\tnum_chunk\x18\x01 \x01(\x05\x12\x10\n\x08\x63hunk_id\x18\x02 \x01(\x05\x12\x10\n\x08num_args\x18\x03 \x01(\x05\x12\x0e\n\x06\x61rg_id\x18\x04 \x01(\x05\x12\r\n\x05\x66lags\x18\x05 \x01(\x03\x12%\n\x05\x64type\x18\x06 \x01(\x0b\x32\x16.tensorpc.protos.dtype\x12\x14\n\x0c\x63hunked_data\x18\x07 \x01(\x0c\x12\x10\n\x08\x66unc_key\x18\x08 \x01(\t\x12\r\n\x05shape\x18\t \x03(\x03\x12\x11\n\texception\x18\n \x01(\t\"%\n\x12HealthCheckRequest\x12\x0f\n\x07service\x18\x01 \x01(\t\" \n\x10HealthCheckReply\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\t\"\x1c\n\x0cHelloRequest\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\t\"\x1a\n\nHelloReply\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\t*\xa4\x01\n\x0c\x45ncodeMethod\x12\x0b\n\x07Unknown\x10\x00\x12\x08\n\x04Json\x10\x01\x12\n\n\x06Pickle\x10\x02\x12\x0f\n\x0bMessagePack\x10\x03\x12\r\n\tJsonArray\x10\x10\x12\x0f\n\x0bPickleArray\x10 \x12\x14\n\x10MessagePackArray\x10\x30\x12\x0f\n\x0bNoArrayMask\x10\x0f\x12\x0e\n\tArrayMask\x10\xf0\x01\x12\t\n\x04Mask\x10\xff\x01\x62\x06proto3')

_ENCODEMETHOD = DESCRIPTOR.enum_types_by_name['EncodeMethod']
EncodeMethod = enum_type_wrapper.EnumTypeWrapper(_ENCODEMETHOD)
Unknown = 0
Json = 1
Pickle = 2
MessagePack = 3
JsonArray = 16
PickleArray = 32
MessagePackArray = 48
NoArrayMask = 15
ArrayMask = 240
Mask = 255


_SIMPLEREPLY = DESCRIPTOR.message_types_by_name['SimpleReply']
_REMOTECALLREQUEST = DESCRIPTOR.message_types_by_name['RemoteCallRequest']
_REMOTEJSONCALLREQUEST = DESCRIPTOR.message_types_by_name['RemoteJsonCallRequest']
_REMOTECALLREPLY = DESCRIPTOR.message_types_by_name['RemoteCallReply']
_REMOTEJSONCALLREPLY = DESCRIPTOR.message_types_by_name['RemoteJsonCallReply']
_REMOTECALLSTREAM = DESCRIPTOR.message_types_by_name['RemoteCallStream']
_HEALTHCHECKREQUEST = DESCRIPTOR.message_types_by_name['HealthCheckRequest']
_HEALTHCHECKREPLY = DESCRIPTOR.message_types_by_name['HealthCheckReply']
_HELLOREQUEST = DESCRIPTOR.message_types_by_name['HelloRequest']
_HELLOREPLY = DESCRIPTOR.message_types_by_name['HelloReply']
SimpleReply = _reflection.GeneratedProtocolMessageType('SimpleReply', (_message.Message,), {
  'DESCRIPTOR' : _SIMPLEREPLY,
  '__module__' : 'rpc_message_pb2'
  # @@protoc_insertion_point(class_scope:tensorpc.protos.SimpleReply)
  })
_sym_db.RegisterMessage(SimpleReply)

RemoteCallRequest = _reflection.GeneratedProtocolMessageType('RemoteCallRequest', (_message.Message,), {
  'DESCRIPTOR' : _REMOTECALLREQUEST,
  '__module__' : 'rpc_message_pb2'
  # @@protoc_insertion_point(class_scope:tensorpc.protos.RemoteCallRequest)
  })
_sym_db.RegisterMessage(RemoteCallRequest)

RemoteJsonCallRequest = _reflection.GeneratedProtocolMessageType('RemoteJsonCallRequest', (_message.Message,), {
  'DESCRIPTOR' : _REMOTEJSONCALLREQUEST,
  '__module__' : 'rpc_message_pb2'
  # @@protoc_insertion_point(class_scope:tensorpc.protos.RemoteJsonCallRequest)
  })
_sym_db.RegisterMessage(RemoteJsonCallRequest)

RemoteCallReply = _reflection.GeneratedProtocolMessageType('RemoteCallReply', (_message.Message,), {
  'DESCRIPTOR' : _REMOTECALLREPLY,
  '__module__' : 'rpc_message_pb2'
  # @@protoc_insertion_point(class_scope:tensorpc.protos.RemoteCallReply)
  })
_sym_db.RegisterMessage(RemoteCallReply)

RemoteJsonCallReply = _reflection.GeneratedProtocolMessageType('RemoteJsonCallReply', (_message.Message,), {
  'DESCRIPTOR' : _REMOTEJSONCALLREPLY,
  '__module__' : 'rpc_message_pb2'
  # @@protoc_insertion_point(class_scope:tensorpc.protos.RemoteJsonCallReply)
  })
_sym_db.RegisterMessage(RemoteJsonCallReply)

RemoteCallStream = _reflection.GeneratedProtocolMessageType('RemoteCallStream', (_message.Message,), {
  'DESCRIPTOR' : _REMOTECALLSTREAM,
  '__module__' : 'rpc_message_pb2'
  # @@protoc_insertion_point(class_scope:tensorpc.protos.RemoteCallStream)
  })
_sym_db.RegisterMessage(RemoteCallStream)

HealthCheckRequest = _reflection.GeneratedProtocolMessageType('HealthCheckRequest', (_message.Message,), {
  'DESCRIPTOR' : _HEALTHCHECKREQUEST,
  '__module__' : 'rpc_message_pb2'
  # @@protoc_insertion_point(class_scope:tensorpc.protos.HealthCheckRequest)
  })
_sym_db.RegisterMessage(HealthCheckRequest)

HealthCheckReply = _reflection.GeneratedProtocolMessageType('HealthCheckReply', (_message.Message,), {
  'DESCRIPTOR' : _HEALTHCHECKREPLY,
  '__module__' : 'rpc_message_pb2'
  # @@protoc_insertion_point(class_scope:tensorpc.protos.HealthCheckReply)
  })
_sym_db.RegisterMessage(HealthCheckReply)

HelloRequest = _reflection.GeneratedProtocolMessageType('HelloRequest', (_message.Message,), {
  'DESCRIPTOR' : _HELLOREQUEST,
  '__module__' : 'rpc_message_pb2'
  # @@protoc_insertion_point(class_scope:tensorpc.protos.HelloRequest)
  })
_sym_db.RegisterMessage(HelloRequest)

HelloReply = _reflection.GeneratedProtocolMessageType('HelloReply', (_message.Message,), {
  'DESCRIPTOR' : _HELLOREPLY,
  '__module__' : 'rpc_message_pb2'
  # @@protoc_insertion_point(class_scope:tensorpc.protos.HelloReply)
  })
_sym_db.RegisterMessage(HelloReply)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _ENCODEMETHOD._serialized_start=952
  _ENCODEMETHOD._serialized_end=1116
  _SIMPLEREPLY._serialized_start=54
  _SIMPLEREPLY._serialized_end=100
  _REMOTECALLREQUEST._serialized_start=103
  _REMOTECALLREQUEST._serialized_end=236
  _REMOTEJSONCALLREQUEST._serialized_start=239
  _REMOTEJSONCALLREQUEST._serialized_end=372
  _REMOTECALLREPLY._serialized_start=374
  _REMOTECALLREPLY._serialized_end=485
  _REMOTEJSONCALLREPLY._serialized_start=487
  _REMOTEJSONCALLREPLY._serialized_end=598
  _REMOTECALLSTREAM._serialized_start=601
  _REMOTECALLSTREAM._serialized_end=818
  _HEALTHCHECKREQUEST._serialized_start=820
  _HEALTHCHECKREQUEST._serialized_end=857
  _HEALTHCHECKREPLY._serialized_start=859
  _HEALTHCHECKREPLY._serialized_end=891
  _HELLOREQUEST._serialized_start=893
  _HELLOREQUEST._serialized_end=921
  _HELLOREPLY._serialized_start=923
  _HELLOREPLY._serialized_end=949
# @@protoc_insertion_point(module_scope)
