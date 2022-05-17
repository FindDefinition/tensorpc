# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: remote_object.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import arraybuf_pb2 as arraybuf__pb2
from . import rpc_message_pb2 as rpc__message__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x13remote_object.proto\x12\x0ftensorpc.protos\x1a\x0e\x61rraybuf.proto\x1a\x11rpc_message.proto2\xc1\t\n\x0cRemoteObject\x12T\n\nRemoteCall\x12\".tensorpc.protos.RemoteCallRequest\x1a .tensorpc.protos.RemoteCallReply\"\x00\x12[\n\x0fRemoteGenerator\x12\".tensorpc.protos.RemoteCallRequest\x1a .tensorpc.protos.RemoteCallReply\"\x00\x30\x01\x12`\n\x0eRemoteJsonCall\x12&.tensorpc.protos.RemoteJsonCallRequest\x1a$.tensorpc.protos.RemoteJsonCallReply\"\x00\x12g\n\x13RemoteJsonGenerator\x12&.tensorpc.protos.RemoteJsonCallRequest\x1a$.tensorpc.protos.RemoteJsonCallReply\"\x00\x30\x01\x12^\n\x10RemoteStreamCall\x12\".tensorpc.protos.RemoteCallRequest\x1a .tensorpc.protos.RemoteCallReply\"\x00(\x01\x30\x01\x12Z\n\x0eServerShutdown\x12#.tensorpc.protos.HealthCheckRequest\x1a!.tensorpc.protos.HealthCheckReply\"\x00\x12W\n\x0bHealthCheck\x12#.tensorpc.protos.HealthCheckRequest\x1a!.tensorpc.protos.HealthCheckReply\"\x00\x12U\n\x0fQueryServerMeta\x12\".tensorpc.protos.RemoteCallRequest\x1a\x1c.tensorpc.protos.SimpleReply\"\x00\x12V\n\x10QueryServiceMeta\x12\".tensorpc.protos.RemoteCallRequest\x1a\x1c.tensorpc.protos.SimpleReply\"\x00\x12_\n\x11\x43hunkedRemoteCall\x12!.tensorpc.protos.RemoteCallStream\x1a!.tensorpc.protos.RemoteCallStream\"\x00(\x01\x30\x01\x12\x62\n\x16\x43lientStreamRemoteCall\x12\".tensorpc.protos.RemoteCallRequest\x1a .tensorpc.protos.RemoteCallReply\"\x00(\x01\x12`\n\x12\x42iStreamRemoteCall\x12\".tensorpc.protos.RemoteCallRequest\x1a .tensorpc.protos.RemoteCallReply\"\x00(\x01\x30\x01\x12H\n\x08SayHello\x12\x1d.tensorpc.protos.HelloRequest\x1a\x1b.tensorpc.protos.HelloReply\"\x00\x62\x06proto3')



_REMOTEOBJECT = DESCRIPTOR.services_by_name['RemoteObject']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _REMOTEOBJECT._serialized_start=76
  _REMOTEOBJECT._serialized_end=1293
# @@protoc_insertion_point(module_scope)
