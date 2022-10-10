from tensorpc.constants import PROTOBUF_VERSION

if PROTOBUF_VERSION >= (4, 0):
    from .protos import remote_object_pb2, rpc_message_pb2, remote_object_pb2_grpc, wsdef_pb2
else:
    from .protos import remote_object_pb2, rpc_message_pb2, remote_object_pb2_grpc, wsdef_pb2
