from tensorpc.constants import PROTOBUF_VERSION

if PROTOBUF_VERSION < (3, 20):
    from .protos_legacy import arraybuf_pb2, remote_object_pb2, rpc_message_pb2, remote_object_pb2_grpc, wsdef_pb2
else:
    from .protos import arraybuf_pb2, remote_object_pb2, rpc_message_pb2, remote_object_pb2_grpc, wsdef_pb2 # type: ignore[no-redef]
