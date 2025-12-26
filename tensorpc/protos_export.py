from tensorpc.constants import PROTOBUF_VERSION

if PROTOBUF_VERSION < (3, 20):
    from .protos.pbv3 import arraybuf_pb2, remote_object_pb2, rpc_message_pb2, remote_object_pb2_grpc, wsdef_pb2
elif PROTOBUF_VERSION[0] < 6:
    from .protos.pbv5 import arraybuf_pb2, remote_object_pb2, rpc_message_pb2, remote_object_pb2_grpc, wsdef_pb2  # type: ignore[no-redef]
else:
    from .protos.pbv6 import arraybuf_pb2, remote_object_pb2, rpc_message_pb2, remote_object_pb2_grpc, wsdef_pb2