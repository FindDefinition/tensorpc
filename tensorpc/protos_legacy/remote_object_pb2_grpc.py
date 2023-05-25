# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from . import rpc_message_pb2 as rpc__message__pb2


class RemoteObjectStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.RemoteCall = channel.unary_unary(
        '/tensorpc.protos.RemoteObject/RemoteCall',
        request_serializer=rpc__message__pb2.RemoteCallRequest.SerializeToString,
        response_deserializer=rpc__message__pb2.RemoteCallReply.FromString,
        )
    self.RemoteGenerator = channel.unary_stream(
        '/tensorpc.protos.RemoteObject/RemoteGenerator',
        request_serializer=rpc__message__pb2.RemoteCallRequest.SerializeToString,
        response_deserializer=rpc__message__pb2.RemoteCallReply.FromString,
        )
    self.RemoteJsonCall = channel.unary_unary(
        '/tensorpc.protos.RemoteObject/RemoteJsonCall',
        request_serializer=rpc__message__pb2.RemoteJsonCallRequest.SerializeToString,
        response_deserializer=rpc__message__pb2.RemoteJsonCallReply.FromString,
        )
    self.RemoteJsonGenerator = channel.unary_stream(
        '/tensorpc.protos.RemoteObject/RemoteJsonGenerator',
        request_serializer=rpc__message__pb2.RemoteJsonCallRequest.SerializeToString,
        response_deserializer=rpc__message__pb2.RemoteJsonCallReply.FromString,
        )
    self.RemoteStreamCall = channel.stream_stream(
        '/tensorpc.protos.RemoteObject/RemoteStreamCall',
        request_serializer=rpc__message__pb2.RemoteCallRequest.SerializeToString,
        response_deserializer=rpc__message__pb2.RemoteCallReply.FromString,
        )
    self.ServerShutdown = channel.unary_unary(
        '/tensorpc.protos.RemoteObject/ServerShutdown',
        request_serializer=rpc__message__pb2.HealthCheckRequest.SerializeToString,
        response_deserializer=rpc__message__pb2.HealthCheckReply.FromString,
        )
    self.HealthCheck = channel.unary_unary(
        '/tensorpc.protos.RemoteObject/HealthCheck',
        request_serializer=rpc__message__pb2.HealthCheckRequest.SerializeToString,
        response_deserializer=rpc__message__pb2.HealthCheckReply.FromString,
        )
    self.QueryServerMeta = channel.unary_unary(
        '/tensorpc.protos.RemoteObject/QueryServerMeta',
        request_serializer=rpc__message__pb2.RemoteCallRequest.SerializeToString,
        response_deserializer=rpc__message__pb2.SimpleReply.FromString,
        )
    self.QueryServiceMeta = channel.unary_unary(
        '/tensorpc.protos.RemoteObject/QueryServiceMeta',
        request_serializer=rpc__message__pb2.RemoteCallRequest.SerializeToString,
        response_deserializer=rpc__message__pb2.SimpleReply.FromString,
        )
    self.ChunkedRemoteCall = channel.stream_stream(
        '/tensorpc.protos.RemoteObject/ChunkedRemoteCall',
        request_serializer=rpc__message__pb2.RemoteCallStream.SerializeToString,
        response_deserializer=rpc__message__pb2.RemoteCallStream.FromString,
        )
    self.ClientStreamRemoteCall = channel.stream_unary(
        '/tensorpc.protos.RemoteObject/ClientStreamRemoteCall',
        request_serializer=rpc__message__pb2.RemoteCallRequest.SerializeToString,
        response_deserializer=rpc__message__pb2.RemoteCallReply.FromString,
        )
    self.BiStreamRemoteCall = channel.stream_stream(
        '/tensorpc.protos.RemoteObject/BiStreamRemoteCall',
        request_serializer=rpc__message__pb2.RemoteCallRequest.SerializeToString,
        response_deserializer=rpc__message__pb2.RemoteCallReply.FromString,
        )
    self.SayHello = channel.unary_unary(
        '/tensorpc.protos.RemoteObject/SayHello',
        request_serializer=rpc__message__pb2.HelloRequest.SerializeToString,
        response_deserializer=rpc__message__pb2.HelloReply.FromString,
        )


class RemoteObjectServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def RemoteCall(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def RemoteGenerator(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def RemoteJsonCall(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def RemoteJsonGenerator(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def RemoteStreamCall(self, request_iterator, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def ServerShutdown(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def HealthCheck(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def QueryServerMeta(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def QueryServiceMeta(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def ChunkedRemoteCall(self, request_iterator, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def ClientStreamRemoteCall(self, request_iterator, context):
    """first RemoteCallRequest will save paramsters of generator call, following are generator data.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def BiStreamRemoteCall(self, request_iterator, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def SayHello(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_RemoteObjectServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'RemoteCall': grpc.unary_unary_rpc_method_handler(
          servicer.RemoteCall,
          request_deserializer=rpc__message__pb2.RemoteCallRequest.FromString,
          response_serializer=rpc__message__pb2.RemoteCallReply.SerializeToString,
      ),
      'RemoteGenerator': grpc.unary_stream_rpc_method_handler(
          servicer.RemoteGenerator,
          request_deserializer=rpc__message__pb2.RemoteCallRequest.FromString,
          response_serializer=rpc__message__pb2.RemoteCallReply.SerializeToString,
      ),
      'RemoteJsonCall': grpc.unary_unary_rpc_method_handler(
          servicer.RemoteJsonCall,
          request_deserializer=rpc__message__pb2.RemoteJsonCallRequest.FromString,
          response_serializer=rpc__message__pb2.RemoteJsonCallReply.SerializeToString,
      ),
      'RemoteJsonGenerator': grpc.unary_stream_rpc_method_handler(
          servicer.RemoteJsonGenerator,
          request_deserializer=rpc__message__pb2.RemoteJsonCallRequest.FromString,
          response_serializer=rpc__message__pb2.RemoteJsonCallReply.SerializeToString,
      ),
      'RemoteStreamCall': grpc.stream_stream_rpc_method_handler(
          servicer.RemoteStreamCall,
          request_deserializer=rpc__message__pb2.RemoteCallRequest.FromString,
          response_serializer=rpc__message__pb2.RemoteCallReply.SerializeToString,
      ),
      'ServerShutdown': grpc.unary_unary_rpc_method_handler(
          servicer.ServerShutdown,
          request_deserializer=rpc__message__pb2.HealthCheckRequest.FromString,
          response_serializer=rpc__message__pb2.HealthCheckReply.SerializeToString,
      ),
      'HealthCheck': grpc.unary_unary_rpc_method_handler(
          servicer.HealthCheck,
          request_deserializer=rpc__message__pb2.HealthCheckRequest.FromString,
          response_serializer=rpc__message__pb2.HealthCheckReply.SerializeToString,
      ),
      'QueryServerMeta': grpc.unary_unary_rpc_method_handler(
          servicer.QueryServerMeta,
          request_deserializer=rpc__message__pb2.RemoteCallRequest.FromString,
          response_serializer=rpc__message__pb2.SimpleReply.SerializeToString,
      ),
      'QueryServiceMeta': grpc.unary_unary_rpc_method_handler(
          servicer.QueryServiceMeta,
          request_deserializer=rpc__message__pb2.RemoteCallRequest.FromString,
          response_serializer=rpc__message__pb2.SimpleReply.SerializeToString,
      ),
      'ChunkedRemoteCall': grpc.stream_stream_rpc_method_handler(
          servicer.ChunkedRemoteCall,
          request_deserializer=rpc__message__pb2.RemoteCallStream.FromString,
          response_serializer=rpc__message__pb2.RemoteCallStream.SerializeToString,
      ),
      'ClientStreamRemoteCall': grpc.stream_unary_rpc_method_handler(
          servicer.ClientStreamRemoteCall,
          request_deserializer=rpc__message__pb2.RemoteCallRequest.FromString,
          response_serializer=rpc__message__pb2.RemoteCallReply.SerializeToString,
      ),
      'BiStreamRemoteCall': grpc.stream_stream_rpc_method_handler(
          servicer.BiStreamRemoteCall,
          request_deserializer=rpc__message__pb2.RemoteCallRequest.FromString,
          response_serializer=rpc__message__pb2.RemoteCallReply.SerializeToString,
      ),
      'SayHello': grpc.unary_unary_rpc_method_handler(
          servicer.SayHello,
          request_deserializer=rpc__message__pb2.HelloRequest.FromString,
          response_serializer=rpc__message__pb2.HelloReply.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'tensorpc.protos.RemoteObject', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))