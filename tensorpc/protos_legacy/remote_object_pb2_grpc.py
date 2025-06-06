# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import rpc_message_pb2 as rpc__message__pb2


class RemoteObjectStub(object):
    """Missing associated documentation comment in .proto file."""

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
        self.ChunkedBiStreamRemoteCall = channel.stream_stream(
                '/tensorpc.protos.RemoteObject/ChunkedBiStreamRemoteCall',
                request_serializer=rpc__message__pb2.RemoteCallStream.SerializeToString,
                response_deserializer=rpc__message__pb2.RemoteCallStream.FromString,
                )
        self.ChunkedRemoteGenerator = channel.stream_stream(
                '/tensorpc.protos.RemoteObject/ChunkedRemoteGenerator',
                request_serializer=rpc__message__pb2.RemoteCallStream.SerializeToString,
                response_deserializer=rpc__message__pb2.RemoteCallStream.FromString,
                )
        self.ChunkedClientStreamRemoteCall = channel.stream_stream(
                '/tensorpc.protos.RemoteObject/ChunkedClientStreamRemoteCall',
                request_serializer=rpc__message__pb2.RemoteCallStream.SerializeToString,
                response_deserializer=rpc__message__pb2.RemoteCallStream.FromString,
                )
        self.RelayStream = channel.stream_stream(
                '/tensorpc.protos.RemoteObject/RelayStream',
                request_serializer=rpc__message__pb2.RemoteCallStream.SerializeToString,
                response_deserializer=rpc__message__pb2.RemoteCallStream.FromString,
                )
        self.SayHello = channel.unary_unary(
                '/tensorpc.protos.RemoteObject/SayHello',
                request_serializer=rpc__message__pb2.HelloRequest.SerializeToString,
                response_deserializer=rpc__message__pb2.HelloReply.FromString,
                )


class RemoteObjectServicer(object):
    """Missing associated documentation comment in .proto file."""

    def RemoteCall(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RemoteGenerator(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RemoteJsonCall(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RemoteJsonGenerator(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RemoteStreamCall(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ServerShutdown(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def HealthCheck(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def QueryServerMeta(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def QueryServiceMeta(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ChunkedRemoteCall(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
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
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ChunkedBiStreamRemoteCall(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ChunkedRemoteGenerator(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ChunkedClientStreamRemoteCall(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RelayStream(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SayHello(self, request, context):
        """Missing associated documentation comment in .proto file."""
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
            'ChunkedBiStreamRemoteCall': grpc.stream_stream_rpc_method_handler(
                    servicer.ChunkedBiStreamRemoteCall,
                    request_deserializer=rpc__message__pb2.RemoteCallStream.FromString,
                    response_serializer=rpc__message__pb2.RemoteCallStream.SerializeToString,
            ),
            'ChunkedRemoteGenerator': grpc.stream_stream_rpc_method_handler(
                    servicer.ChunkedRemoteGenerator,
                    request_deserializer=rpc__message__pb2.RemoteCallStream.FromString,
                    response_serializer=rpc__message__pb2.RemoteCallStream.SerializeToString,
            ),
            'ChunkedClientStreamRemoteCall': grpc.stream_stream_rpc_method_handler(
                    servicer.ChunkedClientStreamRemoteCall,
                    request_deserializer=rpc__message__pb2.RemoteCallStream.FromString,
                    response_serializer=rpc__message__pb2.RemoteCallStream.SerializeToString,
            ),
            'RelayStream': grpc.stream_stream_rpc_method_handler(
                    servicer.RelayStream,
                    request_deserializer=rpc__message__pb2.RemoteCallStream.FromString,
                    response_serializer=rpc__message__pb2.RemoteCallStream.SerializeToString,
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


 # This class is part of an EXPERIMENTAL API.
class RemoteObject(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def RemoteCall(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorpc.protos.RemoteObject/RemoteCall',
            rpc__message__pb2.RemoteCallRequest.SerializeToString,
            rpc__message__pb2.RemoteCallReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RemoteGenerator(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/tensorpc.protos.RemoteObject/RemoteGenerator',
            rpc__message__pb2.RemoteCallRequest.SerializeToString,
            rpc__message__pb2.RemoteCallReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RemoteJsonCall(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorpc.protos.RemoteObject/RemoteJsonCall',
            rpc__message__pb2.RemoteJsonCallRequest.SerializeToString,
            rpc__message__pb2.RemoteJsonCallReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RemoteJsonGenerator(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/tensorpc.protos.RemoteObject/RemoteJsonGenerator',
            rpc__message__pb2.RemoteJsonCallRequest.SerializeToString,
            rpc__message__pb2.RemoteJsonCallReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RemoteStreamCall(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/tensorpc.protos.RemoteObject/RemoteStreamCall',
            rpc__message__pb2.RemoteCallRequest.SerializeToString,
            rpc__message__pb2.RemoteCallReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ServerShutdown(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorpc.protos.RemoteObject/ServerShutdown',
            rpc__message__pb2.HealthCheckRequest.SerializeToString,
            rpc__message__pb2.HealthCheckReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def HealthCheck(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorpc.protos.RemoteObject/HealthCheck',
            rpc__message__pb2.HealthCheckRequest.SerializeToString,
            rpc__message__pb2.HealthCheckReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def QueryServerMeta(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorpc.protos.RemoteObject/QueryServerMeta',
            rpc__message__pb2.RemoteCallRequest.SerializeToString,
            rpc__message__pb2.SimpleReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def QueryServiceMeta(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorpc.protos.RemoteObject/QueryServiceMeta',
            rpc__message__pb2.RemoteCallRequest.SerializeToString,
            rpc__message__pb2.SimpleReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ChunkedRemoteCall(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/tensorpc.protos.RemoteObject/ChunkedRemoteCall',
            rpc__message__pb2.RemoteCallStream.SerializeToString,
            rpc__message__pb2.RemoteCallStream.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ClientStreamRemoteCall(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_unary(request_iterator, target, '/tensorpc.protos.RemoteObject/ClientStreamRemoteCall',
            rpc__message__pb2.RemoteCallRequest.SerializeToString,
            rpc__message__pb2.RemoteCallReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def BiStreamRemoteCall(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/tensorpc.protos.RemoteObject/BiStreamRemoteCall',
            rpc__message__pb2.RemoteCallRequest.SerializeToString,
            rpc__message__pb2.RemoteCallReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ChunkedBiStreamRemoteCall(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/tensorpc.protos.RemoteObject/ChunkedBiStreamRemoteCall',
            rpc__message__pb2.RemoteCallStream.SerializeToString,
            rpc__message__pb2.RemoteCallStream.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ChunkedRemoteGenerator(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/tensorpc.protos.RemoteObject/ChunkedRemoteGenerator',
            rpc__message__pb2.RemoteCallStream.SerializeToString,
            rpc__message__pb2.RemoteCallStream.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ChunkedClientStreamRemoteCall(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/tensorpc.protos.RemoteObject/ChunkedClientStreamRemoteCall',
            rpc__message__pb2.RemoteCallStream.SerializeToString,
            rpc__message__pb2.RemoteCallStream.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RelayStream(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/tensorpc.protos.RemoteObject/RelayStream',
            rpc__message__pb2.RemoteCallStream.SerializeToString,
            rpc__message__pb2.RemoteCallStream.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SayHello(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tensorpc.protos.RemoteObject/SayHello',
            rpc__message__pb2.HelloRequest.SerializeToString,
            rpc__message__pb2.HelloReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
