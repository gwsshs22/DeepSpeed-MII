# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
import mii.grpc_related.proto.modelresponse_pb2 as modelresponse__pb2


class ModelResponseStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Terminate = channel.unary_unary(
                '/modelresponse.ModelResponse/Terminate',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )
        self.EmptyRun = channel.unary_unary(
                '/modelresponse.ModelResponse/EmptyRun',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )
        self.GeneratorReply = channel.unary_unary(
                '/modelresponse.ModelResponse/GeneratorReply',
                request_serializer=modelresponse__pb2.MultiStringRequest.SerializeToString,
                response_deserializer=modelresponse__pb2.MultiGenerationReply.FromString,
                )
        self.GeneratorReplyStream = channel.unary_stream(
                '/modelresponse.ModelResponse/GeneratorReplyStream',
                request_serializer=modelresponse__pb2.MultiStringRequest.SerializeToString,
                response_deserializer=modelresponse__pb2.MultiGenerationReply.FromString,
                )


class ModelResponseServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Terminate(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def EmptyRun(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GeneratorReply(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GeneratorReplyStream(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ModelResponseServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Terminate': grpc.unary_unary_rpc_method_handler(
                    servicer.Terminate,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
            'EmptyRun': grpc.unary_unary_rpc_method_handler(
                    servicer.EmptyRun,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
            'GeneratorReply': grpc.unary_unary_rpc_method_handler(
                    servicer.GeneratorReply,
                    request_deserializer=modelresponse__pb2.MultiStringRequest.FromString,
                    response_serializer=modelresponse__pb2.MultiGenerationReply.SerializeToString,
            ),
            'GeneratorReplyStream': grpc.unary_stream_rpc_method_handler(
                    servicer.GeneratorReplyStream,
                    request_deserializer=modelresponse__pb2.MultiStringRequest.FromString,
                    response_serializer=modelresponse__pb2.MultiGenerationReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'modelresponse.ModelResponse', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ModelResponse(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Terminate(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/modelresponse.ModelResponse/Terminate',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def EmptyRun(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/modelresponse.ModelResponse/EmptyRun',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GeneratorReply(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/modelresponse.ModelResponse/GeneratorReply',
            modelresponse__pb2.MultiStringRequest.SerializeToString,
            modelresponse__pb2.MultiGenerationReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GeneratorReplyStream(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/modelresponse.ModelResponse/GeneratorReplyStream',
            modelresponse__pb2.MultiStringRequest.SerializeToString,
            modelresponse__pb2.MultiGenerationReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
