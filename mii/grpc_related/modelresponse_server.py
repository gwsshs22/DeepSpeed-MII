# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import asyncio
import queue
import sys
import threading
from concurrent import futures
from typing import Dict, Any
from enum import Enum
from dataclasses import dataclass

import grpc
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2

from mii.backend.client import create_channel
from mii.constants import (
    GenerationFinishReason,
    GRPC_MAX_MSG_SIZE,
    TERMINATE_METHOD,
    LB_MAX_WORKER_THREADS,
    SERVER_SHUTDOWN_TIMEOUT,
    STREAM_RESPONSE_QUEUE_TIMEOUT,
)
from mii.grpc_related.proto import modelresponse_pb2_grpc
from mii.grpc_related.task_methods import TASK_METHODS_DICT, TaskMethods


class ServiceBase(modelresponse_pb2_grpc.ModelResponseServicer):
    """
    Base class to provide common features of an inference server
    """
    def __init__(self):
        self._stop_event = threading.Event()

    def Terminate(self, request, context):
        self._stop_event.set()
        return google_dot_protobuf_dot_empty__pb2.Empty()

    def get_stop_event(self):
        return self._stop_event


class ModelResponse(ServiceBase):
    """
    Implementation class of an MII inference server
    """
    def __init__(self, async_pipeline=None):
        super().__init__()
        self.inference_pipeline = async_pipeline
        self.method_name_to_task = {m.method: t for t, m in TASK_METHODS_DICT.items()}
        self.lock = threading.Lock()

    def _get_task_methods(self, method_name: str) -> Dict[str, TaskMethods]:
        if method_name not in self.method_name_to_task:
            raise ValueError(f"unknown method: {method_name}")

        task = self.method_name_to_task[method_name]
        if task not in TASK_METHODS_DICT:
            raise ValueError(f"unknown task: {task}")

        task_methods = TASK_METHODS_DICT[task]
        return task_methods

    def EmptyRun(self, request, context):
        self.inference_pipeline.empty_run()
        return google_dot_protobuf_dot_empty__pb2.Empty()

    def GeneratorReply(self, request, context):
        task_methods = self._get_task_methods("GeneratorReply")

        prompts, kwargs = task_methods.unpack_request_from_proto(request)
        uids_running, uids_complete_order, responses = [], [], []

        # Put requests for all prompts into the pipeline
        for p in prompts:
            request_kwargs = kwargs.copy()
            uid = self.inference_pipeline.put_request(p, request_kwargs)
            uids_running.append(uid)

        # Get responses from the pipeline as they are ready, flush finished uids
        # so new requests can be processed
        while uids_running:
            uid, response = self.inference_pipeline.get_response()
            # TODO: Ugly hack for multi-threading. Will be fixed when we refactor these methods
            if uid == -1:
                uid = uids_running[0]
            responses.append(response)
            self.inference_pipeline.flush_uid(uid)
            uids_complete_order.append(uids_running.index(uid))
            uids_running.remove(uid)

        # Sort responses in the order of prompts
        responses = [
            r for idx,
            r in sorted(zip(uids_complete_order,
                            responses),
                        key=lambda pair: pair[0])
        ]

        return task_methods.pack_response_to_proto(responses)

    def GeneratorReplyStream(self, request, context):
        task_methods = self._get_task_methods("GeneratorReply")

        prompts, kwargs = task_methods.unpack_request_from_proto(request)
        uid = self.inference_pipeline.put_request(prompts[0], kwargs)

        while True:
            response_uid, r = self.inference_pipeline.get_response()
            assert uid == response_uid, "uid mismatch"
            done = r.finish_reason != GenerationFinishReason.NONE
            response = task_methods.pack_response_to_proto([r])
            yield response
            if done:
                break

        self.inference_pipeline.flush_uid(uid)

def _get_grpc_method_name(method):
    return method.split("/")[-1]

class SchedulerMessageType(Enum):
    SCHEDULE_UNARY = 1
    SCHEDULE_STREAM = 2
    TRY_SCHEDULE = 3
    TERMINATE = 4

@dataclass
class SchedulerMessage:
    msg_type: SchedulerMessageType
    request_proto: Any = None
    future: Any = None
    result_queue: Any = None

@dataclass
class ModelReplicaState:
    num_running_reqs: int = 0

class Scheduler:

    def __init__(self, model_config):
        self._terminated = False
        self._msg_queue = asyncio.Queue()
        self._pending_reqs = []
        self._asyncio_loop = asyncio.get_event_loop()
        self._counter = 0

        self._try_schedule_msg = SchedulerMessage(SchedulerMessageType.TRY_SCHEDULE)
        self._expert_parallel = model_config.expert_parallel

        self._pstubs = [
            ParallelStubInvoker(replica.hostname,
                                replica.tensor_parallel_ports)
            for replica in model_config.replica_configs
        ]
        self._num_replicas = len(self._pstubs)
        self._model_replica_states = [ModelReplicaState() for _ in range(self._num_replicas)]
        self._issued_orders = [0 for _ in range(self._num_replicas)]
        self._current_orders = [0 for _ in range(self._num_replicas)]
        self._order_locks = [asyncio.Lock() for _ in range(self._num_replicas)]

        # Start the asyncio loop in a separate thread
        def run_asyncio_loop(loop):
            asyncio.set_event_loop(loop)
            asyncio.run_coroutine_threadsafe(
                self._schedule_loop(),
                loop)

            try:
                loop.run_forever()
            finally:
                loop.close()

        self._loop_thread = threading.Thread(target=run_asyncio_loop, args=(self._asyncio_loop, ))
        self._loop_thread.start()

    def _issue_order(self, replica_id):
        ret = self._issued_orders[replica_id]
        self._issued_orders[replica_id] += 1
        return ret

    async def _schedule_loop(self):
        while not self._terminated:
            msgs = []
            if self._msg_queue.empty():
                msg = await self._msg_queue.get()
                self._msg_queue.task_done()
                msgs.append(msg)
            while True:
                try:
                    msg = self._msg_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                else:
                    msgs.append(msg)
                    self._msg_queue.task_done()

            for msg in msgs:
                if msg.msg_type == SchedulerMessageType.TERMINATE:
                    return
                elif msg.msg_type == SchedulerMessageType.SCHEDULE_STREAM or \
                    msg.msg_type == SchedulerMessageType.SCHEDULE_UNARY:
                    self._pending_reqs.append(msg)

            # Do scheduling based pending requests and model replica states.
            # Currently, we just do simple round-robin.
            # TODO(gwkim): Implement more efficient scheduling policies.
            for pending_req in self._pending_reqs:
                replica_id = self._counter % self._num_replicas
                self._counter += 1
                self._model_replica_states[replica_id].num_running_reqs += 1
                if pending_req.msg_type == SchedulerMessageType.SCHEDULE_UNARY:
                    asyncio.create_task(self._run_unary(replica_id, pending_req.request_proto, pending_req.future, self._issue_order(replica_id)))
                elif pending_req.msg_type == SchedulerMessageType.SCHEDULE_STREAM:
                    asyncio.create_task(self._run_stream(replica_id, pending_req.request_proto, pending_req.result_queue, self._issue_order(replica_id)))
                else:
                    raise ValueError(f"Unexpected pending request: {pending_req}")

            self._pending_reqs.clear()

            # Do empty runs for expert parallelism.
            if self._expert_parallel:
                has_running_replica = False
                idle_replica_ids = []
                for replica_id, model_replica_state in enumerate(self._model_replica_states):
                    if model_replica_state.num_running_reqs > 0:
                        has_running_replica = True
                    else:
                        idle_replica_ids.append(replica_id)

                if has_running_replica:
                    for idle_replica_id in idle_replica_ids:
                        self._model_replica_states[idle_replica_id].num_running_reqs += 1
                        asyncio.create_task(self._run_empty_run(idle_replica_id, self._issue_order(idle_replica_id)))

    def _mark_try_scheduling(self):
        if self._msg_queue.empty():
            self._msg_queue.put_nowait(self._try_schedule_msg)

    async def _run_empty_run(self, replica_id, order):
        async with self._order_locks[replica_id]:
            while self._current_orders[replica_id] != order:
                await asyncio.sleep(0)
            coroutine = self._pstubs[replica_id].EmptyRun()
            self._current_orders[replica_id] += 1
        await coroutine
        self._model_replica_states[replica_id].num_running_reqs -= 1
        self._mark_try_scheduling()

    async def _run_unary(self, replica_id, request_proto, future, order):
        async with self._order_locks[replica_id]:
            while self._current_orders[replica_id] != order:
                await asyncio.sleep(0)
            coroutine = self._pstubs[replica_id].GeneratorReply(request_proto)
            self._current_orders[replica_id] += 1

        res = await coroutine
        if future:
            future.set_result(res)
        self._model_replica_states[replica_id].num_running_reqs -= 1
        self._mark_try_scheduling()

    async def _run_stream(self, replica_id, request_proto, result_queue, order):
        async with self._order_locks[replica_id]:
            while self._current_orders[replica_id] != order:
                await asyncio.sleep(0)
            coroutine = self._pstubs[replica_id].GeneratorReplyStream(request_proto, result_queue)
            self._current_orders[replica_id] += 1

        await coroutine
        self._model_replica_states[replica_id].num_running_reqs -= 1
        self._mark_try_scheduling()

    def _enqueue(self, msg):
        self._msg_queue.put_nowait(msg)

    def enqueue_unary(self, request_proto, context):
        async def _enqueue_unary_coroutine():
            future = asyncio.Future(loop=self._asyncio_loop)
            self._enqueue(SchedulerMessage(
                msg_type=SchedulerMessageType.SCHEDULE_UNARY,
                request_proto=request_proto,
                future=future
            ))
            return await future

        return asyncio.run_coroutine_threadsafe(_enqueue_unary_coroutine(), self._asyncio_loop).result()

    def enqueue_stream(self, request_proto, context):
        result_queue = queue.Queue()

        async def _enqueue_stream_coroutine():
            self._enqueue(SchedulerMessage(
                msg_type=SchedulerMessageType.SCHEDULE_STREAM,
                request_proto=request_proto,
                result_queue=result_queue
            ))

        asyncio.run_coroutine_threadsafe(_enqueue_stream_coroutine(), self._asyncio_loop)

        while True:
            try:
                response_proto = result_queue.get(
                    timeout=STREAM_RESPONSE_QUEUE_TIMEOUT)
                yield response_proto
                if response_proto.response[0].finish_reason != str(
                        GenerationFinishReason.NONE.value):
                    break
            except queue.Empty:
                print(
                    f"Haven't received a streaming response in {STREAM_RESPONSE_QUEUE_TIMEOUT} second(s)"
                )
                break

    async def _async_terminate(self):
        for pstub in self._pstubs:
            ret = await pstub.Terminate()
        return ret

    def terminate(self):
        if self._terminated:
            return

        self._terminated = True
        self._enqueue(SchedulerMessage(SchedulerMessageType.TERMINATE))
        ret = asyncio.run_coroutine_threadsafe(self._async_terminate(), self._asyncio_loop).result()
        self._asyncio_loop.call_soon_threadsafe(self._asyncio_loop.stop)
        self._loop_thread.join()
        return ret

class ParallelStubInvoker:
    """
    Invokes a gRPC method on multiple endpoints in parallel.
    This class aims to call gRPC methods without conversions between proto and python object.
    TensorParallelClient can be used for invocation with the conversions.
    """
    def __init__(self, host, ports):
        # Assumption: target services are all on the same host
        self.stubs = []
        for port in ports:
            channel = create_channel(host, port)
            stub = modelresponse_pb2_grpc.ModelResponseStub(channel)
            self.stubs.append(stub)

    async def EmptyRun(self):
        return await self.stubs[0].EmptyRun(google_dot_protobuf_dot_empty__pb2.Empty())

    async def GeneratorReply(self, proto_request):
        return await self.stubs[0].GeneratorReply(proto_request)

    async def GeneratorReplyStream(self, proto_request, result_queue):
        response = self.stubs[0].GeneratorReplyStream(proto_request)
        async for r in response:
            result_queue.put(r)

    async def Terminate(self):
        responses = []
        for stub in self.stubs:
            responses.append(stub.Terminate(google_dot_protobuf_dot_empty__pb2.Empty()))

        for response in responses:
            ret = await response
        return ret

class LoadBalancingInterceptor(grpc.ServerInterceptor):
    def __init__(self, model_config):
        super().__init__()
        self._scheduler = Scheduler(model_config)

    def intercept_service(self, continuation, handler_call_details):
        next_handler = continuation(handler_call_details)

        method_name = _get_grpc_method_name(handler_call_details.method)
        if method_name == TERMINATE_METHOD:
            self._scheduler.terminate()
            return grpc.unary_unary_rpc_method_handler(
                lambda r, c: next_handler.unary_unary(r, c),
                request_deserializer=next_handler.request_deserializer,
                response_serializer=next_handler.response_serializer)
        
        if next_handler.unary_unary is not None:
            return grpc.unary_unary_rpc_method_handler(
                lambda r, c: self._scheduler.enqueue_unary(r, c),
                request_deserializer=next_handler.request_deserializer,
                response_serializer=next_handler.response_serializer)
        else:
            return grpc.unary_stream_rpc_method_handler(
                lambda r, c: self._scheduler.enqueue_stream(r, c),
                request_deserializer=next_handler.request_deserializer,
                response_serializer=next_handler.response_serializer)

def _do_serve(service_impl, port, interceptors=[]):
    stop_event = service_impl.get_stop_event()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=LB_MAX_WORKER_THREADS),
                         interceptors=interceptors,
                         options=[("grpc.max_send_message_length",
                                   GRPC_MAX_MSG_SIZE),
                                  ("grpc.max_receive_message_length",
                                   GRPC_MAX_MSG_SIZE)])
    modelresponse_pb2_grpc.add_ModelResponseServicer_to_server(service_impl, server)
    server.add_insecure_port(f"[::]:{port}")
    print(f"About to start server at {port}")
    server.start()
    print(f"Started server at {port}")
    stop_event.wait()
    server.stop(SERVER_SHUTDOWN_TIMEOUT)


def serve_inference(async_pipeline, port):
    async_pipeline.start()
    _do_serve(ModelResponse(async_pipeline=async_pipeline), port)
    async_pipeline.shutdown()


def serve_load_balancing(model_config, lb_port):
    _do_serve(ServiceBase(), lb_port, [LoadBalancingInterceptor(model_config)])


if __name__ == "__main__":
    import logging
    logging.basicConfig()
    serve_inference(None, sys.argv[1])
