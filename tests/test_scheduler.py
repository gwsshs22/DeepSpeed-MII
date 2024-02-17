import asyncio
import os
import signal
import multiprocessing
import time
import atexit
import sys
import queue
import random
import threading
from collections import defaultdict
from typing import Any
from dataclasses import dataclass

from mii.grpc_related.task_methods import TASK_METHODS_DICT
from mii.grpc_related.modelresponse_server import serve_inference, serve_load_balancing
from mii.constants import GenerationFinishReason, TaskType
from mii.batching.data_classes import Response
from mii.backend.client import create_channel
from mii.grpc_related.proto import modelresponse_pb2, modelresponse_pb2_grpc

class MockAsyncPipeline:

    def __init__(self):
        self._uid_gen = 0
        self._queues = defaultdict(lambda: queue.Queue())

    def put_request(self, p, gen_args):
        uid = self._uid_gen
        self._uid_gen += 1
        if gen_args.get("stream", False):
            gen_args["remaining_gen"] = 3
        tid = threading.get_ident()
        self._queues[tid].put((uid, gen_args))
        return uid

    def get_response(self):
        tid = threading.get_ident()
        queue = self._queues[tid]
        uid, gen_args = queue.get()
        time.sleep(0.01 + random.random() * 0.01)

        remaining_gen = gen_args.get("remaining_gen", 0)
        if remaining_gen > 0:
            gen_args["remaining_gen"] = remaining_gen - 1
            queue.put((uid, gen_args))
            finish_reason = GenerationFinishReason.NONE
        else:
            finish_reason = GenerationFinishReason.LENGTH

        return uid, Response(generated_text="some text",
                            prompt_length=50,
                            generated_length=25,
                            finish_reason=finish_reason)

    def empty_run(self):
        return

    def flush_uid(self, uid):
        return

    def start(self):
        return
    
    def shutdown(self):
        return

def run_model_server(port):
    serve_inference(MockAsyncPipeline(), port)

def run_load_balancer(replica_ports, lb_port, expert_parallel=False):
    @dataclass
    class MockModelConfig:
        replica_configs: Any = None
        expert_parallel: bool = False
        task: Any = None

    @dataclass
    class MockReplicaConfig:
        hostname: str = "localhost"
        tensor_parallel_ports: Any = None

    replica_configs = [
        MockReplicaConfig(tensor_parallel_ports=ports_in_replica)
        for ports_in_replica in replica_ports
    ]
    serve_load_balancing(MockModelConfig(replica_configs, expert_parallel=expert_parallel), lb_port)


def child_process():
    print(f"Child process ID: {os.getpid()} started")
    try:
        while True:
            # Simulate work
            time.sleep(1)
            print(f"Child process ID: {os.getpid()} is working...")
    except KeyboardInterrupt:
        pass  # Child process can ignore keyboard interrupts

def cleanup_children(children):
    for child in children:
        child.kill()
        child.join()

if __name__ == "__main__":
    # Ensure the parent process and its children are in their own process group
    children = []    # Register cleanup function for normal exits
    parent_pid = os.getpid()
    atexit.register(cleanup_children, children)

    # Handle signals for abnormal termination
    def signal_handler(signum, frame):
        if os.getpid() == parent_pid:
            cleanup_children(children)
            sys.exit(signum)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    def start_child(target):
        p = multiprocessing.Process(target=target)
        p.start()
        children.append(p)

    replica_ports = [[50511], [50512], [50513], [50514]]
    lb_port = 50500
    for ports in replica_ports:
        for p in ports:
            start_child(lambda: run_model_server(p))

    start_child(lambda: run_load_balancer(replica_ports, lb_port, expert_parallel=True))

    asyncio_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(asyncio_loop)

    def run_asyncio_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    threading.Thread(target=run_asyncio_loop, args=(asyncio_loop, )).start()

    channel = create_channel("localhost", lb_port)
    stub = modelresponse_pb2_grpc.ModelResponseStub(channel)
    task_methods = TASK_METHODS_DICT[TaskType.TEXT_GENERATION]

    async def generate(req):
        return await stub.GeneratorReply(req)

    time.sleep(1)

    futs = []
    proto_request = task_methods.pack_request_to_proto(["prompts"], **{})

    # Warm-ups
    for _ in range(1):
        asyncio.run_coroutine_threadsafe(generate(proto_request), asyncio_loop).result()

    print("Warmup done")
    num_reqs = 32
    start_t = time.time()
    for _ in range(num_reqs):
        fut = asyncio.run_coroutine_threadsafe(generate(proto_request), asyncio_loop)
        futs.append(fut)

    for fut in futs:
        fut.result()
    print(f"Done in {time.time() - start_t}s")

    async def put_result(req, callback):
        response_stream = stub.GeneratorReplyStream(req)

        try:
            async for response in response_stream:
                callback(response)
        except StopAsyncIteration:
            pass

    def simple_callback(result):
        print(f"In streaming callback result={result}")

    stream_proto_request = task_methods.pack_request_to_proto(["prompts"], **{ "stream": True })
    futs = []

    num_stream_reqs = 1
    for _ in range(num_stream_reqs):
        fut = asyncio.run_coroutine_threadsafe(put_result(stream_proto_request, simple_callback), asyncio_loop)
        futs.append(fut)

    for fut in futs:
        fut.result()
    
    async def terminate():
        await stub.Terminate(modelresponse_pb2.google_dot_protobuf_dot_empty__pb2.Empty())

    fut = asyncio.run_coroutine_threadsafe(terminate(), asyncio_loop)
    fut.result()
    asyncio_loop.call_soon_threadsafe(asyncio_loop.stop)
