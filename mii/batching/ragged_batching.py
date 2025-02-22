# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import copy
import gc
import os
import queue
import random
import threading
import time
from collections import deque, defaultdict
from functools import cached_property
from typing import Dict, Tuple, List, Any, Union, DefaultDict

import torch
import ujson
import zmq
from deepspeed.accelerator import get_accelerator
from deepspeed.utils.timer import SynchronizedWallClockTimer

from mii.batching.constants import TOP_K_NAME, TOP_P_NAME, TEMP_NAME, SAMPLER_NAME, STOP_NAME
from mii.batching.data_classes import Response, Request, RequestBatch
from mii.batching.generation.logit_processors import TopPLogitProcessor, TopKLogitProcessor, TemperatureLogitProcessor
from mii.batching.generation.samplers import LogitsSampler, GreedySampler
from mii.batching.generation.stop_criterion import EosGenerationStopCriterion, TokenStopCriterion
from mii.batching.postprocess import (
    run_batch_logit_processing,
    run_batch_sampler,
    run_batch_stop_criterion,
)
from mii.batching.utils import sync_debug, profiler
from mii.config import GenerateParamsConfig
from mii.constants import GenerationFinishReason, ZMQ_RECV_TIMEOUT
from mii.logging import logger


class RaggedBatchBase:
    def __init__(self, inference_engine, tokenizer, model_config):
        self.inference_engine = inference_engine
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.model_config = model_config
        self.zmq_port = model_config.zmq_port_number
        if model_config.max_length is not None:
            self.max_length = model_config.max_length
        else:
            self.max_length = inference_engine._policy._checkpoint_engine.model_config.max_seq_length
        self.sync_debug = model_config.sync_debug
        self.profile_model_time = model_config.profile_model_time
        self.run_condition = threading.Condition()
        self.is_shutdown = False

        self.request_queue: queue.Queue = queue.Queue()
        self.result_queues: Dict[int, queue.Queue] = {}
        self.scheduled_requests: RequestBatch = RequestBatch()
        self.buffer = deque()
        self.scheduled_length = 0
        self.scheduled_seq_num = 0
        self.scheduled_req_blocks = 0
        self.empty_run_scheduled = False
        self.empty_run_queue = queue.Queue()

        # TODO: we will need to prune self._post_processors for long running deployments
        self._post_processors = {}
        self.logit_processor = run_batch_logit_processing
        self.sampler = run_batch_sampler
        self.stop_criterion = run_batch_stop_criterion

        self._timers: SynchronizedWallClockTimer = SynchronizedWallClockTimer()
        self._profiled_times: DefaultDict[str, List[int]] = defaultdict(list)
        self._iters: int = 0
        self._num_generated_tokens: int = 0

        self._zmq_context = zmq.Context()
        torch.cuda.synchronize()
        if self.is_first_rank_in_replica:
            self.socket = self._zmq_context.socket(zmq.PUB)
            self.socket.bind(f"tcp://*:{self.zmq_port}")
            time.sleep(1)  # Give the subscriber a change to connect
        else:
            self.socket = self._zmq_context.socket(zmq.SUB)
            self.socket.connect(f"tcp://localhost:{self.zmq_port}")
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self.socket.setsockopt(zmq.RCVTIMEO, ZMQ_RECV_TIMEOUT)

    @cached_property
    def local_rank(self) -> int:
        return get_accelerator().current_device()

    @property
    def is_first_rank_in_replica(self) -> bool:
        return self.local_rank % self.model_config.tensor_parallel == 0
    
    def should_run(self) -> bool:
        return len(self.buffer) > 0 or not self.request_queue.empty() or self.empty_run_scheduled or self.is_shutdown

    def wait_for_run(self) -> None:
        with self.run_condition:
            self.run_condition.wait_for(self.should_run)

    def _shutdown(self) -> None:
        with self.run_condition:
            self.is_shutdown = True
            self.run_condition.notify_all()

    def empty_run(self) -> None:
        assert not self.empty_run_scheduled
        with self.run_condition:
            self.empty_run_scheduled = True
            self.run_condition.notify_all()
        self.empty_run_queue.get()
    
    def put_request_and_notify(self, req) -> None:
        with self.run_condition:
            self.request_queue.put(req)
            self.run_condition.notify_all()

    @profiler
    def generate(self) -> None:
        if self.is_first_rank_in_replica:
            # 1. Schedule requests
            self._reset_scheduler_bookkeeping()
            self.wait_for_run()
            self.schedule_requests()

        # 2. Get a batch of requests, broadcast to all ranks
        scheduled_requests, empty_run = self._bcast_requests()

        if self.is_shutdown:
            return

        if empty_run:
            self.inference_engine.empty_run()
            if self.is_first_rank_in_replica:
                self.empty_run_scheduled = False
                self.empty_run_queue.put(None)
            return

        # 3. Flush for uids that are finished generating
        self.flush(scheduled_requests.requests_to_flush.uids)

        # 4. Put new tokens into inference engine
        if scheduled_requests.requests_to_run:
            next_token_logits = self.put(
                scheduled_requests.requests_to_run.uids,
                scheduled_requests.requests_to_run.tokens,
            )

        # short circuit if not rank 0, only rank 0 does scheduling and postprocessing of logits
        if not self.is_first_rank_in_replica:
            return

        # 5. Launch logit processing and token generation
        running_requests = scheduled_requests.requests_to_run
        running_requests.update_seq_length()
        if running_requests:
            next_tokens, done_tokens = self._process_logits(
                next_token_logits, running_requests
            )
            running_requests.next_tokens = next_tokens
            running_requests.done_tokens = done_tokens

        # 6. Accumulate generated tokens, check completion, and generate output
        for r in running_requests.last_in_prompt:
            r.accumulate_generated_token()
            self._num_generated_tokens += 1
            if r.stop_generation or r.stream:
                self._generate_output(r)
            if not r.stop_generation:
                r.set_next_as_input()
                self.buffer.append(r)

        if self.profile_model_time:
            self._print_profiled_times()

    def _print_profiled_times(self) -> None:
        self._iters += 1
        if not (self._iters % 100 == 0):
            return
        for event, times in self._profiled_times.items():
            mean_time = sum(times) / len(times)
            log_msg = f"{event}: {mean_time}"
            if event == "generate":
                log_msg += f" ({self._num_generated_tokens / sum(times)} tokens/ms)"
            logger.info(log_msg)
        self._profiled_times.clear()
        self._num_generated_tokens = 0

    @sync_debug
    def _bcast_requests(self, force=False) -> RequestBatch:
        if self.is_first_rank_in_replica:
            if self.empty_run_scheduled:
                assert not self.scheduled_requests.requests_to_run
                self.socket.send_string("ER")
                return self.scheduled_requests, True

            if not self.scheduled_requests and not force:
                if self.is_shutdown:
                    self.socket.send_string("ST")
                return self.scheduled_requests, False

            # Rank 0 gets batch of requests and broadcasts to other ranks
            data_dicts = self.scheduled_requests.to_msg_dicts()
            json_data = ujson.dumps(data_dicts)
            self.socket.send_string(json_data)
        else:
            try:
                json_data = self.socket.recv_string()
                if json_data == "ER":
                    self.scheduled_requests = RequestBatch()
                    return self.scheduled_requests, True
                elif json_data == "ST":
                    self._shutdown()
                    return self.scheduled_requests, False

                data_dicts = ujson.loads(json_data)
                self.scheduled_requests = RequestBatch.from_msg_dicts(data_dicts)
            except zmq.Again:
                self.scheduled_requests = RequestBatch()

        return self.scheduled_requests, False

    def _reset_scheduler_bookkeeping(self) -> None:
        self.scheduled_requests = RequestBatch()
        self.scheduled_length = 0
        self.scheduled_seq_num = 0
        self.scheduled_req_blocks = 0

    @sync_debug
    def _process_logits(
            self,
            next_token_logits: torch.Tensor,
            running_requests: RequestBatch) -> Tuple[torch.Tensor,
                                                     torch.Tensor]:
        next_token_logits = next_token_logits[:, :self.vocab_size]
        next_token_logits = self.logit_processor(next_token_logits,
                                                 running_requests,
                                                 self._post_processors)
        next_tokens = self.sampler(next_token_logits,
                                   running_requests,
                                   self._post_processors)
        done_tokens = self.stop_criterion(next_tokens,
                                          running_requests,
                                          self._post_processors)
        next_tokens = next_tokens.to(torch.device("cpu"), non_blocking=False)
        done_tokens = done_tokens.to(torch.device("cpu"), non_blocking=False)
        return next_tokens, done_tokens

    @sync_debug
    def _generate_output(self, r: Request) -> bool:
        outputs = []
        if r.stream:
            outputs.append((
                r.uid,
                [r.next_token],
                r.prompt_length,
                r.num_generated_tokens,
                GenerationFinishReason.NONE,
            ))
        if r.finish_reason != GenerationFinishReason.NONE:
            if r.stream or not r.generated_tokens:
                output_tokens = []
            else:
                output_tokens = torch.cat([t.unsqueeze(0) for t in r.generated_tokens],
                                          dim=0)
                if r.return_full_text:
                    # Avoid returning bos token, refactor this later
                    output_tokens = torch.cat((r.prompt_tokens[1:], output_tokens))
            outputs.append((
                r.uid,
                output_tokens,
                r.prompt_length,
                r.num_generated_tokens,
                r.finish_reason,
            ))
        for output in outputs:
            self.result_queues[r.tid].put_nowait(output)

    def _schedule_token_gen(self, requests: List[Request]) -> None:
        free_blocks = min(self.inference_engine.free_blocks)
        conf_manager = self.inference_engine._config.state_manager

        num_schedulable = min([
            len(requests),
            conf_manager.max_ragged_sequence_count,
            conf_manager.max_ragged_batch_size
        ])

        for r in requests[:num_schedulable]:
            block_capacity = self.inference_engine.get_remaining_block_capacity(r.uid)
            # We can schedule token generation if the last block has a capacity
            if block_capacity > 0:
                self.scheduled_length += 1
                self.scheduled_requests.append(r)
            elif free_blocks > 0:
                # We need a new block
                free_blocks -= 1
                self.scheduled_length += 1
                self.scheduled_req_blocks += 1
                self.scheduled_requests.append(r)

    def _schedule_prompts(self, requests: List[Request]) -> None:
        free_blocks = min(self.inference_engine.free_blocks)
        conf_manager = self.inference_engine._config.state_manager

        for r in requests:
            if free_blocks == 0:
                break

            if r.max_length <= r.seq_length:
                continue

            # Make sure that the engine has enough capacity to process the batch
            if len(self.scheduled_requests) > conf_manager.max_ragged_sequence_count:
                break

            max_batch_size = conf_manager.max_ragged_batch_size - self.scheduled_length
            if max_batch_size <= 0:
                break

            max_blocks = free_blocks - self.scheduled_req_blocks

            # Check capacity to mitigate the deadlock risk
            # We don't schedule requests when we find that a prompt is too long to fit to the KV cache
            if len(r.input_tokens) > 1:
                req_tokens, _ = self.inference_engine.query(r.uid, len(r.input_tokens), max_blocks)
                if req_tokens < len(r.input_tokens):
                    break

            req_tokens = min(len(r.input_tokens), max_batch_size)
            req_tokens, req_blocks = self.inference_engine.query(r.uid, req_tokens, max_blocks)

            if req_tokens <= 0:
                continue

            # Decompose the prompt to fit to the max ragged batch size
            decomposed = req_tokens < len(r.input_tokens)
            remaining_tokens = r.input_tokens[req_tokens:]
            r.input_tokens = r.input_tokens[:req_tokens]
            r.last_in_prompt = not decomposed

            # Schedule the request
            self.scheduled_requests.append(r)

            self.scheduled_req_blocks += req_blocks
            self.scheduled_length += req_tokens

            if decomposed:
                req_remaining = copy.copy(r)
                req_remaining.input_tokens = remaining_tokens
                req_remaining.seq_length = r.seq_length + req_tokens
                req_remaining.last_in_prompt = True

                self.buffer.appendleft(req_remaining)

    def schedule_requests(self) -> None:
        while not self.request_queue.empty():
            r = self.request_queue.get_nowait()
            self.buffer.append(r)

        next_token_gen_reqs = []
        prompt_reqs = []

        for r in self.buffer:
            if r.is_flush_request:
                self.scheduled_requests.append(r)
            else:
                if r.num_generated_tokens > 0:
                    if r.max_length > r.seq_length:
                        next_token_gen_reqs.append(r)
                else:
                    prompt_reqs.append(r)

        # We want to process next token generation first
        self._schedule_token_gen(next_token_gen_reqs)
        self._schedule_prompts(prompt_reqs)

        if len(self.buffer) > 0 and len(self.scheduled_requests) == 0:
            print(
                "Deadlock detected. Resetting KV cache and recomputing requests. Consider limiting number of concurrent requests or decreasing max lengths of prompts/generations."
            )
            self.scheduled_requests = RequestBatch()
            self.reset_request_status()
        else:
            scheduled_requests_ids = set(id(r) for r in self.scheduled_requests)
            self.buffer = deque(
                [r for r in self.buffer if id(r) not in scheduled_requests_ids])

    def _queue_flush_request(self, uid: int) -> None:
        self.request_queue.put_nowait(
            Request(
                tid=None,
                uid=uid,
                input_tokens=None,
                prompt_tokens=None,
                seq_length=None,
                last_in_prompt=None,
                post_processing=None,
                generate_params=None,
            ))

    def reset_request_status(self):
        for r in self.buffer:
            if r.seq_length > 0:
                self._queue_flush_request(r.uid)

        new_buffer = deque()
        for r in self.buffer:
            new_req = copy.copy(r)
            new_req.prompt_tokens = new_req.input_tokens = torch.concat(
                [r.prompt_tokens] + [t.unsqueeze(0) for t in r.generated_tokens])
            new_req.seq_length = 0
            new_req.max_new_tokens = r.max_new_tokens - len(r.generated_tokens)
            new_req.clear_generated_token()
            new_buffer.append(new_req)

        self.buffer = new_buffer

    def make_request(self,
                     tid: int,
                     uid: int,
                     input_tokens: torch.Tensor,
                     kwargs: Dict) -> Request:
        kwargs["prompt_length"] = len(input_tokens)
        kwargs["max_length"] = kwargs.get("max_length", self.max_length)
        generate_params = GenerateParamsConfig(**kwargs)

        post_processing = []

        top_p = generate_params.top_p
        top_p_name = "_".join((TOP_P_NAME, str(top_p)))
        if top_p_name not in self._post_processors:
            self._post_processors[top_p_name] = TopPLogitProcessor(top_p=top_p)
        post_processing.append(top_p_name)

        top_k = generate_params.top_k
        if top_k is not None:
            top_k_name = "_".join((TOP_K_NAME, str(top_k)))
            if top_k_name not in self._post_processors:
                self._post_processors[top_k_name] = TopKLogitProcessor(top_k=top_k)
            post_processing.append(top_k_name)

        temp = generate_params.temperature
        if temp is not None:
            temp_name = "_".join((TEMP_NAME, str(temp)))
            if temp_name not in self._post_processors:
                self._post_processors[temp_name] = TemperatureLogitProcessor(
                    temperature=temp)
            post_processing.append(temp_name)

        do_sample = generate_params.do_sample
        if do_sample:
            sampler_name = "_".join((SAMPLER_NAME, "logits"))
            if sampler_name not in self._post_processors:
                self._post_processors[sampler_name] = LogitsSampler()
        else:
            sampler_name = "_".join((SAMPLER_NAME, "greedy"))
            if sampler_name not in self._post_processors:
                self._post_processors[sampler_name] = GreedySampler()
        post_processing.append(sampler_name)

        stop = generate_params.stop
        if stop != []:
            stop_name = "_".join([STOP_NAME] + stop)
            if stop_name not in self._post_processors:
                self._post_processors[stop_name] = TokenStopCriterion(
                    token=stop,
                    tokenizer=self.tokenizer)
        else:
            stop_name = STOP_NAME
            if STOP_NAME not in self._post_processors:
                self._post_processors[stop_name] = EosGenerationStopCriterion(
                    tokenizer=self.tokenizer)
        post_processing.append(stop_name)

        return Request(
            tid=tid,
            uid=uid,
            input_tokens=input_tokens,
            prompt_tokens=input_tokens,
            seq_length=0,
            last_in_prompt=True,
            post_processing=post_processing,
            generate_params=generate_params,
        )

    def make_response(self,
                      generated_text: str,
                      prompt_length: int,
                      generated_length: int,
                      finish_reason: GenerationFinishReason) -> Response:
        return Response(generated_text=generated_text,
                        prompt_length=prompt_length,
                        generated_length=generated_length,
                        finish_reason=finish_reason)

    def put(self, uids: List[int], tokenized_input: List[torch.Tensor]) -> torch.Tensor:
        # Call inference engine. You can skip checking schedulability because we already checked when scheduling
        return self.inference_engine.put(uids, tokenized_input, do_checks=False)

    def flush(self, uids: List[int]) -> None:
        for uid in uids:
            self.inference_engine.flush(uid)


class MIIPipeline(RaggedBatchBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tid = threading.get_ident()
        self._destroyed = False

    def __call__(self, inputs: Union[str, List[str]], **kwargs) -> List[Response]:
        if self._destroyed:
            raise RuntimeError(
                "The inference engine of this pipeline has been destroyed.")

        if isinstance(inputs, str):
            inputs = [inputs]
        outputs: List[Response] = []
        uids_running: List[int] = list(range(len(inputs)))
        uids_complete_order: List[int] = []

        for uid, input in zip(uids_running, inputs):
            request_kwargs = kwargs.copy()
            self._put_request(uid, input, request_kwargs)

        self.schedule_requests()

        if self.is_first_rank_in_replica:
            # Rank 0 runs generate() until all responses are returned
            while uids_running:
                self.generate()
                while not self.result_queues[self.tid].empty():
                    uid, response = self._get_response()
                    outputs.append(response)
                    self._queue_flush_request(uid)
                    uids_complete_order.append(uid)
                    uids_running.remove(uid)
            # Ensure final flush requests broadcast and
            # kick ranks 1 -> n out of the while loop
            self._bcast_requests(force=True)
        else:
            # Ranks 1 -> n just run generate() until there are no more requests
            while self.scheduled_requests:
                self.generate()

        outputs = [
            r for idx,
            r in sorted(zip(uids_complete_order,
                            outputs),
                        key=lambda pair: pair[0])
        ]

        if self.model_config.all_rank_output:
            outputs = self._bcast_responses(outputs)

        return outputs

    def _put_request(self, uid: int, input: str, kwargs: Dict[str, Any]) -> None:
        self.result_queues[self.tid] = queue.Queue()
        input_tokens = self.tokenizer.encode(input)
        request = self.make_request(self.tid, uid, input_tokens, kwargs)
        self.put_request_and_notify(request)

    def _get_response(self) -> Tuple[int, Response]:
        result = self.result_queues[self.tid].get()
        uid = result[0]
        generated_tokens = self.tokenizer.decode(result[1])
        response = self.make_response(generated_tokens, result[2], result[3], result[4])
        return uid, response

    def _bcast_responses(self, responses: List[Response]) -> List[Response]:
        if self.is_first_rank_in_replica:
            data_dicts = [r.to_msg_dict() for r in responses]
            json_data = ujson.dumps(data_dicts)
            self.socket.send_string(json_data)
        else:
            json_data = self.socket.recv_string()
            data_dicts = ujson.loads(json_data)
            responses = [Response.from_msg_dict(msg) for msg in data_dicts]
        return responses

    def destroy(self) -> None:
        del self.inference_engine
        self.socket.close()
        self._zmq_context.term()
        gc.collect()
        get_accelerator().empty_cache()
        self._destroyed = True


class MIIAsyncPipeline(RaggedBatchBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uids = set()
        self.lock = threading.Lock()
        self.thread = None
        self.UID_RANGE_LB = 1
        self.UID_RANGE_UB = 10000

    def __call__(self) -> None:
        # CUDA device gets reset, must set it again to avoid problems
        get_accelerator().set_device(int(os.getenv("LOCAL_RANK", "0")))
        while True:
            self.generate()

            if (self.is_shutdown and self.request_queue.empty()
                    and all(q.empty() for q in self.result_queues.values())):
                break

    def _get_uid(self) -> int:
        with self.lock:
            uid = random.randrange(self.UID_RANGE_LB, self.UID_RANGE_UB)
            while uid in self.uids:
                uid = random.randrange(self.UID_RANGE_LB, self.UID_RANGE_UB)
            self.uids.add(uid)

        return uid

    def put_request(self, prompt: str, kwargs: Dict) -> int:
        # TODO: We should avoid any request/response work with non-rank 0, but
        # this requires some refactoring how we do the put and request in
        # `ModelResponse`
        #if not self.is_first_rank_in_replica:
        #    return
        if self.is_shutdown:
            raise RuntimeError("The request queue was shutdown.")

        uid = self._get_uid()

        # Temporary hack to avoid non-rank 0 processes not shutting down. See
        # related TODO above.
        if not self.is_first_rank_in_replica:
            return uid

        tid = threading.get_ident()
        with self.lock:
            if tid not in self.result_queues:
                self.result_queues[tid] = queue.Queue()

        input_tokens = self.tokenizer.encode(prompt)
        request = self.make_request(tid, uid, input_tokens, kwargs)
        self.put_request_and_notify(request)

        return uid

    def get_response(self) -> Tuple[int, Response]:
        # TODO: We should avoid any request/response work with non-rank 0, but
        # this requires some refactoring how we do the put and request in
        # `ModelResponse`
        if not self.is_first_rank_in_replica:
            return -1, Response(generated_text="",
                            prompt_length=None,
                            generated_length=None,
                            finish_reason=None)
        tid = threading.get_ident()
        result = self.result_queues[tid].get()
        uid = result[0]
        generated_token_ids = result[1]
        if len(generated_token_ids) == 0:
            generated_text = ""
        else:
            generated_text = self.tokenizer.decode(generated_token_ids)
        response = self.make_response(generated_text, result[2], result[3], result[4])
        return uid, response

    def start(self) -> None:
        self.thread = threading.Thread(target=self, daemon=True)
        self.thread.start()

    def shutdown(self) -> None:
        # Called by the first rank process.
        self._shutdown()
        self.thread.join()

    def flush_uid(self, uid: int) -> None:
        with self.lock:
            if self.is_first_rank_in_replica:
                self._queue_flush_request(uid)
            self.uids.remove(uid)
