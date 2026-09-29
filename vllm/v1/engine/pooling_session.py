# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Backpressured client for resumable pooling input sessions."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Protocol

from vllm.engine.protocol import StreamingInput
from vllm.inputs import EngineInput, PromptType
from vllm.outputs import PoolingRequestOutput
from vllm.pooling_params import PoolingParams


class PoolingSessionError(RuntimeError):
    """The retained pooling request violated its append/output lifecycle."""


class PoolingSessionEngine(Protocol):
    def encode(
        self,
        prompt: AsyncGenerator[StreamingInput, None],
        pooling_params: PoolingParams,
        request_id: str,
    ) -> AsyncGenerator[PoolingRequestOutput, None]: ...


class AsyncPoolingSession:
    """Keep one pooling request alive and return one output per input append.

    Pooling output collection keeps only the latest pending intermediate result,
    so this wrapper does not permit the input producer to advance until the
    caller has received the preceding output. Concurrent callers are serialized
    by the append lock. Closing the session finishes the underlying streaming
    request and releases its engine-owned cache and pooling accumulator.
    """

    def __init__(
        self,
        engine: PoolingSessionEngine,
        *,
        pooling_params: PoolingParams,
        request_id: str,
    ) -> None:
        if not request_id:
            raise ValueError("pooling session request_id must be non-empty")
        if not pooling_params.retain_pooling_state:
            raise ValueError("pooling sessions require retain_pooling_state=True")
        if pooling_params.readout_offsets is not None:
            raise ValueError(
                "readout_offsets are per input; pass them to append(), not to "
                "the session"
            )
        self.request_id = request_id
        self._input_queue: asyncio.Queue[StreamingInput | None] = asyncio.Queue(
            maxsize=1
        )
        self._output_ack: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
        self._append_lock = asyncio.Lock()
        self._closed = False
        self._started = False
        # Tokens appended so far; None once an input's length is unknown
        # (a text prompt), after which absolute readout offsets are refused.
        self._num_tokens: int | None = 0
        self._outputs = engine.encode(
            self._input_stream(),
            pooling_params,
            request_id,
        )

    @property
    def closed(self) -> bool:
        return self._closed

    async def _input_stream(self) -> AsyncGenerator[StreamingInput, None]:
        while (item := await self._input_queue.get()) is not None:
            yield item
            await self._output_ack.get()

    @property
    def num_tokens(self) -> int | None:
        """Tokens appended so far, or None if a text prompt hid the count."""
        return self._num_tokens

    async def append(
        self,
        prompt: PromptType | EngineInput,
        pooling_params: PoolingParams | None = None,
    ) -> PoolingRequestOutput:
        """Append one input and return its output.

        `pooling_params` overrides the session's parameters for this input
        only (e.g. GATHER `readout_offsets`, absolute positions in the whole
        session that must fall inside this input). Invalid offsets are refused
        here, before the engine sees the input, and leave the session usable.
        """
        async with self._append_lock:
            if self._closed:
                raise PoolingSessionError("pooling session is closed")
            input_len = _prompt_token_count(prompt)
            if pooling_params is not None:
                if not pooling_params.retain_pooling_state:
                    raise ValueError(
                        "per-input pooling params require retain_pooling_state=True"
                    )
                offsets = pooling_params.readout_offsets
                if offsets is not None:
                    if self._num_tokens is None or input_len is None:
                        raise ValueError(
                            "readout_offsets require token-id prompts for every "
                            "input of the session"
                        )
                    start, stop = self._num_tokens, self._num_tokens + input_len
                    if not offsets or offsets[0] < start or offsets[-1] >= stop:
                        raise ValueError(
                            f"readout offsets must lie in this input's token "
                            f"positions [{start}, {stop})"
                        )
                pooling_params = pooling_params.clone()
            self._started = True
            await self._input_queue.put(
                StreamingInput(prompt=prompt, sampling_params=pooling_params)
            )
            try:
                output = await anext(self._outputs)
            except StopAsyncIteration as error:
                self._closed = True
                raise PoolingSessionError(
                    "pooling session ended before producing an append output"
                ) from error
            except BaseException:
                self._closed = True
                await self._outputs.aclose()
                raise
            if output.finished:
                self._closed = True
                await self._outputs.aclose()
                raise PoolingSessionError(
                    "pooling session append unexpectedly finished the request"
                )
            await self._output_ack.put(None)
            if self._num_tokens is not None and input_len is not None:
                self._num_tokens += input_len
            else:
                self._num_tokens = None
            return output

    async def close(self) -> None:
        async with self._append_lock:
            if self._closed:
                return
            self._closed = True
            if not self._started:
                await self._outputs.aclose()
                return
            await self._input_queue.put(None)
            try:
                await anext(self._outputs)
            except StopAsyncIteration:
                return
            finally:
                await self._outputs.aclose()
            raise PoolingSessionError(
                "pooling session emitted an output after input completion"
            )

    async def __aenter__(self) -> "AsyncPoolingSession":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()


def _prompt_token_count(prompt: PromptType | EngineInput) -> int | None:
    if isinstance(prompt, dict):
        token_ids = prompt.get("prompt_token_ids")
        if token_ids is not None and "prompt_embeds" not in prompt:
            return len(token_ids)
    return None
