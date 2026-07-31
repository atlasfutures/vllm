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
        self.request_id = request_id
        self._input_queue: asyncio.Queue[StreamingInput | None] = asyncio.Queue(
            maxsize=1
        )
        self._output_ack: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
        self._append_lock = asyncio.Lock()
        self._closed = False
        self._started = False
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

    async def append(
        self,
        prompt: PromptType | EngineInput,
    ) -> PoolingRequestOutput:
        async with self._append_lock:
            if self._closed:
                raise PoolingSessionError("pooling session is closed")
            self._started = True
            await self._input_queue.put(StreamingInput(prompt=prompt))
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
