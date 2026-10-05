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
        if pooling_params.image_token_counts is not None:
            raise ValueError(
                "image_token_counts are per input; pass them to append(), not "
                "to the session"
            )
        self.request_id = request_id
        self._input_queue: asyncio.Queue[StreamingInput | None] = asyncio.Queue(
            maxsize=1
        )
        self._output_ack: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
        self._append_lock = asyncio.Lock()
        self._closed = False
        self._started = False
        # Engine positions appended so far, as vLLM processed them (an image
        # placeholder counts as its expanded tokens). None only if an output
        # did not report its processed prompt.
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
        """Engine positions appended so far (images expanded), or None if
        the engine did not report its processed prompt."""
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

        Positions are vLLM's processed positions: each image placeholder
        counts as the tokens it expands to. Offsets on an input with images
        need `pooling_params.image_token_counts` (one placeholder token per
        image in `prompt_token_ids`, plus the declared expansion of each).
        The engine refuses the input if its expansion differs, which ends the
        session. After each output the session takes its position count from
        the engine's processed prompt, and ends the session if that differs
        from the declared lengths.
        """
        async with self._append_lock:
            if self._closed:
                raise PoolingSessionError("pooling session is closed")
            input_len = _prompt_token_count(prompt, pooling_params)
            if pooling_params is not None:
                if not pooling_params.retain_pooling_state:
                    raise ValueError(
                        "per-input pooling params require retain_pooling_state=True"
                    )
                offsets = pooling_params.readout_offsets
                if offsets is not None:
                    if self._num_tokens is None or input_len is None:
                        raise ValueError(
                            "readout_offsets need this input's engine token "
                            "count: pass a token-id prompt, with "
                            "image_token_counts for a prompt with images"
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
            # Read the processed length before releasing the next input: the
            # engine's prompt list grows with the session.
            processed = output.prompt_token_ids
            expected = (
                None
                if self._num_tokens is None or input_len is None
                else self._num_tokens + input_len
            )
            self._num_tokens = (
                len(processed) if isinstance(processed, list) else expected
            )
            if expected is not None and self._num_tokens != expected:
                # The caller's positions no longer match the engine's.
                self._closed = True
                await self._outputs.aclose()
                raise PoolingSessionError(
                    f"vLLM processed {self._num_tokens} session tokens, but "
                    f"the inputs' declared lengths give {expected}; a prompt "
                    f"with images must carry one placeholder token per image"
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


def _prompt_token_count(
    prompt: PromptType | EngineInput,
    pooling_params: PoolingParams | None,
) -> int | None:
    """The engine positions `prompt` will occupy, or None if unknown before
    processing (a text prompt, embeddings, or images without a declared
    expansion)."""
    if isinstance(prompt, list) and all(type(t) is int for t in prompt):
        return len(prompt)
    if not isinstance(prompt, dict) or "prompt_embeds" in prompt:
        return None
    token_ids = prompt.get("prompt_token_ids")
    if not isinstance(token_ids, list):
        return None
    mm_data = prompt.get("multi_modal_data")
    if not mm_data:
        return len(token_ids)
    # Multimodal preprocessing expands each image placeholder, so the
    # caller's ids give the engine's positions only with the declared
    # expansion (which the engine verifies).
    counts = pooling_params.image_token_counts if pooling_params else None
    if counts is None or not isinstance(mm_data, dict) or set(mm_data) != {"image"}:
        return None
    images = mm_data["image"]
    num_images = len(images) if isinstance(images, list) else 1
    if num_images != len(counts):
        raise ValueError(
            f"image_token_counts declares {len(counts)} images, but the "
            f"prompt carries {num_images}"
        )
    return len(token_ids) - num_images + sum(counts)
