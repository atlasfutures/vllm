# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections.abc import AsyncGenerator

import pytest
import torch

from vllm.engine.protocol import StreamingInput
from vllm.inputs import TokensPrompt
from vllm.outputs import PoolingOutput, PoolingRequestOutput
from vllm.pooling_params import PoolingParams
from vllm.v1.engine.pooling_session import AsyncPoolingSession, PoolingSessionError


class _FakePoolingEngine:
    def __init__(self) -> None:
        self.appended: list[list[int]] = []
        self.input_finished = False

    async def _encode(
        self,
        prompt: AsyncGenerator[StreamingInput, None],
        request_id: str,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        cumulative: list[int] = []
        async for item in prompt:
            chunk = list(item.prompt["prompt_token_ids"])
            self.appended.append(chunk)
            cumulative.extend(chunk)
            yield PoolingRequestOutput(
                request_id=request_id,
                outputs=PoolingOutput(data=torch.tensor([float(len(cumulative))])),
                prompt_token_ids=list(cumulative),
                num_cached_tokens=0,
                finished=False,
            )
        self.input_finished = True

    def encode(
        self,
        prompt: AsyncGenerator[StreamingInput, None],
        pooling_params: PoolingParams,
        request_id: str,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        assert pooling_params.retain_pooling_state is True
        return self._encode(prompt, request_id)


def _params() -> PoolingParams:
    return PoolingParams(task="embed", retain_pooling_state=True)


@pytest.mark.asyncio
async def test_pooling_session_returns_one_cumulative_output_per_append() -> None:
    engine = _FakePoolingEngine()
    session = AsyncPoolingSession(
        engine,
        pooling_params=_params(),
        request_id="session-1",
    )

    first = await session.append(TokensPrompt(prompt_token_ids=[1, 2, 3]))
    second = await session.append(TokensPrompt(prompt_token_ids=[4, 5]))
    await session.close()

    assert first.prompt_token_ids == [1, 2, 3]
    assert second.prompt_token_ids == [1, 2, 3, 4, 5]
    assert engine.appended == [[1, 2, 3], [4, 5]]
    assert engine.input_finished is True
    assert session.closed is True


@pytest.mark.asyncio
async def test_pooling_session_serializes_concurrent_appends() -> None:
    engine = _FakePoolingEngine()
    session = AsyncPoolingSession(
        engine,
        pooling_params=_params(),
        request_id="session-2",
    )

    first, second = await asyncio.gather(
        session.append(TokensPrompt(prompt_token_ids=[1])),
        session.append(TokensPrompt(prompt_token_ids=[2])),
    )
    await session.close()

    assert first.prompt_token_ids == [1]
    assert second.prompt_token_ids == [1, 2]
    assert engine.appended == [[1], [2]]


@pytest.mark.asyncio
async def test_pooling_session_rejects_append_after_close() -> None:
    session = AsyncPoolingSession(
        _FakePoolingEngine(),
        pooling_params=_params(),
        request_id="session-3",
    )
    await session.close()

    with pytest.raises(PoolingSessionError, match="closed"):
        await session.append(TokensPrompt(prompt_token_ids=[1]))


def test_pooling_session_requires_retained_state() -> None:
    with pytest.raises(ValueError, match="retain_pooling_state"):
        AsyncPoolingSession(
            _FakePoolingEngine(),
            pooling_params=PoolingParams(task="embed"),
            request_id="session-4",
        )
