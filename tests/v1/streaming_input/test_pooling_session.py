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

_PAD = 99


class _FakePoolingEngine:
    def __init__(self) -> None:
        self.appended: list[list[int]] = []
        self.params: list[PoolingParams | None] = []
        self.input_finished = False

    async def _encode(
        self,
        prompt: AsyncGenerator[StreamingInput, None],
        request_id: str,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        cumulative: list[int] = []
        async for item in prompt:
            prompt_ids = item.prompt
            images: list[int] = []
            if isinstance(prompt_ids, dict):
                images = list(
                    (prompt_ids.get("multi_modal_data") or {}).get("image", [])
                )
                prompt_ids = prompt_ids["prompt_token_ids"]
            chunk = []
            for token in prompt_ids:
                # Like vLLM's processor: an image placeholder expands to the
                # image's tokens (a fake image is its token count).
                chunk += [token] * (images.pop(0) if token == _PAD else 1)
            self.appended.append(chunk)
            self.params.append(item.sampling_params)
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


def _gather(offsets: list[int] | None) -> PoolingParams:
    return PoolingParams(
        task="token_embed", retain_pooling_state=True, readout_offsets=offsets
    )


@pytest.mark.asyncio
async def test_pooling_session_sends_per_input_readout_offsets() -> None:
    engine = _FakePoolingEngine()
    session = AsyncPoolingSession(
        engine,
        pooling_params=PoolingParams(task="token_embed", retain_pooling_state=True),
        request_id="session-5",
    )

    await session.append(TokensPrompt(prompt_token_ids=[1, 2, 3]), _gather([0, 2]))
    await session.append(TokensPrompt(prompt_token_ids=[4]))
    await session.append(TokensPrompt(prompt_token_ids=[5, 6]), _gather([4, 5]))
    await session.close()

    assert [p and p.readout_offsets for p in engine.params] == [[0, 2], None, [4, 5]]
    assert session.num_tokens == 6


@pytest.mark.asyncio
async def test_pooling_session_refuses_offsets_outside_the_input() -> None:
    engine = _FakePoolingEngine()
    session = AsyncPoolingSession(
        engine,
        pooling_params=PoolingParams(task="token_embed", retain_pooling_state=True),
        request_id="session-6",
    )
    await session.append(TokensPrompt(prompt_token_ids=[1, 2, 3]), _gather([2]))

    for offsets in ([2, 4], [5]):  # an earlier input's position; past the end
        with pytest.raises(ValueError, match=r"\[3, 5\)"):
            await session.append(
                TokensPrompt(prompt_token_ids=[4, 5]), _gather(offsets)
            )

    # The refusal reached no engine and left the session usable.
    await session.append(TokensPrompt(prompt_token_ids=[4, 5]), _gather([3, 4]))
    await session.close()
    assert engine.appended == [[1, 2, 3], [4, 5]]


def test_pooling_session_refuses_session_level_offsets() -> None:
    with pytest.raises(ValueError, match="per input"):
        AsyncPoolingSession(
            _FakePoolingEngine(), pooling_params=_gather([0]), request_id="session-7"
        )


@pytest.mark.asyncio
async def test_pooling_session_tracks_raw_token_list_prompts() -> None:
    """A raw list[int] prompt has a known length, so later offsets stay usable."""
    engine = _FakePoolingEngine()
    session = AsyncPoolingSession(
        engine,
        pooling_params=PoolingParams(task="token_embed", retain_pooling_state=True),
        request_id="session-8",
    )
    await session.append([1, 2, 3])
    await session.append([4, 5], _gather([3, 4]))
    await session.close()
    assert session.num_tokens == 5


@pytest.mark.asyncio
async def test_pooling_session_counts_processed_tokens_after_a_multimodal_prompt():
    """An image prompt without declared counts takes no offsets, but the next
    input's offsets are placed after the engine's expanded count."""
    engine = _FakePoolingEngine()
    session = AsyncPoolingSession(
        engine,
        pooling_params=PoolingParams(task="token_embed", retain_pooling_state=True),
        request_id="session-9",
    )
    image_prompt = TokensPrompt(
        prompt_token_ids=[1, _PAD, 2], multi_modal_data={"image": [4]}
    )
    with pytest.raises(ValueError, match="image_token_counts"):
        await session.append(image_prompt, _gather([2]))
    await session.append(image_prompt)
    assert session.num_tokens == 6
    with pytest.raises(ValueError, match=r"\[6, 7\)"):
        await session.append(TokensPrompt(prompt_token_ids=[3]), _gather([2]))
    await session.append(TokensPrompt(prompt_token_ids=[3]), _gather([6]))
    await session.close()


def _gather_images(offsets: list[int], counts: list[int]) -> PoolingParams:
    return PoolingParams(
        task="token_embed",
        retain_pooling_state=True,
        readout_offsets=offsets,
        image_token_counts=counts,
    )


@pytest.mark.asyncio
async def test_pooling_session_places_offsets_after_an_image_in_the_first_append():
    engine = _FakePoolingEngine()
    session = AsyncPoolingSession(
        engine,
        pooling_params=PoolingParams(task="token_embed", retain_pooling_state=True),
        request_id="session-10",
    )
    # [1, 99x5, 2] spans [0, 7); the old one-pad count gave [0, 3).
    await session.append(
        TokensPrompt(prompt_token_ids=[1, _PAD, 2], multi_modal_data={"image": [5]}),
        _gather_images([5, 6], [5]),
    )
    assert session.num_tokens == 7
    with pytest.raises(ValueError, match=r"\[7, 9\)"):
        await session.append(TokensPrompt(prompt_token_ids=[3, 4]), _gather([3, 4]))
    await session.append(TokensPrompt(prompt_token_ids=[3, 4]), _gather([7, 8]))
    await session.close()
    assert engine.appended == [[1, 99, 99, 99, 99, 99, 2], [3, 4]]


@pytest.mark.asyncio
async def test_pooling_session_places_offsets_after_an_image_in_a_later_append():
    engine = _FakePoolingEngine()
    session = AsyncPoolingSession(
        engine,
        pooling_params=PoolingParams(task="token_embed", retain_pooling_state=True),
        request_id="session-11",
    )
    await session.append(TokensPrompt(prompt_token_ids=[1, 2]), _gather([1]))
    # Two images, expanding to 3 and 4: [3, 99x3, 4, 99x4, 5] spans [2, 12).
    with pytest.raises(ValueError, match=r"\[2, 12\)"):
        await session.append(
            TokensPrompt(
                prompt_token_ids=[3, _PAD, 4, _PAD, 5],
                multi_modal_data={"image": [3, 4]},
            ),
            _gather_images([12], [3, 4]),
        )
    await session.append(
        TokensPrompt(
            prompt_token_ids=[3, _PAD, 4, _PAD, 5],
            multi_modal_data={"image": [3, 4]},
        ),
        _gather_images([2, 6, 11], [3, 4]),
    )
    await session.append(TokensPrompt(prompt_token_ids=[6]), _gather([12]))
    await session.close()
    assert session.num_tokens == 13


@pytest.mark.asyncio
async def test_pooling_session_refuses_counts_for_another_number_of_images():
    engine = _FakePoolingEngine()
    session = AsyncPoolingSession(
        engine,
        pooling_params=PoolingParams(task="token_embed", retain_pooling_state=True),
        request_id="session-12",
    )
    with pytest.raises(ValueError, match="declares 2 images, but the prompt carries 1"):
        await session.append(
            TokensPrompt(prompt_token_ids=[_PAD], multi_modal_data={"image": [3]}),
            _gather_images([0], [3, 3]),
        )
    # A non-list image value has processor-defined item semantics, so its
    # length is unknown and its offsets are refused, not its count.
    with pytest.raises(ValueError, match="engine token count"):
        await session.append(
            TokensPrompt(prompt_token_ids=[_PAD], multi_modal_data={"image": 3}),
            _gather_images([0], [3, 3]),
        )
    # Refused before the engine; the session stays usable.
    await session.append(TokensPrompt(prompt_token_ids=[1]), _gather([0]))
    await session.close()
    assert engine.appended == [[1]]


@pytest.mark.asyncio
async def test_pooling_session_ends_when_the_engine_count_differs_from_declared():
    """A caller whose prompt already carries an expanded run would misplace every
    later offset; the session ends rather than continue on wrong positions."""
    engine = _FakePoolingEngine()
    session = AsyncPoolingSession(
        engine,
        pooling_params=PoolingParams(task="token_embed", retain_pooling_state=True),
        request_id="session-13",
    )
    with pytest.raises(PoolingSessionError, match="processed 7 session tokens"):
        await session.append(
            TokensPrompt(
                prompt_token_ids=[1, _PAD, 2], multi_modal_data={"image": [5]}
            ),
            _gather_images([0], [4]),
        )
    assert session.closed is True


def test_pooling_session_refuses_session_level_image_token_counts() -> None:
    with pytest.raises(ValueError, match="per input"):
        AsyncPoolingSession(
            _FakePoolingEngine(),
            pooling_params=PoolingParams(
                task="token_embed", retain_pooling_state=True, image_token_counts=[3]
            ),
            request_id="session-14",
        )
