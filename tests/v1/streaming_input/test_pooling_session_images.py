# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GATHER pooling sessions with images, end to end on a GPU.

A retained session's readouts must equal the one-shot readouts of the same
expanded prompt when images arrive in the first append or in a later one: the
session places offsets in vLLM's expanded positions (each image placeholder
counts as its expanded tokens) and the engine refuses a declared expansion that
differs from its own.
"""

import uuid

import pytest
import pytest_asyncio
from PIL import Image

from tests.models.utils import check_embeddings_close
from vllm.config import PoolerConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.platforms import current_platform
from vllm.pooling_params import PoolingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.pooling_session import AsyncPoolingSession

if not current_platform.is_cuda():
    pytest.skip(reason="V1 currently only supported on CUDA.", allow_module_level=True)

MODEL = "Qwen/Qwen3.5-0.8B"

# Segments of an expanded prompt: a run of text token ids, or an image.
TEXT_A = list(range(1000, 1040))
TEXT_B = list(range(2000, 2025))
TEXT_C = list(range(3000, 3030))
TEXT_D = list(range(4000, 4020))


def _image(width: int, height: int, seed: int) -> Image.Image:
    image = Image.new("RGB", (width, height))
    image.putdata(
        [
            ((x * 7 + seed) % 256, (y * 5 + seed) % 256, (x + y + seed) % 256)
            for y in range(height)
            for x in range(width)
        ]
    )
    return image


IMAGE_1 = _image(320, 240, seed=1)
IMAGE_2 = _image(200, 360, seed=2)


def _gather(
    offsets: list[int] | None,
    counts: list[int] | None = None,
    retain: bool = True,
) -> PoolingParams:
    return PoolingParams(
        task="token_embed",
        use_activation=False,
        retain_pooling_state=retain,
        readout_offsets=offsets,
        image_token_counts=counts,
    )


# Vision tower in the model dtype (bf16), or in float32 run once per image
# with its embeddings cast to bf16 at language-model entry.
ENCODER_OPTIONS = {
    "bf16_encoder": {},
    "fp32_per_item_encoder": {
        "mm_encoder_dtype": "float32",
        "mm_encoder_per_item": True,
        "mm_encoder_attn_backend": "TORCH_SDPA",
    },
}


# One engine per encoder option: each test reuses its loop.
@pytest_asyncio.fixture(
    scope="module", loop_scope="module", params=sorted(ENCODER_OPTIONS)
)
async def engine(request):
    llm = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            **ENCODER_OPTIONS[request.param],
            model=MODEL,
            runner="pooling",
            dtype="bfloat16",
            max_model_len=8192,
            max_num_seqs=4,
            enforce_eager=True,
            enable_prefix_caching=False,
            gpu_memory_utilization=0.5,
            limit_mm_per_prompt={"image": 2, "video": 0},
            pooler_config=PoolerConfig(
                seq_pooling_type="MEAN",
                tok_pooling_type="GATHER",
                use_activation=False,
            ),
        )
    )
    try:
        yield llm
    finally:
        llm.shutdown()


async def _encode(llm: AsyncLLM, prompt, params: PoolingParams):
    final = None
    async for output in llm.encode(prompt, params, request_id=uuid.uuid4().hex):
        final = output
    assert final is not None
    return final


async def _image_token_count(llm: AsyncLLM, image: Image.Image) -> int:
    """The tokens vLLM's processor expands this image's placeholder to."""
    hf = llm.model_config.hf_config
    ids = [hf.vision_start_token_id, hf.image_token_id, hf.vision_end_token_id]
    final = await _encode(
        llm,
        TokensPrompt(prompt_token_ids=ids, multi_modal_data={"image": [image]}),
        _gather(None, retain=False),
    )
    return len(final.prompt_token_ids) - 2


def _chunk(llm: AsyncLLM, segments: list, counts: dict[int, int]):
    """Collapsed prompt, its images, their declared counts, its expanded length
    and the expanded position that closes each segment."""
    hf = llm.model_config.hf_config
    ids: list[int] = []
    images: list[Image.Image] = []
    expanded = 0
    ends: list[int] = []
    for segment in segments:
        if isinstance(segment, Image.Image):
            ids += [hf.vision_start_token_id, hf.image_token_id, hf.vision_end_token_id]
            images.append(segment)
            expanded += counts[id(segment)] + 2
        else:
            ids += segment
            expanded += len(segment)
        ends.append(expanded - 1)
    prompt = TokensPrompt(prompt_token_ids=ids)
    if images:
        prompt["multi_modal_data"] = {"image": images}
    return prompt, [counts[id(i)] for i in images], expanded, ends


# Each case lists its appends; one-shot runs their concatenation.
CASES = {
    "image_in_first_append": [[TEXT_A, IMAGE_1, TEXT_B], [TEXT_C]],
    "image_in_later_append": [[TEXT_A], [TEXT_B, IMAGE_1, TEXT_C], [TEXT_D]],
    "images_in_first_and_later_appends": [
        [TEXT_A, IMAGE_1, TEXT_B],
        [TEXT_C],
        [TEXT_D, IMAGE_2, TEXT_A],
    ],
}


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize("case", sorted(CASES))
async def test_image_session_readouts_equal_one_shot(engine, case):
    appends = CASES[case]
    counts = {
        id(image): await _image_token_count(engine, image)
        for image in (IMAGE_1, IMAGE_2)
    }

    # One-shot over the whole prompt, reading the close of every segment.
    prompt, image_counts, expanded, ends = _chunk(
        engine, [s for chunk in appends for s in chunk], counts
    )
    one_shot = await _encode(
        engine, prompt, _gather(ends, image_counts or None, retain=False)
    )
    assert len(one_shot.prompt_token_ids) == expanded
    reference = one_shot.outputs.data.float()
    assert reference.shape[0] == len(ends)

    rows = []
    base = 0
    async with AsyncPoolingSession(
        engine,
        pooling_params=_gather(None),
        request_id=uuid.uuid4().hex,
    ) as session:
        for chunk in appends:
            prompt, image_counts, length, chunk_ends = _chunk(engine, chunk, counts)
            offsets = [base + end for end in chunk_ends]
            output = await session.append(
                prompt, _gather(offsets, image_counts or None)
            )
            base += length
            # The session counts the expanded positions vLLM processed.
            assert session.num_tokens == base == len(output.prompt_token_ids)
            rows += output.outputs.data.float().tolist()

    check_embeddings_close(
        embeddings_0_lst=reference.tolist(),
        embeddings_1_lst=rows,
        name_0="one_shot",
        name_1="session",
    )


@pytest.mark.asyncio(loop_scope="module")
async def test_declared_image_token_count_mismatch_is_refused(engine):
    n = await _image_token_count(engine, IMAGE_1)
    prompt, _, expanded, ends = _chunk(
        engine, [TEXT_A, IMAGE_1, TEXT_B], {id(IMAGE_1): n}
    )

    with pytest.raises(ValueError, match="declared expansion"):
        await _encode(engine, prompt, _gather([ends[-1]], [n + 1], retain=False))

    final = await _encode(engine, prompt, _gather([ends[-1]], [n], retain=False))
    assert len(final.prompt_token_ids) == expanded
