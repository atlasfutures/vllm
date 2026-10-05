# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`mm_encoder_dtype` and `mm_encoder_per_item` on Qwen3-VL image inputs."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.multimodal import MultiModalConfig
from vllm.model_executor.models.interfaces import supports_mm_encoder_dtype
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    _apply_mm_encoder_dtype,
)

MERGE = 2
HIDDEN = 8
# Two images of 16 and 12 patch rows (4 and 3 merged tokens).
GRID_THW = torch.tensor([[1, 4, 4], [1, 2, 6]])


class FakeVisual:
    """Records each call; returns one float32 row per merged patch group."""

    spatial_merge_size = MERGE
    dtype = torch.float32

    def __init__(self):
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __call__(self, pixel_values, grid_thw):
        self.calls.append((pixel_values, grid_thw))
        rows = pixel_values[:: MERGE * MERGE, :1]
        return rows.expand(-1, HIDDEN).to(torch.float32)


def _fake_model(**mm_options):
    return SimpleNamespace(
        visual=FakeVisual(),
        use_data_parallel=False,
        multimodal_config=MultiModalConfig(**mm_options),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )


def _image_input():
    pixel_values = torch.arange(28, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
    return {
        "type": "pixel_values",
        "pixel_values": pixel_values,
        "image_grid_thw": GRID_THW,
    }


def test_per_item_fp32_runs_one_call_per_image_and_casts_to_lm_dtype():
    model = _fake_model(mm_encoder_dtype="float32", mm_encoder_per_item=True)
    image_input = _image_input()

    out = Qwen3VLForConditionalGeneration._process_image_input(model, image_input)

    calls = model.visual.calls
    assert len(calls) == 2
    pixel_values = image_input["pixel_values"]
    torch.testing.assert_close(calls[0][0], pixel_values[:16])
    torch.testing.assert_close(calls[1][0], pixel_values[16:])
    assert calls[0][1].tolist() == [[1, 4, 4]]
    assert calls[1][1].tolist() == [[1, 2, 6]]
    assert [e.shape for e in out] == [(4, HIDDEN), (3, HIDDEN)]
    assert all(e.dtype == torch.bfloat16 for e in out)


def test_default_runs_one_batched_call_and_keeps_encoder_dtype():
    model = _fake_model()

    out = Qwen3VLForConditionalGeneration._process_image_input(model, _image_input())

    assert len(model.visual.calls) == 1
    assert model.visual.calls[0][1] is GRID_THW
    assert [e.shape for e in out] == [(4, HIDDEN), (3, HIDDEN)]
    assert all(e.dtype == torch.float32 for e in out)


def test_precomputed_image_embeds_are_cast_to_lm_dtype():
    model = _fake_model(mm_encoder_dtype="float32")
    image_input = {
        "type": "image_embeds",
        "image_embeds": torch.zeros(7, HIDDEN, dtype=torch.bfloat16),
        "image_grid_thw": GRID_THW,
    }

    out = Qwen3VLForConditionalGeneration._process_image_input(model, image_input)

    assert not model.visual.calls
    assert [e.shape for e in out] == [(4, HIDDEN), (3, HIDDEN)]
    assert all(e.dtype == torch.bfloat16 for e in out)


@pytest.mark.parametrize(
    ("mm_encoder_dtype", "expected"),
    [(None, torch.bfloat16), ("float32", torch.float32)],
)
def test_init_hook_casts_vision_tower(mm_encoder_dtype, expected):
    model = SimpleNamespace(
        visual=torch.nn.Linear(2, 2, dtype=torch.bfloat16),
        multimodal_config=MultiModalConfig(mm_encoder_dtype=mm_encoder_dtype),
    )

    _apply_mm_encoder_dtype(model, quant_config=None)

    assert model.visual.weight.dtype == expected


def test_init_hook_refuses_quantized_model():
    model = SimpleNamespace(
        visual=torch.nn.Linear(2, 2, dtype=torch.bfloat16),
        multimodal_config=MultiModalConfig(mm_encoder_dtype="float32"),
    )
    quant_config = SimpleNamespace(get_name=lambda: "fp8")

    with pytest.raises(ValueError, match="quantized"):
        _apply_mm_encoder_dtype(model, quant_config=quant_config)


@pytest.mark.parametrize("use_data_parallel", [False, True])
def test_vision_tower_left_in_model_dtype_is_refused(use_data_parallel):
    """A loader that replaced the cast parameters fails loudly, not silently,
    on the data-parallel encoder path too."""
    model = _fake_model(mm_encoder_dtype="float32")
    model.use_data_parallel = use_data_parallel
    model.visual.dtype = torch.bfloat16

    with pytest.raises(RuntimeError, match="vision tower is torch.bfloat16"):
        Qwen3VLForConditionalGeneration._process_image_input(model, _image_input())


def test_support_is_declared_per_class_not_inherited():
    assert supports_mm_encoder_dtype(Qwen3VLForConditionalGeneration)
    assert supports_mm_encoder_dtype(Qwen3_5ForConditionalGeneration)
    assert not supports_mm_encoder_dtype(Qwen3_5MoeForConditionalGeneration)
