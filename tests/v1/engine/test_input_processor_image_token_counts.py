# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A pooling request's declared image expansion must match vLLM's processor."""

import pytest

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.pooling_params import PoolingParams
from vllm.v1.engine.input_processor import _check_image_token_counts


def _feature(modality: str, offset: int, length: int) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality=modality,
        identifier=f"{modality}-{offset}",
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


def _params(counts: list[int] | None) -> PoolingParams:
    return PoolingParams(task="token_embed", image_token_counts=counts)


def test_matching_declared_image_token_counts_pass() -> None:
    features = [_feature("image", 1, 700), _feature("image", 710, 1050)]
    _check_image_token_counts(_params([700, 1050]), features)
    # Nothing declared, nothing checked.
    _check_image_token_counts(_params(None), features)
    _check_image_token_counts(_params([]), None)


@pytest.mark.parametrize(
    ("counts", "features"),
    [
        ([700], [_feature("image", 1, 701)]),  # another expansion
        ([700, 64], [_feature("image", 1, 700)]),  # an image missing
        ([700], None),  # no images processed
        ([700], [_feature("image", 1, 700), _feature("image", 800, 64)]),
    ],
)
def test_differing_image_token_counts_are_refused(counts, features) -> None:
    with pytest.raises(ValueError, match="declared expansion"):
        _check_image_token_counts(_params(counts), features)


def test_non_image_features_are_not_counted() -> None:
    features = [_feature("audio", 0, 30), _feature("image", 40, 64)]
    _check_image_token_counts(_params([64]), features)


@pytest.mark.parametrize("counts", [[0], [-3], [2.0]])
def test_image_token_counts_must_be_positive_ints(counts) -> None:
    with pytest.raises(ValueError, match="positive token counts"):
        _params(counts)
