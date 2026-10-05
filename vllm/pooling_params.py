# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from typing import Any

import msgspec

from vllm.config import ModelConfig, PoolerConfig
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind
from vllm.tasks import PoolingTask, check_removed_pooling_task

logger = init_logger(__name__)


class LateInteractionParams(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    """Metadata for worker-side late-interaction scoring.

    Attributes:
        mode:
            - "cache_query": cache query token embeddings
            - "score_doc": score a document against a cached query.
        query_key: stable key used for both DP routing and worker cache lookup.
        query_uses: expected number of document requests
    """

    mode: str
    query_key: str
    query_uses: int | None = None


class PoolingParams(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    """API parameters for pooling models.

    Attributes:
        use_activation: Whether to apply activation function to the pooler outputs.
            `None` uses the pooler's default, which is `True` in most cases.
        dimensions: Reduce the dimensions of embeddings
            if model support matryoshka representation.
    """

    # --8<-- [start:common-pooling-params]
    use_activation: bool | None = None
    # --8<-- [end:common-pooling-params]

    ## for embeddings models
    # --8<-- [start:embed-pooling-params]
    dimensions: int | None = None
    # --8<-- [end:embed-pooling-params]

    ## for step pooling models
    step_tag_id: int | None = None
    returned_token_ids: list[int] | None = None

    ## Internal use only
    task: PoolingTask | None = None
    requires_token_ids: bool = False
    skip_reading_prefix_cache: bool | None = None
    late_interaction_params: LateInteractionParams | None = None
    extra_kwargs: dict[str, Any] | None = None
    output_kind: RequestOutputKind = RequestOutputKind.FINAL_ONLY
    # Retain sequence-pooling accumulator state between resumable streaming
    # input chunks. This is an engine-internal session primitive; ordinary
    # pooling requests remain one-shot and clear their state after output.
    retain_pooling_state: bool = False
    # GATHER token pooling: absolute token positions whose final hidden states
    # are returned, strictly increasing. For a retained streaming session the
    # positions index the concatenation of every input so far and must lie in
    # the input being processed. None selects the input's last token.
    readout_offsets: list[int] | None = None
    # Declared expansion of this input's images: for each image in the
    # prompt's multi_modal_data, in prompt order, the number of prompt
    # positions vLLM's processor must expand its placeholder to. The request is
    # refused when the processed placeholders differ. A retained session also
    # uses it to place this input's readout offsets before the engine runs.
    image_token_counts: list[int] | None = None

    @property
    def all_parameters(self) -> list[str]:
        return ["dimensions", "use_activation"]

    @property
    def valid_parameters(self):
        return {
            "embed": ["dimensions", "use_activation"],
            "classify": ["use_activation"],
            "token_embed": ["dimensions", "use_activation"],
            "token_classify": ["use_activation"],
        }

    def clone(self) -> "PoolingParams":
        """Returns a deep copy of the PoolingParams instance."""
        return deepcopy(self)

    def verify(self, model_config: ModelConfig) -> None:
        # plugin task uses io_processor.parse_request to verify inputs,
        # skipping PoolingParams verify
        if self.task == "plugin":
            if self.skip_reading_prefix_cache is None:
                self.skip_reading_prefix_cache = True
            return

        # skipping verify, let plugins configure and validate pooling params
        if self.task not in self.valid_parameters:
            return

        # NOTE: Task validation needs to done against the model instance,
        # which is not available in model config. So, it's not included
        # in this method
        self._merge_default_parameters(model_config)
        self._set_default_parameters(model_config)
        self._verify_valid_parameters()

    def _merge_default_parameters(self, model_config: ModelConfig) -> None:
        pooler_config = model_config.pooler_config
        if pooler_config is None:
            return

        if self.task is None:
            raise ValueError("task must be set before merging parameters")

        token_task = self.task in ("token_embed", "token_classify")
        gather = token_task and pooler_config.tok_pooling_type == "GATHER"
        if self.retain_pooling_state and not (
            gather or (not token_task and pooler_config.seq_pooling_type == "MEAN")
        ):
            raise ValueError(
                "retain_pooling_state is supported only for sequence MEAN "
                "pooling and token GATHER pooling"
            )
        if self.readout_offsets is not None:
            if not gather:
                raise ValueError("readout_offsets requires token GATHER pooling")
            offsets = self.readout_offsets
            if (
                not offsets
                or any(type(o) is not int or o < 0 for o in offsets)
                or any(a >= b for a, b in zip(offsets, offsets[1:]))
            ):
                raise ValueError(
                    "readout_offsets must be a non-empty, strictly increasing "
                    "list of non-negative token positions"
                )
        valid_parameters = self.valid_parameters[self.task]

        for k in valid_parameters:
            if getattr(pooler_config, k, None) is None:
                continue

            if getattr(self, k, None) is None:
                setattr(self, k, getattr(pooler_config, k))

        if self.skip_reading_prefix_cache is None:
            # If prefix caching is enabled,
            # the output of all pooling may less than n_prompt_tokens,
            # we need to skip reading cache at this request.
            # Causal MEAN accumulates every prompt hidden state, so a cached
            # prefix (including one served by a KV connector, which local
            # prefix-caching support flags do not gate) would leave the
            # accumulator short of the prompt length.
            self.skip_reading_prefix_cache = (
                self.task in ["token_embed", "token_classify"]
                or pooler_config.seq_pooling_type == "MEAN"
            )

        self._verify_step_pooling(pooler_config, valid_parameters)

    def _verify_step_pooling(
        self,
        pooler_config: PoolerConfig,
        valid_parameters: list[str],
    ):
        step_pooling_parameters = ["step_tag_id", "returned_token_ids"]
        if pooler_config.tok_pooling_type != "STEP":
            invalid_parameters = []
            for k in step_pooling_parameters:
                if getattr(self, k, None) is not None:
                    invalid_parameters.append(k)

            if invalid_parameters:
                raise ValueError(
                    f"Task {self.task} only supports {valid_parameters} "
                    f"parameters, does not support "
                    f"{invalid_parameters} parameters"
                )
        else:
            for k in step_pooling_parameters:
                if getattr(pooler_config, k, None) is None:
                    continue

                if getattr(self, k, None) is None:
                    setattr(self, k, getattr(pooler_config, k))

    def _set_default_parameters(self, model_config: ModelConfig):
        if self.task in ["embed", "token_embed"]:
            if self.use_activation is None:
                self.use_activation = True

            if self.dimensions is not None:
                dimensions = self.dimensions
                model_name = model_config.served_model_name
                embedding_size = model_config.embedding_size
                valid_range = f"[1, {embedding_size}]"
                dimensions_in_range = 1 <= dimensions <= embedding_size
                if not model_config.is_matryoshka:
                    raise ValueError(
                        f"Model {model_name!r} does not support Matryoshka "
                        f"embeddings; dimensions must be unset "
                        f"(received dimensions={dimensions})."
                    )

                if not dimensions_in_range:
                    raise ValueError(
                        f"Model {model_name!r} only supports dimensions in "
                        f"range {valid_range}, got {dimensions}."
                    )

                mds = model_config.matryoshka_dimensions
                if mds is not None and dimensions not in mds:
                    raise ValueError(
                        f"Model {model_name!r} only supports Matryoshka "
                        f"dimensions {str(mds)}, got {dimensions}."
                    )

        elif self.task in ["classify", "token_classify"]:
            if self.use_activation is None:
                self.use_activation = True
        else:
            raise ValueError(f"Unknown pooling task: {self.task!r}")

    def _verify_valid_parameters(self):
        if self.task is None:
            raise ValueError("task must be set before verifying parameters")
        valid_parameters = self.valid_parameters[self.task]
        invalid_parameters = []
        for k in self.all_parameters:
            if k in valid_parameters:
                continue

            if getattr(self, k, None) is not None:
                invalid_parameters.append(k)

        if invalid_parameters:
            raise ValueError(
                f"Task {self.task!r} only supports {valid_parameters} "
                f"parameters, does not support "
                f"{invalid_parameters} parameters"
            )

    def __repr__(self) -> str:
        return (
            f"PoolingParams("
            f"task={self.task}, "
            f"dimensions={self.dimensions}, "
            f"use_activation={self.use_activation}, "
            f"step_tag_id={self.step_tag_id}, "
            f"returned_token_ids={self.returned_token_ids}, "
            f"requires_token_ids={self.requires_token_ids}, "
            f"skip_reading_prefix_cache={self.skip_reading_prefix_cache}, "
            f"late_interaction_params={self.late_interaction_params}, "
            f"extra_kwargs={self.extra_kwargs}, "
            f"readout_offsets={self.readout_offsets}, "
            f"image_token_counts={self.image_token_counts})"
        )

    def __post_init__(self) -> None:
        check_removed_pooling_task(self.task)
        if self.output_kind != RequestOutputKind.FINAL_ONLY:
            raise ValueError(
                "For pooling output_kind has to be FINAL_ONLY, "
                f"got {self.output_kind!r}"
            )
        counts = self.image_token_counts
        if counts is not None and any(type(n) is not int or n < 1 for n in counts):
            raise ValueError(
                "image_token_counts must be a list of positive token counts, "
                f"got {counts!r}"
            )
