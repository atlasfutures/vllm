# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from bisect import bisect_left
from collections.abc import Set
from typing import TypeAlias

import torch
import torch.nn as nn

from vllm.config import get_current_vllm_config
from vllm.config.pooler import TokenPoolingType
from vllm.model_executor.layers.pooler import PoolingParamsUpdate
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

TokenPoolingMethodOutputItem: TypeAlias = torch.Tensor | None


class TokenPoolingMethod(nn.Module, ABC):
    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed", "token_classify"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate()

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> list[TokenPoolingMethodOutputItem]:
        raise NotImplementedError


class AllPool(TokenPoolingMethod):
    def __init__(self):
        super().__init__()

        vllm_config = get_current_vllm_config()
        scheduler_config = vllm_config.scheduler_config

        self.enable_chunked_prefill = scheduler_config.enable_chunked_prefill

    def extra_repr(self) -> str:
        return f"enable_chunked_prefill={self.enable_chunked_prefill}"

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> list[TokenPoolingMethodOutputItem]:
        pooling_cursor = pooling_metadata.get_pooling_cursor()
        # Use the already-CPU num_scheduled_tokens tensor so `.tolist()`
        # doesn't trigger a GPU->CPU sync. torch.split produces the same
        # consecutive slices as indexing with first/last per-sequence indices.
        hidden_states_lst = list(
            torch.split(hidden_states, pooling_cursor.num_scheduled_tokens_cpu.tolist())
        )

        if not self.enable_chunked_prefill:
            return hidden_states_lst

        pooling_states = pooling_metadata.pooling_states

        # If chunked_prefill is enabled
        # 1. first store the chunked hidden_states in pooling_states.hidden_states_cache
        for p, hs_chunk in zip(pooling_states, hidden_states_lst):
            p.hidden_states_cache.append(hs_chunk)

        # 2. Once prefill is finished, send hidden_states_cache to PoolerHead
        output_list = list[TokenPoolingMethodOutputItem]()
        for p, finished in zip(pooling_states, pooling_cursor.is_finished()):
            if finished:
                hidden_states_cache = p.hidden_states_cache
                if len(hidden_states_cache) == 1:
                    output_list.append(hidden_states_cache[0])
                else:
                    output_list.append(torch.concat(hidden_states_cache, dim=0))
                p.clean()
            else:
                output_list.append(None)

        return output_list


class StepPool(AllPool):
    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> list[TokenPoolingMethodOutputItem]:
        pooled_data_lst = super().forward(hidden_states, pooling_metadata)
        # Use the CPU copy of prompt_token_ids so the step_tag_id mask can be
        # resolved to indices without a d2h sync from boolean indexing.
        prompt_token_ids_cpu = pooling_metadata.get_prompt_token_ids_cpu()
        pooling_params = pooling_metadata.pooling_params

        pooled_data = list[torch.Tensor | None]()
        for data, token_id_cpu, pooling_param in zip(
            pooled_data_lst, prompt_token_ids_cpu, pooling_params
        ):
            # for unfinished chunked prefill
            if data is None:
                pooled_data.append(None)
            else:
                step_tag_id = pooling_param.step_tag_id
                returned_token_ids = pooling_param.returned_token_ids

                if returned_token_ids is not None and len(returned_token_ids) > 0:
                    data = data[:, returned_token_ids]

                if step_tag_id is not None:
                    idx_cpu = (token_id_cpu == step_tag_id).nonzero(as_tuple=True)[0]
                    idx = idx_cpu.to(data.device, non_blocking=True)
                    data = data[idx]

                pooled_data.append(data)

        return pooled_data


class GatherPool(TokenPoolingMethod):
    """Final hidden states at the token positions a request names.

    `PoolingParams.readout_offsets` are absolute positions in the request's
    token sequence; for a retained streaming session that is the concatenation
    of every input appended so far, so a position never changes meaning as the
    session grows. Every offset must lie inside the input being processed
    (`[tokens before this input, tokens after it)`); `None` selects the input's
    last token. The output for an input is `[len(offsets), hidden]`, in offset
    order.

    Rows are selected from each scheduled chunk as it runs and only those rows
    are kept, so chunked prefill needs no all-token cache. A preempted request
    recomputes from position 0 with its state cleaned and gathers again.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> list[TokenPoolingMethodOutputItem]:
        pooling_cursor = pooling_metadata.get_pooling_cursor()
        scheduled = pooling_cursor.num_scheduled_tokens_cpu.tolist()
        seq_lens = pooling_cursor.seq_lens_cpu.tolist()
        prompt_lens = pooling_cursor.prompt_lens_cpu.tolist()
        params_list = pooling_metadata.pooling_params

        # One index_select for the whole batch; indices are built on the CPU
        # from CPU-resident counts, so selection needs no device sync.
        row_index: list[int] = []
        spans: list[tuple[int, int]] = []
        first_row = 0
        for params, num_scheduled, seq_len, prompt_len in zip(
            params_list, scheduled, seq_lens, prompt_lens
        ):
            chunk_start = seq_len - num_scheduled
            offsets = _readout_offsets(params, prompt_len)
            low = bisect_left(offsets, chunk_start)
            high = bisect_left(offsets, seq_len)
            begin = len(row_index)
            row_index.extend(
                first_row + offset - chunk_start for offset in offsets[low:high]
            )
            spans.append((begin, len(row_index)))
            first_row += num_scheduled

        gathered = None
        if row_index:
            index = torch.tensor(row_index, dtype=torch.long).to(
                hidden_states.device, non_blocking=True
            )
            gathered = hidden_states.index_select(0, index)

        output_list: list[TokenPoolingMethodOutputItem] = []
        for state, params, (begin, end), finished, prompt_len in zip(
            pooling_metadata.pooling_states,
            params_list,
            spans,
            pooling_cursor.is_finished().tolist(),
            prompt_lens,
        ):
            if end > begin:
                assert gathered is not None
                state.hidden_states_cache.append(gathered[begin:end])
            if not finished:
                output_list.append(None)
                continue
            rows = state.hidden_states_cache
            expected = len(_readout_offsets(params, prompt_len))
            got = sum(row.shape[0] for row in rows)
            state.clean()
            if got != expected:
                raise RuntimeError(
                    f"GATHER pooling collected {got} of {expected} readout rows; "
                    "readout_offsets must lie inside the current input"
                )
            output_list.append(rows[0] if len(rows) == 1 else torch.cat(rows, dim=0))
        return output_list


def _readout_offsets(params, prompt_len: int) -> list[int]:
    offsets = params.readout_offsets
    return [prompt_len - 1] if offsets is None else offsets


def get_tok_pooling_method(pooling_type: TokenPoolingType | str):
    if pooling_type == "ALL":
        return AllPool()
    if pooling_type == "STEP":
        return StepPool()
    if pooling_type == "GATHER":
        return GatherPool()

    raise NotImplementedError(f"Unknown tokenwise pooling type: {pooling_type!r}")
