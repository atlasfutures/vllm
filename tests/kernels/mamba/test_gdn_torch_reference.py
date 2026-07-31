# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    torch_reference_chunk_gated_delta_rule,
)


def test_torch_reference_preserves_packed_state_orientation() -> None:
    torch.manual_seed(0)
    lengths = (5, 7)
    heads, key_dim, value_dim = 2, 3, 4
    total = sum(lengths)
    q = torch.randn(1, total, heads, key_dim)
    k = torch.randn_like(q)
    v = torch.randn(1, total, heads, value_dim)
    g = -torch.rand(1, total, heads)
    beta = torch.sigmoid(torch.randn(1, total, heads))
    state_vk = torch.randn(len(lengths), heads, value_dim, key_dim)
    offsets = torch.tensor([0, lengths[0], total], dtype=torch.int32)

    output, final_state = torch_reference_chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state_vk,
        output_final_state=True,
        cu_seqlens=offsets,
    )

    expected_outputs = []
    expected_states = []
    for index, (start, end) in enumerate(zip(offsets, offsets[1:])):
        segment_output, segment_state = torch_reference_chunk_gated_delta_rule(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            initial_state=state_vk[index : index + 1],
            output_final_state=True,
        )
        expected_outputs.append(segment_output)
        expected_states.append(segment_state)

    torch.testing.assert_close(output, torch.cat(expected_outputs, dim=1))
    torch.testing.assert_close(final_state, torch.cat(expected_states))
