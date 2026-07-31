# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn import functional as F

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    torch_reference_chunk_gated_delta_rule,
    torch_reference_post_conv_prep,
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


def test_torch_reference_post_conv_prep_matches_transformers_order() -> None:
    torch.manual_seed(1)
    tokens = 5
    num_k_heads = 2
    num_v_heads = 4
    head_k_dim = 3
    head_v_dim = 6
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    conv_output = torch.randn(tokens, key_dim * 2 + value_dim)
    a = torch.randn(tokens, num_v_heads)
    b = torch.randn(tokens, num_v_heads)
    A_log = torch.randn(num_v_heads)
    dt_bias = torch.randn(num_v_heads)

    query, key, value, g, beta = torch_reference_post_conv_prep(
        conv_output,
        a,
        b,
        A_log,
        dt_bias,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    )

    expected_query, expected_key, expected_value = torch.split(
        conv_output,
        [key_dim, key_dim, value_dim],
        dim=-1,
    )
    expected_query = expected_query.reshape(
        1, tokens, num_k_heads, head_k_dim
    ).repeat_interleave(num_v_heads // num_k_heads, dim=2)
    expected_key = expected_key.reshape(
        1, tokens, num_k_heads, head_k_dim
    ).repeat_interleave(num_v_heads // num_k_heads, dim=2)
    expected_value = expected_value.reshape(1, tokens, num_v_heads, head_v_dim)

    torch.testing.assert_close(query, expected_query)
    torch.testing.assert_close(key, expected_key)
    torch.testing.assert_close(value, expected_value)
    torch.testing.assert_close(beta, b.sigmoid().unsqueeze(0))
    torch.testing.assert_close(
        g,
        (-A_log.float().exp() * F.softplus(a.float() + dt_bias)).unsqueeze(0),
    )
