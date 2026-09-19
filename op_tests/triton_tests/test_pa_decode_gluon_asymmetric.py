# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math

import pytest
import torch

import aiter
from aiter import per_tensor_quant
from aiter.ops.triton.gluon.pa_decode_gluon import pa_decode_gluon
import aiter.ops.triton.utils._triton.arch_info as arch_info


QK_HEAD_SIZE = 192
VALUE_HEAD_SIZE = 128
BLOCK_SIZE = 64
CONTEXT_LENGTH = 512
CONTEXT_PARTITION_SIZE = 256
NUM_QUERY_HEADS = 16
NUM_KV_HEADS = 1


def _make_vectorized_caches(cache_dtype: torch.dtype):
    batch_size = 2
    blocks_per_sequence = CONTEXT_LENGTH // BLOCK_SIZE
    num_blocks = batch_size * blocks_per_sequence

    key = (
        torch.randn(
            num_blocks,
            BLOCK_SIZE,
            NUM_KV_HEADS,
            QK_HEAD_SIZE,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.2
    )
    value = (
        torch.randn(
            num_blocks,
            BLOCK_SIZE,
            NUM_KV_HEADS,
            VALUE_HEAD_SIZE,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.2
    )

    elements_per_vector = 16 // cache_dtype.itemsize
    key_cache = (
        key.view(
            num_blocks,
            BLOCK_SIZE,
            NUM_KV_HEADS,
            QK_HEAD_SIZE // elements_per_vector,
            elements_per_vector,
        )
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    value_cache = (
        value.permute(0, 2, 3, 1)
        .contiguous()
        .view(
            num_blocks,
            NUM_KV_HEADS,
            VALUE_HEAD_SIZE,
            BLOCK_SIZE // elements_per_vector,
            elements_per_vector,
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )

    if cache_dtype == aiter.dtypes.fp8:
        key_cache, key_scale = per_tensor_quant(
            key_cache, quant_dtype=aiter.dtypes.fp8
        )
        value_cache, value_scale = per_tensor_quant(
            value_cache, quant_dtype=aiter.dtypes.fp8
        )
        key = (
            key_cache.permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, QK_HEAD_SIZE)
            .float()
            * key_scale.float()
        )
        value = (
            value_cache.permute(0, 2, 4, 1, 3)
            .contiguous()
            .view(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, VALUE_HEAD_SIZE)
            .float()
            * value_scale.float()
        )
    else:
        key_scale = None
        value_scale = None
        key = key.float()
        value = value.float()

    block_tables = torch.arange(
        num_blocks, device="cuda", dtype=torch.int32
    ).view(batch_size, blocks_per_sequence)
    return key_cache, value_cache, key, value, key_scale, value_scale, block_tables


def _reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_tables: torch.Tensor,
    query_length: int,
    sliding_window: int,
    sinks: torch.Tensor | None,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(QK_HEAD_SIZE)
    key_positions = torch.arange(CONTEXT_LENGTH, device="cuda")
    query_positions = torch.arange(query_length, device="cuda")
    upper_bound = CONTEXT_LENGTH - query_length + query_positions

    mask = key_positions[None, :] <= upper_bound[:, None]
    if sliding_window:
        # Match the decode kernel's combined causal/sliding-window frontier.
        lower_bound = CONTEXT_LENGTH - sliding_window + 1 + query_positions
        mask &= key_positions[None, :] >= lower_bound[:, None]

    outputs = []
    batch_size = block_tables.shape[0]
    for batch_idx in range(batch_size):
        block_ids = block_tables[batch_idx].long()
        sequence_key = key[block_ids].reshape(
            -1, NUM_KV_HEADS, QK_HEAD_SIZE
        )[:CONTEXT_LENGTH, 0]
        sequence_value = value[block_ids].reshape(
            -1, NUM_KV_HEADS, VALUE_HEAD_SIZE
        )[:CONTEXT_LENGTH, 0]
        sequence_query = query[
            batch_idx * query_length : (batch_idx + 1) * query_length
        ].float()

        logits = (
            torch.einsum("qhd,kd->hqk", sequence_query, sequence_key) * scale
        )
        logits = logits.masked_fill(~mask[None, :, :], -3.4e38)
        logits_max = logits.max(dim=-1, keepdim=True).values
        unnormalized = torch.exp(logits - logits_max)
        denominator = unnormalized.sum(dim=-1, keepdim=True)
        if sinks is not None:
            denominator += torch.exp(sinks.float()[:, None, None] - logits_max)
        probabilities = unnormalized / denominator
        outputs.append(
            torch.einsum("hqk,kd->qhd", probabilities, sequence_value)
        )

    return torch.cat(outputs).to(torch.bfloat16)


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, aiter.dtypes.fp8])
@pytest.mark.parametrize("query_length", [1, 4])
@pytest.mark.parametrize("sliding_window", [0, 128])
def test_pa_decode_qk192_v128_vectorized_5d(
    cache_dtype: torch.dtype, query_length: int, sliding_window: int
):
    if arch_info.get_arch() not in ("gfx942", "gfx950"):
        pytest.skip("asymmetric Gluon paged decode supports gfx942/gfx950")

    torch.manual_seed(7)
    batch_size = 2
    query = (
        torch.randn(
            batch_size * query_length,
            NUM_QUERY_HEADS,
            QK_HEAD_SIZE,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.2
    )
    (
        key_cache,
        value_cache,
        key,
        value,
        key_scale,
        value_scale,
        block_tables,
    ) = _make_vectorized_caches(cache_dtype)
    context_lengths = torch.full(
        (batch_size,), CONTEXT_LENGTH, device="cuda", dtype=torch.int32
    )
    sinks = None
    if sliding_window:
        sinks = torch.linspace(
            -1.0,
            -0.25,
            NUM_QUERY_HEADS,
            device="cuda",
            dtype=torch.bfloat16,
        )

    output = torch.empty(
        batch_size * query_length,
        NUM_QUERY_HEADS,
        VALUE_HEAD_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    pa_decode_gluon(
        output,
        query,
        key_cache,
        value_cache,
        context_lengths,
        block_tables,
        1.0 / math.sqrt(QK_HEAD_SIZE),
        query_length,
        max_context_partition_num=2,
        context_partition_size=CONTEXT_PARTITION_SIZE,
        compute_type=torch.bfloat16,
        key_scale=key_scale,
        value_scale=value_scale,
        sinks=sinks,
        sliding_window=sliding_window,
        ps=True,
    )

    expected = _reference(
        query,
        key,
        value,
        block_tables,
        query_length,
        sliding_window,
        sinks,
    )
    torch.testing.assert_close(output, expected, atol=6e-3, rtol=6e-3)
