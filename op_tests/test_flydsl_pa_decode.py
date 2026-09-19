# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Full-reference coverage for AITER's opt-in FlyDSL PA decode API."""

import random
from typing import List, Optional, Tuple, Union

import pytest
import torch
import triton

pytest.importorskip("flydsl")
from aiter import dtypes, per_tensor_quant
from aiter.ops.flydsl.pa_decode import pa_decode_ps_launch as flydsl_ps_launch

HAS_FLYDSL_PS = True
UNIFORM_RANGE = (-1, 1)
STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
}
_ARCH = (
    torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if torch.cuda.is_available()
    else ""
)
_requires_tile_pa = pytest.mark.skipif(
    _ARCH not in ("gfx942", "gfx950"),
    reason="PA decode requires gfx942 or gfx950",
)


def setup_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_kv_cache_torch_dtype(
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
) -> torch.dtype:
    if isinstance(cache_dtype, str):
        if cache_dtype == "auto":
            if isinstance(model_dtype, str):
                return STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            if isinstance(model_dtype, torch.dtype):
                return model_dtype
            raise ValueError(f"Invalid model dtype: {model_dtype}")
        if cache_dtype in ["half", "bfloat16", "float"]:
            return STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        if cache_dtype == "fp8":
            return torch.uint8
        raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    if isinstance(cache_dtype, torch.dtype):
        return cache_dtype
    raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")


def create_kv_cache(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: int = 0,
    device: Optional[str] = "cuda",
    itemsize: int = 1,
    value_head_size: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(f"Does not support fp8 key cache with head_size={head_size}")
    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)
    elements_per_vector = 16 // itemsize
    value_head_size = head_size if value_head_size is None else value_head_size
    key_cache_shape = (
        num_blocks,
        num_heads,
        head_size // elements_per_vector,
        block_size,
        elements_per_vector,
    )
    value_cache_shape = (num_blocks, num_heads, value_head_size, block_size)
    key_caches: List[torch.Tensor] = []
    value_caches: List[torch.Tensor] = []
    setup_seed(seed)
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape, dtype=torch_dtype, device=device)
        value_cache = torch.empty(
            size=value_cache_shape, dtype=torch_dtype, device=device
        )
        key_cache.uniform_(*UNIFORM_RANGE)
        value_cache.uniform_(*UNIFORM_RANGE)
        key_caches.append(key_cache)
        value_caches.append(value_cache)
    return key_caches, value_caches


def reference_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
    output_dtype: torch.dtype,
    is_causal: bool = True,
    sliding_window=0,
) -> torch.Tensor:
    """Reference implementation of masked attention."""
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    num_query_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    s_q = query.shape[0]
    s_k = key.shape[0]
    key = key.repeat_interleave(num_query_heads // num_kv_heads, dim=1)
    value = value.repeat_interleave(num_query_heads // num_kv_heads, dim=1)

    attention_weights = torch.einsum("qhd,khd->hqk", query, key) * softmax_scale

    if is_causal:
        query_len = query.shape[0]
        key_len = key.shape[0]
        attention_bias = torch.zeros(
            query_len, key_len, dtype=torch.float32, device=query.device
        )
        causal_mask = torch.ones(
            query_len, key_len, dtype=torch.bool, device=query.device
        ).tril(diagonal=key_len - query_len)
        # attention_bias.masked_fill_(causal_mask.logical_not(), float(-3.4e38))
        attention_bias.masked_fill_(causal_mask.logical_not(), float(-3.4e38))
        attention_weights += attention_bias

    # Handle position calculation for both context and generation phases
    if s_q == s_k:
        # Context phase: standard position calculation
        query_positions = torch.arange(s_q, device=query.device)
        key_positions = torch.arange(s_k, device=query.device)
    else:
        # Generation phase: query is at position s_k (after the cache)
        query_positions = torch.arange(
            s_k - s_q, s_k, device=query.device
        )  # [s_k] for s_q=1
        key_positions = torch.arange(s_k, device=query.device)  # [0,1,2,...,s_k-1]

    # Create position difference matrix: query_pos - key_pos
    pos_diff = query_positions.unsqueeze(1) - key_positions.unsqueeze(0)  # [s_q, s_k]

    # Fallback: initialize the mask to all True, then progressively tighten with AND
    window_mask = torch.ones_like(attention_weights, dtype=torch.bool)
    if sliding_window > 0:
        # Sliding window mask: allow attention only if 0 <= pos_diff < sliding_window_size
        # sliding window size does not cover the diagonals
        sliding_window_mask = pos_diff >= sliding_window + 1
        window_mask &= sliding_window_mask

    if sliding_window > 0:
        attention_weights.masked_fill_(window_mask, float("-inf"))
    # torch.save(attention_weights, "/data00/fengjunda.aml/debug/attention_weights.pt")

    attention_weights = torch.softmax(attention_weights, dim=-1)
    output = torch.einsum("hqk,khd->qhd", attention_weights, value)
    return output.to(output_dtype)


def torch_mha_extend(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    query_output_indptr: torch.Tensor,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    sliding_window=0,
) -> torch.Tensor:
    """PyTorch reference implementation of paged attention."""
    num_blocks, num_heads, value_head_size, block_size = value_cache.shape
    key_head_size = key_cache.shape[2] * key_cache.shape[4]
    softmax_scale = 1.0 / (key_head_size**0.5)

    output_dtype = query.dtype
    kv_dtype = key_cache.dtype

    queries_split = torch.tensor_split(query, query_output_indptr.tolist()[1:])
    key_cache_flat = (
        key_cache.permute(0, 3, 1, 2, 4).contiguous().view(-1, num_heads, key_head_size)
    )
    value_cache_flat = (
        value_cache.permute(0, 3, 1, 2)
        .contiguous()
        .view(-1, num_heads, value_head_size)
    )

    batch_size = query_output_indptr.shape[0] - 1
    outputs = []

    for batch_idx in range(batch_size):
        current_query = queries_split[batch_idx]
        current_block_table = block_tables[batch_idx]
        current_context_length = context_lengths[batch_idx].item()

        token_indices = (
            current_block_table.repeat_interleave(block_size)[:current_context_length]
            * block_size
            + torch.arange(current_context_length, device=current_block_table.device)
            % block_size
        )

        gathered_keys = (
            key_cache_flat.view(torch.int8)[token_indices]
            .view(kv_dtype)
            .to(torch.float)
        )
        if key_scale is not None:
            gathered_keys *= key_scale[:, token_indices].t().unsqueeze(-1)

        gathered_values = (
            value_cache_flat.view(torch.int8)[token_indices]
            .view(kv_dtype)
            .to(torch.float)
        )
        if value_scale is not None:
            gathered_values *= value_scale[:, token_indices].t().unsqueeze(-1)

        attention_output = reference_masked_attention(
            current_query,
            gathered_keys,
            gathered_values,
            softmax_scale,
            output_dtype,
            is_causal=True,
            sliding_window=sliding_window,
        )
        outputs.append(attention_output)

    return torch.cat(outputs)


def quantize_kv_cache_per_tensor(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    quant_dtype: torch.dtype,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    num_blocks, num_heads, _, block_size = value_cache.shape
    key_head_dim = key_cache.shape[2] * key_cache.shape[4]
    elements_per_vector = 16 // quant_dtype.itemsize
    key_cache_reshaped = (
        key_cache.permute(0, 1, 3, 2, 4)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )
    key_cache_reshaped = (
        key_cache_reshaped.view(
            num_blocks,
            num_heads,
            block_size,
            key_head_dim // elements_per_vector,
            elements_per_vector,
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    quantized_keys, key_scales_original = per_tensor_quant(
        key_cache_reshaped, quant_dtype=quant_dtype
    )
    quantized_values, value_scales_original = per_tensor_quant(
        value_cache, quant_dtype=quant_dtype
    )
    key_scales_flat = key_scales_original.expand(num_heads, num_blocks * block_size)
    value_scales_flat = value_scales_original.expand(num_heads, num_blocks * block_size)
    return (
        quantized_keys,
        key_scales_flat,
        quantized_values,
        value_scales_flat,
        key_scales_original,
        value_scales_original,
    )


def shuffle_value_cache_layout(value_cache: torch.Tensor) -> torch.Tensor:
    elements_per_vector = 16 // value_cache.element_size()
    num_blocks, num_kv_heads, head_size, block_size = value_cache.shape
    value_cache_reshaped = value_cache.view(
        num_blocks,
        num_kv_heads,
        head_size,
        block_size // elements_per_vector,
        elements_per_vector,
    )
    return value_cache_reshaped.permute(0, 1, 3, 2, 4).contiguous()


@_requires_tile_pa
@pytest.mark.parametrize(
    ("compute_type", "query_length", "value_head_size", "num_partitions"),
    [
        pytest.param("bf16", 1, 128, 4, id="bf16-qlen1-v128"),
        pytest.param("bf16", 1, 192, 4, id="bf16-qlen1-v192"),
        pytest.param("bf16", 4, 128, 4, id="bf16-qlen4-v128"),
        pytest.param("bf16", 4, 192, 4, id="bf16-qlen4-v192"),
        pytest.param("fp8", 4, 128, 4, id="fp8-qlen4-v128"),
        pytest.param("fp8", 4, 192, 4, id="fp8-qlen4-v192"),
        pytest.param("bf16", 8, 128, 1, id="bf16-qlen8-v128-one-shot"),
        pytest.param("bf16", 8, 128, 4, id="bf16-qlen8-v128-split"),
        pytest.param("fp8", 8, 128, 1, id="fp8-qlen8-v128-one-shot"),
        pytest.param("fp8", 8, 128, 4, id="fp8-qlen8-v128-split"),
    ],
)
@pytest.mark.parametrize("head_size", [128, 192])
@pytest.mark.parametrize("num_heads", [(16, 1)])
@pytest.mark.parametrize("context_length", [1027])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize(
    "entrypoint",
    ["direct", "ps-allocate", "ps-preallocated"],
    ids=["direct", "ps-allocate", "ps-preallocated"],
)
def test_tile_pa_vectorized_5d_matches_torch(
    compute_type: str,
    query_length: int,
    value_head_size: int,
    num_partitions: int,
    head_size: int,
    num_heads: Tuple[int, int],
    context_length: int,
    batch_size: int,
    block_size: int,
    entrypoint: str,
) -> None:
    from aiter.ops.flydsl.pa_decode import pa_decode_tile

    num_query_heads, num_kv_heads = num_heads
    blocks_per_sequence = triton.cdiv(context_length, block_size)
    total_blocks = batch_size * blocks_per_sequence
    device = torch.device("cuda:0")
    cache_dtype = dtypes.d_dtypes[compute_type]

    setup_seed(20260821)
    query = torch.empty(
        batch_size * query_length,
        num_query_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-0.5, 0.5)
    key_caches, value_caches = create_kv_cache(
        total_blocks,
        block_size,
        1,
        num_kv_heads,
        head_size,
        "auto",
        torch.bfloat16,
        seed=20260821,
        device=str(device),
        itemsize=1 if compute_type == "fp8" else cache_dtype.itemsize,
        value_head_size=value_head_size,
    )
    key_cache = key_caches[0]
    value_cache = value_caches[0]
    key_scale_flat = None
    value_scale_flat = None
    key_scale = None
    value_scale = None
    if compute_type == "fp8":
        (
            key_cache,
            key_scale_flat,
            value_cache,
            value_scale_flat,
            key_scale,
            value_scale,
        ) = quantize_kv_cache_per_tensor(
            key_cache,
            value_cache,
            quant_dtype=cache_dtype,
        )
    block_tables = torch.arange(
        total_blocks - 1, -1, -1, dtype=torch.int32, device=device
    ).reshape(batch_size, blocks_per_sequence)
    context_lengths = torch.full(
        (batch_size,), context_length, dtype=torch.int32, device=device
    )
    query_output_indptr = torch.arange(
        0,
        (batch_size + 1) * query_length,
        query_length,
        dtype=torch.int32,
        device=device,
    )
    expected = torch_mha_extend(
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        query_output_indptr,
        key_scale_flat,
        value_scale_flat,
    )
    partial_shape = (
        batch_size,
        num_kv_heads,
        num_partitions,
        query_length * (num_query_heads // num_kv_heads),
    )
    actual = torch.full(
        (batch_size * query_length, num_query_heads, value_head_size),
        float("nan"),
        dtype=query.dtype,
        device=device,
    )
    pmax = torch.empty(partial_shape, dtype=torch.float32, device=device)
    psum = torch.empty(partial_shape, dtype=torch.float32, device=device)
    pout = torch.empty(
        (*partial_shape, value_head_size), dtype=query.dtype, device=device
    )
    value_cache = shuffle_value_cache_layout(value_cache)
    if entrypoint == "direct":
        pa_decode_tile(
            output=actual,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            context_lengths=context_lengths,
            key_scale=key_scale,
            value_scale=value_scale,
            softmax_scale=head_size**-0.5,
            num_partitions=num_partitions,
            pmax=pmax,
            psum=psum,
            pout=pout,
        )
    else:
        if not HAS_FLYDSL_PS:
            pytest.skip("FlyDSL `pa_decode_ps_launch` is not available")
        kv_page_indices = block_tables.flatten()
        kv_indptr = torch.arange(
            0,
            (batch_size + 1) * blocks_per_sequence,
            blocks_per_sequence,
            dtype=torch.int32,
            device=device,
        )

        def launch_ps():
            return flydsl_ps_launch(
                output=actual,
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                context_lengths=context_lengths,
                kv_page_indices=kv_page_indices,
                kv_indptr=kv_indptr,
                softmax_scale=head_size**-0.5,
                key_scale=key_scale,
                value_scale=value_scale,
                block_tables=block_tables,
                max_context_partition_num=num_partitions,
                exp_sums=psum if entrypoint == "ps-preallocated" else None,
                max_logits=pmax if entrypoint == "ps-preallocated" else None,
                temporary_output=pout if entrypoint == "ps-preallocated" else None,
            )

        path = launch_ps()
        assert path == "ps_small_block"
        if (
            entrypoint == "ps-preallocated"
            and (query_length == 8 or (compute_type == "bf16" and query_length == 4))
            and value_head_size == 128
        ):
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                launch_ps()
            torch.cuda.current_stream().wait_stream(capture_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                launch_ps()
            actual.fill_(float("nan"))
            graph.replay()
    torch.cuda.synchronize()

    tolerance = 2.0e-2 if compute_type == "fp8" else 5.0e-3
    assert bool(torch.isfinite(actual).all().item())
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@_requires_tile_pa
@pytest.mark.parametrize(
    "compute_type,head_size,batch_size,num_heads,num_partitions,block_size",
    [
        ("bf16", 128, 9, (16, 1), 3, 16),
        ("bf16", 192, 17, (18, 2), 1, 64),
        ("fp8", 128, 9, (32, 2), 3, 16),
        ("fp8", 128, 17, (48, 2), 4, 64),
    ],
)
def test_tile_pa_qlen8_grouped_grid_tail(
    compute_type, head_size, batch_size, num_heads, num_partitions, block_size
):
    """Partial sequence blocks preserve query-group and KV-head addressing."""
    test_tile_pa_vectorized_5d_matches_torch(
        compute_type=compute_type,
        query_length=8,
        value_head_size=128,
        num_partitions=num_partitions,
        head_size=head_size,
        num_heads=num_heads,
        context_length=1031,
        batch_size=batch_size,
        block_size=block_size,
        entrypoint="ps-preallocated",
    )


@_requires_tile_pa
@pytest.mark.parametrize("num_partitions", [1, 4])
@pytest.mark.parametrize(
    "compute_type,query_length,num_heads",
    [
        ("bf16", 1, (16, 1)),
        ("fp8", 1, (16, 1)),
        ("bf16", 4, (16, 1)),
        ("fp8", 4, (16, 1)),
        ("bf16", 8, (16, 1)),
        ("fp8", 8, (16, 1)),
        ("bf16", 5, (16, 1)),  # Padded local M-tiles in the final group.
        ("bf16", 8, (17, 1)),  # Three groups and a partial final M-tile.
        ("bf16", 8, (32, 2)),  # Query groups must not become KV-head IDs.
    ],
)
@pytest.mark.parametrize("query_value", [0.0, -512.0], ids=["zero", "negative"])
def test_tile_pa_constant_query_causal_frontier(
    compute_type, num_partitions, query_length, num_heads, query_value
):
    """Constant logits must stay finite and see their own causal prefix."""
    from aiter.ops.flydsl.pa_decode import pa_decode_tile

    context_length = 9
    num_query_heads, num_kv_heads = num_heads
    cache_dtype = dtypes.d_dtypes[compute_type]
    vector = 16 // cache_dtype.itemsize
    query = torch.full(
        (query_length, num_query_heads, 192),
        query_value,
        dtype=torch.bfloat16,
        device="cuda",
    )
    key = torch.ones(
        (2, num_kv_heads, 192 // vector, 64, vector), dtype=cache_dtype, device="cuda"
    )
    # Constant scores make the visible values' arithmetic mean an independent oracle.
    # A nonzero physical page and large finite padding catch page/tail mistakes.
    value = torch.full(
        (2, num_kv_heads, 128, 64), 64.0, dtype=torch.bfloat16, device="cuda"
    )
    head_offsets = torch.arange(num_kv_heads, dtype=torch.float32, device="cuda") * 16
    value[1, :, :, :context_length] = head_offsets[:, None, None] + torch.arange(
        1, context_length + 1, device="cuda"
    )
    value = shuffle_value_cache_layout(value.to(cache_dtype))
    table = torch.tensor([[1]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([context_length], dtype=torch.int32, device="cuda")
    scale = (
        torch.ones(1, dtype=torch.float32, device="cuda")
        if compute_type == "fp8"
        else None
    )
    output = torch.full(
        (query_length, num_query_heads, 128),
        float("nan"),
        dtype=query.dtype,
        device="cuda",
    )

    pa_decode_tile(
        output,
        query,
        key,
        value,
        table,
        lengths,
        scale,
        scale,
        num_partitions=num_partitions,
    )
    visible = (
        context_length - query_length + torch.arange(1, query_length + 1, device="cuda")
    )
    expected = ((visible.float() + 1) / 2).view(query_length, 1, 1)
    expected = (
        expected
        + head_offsets.repeat_interleave(num_query_heads // num_kv_heads)[None, :, None]
    )
    expected = expected.expand_as(output)
    tolerance = 2.0e-2 if compute_type == "fp8" else 5.0e-3
    torch.testing.assert_close(output.float(), expected, rtol=tolerance, atol=tolerance)


@_requires_tile_pa
@pytest.mark.parametrize(
    ("compute_type", "block_size", "sliding_window"),
    [
        pytest.param("bf16", 1024, 0, id="bf16-page1024"),
        pytest.param("fp8", 64, 128, id="asymmetric-fp8-sliding-window"),
    ],
)
def test_pa_decode_ps_rejects_unsupported_bf16_asymmetric_paths(
    compute_type: str, block_size: int, sliding_window: int
) -> None:
    if not HAS_FLYDSL_PS:
        pytest.skip("FlyDSL `pa_decode_ps_launch` is not available")

    device = torch.device("cuda:0")
    head_size = 192
    value_head_size = 128
    num_query_heads = 16
    num_kv_heads = 1
    cache_dtype = dtypes.d_dtypes[compute_type]
    vector_width = 8 if compute_type == "bf16" else 16
    query = torch.empty(
        1, num_query_heads, head_size, dtype=torch.bfloat16, device=device
    )
    output = torch.empty(
        1, num_query_heads, value_head_size, dtype=query.dtype, device=device
    )
    key_cache = torch.empty(
        1,
        num_kv_heads,
        head_size // vector_width,
        block_size,
        vector_width,
        dtype=cache_dtype,
        device=device,
    )
    value_cache = torch.empty(
        1,
        num_kv_heads,
        block_size // vector_width,
        value_head_size,
        vector_width,
        dtype=cache_dtype,
        device=device,
    )
    scale = None
    if compute_type == "fp8":
        scale = torch.ones(1, dtype=torch.float32, device=device)

    with pytest.raises(
        ValueError,
        match="BF16 KV and asymmetric value dimensions currently require",
    ):
        flydsl_ps_launch(
            output=output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            context_lengths=torch.ones(1, dtype=torch.int32, device=device),
            kv_page_indices=torch.zeros(1, dtype=torch.int32, device=device),
            kv_indptr=torch.tensor([0, 1], dtype=torch.int32, device=device),
            softmax_scale=head_size**-0.5,
            key_scale=scale,
            value_scale=scale,
            sliding_window=sliding_window,
            block_tables=torch.zeros((1, 1), dtype=torch.int32, device=device),
            max_context_partition_num=1,
        )


@_requires_tile_pa
def test_pa_decode_ps_rejects_non_divisible_gqa_heads() -> None:
    if not HAS_FLYDSL_PS:
        pytest.skip("FlyDSL `pa_decode_ps_launch` is not available")

    device = torch.device("cuda:0")
    query = torch.empty(1, 3, 192, dtype=torch.bfloat16, device=device)
    output = torch.empty(1, 3, 128, dtype=query.dtype, device=device)
    key_cache = torch.empty(1, 2, 24, 64, 8, dtype=torch.bfloat16, device=device)
    value_cache = torch.empty(1, 2, 8, 128, 8, dtype=torch.bfloat16, device=device)

    with pytest.raises(ValueError, match="must be divisible"):
        flydsl_ps_launch(
            output=output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            context_lengths=torch.ones(1, dtype=torch.int32, device=device),
            kv_page_indices=torch.zeros(1, dtype=torch.int32, device=device),
            kv_indptr=torch.tensor([0, 1], dtype=torch.int32, device=device),
            softmax_scale=192**-0.5,
            block_tables=torch.zeros((1, 1), dtype=torch.int32, device=device),
            max_context_partition_num=1,
        )


@_requires_tile_pa
@pytest.mark.parametrize("compute_type", ["fp8"])
@pytest.mark.parametrize("num_partitions", [1])
@pytest.mark.parametrize("head_size", [192])
@pytest.mark.parametrize("num_heads", [(16, 1)])
@pytest.mark.parametrize("query_length", [1])
@pytest.mark.parametrize("context_length", [2])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("trans_v", [True])
@pytest.mark.parametrize("block_size", [64])
def test_fp8_cache_offset_above_2gib(
    compute_type: str,
    num_partitions: int,
    head_size: int,
    num_heads: Tuple[int, int],
    query_length: int,
    context_length: int,
    batch_size: int,
    trans_v: bool,
    block_size: int,
) -> None:
    assert trans_v
    num_query_heads, num_kv_heads = num_heads
    cache_dtype = dtypes.d_dtypes[compute_type]
    page_bytes = num_kv_heads * head_size * block_size * cache_dtype.itemsize
    high_page = triton.cdiv(2**31, page_bytes)
    total_blocks = high_page + 1
    device = torch.device("cuda:0")

    query = torch.ones(
        batch_size * query_length,
        num_query_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    key_cache = torch.empty(
        total_blocks,
        num_kv_heads,
        head_size // 16,
        block_size,
        16,
        dtype=cache_dtype,
        device=device,
    )
    value_cache = torch.empty(
        total_blocks,
        num_kv_heads,
        block_size // 16,
        head_size,
        16,
        dtype=cache_dtype,
        device=device,
    )
    # Rounded-up tiles use page 0 for bounded fallback loads. Keep both that
    # page and the selected high page finite so masked PV lanes cannot see NaNs.
    key_cache[0].zero_()
    value_cache[0].zero_()
    key_cache[high_page].zero_()
    value_cache[high_page].zero_()

    # Make the output sensitive to both selected K/V token addresses. If K
    # wraps to page 0, the opposing values average to zero.
    key_cache[high_page, 0, :, 0, :].fill_(-0.25)
    key_cache[high_page, 0, :, 1, :].fill_(0.25)
    value_cache[high_page, 0, 0, :, 0].fill_(-1.0)
    value_cache[high_page, 0, 0, :, 1].fill_(1.0)

    selected_keys = (
        key_cache[high_page]
        .permute(2, 0, 1, 3)
        .reshape(block_size, num_kv_heads, head_size)[:context_length]
    )
    selected_values = (
        value_cache[high_page]
        .permute(1, 3, 0, 2)
        .reshape(block_size, num_kv_heads, head_size)[:context_length]
    )
    expected = reference_masked_attention(
        query,
        selected_keys,
        selected_values,
        head_size**-0.5,
        query.dtype,
    )
    assert expected.float().abs().min().item() > 0.5

    block_tables = torch.full(
        (batch_size, 1), high_page, dtype=torch.int32, device=device
    )
    context_lengths = torch.full(
        (batch_size,), context_length, dtype=torch.int32, device=device
    )
    scale = torch.ones(1, dtype=torch.float32, device=device)
    actual = torch.full_like(query, float("nan"))
    from aiter.ops.flydsl.pa_decode import pa_decode_tile

    pa_decode_tile(
        output=actual,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        context_lengths=context_lengths,
        key_scale=scale,
        value_scale=scale,
        softmax_scale=head_size**-0.5,
        num_partitions=num_partitions,
    )
    torch.cuda.synchronize()

    assert bool(torch.isfinite(actual).all().item())
    torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=2.0e-2)
