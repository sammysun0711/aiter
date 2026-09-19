# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Full-reference coverage for AITER's opt-in FlyDSL PA decode API."""

import random
import sys
from types import SimpleNamespace

import pytest
import torch
import triton

pytest.importorskip("flydsl")
from aiter import dtypes, per_tensor_quant
from aiter.ops.attention import pa_decode_flydsl
from aiter.ops.flydsl import flydsl_pa_decode_ps, flydsl_pa_decode_tile
from aiter.ops.flydsl import pa_decode as tile
from aiter.ops.flydsl.pa_decode import pa_decode_ps_launch as flydsl_ps_launch


def _custom_op_kwargs(args):
    return {
        "output": args["output"],
        "query": args["query"],
        "key_cache": args["key_cache"],
        "value_cache": args["value_cache"],
        "context_lengths": args["context_lengths"],
        "block_tables": args["block_tables"],
        "softmax_scale": args.get("softmax_scale", args["query"].shape[-1] ** -0.5),
        "query_length": args["query"].shape[0] // args["context_lengths"].shape[0],
        "max_context_partition_num": args.get("num_partitions") or 0,
        "compute_type": args["key_cache"].dtype,
        "key_scale": args["key_scale"],
        "value_scale": args["value_scale"],
        "max_logits": args.get("pmax"),
        "exp_sums": args.get("psum"),
        "temporary_output": args.get("pout"),
    }


def _registered_tile(
    output,
    query,
    key_cache,
    value_cache,
    block_tables,
    context_lengths,
    key_scale,
    value_scale,
    softmax_scale=None,
    *,
    num_partitions=None,
    pmax=None,
    psum=None,
    pout=None,
):
    torch.ops.aiter.pa_decode_flydsl(
        **_custom_op_kwargs(
            {
                "output": output,
                "query": query,
                "key_cache": key_cache,
                "value_cache": value_cache,
                "block_tables": block_tables,
                "context_lengths": context_lengths,
                "key_scale": key_scale,
                "value_scale": value_scale,
                "softmax_scale": (
                    query.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
                ),
                "num_partitions": num_partitions,
                "pmax": pmax,
                "psum": psum,
                "pout": pout,
            }
        )
    )


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
    cache_dtype: str | torch.dtype | None,
    model_dtype: str | torch.dtype | None = None,
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
    cache_dtype: str | torch.dtype | None,
    model_dtype: str | torch.dtype | None = None,
    seed: int = 0,
    device: str | None = "cuda",
    itemsize: int = 1,
    value_head_size: int | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
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
    key_caches: list[torch.Tensor] = []
    value_caches: list[torch.Tensor] = []
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
        attention_bias.masked_fill_(causal_mask.logical_not(), (-3.4e38))
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
    key_scale: torch.Tensor | None = None,
    value_scale: torch.Tensor | None = None,
    sliding_window=0,
) -> torch.Tensor:
    """PyTorch reference implementation of paged attention."""
    _num_blocks, num_heads, value_head_size, block_size = value_cache.shape
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
) -> tuple[
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
    num_heads: tuple[int, int],
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
    num_heads: tuple[int, int],
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


@_requires_tile_pa
class TestBF16Wave:
    @pytest.fixture
    def force_wave(self, monkeypatch):
        from flydsl.runtime.device import get_rocm_arch

        from aiter.ops.flydsl import pa_decode as tile
        from aiter.ops.flydsl.kernels.pa_decode_bf16_wave import (
            compile_pa_decode_bf16_wave,
        )

        if str(get_rocm_arch()).split(":", 1)[0] != "gfx950":
            pytest.skip("The async BF16 specialization requires gfx950")

        def compile_small_case(**kw):
            assert kw["query_length"] == 8 and kw["query_group_size"] == 16
            assert kw["kv_dtype"] == "bf16" and kw["block_size"] == 64
            assert kw["v_head_dim"] == 128
            return compile_pa_decode_bf16_wave(
                kw["head_dim"], kw["num_partitions"], kw["softmax_scale"]
            )

        # Production preserves the generic kernel for these deliberately small grids.
        monkeypatch.setattr(tile, "compile_pa_decode_tile", compile_small_case)

    @pytest.mark.parametrize("entrypoint", ["direct", "ps-allocate", "ps-preallocated"])
    @pytest.mark.parametrize(
        "head_dim,context,parts,heads",
        [
            (128, 8, 1, (16, 1)),
            (192, 9, 4, (32, 2)),
            (128, 63, 3, (32, 2)),
            (192, 64, 1, (16, 1)),
            (128, 65, 4, (16, 1)),
            (192, 127, 3, (32, 2)),
            (128, 129, 1, (32, 2)),
            (192, 1027, 8, (16, 1)),
        ],
    )
    def test_bf16_wave_reference(
        self, force_wave, head_dim, context, parts, heads, entrypoint
    ):
        test_tile_pa_vectorized_5d_matches_torch(
            compute_type="bf16",
            query_length=8,
            value_head_size=128,
            num_partitions=parts,
            head_size=head_dim,
            num_heads=heads,
            context_length=context,
            batch_size=3,
            block_size=64,
            entrypoint=entrypoint,
        )

    @pytest.mark.parametrize("parts", [1, 4])
    @pytest.mark.parametrize("heads", [(16, 1), (32, 2)])
    @pytest.mark.parametrize("query_value", [0.0, -512.0])
    def test_bf16_wave_constant_logits(self, force_wave, parts, heads, query_value):
        test_tile_pa_constant_query_causal_frontier(
            "bf16", parts, 8, heads, query_value
        )

    @pytest.mark.parametrize("head_dim", [128, 192])
    @pytest.mark.parametrize("parts", [1, 8])
    def test_bf16_wave_varlen_strides_and_replay(self, force_wave, head_dim, parts):
        from aiter.ops.flydsl.pa_decode import pa_decode_tile

        setup_seed(20260919)
        batch, qlen, qheads, kvheads = 3, 8, 32, 2
        query = torch.randn(
            batch * qlen, qheads * 2, head_dim, device="cuda", dtype=torch.bfloat16
        )[:, ::2]
        output_storage = torch.full(
            (batch * qlen, qheads * 2, 128),
            float("nan"),
            device="cuda",
            dtype=query.dtype,
        )
        output = output_storage[:, ::2]
        keys, values = create_kv_cache(
            10,
            64,
            1,
            kvheads,
            head_dim,
            "auto",
            torch.bfloat16,
            seed=20260919,
            device="cuda",
            itemsize=2,
            value_head_size=128,
        )
        table = torch.tensor(
            [[9, 8, 7], [6, 5, 4], [3, 2, 1]], device="cuda", dtype=torch.int32
        )
        lengths = torch.tensor([8, 65, 129], device="cuda", dtype=torch.int32)
        indptr = torch.arange(
            0, (batch + 1) * qlen, qlen, device="cuda", dtype=torch.int32
        )
        expected = torch_mha_extend(
            query, keys[0], values[0], table, lengths, indptr, None, None
        )
        value = shuffle_value_cache_layout(values[0])
        partial_shape = (batch, kvheads, parts, 128)
        pmax = torch.empty(partial_shape, device="cuda", dtype=torch.float32)
        psum = torch.empty_like(pmax)
        pout = torch.empty((*partial_shape, 128), device="cuda", dtype=query.dtype)

        def launch():
            pa_decode_tile(
                output,
                query,
                keys[0],
                value,
                table,
                lengths,
                None,
                None,
                num_partitions=parts,
                pmax=pmax,
                psum=psum,
                pout=pout,
            )

        launch()
        torch.testing.assert_close(output, expected, rtol=0.005, atol=0.005)
        first = output.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            launch()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            launch()
        for _ in range(3):
            output.fill_(float("nan"))
            graph.replay()
            torch.testing.assert_close(output, first, rtol=0, atol=0)

        # Replay must observe changed KV/query data and a moved causal frontier.
        query.mul_(0.5)
        keys[0].neg_()
        values[0].mul_(0.5)
        value.copy_(shuffle_value_cache_layout(values[0]))
        lengths.copy_(torch.tensor([9, 64, 128], device="cuda", dtype=lengths.dtype))
        expected = torch_mha_extend(
            query, keys[0], values[0], table, lengths, indptr, None, None
        )
        output.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0.005, atol=0.005)
        updated = output.clone()
        output.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(output, updated, rtol=0, atol=0)
        assert bool(torch.isnan(output_storage[:, 1::2]).all().item())

    def test_bf16_wave_page_offset_above_2gib(self, force_wave):
        from aiter.ops.flydsl.pa_decode import pa_decode_tile

        # Both K and V page bases exceed a signed 32-bit byte offset.
        page = (1 << 31) // (128 * 64 * 2) + 3
        key = torch.empty(
            (page + 1, 1, 192 // 8, 64, 8), device="cuda", dtype=torch.bfloat16
        )
        value = torch.empty(
            (page + 1, 1, 64 // 8, 128, 8), device="cuda", dtype=torch.bfloat16
        )
        key[page].fill_(1)
        value[page].fill_(2)
        query = torch.zeros((8, 16, 192), device="cuda", dtype=torch.bfloat16)
        output = torch.full(
            (8, 16, 128), float("nan"), device="cuda", dtype=query.dtype
        )
        table = torch.tensor([[page]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([9], device="cuda", dtype=torch.int32)
        for parts in (1, 4):
            pa_decode_tile(
                output,
                query,
                key,
                value,
                table,
                lengths,
                None,
                None,
                num_partitions=parts,
            )
            torch.testing.assert_close(
                output, torch.full_like(output, 2), rtol=0, atol=0
            )

    @pytest.mark.parametrize(
        "batch,parts,expect_wave", [(31, 8, False), (32, 8, True), (256, 1, True)]
    )
    def test_bf16_wave_dispatch(self, monkeypatch, batch, parts, expect_wave):
        from flydsl.runtime.device import get_rocm_arch

        from aiter.ops.flydsl import pa_decode as tile
        from aiter.ops.flydsl.kernels import pa_decode_bf16_wave as wave

        if str(get_rocm_arch()).split(":", 1)[0] != "gfx950":
            pytest.skip("The async BF16 specialization requires gfx950")
        selected = []
        original_wave = wave.compile_pa_decode_bf16_wave
        original_tile = tile.compile_pa_decode_tile

        def record_wave(*args, **kw):
            selected.append("wave")
            return original_wave(*args, **kw)

        def record_tile(*args, **kw):
            selected.append("tile")
            return original_tile(*args, **kw)

        monkeypatch.setattr(wave, "compile_pa_decode_bf16_wave", record_wave)
        monkeypatch.setattr(tile, "compile_pa_decode_tile", record_tile)
        test_tile_pa_vectorized_5d_matches_torch(
            compute_type="bf16",
            query_length=8,
            value_head_size=128,
            num_partitions=parts,
            head_size=192,
            num_heads=(16, 1),
            context_length=65,
            batch_size=batch,
            block_size=64,
            entrypoint="direct",
        )
        assert selected == ["wave" if expect_wave else "tile"]


@_requires_tile_pa
class TestFP8Wave:
    @pytest.fixture(scope="class", autouse=True)
    def fp8_wave(self):
        from flydsl.runtime.device import get_rocm_arch

        from aiter.ops.flydsl import pa_decode as tile

        if str(get_rocm_arch()).split(":", 1)[0] != "gfx950":
            pytest.skip("The FP8 wave specialization requires gfx950")
        return tile

    @pytest.mark.parametrize("entrypoint", ["direct", "ps-allocate", "ps-preallocated"])
    @pytest.mark.parametrize(
        "dim,context,parts,heads",
        [
            (128, 8, 1, (16, 1)),
            (192, 9, 4, (32, 2)),
            (128, 63, 3, (32, 2)),
            (192, 64, 1, (16, 1)),
            (128, 65, 4, (16, 1)),
            (192, 127, 3, (32, 2)),
            (128, 129, 1, (32, 2)),
            (192, 193, 1, (16, 1)),
            (128, 257, 4, (32, 2)),
            (192, 1027, 8, (16, 1)),
            (128, 2049, 1, (16, 1)),
            (192, 2049, 32, (32, 2)),
        ],
    )
    def test_fp8_wave_reference(self, dim, context, parts, heads, entrypoint):
        test_tile_pa_vectorized_5d_matches_torch(
            compute_type="fp8",
            query_length=8,
            value_head_size=128,
            num_partitions=parts,
            head_size=dim,
            num_heads=heads,
            context_length=context,
            batch_size=3,
            block_size=64,
            entrypoint=entrypoint,
        )

    @pytest.mark.parametrize("dim", [128, 192])
    @pytest.mark.parametrize("context", [65, 769])
    @pytest.mark.parametrize("batch,parts", [(9, 1), (17, 3), (64, 1), (65, 1)])
    def test_fp8_wave_grouped_grid_boundaries(self, dim, context, batch, parts):
        # Exercise partial eight-sequence groups and both sides of the small-grid gate.
        test_tile_pa_vectorized_5d_matches_torch(
            compute_type="fp8",
            query_length=8,
            value_head_size=128,
            num_partitions=parts,
            head_size=dim,
            num_heads=(16, 1),
            context_length=context,
            batch_size=batch,
            block_size=64,
            entrypoint="direct",
        )

    @pytest.mark.parametrize("parts", [1, 4])
    @pytest.mark.parametrize("heads", [(16, 1), (32, 2)])
    @pytest.mark.parametrize("query_value", [0.0, -512.0])
    def test_fp8_wave_constant_frontier(self, fp8_wave, parts, heads, query_value):
        qheads, kvheads = heads
        query = torch.full(
            (8, qheads, 192), query_value, device="cuda", dtype=torch.bfloat16
        )
        key = torch.ones(
            (2, kvheads, 12, 64, 16), device="cuda", dtype=torch.float8_e4m3fn
        )
        values = torch.full((2, kvheads, 128, 64), 64.0, device="cuda")
        offsets = torch.arange(kvheads, device="cuda") * 16
        values[1, :, :, :9] = offsets[:, None, None] + torch.arange(
            1, 10, device="cuda"
        )
        values = values.to(key.dtype)
        value = shuffle_value_cache_layout(values)
        table = torch.tensor([[1]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([9], device="cuda", dtype=torch.int32)
        scale = torch.ones(1, device="cuda")
        output = torch.empty((8, qheads, 128), device="cuda", dtype=query.dtype)
        fp8_wave.pa_decode_tile(
            output,
            query,
            key,
            value,
            table,
            lengths,
            scale,
            scale,
            num_partitions=parts,
        )
        # The oracle must average stored FP8 values, including rounding above 16.
        prefix = values[1, :, 0, :9].float().cumsum(-1)
        visible = torch.arange(2, 10, device="cuda")
        expected = (
            (prefix[:, visible - 1] / visible)
            .T.repeat_interleave(16, 1)[:, :, None]
            .expand_as(output)
        )
        torch.testing.assert_close(output.float(), expected, rtol=0.02, atol=0.02)

    @pytest.mark.parametrize("dim", [128, 192])
    @pytest.mark.parametrize("parts", [1, 4, 32])
    @pytest.mark.parametrize("zero_query", [False, True])
    @pytest.mark.parametrize(
        "contexts", [(8, 321, 1089), (511, 512, 513), (513, 769, 1089), (63, 255, 1027)]
    )
    def test_fp8_wave_page_signatures_and_mutated_graph(
        self, fp8_wave, dim, parts, zero_query, contexts
    ):
        setup_seed(20260919)
        batch, qlen, qheads, kvheads = 3, 8, 32, 2
        max_blocks = max((length + 63) // 64 for length in contexts)
        pages = batch * max_blocks + 1
        query = torch.randn(
            batch * qlen, qheads * 2, dim, device="cuda", dtype=torch.bfloat16
        )[:, ::2]
        if zero_query:
            query.zero_()
        storage = torch.full(
            (batch * qlen, qheads * 2, 128),
            float("nan"),
            device="cuda",
            dtype=query.dtype,
        )
        output = storage[:, ::2]
        key = torch.randn(pages, kvheads, dim // 16, 64, 16, device="cuda").to(
            torch.float8_e4m3fn
        )
        # Distinct values for pages, tokens, heads, and channels expose stale ring slots.
        pg = torch.arange(pages, device="cuda")[:, None, None, None]
        hd = torch.arange(kvheads, device="cuda")[None, :, None, None]
        ch = torch.arange(128, device="cuda")[None, None, :, None]
        tok = torch.arange(64, device="cuda")[None, None, None, :]
        values = ((pg * 5 + hd * 3 + ch + tok * 7) % 15 - 7).to(torch.float8_e4m3fn)
        value = shuffle_value_cache_layout(values)
        table = (
            torch.randperm(pages, device="cuda", dtype=torch.int64)[
                : batch * max_blocks
            ]
            .to(torch.int32)
            .reshape(batch, max_blocks)
        )
        lengths = torch.tensor(contexts, device="cuda", dtype=torch.int32)
        indptr = torch.arange(
            0, (batch + 1) * qlen, qlen, device="cuda", dtype=torch.int32
        )
        ks = torch.tensor([0.37], device="cuda")
        vs = torch.tensor([0.63], device="cuda")
        partial_shape = (batch, kvheads, parts, 128)
        pmax = torch.empty(partial_shape, device="cuda")
        psum = torch.empty_like(pmax)
        pout = torch.empty((*partial_shape, 128), device="cuda", dtype=query.dtype)

        def reference():
            return torch_mha_extend(
                query,
                key,
                values,
                table,
                lengths,
                indptr,
                ks.expand(kvheads, pages * 64),
                vs.expand(kvheads, pages * 64),
            )

        def launch():
            fp8_wave.pa_decode_tile(
                output,
                query,
                key,
                value,
                table,
                lengths,
                ks,
                vs,
                num_partitions=parts,
                pmax=pmax,
                psum=psum,
                pout=pout,
            )

        expected = reference()
        launch()
        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02)
        first = output.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            launch()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            launch()
        for _ in range(32):
            output.fill_(float("nan"))
            graph.replay()
            torch.testing.assert_close(output, first, rtol=0, atol=0)

        query.mul_(0.5)
        key.copy_((-key.float()).to(key.dtype))
        values.copy_((-values.float()).to(values.dtype))
        value.copy_(shuffle_value_cache_layout(values))
        table.copy_(table.flip(1))
        lengths.copy_(
            torch.tensor(
                [contexts[2], contexts[0], contexts[1]],
                device="cuda",
                dtype=lengths.dtype,
            )
        )
        ks.fill_(0.23)
        vs.fill_(0.71)
        expected = reference()
        output.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02)
        updated = output.clone()
        for _ in range(32):
            output.fill_(float("nan"))
            graph.replay()
            torch.testing.assert_close(output, updated, rtol=0, atol=0)
        assert bool(torch.isnan(storage[:, 1::2]).all().item())

    @pytest.mark.parametrize("dim", [128, 192])
    @pytest.mark.parametrize("parts", [1, 4])
    @pytest.mark.parametrize("context", [257, 769])
    def test_fp8_wave_high_pages(self, fp8_wave, dim, parts, context):
        high = (1 << 31) // (128 * 64) + 3
        key = torch.empty(
            (high + 1, 1, dim // 16, 64, 16), device="cuda", dtype=torch.float8_e4m3fn
        )
        value = torch.empty((high + 1, 1, 4, 128, 16), device="cuda", dtype=key.dtype)
        key[0].zero_()
        value[0].zero_()
        key[high - 1].fill_(-0.25)
        key[high].fill_(0.25)
        value[high - 1].fill_(-1)
        value[high].fill_(1)
        query = torch.ones((8, 16, dim), device="cuda", dtype=torch.bfloat16)
        output = torch.empty((8, 16, 128), device="cuda", dtype=query.dtype)
        table = torch.tensor(
            [[high - (page % 2) for page in range((context + 63) // 64)]],
            device="cuda",
            dtype=torch.int32,
        )
        lengths = torch.tensor([context], device="cuda", dtype=torch.int32)
        scale = torch.ones(1, device="cuda")
        selected_k = torch.cat(
            [key[p].permute(2, 0, 1, 3).reshape(64, 1, dim) for p in table[0].tolist()]
        )[:context]
        selected_v = torch.cat(
            [
                value[p].permute(1, 3, 0, 2).reshape(64, 1, 128)
                for p in table[0].tolist()
            ]
        )[:context]
        expected = reference_masked_attention(
            query,
            selected_k.float(),
            selected_v.float(),
            dim**-0.5,
            query.dtype,
            is_causal=True,
        )
        assert expected.float().abs().min().item() > 0.5
        fp8_wave.pa_decode_tile(
            output,
            query,
            key,
            value,
            table,
            lengths,
            scale,
            scale,
            num_partitions=parts,
        )
        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02)

    @pytest.mark.parametrize(
        "case,expected",
        [
            ("d128", "fp8-small"),
            ("d192", "fp8-small"),
            ("grid64", "fp8-small"),
            ("grid65", "fp8-wave"),
            ("qlen1", "generic"),
            ("qlen4", "generic"),
            ("gqa8", "generic"),
            ("page16", "generic"),
            ("d64", "generic"),
            ("v192", "generic"),
            ("fp16-query", "generic"),
            ("per-token", "generic"),
            ("plain-v", "generic"),
            ("zero-scale", "generic"),
            ("negative-scale", "generic"),
            ("gfx942", "generic"),
        ],
    )
    def test_fp8_wave_dispatch(self, monkeypatch, case, expected):
        from aiter.ops.flydsl import pa_decode as tile
        from aiter.ops.flydsl.kernels import pa_decode_fp8_small as small
        from aiter.ops.flydsl.kernels import pa_decode_fp8_wave as wave

        selected = []

        def record_wave(*args, **kwargs):
            selected.append("fp8-wave")
            return {"launch": object()}

        def record_small(*args, **kwargs):
            selected.append("fp8-small")
            return {"launch": object()}

        def record_generic(**kwargs):
            selected.append("generic")
            return {"launch": object()}

        monkeypatch.setattr(wave, "compile_pa_decode_fp8_wave", record_wave)
        monkeypatch.setattr(small, "compile_pa_decode_fp8_small", record_small)
        monkeypatch.setattr(tile, "compile_pa_decode_tile", record_generic)
        monkeypatch.setattr(tile, "_run_compiled", lambda *args: None)
        batch, qlen, heads, dim, vdim, page = 2, 8, 16, 128, 128, 64
        qdtype, kvdtype = torch.bfloat16, torch.float8_e4m3fn
        softmax_scale = None
        if case == "d192":
            dim = 192
        elif case == "grid64":
            batch = 64
        elif case == "grid65":
            batch = 65
        elif case == "qlen1":
            qlen = 1
        elif case == "qlen4":
            qlen = 4
        elif case == "gqa8":
            heads = 8
        elif case == "page16":
            page = 16
        elif case == "d64":
            dim = 64
        elif case == "v192":
            vdim = 192
        elif case == "fp16-query":
            qdtype = torch.float16
        elif case == "zero-scale":
            softmax_scale = 0.0
        elif case == "negative-scale":
            softmax_scale = -0.5
        elif case == "gfx942":
            kvdtype = torch.float8_e4m3fnuz
            monkeypatch.setattr(tile, "get_rocm_arch", lambda: "gfx942")

        query = torch.empty((batch * qlen, heads, dim), dtype=qdtype, device="cuda")
        output = torch.empty((batch * qlen, heads, vdim), dtype=qdtype, device="cuda")
        key = torch.empty((1, 1, dim // 16, page, 16), dtype=kvdtype, device="cuda")
        vshape = (
            (1, 1, vdim, page) if case == "plain-v" else (1, 1, page // 16, vdim, 16)
        )
        value = torch.empty(vshape, dtype=kvdtype, device="cuda")
        table = torch.empty((batch, 1), dtype=torch.int32, device="cuda")
        lengths = torch.empty((batch,), dtype=torch.int32, device="cuda")
        scale_shape = (1, 1, page) if case == "per-token" else (1,)
        scale = torch.empty(scale_shape, dtype=torch.float32, device="cuda")
        tile.pa_decode_tile(
            output,
            query,
            key,
            value,
            table,
            lengths,
            scale,
            scale,
            softmax_scale,
            num_partitions=1,
        )
        assert selected == [expected]


@_requires_tile_pa
class TestMetadata:
    @pytest.mark.parametrize("compute_type", ["bf16", "fp8"])
    @pytest.mark.parametrize("entrypoint", ["direct", "ps-preallocated"])
    @pytest.mark.parametrize(
        "invalid,match",
        [
            ("table_rows", "block_tables must have shape"),
            ("table_rank", "block_tables must have shape"),
            ("table_empty", "block_tables must have shape"),
            ("table_row_stride", "block_tables must be contiguous"),
            ("table_column_stride", "block_tables must be contiguous"),
            ("length_stride", "context_lengths must be contiguous"),
            ("length_rank", "context_lengths must be a non-empty 1D"),
            ("table_device", "block_tables must be on"),
            ("length_device", "context_lengths must be on"),
        ],
    )
    def test_tile_rejects_unsupported_metadata(
        self, monkeypatch, compute_type, entrypoint, invalid, match
    ):
        from aiter.ops.flydsl import pa_decode as tile
        from aiter.ops.flydsl.kernels import pa_decode_bf16_wave as wave
        from aiter.ops.flydsl.kernels import pa_decode_fp8_small as small
        from aiter.ops.flydsl.kernels import pa_decode_fp8_wave as fp8_wave

        def unexpected_compile(*args, **kwargs):
            pytest.fail("Invalid metadata reached kernel compilation or launch")

        monkeypatch.setattr(tile, "compile_pa_decode_tile", unexpected_compile)
        monkeypatch.setattr(wave, "compile_pa_decode_bf16_wave", unexpected_compile)
        monkeypatch.setattr(fp8_wave, "compile_pa_decode_fp8_wave", unexpected_compile)
        monkeypatch.setattr(small, "compile_pa_decode_fp8_small", unexpected_compile)
        monkeypatch.setattr(tile, "_run_compiled", unexpected_compile)
        batch, heads, dim, page, parts = 32, 16, 128, 64, 8
        dtype = dtypes.d_dtypes[compute_type]
        vector = 16 // dtype.itemsize
        query = torch.empty(
            (batch * 8, heads, dim), dtype=torch.bfloat16, device="cuda"
        )
        output = torch.empty_like(query)
        key = torch.empty(
            (batch * 2, 1, dim // vector, page, vector), dtype=dtype, device="cuda"
        )
        value = torch.empty(
            (batch * 2, 1, page // vector, dim, vector), dtype=dtype, device="cuda"
        )
        table = torch.zeros((batch, 2), dtype=torch.int32, device="cuda")
        lengths = torch.full((batch,), 65, dtype=torch.int32, device="cuda")
        scale = (
            torch.ones(1, dtype=torch.float32, device="cuda")
            if compute_type == "fp8"
            else None
        )
        partial_shape = (batch, 1, parts, 8 * heads)
        pmax = torch.empty(partial_shape, dtype=torch.float32, device="cuda")
        psum = torch.empty_like(pmax)
        pout = torch.empty((*partial_shape, dim), dtype=query.dtype, device="cuda")
        if invalid == "table_rows":
            table = torch.zeros((batch + 1, 2), dtype=table.dtype, device=table.device)
        elif invalid == "table_rank":
            table = table.unsqueeze(-1)
        elif invalid == "table_empty":
            table = table[:, :0]
        elif invalid == "table_row_stride":
            table = torch.zeros((batch, 3), dtype=table.dtype, device=table.device)[
                :, :2
            ]
        elif invalid == "table_column_stride":
            table = torch.zeros((batch, 4), dtype=table.dtype, device=table.device)[
                :, ::2
            ]
        elif invalid == "length_stride":
            lengths = torch.full(
                (batch * 2,), 65, dtype=lengths.dtype, device=lengths.device
            )[::2]
        elif invalid == "length_rank":
            lengths = lengths.unsqueeze(-1)
        elif invalid == "table_device":
            table = table.cpu()
        elif invalid == "length_device":
            lengths = lengths.cpu()

        with pytest.raises(ValueError, match=match):
            if entrypoint == "direct":
                tile.pa_decode_tile(
                    output,
                    query,
                    key,
                    value,
                    table,
                    lengths,
                    scale,
                    scale,
                    num_partitions=parts,
                )
            else:
                flydsl_ps_launch(
                    output=output,
                    query=query,
                    key_cache=key,
                    value_cache=value,
                    context_lengths=lengths,
                    kv_page_indices=torch.zeros(
                        batch * 2, dtype=torch.int32, device="cuda"
                    ),
                    kv_indptr=torch.arange(
                        0, (batch + 1) * 2, 2, dtype=torch.int32, device="cuda"
                    ),
                    softmax_scale=dim**-0.5,
                    key_scale=scale,
                    value_scale=scale,
                    block_tables=table,
                    max_context_partition_num=parts,
                    max_logits=pmax,
                    exp_sums=psum,
                    temporary_output=pout,
                )

    @pytest.mark.parametrize("compute_type", ["bf16", "fp8"])
    @pytest.mark.parametrize("entrypoint", ["direct", "ps-preallocated"])
    def test_tile_dense_metadata_storage_offsets(
        self, monkeypatch, compute_type, entrypoint
    ):
        original_reference = torch_mha_extend
        seen = []

        def reference(
            query, key, value, table, lengths, indptr, ks, vs, *args, **kwargs
        ):
            padded_table = torch.empty(
                (table.shape[0] + 1, table.shape[1]),
                dtype=table.dtype,
                device=table.device,
            )
            padded_table[1:].copy_(table)
            table.set_(padded_table[1:])
            padded_lengths = torch.empty(
                (lengths.numel() + 1,), dtype=lengths.dtype, device=lengths.device
            )
            padded_lengths[1:].copy_(lengths)
            lengths.set_(padded_lengths[1:])
            assert table.is_contiguous() and lengths.is_contiguous()
            assert table.storage_offset() > 0 and lengths.storage_offset() > 0
            seen.append(True)
            return original_reference(
                query, key, value, table, lengths, indptr, ks, vs, *args, **kwargs
            )

        monkeypatch.setattr(sys.modules[__name__], "torch_mha_extend", reference)
        test_tile_pa_vectorized_5d_matches_torch(
            compute_type=compute_type,
            query_length=8,
            value_head_size=128,
            num_partitions=8,
            head_size=128,
            num_heads=(16, 1),
            context_length=65,
            batch_size=32,
            block_size=64,
            entrypoint=entrypoint,
        )
        assert seen == [True]


def make_case(dtype="bf16", parts=4):
    kv_dtype = torch.bfloat16 if dtype == "bf16" else torch.float8_e4m3fn
    vector = 16 // kv_dtype.itemsize
    shape = (2, 1, parts, 128)
    scale = torch.ones(1, device="cuda") if dtype == "fp8" else None
    return {
        "output": torch.empty((16, 16, 128), device="cuda", dtype=torch.bfloat16),
        "query": torch.zeros((16, 16, 128), device="cuda", dtype=torch.bfloat16),
        "key_cache": torch.zeros(
            (2, 1, 128 // vector, 64, vector), device="cuda", dtype=kv_dtype
        ),
        "value_cache": torch.ones(
            (2, 1, 64 // vector, 128, vector), device="cuda", dtype=kv_dtype
        ),
        "block_tables": torch.tensor([[0], [1]], device="cuda", dtype=torch.int32),
        "context_lengths": torch.full((2,), 8, device="cuda", dtype=torch.int32),
        "key_scale": scale,
        "value_scale": scale,
        "num_partitions": parts,
        "pmax": torch.empty(shape, device="cuda"),
        "psum": torch.empty(shape, device="cuda"),
        "pout": torch.empty((*shape, 128), device="cuda", dtype=torch.bfloat16),
    }


@_requires_tile_pa
class TestAPI:
    def test_public_exports(self):
        assert flydsl_pa_decode_tile is tile.pa_decode_tile
        assert flydsl_pa_decode_ps is tile.pa_decode_ps_launch

    @pytest.mark.parametrize(
        "name", ["key_cache", "value_cache", "output", "pmax", "psum", "pout"]
    )
    def test_rejects_wrong_device_before_compile(self, monkeypatch, name):
        args = make_case()
        args[name] = args[name].cpu()
        monkeypatch.setattr(
            tile,
            "compile_pa_decode_tile",
            lambda **kw: pytest.fail("unexpected compilation"),
        )
        with pytest.raises(ValueError, match=name):
            flydsl_pa_decode_tile(**args)

    @pytest.mark.parametrize(
        "name", ["key_cache", "value_cache", "pmax", "psum", "pout"]
    )
    def test_rejects_noncontiguous_storage(self, monkeypatch, name):
        args = make_case()
        original = args[name]
        args[name] = torch.empty(
            (*original.shape[:-1], original.shape[-1] * 2),
            device="cuda",
            dtype=original.dtype,
        )[..., ::2]
        monkeypatch.setattr(
            tile,
            "compile_pa_decode_tile",
            lambda **kw: pytest.fail("unexpected compilation"),
        )
        with pytest.raises(ValueError, match=name):
            flydsl_pa_decode_tile(**args)

    @pytest.mark.parametrize("parts", [-1, 1.5, True])
    def test_rejects_invalid_partition_count(self, parts):
        args = make_case()
        args["num_partitions"] = parts
        with pytest.raises(ValueError, match="num_partitions"):
            flydsl_pa_decode_tile(**args)

    @pytest.mark.parametrize("name", ["pmax", "psum", "pout"])
    def test_rejects_partial_workspace_set(self, name):
        args = make_case()
        args[name] = None
        with pytest.raises(ValueError, match="provide all"):
            flydsl_pa_decode_tile(**args)

    @pytest.mark.parametrize("name", ["pmax", "psum", "pout"])
    def test_rejects_wrong_workspace_dtype(self, name):
        args = make_case()
        args[name] = args[name].to(torch.float64)
        with pytest.raises(ValueError, match=name):
            flydsl_pa_decode_tile(**args)

    @pytest.mark.parametrize("name", ["key_scale", "value_scale"])
    def test_rejects_multielement_scalar_scale(self, name):
        args = make_case("fp8")
        args[name] = torch.ones(2, device="cuda")
        with pytest.raises(ValueError, match="one element"):
            flydsl_pa_decode_tile(**args)

    @pytest.mark.parametrize("dtype", ["bf16", "fp8"])
    def test_np1_preallocated_dummy(self, dtype):
        args = make_case(dtype, 1)
        args["psum"] = args["pout"] = None
        flydsl_pa_decode_tile(**args)
        torch.testing.assert_close(
            args["output"], torch.ones_like(args["output"]), rtol=0, atol=0
        )

    @pytest.mark.parametrize("dtype", ["bf16", "fp8"])
    def test_explicit_stream_and_replay(self, dtype):
        args = make_case(dtype)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        args["stream"] = stream
        flydsl_pa_decode_tile(**args)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            flydsl_pa_decode_tile(**args)
        args["output"].zero_()
        graph.replay()
        torch.testing.assert_close(
            args["output"], torch.ones_like(args["output"]), rtol=0, atol=0
        )

    def test_capture_rejects_python_scales_before_compile(self, monkeypatch):
        args = make_case("fp8")
        args["key_scale"] = args["value_scale"] = 1.0
        monkeypatch.setattr(tile, "_is_current_stream_capturing", lambda: True)
        with pytest.raises(ValueError, match="pre-created FP8 scale tensors"):
            flydsl_pa_decode_tile(**args)

    @pytest.mark.parametrize("kind", ["python", "cpu", "dtype"])
    @pytest.mark.parametrize("entrypoint", ["direct", "ps"])
    def test_explicit_stream_rejects_scale_conversion(self, kind, entrypoint):
        args = make_case("fp8")
        args["stream"] = torch.cuda.Stream()
        if kind == "python":
            args["key_scale"] = 1.0
        elif kind == "cpu":
            args["key_scale"] = args["key_scale"].cpu()
        else:
            args["key_scale"] = args["key_scale"].to(torch.float16)
        with pytest.raises(
            ValueError, match="Explicit-stream PA decode requires pre-created"
        ):
            if entrypoint == "direct":
                flydsl_pa_decode_tile(**args)
            else:
                parts = args.pop("num_partitions")
                pmax, psum, pout = (args.pop(name) for name in ("pmax", "psum", "pout"))
                flydsl_pa_decode_ps(
                    **args,
                    kv_page_indices=None,
                    kv_indptr=None,
                    softmax_scale=128**-0.5,
                    max_context_partition_num=parts,
                    max_logits=pmax,
                    exp_sums=psum,
                    temporary_output=pout,
                )

    @pytest.mark.parametrize("dtype", ["bf16", "fp8"])
    def test_automatic_splits_remain_capped(self, dtype):
        args = make_case(dtype)
        args["num_partitions"] = None
        args["pmax"] = args["psum"] = args["pout"] = None
        assert 4 <= tile.get_recommended_splits(2, 1, 4) <= 8
        flydsl_pa_decode_tile(**args)
        torch.testing.assert_close(
            args["output"], torch.ones_like(args["output"]), rtol=0, atol=0
        )

    @pytest.mark.parametrize("context", [8, 65, 511, 512, 513])
    @pytest.mark.parametrize("dim", [128, 192])
    def test_throughput_grid_fp8_short_context(self, dim, context):
        test_tile_pa_vectorized_5d_matches_torch(
            compute_type="fp8",
            query_length=8,
            value_head_size=128,
            num_partitions=8,
            head_size=dim,
            num_heads=(16, 1),
            context_length=context,
            batch_size=32,
            block_size=64,
            entrypoint="ps-preallocated",
        )

    @pytest.mark.parametrize("dim", [128, 192])
    @pytest.mark.parametrize("page", [16, 64])
    @pytest.mark.parametrize("query_dtype", [torch.bfloat16, torch.float16])
    def test_fp8_per_token_and_4d_value_fallback(self, dim, page, query_dtype):
        setup_seed(20260919)
        batch, context, heads, kv_heads, qlen, parts = 2, 1027, 32, 2, 8, 4
        blocks = (context + page - 1) // page
        pages = batch * blocks
        query = torch.empty(
            (batch * qlen, heads, dim), device="cuda", dtype=query_dtype
        ).uniform_(-0.5, 0.5)
        key = (
            torch.empty((pages, kv_heads, dim // 16, page, 16), device="cuda")
            .uniform_(-1, 1)
            .to(torch.float8_e4m3fn)
        )
        value = (
            torch.empty((pages, kv_heads, 128, page), device="cuda")
            .uniform_(-1, 1)
            .to(key.dtype)
        )
        ks = torch.empty((pages, kv_heads, page), device="cuda").uniform_(0.5, 1.0)
        vs = torch.empty_like(ks).uniform_(0.25, 0.75)
        table = torch.arange(
            pages - 1, -1, -1, device="cuda", dtype=torch.int32
        ).reshape(batch, blocks)
        lengths = torch.full((batch,), context, device="cuda", dtype=torch.int32)
        indptr = torch.arange(
            0, (batch + 1) * qlen, qlen, device="cuda", dtype=torch.int32
        )
        expected = torch_mha_extend(
            query,
            key,
            value,
            table,
            lengths,
            indptr,
            ks.permute(1, 0, 2).reshape(kv_heads, -1),
            vs.permute(1, 0, 2).reshape(kv_heads, -1),
        )
        output = torch.empty_like(expected)
        flydsl_pa_decode_tile(
            output, query, key, value, table, lengths, ks, vs, num_partitions=parts
        )
        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02)

    @pytest.mark.parametrize("dtype", ["bf16", "fp8"])
    @pytest.mark.parametrize("dim", [128, 192])
    @pytest.mark.parametrize("parts", [3, 65])
    def test_reducer_partition_boundaries(self, dtype, dim, parts):
        """Cover non-power-of-two splits and the reducer's multi-wave branch."""
        test_tile_pa_vectorized_5d_matches_torch(
            compute_type=dtype,
            query_length=8,
            value_head_size=128,
            num_partitions=parts,
            head_size=dim,
            num_heads=(32, 2),
            context_length=129,
            batch_size=2,
            block_size=64,
            entrypoint="ps-preallocated",
        )


class TestStaticPlan:
    @pytest.mark.parametrize(
        "dtype,batch,parts,expected",
        [
            ("bf16", 31, 8, "tile"),
            ("bf16", 32, 8, "bf16_wave"),
            ("bf16", 96, 16, "bf16_wave"),
            ("fp8", 8, 8, "fp8_small"),
            ("fp8", 9, 8, "fp8_wave"),
            ("fp8", 64, 1, "fp8_small"),
            ("fp8", 65, 1, "fp8_wave"),
        ],
    )
    @pytest.mark.parametrize("dim", [128, 192])
    def test_kernel_and_workspace(
        self, monkeypatch, dtype, batch, parts, expected, dim
    ):
        from aiter.ops.flydsl.kernels.pa_decode_plan import make_pa_decode_plan

        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda *a: pytest.fail("explicit planning read GPU properties"),
        )
        plan = make_pa_decode_plan(
            arch="gfx950",
            num_seqs=batch,
            num_kv_heads=1,
            query_group_size=16,
            query_length=8,
            head_dim=dim,
            value_head_dim=128,
            block_size=64,
            query_dtype="bf16",
            kv_dtype=dtype,
            per_token_kv=False,
            trans_v=True,
            softmax_scale=None,
            num_partitions=parts,
        )
        assert plan.kernel == expected
        assert plan.num_partitions == parts
        assert plan.scalar_shape == (batch, 1, parts, 128)
        assert plan.output_shape == (batch, 1, parts, 128, 128)

    def test_automatic_policy(self, monkeypatch):
        from aiter.ops.flydsl.kernels.pa_decode_plan import get_recommended_splits

        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda *a: SimpleNamespace(multi_processor_count=256),
        )
        for batch in (1, 3, 32, 96, 200):
            for heads in (1, 2):
                for blocks in (4, 16):
                    expected = max(
                        4, min(-(-512 // (batch * heads * blocks)) * blocks, 8)
                    )
                    assert get_recommended_splits(batch, heads, blocks) == expected
        assert get_recommended_splits(3, 1, sliding_window=256, query_length=8) == 3


@_requires_tile_pa
class TestCustomOp:
    def test_schema_and_export(self):
        import aiter

        assert aiter.pa_decode_flydsl is pa_decode_flydsl
        schema = torch.ops.aiter.pa_decode_flydsl.default._schema
        mutated = {
            a.name
            for a in schema.arguments
            if a.alias_info is not None and a.alias_info.is_write
        }
        assert mutated == {"output", "exp_sums", "max_logits", "temporary_output"}
        assert not schema.returns
        assert "work_plan" not in {argument.name for argument in schema.arguments}

    @pytest.mark.parametrize("dtype", ["bf16", "fp8"])
    @pytest.mark.parametrize("dim", [128, 192])
    @pytest.mark.parametrize("context", [8, 513, 1027])
    @pytest.mark.parametrize("parts", [1, 8])
    @pytest.mark.parametrize("entrypoint", ["python", "torch"])
    def test_reference(self, monkeypatch, dtype, dim, context, parts, entrypoint):
        calls = []
        original = tile.pa_decode_tile

        def record(*args, **kwargs):
            assert not args
            calls.append(kwargs)
            return original(**kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(tile, "pa_decode_tile", record)
            test_tile_pa_vectorized_5d_matches_torch(
                compute_type=dtype,
                query_length=8,
                value_head_size=128,
                num_partitions=parts,
                head_size=dim,
                num_heads=(16, 1),
                context_length=context,
                batch_size=32 if parts == 8 else 3,
                block_size=64,
                entrypoint="direct",
            )
        assert len(calls) == 1
        args = calls[0]
        expected = args["output"].clone()
        args["output"].fill_(float("nan"))
        operation = (
            pa_decode_flydsl
            if entrypoint == "python"
            else torch.ops.aiter.pa_decode_flydsl
        )
        assert operation(**_custom_op_kwargs(args)) is None
        torch.testing.assert_close(args["output"], expected, rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", ["bf16", "fp8"])
    @pytest.mark.parametrize("parts", [1, 4])
    def test_operator_contract(self, dtype, parts):
        args = make_case(dtype, parts)
        kwargs = _custom_op_kwargs(args)
        for key in ("output", "pmax", "psum", "pout"):
            args[key].zero_()
        writable = {"output", "exp_sums", "max_logits", "temporary_output"}
        before = {
            key: value.view(torch.uint8).clone()
            for key, value in kwargs.items()
            if isinstance(value, torch.Tensor) and key not in writable
        }
        torch.ops.aiter.pa_decode_flydsl(**kwargs)
        for key, expected in before.items():
            assert torch.equal(kwargs[key].view(torch.uint8), expected)
        # SchemaCheckMode uses allclose, which does not support CUDA FP8 in
        # PyTorch 2.9. Check those read-only bytes above; keep FakeTensor/AOT.
        checks_to_run = ("test_faketensor", "test_aot_dispatch_dynamic")
        if dtype == "bf16":
            checks_to_run = ("test_schema", *checks_to_run)
        checks = torch.library.opcheck(
            torch.ops.aiter.pa_decode_flydsl.default,
            (),
            kwargs,
            test_utils=checks_to_run,
        )
        assert all(value == "SUCCESS" for value in checks.values())

    @pytest.mark.parametrize("dtype", ["bf16", "fp8"])
    def test_fake_does_not_launch(self, monkeypatch, dtype):
        from torch._subclasses.fake_tensor import FakeTensorMode

        kwargs = _custom_op_kwargs(make_case(dtype))
        monkeypatch.setattr(
            tile,
            "pa_decode_tile",
            lambda *a, **k: pytest.fail("FakeTensor dispatched a GPU kernel"),
        )
        with FakeTensorMode() as mode:
            fake = {
                key: (
                    mode.from_tensor(value)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in kwargs.items()
            }
            assert torch.ops.aiter.pa_decode_flydsl(**fake) is None

    @pytest.mark.parametrize("dtype", ["bf16", "fp8"])
    def test_capture_and_compile(self, dtype):
        args = make_case(dtype)
        kwargs = _custom_op_kwargs(args)
        op = torch.ops.aiter.pa_decode_flydsl
        op(**kwargs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(**kwargs)
        args["output"].zero_()
        graph.replay()
        torch.testing.assert_close(
            args["output"], torch.ones_like(args["output"]), rtol=0, atol=0
        )

        def run(
            output, query, key, value, lengths, table, ks, vs, sums, maxima, partial
        ):
            torch.ops.aiter.pa_decode_flydsl(
                output,
                query,
                key,
                value,
                lengths,
                table,
                128**-0.5,
                8,
                4,
                compute_type=key.dtype,
                key_scale=ks,
                value_scale=vs,
                exp_sums=sums,
                max_logits=maxima,
                temporary_output=partial,
            )
            return output

        compiled = torch.compile(run, backend="aot_eager", fullgraph=True, dynamic=True)
        args["output"].zero_()
        result = compiled(
            *(
                kwargs[key]
                for key in (
                    "output",
                    "query",
                    "key_cache",
                    "value_cache",
                    "context_lengths",
                    "block_tables",
                    "key_scale",
                    "value_scale",
                    "exp_sums",
                    "max_logits",
                    "temporary_output",
                )
            )
        )
        torch.testing.assert_close(result, torch.ones_like(result), rtol=0, atol=0)

    @pytest.mark.parametrize("dim", [128, 192])
    @pytest.mark.parametrize("parts", [1, 8])
    @pytest.mark.skipif(_ARCH != "gfx950", reason="FP8 wave replay requires gfx950")
    def test_mutated_context_graph(self, dim, parts):
        TestFP8Wave().test_fp8_wave_page_signatures_and_mutated_graph(
            SimpleNamespace(pa_decode_tile=_registered_tile),
            dim,
            parts,
            False,
            (511, 512, 513),
        )

    def test_per_token_four_dimensional_scales(self):
        args = make_case("fp8")
        args["key_scale"] = torch.ones((2, 1, 64, 1), device="cuda")
        args["value_scale"] = (
            (torch.arange(1, 65, device="cuda", dtype=torch.float32) / 100)
            .reshape(1, 1, 64, 1)
            .expand(2, 1, 64, 1)
            .contiguous()
        )
        kwargs = _custom_op_kwargs(args)
        torch.ops.aiter.pa_decode_flydsl(**kwargs)
        expected = (
            ((torch.arange(1, 9, device="cuda", dtype=torch.float32) + 1) / 200)
            .repeat(2)
            .reshape(16, 1, 1)
            .expand_as(args["output"])
        )
        torch.testing.assert_close(
            args["output"].float(), expected, rtol=0.02, atol=0.02
        )

    @pytest.mark.parametrize(
        "changes,match",
        [
            ({"query_length": 4}, "query_length"),
            ({"context_partition_size": 64}, "context_partition_size"),
            ({"compute_type": torch.float16}, "compute_type"),
            ({"sliding_window": 64}, "full attention"),
            ({"sliding_window": -2}, "full attention"),
        ],
    )
    def test_rejects_unsupported_contract(self, changes, match):
        kwargs = _custom_op_kwargs(make_case())
        kwargs.update(changes)
        with pytest.raises(ValueError, match=match):
            torch.ops.aiter.pa_decode_flydsl(**kwargs)

    @pytest.mark.parametrize("name", ["query_scale", "alibi_slopes", "sinks"])
    def test_rejects_unsupported_tensor_options(self, name):
        kwargs = _custom_op_kwargs(make_case())
        kwargs[name] = torch.ones(1, device="cuda")
        with pytest.raises(ValueError):
            torch.ops.aiter.pa_decode_flydsl(**kwargs)

    @pytest.mark.parametrize("dtype", ["bf16", "fp8"])
    def test_rejects_padded_block_table(self, dtype):
        kwargs = _custom_op_kwargs(make_case(dtype))
        kwargs["block_tables"] = torch.zeros((2, 2), device="cuda", dtype=torch.int32)[
            :, :1
        ]
        with pytest.raises(ValueError, match="block_tables must be contiguous"):
            torch.ops.aiter.pa_decode_flydsl(**kwargs)

    def test_registration_does_not_import_decode_backend(self):
        import os
        import subprocess

        code = """
import importlib.abc
import sys
attempts = []
class BlockDecode(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "aiter.ops.flydsl.pa_decode":
            attempts.append(fullname)
            raise ModuleNotFoundError("decode backend intentionally unavailable", name=fullname)
sys.meta_path.insert(0, BlockDecode())
import aiter
import torch
assert callable(aiter.pa_decode_flydsl)
assert torch.ops.aiter.pa_decode_flydsl.default._schema
assert "aiter.ops.flydsl.pa_decode" not in sys.modules
assert not attempts
"""
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            check=True,
            capture_output=True,
            text=True,
        )

    @pytest.mark.parametrize("dtype", ["bf16", "fp8"])
    def test_automatic_splits(self, dtype):
        args = make_case(dtype)
        args["num_partitions"] = 0
        args["pmax"] = args["psum"] = args["pout"] = None
        torch.ops.aiter.pa_decode_flydsl(**_custom_op_kwargs(args))
        torch.testing.assert_close(
            args["output"], torch.ones_like(args["output"]), rtol=0, atol=0
        )
