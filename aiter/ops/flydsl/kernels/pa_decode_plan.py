# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 FlyDSL Project Contributors

"""Static launch planning for FlyDSL PA decode.

Unlike PR #4332's dynamic GPU worklists, this plan has no device metadata
and does not read context lengths. It preserves the validated split policy,
kernel selection and normalized-partial layout. Dynamic scheduling is deferred.
"""

import functools
from dataclasses import dataclass
from typing import Literal

import torch

KV_COMPUTE_BLOCK = 256


def get_recommended_splits(
    num_sequences: int,
    num_kv_heads: int,
    split_kv_blocks: int = 1,
    *,
    sliding_window: int = 0,
    context_partition_size: int = KV_COMPUTE_BLOCK,
    query_length: int = 1,
) -> int:
    """Preserve the existing Gluon split heuristic and upper clamp of eight."""
    if sliding_window > 0:
        window_token_count = sliding_window + query_length
        return -(-(window_token_count - 1) // context_partition_size) + 1
    properties = torch.cuda.get_device_properties(torch.device("cuda"))
    capacity = properties.multi_processor_count * 2
    denominator = max(1, num_sequences * num_kv_heads * split_kv_blocks)
    partitions = -(-capacity // denominator) * split_kv_blocks
    return max(4, min(partitions, 8))


@dataclass(frozen=True)
class PADecodePlan:
    """Host-only static launch selection; no GPU worklist or packed scratch."""

    kernel: Literal["tile", "bf16_wave", "fp8_wave", "fp8_small"]
    num_partitions: int
    scalar_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int, int, int, int]


@functools.lru_cache(maxsize=256)
def _static_plan(
    arch,
    num_seqs,
    num_kv_heads,
    query_group_size,
    query_length,
    head_dim,
    value_head_dim,
    block_size,
    query_dtype,
    kv_dtype,
    per_token_kv,
    trans_v,
    positive_scale,
    num_partitions,
):
    wave_geometry = (
        arch == "gfx950"
        and query_dtype == "bf16"
        and query_length == 8
        and query_group_size == 16
        and block_size == 64
        and head_dim in (128, 192)
        and value_head_dim == 128
        and positive_scale
    )
    grid_size = num_seqs * num_kv_heads * num_partitions
    kernel = "tile"
    if wave_geometry and kv_dtype == "bf16" and grid_size >= 256:
        kernel = "bf16_wave"
    elif wave_geometry and kv_dtype == "fp8" and not per_token_kv and trans_v:
        kernel = "fp8_small" if grid_size <= 64 else "fp8_wave"
    shape = (num_seqs, num_kv_heads, num_partitions, query_length * query_group_size)
    return PADecodePlan(kernel, num_partitions, shape, (*shape, value_head_dim))


def make_pa_decode_plan(
    *,
    arch: str,
    num_seqs: int,
    num_kv_heads: int,
    query_group_size: int,
    query_length: int,
    head_dim: int,
    value_head_dim: int,
    block_size: int,
    query_dtype: str,
    kv_dtype: str,
    per_token_kv: bool,
    trans_v: bool,
    softmax_scale: float | None,
    num_partitions: int | None,
) -> PADecodePlan:
    """Plan validated host geometry without allocation, readback or GPU work."""
    if not num_partitions:
        num_partitions = get_recommended_splits(
            num_seqs, num_kv_heads, split_kv_blocks=KV_COMPUTE_BLOCK // block_size
        )
    return _static_plan(
        arch,
        num_seqs,
        num_kv_heads,
        query_group_size,
        query_length,
        head_dim,
        value_head_dim,
        block_size,
        query_dtype,
        kv_dtype,
        per_token_kv,
        trans_v,
        softmax_scale is None or 0.0 < softmax_scale < float("inf"),
        num_partitions,
    )
