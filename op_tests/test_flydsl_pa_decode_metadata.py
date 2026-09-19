# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""The tile ABI requires dense metadata, but permits nonzero storage offsets."""

import pytest
import torch

from op_tests import test_flydsl_pa_decode as test_pa

pytestmark = test_pa._requires_tile_pa


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
    monkeypatch, compute_type, entrypoint, invalid, match
):
    from aiter.ops.flydsl.kernels import pa_decode_bf16_wave as wave
    from aiter.ops.flydsl.kernels import pa_decode_fp8_small as small
    from aiter.ops.flydsl.kernels import pa_decode_fp8_wave as fp8_wave
    from aiter.ops.flydsl import pa_decode as tile

    def unexpected_compile(*args, **kwargs):
        pytest.fail("Invalid metadata reached kernel compilation or launch")

    monkeypatch.setattr(tile, "compile_pa_decode_tile", unexpected_compile)
    monkeypatch.setattr(wave, "compile_pa_decode_bf16_wave", unexpected_compile)
    monkeypatch.setattr(fp8_wave, "compile_pa_decode_fp8_wave", unexpected_compile)
    monkeypatch.setattr(small, "compile_pa_decode_fp8_small", unexpected_compile)
    monkeypatch.setattr(tile, "_run_compiled", unexpected_compile)
    batch, heads, dim, page, parts = 32, 16, 128, 64, 8
    dtype = test_pa.dtypes.d_dtypes[compute_type]
    vector = 16 // dtype.itemsize
    query = torch.empty((batch * 8, heads, dim), dtype=torch.bfloat16, device="cuda")
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
        table = torch.zeros((batch, 3), dtype=table.dtype, device=table.device)[:, :2]
    elif invalid == "table_column_stride":
        table = torch.zeros((batch, 4), dtype=table.dtype, device=table.device)[:, ::2]
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
            test_pa.flydsl_ps_launch(
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
def test_tile_dense_metadata_storage_offsets(monkeypatch, compute_type, entrypoint):
    original_reference = test_pa.torch_mha_extend
    seen = []

    def reference(query, key, value, table, lengths, indptr, ks, vs, *args, **kwargs):
        padded_table = torch.empty(
            (table.shape[0] + 1, table.shape[1]), dtype=table.dtype, device=table.device
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

    monkeypatch.setattr(test_pa, "torch_mha_extend", reference)
    test_pa.test_tile_pa_vectorized_5d_matches_torch(
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
