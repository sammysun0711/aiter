# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""AITER API contracts in addition to the imported kernel regression suite."""

import pytest
import torch

from aiter.ops.flydsl import flydsl_pa_decode_ps, flydsl_pa_decode_tile
from aiter.ops.flydsl import pa_decode as tile
from op_tests import test_flydsl_pa_decode as reference

pytestmark = reference._requires_tile_pa


def make_case(dtype="bf16", parts=4):
    kv_dtype = torch.bfloat16 if dtype == "bf16" else torch.float8_e4m3fn
    vector = 16 // kv_dtype.itemsize
    shape = (2, 1, parts, 128)
    scale = torch.ones(1, device="cuda") if dtype == "fp8" else None
    return dict(
        output=torch.empty((16, 16, 128), device="cuda", dtype=torch.bfloat16),
        query=torch.zeros((16, 16, 128), device="cuda", dtype=torch.bfloat16),
        key_cache=torch.zeros(
            (2, 1, 128 // vector, 64, vector), device="cuda", dtype=kv_dtype
        ),
        value_cache=torch.ones(
            (2, 1, 64 // vector, 128, vector), device="cuda", dtype=kv_dtype
        ),
        block_tables=torch.tensor([[0], [1]], device="cuda", dtype=torch.int32),
        context_lengths=torch.full((2,), 8, device="cuda", dtype=torch.int32),
        key_scale=scale,
        value_scale=scale,
        num_partitions=parts,
        pmax=torch.empty(shape, device="cuda"),
        psum=torch.empty(shape, device="cuda"),
        pout=torch.empty((*shape, 128), device="cuda", dtype=torch.bfloat16),
    )


def test_public_exports():
    assert flydsl_pa_decode_tile is tile.pa_decode_tile
    assert flydsl_pa_decode_ps is tile.pa_decode_ps_launch


@pytest.mark.parametrize(
    "name", ["key_cache", "value_cache", "output", "pmax", "psum", "pout"]
)
def test_rejects_wrong_device_before_compile(monkeypatch, name):
    args = make_case()
    args[name] = args[name].cpu()
    monkeypatch.setattr(
        tile,
        "compile_pa_decode_tile",
        lambda **kw: pytest.fail("unexpected compilation"),
    )
    with pytest.raises(ValueError, match=name):
        flydsl_pa_decode_tile(**args)


@pytest.mark.parametrize("name", ["key_cache", "value_cache", "pmax", "psum", "pout"])
def test_rejects_noncontiguous_storage(monkeypatch, name):
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
def test_rejects_invalid_partition_count(parts):
    args = make_case()
    args["num_partitions"] = parts
    with pytest.raises(ValueError, match="num_partitions"):
        flydsl_pa_decode_tile(**args)


@pytest.mark.parametrize("name", ["pmax", "psum", "pout"])
def test_rejects_partial_workspace_set(name):
    args = make_case()
    args[name] = None
    with pytest.raises(ValueError, match="provide all"):
        flydsl_pa_decode_tile(**args)


@pytest.mark.parametrize("name", ["pmax", "psum", "pout"])
def test_rejects_wrong_workspace_dtype(name):
    args = make_case()
    args[name] = args[name].to(torch.float64)
    with pytest.raises(ValueError, match=name):
        flydsl_pa_decode_tile(**args)


@pytest.mark.parametrize("name", ["key_scale", "value_scale"])
def test_rejects_multielement_scalar_scale(name):
    args = make_case("fp8")
    args[name] = torch.ones(2, device="cuda")
    with pytest.raises(ValueError, match="one element"):
        flydsl_pa_decode_tile(**args)


@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
def test_np1_preallocated_dummy(dtype):
    args = make_case(dtype, 1)
    args["psum"] = args["pout"] = None
    flydsl_pa_decode_tile(**args)
    torch.testing.assert_close(
        args["output"], torch.ones_like(args["output"]), rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
def test_explicit_stream_and_replay(dtype):
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


def test_capture_rejects_python_scales_before_compile(monkeypatch):
    args = make_case("fp8")
    args["key_scale"] = args["value_scale"] = 1.0
    monkeypatch.setattr(tile, "_is_current_stream_capturing", lambda: True)
    with pytest.raises(ValueError, match="pre-created FP8 scale tensors"):
        flydsl_pa_decode_tile(**args)


@pytest.mark.parametrize("kind", ["python", "cpu", "dtype"])
@pytest.mark.parametrize("entrypoint", ["direct", "ps"])
def test_explicit_stream_rejects_scale_conversion(kind, entrypoint):
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
def test_automatic_splits_remain_capped(dtype):
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
def test_throughput_grid_fp8_short_context(dim, context):
    reference.test_tile_pa_vectorized_5d_matches_torch(
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
def test_fp8_per_token_and_4d_value_fallback(dim, page, query_dtype):
    reference.setup_seed(20260919)
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
    table = torch.arange(pages - 1, -1, -1, device="cuda", dtype=torch.int32).reshape(
        batch, blocks
    )
    lengths = torch.full((batch,), context, device="cuda", dtype=torch.int32)
    indptr = torch.arange(0, (batch + 1) * qlen, qlen, device="cuda", dtype=torch.int32)
    expected = reference.torch_mha_extend(
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
