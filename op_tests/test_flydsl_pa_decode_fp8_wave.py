# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Reference and replay coverage for the gfx950 qlen8 FP8 specialization."""

import pytest
import torch

from op_tests import test_flydsl_pa_decode as test_pa

pytestmark = test_pa._requires_tile_pa


@pytest.fixture(scope="module", autouse=True)
def fp8_wave():
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
def test_fp8_wave_reference(dim, context, parts, heads, entrypoint):
    test_pa.test_tile_pa_vectorized_5d_matches_torch(
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
def test_fp8_wave_grouped_grid_boundaries(dim, context, batch, parts):
    # Exercise partial eight-sequence groups and both sides of the small-grid gate.
    test_pa.test_tile_pa_vectorized_5d_matches_torch(
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
def test_fp8_wave_constant_frontier(fp8_wave, parts, heads, query_value):
    qheads, kvheads = heads
    query = torch.full(
        (8, qheads, 192), query_value, device="cuda", dtype=torch.bfloat16
    )
    key = torch.ones((2, kvheads, 12, 64, 16), device="cuda", dtype=torch.float8_e4m3fn)
    values = torch.full((2, kvheads, 128, 64), 64.0, device="cuda")
    offsets = torch.arange(kvheads, device="cuda") * 16
    values[1, :, :, :9] = offsets[:, None, None] + torch.arange(1, 10, device="cuda")
    values = values.to(key.dtype)
    value = test_pa.shuffle_value_cache_layout(values)
    table = torch.tensor([[1]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([9], device="cuda", dtype=torch.int32)
    scale = torch.ones(1, device="cuda")
    output = torch.empty((8, qheads, 128), device="cuda", dtype=query.dtype)
    fp8_wave.pa_decode_tile(
        output, query, key, value, table, lengths, scale, scale, num_partitions=parts
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
    fp8_wave, dim, parts, zero_query, contexts
):
    test_pa.setup_seed(20260919)
    batch, qlen, qheads, kvheads = 3, 8, 32, 2
    max_blocks = max((length + 63) // 64 for length in contexts)
    pages = batch * max_blocks + 1
    query = torch.randn(
        batch * qlen, qheads * 2, dim, device="cuda", dtype=torch.bfloat16
    )[:, ::2]
    if zero_query:
        query.zero_()
    storage = torch.full(
        (batch * qlen, qheads * 2, 128), float("nan"), device="cuda", dtype=query.dtype
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
    value = test_pa.shuffle_value_cache_layout(values)
    table = (
        torch.randperm(pages, device="cuda", dtype=torch.int64)[: batch * max_blocks]
        .to(torch.int32)
        .reshape(batch, max_blocks)
    )
    lengths = torch.tensor(contexts, device="cuda", dtype=torch.int32)
    indptr = torch.arange(0, (batch + 1) * qlen, qlen, device="cuda", dtype=torch.int32)
    ks = torch.tensor([0.37], device="cuda")
    vs = torch.tensor([0.63], device="cuda")
    partial_shape = (batch, kvheads, parts, 128)
    pmax = torch.empty(partial_shape, device="cuda")
    psum = torch.empty_like(pmax)
    pout = torch.empty((*partial_shape, 128), device="cuda", dtype=query.dtype)

    def reference():
        return test_pa.torch_mha_extend(
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
    value.copy_(test_pa.shuffle_value_cache_layout(values))
    table.copy_(table.flip(1))
    lengths.copy_(
        torch.tensor(
            [contexts[2], contexts[0], contexts[1]], device="cuda", dtype=lengths.dtype
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
def test_fp8_wave_high_pages(fp8_wave, dim, parts, context):
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
        [value[p].permute(1, 3, 0, 2).reshape(64, 1, 128) for p in table[0].tolist()]
    )[:context]
    expected = test_pa.reference_masked_attention(
        query,
        selected_k.float(),
        selected_v.float(),
        dim**-0.5,
        query.dtype,
        is_causal=True,
    )
    assert expected.float().abs().min().item() > 0.5
    fp8_wave.pa_decode_tile(
        output, query, key, value, table, lengths, scale, scale, num_partitions=parts
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
def test_fp8_wave_dispatch(monkeypatch, case, expected):
    from aiter.ops.flydsl.kernels import pa_decode_fp8_small as small
    from aiter.ops.flydsl.kernels import pa_decode_fp8_wave as wave
    from aiter.ops.flydsl import pa_decode as tile

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
    vshape = (1, 1, vdim, page) if case == "plain-v" else (1, 1, page // 16, vdim, 16)
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
