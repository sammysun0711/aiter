# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Small reference cases explicitly exercise the throughput-only BF16 kernel."""

import pytest
import torch

from op_tests import test_flydsl_pa_decode as test_pa

pytestmark = test_pa._requires_tile_pa


@pytest.fixture
def force_wave(monkeypatch):
    from flydsl.runtime.device import get_rocm_arch
    from aiter.ops.flydsl import pa_decode as tile
    from aiter.ops.flydsl.kernels.pa_decode_bf16_wave import compile_pa_decode_bf16_wave

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
def test_bf16_wave_reference(force_wave, head_dim, context, parts, heads, entrypoint):
    test_pa.test_tile_pa_vectorized_5d_matches_torch(
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
def test_bf16_wave_constant_logits(force_wave, parts, heads, query_value):
    test_pa.test_tile_pa_constant_query_causal_frontier(
        "bf16", parts, 8, heads, query_value
    )


@pytest.mark.parametrize("head_dim", [128, 192])
@pytest.mark.parametrize("parts", [1, 8])
def test_bf16_wave_varlen_strides_and_replay(force_wave, head_dim, parts):
    from aiter.ops.flydsl.pa_decode import pa_decode_tile

    test_pa.setup_seed(20260919)
    batch, qlen, qheads, kvheads = 3, 8, 32, 2
    query = torch.randn(
        batch * qlen, qheads * 2, head_dim, device="cuda", dtype=torch.bfloat16
    )[:, ::2]
    output_storage = torch.full(
        (batch * qlen, qheads * 2, 128), float("nan"), device="cuda", dtype=query.dtype
    )
    output = output_storage[:, ::2]
    keys, values = test_pa.create_kv_cache(
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
    indptr = torch.arange(0, (batch + 1) * qlen, qlen, device="cuda", dtype=torch.int32)
    expected = test_pa.torch_mha_extend(
        query, keys[0], values[0], table, lengths, indptr, None, None
    )
    value = test_pa.shuffle_value_cache_layout(values[0])
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
    value.copy_(test_pa.shuffle_value_cache_layout(values[0]))
    lengths.copy_(torch.tensor([9, 64, 128], device="cuda", dtype=lengths.dtype))
    expected = test_pa.torch_mha_extend(
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


def test_bf16_wave_page_offset_above_2gib(force_wave):
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
    output = torch.full((8, 16, 128), float("nan"), device="cuda", dtype=query.dtype)
    table = torch.tensor([[page]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([9], device="cuda", dtype=torch.int32)
    for parts in (1, 4):
        pa_decode_tile(
            output, query, key, value, table, lengths, None, None, num_partitions=parts
        )
        torch.testing.assert_close(output, torch.full_like(output, 2), rtol=0, atol=0)


@pytest.mark.parametrize(
    "batch,parts,expect_wave", [(31, 8, False), (32, 8, True), (256, 1, True)]
)
def test_bf16_wave_dispatch(monkeypatch, batch, parts, expect_wave):
    from flydsl.runtime.device import get_rocm_arch
    from aiter.ops.flydsl.kernels import pa_decode_bf16_wave as wave
    from aiter.ops.flydsl import pa_decode as tile

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
    test_pa.test_tile_pa_vectorized_5d_matches_torch(
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
