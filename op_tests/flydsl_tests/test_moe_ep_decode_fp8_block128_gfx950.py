# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Numerical and dynamic-graph coverage for the block-FP8 EP decode backend."""
import pytest
import torch

pytest.importorskip("flydsl")
if not torch.cuda.is_available() or not torch.cuda.get_device_properties(
    0
).gcnArchName.startswith("gfx950"):
    pytest.skip("The block-FP8 EP kernel requires gfx950", allow_module_level=True)

from aiter.ops.flydsl.kernels.moe_ep_decode_fp8_block128_gfx950.configuration import (
    Stage1,
    OutputMode,
    WeightCache,
)
from aiter.ops.flydsl.moe_ep_decode_fp8_block128_gfx950 import (
    fused_moe_ep_decode_fp8_block128_gfx950,
)
from aiter.ops.shuffle import shuffle_weight

FP8 = torch.float8_e4m3fn


def quantize_weights(weight):
    experts, n, k = weight.shape
    blocks = weight.reshape(experts, n // 128, 128, k // 128, 128)
    scales = blocks.abs().amax((2, 4)).clamp_min(1e-10) / 448
    q = (blocks / scales[:, :, None, :, None]).to(FP8).reshape_as(weight)
    dequantized = (
        q.float().reshape_as(blocks) * scales[:, :, None, :, None]
    ).reshape_as(weight)
    return shuffle_weight(q, (16, 16)), scales.contiguous(), dequantized


def reference(x, scale, w1, w2, ids, weights, mask):
    """Explicit expert dispatch and two FP32 matmuls with FP8 intermediate."""
    rows, hidden = x.shape
    x = (x.float().reshape(rows, hidden // 128, 128) * scale[:, :, None]).reshape(
        rows, hidden
    )
    intermediate = w2.shape[-1]
    partial = torch.zeros(rows, ids.shape[1], hidden, device=x.device)
    global_ids = mask.nonzero().flatten().tolist()
    for local, global_id in enumerate(global_ids):
        row, slot = ((ids == global_id) & (weights != 0)).nonzero(as_tuple=True)
        if not row.numel():
            continue
        gate, up = (x[row] @ w1[local].T).chunk(2, dim=-1)
        z = (torch.nn.functional.silu(gate) * up).reshape(-1, intermediate // 128, 128)
        z_scale = z.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448
        z = ((z / z_scale).to(FP8).float() * z_scale).reshape(-1, intermediate)
        value = (z @ w2[local].T) * weights[row, slot, None]
        partial[row, slot] = value.to(torch.bfloat16).float()
    return partial.sum(1).to(torch.bfloat16)


def assert_close(actual, expected):
    assert torch.isfinite(actual).all()
    difference = (actual.float() - expected.float()).square().sum(-1)
    denominator = (actual.float().square() + expected.float().square()).sum(-1)
    assert (difference / denominator.clamp_min(1e-20)).max() < 2e-3
    zero_rows = expected.eq(0).all(-1)
    assert actual[zero_rows].eq(0).all()


@pytest.mark.parametrize(
    "hidden,intermediate,topk",
    [
        (1024, 128, 1),
        (1024, 256, 4),
        (2048, 256, 8),
        (4096, 256, 4),
        (6144, 2048, 8),
    ],
)
@pytest.mark.parametrize(
    "source_hint",
    [
        None,
        8,
        "scalar_projection",
        "tiled_projection",
        "reuse_cache",
        "m16_reduce",
        "m32_atomic",
        "m64_projection",
        "auto_fused",
        "auto_projection",
    ],
)
def test_dynamic_count_mask_and_padding(
    hidden, intermediate, topk, source_hint, monkeypatch
):
    torch.manual_seed(1234)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    capacity, experts, global_experts = 64, 4, 12
    w1, s1, ref_w1 = quantize_weights(
        torch.randn(experts, 2 * intermediate, hidden, device="cuda") * 0.04
    )
    w2, s2, ref_w2 = quantize_weights(
        torch.randn(experts, hidden, intermediate, device="cuda") * 0.04
    )
    x = torch.randn(capacity, hidden, device="cuda").to(FP8)
    scale = torch.rand(capacity, hidden // 128, device="cuda") * 0.5 + 0.1
    ids = (
        torch.rand(capacity, global_experts, device="cuda")
        .topk(topk, -1)
        .indices.to(torch.int32)
    )
    weights = torch.rand(capacity, topk, device="cuda")
    weights /= weights.sum(-1, keepdim=True)
    mask = torch.zeros(global_experts, device="cuda", dtype=torch.int32)
    mask[[0, 2, 5, 11]] = 1
    count = torch.zeros(1, device="cuda", dtype=torch.int32)

    from aiter.ops.flydsl.moe_ep_decode_fp8_block128_gfx950 import MoeConfig

    config = None
    if source_hint in ("auto_fused", "auto_projection"):
        projection = source_hint == "auto_projection"
        source_hint = None
        config = MoeConfig(
            stage1=Stage1.SEPARATE_FP8_QUANT if projection else Stage1.FUSED_FP8_QUANT,
            stage1_cache=WeightCache.ADAPTIVE,
            stage2_cache=(WeightCache.STREAM if projection else WeightCache.ADAPTIVE),
            down_adaptive_m=True,
        )
    if source_hint == "m64_projection":
        source_hint = None
        config = MoeConfig(
            block_m=64,
            stage1_n=128,
            stage1=Stage1.SEPARATE_FP8_QUANT_TRANSPOSED,
            quant_tiled=True,
            down_adaptive_m=True,
        )
    if source_hint in ("m16_reduce", "m32_atomic"):
        atomic = source_hint == "m32_atomic"
        source_hint = None
        config = MoeConfig(
            block_m=32 if atomic else 16,
            output=(OutputMode.ATOMIC if atomic else OutputMode.PARTIAL),
        )
    if source_hint in ("scalar_projection", "tiled_projection"):
        tiled = source_hint == "tiled_projection"
        source_hint = None
        config = MoeConfig(
            stage1_n=128, stage1=Stage1.SEPARATE_FP8_QUANT_TRANSPOSED, quant_tiled=tiled
        )

    if source_hint == "reuse_cache":
        source_hint = None
        config = MoeConfig(
            stage2_n=128,
            stage1_cache=WeightCache.CACHE,
            stage2_cache=WeightCache.ADAPTIVE,
            down_adaptive_m=True,
        )

    def run():
        return fused_moe_ep_decode_fp8_block128_gfx950(
            x,
            w1,
            w2,
            ids,
            weights,
            scale,
            s1,
            s2,
            mask,
            count,
            source_hint,
            config=config,
        )

    # Compile with no received rows, then exercise positive counts in the same graph.
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for iteration, n in enumerate([1, 33, capacity, 0, 19]):
        count.fill_(n)
        scale.mul_(0.9)
        mask.zero_()
        mask[[1, 3, 7, 9] if iteration % 2 == 0 else [0, 2, 5, 11]] = 1
        ids.copy_((ids + 1) % global_experts)
        if n:
            # Duplicate IDs in zero-weight padding must not read uninitialized partials.
            ids[n - 1].fill_(1)
            weights[n - 1].zero_()
        graph.replay()
        torch.cuda.synchronize()
        if n:
            expected = reference(
                x[:n], scale[:n], ref_w1, ref_w2, ids[:n], weights[:n], mask
            )
            assert_close(output[:n], expected)


@pytest.mark.parametrize("projection_m", [None, 32, 64])
def test_all_local_topk_graph_counts(projection_m):
    """Exercise hot routes and every M16 tail boundary in the captured graph."""
    torch.manual_seed(4321)
    experts, hidden, intermediate, capacity, topk = 8, 1024, 128, 64, 8
    w1, s1, ref_w1 = quantize_weights(
        torch.randn(experts, 2 * intermediate, hidden, device="cuda") * 0.04
    )
    w2, s2, ref_w2 = quantize_weights(
        torch.randn(experts, hidden, intermediate, device="cuda") * 0.04
    )
    x = torch.randn(capacity, hidden, device="cuda").to(FP8)
    scale = torch.rand(capacity, hidden // 128, device="cuda") + 0.1
    ids = torch.arange(experts, device="cuda", dtype=torch.int32).repeat(capacity, 1)
    weights = torch.rand(capacity, topk, device="cuda")
    weights /= weights.sum(-1, keepdim=True)
    mask = torch.zeros(2 * experts, device="cuda", dtype=torch.int32)
    mask[:experts] = 1
    count = torch.tensor([capacity], device="cuda", dtype=torch.int32)
    from aiter.ops.flydsl.moe_ep_decode_fp8_block128_gfx950 import MoeConfig

    config = (
        None
        if projection_m is None
        else MoeConfig(
            block_m=projection_m,
            stage1_n=128,
            stage1=Stage1.SEPARATE_FP8_QUANT_TRANSPOSED,
            output=OutputMode.ATOMIC if projection_m == 16 else OutputMode.PARTIAL,
            quant_tiled=True,
            down_adaptive_m=True,
        )
    )

    def run():
        return fused_moe_ep_decode_fp8_block128_gfx950(
            x, w1, w2, ids, weights, scale, s1, s2, mask, count, 4, config=config
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for n in [capacity, 49, 48, 33, 32, 17, 16, 1, 0, capacity]:
        count.fill_(n)
        graph.replay()
        torch.cuda.synchronize()
        if n:
            expected = reference(
                x[:n], scale[:n], ref_w1, ref_w2, ids[:n], weights[:n], mask
            )
            assert_close(output[:n], expected)


def test_singleton_expert_descriptor_bounds():
    """The last packed expert descriptor must be read within its one-word bounds."""
    from aiter.ops.flydsl.moe_ep_decode_fp8_block128_gfx950 import MoeConfig

    torch.manual_seed(765)
    hidden, intermediate, experts = 1024, 128, 2
    w1, s1, ref_w1 = quantize_weights(
        torch.randn(experts, 2 * intermediate, hidden, device="cuda") * 0.04
    )
    w2, s2, ref_w2 = quantize_weights(
        torch.randn(experts, hidden, intermediate, device="cuda") * 0.04
    )
    x = torch.randn(1, hidden, device="cuda").to(FP8)
    scale = torch.ones(1, hidden // 128, device="cuda")
    ids = torch.tensor([[1]], device="cuda", dtype=torch.int32)
    weights = torch.ones(1, 1, device="cuda")
    mask = torch.zeros(2 * experts, device="cuda", dtype=torch.int32)
    mask[:experts] = 1
    count = torch.ones(1, device="cuda", dtype=torch.int32)
    config = MoeConfig(stage2_n=128, stage2_cache=WeightCache.ADAPTIVE)
    output = fused_moe_ep_decode_fp8_block128_gfx950(
        x, w1, w2, ids, weights, scale, s1, s2, mask, count, config=config
    )
    expected = reference(x, scale, ref_w1, ref_w2, ids, weights, mask)
    assert_close(output, expected)


@pytest.mark.parametrize(
    "block_m,workers,adaptive,reuse,force_opus",
    [
        (16, 1, False, False, False),
        (32, 0, False, False, False),
        (32, 1, False, False, False),
        (64, 1, False, False, False),
        (32, 1, True, False, False),
        (64, 1, True, False, False),
        (32, 1, True, True, False),
        (64, 1, True, True, False),
        (32, 1, True, True, True),
    ],
)
def test_parallel_n_shared_peaks_graph_replay(
    block_m, workers, adaptive, reuse, force_opus, monkeypatch
):
    """Exercise padded M tails and reuse of the shared maxima by worker loops."""
    from aiter.ops.flydsl.moe_ep_decode_fp8_block128_gfx950 import MoeConfig

    if force_opus:
        from aiter.ops.flydsl import moe_ep_decode_fp8_block128_gfx950 as backend

        monkeypatch.setattr(backend, "_use_local_sort", lambda *args: False)

    torch.manual_seed(910)
    capacity, experts, hidden, intermediate, topk = 1024, 8, 1024, 512, 8
    config = MoeConfig(
        block_m=block_m,
        stage1_workers=workers,
        stage2_n=128,
        stage2_workers=4,
        stage1=Stage1.FUSED_FP8_QUANT_PARALLEL,
        stage1_cache=WeightCache.CACHE,
        stage2_cache=WeightCache.ADAPTIVE if reuse else WeightCache.CACHE,
        output=OutputMode.ATOMIC if block_m == 16 else OutputMode.PARTIAL,
        down_adaptive_m=adaptive,
    )
    w1, s1, ref_w1 = quantize_weights(
        torch.randn(experts, 2 * intermediate, hidden, device="cuda") * 0.04
    )
    w2, s2, ref_w2 = quantize_weights(
        torch.randn(experts, hidden, intermediate, device="cuda") * 0.04
    )
    x = torch.randn(capacity, hidden, device="cuda").to(FP8)
    scale = torch.rand(capacity, hidden // 128, device="cuda") + 0.1
    ids = torch.arange(experts, device="cuda", dtype=torch.int32).repeat(capacity, 1)
    weights = torch.rand(capacity, topk, device="cuda")
    weights /= weights.sum(-1, keepdim=True)
    mask = torch.zeros(2 * experts, device="cuda", dtype=torch.int32)
    mask[:experts] = 1
    count = torch.tensor([capacity], device="cuda", dtype=torch.int32)

    def run():
        return fused_moe_ep_decode_fp8_block128_gfx950(
            x, w1, w2, ids, weights, scale, s1, s2, mask, count, config=config
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for n in [capacity, 33, 0, 513, capacity]:
        count.fill_(n)
        scale.mul_(0.9)
        graph.replay()
        torch.cuda.synchronize()
        if n:
            expected = reference(
                x[:n], scale[:n], ref_w1, ref_w2, ids[:n], weights[:n], mask
            )
            assert_close(output[:n], expected)
