# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Numerical and dynamic-graph coverage for the block-FP8 EP decode backend."""
import pytest
import torch

pytest.importorskip("flydsl")
if (
    not torch.cuda.is_available()
    or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950")
):
    pytest.skip("The block-FP8 EP kernel requires gfx950", allow_module_level=True)

from aiter.ops.flydsl.moe_ep_decode_fp8_block128_gfx950 import fused_moe_ep_decode_fp8_block128_gfx950
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
@pytest.mark.parametrize("source_hint", [None, 8])
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

    def run():
        return fused_moe_ep_decode_fp8_block128_gfx950(
            x, w1, w2, ids, weights, scale, s1, s2, mask, count, source_hint
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


def test_all_local_topk_atomic_accumulation():
    """Exercise eight local contributions per token, beyond typical EP8 load."""
    torch.manual_seed(4321)
    experts, hidden, intermediate, capacity, topk = 8, 1024, 128, 64, 8
    w1, s1, ref_w1 = quantize_weights(torch.randn(experts, 2 * intermediate, hidden, device="cuda") * .04)
    w2, s2, ref_w2 = quantize_weights(torch.randn(experts, hidden, intermediate, device="cuda") * .04)
    x = torch.randn(capacity, hidden, device="cuda").to(FP8)
    scale = torch.rand(capacity, hidden // 128, device="cuda") + .1
    ids = torch.arange(experts, device="cuda", dtype=torch.int32).repeat(capacity, 1)
    weights = torch.rand(capacity, topk, device="cuda")
    weights /= weights.sum(-1, keepdim=True)
    mask = torch.ones(experts, device="cuda", dtype=torch.int32)
    count = torch.tensor([capacity], device="cuda", dtype=torch.int32)

    def run():
        return fused_moe_ep_decode_fp8_block128_gfx950(x, w1, w2, ids, weights, scale, s1, s2, mask, count, 4)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for n in [capacity, 17, 0, capacity]:
        count.fill_(n)
        graph.replay()
        torch.cuda.synchronize()
        if n:
            expected = reference(x[:n], scale[:n], ref_w1, ref_w2, ids[:n], weights[:n], mask)
            assert_close(output[:n], expected)
