# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
import aiter.fused_moe as fused_moe_mod
from aiter import QuantType, dtypes, get_gfx, get_hip_quant
from aiter.ops.rmsnorm import (
    mimo_add_rmsnorm_fp8_group_quant,
    mimo_rmsnorm_fp8_group_quant,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="MiMo fused RMSNorm/group-quant kernels require gfx950",
)


def _production_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None,
    epsilon: float,
):
    normalized = torch.empty_like(x)
    residual_out = torch.empty_like(x) if residual is not None else None
    if residual is None:
        aiter.rmsnorm(normalized, x, weight, epsilon)
    else:
        aiter.add_rmsnorm(
            normalized,
            x,
            residual,
            residual_out,
            weight,
            epsilon,
        )

    quantized, scale = get_hip_quant(QuantType.per_1x128)(
        normalized,
        quant_dtype=dtypes.fp8,
        transpose_scale=True,
    )
    return normalized, quantized, scale, residual_out


def _fused_result(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None,
    epsilon: float,
):
    m, n = x.shape
    normalized = torch.empty_like(x)
    quantized = torch.empty_like(x, dtype=dtypes.fp8)
    scale = torch.empty((m, n // 128), dtype=torch.float32, device=x.device)
    residual_out = torch.empty_like(x) if residual is not None else None
    if residual is None:
        mimo_rmsnorm_fp8_group_quant(
            quantized,
            normalized,
            scale,
            x,
            weight,
            epsilon,
        )
    else:
        mimo_add_rmsnorm_fp8_group_quant(
            quantized,
            normalized,
            scale,
            x,
            residual,
            residual_out,
            weight,
            epsilon,
        )
    return normalized, quantized, scale, residual_out


@pytest.mark.parametrize("m", [1, 7, 129])
@pytest.mark.parametrize("add_residual", [False, True])
def test_mimo_rmsnorm_fp8_group_quant_matches_production_path(m, add_residual):
    torch.manual_seed(2026 + m)
    epsilon = 1e-6
    x = (torch.randn((m, 6144), dtype=torch.bfloat16, device="cuda") / 4).contiguous()
    weight = (
        1
        + torch.randn((6144,), dtype=torch.bfloat16, device="cuda") / 16
    ).contiguous()
    residual = (
        (torch.randn_like(x) / 4).contiguous() if add_residual else None
    )

    reference = _production_reference(x, weight, residual, epsilon)
    actual = _fused_result(x, weight, residual, epsilon)

    for actual_tensor, reference_tensor in zip(actual, reference):
        if reference_tensor is None:
            assert actual_tensor is None
        else:
            assert torch.equal(actual_tensor, reference_tensor)


def test_prequantized_fused_moe_2stage_transposes_per_1x128_scales(monkeypatch):
    calls = {"partial_transpose": 0, "quant_transpose_scale": []}

    def fake_partial_transpose(dst, src, num_rows=None):
        calls["partial_transpose"] += 1
        dst.copy_(src + 1.0)

    def fake_quant(x, *, scale=None, quant_dtype=None, num_rows=None, **kwargs):
        calls["quant_transpose_scale"].append(kwargs.get("transpose_scale", False))
        out = torch.empty_like(x, dtype=quant_dtype)
        out_scale = torch.empty((x.shape[0], x.shape[1] // 128), device=x.device)
        return out, out_scale

    stage1_inputs = {}
    stage2_inputs = {}

    def fake_stage1(
        a1,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        a2,
        topk,
        *,
        a1_scale=None,
        **kwargs,
    ):
        stage1_inputs["a1_scale"] = a1_scale
        return torch.empty((a1.shape[0], topk, w2.shape[-1]), dtype=torch.bfloat16)

    def fake_stage2(
        a2,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        moe_out,
        topk,
        *,
        a2_scale=None,
        **kwargs,
    ):
        stage2_inputs["a2_scale"] = a2_scale
        return moe_out

    fake_stage1.func = None
    fake_stage1.transpose_quant = True
    fake_stage2.func = None
    fake_stage2.transpose_quant = True

    monkeypatch.setattr(fused_moe_mod.aiter, "partial_transpose", fake_partial_transpose)
    monkeypatch.setattr(fused_moe_mod, "get_quant", lambda quant_type: fake_quant)
    monkeypatch.setattr(
        fused_moe_mod,
        "get_2stage_cfgs",
        lambda *args, **kwargs: fused_moe_mod.MOEMetadata(
            fake_stage1,
            fake_stage2,
            block_m=256,
            ksplit=0,
        ),
    )

    token_num = 2
    model_dim = 128
    inter_dim = 128
    hidden_states = torch.empty((token_num, model_dim), dtype=dtypes.fp8)
    a1_scale = torch.ones((token_num, model_dim // 128), dtype=torch.float32)
    w1 = torch.empty((1, inter_dim * 2, model_dim), dtype=dtypes.fp8)
    w2 = torch.empty((1, model_dim, inter_dim), dtype=dtypes.fp8)
    sorted_ids = torch.zeros((token_num,), dtype=torch.int32)
    sorted_weights = torch.ones((token_num,), dtype=torch.float32)
    sorted_expert_ids = torch.zeros((1,), dtype=torch.int32)
    num_valid_ids = torch.tensor([token_num], dtype=torch.int32)
    moe_out = torch.empty((token_num, model_dim), dtype=torch.bfloat16)

    fused_moe_mod.fused_moe_2stages(
        hidden_states,
        w1,
        w2,
        topk=1,
        sorted_ids=sorted_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        moe_out=moe_out,
        isG1U1=True,
        block_size_M=256,
        quant_type=QuantType.per_1x128,
        q_dtype_a=dtypes.fp8,
        q_dtype_w=dtypes.fp8,
        a1_scale=a1_scale,
        a1_scale_is_transposed=False,
    )

    assert calls["partial_transpose"] == 1
    assert torch.equal(stage1_inputs["a1_scale"], a1_scale + 1.0)
    assert calls["quant_transpose_scale"] == [True]
    assert stage2_inputs["a2_scale"] is not None
