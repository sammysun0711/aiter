# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
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
