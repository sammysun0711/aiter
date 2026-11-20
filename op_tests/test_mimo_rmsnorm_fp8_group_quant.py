# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from unittest.mock import Mock

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
    not torch.cuda.is_available() or get_gfx() not in ("gfx942", "gfx950"),
    reason="MiMo fused RMSNorm/group-quant kernels require gfx942/gfx950",
)
FN_COMPAT_ENV = "AITER_MIMO_RMSNORM_FP8_FN_COMPAT"
MIMO_WIDTHS = (2048, 4096, 6144)


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


def _assert_matches_production(actual, reference, *, allow_subnormal_ulp=False):
    for observed, expected in zip(actual, reference):
        if expected is None:
            assert observed is None
        elif (
            not allow_subnormal_ulp
            or observed.dtype != dtypes.fp8
            or torch.equal(observed, expected)
        ):
            assert torch.equal(observed, expected)
        else:
            # Keep BF16 normalization/residuals and FP32 scales bitwise exact.
            # At FP8 subnormal midpoints the fused/register and separate/load
            # paths can differ by one representable subnormal step. Normal
            # FP8 values must still match bitwise; this is not a relative-error
            # allowance across the FP8 range.
            a, b = observed.float(), expected.float()
            mismatch = a != b
            tiny = torch.finfo(dtypes.fp8).smallest_normal
            assert torch.all(a[mismatch].abs() <= tiny)
            assert torch.all(b[mismatch].abs() <= tiny)
            assert torch.all((a[mismatch] - b[mismatch]).abs() <= tiny / 8)


@pytest.mark.parametrize("m", [1, 7, 129, 8192])
@pytest.mark.parametrize("n", MIMO_WIDTHS)
@pytest.mark.parametrize("add_residual", [False, True])
@pytest.mark.parametrize("fn_compat", [False, True])
def test_mimo_rmsnorm_fp8_group_quant_matches_production_path(
    m, n, add_residual, fn_compat, monkeypatch
):
    monkeypatch.setenv(FN_COMPAT_ENV, str(int(fn_compat)))
    torch.manual_seed(2026 + m)
    epsilon = 1e-6
    x = (torch.randn((m, n), dtype=torch.bfloat16, device="cuda") / 4).contiguous()
    weight = (
        1 + torch.randn((n,), dtype=torch.bfloat16, device="cuda") / 16
    ).contiguous()
    residual = (torch.randn_like(x) / 4).contiguous() if add_residual else None

    reference = _production_reference(x, weight, residual, epsilon)
    actual = _fused_result(x, weight, residual, epsilon)

    _assert_matches_production(actual, reference, allow_subnormal_ulp=m >= 8192)


@pytest.mark.parametrize("n", MIMO_WIDTHS)
@pytest.mark.parametrize("add_residual", [False, True])
def test_mimo_fn_compat_group_range(monkeypatch, n, add_residual):
    x = torch.ones((3, n), dtype=torch.bfloat16, device="cuda")
    x[1].neg_()
    x[2].zero_()
    weight = (
        (torch.arange(n // 128, device="cuda") % 4 + 1)
        .repeat_interleave(128)
        .bfloat16()
    )
    weight[:128] = 0
    weight[::4] *= -0.5
    residual = x * 0.5 if add_residual else None

    monkeypatch.delenv(FN_COMPAT_ENV, raising=False)
    native = _fused_result(x, weight, residual, 1e-6)
    monkeypatch.setenv(FN_COMPAT_ENV, "1")
    compat = _fused_result(x, weight, residual, 1e-6)
    assert torch.equal(compat[0], native[0])
    if add_residual:
        assert torch.equal(compat[3], native[3])

    blocks = native[0].cpu().float().view(3, n // 128, 128)
    amax = blocks.abs().amax(-1).clamp_min(1e-10)
    native_max = torch.finfo(dtypes.fp8).max
    compat_max = 224.0 if get_gfx() == "gfx942" else native_max
    for (_, q, raw_scale, _), limit in ((native, native_max), (compat, compat_max)):
        scale = amax * (1.0 / limit)
        expected = (blocks * scale.reciprocal().unsqueeze(-1)).to(dtypes.fp8)
        assert torch.equal(raw_scale.cpu().view(n // 128, 3).T, scale)
        assert torch.equal(
            q.cpu().view(torch.uint8), expected.flatten(1).view(torch.uint8)
        )


@pytest.mark.parametrize("n", MIMO_WIDTHS)
@pytest.mark.parametrize("add_residual", [False, True])
@pytest.mark.parametrize("fn_compat", [False, True])
def test_mimo_rmsnorm_fp8_group_quant_replays_graph(
    n, add_residual, fn_compat, monkeypatch
):
    monkeypatch.setenv(FN_COMPAT_ENV, str(int(fn_compat)))
    torch.manual_seed(942)
    x = torch.randn((17, n), dtype=torch.bfloat16, device="cuda") / 4
    weight = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn_like(x) / 4 if add_residual else None
    _fused_result(x, weight, residual, 1e-6)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _fused_result(x, weight, residual, 1e-6)
    for factor in (0.0, 0.25, 1.0):
        x.fill_(factor)
        if residual is not None:
            residual.fill_(factor * 0.5)
        with monkeypatch.context() as replay_env:
            replay_env.setenv(FN_COMPAT_ENV, str(int(not fn_compat)))
            graph.replay()
        reference = _production_reference(x, weight, residual, 1e-6)
        for observed, expected in zip(actual, reference):
            if expected is not None:
                assert torch.equal(observed, expected)


@pytest.mark.parametrize(
    "path, rows, width",
    [
        ("mimo", 5, 2048),
        ("mimo", 5, 4096),
        ("split", 5, 512),
        ("generic", 5, 512),
        ("standalone", 129, 512),
    ],
)
def test_per128_fn_compat_dequant_vs_cpu_fn(monkeypatch, path, rows, width):
    monkeypatch.setenv(FN_COMPAT_ENV, "1")
    x = torch.randn(
        rows, width, device="cpu", generator=torch.Generator().manual_seed(33 + width)
    ).bfloat16()
    w = (
        torch.rand(
            width, device="cpu", generator=torch.Generator().manual_seed(34 + width)
        )
        + 0.5
    ).bfloat16()
    eps = 1e-6
    y = x.float()
    if path != "standalone":
        y = (
            ((y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)) * w.float())
            .bfloat16()
            .float()
        )
    blocks = y.reshape(rows, -1, 128)
    scale = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448
    golden = (blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale

    x, w = x.cuda(), w.cuda()
    if path in ("mimo", "split"):
        run = _fused_result if path == "mimo" else _production_reference
        _, q, raw, _ = run(x, w, None, eps)
        scale = raw.view(width // 128, rows).T
    elif path == "generic":
        q = torch.empty_like(x, dtype=dtypes.fp8)
        scale = torch.empty((rows, width // 128), dtype=torch.float32, device=x.device)
        aiter.rmsnorm_quant(q, x, scale, w, eps, 128)
    else:
        q, scale = get_hip_quant(QuantType.per_1x128)(x, quant_dtype=dtypes.fp8)
    actual = q.cpu().float().reshape_as(blocks) * scale.cpu().unsqueeze(-1)
    assert torch.isfinite(actual).all()
    bad = ~torch.isclose(actual, golden, rtol=0.01, atol=0.01)
    assert bad.sum().item() * 20 <= golden.numel()


def test_token_and_group_of_width_128(monkeypatch):
    x = torch.ones(128, dtype=torch.bfloat16, device="cuda")
    for flag in ("0", "1"):
        monkeypatch.setenv(FN_COMPAT_ENV, flag)
        token, _ = get_hip_quant(QuantType.per_Token)(
            x.view(1, 128), quant_dtype=dtypes.fp8
        )
        grouped, _ = get_hip_quant(QuantType.per_1x128)(x, quant_dtype=dtypes.fp8)
        native_max = torch.finfo(dtypes.fp8).max
        selected = 224 if flag == "1" and get_gfx() == "gfx942" else native_max
        assert torch.all(token.float() == native_max)
        assert torch.all(grouped.float() == selected)


def test_per128_flattens_contiguous_rows(monkeypatch):
    from aiter.ops import quant

    x = torch.empty((1, 2, 128), dtype=torch.bfloat16, device="cuda").transpose(0, 1)
    assert x.is_contiguous() and x.stride(-2) != 128
    launch = Mock()
    monkeypatch.setattr(quant, "dynamic_per_group_scaled_quant", launch)
    q, scale = quant.per_group_quant_hip(x, quant_dtype=dtypes.fp8)
    launch.assert_called_once()
    rows = launch.call_args.args[1]
    assert rows.shape == (2, 128) and rows.stride() == (128, 1)
    assert rows.data_ptr() == x.data_ptr()
    assert q.shape == x.shape and scale.shape == (2, 1, 1)


@pytest.mark.parametrize("flag", ["0", "1"])
def test_allreduce_per128_compat_dispatch(monkeypatch, flag):
    from aiter.dist.device_communicators.communicator_cuda import CudaCommunicator

    x = torch.ones((3, 256), dtype=torch.bfloat16, device="cuda")
    comm = Mock(world_size=2, _ar_1stage_max_kb=-1, _ar_1stage_override=True)
    comm.fused_allreduce_rmsnorm.return_value = (x, x)
    fused = comm.ca_comm.custom_fused_ar_rms_per_group_quant
    monkeypatch.setenv(FN_COMPAT_ENV, flag)
    # Force the existing fallback to obtain its real HIP FP8 output and scales.
    fused.return_value = None
    run = CudaCommunicator.fused_allreduce_rmsnorm_quant_per_group
    expected = run(comm, x, x, x[0], 1e-6, emit_bf16=True, transpose_scale=True)
    use_fallback = flag == "1" and get_gfx() == "gfx942"
    assert torch.all(
        expected[0].float() == (224 if use_fallback else torch.finfo(dtypes.fp8).max)
    )
    fused.return_value = expected
    comm.reset_mock()
    actual = run(comm, x, x, x[0], 1e-6, emit_bf16=True, transpose_scale=True)
    assert fused.call_count == int(not use_fallback)
    assert comm.fused_allreduce_rmsnorm.call_count == int(use_fallback)
    assert actual[2].stride() == (1, x.size(0))
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.parametrize("use_num_local_tokens", [False, True])
def test_prequantized_fused_moe_2stage_transposes_per_1x128_scales(
    monkeypatch, use_num_local_tokens
):
    calls = {"partial_transpose": 0, "quant_transpose_scale": []}

    def fake_partial_transpose(dst, src, num_rows=None):
        assert num_rows is not None
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

    monkeypatch.setattr(
        fused_moe_mod.aiter, "partial_transpose", fake_partial_transpose
    )
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
    model_dim = 256
    inter_dim = 128
    hidden_states = torch.empty((token_num, model_dim), dtype=dtypes.fp8)
    a1_scale = torch.arange(
        token_num * (model_dim // 128), dtype=torch.float32
    ).reshape(token_num, model_dim // 128)
    w1 = torch.empty((1, inter_dim * 2, model_dim), dtype=dtypes.fp8)
    w2 = torch.empty((1, model_dim, inter_dim), dtype=dtypes.fp8)
    sorted_ids = torch.zeros((token_num,), dtype=torch.int32)
    sorted_weights = torch.ones((token_num,), dtype=torch.float32)
    sorted_expert_ids = torch.zeros((1,), dtype=torch.int32)
    num_valid_ids = torch.tensor([token_num], dtype=torch.int32)
    moe_out = torch.empty((token_num, model_dim), dtype=torch.bfloat16)
    num_local_tokens = (
        torch.tensor([token_num], dtype=torch.int32) if use_num_local_tokens else None
    )

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
        num_local_tokens=num_local_tokens,
    )

    expected_a1_scale = (
        a1_scale + 1.0
        if use_num_local_tokens
        else a1_scale.transpose(0, 1).contiguous().view_as(a1_scale)
    )
    assert calls["partial_transpose"] == int(use_num_local_tokens)
    assert torch.equal(stage1_inputs["a1_scale"], expected_a1_scale)
    assert calls["quant_transpose_scale"] == [True]
    assert stage2_inputs["a2_scale"] is not None


@pytest.mark.parametrize("use_num_local_tokens", [False, True])
def test_prequantized_fused_moe_1stage_transposes_per_1x128_scales(
    monkeypatch, use_num_local_tokens
):
    calls = {"partial_transpose": 0}
    captured = {}

    def fake_partial_transpose(dst, src, num_rows=None):
        assert num_rows is not None
        calls["partial_transpose"] += 1
        dst.copy_(src + 1.0)

    def fake_fmoe(
        moe_buf,
        a1,
        w1,
        w2,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        a1_scale,
        w1_scale,
        w2_scale,
        kernel_name,
        **kwargs,
    ):
        captured["a1_scale"] = a1_scale

    monkeypatch.setattr(
        fused_moe_mod.aiter, "partial_transpose", fake_partial_transpose
    )
    monkeypatch.setattr(fused_moe_mod.aiter, "fmoe_fp8_blockscale_g1u1", fake_fmoe)

    token_num = 2
    model_dim = 256
    inter_dim = 128
    hidden_states = torch.empty((token_num, model_dim), dtype=dtypes.fp8)
    a1_scale = torch.arange(
        token_num * (model_dim // 128), dtype=torch.float32
    ).reshape(token_num, model_dim // 128)
    w1 = torch.empty((1, inter_dim * 2, model_dim), dtype=dtypes.fp8)
    w2 = torch.empty((1, model_dim, inter_dim), dtype=dtypes.fp8)
    sorted_ids = torch.zeros((token_num,), dtype=torch.int32)
    sorted_weights = torch.ones((token_num,), dtype=torch.float32)
    sorted_expert_ids = torch.zeros((1,), dtype=torch.int32)
    num_valid_ids = torch.tensor([token_num], dtype=torch.int32)
    moe_buf = torch.empty((token_num, model_dim), dtype=torch.bfloat16)
    num_local_tokens = (
        torch.tensor([token_num], dtype=torch.int32) if use_num_local_tokens else None
    )

    fused_moe_mod.fused_moe_1stage(
        hidden_states,
        w1,
        w2,
        topk=1,
        sorted_ids=sorted_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        moe_buf=moe_buf,
        isG1U1=True,
        block_size_M=256,
        quant_type=QuantType.per_1x128,
        q_dtype_a=dtypes.fp8,
        q_dtype_w=dtypes.fp8,
        w1_scale=torch.ones((1, inter_dim * 2 // 128, model_dim // 128)),
        w2_scale=torch.ones((1, model_dim // 128, inter_dim // 128)),
        a1_scale=a1_scale,
        a1_scale_is_transposed=False,
        num_local_tokens=num_local_tokens,
        M=token_num,
        device=hidden_states.device,
        doweight_stage1=False,
    )

    expected_a1_scale = (
        a1_scale + 1.0
        if use_num_local_tokens
        else a1_scale.transpose(0, 1).contiguous().view_as(a1_scale)
    )
    assert calls["partial_transpose"] == int(use_num_local_tokens)
    assert torch.equal(captured["a1_scale"], expected_a1_scale)
