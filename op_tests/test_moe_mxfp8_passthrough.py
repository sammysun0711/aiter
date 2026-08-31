# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Numerical test for the MXFP8 activation passthrough in fused_moe.

The passthrough lets a caller hand over activations that are already fp8 with
group-32 e8m0 microscales, so fused_moe sorts the scale and skips
requantization. Its entire claim is that taking the shortcut changes nothing,
and that is testable exactly: pre-quantize with the same HIP MX quantizer the
internal path uses, and both sides feed identical fp8 bytes and identical scale
values to the same GEMM. The only thing left varying is which code sorts the
scale, so the two outputs must be *bit-identical* -- not merely close.

Exactness matters here because the failure mode is silent. The scale is consumed
bytewise, so a mis-sorted or mis-strided scale returns plausible-looking wrong
numbers rather than raising, and any tolerance wide enough to absorb quantizer
noise would also absorb a real defect. There is no quantizer noise to absorb.

Scope, stated so it is not overclaimed: this pins the passthrough against the
path it replaces. It is not an absolute-correctness test for the a8w4 MoE, which
op_tests/test_moe_2stage.py and op_tests/flydsl_tests/test_flydsl_moe_a8w4.py
already cover against a torch reference.

The harness mirrors how SGLang's MoRI dispatch calls this: per_1x32 MXFP4 expert
weights in the shuffled a8w4 layout and GateMode.INTERLEAVE. Swiglu tests force
the decode-sized activation policy to FP8. Plain-SiLU interleaved MXFP4 selects
FP8 directly on gfx950, so its regression runs with the default threshold.
"""

import os

import pytest
import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    fused_moe,
    fused_topk,
    get_2stage_cfgs,
    moe_sorting,
    torch_moe_stage2,
)
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.opus.moe_stage2_a8w4 import (
    opus_moe_stage2_a8w4_fwd,
    stage2_launch_config,
)
from aiter.ops.quant import (
    dynamic_per_group_scaled_quant,
    mxfp4_moe_sort_fwd,
    per_1x32_f4_quant,
    per_1x32_f8_scale_f8_quant,
    per_1x32_mx_quant_hip,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm device"),
    pytest.mark.skipif(get_gfx() not in ("gfx950",), reason="gfx950 a8w4 MoE required"),
]

# Decode-range token counts, including the single-token EP rank case.
TOKENS = [1, 8, 112]
MODEL_DIM = 2048
INTER_DIM = 512
EXPERTS = 8
TOPK = 2


@pytest.fixture(autouse=True)
def _force_fp8_activation_path():
    """Match serving: keep q_dtype_a at fp8 even for decode-sized batches."""
    old = os.environ.get("AITER_BF16_FP8_MOE_BOUND")
    os.environ["AITER_BF16_FP8_MOE_BOUND"] = "0"
    yield
    if old is None:
        os.environ.pop("AITER_BF16_FP8_MOE_BOUND", None)
    else:
        os.environ["AITER_BF16_FP8_MOE_BOUND"] = old


def _build(tokens, device="cuda", dtype=dtypes.bf16):
    torch.manual_seed(0)
    x = torch.randn(tokens, MODEL_DIM, dtype=dtype, device=device) / 10
    w1 = torch.randn(EXPERTS, INTER_DIM * 2, MODEL_DIM, dtype=dtype, device=device) / 10
    w2 = torch.randn(EXPERTS, MODEL_DIM, INTER_DIM, dtype=dtype, device=device) / 10
    score = torch.randn(tokens, EXPERTS, dtype=dtype, device=device)
    topk_weight, topk_ids = fused_topk(x, score, TOPK, True)

    w1_q, w1_scale = per_1x32_f4_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(EXPERTS, INTER_DIM * 2, MODEL_DIM // 2)
    w2_q = w2_q.view(EXPERTS, MODEL_DIM, INTER_DIM // 2)
    w1_shuffled = shuffle_weight_a16w4(w1_q, 16, True)
    w2_shuffled = shuffle_weight_a16w4(w2_q, 16, False)
    w1_shuffled.is_shuffled = True
    w2_shuffled.is_shuffled = True

    return (
        x,
        w1_shuffled,
        w2_shuffled,
        {
            "topk_weight": topk_weight,
            "topk_ids": topk_ids,
            "quant_type": QuantType.per_1x32,
            "activation": ActivationType.Swiglu,
            "gate_mode": GateMode.INTERLEAVE.value,
            "w1_scale": shuffle_scale_a16w4(w1_scale, EXPERTS, True),
            "w2_scale": shuffle_scale_a16w4(w2_scale, EXPERTS, False),
            "dtype": dtypes.bf16,
        },
    )


@pytest.mark.parametrize("tokens", TOKENS)
def test_passthrough_is_bit_identical_to_internal_quantization(tokens):
    """Taking the shortcut must change nothing at all."""
    x, w1, w2, kwargs = _build(tokens)

    internal = fused_moe(x, w1, w2, **kwargs)

    # Same quantizer the internal path uses, so the GEMM sees identical bytes.
    a1, a1_scale = per_1x32_mx_quant_hip(
        x,
        scale=None,
        quant_dtype=dtypes.fp8,
        scale_type=dtypes.fp8_e8m0,
        shuffle=False,
    )
    assert a1.dtype == dtypes.fp8 and a1_scale.dtype == dtypes.fp8_e8m0
    passthrough = fused_moe(a1, w1, w2, a1_scale=a1_scale, **kwargs)

    assert torch.isfinite(passthrough).all(), "passthrough produced non-finite output"
    assert torch.equal(passthrough, internal), (
        "passthrough is not bit-identical to internal requantization; the sorted "
        "activation scale is not the scale that belongs to these tokens "
        f"(max |delta| = {(passthrough.float() - internal.float()).abs().max().item():.3e})"
    )


def test_silu_interleave_uses_a8w4_below_default_threshold(monkeypatch):
    """Plain-SiLU interleaved MXFP4 must not fall into unsupported A16W4."""
    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "256")
    monkeypatch.setenv("AITER_FLYDSL_FORCE", "0")
    x, w1, w2, kwargs = _build(8)
    kwargs["activation"] = ActivationType.Silu

    internal = fused_moe(x, w1, w2, **kwargs)
    a1, a1_scale = per_1x32_mx_quant_hip(
        x,
        scale=None,
        quant_dtype=dtypes.fp8,
        scale_type=dtypes.fp8_e8m0,
        shuffle=False,
    )
    passthrough = fused_moe(a1, w1, w2, a1_scale=a1_scale, **kwargs)

    assert torch.isfinite(internal).all()
    assert torch.equal(passthrough, internal)


def test_ep_config_key_supports_legacy_and_routed_only_topk():
    """Both EP route conventions must select the routed-top-k=8 MiMo row."""
    common = (8192, 6144, 2048, 24)
    kwargs = {
        "dtype": torch.bfloat16,
        "q_dtype_a": dtypes.fp8,
        "q_dtype_w": dtypes.fp4x2,
        "q_type": QuantType.per_1x32,
        "use_g1u1": True,
        "activation": ActivationType.Silu,
        "doweight_stage1": False,
        "hidden_pad": 0,
        "intermediate_pad": 0,
        "is_shuffled": True,
        "gate_mode": GateMode.INTERLEAVE.value,
        "is_ep": True,
    }

    get_2stage_cfgs.cache_clear()
    legacy = get_2stage_cfgs(*common, 9, ep_has_fake_route=True, **kwargs)
    routed_only = get_2stage_cfgs(*common, 8, ep_has_fake_route=False, **kwargs)

    legacy_names = (
        legacy.stage1.keywords["kernelName"],
        legacy.stage2.keywords["kernelName"],
    )
    routed_only_names = (
        routed_only.stage1.keywords["kernelName"],
        routed_only.stage2.keywords["kernelName"],
    )
    assert legacy_names == routed_only_names
    assert legacy_names[0] == (
        "flydsl_moe1_afp8_wfp4_bf16_t64x128x256_w3_gui_persist_fp8"
    )
    assert legacy_names[1] == (
        "flydsl_moe2_afp8_wfp4_bf16_t64x128x256_atomic_bnt2_persist"
    )


@pytest.mark.parametrize("token", [32768, 131072])
@pytest.mark.parametrize("local_experts", [24, 48])
def test_mimo_production_capacity_selects_persistent_pair(token, local_experts):
    """Both target EP topologies and production tiers use the persistent pair."""

    get_2stage_cfgs.cache_clear()
    metadata = get_2stage_cfgs(
        token,
        6144,
        2048,
        local_experts,
        8,
        torch.bfloat16,
        dtypes.fp8,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        True,
        GateMode.INTERLEAVE.value,
        is_ep=True,
        ep_has_fake_route=False,
    )

    assert metadata.block_m == 128
    assert metadata.stage1.keywords["kernelName"] == (
        "flydsl_moe1_afp8_wfp4_bf16_t128x256x256_bnt0_gui_persist_split_ph8_fp8"
    )
    assert metadata.stage2.keywords["kernelName"] == (
        "flydsl_moe2_afp8_wfp4_bf16_t64x256x256_atomic_persist_async_sbm128"
    )


@pytest.mark.parametrize(
    ("tokens", "local_experts", "persistent"),
    [
        pytest.param(4096, 24, False, id="ep16-m4k-normal"),
        pytest.param(8192, 24, True, id="ep16-m8k-persistent"),
        pytest.param(16384, 24, True, id="ep16-m16k-persistent"),
        pytest.param(32768, 24, True, id="ep16-m32k-persistent"),
        pytest.param(8192, 48, False, id="ep8-m8k-normal"),
        pytest.param(16384, 48, False, id="ep8-m16k-normal"),
        pytest.param(32768, 48, True, id="ep8-m32k-persistent"),
    ],
)
def test_mimo_sparse_stage1_persistence_policy(tokens, local_experts, persistent):
    get_2stage_cfgs.cache_clear()
    metadata = get_2stage_cfgs(
        tokens,
        6144,
        2048,
        local_experts,
        8,
        torch.bfloat16,
        dtypes.fp8,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        True,
        GateMode.INTERLEAVE.value,
        is_ep=True,
        ep_has_fake_route=False,
    )

    kernel_name = metadata.stage1.keywords["kernelName"]
    assert ("_persist" in kernel_name) is persistent


def test_routed_only_ep_passes_safe_atomic_capacity(monkeypatch):
    import importlib

    fused_moe_module = importlib.import_module("aiter.fused_moe")
    captured = {}

    def fake_stage2(**kwargs):
        captured.update(kwargs)
        return kwargs["out"]

    monkeypatch.setattr(
        fused_moe_module.aiter.ops.flydsl, "flydsl_moe_stage2", fake_stage2
    )
    token_num, topk, model_dim = 64, 8, 128
    out = torch.empty((token_num, model_dim))
    fused_moe_module._flydsl_stage2_wrapper(
        inter_states=torch.empty((token_num, topk, 32)),
        w1=torch.empty(0),
        w2=torch.empty(0),
        sorted_token_ids=torch.empty(0, dtype=torch.int32),
        sorted_expert_ids=torch.empty(0, dtype=torch.int32),
        num_valid_ids=torch.empty(0, dtype=torch.int32),
        out=out,
        topk=topk,
        kernelName="flydsl_moe2_afp8_wfp4_bf16_t64x256x256_atomic",
        expert_mask=torch.empty(384, dtype=torch.int32),
        topk_ids=torch.empty((token_num, topk), dtype=torch.int32),
        ep_has_fake_route=False,
    )

    assert captured["atomic_token_capacity"] == token_num // topk

    captured.clear()
    fused_moe_module._flydsl_stage2_wrapper(
        inter_states=torch.empty((token_num, topk, 32)),
        w1=torch.empty(0),
        w2=torch.empty(0),
        sorted_token_ids=torch.empty(0, dtype=torch.int32),
        sorted_expert_ids=torch.empty(0, dtype=torch.int32),
        num_valid_ids=torch.empty(0, dtype=torch.int32),
        out=out,
        topk=topk,
        kernelName="flydsl_moe2_afp8_wfp4_bf16_t64x256x256_atomic",
        expert_mask=torch.empty(384, dtype=torch.int32),
        topk_ids=torch.empty((token_num, topk), dtype=torch.int32),
        ep_has_fake_route=True,
    )
    assert captured["atomic_token_capacity"] is None


def test_routed_only_ep_topk_reaches_fused_moe():
    """The public API must accept an EP mask without requiring a fake route."""
    x, w1, w2, kwargs = _build(8)
    kwargs["activation"] = ActivationType.Silu
    expert_mask = torch.ones(EXPERTS, dtype=torch.int32, device=x.device)

    reference = fused_moe(x, w1, w2, **kwargs)
    routed_only = fused_moe(
        x,
        w1,
        w2,
        expert_mask=expert_mask,
        ep_has_fake_route=False,
        **kwargs,
    )

    torch.testing.assert_close(routed_only, reference, atol=1.0, rtol=0.05)


def test_ep_a8w4_quantizes_only_num_local_tokens(monkeypatch):
    """The padded graph capacity must not drive stage-1 input quantization."""
    import importlib

    fused_moe_module = importlib.import_module("aiter.fused_moe")
    original_quant = fused_moe_module.fused_dynamic_mxfp8_quant_moe_sort
    captured_num_rows = []

    def capture_num_rows(*args, **kwargs):
        captured_num_rows.append(kwargs.get("num_rows"))
        return original_quant(*args, **kwargs)

    monkeypatch.setattr(
        fused_moe_module,
        "fused_dynamic_mxfp8_quant_moe_sort",
        capture_num_rows,
    )

    tokens, valid_tokens = 16, 8
    x, w1, w2, kwargs = _build(tokens)
    kwargs["activation"] = ActivationType.Silu
    num_local_tokens = torch.tensor([valid_tokens], dtype=torch.int32, device=x.device)
    expert_mask = torch.ones(EXPERTS, dtype=torch.int32, device=x.device)

    padded = fused_moe(
        x,
        w1,
        w2,
        expert_mask=expert_mask,
        num_local_tokens=num_local_tokens,
        ep_has_fake_route=False,
        **kwargs,
    )

    assert captured_num_rows == [num_local_tokens, num_local_tokens]
    assert torch.isfinite(padded[:valid_tokens]).all()

    def force_full_quant(*args, **kwargs):
        kwargs["num_rows"] = None
        return original_quant(*args, **kwargs)

    monkeypatch.setattr(
        fused_moe_module,
        "fused_dynamic_mxfp8_quant_moe_sort",
        force_full_quant,
    )
    full_quant = fused_moe(
        x,
        w1,
        w2,
        expert_mask=expert_mask,
        num_local_tokens=num_local_tokens,
        ep_has_fake_route=False,
        **kwargs,
    )
    torch.testing.assert_close(
        padded[:valid_tokens], full_quant[:valid_tokens], atol=0, rtol=0
    )


def test_device_row_limited_mxfp8_quant_graph_replay():
    """Persistent row-limited quantization must track the replay-time row count."""
    torch.manual_seed(13)
    capacity, cols = 8192, 512
    x = torch.randn((capacity, cols), dtype=torch.bfloat16, device="cuda") / 8
    full_out = torch.empty((capacity, cols), dtype=dtypes.fp8, device="cuda")
    full_scale = torch.empty(
        (capacity, cols // 32), dtype=dtypes.fp8_e8m0, device="cuda"
    )
    dynamic_per_group_scaled_quant(full_out, x, full_scale, 32, False)

    out = torch.empty_like(full_out)
    scale = torch.empty_like(full_scale)
    num_rows = torch.tensor([257], dtype=torch.int32, device="cuda")
    dynamic_per_group_scaled_quant(
        out, x, scale, 32, False, num_rows=num_rows, num_rows_factor=1
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dynamic_per_group_scaled_quant(
            out, x, scale, 32, False, num_rows=num_rows, num_rows_factor=1
        )

    for live_rows in (257, 1025):
        num_rows.fill_(live_rows)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out[:live_rows], full_out[:live_rows])
        assert torch.equal(scale[:live_rows], full_scale[:live_rows])


def _mimo_opus_metadata(tokens, output_dtype):
    return get_2stage_cfgs(
        tokens,
        6144,
        256,
        384,
        8,
        torch.bfloat16,
        dtypes.fp8,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        True,
        GateMode.INTERLEAVE.value,
        opus_stage2_output_dtype=output_dtype,
    )


@pytest.mark.parametrize(
    ("tokens", "sort_block_m", "stage1_schedule"),
    [
        (8192, 64, "t64x256x256_w4_bnt0_gui_xcd4_fp8"),
        (16384, 128, "t128x256x256_bnt0_gui_xcd4_ph8_fp8"),
        (32768, 64, "t64x256x256_w2_bnt0_gui_xcd4_fp8"),
        (65536, 64, "t64x256x256_w2_bnt0_gui_xcd4_fp8"),
    ],
)
def test_mimo_opus_stage2_output_dtype_selects_matching_kernel(
    tokens, sort_block_m, stage1_schedule
):
    """The public config switch must cover every MiMo production token tier."""

    get_2stage_cfgs.cache_clear()
    auto = _mimo_opus_metadata(tokens, "auto")
    fp8 = _mimo_opus_metadata(tokens, "fp8")
    bf16 = _mimo_opus_metadata(tokens, "bf16")

    auto_name = auto.stage2.keywords["kernelName"]
    fp8_name = fp8.stage2.keywords["kernelName"]
    bf16_name = bf16.stage2.keywords["kernelName"]

    assert (
        auto.stage1.keywords["kernelName"]
        == f"flydsl_moe1_afp8_wfp4_bf16_{stage1_schedule}"
    )
    suffix = f"t64x256x256_sbm{sort_block_m}_rbn6144_xw8"
    assert auto_name == fp8_name
    assert auto_name == f"opus_moe2_layout_afp8_wfp4_fp8_{suffix}"
    assert bf16_name == f"opus_moe2_layout_afp8_wfp4_bf16_{suffix}"


def test_opus_stage2_output_dtype_does_not_reject_non_opus_decode_config():
    """The global route-output preference must not disturb decode kernels."""

    get_2stage_cfgs.cache_clear()
    metadata = get_2stage_cfgs(
        128,
        6144,
        2048,
        24,
        8,
        torch.bfloat16,
        dtypes.fp8,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        True,
        GateMode.INTERLEAVE.value,
        opus_stage2_output_dtype="bf16",
    )

    assert metadata.stage2.keywords["kernelName"].startswith("flydsl_moe2_")


def test_opus_stage2_output_dtype_rejects_unknown_value():
    get_2stage_cfgs.cache_clear()
    with pytest.raises(ValueError, match="opus_stage2_output_dtype"):
        _mimo_opus_metadata(8192, "int8")


def test_opus_stage2_route_output_formats_match_torch():
    """Both registered W=8 route formats must preserve stage-2 accuracy."""

    torch.manual_seed(11)
    torch.cuda.manual_seed(11)
    tokens, model_dim, inter_dim, experts, topk, block_m = 128, 6144, 256, 1, 1, 64

    inter = (
        torch.randn((tokens, topk, inter_dim), dtype=torch.bfloat16, device="cuda") / 8
    )
    w2 = (
        torch.randn(
            (experts, model_dim, inter_dim), dtype=torch.bfloat16, device="cuda"
        )
        / 8
    )
    topk_ids = torch.zeros((tokens, topk), dtype=torch.int32, device="cuda")
    topk_weights = torch.ones((tokens, topk), dtype=torch.float32, device="cuda")
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids,
        topk_weights,
        experts,
        model_dim,
        torch.bfloat16,
        block_m,
    )

    inter_q, inter_scale = per_1x32_f8_scale_f8_quant(
        inter, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    w2_q, w2_scale = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2)
    w2_q = w2_q.view(experts, model_dim, inter_dim // 2)
    w1_q = torch.empty(
        (experts, inter_dim * 2, model_dim // 2),
        dtype=dtypes.fp4x2,
        device="cuda",
    )
    reference = torch_moe_stage2(
        inter_q,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=torch.bfloat16,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale,
        a2_scale=inter_scale,
        doweight=True,
    )

    inter_scale_sorted = mxfp4_moe_sort_fwd(
        inter_scale,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=tokens,
        cols=inter_dim,
    )
    w2_shuffled = shuffle_weight_a16w4(w2_q, 16, False)
    w2_scale_shuffled = shuffle_scale_a16w4(w2_scale, experts, False)

    outputs = {}
    kernel_names = {
        "fp8": "opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm64_rbn6144_xw8",
        "bf16": "opus_moe2_afp8_wfp4_bf16_t64x256x256_sbm64_rbn6144_xw8",
    }
    from csrc.opus_moe.opus_moe_common import opus_a8w4_stage2_instance_from_name

    for output_dtype, kernel_name in kernel_names.items():
        instance = opus_a8w4_stage2_instance_from_name(kernel_name)
        assert instance is not None
        outputs[output_dtype] = opus_moe_stage2_a8w4_fwd(
            inter_q,
            w2_shuffled,
            inter_scale_sorted,
            w2_scale_shuffled,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            launch=stage2_launch_config(instance.kid),
            inter_dim_pad=0,
            token_num=tokens,
            topk=topk,
        )
        torch.testing.assert_close(
            outputs[output_dtype], reference, atol=1.0, rtol=0.05
        )

    fp8_error = (outputs["fp8"].float() - reference.float()).abs().mean()
    bf16_error = (outputs["bf16"].float() - reference.float()).abs().mean()
    assert bf16_error <= fp8_error
