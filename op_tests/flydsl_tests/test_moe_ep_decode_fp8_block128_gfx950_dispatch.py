# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The opt-in backend must delegate unsupported metadata to native AITER."""
import importlib
import pytest
import torch

pytest.importorskip("flydsl")
if not torch.cuda.is_available() or not torch.cuda.get_device_properties(
    0
).gcnArchName.startswith("gfx950"):
    pytest.skip("Requires gfx950", allow_module_level=True)

import aiter
from aiter.ops.flydsl.kernels.moe_ep_decode_fp8_block128_gfx950.configuration import (
    Stage1,
    OutputMode,
    WeightCache,
)
from aiter.ops.flydsl import moe_ep_decode_fp8_block128_gfx950 as backend

fm = importlib.import_module("aiter.fused_moe")


class NativePath(RuntimeError):
    pass


def inputs():
    cap, h, inter, experts, topk = 8, 1024, 128, 4, 2
    w1 = torch.empty(experts, 2 * inter, h, dtype=torch.float8_e4m3fn, device="cuda")
    w2 = torch.empty(experts, h, inter, dtype=torch.float8_e4m3fn, device="cuda")
    w1.is_shuffled = w2.is_shuffled = True
    return dict(
        hidden_states=torch.empty(cap, h, dtype=torch.float8_e4m3fn, device="cuda"),
        w1=w1,
        w2=w2,
        topk_ids=torch.empty(cap, topk, dtype=torch.int32, device="cuda"),
        topk_weight=torch.empty(cap, topk, device="cuda"),
        a1_scale=torch.empty(cap, h // 128, device="cuda"),
        w1_scale=torch.empty(experts, 2 * inter // 128, h // 128, device="cuda"),
        w2_scale=torch.empty(experts, h // 128, inter // 128, device="cuda"),
        expert_mask=torch.ones(8, dtype=torch.int32, device="cuda"),
        num_local_tokens=torch.zeros(1, dtype=torch.int32, device="cuda"),
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.per_1x128,
        dtype=torch.bfloat16,
        ep_has_fake_route=False,
        num_tokens_before_dispatch=8,
    )


def native(*args, **kwargs):
    raise NativePath("Native configuration selection reached")


@pytest.mark.parametrize(
    "unsupported",
    [
        "ids_dtype",
        "count_dtype",
        "count_shape",
        "scale_dtype",
        "mask_rank",
        "non_ep_experts",
        "strided_input",
    ],
)
def test_unsupported_metadata_falls_back(unsupported, monkeypatch):
    kwargs = inputs()
    if unsupported == "ids_dtype":
        kwargs["topk_ids"] = kwargs["topk_ids"].long()
    elif unsupported == "count_dtype":
        kwargs["num_local_tokens"] = kwargs["num_local_tokens"].long()
    elif unsupported == "count_shape":
        kwargs["num_local_tokens"] = torch.zeros(2, dtype=torch.int32, device="cuda")
    elif unsupported == "scale_dtype":
        kwargs["a1_scale"] = kwargs["a1_scale"].half()
    elif unsupported == "mask_rank":
        kwargs["expert_mask"] = kwargs["expert_mask"].view(1, -1)
    elif unsupported == "non_ep_experts":
        kwargs["expert_mask"] = kwargs["expert_mask"][: kwargs["w2"].shape[0]]
    else:
        kwargs["hidden_states"] = torch.empty(
            8, 2048, dtype=torch.float8_e4m3fn, device="cuda"
        )[:, ::2]
    monkeypatch.setattr(fm, "get_2stage_cfgs", native)
    for enabled in [False, True]:
        monkeypatch.setattr(
            fm, "_USE_FLYDSL_MOE_EP_DECODE_FP8_BLOCK128_GFX950", enabled
        )
        with pytest.raises(NativePath):
            fm.fused_moe(**kwargs)


def test_non_ep_direct_entry_rejected():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        kwargs = inputs()
        kwargs["expert_mask"] = kwargs["expert_mask"][: kwargs["w2"].shape[0]]
        args = tuple(
            kwargs[name]
            for name in (
                "hidden_states",
                "w1",
                "w2",
                "topk_ids",
                "topk_weight",
                "a1_scale",
                "w1_scale",
                "w2_scale",
                "expert_mask",
                "num_local_tokens",
            )
        )
        assert not backend.is_supported_ep_decode_fp8_block128_gfx950_input(*args)
        with pytest.raises(ValueError, match="fewer local than global experts"):
            backend.fused_moe_ep_decode_fp8_block128_gfx950(*args)


@pytest.mark.parametrize("source_tokens", [0, 1, 3, 8])
def test_supported_metadata_selects_opt_in(monkeypatch, source_tokens):
    kwargs = inputs()
    kwargs["num_tokens_before_dispatch"] = source_tokens
    marker = torch.empty(8, 1024, dtype=torch.bfloat16, device="cuda")
    monkeypatch.setattr(fm, "_USE_FLYDSL_MOE_EP_DECODE_FP8_BLOCK128_GFX950", True)
    monkeypatch.setattr(fm, "get_2stage_cfgs", native)
    calls = []

    def selected(*args, **options):
        calls.append(options)
        return marker

    monkeypatch.setattr(backend, "fused_moe_ep_decode_fp8_block128_gfx950", selected)
    assert fm.fused_moe(**kwargs).data_ptr() == marker.data_ptr()
    assert calls == [{"num_tokens_before_dispatch": source_tokens, "config": None}]


def test_public_dispatch_uses_dedicated_profile(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from aiter.ops.flydsl.kernels.moe_ep_decode_fp8_block128_gfx950.configuration import (
        MoeConfig,
    )
    import csv
    from aiter.ops.flydsl.kernels.moe_ep_decode_fp8_block128_gfx950.configuration import (
        Stage1,
    )

    kwargs = inputs()
    config = MoeConfig(stage2_n=128, stage1=Stage1.FUSED_FP8_QUANT_PARALLEL)
    path = tmp_path / "fmoe.csv"
    first, second = config.kernel_names()
    row = dict(
        gfx="gfx950",
        cu_num=fm.get_cu_num(),
        token=8,
        model_dim=1024,
        inter_dim=128,
        expert=4,
        global_expert=8,
        topk=2,
        source_tokens=8,
        act_type="ActivationType.Silu",
        dtype="torch.bfloat16",
        q_dtype_a="torch.float8_e4m3fn",
        q_dtype_w="torch.float8_e4m3fn",
        q_type="QuantType.per_1x128",
        use_g1u1=1,
        doweight_stage1=0,
        kernelName1=first,
        kernelName2=second,
    )
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    monkeypatch.setattr(
        fm,
        "AITER_CONFIGS",
        SimpleNamespace(AITER_CONFIG_FMOE_FILE=str(path)),
    )
    monkeypatch.setattr(fm, "_USE_FLYDSL_MOE_EP_DECODE_FP8_BLOCK128_GFX950", True)
    monkeypatch.setattr(fm, "get_2stage_cfgs", native)

    def selected(*args, **options):
        assert options["config"].kernel_names() == config.kernel_names()
        return torch.empty(8, 1024, dtype=torch.bfloat16, device="cuda")

    monkeypatch.setattr(backend, "fused_moe_ep_decode_fp8_block128_gfx950", selected)
    fm.fused_moe(**kwargs)


@pytest.mark.parametrize("block_m", [16, 32, 64])
def test_explicit_block_m_is_honored(monkeypatch, block_m):
    kwargs = inputs()
    kwargs["block_size_M"] = block_m
    monkeypatch.setattr(fm, "_USE_FLYDSL_MOE_EP_DECODE_FP8_BLOCK128_GFX950", True)
    monkeypatch.setattr(fm, "get_2stage_cfgs", native)

    def selected(*args, **options):
        assert options["config"].block_m == block_m
        assert options["config"].uses_atomic_output == (block_m == 16)
        return torch.empty(8, 1024, dtype=torch.bfloat16, device="cuda")

    monkeypatch.setattr(backend, "fused_moe_ep_decode_fp8_block128_gfx950", selected)
    fm.fused_moe(**kwargs)


def test_unsupported_explicit_block_m_uses_native(monkeypatch):
    monkeypatch.setattr(fm, "_USE_FLYDSL_MOE_EP_DECODE_FP8_BLOCK128_GFX950", True)
    monkeypatch.setattr(fm, "get_2stage_cfgs", native)
    with pytest.raises(NativePath):
        fm.fused_moe(**inputs(), block_size_M=48)


def test_projection_workspace_limit_without_allocating():
    from torch._subclasses.fake_tensor import FakeTensorMode

    # All input buffers and Q fit below2GiB, but FP32 gate/up scratch would not.
    with FakeTensorMode():
        cap, h, inter, experts, topk = 2048, 1024, 8192, 32, 32
        args = [
            torch.empty(cap, h, dtype=torch.float8_e4m3fn, device="cuda"),
            torch.empty(
                experts, 2 * inter, h, dtype=torch.float8_e4m3fn, device="cuda"
            ),
            torch.empty(experts, h, inter, dtype=torch.float8_e4m3fn, device="cuda"),
            torch.empty(cap, topk, dtype=torch.int32, device="cuda"),
            torch.empty(cap, topk, device="cuda"),
            torch.empty(cap, h // 128, device="cuda"),
            torch.empty(experts, 2 * inter // 128, h // 128, device="cuda"),
            torch.empty(experts, h // 128, inter // 128, device="cuda"),
            torch.empty(64, dtype=torch.int32, device="cuda"),
            torch.empty(1, dtype=torch.int32, device="cuda"),
        ]
        assert not backend.is_supported_ep_decode_fp8_block128_gfx950_input(
            *args, num_tokens_before_dispatch=4
        )
        with pytest.raises(ValueError, match="projection workspace"):
            backend._validate_inputs(*args, num_tokens_before_dispatch=4)
        assert backend.is_supported_ep_decode_fp8_block128_gfx950_input(*args)
