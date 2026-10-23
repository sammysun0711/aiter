# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU tests for the profile codec and source/capacity lookup."""
import csv
import pytest

# AITER's top-level import initializes its GPU backends. Load these two pure
# modules directly so profile selection remains testable on a CPU-only host.
import importlib.util
from pathlib import Path
import sys

_ROOT = Path(__file__).parents[2]
_NAME = "aiter.ops.flydsl.kernels.moe_ep_decode_fp8_block128_gfx950.configuration"


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load("aiter.utility.moe_config", "aiter/utility/moe_config.py")
_profiles = _load(
    _NAME, "aiter/ops/flydsl/kernels/moe_ep_decode_fp8_block128_gfx950/configuration.py"
)

MoeConfig, TAG = _profiles.MoeConfig, _profiles.TAG
Stage1, OutputMode, WeightCache = (
    _profiles.Stage1,
    _profiles.OutputMode,
    _profiles.WeightCache,
)
lookup_profile = _profiles.lookup_profile


def write_profiles(path, rows):
    previous = list(csv.DictReader(path.open())) if path.exists() else []
    rows = previous + rows
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _profiles._read_profiles.cache_clear()
    lookup_profile.cache_clear()


def profile(tokens, config, source=-1):
    first, second = config.kernel_names()
    return dict(
        gfx="gfx950",
        cu_num=256,
        token=tokens,
        model_dim=6144,
        inter_dim=2048,
        expert=48,
        global_expert=384,
        topk=8,
        source_tokens=source,
        act_type="ActivationType.Silu",
        dtype="torch.bfloat16",
        q_dtype_a="torch.float8_e4m3fn",
        q_dtype_w="torch.float8_e4m3fn",
        q_type="QuantType.per_1x128",
        use_g1u1=1,
        doweight_stage1=0,
        kernelName1=first,
        kernelName2=second,
        _tag=TAG,
    )


@pytest.mark.parametrize(
    "config",
    [
        MoeConfig(),
        MoeConfig(stage2_n=128, stage2_cache=WeightCache.ADAPTIVE),
        MoeConfig(
            block_m=16, stage1=Stage1.SEPARATE_FP8_QUANT, output=OutputMode.ATOMIC
        ),
        MoeConfig(block_m=16),
        MoeConfig(block_m=32, output=OutputMode.ATOMIC),
        MoeConfig(
            stage1_cache=_profiles.WeightCache.ADAPTIVE,
            stage2_cache=_profiles.WeightCache.ADAPTIVE,
        ),
        MoeConfig(
            stage1=Stage1.SEPARATE_FP8_QUANT,
            stage1_cache=_profiles.WeightCache.ADAPTIVE,
            stage2_cache=_profiles.WeightCache.STREAM,
        ),
        MoeConfig(stage1=Stage1.SEPARATE_FP8_QUANT_TRANSPOSED, quant_tiled=True),
        MoeConfig(
            block_m=64, stage1=Stage1.FUSED_FP8_QUANT_PARALLEL, down_adaptive_m=True
        ),
        MoeConfig(
            block_m=64,
            stage1_n=128,
            stage2_n=128,
            stage1_workers=4,
            stage2_workers=4,
            stage1=Stage1.SEPARATE_FP8_QUANT_TRANSPOSED,
        ),
        MoeConfig(
            stage1_n=128,
            stage1=Stage1.SEPARATE_FP8_QUANT,
            stage1_cache=True,
            stage2_cache=False,
        ),
    ],
)
def test_kernel_name_roundtrip(config):
    names = config.kernel_names()
    assert MoeConfig.from_kernel_names(*names).kernel_names() == names


def test_source_workload_and_exact_override(tmp_path):
    path = tmp_path / "fmoe.csv"
    small, large = MoeConfig(stage1=Stage1.SEPARATE_FP8_QUANT), MoeConfig(
        stage2_n=128, stage1_cache=True, stage2_cache=True
    )
    write_profiles(path, [profile(64, small), profile(1024, large)])

    def lookup(capacity, source=None, global_e=384):
        return lookup_profile(path, 256, capacity, 6144, 2048, 48, global_e, 8, source)

    assert lookup(1024, 8).kernel_names() == small.kernel_names()
    assert lookup(1024).kernel_names() == large.kernel_names()
    assert lookup(672, 84) is None
    assert lookup(1024, 8, global_e=768) is None
    exact = MoeConfig(
        block_m=16, stage1=Stage1.SEPARATE_FP8_QUANT, output=OutputMode.ATOMIC
    )
    write_profiles(path, [profile(1024, exact, source=8)])
    assert lookup(1024, 8).kernel_names() == exact.kernel_names()


def test_reject_mismatched_stage_tiles():
    first, _ = MoeConfig(block_m=32).kernel_names()
    _, second = MoeConfig(block_m=64, stage1=Stage1.SEPARATE_FP8_QUANT).kernel_names()
    with pytest.raises(ValueError, match="kernel names"):
        MoeConfig.from_kernel_names(first, second)


def test_legacy_defaults_and_cache_aliases_are_canonical():
    first = _profiles.PREFIX + "s1_m16_n64_proj_t0_c0_w0_k1_a0_np0"
    second = _profiles.PREFIX + "s2_m16_n64_c0_w8_a1_am0"
    c = MoeConfig.from_kernel_names(first, second)
    assert c.stage1_workers == 8
    assert c.output == OutputMode.ATOMIC
    assert MoeConfig.from_kernel_names(*c.kernel_names()) == c

    first = first.replace("_m16_", "_m32_").replace("_proj_", "_fused_")
    second = second.replace("_m16_", "_m32_") + "_rc1"
    c = MoeConfig.from_kernel_names(first, second)
    assert c.stage2_cache == WeightCache.ADAPTIVE
    assert "_c2_" in c.kernel_names()[1]
    assert c.output == OutputMode.PARTIAL


def test_retired_schedule_flags_normalize_to_maintained_implementations():
    config = MoeConfig(stage1=Stage1.SEPARATE_FP8_QUANT_TRANSPOSED, stage1_n=128)
    one, two = config.kernel_names()
    old_one = one.replace("_a0_", "_a1_").replace("_sa1", "")
    old_two = two.replace("_a1_", "_a0_").replace("_rv8", "_fr1")
    assert MoeConfig.from_kernel_names(old_one, old_two) == config
