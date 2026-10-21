# SPDX-License-Identifier: MIT
"""Native FP8 codegen must stay scoped to gfx942 LINEAR per-tensor kernels."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

GENERATOR = (
    Path(__file__).resolve().parents[1]
    / "3rdparty/composable_kernel/example/ck_tile/01_fmha/generate.py"
)
pytestmark = pytest.mark.skipif(not GENERATOR.exists(), reason="requires CK source")


def generate(tmp_path, tile, target="gfx942", dtype="fp8bf16"):
    scale = "pertensor" if dtype == "fp8bf16" else "nqscale"
    pattern = (
        f"*d192_{dtype}_group_ps1*_nlogits_nbias_mask_nlse_ndropout_"
        f"{scale}_sink_linear_sglang"
    )
    subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "-d",
            "batch_prefill",
            "--targets",
            target,
            "--receipt",
            "200",
            "--filter",
            pattern,
            "--output_dir",
            str(tmp_path),
        ],
        env={
            **os.environ,
            "AITER_MIMO_GFX942_FP8_LINEAR_TILE": tile,
            "AITER_MIMO_GFX942_LINEAR_TILE": "128x64",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    return sorted(tmp_path.glob(f"fmha_batch_prefill_d192_{dtype}_group_ps1_*.cpp"))


@pytest.mark.parametrize(
    "tile", ["64x64", "64x96", "64x128", "128x64", "128x96", "128x128"]
)
def test_fp8_linear_native_tiles(tmp_path, tile):
    files = generate(tmp_path, tile)
    assert len(files) == 2  # BUFFER_LOAD and GLOBAL_LOAD_LDS.
    m, n = tile.split("x")
    for file in files:
        source = file.read_text()
        assert f"ck_tile::sequence<{m}, {n}, 32, 128, 32, 192>" in source
        assert "BlockAttentionQuantScaleEnum::PERTENSOR" in source
        assert "BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT" in source
        assert "BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D" in source


def test_fp8_linear_generic_excludes_native(tmp_path):
    assert generate(tmp_path, "generic") == []


def test_fp8_linear_override_does_not_enable_gfx950(tmp_path):
    assert generate(tmp_path, "128x64", target="gfx950") == []


def test_fp8_linear_override_does_not_change_bf16(tmp_path):
    files = generate(tmp_path, "64x96", dtype="bf16")
    assert len(files) == 2
    for file in files:
        assert "ck_tile::sequence<128, 64, 32, 128, 32, 192>" in file.read_text()
