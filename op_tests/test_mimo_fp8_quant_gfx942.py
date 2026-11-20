# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Customer-style MiMo dequantization checks against independent PyTorch CPU goldens.

Run with python -m pytest -q op_tests/test_mimo_fp8_quant_gfx942.py to print the Markdown tables.
"""

from itertools import groupby
from typing import NamedTuple

import pytest
import torch

# Load the terminal-summary hook when pytest collects this standalone test file.
pytest_plugins = (__name__,)

GROUP = 128
EPSILON = 1e-6
ATOL = RTOL = 1e-2
FN = torch.float8_e4m3fn
FNUZ = torch.float8_e4m3fnuz
CASES = {
    "rmsnorm_quant_k4096": (5, 4096),
    "rmsnorm_quant_k512": (5, 512),
    "rmsnorm_quant_k2048": (5, 2048),
    "quant_to_fp8_contiguous": (129, 512),
}


class Comparison(NamedTuple):
    label: str
    arch: str
    compat: bool
    golden_dtype: torch.dtype


COMPARISONS = [
    Comparison("AITER FNUZ /240 vs CPU FNUZ /240 (gfx942)", "gfx942", False, FNUZ),
    Comparison("AITER FN /448 vs CPU FN /448 (gfx950)", "gfx950", False, FN),
    Comparison("AITER FNUZ /224 vs CPU FNUZ /224 (gfx942)", "gfx942", True, FNUZ),
    Comparison("AITER FNUZ /224 vs CPU FN /448 (gfx942)", "gfx942", True, FN),
]


def _build(case):
    rows, width = CASES[case]
    x = torch.randn(
        (rows, width),
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator(device="cpu").manual_seed(33 + width),
    )
    inputs = {"x": x.bfloat16()}
    if case != "quant_to_fp8_contiguous":
        w = torch.rand(
            width,
            device="cpu",
            dtype=torch.float32,
            generator=torch.Generator(device="cpu").manual_seed(34 + width),
        )
        inputs["weight"] = (w + 0.5).bfloat16()
    return inputs


def _run(inputs, gpu_dtype):
    from aiter import QuantType, get_hip_quant, rmsnorm
    from aiter.ops.rmsnorm import mimo_rmsnorm_fp8_group_quant

    x = inputs["x"].cuda()
    rows, width = x.shape
    if "weight" in inputs:
        weight = inputs["weight"].cuda()
        normalized = torch.empty_like(x)
        if width == 512:
            # Reconstruct the K=512 wrapper with the unfused production path.
            rmsnorm(normalized, x, weight, EPSILON)
            q, raw_scale = get_hip_quant(QuantType.per_1x128)(
                normalized, quant_dtype=gpu_dtype, transpose_scale=True
            )
        else:
            q = torch.empty_like(x, dtype=gpu_dtype)
            raw_scale = torch.empty(
                (rows, width // GROUP), device=x.device, dtype=torch.float32
            )
            mimo_rmsnorm_fp8_group_quant(q, normalized, raw_scale, x, weight, EPSILON)
        scale = raw_scale.view(width // GROUP, rows).T
    else:
        q, scale = get_hip_quant(QuantType.per_1x128)(
            x, quant_dtype=gpu_dtype, transpose_scale=False
        )
    assert q.dtype == gpu_dtype
    return {"dequant": (q.float() * scale.repeat_interleave(GROUP, dim=1)).cpu()}


def _cpu_golden(inputs, comparison):
    y = inputs["x"].float()
    dtype = comparison.golden_dtype
    if "weight" in inputs:
        inv_rms = torch.rsqrt(y.square().mean(dim=1, keepdim=True) + EPSILON)
        weighted = (y * inv_rms) * inputs["weight"].float()
        if dtype == FNUZ:
            # Opus' default gfx942 FP32 -> BF16 conversion truncates finite values.
            bits = weighted.contiguous().view(torch.int32) >> 16
            y = bits.to(torch.int16).view(torch.bfloat16).float()
        else:
            y = weighted.bfloat16().float()
    blocks = y.unflatten(1, (-1, GROUP))
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    if dtype == FNUZ:
        limit = 224 if comparison.compat else 240
        scale = amax * (1.0 / limit)
        scaled = blocks * scale.reciprocal()
    else:
        limit = 448
        scale = amax / limit
        scaled = blocks / scale
    q = scaled.clamp(-limit, limit).to(dtype)
    return {"dequant": (q.float() * scale).flatten(1)}


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize(
    "comparison",
    COMPARISONS,
    ids=["native-fnuz", "native-fn", "compat-fnuz", "compat-vs-fn"],
)
def test_mimo_fp8_cpu_golden(case, comparison, monkeypatch, record_property):
    # Record the identity before skipping so a report distinguishes missing hardware.
    record_property("mimo_fp8_case", (comparison.label, case))
    record_property(
        "mimo_fp8_order", (COMPARISONS.index(comparison), list(CASES).index(case))
    )
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("A ROCm GPU is required")
    gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = gpu.gcnArchName.split(":")[0]
    record_property("mimo_fp8_device", f"{gpu.name} ({arch})")
    if arch != comparison.arch:
        pytest.skip(
            f"This comparison requires {comparison.arch}; available GPU is {arch}"
        )
    monkeypatch.setenv("AITER_MIMO_RMSNORM_FP8_FN_COMPAT", str(int(comparison.compat)))

    inputs = _build(case)
    actual = _run(inputs, FNUZ if arch == "gfx942" else FN)["dequant"]
    golden = _cpu_golden(inputs, comparison)["dequant"]
    assert actual.shape == golden.shape and actual.device == golden.device
    finite = torch.isfinite(actual) & torch.isfinite(golden)
    close = torch.isclose(actual, golden, atol=ATOL, rtol=RTOL)
    bad = (~(finite & close)).sum().item()
    total = golden.numel()
    stats = ["—"] * 4
    if finite.all():
        units = (actual - golden).abs() / (ATOL + RTOL * golden.abs())
        levels = units.new_tensor([1.0, 0.99, 0.95, 0.90])
        stats = [f"{v:.3f}" for v in torch.quantile(units.flatten(), levels).tolist()]
    record_property("mimo_fp8_stats", (total, bad, *stats))
    assert finite.all(), f"{case} produced non-finite dequantized values"
    assert (
        bad * 20 <= total
    ), f"{case}: {bad}/{total} values outside atol={ATOL}, rtol={RTOL} (5% budget)"


def pytest_terminal_summary(terminalreporter):
    rows = []
    devices = set()
    for reports in terminalreporter.stats.values():
        for report in reports:
            if getattr(report, "when", None) != "call":
                continue
            props = dict(report.user_properties)
            if "mimo_fp8_case" not in props:
                continue
            title, case = props["mimo_fp8_case"]
            if "mimo_fp8_device" in props:
                devices.add(props["mimo_fp8_device"])
            status = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP"}[
                report.outcome
            ]
            if "mimo_fp8_stats" in props:
                total, bad, *errors = props["mimo_fp8_stats"]
                stats = [total, bad, f"{bad / total:.3%}", *errors]
            else:
                stats = ["—"] * 7
            cells = [case, "dequant", status, *stats, "1e-2 / 1e-2"]
            rows.append(
                (
                    props["mimo_fp8_order"],
                    title,
                    "| " + " | ".join(map(str, cells)) + " |",
                )
            )
    if not rows:
        return

    lines = [
        "# MiMo FP8 dequantization accuracy",
        "",
        "Hardware — " + (", ".join(sorted(devices)) or "no ROCm GPU available"),
        "",
        "A bad point exceeds `0.01 + 0.01 * abs(CPU golden)` in absolute error. "
        "PASS allows up to 5% bad points and requires finite outputs. "
        "Max and percentiles use absolute error divided by that tolerance, over all elements. "
        "SKIP indicates an unavailable GPU format.",
        "",
        "The FNUZ CPU golden uses gfx942 BF16 truncation and reciprocal-multiply quantization. "
        "The FN CPU golden uses nearest-even BF16 and FP32 division. Each table names the actual "
        "GPU storage format and group-scale divisor.",
        "",
        "RMSNorm uses five rows, epsilon 1e-6, and CPU seeds 33+K for activations and 34+K for weights. "
        "K=4096 and K=2048 use the MiMo fused op. K=512 uses separate AITER RMSNorm and HIP quantization. "
        "The standalone case reconstructs 66,048 elements as contiguous BF16 [129, 512] with seed 545.",
    ]
    for title, group in groupby(sorted(rows), key=lambda row: row[1]):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Kernel | 输出 | 结果 | 总点数 | 坏点数 | 坏点数 / 总点数 (%) | max | p99 | p95 | p90 | atol / rtol |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        lines.extend(row[2] for row in group)
    terminalreporter.write_line("\n" + "\n".join(lines))
