# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Export an all-shape MI300X PTPC / block-scale comparison from graph replay samples."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
SIZES = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
PROJECTIONS = ((1536, 4096), (32768, 1536), (4096, 32768))
LABELS = {
    "ptpc_torch": "PyTorch, ordinary",
    "ptpc_tgemm_plain": "tgemm hipBLASLt, ordinary",
    "ptpc_tgemm_packed": "tgemm hipBLASLt, packed",
    "ptpc_aiter_native": "AITER PTPC operator, packed",
    "block_triton": "Triton, ordinary",
    "block_ck_plain": "Classic CK, ordinary",
    "block_ck": "Classic CK, packed",
    "block_cktile": "CK Tile, packed",
    "block_asm": "ASM, packed",
    "block_opus": "OPUS, packed",
    "block_flydsl": "FlyDSL original tuned port, packed",
    "block_lds_m": "FlyDSL two-stage LDS, M-first",
    "block_lds_n": "FlyDSL two-stage LDS, N-first",
    "block_register": "FlyDSL transposed register cache, M-first",
    "tensor_channel_torch": "PyTorch, ordinary",
    "tensor_channel_tgemm_plain": "tgemm hipBLASLt, ordinary",
    "tensor_channel_tgemm_packed": "tgemm hipBLASLt, packed",
    "tensor_channel_aiter_native": "AITER PTPC operator, packed",
}
GROUPS = ("ptpc", "block", "tensor_channel")
TITLES = {
    "ptpc": "Per-token activation and per-channel weight",
    "block": "K128 block scaling",
    "tensor_channel": "One activation multiplier and per-channel weight",
}


def keys(group):
    return [name for name in LABELS if name.startswith(group + "_")]


def export(raw, output):
    records, wide = {}, []
    for n, k in PROJECTIONS:
        for m in SIZES:
            shape = m, n, k
            data = json.loads((raw / ("-".join(map(str, shape)) + ".json")).read_text())
            assert tuple(data["shape"]) == shape and set(data["methods"]) == set(LABELS)
            methods = data["methods"]
            records[shape] = methods
            row = {"M": m, "N": n, "K": k}
            chunks, missing = [], []
            for name in LABELS:
                item = methods[name]
                col = name + "_tflops"
                if item["status"] != "passed":
                    row[col] = ""
                    missing.append(f'{name}: {item["detail"]}')
                    continue
                samples = item["rounds_us"]
                assert len(samples) == data["rounds"] and all(
                    len(g) == data["groups"] for g in samples
                )
                us = median(v for g in samples for v in g)
                assert math.isclose(
                    2 * m * n * k / us / 1e6, item["tflops"], rel_tol=1e-12
                )
                assert item["relative_rms"] < 0.006 and item["max_over_rms"] < 0.035
                assert item["rows"] * item["calls"] == m
                row[col] = item["tflops"]
                if item["calls"] > 1:
                    chunks.append(f'{name}={item["calls"]}x{item["rows"]}')
            for group in GROUPS:
                winner = max(
                    (key for key in keys(group) if methods[key]["status"] == "passed"),
                    key=lambda key: methods[key]["tflops"],
                )
                row[group + "_winner"] = winner
                row[group + "_best_tflops"] = methods[winner]["tflops"]
            row["ptpc_vs_block_pct"] = 100 * (
                row["ptpc_best_tflops"] / row["block_best_tflops"] - 1
            )
            row["row_chunks"] = " | ".join(chunks)
            row["unavailable"] = " | ".join(missing)
            wide.append(row)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "mi300x-fp8-ptpc-vs-blockscale.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(wide[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(wide)
    return records, wide


def markdown(records, raw, output):
    guide = ROOT.parents[2] / "docs/mi300x-fp8-gemm-tuning-guide.md"
    report = [
        "# MI300X FP8 GEMM comparison",
        (
            f"Median TFLOP/s on one 304-CU gfx942 GPU. Each cell includes all row chunks. "
            f"† marks a chunked result. A dash indicates an unavailable or rejected measurement. "
            f"The CSV gives the chunk sizes and reasons. [Tuning guide]({os.path.relpath(guide,output)}) · "
            f"[Unrounded CSV](mi300x-fp8-ptpc-vs-blockscale.csv) · [Raw timing samples]({os.path.relpath(raw,output)}/)"
        ),
    ]
    for n, k in PROJECTIONS:
        report.append(f"## Projection {k:,} → {n:,}")
        for group in GROUPS:
            report.append("### " + TITLES[group])
            table = [
                "| Kernel | " + " | ".join(f"M={m:,}" for m in SIZES) + " |",
                "|:---|" + "---:|" * len(SIZES),
            ]
            for name in keys(group):
                cells = []
                for m in SIZES:
                    rec = records[m, n, k][name]
                    cells.append(
                        f"{rec['tflops']:,.1f}" + ("†" if rec["calls"] > 1 else "")
                        if rec["status"] == "passed"
                        else "—"
                    )
                table.append("| " + LABELS[name] + " | " + " | ".join(cells) + " |")
            report.append("\n".join(table))
    (output / "mi300x-fp8-ptpc-vs-blockscale.md").write_text("\n\n".join(report) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "results/cross-scaling")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    records, _ = export(args.raw, args.output)
    markdown(records, args.raw, args.output)
