# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Time native block-scale tuner candidates that pass a sampled signed-input FP64 check."""

import argparse
import csv
import math
import sys
from functools import partial
from pathlib import Path
from statistics import median

import torch
from bench_fp8_scaling_comparison import capture, time_group
from common import Inputs, error_metric, opus_offsets_fit, packed_scales, read_csv

# Import the checkout's tuner and its sibling kernel registries.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "csrc/ck_gemm_a8w8_blockscale")
)
from gemm_a8w8_blockscale_tune import BLOCK_PER_CU_MAX, GemmA8W8BlockScaleTuner

COLUMNS = [
    "gfx",
    "cu_num",
    "M",
    "N",
    "K",
    "libtype",
    "kernelId",
    "splitK",
    "us",
    "kernelName",
    "tflops",
    "bw",
    "errRatio",
]


def candidates(tuner, shape, family, packed):
    key = ("gfx942", 304, *shape)
    if family == "ck":
        return tuner.get_gemm_a8w8_blockscale_tune_task(key, False, 0, packed, {})
    if family == "cktile":
        return tuner.get_gemm_a8w8_blockscale_cktile_tune_task(
            key, False, 0, packed, range(1, BLOCK_PER_CU_MAX + 1), {}
        )
    if family == "asm":
        return tuner.get_gemm_a8w8_blockscale_asm_tune_task(key, True, 0, packed, {})
    if family == "opus":
        return tuner.get_gemm_a8w8_blockscale_opus_tune_task(key, 0, packed, {})
    return tuner.get_gemm_a8w8_blockscale_flydsl_tune_task(key, 0, packed, {})


def qualify(shape, family, layout, tuner, profile):
    packed = layout == "packed"
    m, n, k = shape
    if family == "opus" and not opus_offsets_fit(*shape):
        print(f"{shape}: OPUS signed row offset exceeds INT_MAX", flush=True)
        return None
    d = Inputs(shape)
    ri, ci, _, reference = d.reference()
    tensors = {
        "x": d.x,
        "weight_shuffle": d.packed,
        "weight": d.w,
        "x_scale": d.bx,
        "x_scale_t": packed_scales(d.bx),
        "w_scale": d.bw,
        "out": d.out,
        "zero_bias": torch.zeros((1, n), dtype=torch.float32, device="cuda"),
    }
    stream = torch.cuda.Stream()
    best = None
    for task in candidates(tuner, shape, family, packed):
        _, kid, split, name, _, _ = task[0]
        keys, *params = task[4]
        fn = partial(task[3], *(tensors[key] for key in keys), *params)
        row = {
            "gfx": "gfx942",
            "cu_num": 304,
            "M": m,
            "N": n,
            "K": k,
            "libtype": family,
            "kernelId": kid,
            "splitK": split,
            "kernelName": name or tuner.getKernelName(kid, family, packed),
        }
        try:
            checks = [
                error_metric(fn, d.out, ri, ci, reference)
                for _ in range(5 if family == "opus" else 2)
            ]
            graph = capture(fn, stream, 3)
            checks.append(error_metric(graph.replay, d.out, ri, ci, reference))
            us = median(time_group(graph, 3) for _ in range(5))
            checks.append(error_metric(graph.replay, d.out, ri, ci, reference))
        except ValueError as exc:
            if "sampled numerical check failed" not in str(exc):
                raise
            profile.writerow(dict(row, weight_layout=layout, reason=str(exc)))
            continue
        except RuntimeError as exc:
            if not any(
                s in str(exc).lower()
                for s in ("not supported", "unsupported", "alignment check")
            ):
                raise
            profile.writerow(dict(row, weight_layout=layout, reason=str(exc)))
            continue
        sample = d.out[ri[:, None], ci[None, :]].cpu().double()
        bad = (
            (~torch.isclose(sample, reference, rtol=0.01, atol=0.01))
            .double()
            .mean()
            .item()
        )
        row.update(
            us=us,
            tflops=2 * math.prod(shape) / us / 1e6,
            bw=(m * k + n * k + 2 * m * n) / us / 1000,
            errRatio=bad,
        )
        profile.writerow(
            dict(
                row,
                weight_layout=layout,
                relative_rms=max(x for x, _ in checks),
                max_over_rms=max(y for _, y in checks),
            )
        )
        if best is None or us < best["us"]:
            best = row
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--shape", nargs=3, type=int, metavar=("M", "N", "K"))
    inputs.add_argument("--input", type=Path, help="CSV with M, N and K columns")
    p.add_argument(
        "--layout",
        choices=("ordinary", "packed"),
        default="packed",
        help="Weight layout to qualify. Write a separate runtime CSV for each layout.",
    )
    p.add_argument(
        "--family", choices=("ck", "cktile", "asm", "opus", "flydsl"), required=True
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Runtime CSV; candidate diagnostics go beside it",
    )
    args = p.parse_args()
    shapes = (
        [tuple(args.shape)]
        if args.shape
        else [tuple(int(r[t]) for t in ("M", "N", "K")) for r in read_csv(args.input)]
    )
    props = torch.cuda.get_device_properties(0)
    if not props.gcnArchName.startswith("gfx942") or props.multi_processor_count != 304:
        p.error("the MI300X replay requires 304-CU gfx942")
    tuner = GemmA8W8BlockScaleTuner("MI300X", ["gfx", "cu_num", "M", "N", "K"], ["us"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    winners = []
    with args.output.with_suffix(".profile.csv").open("w", newline="") as fp:
        profile = csv.DictWriter(
            fp, [*COLUMNS, "weight_layout", "relative_rms", "max_over_rms", "reason"]
        )
        profile.writeheader()
        for shape in shapes:
            row = qualify(shape, args.family, args.layout, tuner, profile)
            fp.flush()
            if row:
                winners.append(row)
            print(
                shape,
                row["kernelName"] if row else "no qualified direct row",
                flush=True,
            )
    # Publish the runtime table only once all requested shapes have been screened.
    tmp = args.output.with_suffix(".tmp")
    with tmp.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, COLUMNS)
        writer.writeheader()
        writer.writerows(winners)
    tmp.replace(args.output)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
