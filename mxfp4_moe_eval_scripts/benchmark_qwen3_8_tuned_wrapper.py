#!/usr/bin/env python3
"""Benchmark the deployable single-layout tuned Qwen3.8 A16W4 wrapper."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from qwen3_8_mi308x_a16w4 import fused_moe, prepare_checkpoint_weights


DEFAULT_M = [1, 4, 8, 16, 32, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768]
HIDDEN = 8192
INTER = 256
EXPERTS = 512


def timed(fn, m: int) -> dict:
    """Measure the complete wrapper with CUDA events.

    AITER supplies all routing and compute operations; this helper only records
    end-to-end latency around the Qwen-specific composition.
    """
    reps = 15 if m <= 256 else 9 if m <= 2048 else 5 if m <= 16384 else 3
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end) * 1000.0))
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples": reps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=DEFAULT_M)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/qwen3_8_tuned_native_wrapper_tp8.json"),
    )
    args = parser.parse_args()

    torch.manual_seed(8128)
    w13 = torch.randint(
        0, 256, (EXPERTS, 2 * INTER, HIDDEN // 2), device="cuda", dtype=torch.uint8
    )
    w13_scale = torch.full(
        (EXPERTS, 2 * INTER, HIDDEN // 32), 120, device="cuda", dtype=torch.uint8
    )
    w2 = torch.randint(
        0, 256, (EXPERTS, HIDDEN, INTER // 2), device="cuda", dtype=torch.uint8
    )
    w2_scale = torch.full(
        (EXPERTS, HIDDEN, INTER // 32), 120, device="cuda", dtype=torch.uint8
    )
    weights = prepare_checkpoint_weights(w13, w13_scale, w2, w2_scale)
    rows = []
    output = args.output
    for m in args.m:
        x = torch.randn((m, HIDDEN), device="cuda", dtype=torch.bfloat16) * 0.02
        logits = torch.randn((m, EXPERTS), device="cuda", dtype=torch.float32)

        def run():
            return fused_moe(x, weights, logits)

        result = timed(run, m)
        row = {
            "m": m,
            **result,
            "effective_tflops": 6 * m * 10 * HIDDEN * INTER / (result["median_us"] * 1e-6) / 1e12,
        }
        rows.append(row)
        output.write_text(json.dumps({"scope": "routing plus tuned native A16W4 expert core", "results": rows}, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
