#!/usr/bin/env python3
"""Tune Qwen3.8 TP8 A16W4 MoE tiles on gfx942.

This benchmark intentionally uses the production local shape:

* hidden size 8192
* TP-local expert intermediate size 256
* 512 experts
* top-k 10

It times stage 1 and stage 2 independently.  Stage-2 timings include the
existing grouped reduction because ``moe_gemm_a16w4`` performs that reduction
before returning.  Candidate outputs are checked against the current fallback
configuration before they are eligible to win.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from aiter.ops.triton.moe.moe_op_gemm_a16w4 import (
    get_kernel_config,
    moe_gemm_a16w4,
)
from aiter.ops.triton.moe.moe_routing.routing import routing


HIDDEN = 8192
INTERMEDIATE = 256
EXPERTS = 512
TOPK = 10


def _candidate_key(config: dict) -> tuple:
    return tuple(sorted(config.items()))


def _variants(seed: dict, stage: str) -> list[dict]:
    """Generate a bounded coordinate-descent search around ``seed``."""
    dimensions = [
        ("block_n", [32, 64, 128, 256, 512]),
        ("block_k", [128, 256]),
        ("num_warps", [2, 4, 8]),
        ("num_stages", [1, 2, 3]),
        ("group_m", [1, 2, 4, 8]),
        ("xcd_swizzle", [1, 8]),
        ("waves_per_eu", [0, 1, 2, 3, 4]),
        ("w_cache_modifier", [None, ".cg"]),
        # gfx942/MI300 tuning guidance recommends mfma_16x16 for GEMM.
        # Keep this fixed rather than accepting a faster-looking MFMA32 result.
        ("matrix_instr_nonkdim", [16]),
        ("kpack", [1, 2]),
        ("split_k", [1, 2, 4] if stage == "stage1" else [1]),
    ]
    seen = set()
    ret = []
    for key, values in dimensions:
        for value in values:
            candidate = dict(seed)
            candidate[key] = value
            frozen = _candidate_key(candidate)
            if frozen not in seen:
                seen.add(frozen)
                ret.append(candidate)
    return ret


def _measure(fn, warmup: int, reps: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(reps):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        output = fn()
        end.record()
        end.synchronize()
        samples.append(float(begin.elapsed_time(end)))

    return {
        "output": output,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _measure_pair(fn_a, fn_b, warmup: int, reps: int) -> tuple[dict, dict]:
    """Measure two finalists in alternating order to reduce clock-drift bias."""
    for _ in range(warmup):
        fn_a()
        fn_b()
    torch.cuda.synchronize()

    samples_a = []
    samples_b = []
    output_a = None
    output_b = None

    def measure_once(fn):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        output = fn()
        end.record()
        end.synchronize()
        return output, float(begin.elapsed_time(end))

    for rep in range(reps):
        if rep % 2 == 0:
            output_a, elapsed_a = measure_once(fn_a)
            output_b, elapsed_b = measure_once(fn_b)
        else:
            output_b, elapsed_b = measure_once(fn_b)
            output_a, elapsed_a = measure_once(fn_a)
        samples_a.append(elapsed_a)
        samples_b.append(elapsed_b)

    def summarize(output, samples):
        return {
            "output": output,
            "median_ms": statistics.median(samples),
            "mean_ms": statistics.mean(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
        }

    return summarize(output_a, samples_a), summarize(output_b, samples_b)


def _accuracy(reference: torch.Tensor, actual: torch.Tensor) -> dict:
    ref = reference.float()
    out = actual.float()
    diff = (ref - out).abs()
    denom = torch.linalg.vector_norm(ref).clamp_min(1.0e-30)
    rel_l2 = torch.linalg.vector_norm(diff) / denom
    return {
        "finite": bool(torch.isfinite(out).all().item()),
        "max_abs": float(diff.max().item()),
        "rel_l2": float(rel_l2.item()),
    }


def _is_valid(metrics: dict) -> bool:
    return (
        metrics["finite"]
        and metrics["max_abs"] <= 6.25e-2
        and metrics["rel_l2"] <= 1.0e-2
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--stage", choices=["stage1", "stage2", "both"], default="both")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--final-reps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=8128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    logits = torch.randn((args.m, EXPERTS), device="cuda", dtype=torch.float32)
    hidden_states = (
        torch.randn((args.m, HIDDEN), device="cuda", dtype=torch.bfloat16) * 0.02
    )
    routing_data, gather_idx, scatter_idx = routing(logits, TOPK)

    w13_storage = torch.randint(
        0,
        256,
        (EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2),
        device="cuda",
        dtype=torch.uint8,
    )
    w13_scale_storage = torch.full(
        (EXPERTS, 2 * INTERMEDIATE, HIDDEN // 32),
        120,
        device="cuda",
        dtype=torch.uint8,
    )
    w2_storage = torch.randint(
        0,
        256,
        (EXPERTS, HIDDEN, INTERMEDIATE // 2),
        device="cuda",
        dtype=torch.uint8,
    )
    w2_scale_storage = torch.full(
        (EXPERTS, HIDDEN, INTERMEDIATE // 32),
        120,
        device="cuda",
        dtype=torch.uint8,
    )

    w13 = w13_storage.transpose(1, 2)
    w13_scale = w13_scale_storage.transpose(1, 2)
    w2 = w2_storage.transpose(1, 2)
    w2_scale = w2_scale_storage.transpose(1, 2)

    def stage1(config=None):
        return moe_gemm_a16w4(
            hidden_states,
            w13,
            None,
            w13_scale,
            routing_data=routing_data,
            gather_indx=gather_idx,
            out_dtype=torch.bfloat16,
            kernel_config=config,
        )

    baseline_stage1_config = get_kernel_config(
        gather_idx.shape[0], 2 * INTERMEDIATE, HIDDEN, routing_data
    )
    baseline_stage1 = stage1(baseline_stage1_config)
    intermediate = (
        F.silu(baseline_stage1[:, :INTERMEDIATE].float())
        * baseline_stage1[:, INTERMEDIATE:].float()
    ).to(torch.bfloat16)

    def stage2(config=None):
        return moe_gemm_a16w4(
            intermediate,
            w2,
            None,
            w2_scale,
            routing_data=routing_data,
            scatter_indx=scatter_idx,
            gammas=routing_data.gate_scal,
            out_dtype=torch.bfloat16,
            kernel_config=config,
        )

    baseline_stage2_config = get_kernel_config(
        gather_idx.shape[0], HIDDEN, INTERMEDIATE, routing_data
    )
    baseline_stage2 = stage2(baseline_stage2_config)
    torch.cuda.synchronize()

    functions = {"stage1": stage1, "stage2": stage2}
    references = {"stage1": baseline_stage1, "stage2": baseline_stage2}
    initial = {
        "stage1": baseline_stage1_config,
        "stage2": baseline_stage2_config,
    }
    stages = ["stage1", "stage2"] if args.stage == "both" else [args.stage]
    report = {
        "seed": args.seed,
        "shape": {
            "m": args.m,
            "expanded_m": int(gather_idx.shape[0]),
            "expert_blocks": int(
                routing_data.n_blocks(gather_idx.shape[0], routing_data.block_m)
            ),
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "experts": EXPERTS,
            "topk": TOPK,
            "block_m": routing_data.block_m,
        },
        "stages": {},
    }

    for stage in stages:
        fn = functions[stage]
        reference = references[stage]
        best_config = dict(initial[stage])
        attempts = []

        for pass_index in range(args.passes):
            candidates = _variants(best_config, stage)
            pass_results = []
            for index, config in enumerate(candidates):
                row = {"pass": pass_index, "candidate": index, "config": config}
                try:
                    measured = _measure(
                        lambda config=config: fn(config), args.warmup, args.reps
                    )
                    accuracy = _accuracy(reference, measured.pop("output"))
                    row.update(measured)
                    row.update(accuracy)
                    row["valid"] = _is_valid(accuracy)
                except Exception as exc:
                    torch.cuda.synchronize()
                    row.update(
                        {
                            "valid": False,
                            "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
                        }
                    )
                attempts.append(row)
                if row["valid"]:
                    pass_results.append(row)
                print(json.dumps({"stage": stage, **row}, sort_keys=True), flush=True)

            if not pass_results:
                raise RuntimeError(f"No valid {stage} candidates in pass {pass_index}")
            best_config = dict(
                min(pass_results, key=lambda x: x["median_ms"])["config"]
            )

        baseline, final = _measure_pair(
            lambda: fn(initial[stage]),
            lambda: fn(best_config),
            args.warmup + 2,
            args.final_reps,
        )
        final_accuracy = _accuracy(reference, final.pop("output"))
        baseline_accuracy = _accuracy(reference, baseline.pop("output"))
        report["stages"][stage] = {
            "baseline_config": initial[stage],
            "baseline": {**baseline, **baseline_accuracy},
            "best_config": best_config,
            "best": {**final, **final_accuracy},
            "speedup": baseline["median_ms"] / final["median_ms"],
            "attempts": attempts,
        }

    print("FINAL " + json.dumps(report, sort_keys=True), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
