# SPDX-License-Identifier: MIT
"""Measure guarded MiMo BF16/FP8 LINEAR tiles in isolated subprocesses."""

import argparse
import json
import os
from pathlib import Path

import torch
import triton.testing
from ck_prefill_guard import check_accuracy, make_case, reference

from aiter import mha_batch_prefill_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--padding-baseline", action="store_true")
    parser.add_argument("--cache-dtype", choices=["bf16", "fp8"], default="bf16")
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--load-mode", choices=["global", "buffer"], default="global")
    ns = parser.parse_args()
    if ns.padding_baseline and ns.cache_dtype != "bf16":
        parser.error("--padding-baseline is only supported for BF16")
    rows = []
    for q_lens, kv_lens in [
        ([129], [257]),
        ([512], [16384]),
        ([3968], [16384]),
        ([16384], [16384]),
        ([32768], [65536]),
        ([3968, 16384, 12416], [16384, 16384, 12416]),
    ]:
        args, sink, k_ref, v_ref, scales = make_case(
            ns.library,
            q_lens,
            kv_lens,
            cache_dtype=ns.cache_dtype,
            load_mode=ns.load_mode,
        )

        def run(args=args, sink=sink, scales=scales):
            live_args = args
            if ns.padding_baseline:
                k = torch.nn.functional.pad(args[1], (0, 64))[..., :192]
                v = torch.nn.functional.pad(args[2], (0, 128))[..., :128]
                live_args = (args[0], k, v, *args[3:])
            return mha_batch_prefill_func(
                *live_args,
                causal=True,
                window_size=(ns.window, -1),
                sink_ptr=sink,
                **scales,
            )

        actual = run()
        q_ref = args[0].float() * scales.get("q_descale", 1.0)
        expected = reference(q_ref, k_ref, v_ref, q_lens, kv_lens, sink, ns.window)
        errors = check_accuracy(actual, expected, ns.cache_dtype)
        timings = [
            triton.testing.do_bench_cudagraph(run, rep=100) * 1000 for _ in range(3)
        ]
        row = {
            "tile": os.getenv(
                "AITER_MIMO_GFX942_FP8_LINEAR_TILE"
                if ns.cache_dtype == "fp8"
                else "AITER_MIMO_GFX942_LINEAR_TILE",
                "default",
            ),
            "cache_dtype": ns.cache_dtype,
            "window": ns.window,
            "load_mode": ns.load_mode,
            "padding": ns.padding_baseline,
            "q_lens": q_lens,
            "kv_lens": kv_lens,
            "median_us": sorted(timings)[1],
            "samples_us": timings,
            **errors,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    ns.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
    os._exit(0)
