#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reference-checked qlen-1/4/8 PA tile benchmark (requires AITER test helpers).

Run from the AITER repository root with the selected FlyDSL runtime and AITER
checkout on PYTHONPATH. No FlyDSL source checkout is needed. Query/output are BF16; --dtypes selects KV dtype.
Setup, full-reference checks, and compilation are outside the timed region.
AITER's profiler reports average GPU time, including the split reducer, with
its normal sample filtering. This is not median or host/API latency.
Effective bandwidth counts valid KV, Q/O, scales, and page/context metadata
once per call. It is logical decimal TB/s, not hardware-counter HBM traffic.

Example:
    python op_tests/op_benchmarks/flydsl/bench_pa_decode_tile.py --batch 32 --context 16384 -o decode.csv
"""

import argparse
import csv
import functools
import hashlib
import itertools
import json
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dtypes", nargs="+", choices=("bf16", "fp8"), default=["bf16", "fp8"]
    )
    parser.add_argument(
        "--qlens", nargs="+", type=int, choices=(1, 4, 8), default=[1, 4, 8]
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        choices=("128,128", "192,128", "192,192"),
        default=["128,128", "192,128", "192,192"],
        metavar="QK,V",
        help="Width pairs: 128,128 192,128 192,192",
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument(
        "--context",
        type=int,
        default=16384,
        help="KV length, including the verification tokens",
    )
    parser.add_argument("--page-size", type=int, choices=(16, 64), default=64)
    parser.add_argument(
        "--parts",
        type=int,
        default=8,
        help="Fixed number of KV partitions (not auto-tuned)",
    )
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument(
        "--iters", type=int, default=50, help="Requested samples per profiler round"
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup calls per round")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write results to a new CSV file; existing files are not overwritten",
    )
    args = parser.parse_args()
    for name in ("batch", "context", "parts", "q_heads", "kv_heads", "rounds"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.iters < 2 or args.warmup < 0:
        parser.error("--iters must be at least 2 and --warmup must be nonnegative")
    if args.context < max(args.qlens):
        parser.error("--context must be at least the largest query length")
    if args.q_heads % args.kv_heads:
        parser.error("--q-heads must be divisible by --kv-heads")
    if args.output and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    return args


def benchmark_case(args, dtype, qlen, qk_dim, v_dim):
    import torch
    from aiter.test_common import run_perftest

    from aiter.ops.flydsl import pa_decode as tile
    from op_tests import test_flydsl_pa_decode as test_pa

    original = tile.pa_decode_tile
    calls = []

    def record(*pos, **kw):
        # Forward the real launch; retain its inputs/workspaces after the test.
        calls.append((pos, kw))
        return original(*pos, **kw)

    tile.pa_decode_tile = record
    try:
        test_pa.test_tile_pa_vectorized_5d_matches_torch(
            compute_type=dtype,
            query_length=qlen,
            value_head_size=v_dim,
            num_partitions=args.parts,
            head_size=qk_dim,
            num_heads=(args.q_heads, args.kv_heads),
            context_length=args.context,
            batch_size=args.batch,
            block_size=args.page_size,
            entrypoint="direct",
        )
    finally:
        tile.pa_decode_tile = original

    assert len(calls) == 1, "expected one reference-checked direct launch"
    pos, kw = calls[0]
    expected = kw["output"].clone()
    fn = functools.partial(original, *pos, **kw)
    averages = []
    for _ in range(args.rounds):
        _, us = run_perftest(
            fn, num_iters=args.iters, num_warmup=args.warmup, num_rotate_args=1
        )
        averages.append(float(us))
        torch.testing.assert_close(kw["output"], expected, rtol=0, atol=0)

    # Uniform contexts reference each physical page once. Count valid KV only,
    # plus Q/O, scalar scales and metadata; exclude padding and split scratch.
    kv_bytes = (
        args.batch
        * args.context
        * args.kv_heads
        * (qk_dim + v_dim)
        * (2 if dtype == "bf16" else 1)
    )
    logical_bytes = kv_bytes
    for name in (
        "query",
        "output",
        "block_tables",
        "context_lengths",
        "key_scale",
        "value_scale",
    ):
        tensor = kw[name]
        if tensor is not None:
            logical_bytes += tensor.numel() * tensor.element_size()
    avg_us = statistics.mean(averages)

    return {
        "kv_dtype": dtype,
        "qlen": qlen,
        "qk_dim": qk_dim,
        "v_dim": v_dim,
        "batch": args.batch,
        "context": args.context,
        "q_heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "page_size": args.page_size,
        "parts": args.parts,
        "iters": args.iters,
        "warmup": args.warmup,
        "rounds": args.rounds,
        "avg_us": avg_us,
        "avg_us_rounds": json.dumps(averages),
        "kv_bytes": kv_bytes,
        "kv_tb_s": kv_bytes / (avg_us * 1e6),
        "logical_bytes": logical_bytes,
        "effective_tb_s": logical_bytes / (avg_us * 1e6),
        "query_sha256": hashlib.sha256(
            kw["query"].view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest(),
        "output_sha256": hashlib.sha256(
            expected.view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest(),
        "accuracy": "PASS-full-reference-and-repeated-output",
    }


def main():
    args = parse_args()
    # Keep --help and argument validation independent of GPU/runtime imports.
    import torch

    import flydsl
    from flydsl.runtime.device import get_rocm_arch
    from aiter.ops.flydsl import pa_decode as tile

    print(
        "METADATA "
        + json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "arch": str(get_rocm_arch()),
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "flydsl": flydsl.__version__,
                "runtime": flydsl.__file__,
                "source": tile.__file__,
                "source_sha256": hashlib.sha256(
                    Path(tile.__file__).read_bytes()
                ).hexdigest(),
                "timing": "mean-of-aiter-profiler-round-averages-us",
            }
        ),
        flush=True,
    )
    results = []
    for dtype, qlen, shape in itertools.product(args.dtypes, args.qlens, args.shapes):
        qk_dim, v_dim = map(int, shape.split(","))
        print(
            f"Running {dtype} qlen={qlen} D{qk_dim}/V{v_dim} B={args.batch} KV={args.context}",
            flush=True,
        )
        row = benchmark_case(args, dtype, qlen, qk_dim, v_dim)
        results.append(row)
        print("RESULT " + json.dumps(row), flush=True)

    print("\nKV dtype  Qlen  QK/V       Average us  Accuracy")
    for row in results:
        shape = f"{row['qk_dim']}/{row['v_dim']}"
        print(
            f"{row['kv_dtype']:8} {row['qlen']:5}  {shape:9} {row['avg_us']:11.4f}  PASS"
        )
    if args.output:
        with args.output.open("x", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"CSV: {args.output}")


if __name__ == "__main__":
    main()
