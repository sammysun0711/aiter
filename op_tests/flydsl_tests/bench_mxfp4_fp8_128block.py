# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Benchmark MXFP4 -> FP8 conversion at MiMo-V2.6 expert weight shapes.

Measures both conversion kernels on a HIP graph with preallocated outputs.
Reports repeated access and single conversions after a 512 MiB cache flush.
Excludes allocation, compilation, host validation, cache flushing, shuffle and MoE.
"""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch

from aiter.ops.flydsl import mxfp4_to_fp8_128block


def benchmark(experts, n, k, *, device, seed, repeats, hot_rounds, cold_rounds, flush):
    generator = torch.Generator(device=device).manual_seed(seed)
    packed = torch.empty((experts, n, k // 2), dtype=torch.uint8, device=device)
    packed.random_(0, 256, generator=generator)
    codes = torch.empty((experts, n, k // 32), dtype=torch.uint8, device=device)
    codes.random_(123, 132, generator=generator)
    q, s = mxfp4_to_fp8_128block(packed, codes)

    def run():
        mxfp4_to_fp8_128block(packed, codes, out=q, out_scale=s)

    # Independent reference: validate every tile scale and sampled FP8 bytes,
    # including first/last elements and offsets beyond 4 GiB in large cases.
    headroom = 5 if q.dtype == torch.float8_e4m3fnuz else 6
    maxima = codes.reshape(experts, n // 128, 128, k // 128, 4).amax((2, 4))
    expected_s = torch.exp2((maxima.int() - 127 - headroom).clamp_min(-126).float())
    ids = torch.linspace(0, q.numel() - 1, min(4096, q.numel()), dtype=torch.int64)
    ids_device = ids.to(device)
    selected_packed = packed.flatten()[ids_device // 2].cpu().long()
    nibbles = torch.where(ids % 2 == 0, selected_packed & 15, selected_packed >> 4)
    selected_codes = codes.flatten()[ids_device // 32].cpu()
    tile_ids = (
        ids // (n * k) * (n // 128) * (k // 128)
        + (ids // k % n // 128) * (k // 128)
        + ids % k // 128
    )
    selected_scale = expected_s.flatten()[tile_ids.to(device)].cpu().double()
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float64,
    )
    weights = lut[nibbles] * torch.exp2(selected_codes.double() - 127)
    expected_q = (weights / selected_scale).float().to(q.dtype).view(torch.uint8)

    def validate():
        torch.testing.assert_close(s, expected_s, rtol=0, atol=0)
        actual = q.view(torch.uint8).flatten()[ids_device].cpu()
        torch.testing.assert_close(actual, expected_q, rtol=0, atol=0)

    for _ in range(5):
        run()
    torch.cuda.synchronize(device)
    validate()
    hot = torch.cuda.CUDAGraph()
    with torch.cuda.graph(hot):
        for _ in range(repeats):
            run()
    cold = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cold):
        run()
    # Both graph paths must write both outputs on replay.
    for graph in (hot, cold):
        q.view(torch.uint8).fill_(85)
        s.fill_(-1)
        graph.replay()
        validate()

    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    samples = {"repeated_us": [], "flushed_us": []}
    for mode, rounds, graph, divisor in (
        ("repeated_us", hot_rounds, hot, repeats),
        ("flushed_us", cold_rounds, cold, 1),
    ):
        for _ in range(rounds):
            if mode == "flushed_us":
                flush.add_(1)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples[mode].append(start.elapsed_time(end) * 1000 / divisor)
        validate()
    return {
        "logical_shape": [experts, n, k],
        "packed_shape": list(packed.shape),
        "scale_shape": list(s.shape),
        "output_dtype": str(q.dtype),
        "validated_fp8_elements": ids.numel(),
        "all_tile_scales_exact": True,
        "sampled_fp8_bytes_exact": True,
        "samples_us": samples,
        **{key: statistics.median(value) for key, value in samples.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experts", type=int, nargs="+", default=[1, 48, 62, 96, 192, 384]
    )
    parser.add_argument("--tp", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--hot-rounds", type=int, default=7)
    parser.add_argument("--cold-rounds", type=int, default=28)
    parser.add_argument("--flush-mib", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(
            args.experts
            + args.tp
            + [args.repeats, args.hot_rounds, args.cold_rounds, args.flush_mib]
        )
        <= 0
    ):
        parser.error("dimensions and iteration counts must be positive")
    if any(2048 % tp or (2048 // tp) % 128 for tp in args.tp):
        parser.error(
            "TP must divide 2048 and leave a multiple of 128 intermediate channels"
        )
    if not torch.version.hip or not torch.cuda.is_available():
        parser.error("Requires ROCm")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    props = torch.cuda.get_device_properties(device)
    if props.gcnArchName.split(":")[0] not in ("gfx942", "gfx950"):
        parser.error("Requires gfx942 or gfx950")
    flush = torch.zeros(
        args.flush_mib * 1024**2 // 4, dtype=torch.float32, device=device
    )
    root = Path(__file__).resolve().parents[2]
    sources = [
        "aiter/ops/flydsl/mxfp4_fp8.py",
        "aiter/ops/flydsl/kernels/mxfp4_fp8_128block.py",
        "aiter/ops/flydsl/kernels/mxfp4_fp8_common.py",
        "op_tests/flydsl_tests/bench_mxfp4_fp8_128block.py",
    ]
    report = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device_index": args.device,
        "device": props.name,
        "arch": props.gcnArchName,
        "total_memory_bytes": props.total_memory,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "flydsl": importlib.metadata.version("flydsl"),
        "source_sha256": {
            p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources
        },
        "methodology": {
            "seed": args.seed,
            "synthetic_inputs": True,
            "e8m0_codes_inclusive": [123, 131],
            "repeats": args.repeats,
            "hot_rounds": args.hot_rounds,
            "cold_rounds": args.cold_rounds,
            "flush_mib": args.flush_mib,
            "timing": "HIP graph + GPU events; median; both kernels; preallocated outputs",
            "clocks_locked": False,
            "excludes": [
                "allocation",
                "compilation",
                "host validation",
                "cache flush",
                "shuffle",
                "GEMM",
                "MoE",
            ],
        },
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for tp in args.tp:
        inter = 2048 // tp
        for experts in args.experts:
            for projection, n, k in (
                ("gate_up", 2 * inter, 6144),
                ("down", 6144, inter),
            ):
                row = benchmark(
                    experts,
                    n,
                    k,
                    device=device,
                    seed=args.seed,
                    repeats=args.repeats,
                    hot_rounds=args.hot_rounds,
                    cold_rounds=args.cold_rounds,
                    flush=flush,
                )
                row.update({"tp": tp, "experts": experts, "projection": projection})
                report["results"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    json.dumps({k: v for k, v in row.items() if k != "samples_us"}),
                    flush=True,
                )
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
