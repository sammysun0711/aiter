#!/usr/bin/env python3
"""Tune MiMo QKV CK B-preshuffle GEMMs with a concurrent TP all-reduce.

Example:

    torchrun --standalone --nproc-per-node=8 \
      op_tests/op_benchmarks/bench_mi325x_mimo_qkv_ck_overlap.py \
      --channel 112 --output /tmp/mi325x_qkv_overlap.json

The script calls only ``gemm_a8w8_blockscale_bpreshuffle_tune``. It does not
use the production PyHIP GEMM dispatcher.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight


N = 3392
K = 6144
SCALE_BLOCK = 128
KERNEL_NAMES = {
    2: "a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3",
    15: "a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1",
    17: "a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1",
}


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def make_data(m: int, device: torch.device):
    x = torch.zeros((m, K), dtype=torch.uint8, device=device).view(dtypes.fp8)
    weight = torch.zeros((N, K), dtype=torch.uint8, device=device).view(dtypes.fp8)
    weight = shuffle_weight(weight, layout=(16, 16))
    x_scale = torch.ones((m, K // SCALE_BLOCK), dtype=torch.float32, device=device)
    x_scale = x_scale.transpose(0, 1).contiguous().view_as(x_scale)
    w_scale = torch.ones(
        ((N + SCALE_BLOCK - 1) // SCALE_BLOCK, K // SCALE_BLOCK),
        dtype=torch.float32,
        device=device,
    )
    out = torch.empty((m, N), dtype=torch.bfloat16, device=device)
    # One child all-reduce over [M, hidden_dim], matching the TBO trace.
    comm = torch.ones((m * K,), dtype=torch.bfloat16, device=device)
    return x, weight, x_scale, w_scale, out, comm


def run_kernel(data, kernel_id: int):
    x, weight, x_scale, w_scale, out, _comm = data
    return aiter.gemm_a8w8_blockscale_bpreshuffle_tune(
        x, weight, x_scale, w_scale, out, kernel_id, 0
    )


def benchmark_standalone(data, kernel_id, warmup, iterations, repeats):
    for _ in range(warmup):
        run_kernel(data, kernel_id)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            run_kernel(data, kernel_id)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return samples


def benchmark_allreduce(comm, warmup, iterations, repeats):
    for _ in range(warmup):
        dist.all_reduce(comm, async_op=True).wait()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        dist.barrier()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for _ in range(iterations):
            dist.all_reduce(comm, async_op=True).wait()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - begin) * 1e6 / iterations)
    return samples


def benchmark_overlap(data, kernel_id, comm_stream, warmup, iterations, repeats):
    comm = data[-1]
    for _ in range(warmup):
        with torch.cuda.stream(comm_stream):
            work = dist.all_reduce(comm, async_op=True)
        run_kernel(data, kernel_id)
        work.wait()
    torch.cuda.synchronize()

    gemm_samples = []
    pair_samples = []
    for _ in range(repeats):
        dist.barrier()
        torch.cuda.synchronize()
        events = []
        begin = time.perf_counter()
        for _ in range(iterations):
            with torch.cuda.stream(comm_stream):
                work = dist.all_reduce(comm, async_op=True)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_kernel(data, kernel_id)
            end.record()
            work.wait()
            events.append((start, end))
        torch.cuda.synchronize()
        pair_samples.append((time.perf_counter() - begin) * 1e6 / iterations)
        gemm_samples.append(
            sum(start.elapsed_time(end) * 1000.0 for start, end in events)
            / iterations
        )
    return gemm_samples, pair_samples


def gather_rank_medians(*sample_sets):
    local = torch.tensor(
        [statistics.median(samples) for samples in sample_sets],
        dtype=torch.float64,
        device="cuda",
    )
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return [item.cpu().tolist() for item in gathered]


def summarize(values):
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=int, default=112)
    parser.add_argument("--tokens", type=parse_int_list, default=[8192, 16384, 32768])
    parser.add_argument("--kernel-ids", type=parse_int_list, default=[2, 15, 17])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    unknown = sorted(set(args.kernel_ids) - set(KERNEL_NAMES))
    if unknown:
        raise ValueError(f"unsupported kernel IDs: {unknown}")

    os.environ["NCCL_MIN_NCHANNELS"] = str(args.channel)
    os.environ["NCCL_MAX_NCHANNELS"] = str(args.channel)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    comm_stream = torch.cuda.Stream(device=device)

    results = []
    for m in args.tokens:
        data = make_data(m, device)
        dist.barrier()
        ar_samples = benchmark_allreduce(
            data[-1], args.warmup, args.iterations, args.repeats
        )

        for kernel_id in args.kernel_ids:
            standalone = benchmark_standalone(
                data, kernel_id, args.warmup, args.iterations, args.repeats
            )
            overlap_gemm, pair_wall = benchmark_overlap(
                data,
                kernel_id,
                comm_stream,
                args.warmup,
                args.iterations,
                args.repeats,
            )
            rank_values = gather_rank_medians(standalone, overlap_gemm, pair_wall, ar_samples)
            if rank == 0:
                standalone_by_rank = [item[0] for item in rank_values]
                overlap_by_rank = [item[1] for item in rank_values]
                pair_by_rank = [item[2] for item in rank_values]
                ar_by_rank = [item[3] for item in rank_values]
                results.append(
                    {
                        "M": m,
                        "N": N,
                        "K": K,
                        "kernel_id": kernel_id,
                        "kernel_name": KERNEL_NAMES[kernel_id],
                        "standalone_us_by_rank": standalone_by_rank,
                        "standalone_us": summarize(standalone_by_rank),
                        "overlap_gemm_us_by_rank": overlap_by_rank,
                        "overlap_gemm_us": summarize(overlap_by_rank),
                        "pair_wall_us_by_rank": pair_by_rank,
                        "pair_wall_us": summarize(pair_by_rank),
                        "allreduce_us_by_rank": ar_by_rank,
                        "allreduce_us": summarize(ar_by_rank),
                    }
                )

        del data
        torch.cuda.empty_cache()
        dist.barrier()

    if rank == 0:
        winners = {}
        for m in args.tokens:
            rows = [row for row in results if row["M"] == m]
            winners[str(m)] = {
                "standalone": min(rows, key=lambda row: row["standalone_us"]["max"])[
                    "kernel_id"
                ],
                "overlap_gemm": min(
                    rows, key=lambda row: row["overlap_gemm_us"]["max"]
                )["kernel_id"],
                "pair_wall": min(rows, key=lambda row: row["pair_wall_us"]["max"])[
                    "kernel_id"
                ],
            }
        payload = {
            "platform": torch.cuda.get_device_name(0),
            "world_size": dist.get_world_size(),
            "channel": args.channel,
            "nccl_min_nchannels": os.environ["NCCL_MIN_NCHANNELS"],
            "nccl_max_nchannels": os.environ["NCCL_MAX_NCHANNELS"],
            "tokens": args.tokens,
            "kernel_ids": args.kernel_ids,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "winner_summary": winners,
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(winners, indent=2))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
