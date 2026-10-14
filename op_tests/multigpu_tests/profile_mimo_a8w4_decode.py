# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Profile MiMo A8W4 dynamic routing and communication without changing wave roles.

HSA_ENABLE_IPC_MODE_LEGACY=1 AITER_MEGAMOE_TP_PROFILE=1 PYTHONPATH=. \
  torchrun --standalone --nproc_per_node=8 \
  op_tests/multigpu_tests/profile_mimo_a8w4_decode.py \
  --output-dir results/mimo-a8w4-decode/profile

Mega timestamps use s_memrealtime (100 MHz): 100 ticks/us. Each CTA records
its own intervals; concurrent compute/service intervals cannot be summed.
Split components are isolated CUDA-graph measurements with fixed inputs,
so their sum is an estimate, not an end-to-end latency.
"""

import argparse
import functools
import importlib
import itertools
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.communication import destroy_dist_env, init_dist_env
from aiter.ops.flydsl.kernels.mega_moe_tp.mega_moe_tp_config import (
    PROFILE_BASE,
    PROFILE_SLOTS,
)
from op_tests.multigpu_tests import bench_mega_moe_tp as bench
from op_tests.multigpu_tests import test_mega_moe_tp as support

INTERVALS = {
    "init + quant + barrier": (0, 1),
    "compute metadata wait": (2, 3),
    "dynamic plan": (4, 5),
    "compute units incl route scan": (5, 6),
    "compute final comm": (6, 7),
    "AG send": (1, 8),
    "AG wave output/poll": (8, 9),
    "A loader": (1, 10),
    "loader output/signal": (10, 11),
    "service output": (1, 12),
    "roles critical duration": (1, 13),
    "final completion": (13, 14),
    "CTA total": (0, 14),
}


def first_output(fn):
    value = fn()
    return value[0] if isinstance(value, tuple) else value


def main():
    if get_gfx() != "gfx950":
        return 0
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, nargs="+", default=[8, 32, 64, 128, 248, 256])
    p.add_argument(
        "--routes",
        nargs="+",
        choices=["random", "balanced", "hot"],
        default=["random", "hot"],
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--lb-min", type=int, help="Override MiMo LB crossover (0 forces dynamic)"
    )
    p.add_argument(
        "--mega-only", action="store_true", help="Skip unqualified split configurations"
    )
    args = p.parse_args()
    os.environ["AITER_MEGAMOE_TP_PROFILE"] = "1"
    rank, world, local = (
        int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    )
    if world != 8 or any(t <= 0 or t > 2048 or t % world for t in args.tokens):
        p.error("requires TP8 and 8-2048 global tokens divisible by 8")
    torch.cuda.set_device(local)
    init_dist_env(world, rank, local_rank=local)
    ctx = SimpleNamespace(rank=rank, world=world, device=torch.device("cuda", local))

    def all_ok(ok):
        t = torch.tensor(int(ok), device=ctx.device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return bool(t.item())

    ctx.all_ok = all_ok
    if rank:
        aiter.logger.setLevel("WARNING")
    shape = support.MODELS["mimo"]
    wt = support.build_weights(shape, ctx, 123)
    split_wt = bench.split_weights(wt, 384)
    timing_args = SimpleNamespace(capture_warmup=20, graph_calls=10, rounds=3, iters=20)
    bench.STATE = SimpleNamespace(ctx=ctx, wt=wt, split_wt=split_wt, args=timing_args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries, split_rows = [], []
    layer = support.new_layer(
        shape,
        wt,
        ctx,
        "ag_rs",
        max(args.tokens) // world,
        **({"lb_min": args.lb_min} if args.lb_min is not None else {}),
    )
    try:
        layer.prepare([t // world for t in args.tokens])
        for tokens, route in itertools.product(args.tokens, args.routes):
            cfg = layer.launch_config(tokens // world)
            c = support.make_case(shape, wt, ctx, "ag_rs", tokens, route, 123)
            for _ in range(20):
                layer(c.x, c.w, c.ids)
            y = support.call(layer, c)
            support.check(y, c, support.RTOL["fp8"])
            start = PROFILE_BASE // 4
            stamps = (
                layer.engine.ctrl[start:]
                .view(torch.int64)
                .reshape(-1, PROFILE_SLOTS)
                .cpu()
            )
            stamp_frame = pd.DataFrame(
                stamps.numpy(), columns=[f"slot{i}" for i in range(PROFILE_SLOTS)]
            )
            stamp_frame.to_csv(
                args.output_dir / f"mega-M{tokens}-{route}-rank{rank}.csv",
                index_label="cta",
            )
            gathered = [None] * world
            dist.all_gather_object(gathered, stamps.numpy().tolist())
            if rank == 0:
                all_stamps = torch.tensor(gathered, dtype=torch.int64).reshape(
                    -1, PROFILE_SLOTS
                )
                for name, (a, b) in INTERVALS.items():
                    if cfg.lb and name == "dynamic plan":
                        name = "LB plan + route scatter"
                    if cfg.lb and name == "compute units incl route scan":
                        name = "LB G1 + export + G2 phase"
                    valid = (all_stamps[:, a] > 0) & (
                        all_stamps[:, b] >= all_stamps[:, a]
                    )
                    values = (
                        all_stamps[valid, b] - all_stamps[valid, a]
                    ).double() / 100
                    if values.numel():
                        summaries.append(
                            {
                                "tokens": tokens,
                                "route": route,
                                "interval": name,
                                "mean_us": float(values.mean()),
                                "median_us": float(values.median()),
                                "p90_us": float(values.quantile(0.9)),
                                "max_us": float(values.max()),
                                "schedule": "lb" if cfg.lb else "dynamic",
                            }
                        )
                for name, slot in (
                    ("route scan sum", 16),
                    ("GEMM1 sum", 17),
                    ("GEMM2 sum", 18),
                ):
                    if cfg.lb and slot == 17:
                        name = "LB GEMM1 + intermediate export sum"
                    elif cfg.lb and slot == 18:
                        name = "LB GEMM2 phase incl import/claim/waits"
                    values = all_stamps[:, slot].double() / 100
                    summaries.append(
                        {
                            "tokens": tokens,
                            "route": route,
                            "interval": name,
                            "mean_us": float(values.mean()),
                            "median_us": float(values.median()),
                            "p90_us": float(values.quantile(0.9)),
                            "max_us": float(values.max()),
                            "schedule": "lb" if cfg.lb else "dynamic",
                        }
                    )
                pd.DataFrame(summaries).to_csv(
                    args.output_dir / "mega-stages.csv", index=False
                )
            # Split isolated components preserve the same routing/quantization
            # and the selected installed config, with each dependency ready.
            if args.mega_only:
                continue
            split = bench.Split(shape, split_wt, tokens, bench.Collectives("custom"))
            w, ids = split.routes(c)
            sr = split.sort(w, ids)
            a, s = split.quant(c.x)
            ag, sg = split.comm.gather(a), split.comm.gather(s)
            fmoe = importlib.import_module("aiter.fused_moe")
            fmoe.kernel_bench_callable = []
            try:
                partial = split.compute(ag, sg, w, ids, sr)
                kernels = list(fmoe.kernel_bench_callable)
            finally:
                fmoe.kernel_bench_callable = None

            def quant(split=split, c=c):
                return split.quant(c.x)[0]

            def route_ag(split=split, c=c):
                return split.routes(c)[1]

            def sort(split=split, w=w, ids=ids):
                return split.sort(w, ids)[0]

            functions = {
                "local quant": quant,
                "route AG": route_ag,
                "sort": sort,
                "activation AG": functools.partial(split.comm.gather, a),
                "scale AG": functools.partial(split.comm.gather, s),
                "compute incl scales": functools.partial(
                    split.compute, ag, sg, w, ids, sr
                ),
                "RS": functools.partial(split.comm.scatter, partial),
            }
            for name, fn in kernels:
                # Stage1 returns (payload, scales); graph_time clones one result.
                functions[name] = functools.partial(first_output, fn)
            row = {
                "tokens": tokens,
                "route": route,
                "gemm1": split.kernels[0],
                "gemm2": split.kernels[1],
            }
            for name, fn in functions.items():
                row[name + " us"] = bench.graph_time(fn)[0]
            split_rows.append(row)
            if rank == 0:
                pd.DataFrame(split_rows).to_csv(
                    args.output_dir / "split-stages.csv", index=False
                )
        if rank == 0:
            aiter.logger.info(
                "Mega diagnostic intervals:\n%s",
                pd.DataFrame(summaries).to_markdown(index=False),
            )
            aiter.logger.info(
                "Split isolated stages:\n%s",
                pd.DataFrame(split_rows).to_markdown(index=False),
            )
            (args.output_dir / "metadata.json").write_text(
                json.dumps(bench.metadata(timing_args), indent=2) + "\n"
            )
    finally:
        bench.barrier()
        del layer
        bench.barrier()
        destroy_dist_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
