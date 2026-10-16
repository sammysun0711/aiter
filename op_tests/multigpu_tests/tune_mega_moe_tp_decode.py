# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Measure existing Mega dynamic variants on MiMo TP8, preserving accuracy.

HSA_ENABLE_IPC_MODE_LEGACY=1 PYTHONPATH=. torchrun --standalone \
  --nproc_per_node=8 op_tests/multigpu_tests/tune_mega_moe_tp_decode.py \
  --output-dir results/mimo-a8w4-decode/mega-tuning

Explicit LaunchCfg overrides are experimental, confined to this tuner.
"""

import argparse
import dataclasses
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
from op_tests.multigpu_tests import bench_mega_moe_tp as bench
from op_tests.multigpu_tests import test_mega_moe_tp as support


def main():
    if get_gfx() != "gfx950":
        return 0
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, nargs="+", default=[8, 32, 64, 128, 248, 256])
    p.add_argument(
        "--routes",
        nargs="+",
        choices=["balanced", "random", "hot"],
        default=["random", "hot"],
    )
    p.add_argument("--mt", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--npp", type=int, nargs="+", default=[1, 2])
    p.add_argument("--dp", type=int, nargs="+", default=[0])
    p.add_argument("--grid", type=int, nargs="+", default=[256])
    p.add_argument("--llr", nargs="+", choices=["auto", "off", "on"], default=["auto"])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--schedule", choices=["auto", "dynamic", "lb"], default="auto")
    p.add_argument("--lp", type=int, nargs="+", default=[1])
    p.add_argument("--lb-q", type=int, nargs="+", default=[1])
    p.add_argument("--prefetch", nargs="+", choices=["off", "on"], default=["off"])
    args = p.parse_args()
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
    timing = SimpleNamespace(capture_warmup=20, graph_calls=10, rounds=3, iters=20)
    bench.STATE = SimpleNamespace(ctx=ctx, wt=wt, args=timing)
    layer = support.new_layer(
        shape,
        wt,
        ctx,
        "ag_rs",
        max(args.tokens) // world,
        **(
            {"lb_min": 0 if args.schedule == "dynamic" else 1}
            if args.schedule != "auto"
            else {}
        )
    )
    rows = []
    try:
        for tokens, route in itertools.product(args.tokens, args.routes):
            c = support.make_case(shape, wt, ctx, "ag_rs", tokens, route, 123)
            local_tokens = tokens // world
            base = layer.launch_config(local_tokens)
            for mt, npp, llr, dp, grid, lp, lbq, prefetch in itertools.product(
                args.mt,
                args.npp,
                args.llr,
                args.dp,
                args.grid,
                args.lp,
                args.lb_q,
                args.prefetch,
            ):
                layer.engine.n_cta = grid
                layer.engine._args = (None, None)
                cfg = dataclasses.replace(
                    base,
                    mt=mt,
                    npp=npp,
                    llr=(tokens <= 128 and llr == "auto") or llr == "on",
                    dp=dp,
                    lp=lp if base.lb else 0,
                    pf=base.lb and prefetch == "on",
                )
                if layer.engine.lb_q != lbq:
                    # lb_q is part of the compiled variant, not LaunchCfg.
                    layer.engine._launchers.clear()
                    layer.engine._armed.clear()
                    layer.engine.lb_q = lbq
                layer.engine._cfgs[local_tokens] = cfg
                layer.prepare([local_tokens])
                eager = support.call(layer, c).clone()
                err = support.check(eager, c, support.RTOL["fp8"])
                us, low, high, replay = bench.graph_time(
                    lambda layer=layer, c=c: layer(c.x, c.w, c.ids)
                )
                layer.check_errors()
                same = ctx.all_ok(torch.equal(eager, replay))
                if not same:
                    raise AssertionError("Mega graph replay differs from eager")
                row = {
                    "tokens": tokens,
                    "route": route,
                    "mt": mt,
                    "npp": npp,
                    "dp": dp,
                    "grid": grid,
                    "llr": llr,
                    "us": us,
                    "min_us": low,
                    "max_us": high,
                    "rel_l2": err,
                    "graph_exact": same,
                    "lb": cfg.lb,
                    "lp": cfg.lp,
                    "lb_q": lbq,
                    "prefetch": prefetch,
                }
                rows.append(row)
                if rank == 0:
                    args.output_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(rows).to_csv(
                        args.output_dir / "candidates.csv", index=False
                    )
                    print("MEGA_VARIANT", json.dumps(row), flush=True)
            layer.engine._cfgs[local_tokens] = base
        if rank == 0:
            aiter.logger.info(
                "Mega dynamic tuning:\n%s", pd.DataFrame(rows).to_markdown(index=False)
            )
            (args.output_dir / "metadata.json").write_text(
                json.dumps(bench.metadata(timing), indent=2) + "\n"
            )
    finally:
        bench.barrier()
        del layer
        bench.barrier()
        destroy_dist_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
