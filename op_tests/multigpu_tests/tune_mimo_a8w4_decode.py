# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qualify existing small-M split kernels on MiMo's TP8 local geometry.

PYTHONPATH=. python op_tests/multigpu_tests/tune_mimo_a8w4_decode.py \
  --output-dir results/mimo-a8w4-decode/tuning

Single-GPU tuning; global M is fed to one rank's local-I compute. Measures
sort + quantized two-stage compute, never the reference or TP communication.
Winning rows must still pass the distributed benchmark before installation.
"""

import argparse
import importlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest
from csrc.opus_moe.opus_moe_common import (
    OPUS_A8W4_DECODE_STAGE2_INSTANCES,
    opus_a8w4_stage1_instances_for_shape,
)
from op_tests.multigpu_tests import bench_mega_moe_tp as bench
from op_tests.multigpu_tests import test_mega_moe_tp as support

STATE = None


def timing(fn):
    # Captured function is local, so no cross-rank synchronization is needed.
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            fn()
    torch.cuda.synchronize()
    _, us = run_perftest(graph.replay, num_iters=20, num_warmup=3, use_cuda_event=True)
    return us / 10


def candidates():
    for bm, tn, wpe in itertools.product((16, 32, 64), (128, 256), (1, 2, 3)):
        yield (
            f"flydsl_moe1_afp8_wfp4_bf16_t{bm}x{tn}x256"
            + (f"_w{wpe}" if wpe > 1 else "")
            + "_gui_fp8",
            bm,
        )
    for inst in opus_a8w4_stage1_instances_for_shape(
        model_dim=6144, logical_inter_dim=256, inter_dim_pad=0
    ):
        if inst.block_m <= 64:
            yield inst.profile_name, inst.block_m


@benchmark()
def tune_case(tokens, seed):
    wt, shape, device = STATE.wt, support.MODELS["mimo"], STATE.ctx.device
    gen = torch.Generator(device=device).manual_seed(seed + tokens)
    x = torch.randn(
        (tokens, shape.model_dim), dtype=dtypes.bf16, device=device, generator=gen
    )
    w, ids = support.route(tokens, shape, "random", gen, device)
    # Split GEMM1 quantizes FP32 activation directly; Mega first rounds it
    # to BF16. Use split's own reference for strict qualification.
    ref = support.torch_partial(shape, wt, x, w, ids, intermediate_dtype=dtypes.fp32)
    quant = aiter.get_hip_quant(support.QUANT_TYPE)
    a, scales = quant(x, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0)
    profiles = []
    for g1, bm in candidates():
        # Atomic/BF16 output keeps the split comparator's FP8 activation error
        # separate from Mega's lossy output communication.
        g2s = [
            inst
            for inst in OPUS_A8W4_DECODE_STAGE2_INSTANCES
            if inst.sort_block_m == bm
            and not inst.route_out_fp8
            and inst.supports_tuner_token(tokens)
        ]
        for inst in g2s:
            row = {
                "tokens": tokens,
                "seed": seed,
                "gemm1": g1,
                "gemm2": inst.name,
                "block_m": bm,
            }
            try:
                split = bench.Split(
                    shape, STATE.split_wt, tokens, None, (g1, inst.name, bm)
                )
                fn = lambda split=split: split.compute(
                    a, scales, w, ids, split.sort(w, ids)
                )
                y = fn()
                torch.cuda.synchronize()
                e = float((y.float() - ref).norm() / ref.norm())
                if not bool(torch.isfinite(y).all()) or not e < 0.01:
                    raise AssertionError(f"relative L2 {e} >= 0.01 or nonfinite")
                err = checkAllclose(
                    ref.float(),
                    y.float(),
                    rtol=0.1,
                    atol=0.1 * float(ref.square().mean().sqrt()),
                    printLog=False,
                )
                us = timing(fn)
                fmoe = importlib.import_module("aiter.fused_moe")
                fmoe.kernel_bench_callable = []
                try:
                    fn()
                    stages = list(fmoe.kernel_bench_callable)
                finally:
                    fmoe.kernel_bench_callable = None
                row.update({"us": us, "rel_l2": e, "err": err, "status": "PASS"})
                for name, stage in stages:
                    row[name + " us"] = timing(stage)
            except (ValueError, RuntimeError, AssertionError) as exc:
                row.update({"status": "FAIL", "reason": str(exc)[:300]})
            profiles.append(row)
            STATE.profiles.append(row)
            pd.DataFrame(STATE.profiles).to_csv(
                STATE.out / "candidates.csv", index=False
            )
    good = [r for r in profiles if r["status"] == "PASS"]
    if not good:
        raise RuntimeError(f"No qualified candidate for M={tokens}")
    winner = min(good, key=lambda r: r["us"])
    STATE.winners.append(winner)
    (STATE.out / "winners.json").write_text(json.dumps(STATE.winners, indent=2) + "\n")
    flops = 6 * tokens * shape.topk * shape.model_dim * wt.inter
    return {
        "gfx": get_gfx(),
        **winner,
        "TFLOPS": flops / winner["us"] / 1e6,
        "TB/s": (wt.w1.numel() + wt.w2.numel()) / winner["us"] / 1e6,
        "qualified": len(good),
        "attempted": len(profiles),
    }


def main():
    global STATE
    if get_gfx() != "gfx950":
        return 0
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", nargs="+", type=int, default=[8, 32, 64, 128, 248, 256])
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    torch.cuda.set_device(0)
    ctx = SimpleNamespace(rank=0, world=8, device=torch.device("cuda", 0))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    STATE = SimpleNamespace(
        ctx=ctx,
        wt=support.build_weights(support.MODELS["mimo"], ctx, 123),
        out=args.output_dir,
        profiles=[],
        winners=[],
    )
    STATE.split_wt = bench.split_weights(STATE.wt, 384)
    rows = [tune_case(m, 123) for m in args.tokens]
    aiter.logger.info(
        "MiMo A8W4 local compute tuning:\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
