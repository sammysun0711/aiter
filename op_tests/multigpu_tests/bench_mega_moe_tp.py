# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Reproducible MiMo TP8 A8W4 split / Mega comparison on gfx950.

HSA_ENABLE_IPC_MODE_LEGACY=1 PYTHONPATH=. torchrun --standalone \
  --nproc_per_node=8 op_tests/multigpu_tests/bench_mega_moe_tp.py \
  --tokens 8 32 64 128 248 256 --output-dir results/mimo-a8w4-decode

Both paths use MXFP8/E8M0 group-32 activations and MXFP4 weights. The
split path quantizes locally before custom AllGather, sorts, runs the tuned
two-stage MoE, and custom ReduceScatters BF16 partials. Mega's output wire
is selected with --comm-dtypes. References are never timed. Each graph
contains ten layer calls; each of three timed rounds reports the slowest
rank, and the final latency is the median of those rounds.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import gc
import hashlib
import importlib.metadata
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter import dtypes
from aiter.dist.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from aiter.dist.parallel_state import graph_capture
from aiter.fused_moe import (
    MOEMetadata,
    _flydsl_stage1_wrapper,
    fused_moe_2stages,
    get_2stage_cfgs,
    get_padded_M,
    moe_sorting,
    stage2_uses_route_reduce,
)
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.communication import destroy_dist_env, init_dist_env
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.opus.moe_stage1_a8w4 import opus_a8w4_stage1_wrapper
from aiter.ops.opus.moe_stage2_a8w4 import (
    opus_a8w4_stage2_wrapper,
    stage2_launch_config,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.test_common import benchmark, checkAllclose, run_perftest
from csrc.opus_moe.opus_moe_common import opus_a8w4_stage2_instance_from_name
from op_tests.multigpu_tests import test_mega_moe_tp as support


def split_weights(wt, experts):
    """A8W4 tuned kernels consume the gate/up-interleaved MFMA layout.

    Mega consumes the standard shuffle. Both are prepared once from identical
    unshuffled MXFP4 values/scales; weight conversion is never timed.
    """
    return dataclasses.replace(
        wt,
        w1=shuffle_weight_a16w4(wt.w1_ref, 16, True),
        w1_scale=shuffle_scale_a16w4(wt.w1_scale_ref, experts, True),
        w2=shuffle_weight_a16w4(wt.w2_ref, 16, False),
        w2_scale=shuffle_scale_a16w4(wt.w2_scale_ref, experts, False),
    )


class Collectives:
    """Byte payloads are bitcast, never numerically converted, for custom AG."""

    def __init__(self, backend):
        self.backend = backend

    def gather(self, x):
        if self.backend == "custom":
            wire = x.view(torch.bfloat16) if x.element_size() == 1 else x
            y = tensor_model_parallel_all_gather(wire, use_custom=True, dim=0)
            return y.view(x.dtype)
        y = torch.empty(
            (dist.get_world_size() * x.shape[0], *x.shape[1:]),
            device=x.device,
            dtype=x.dtype,
        )
        dist.all_gather_into_tensor(y, x)
        return y

    def scatter(self, x):
        if self.backend == "custom":
            return tensor_model_parallel_reduce_scatter(x, use_custom=True, dim=0)
        y = torch.empty(
            (x.shape[0] // dist.get_world_size(), *x.shape[1:]),
            device=x.device,
            dtype=x.dtype,
        )
        dist.reduce_scatter_tensor(y, x)
        return y


class Split:
    def __init__(self, shape, wt, tokens, comm, kernels=None):
        self.shape, self.wt, self.comm = shape, wt, comm
        if kernels is None:
            self.md = get_2stage_cfgs(
                get_padded_M(tokens),
                shape.model_dim,
                wt.inter,
                shape.experts,
                shape.topk,
                dtypes.bf16,
                dtypes.fp8,
                dtypes.fp4x2,
                support.QUANT_TYPE,
                True,
                shape.act,
                False,
                0,
                0,
                True,
                GateMode.SEPARATED.value,
            )
        else:
            g1, g2, bm = kernels
            stage1 = (
                opus_a8w4_stage1_wrapper
                if g1.startswith("opus_")
                else _flydsl_stage1_wrapper
            )
            instance = opus_a8w4_stage2_instance_from_name(g2)
            self.md = MOEMetadata(
                stage1=functools.partial(
                    stage1, kernelName=g1, activation=shape.act, block_m=bm
                ),
                stage2=functools.partial(
                    opus_a8w4_stage2_wrapper,
                    kernelName=g2,
                    launch=stage2_launch_config(instance.kid),
                ),
                block_m=bm,
                ksplit=0,
                run_1stage=False,
                fuse_quant="fp8",
                prequant=True,
            )
        # The split wire is always MXFP8, including when a heuristic config
        # is selected. Qualification errors remain failures, never fast rows.
        if not self.md.prequant:
            raise ValueError("A8W4 split requires a prequantized-input GEMM1")
        self.quant = functools.partial(
            aiter.get_hip_quant(support.QUANT_TYPE),
            quant_dtype=dtypes.fp8,
            scale_type=dtypes.fp8_e8m0,
        )

    @property
    def kernels(self):
        names = []
        for stage in (self.md.stage1, self.md.stage2):
            kw = getattr(stage, "keywords", {})
            names.append(
                kw.get(
                    "kernelName",
                    kw.get("kernelName1", kw.get("kernelName2", "heuristic")),
                )
            )
        return names

    def routes(self, c):
        # Pack route weights and IDs into one 8-byte record, like the
        # recovered benchmark, so metadata needs one AllGather launch.
        meta = torch.stack((c.w.view(torch.int32), c.ids), dim=-1).reshape(
            c.x.shape[0], -1
        )
        all_meta = self.comm.gather(meta)
        packed = all_meta.reshape(-1, self.shape.topk, 2)
        w, ids = (
            packed[..., 0].contiguous().view(torch.float32),
            packed[..., 1].contiguous(),
        )
        return w, ids

    def sort(self, w, ids):
        md, shape = self.md, self.shape
        return moe_sorting(
            ids,
            w,
            shape.experts,
            shape.model_dim,
            dtypes.bf16,
            md.block_m,
            accumulate=not stage2_uses_route_reduce(md.stage2),
            **({"output_aux": md.output_aux} if md.output_aux else {"flat": md.flat}),
            output=None,
        )

    def compute(self, a, scales, w, ids, sorted_ret):
        md, wt, shape = self.md, self.wt, self.shape
        si, sw, se, nv, out, *aux = sorted_ret
        return fused_moe_2stages(
            a,
            wt.w1,
            wt.w2,
            shape.topk,
            si,
            sw,
            se,
            nv,
            out,
            True,
            md.block_m,
            activation=shape.act,
            quant_type=support.QUANT_TYPE,
            q_dtype_a=dtypes.fp8,
            q_dtype_w=dtypes.fp4x2,
            w1_scale=wt.w1_scale,
            w2_scale=wt.w2_scale,
            a1_scale=scales,
            topk_ids=ids,
            topk_weights=w,
            gate_mode=GateMode.SEPARATED.value,
            m_indices=aux[0] if aux else None,
            reverse_sorted=aux[1] if len(aux) > 1 else None,
            routing_num_experts=shape.experts,
            _metadata_transform=lambda _: md,
        )

    def __call__(self, c):
        w, ids = self.routes(c)
        sorted_ret = self.sort(w, ids)
        a, s = self.quant(c.x)
        a, s = self.comm.gather(a), self.comm.gather(s)
        return self.comm.scatter(self.compute(a, s, w, ids, sorted_ret).contiguous())


STATE = None


def rank_max(value):
    t = torch.tensor(float(value), device=STATE.ctx.device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def barrier():
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()


def graph_time(fn):
    args = STATE.args
    for _ in range(args.capture_warmup):
        fn()
    barrier()
    g = torch.cuda.CUDAGraph()
    with graph_capture(), torch.cuda.graph(g):
        for _ in range(args.graph_calls):
            output = fn()
    barrier()
    rounds = []
    for _ in range(args.rounds):
        barrier()
        _, us = run_perftest(
            g.replay,
            num_iters=args.iters,
            num_warmup=5,
            use_cuda_event=True,
        )
        rounds.append(rank_max(us / args.graph_calls))
    g.replay()
    torch.cuda.synchronize()
    actual = output.clone()
    del g, output
    gc.collect()
    barrier()
    return float(pd.Series(rounds).median()), min(rounds), max(rounds), actual


def check(output, ref, name, threshold):
    if not STATE.ctx.all_ok(bool(torch.isfinite(output).all())):
        raise AssertionError(f"{name}: nonfinite output")
    error = support.rel_l2(output, ref)
    if not error < threshold:
        raise AssertionError(f"{name}: relative L2 {error} >= {threshold}")
    err = checkAllclose(
        ref.float(),
        output.float(),
        rtol=0.1,
        atol=0.1 * float(ref.float().square().mean().sqrt()),
        msg=f"{name}: ",
        printLog=STATE.ctx.rank == 0,
    )
    return error, rank_max(err)


@benchmark()
def bench_decode(tokens, route, seed, comm_dtype, backend):
    ctx, shape, wt, args = STATE.ctx, support.MODELS["mimo"], STATE.wt, STATE.args
    c = support.make_case(shape, wt, ctx, "ag_rs", tokens, route, seed)
    comm = Collectives(backend)
    candidates = {}
    split = None
    if "split" in args.implementations:
        split = Split(shape, STATE.split_wt, tokens, comm)
        candidates["split"] = lambda: split(c)
        split_ref = support.make_case(
            shape,
            wt,
            ctx,
            "ag_rs",
            tokens,
            route,
            seed,
            intermediate_dtype=dtypes.fp32,
        ).ref
    max_tokens = max(args.tokens)
    max_local = max_tokens // ctx.world
    layer = support.new_layer(
        shape,
        wt,
        ctx,
        "ag_rs",
        max_local,
        comm_dtype=comm_dtype,
        **({"lb_min": args.lb_min} if args.lb_min is not None else {}),
    )
    layer.prepare([tokens // ctx.world])
    if "mega" in args.implementations:
        candidates["mega"] = lambda layer=layer: layer(c.x, c.w, c.ids)
    support.S = support.Session(ctx, max_local, max_tokens, {}, {})
    flops, nbytes = support._work(shape, wt, c, tokens)
    ret = {"gfx": get_gfx(), "act_dtype": "fp8", "wire": "fp8_e8m0_1x32"}
    if split is not None:
        ret["gemm1"], ret["gemm2"] = split.kernels
        ret["split block_m"] = split.md.block_m
    ret["mega config"] = str(layer.launch_config(tokens // ctx.world))
    for name, fn in candidates.items():
        eager = fn().clone()
        barrier()
        e, err = check(
            eager,
            c.ref if name == "mega" else split_ref,
            name,
            support.RTOL[comm_dtype] if name == "mega" else 0.01,
        )
        ret[f"{name} common_ref_rel_l2"] = support.rel_l2(eager, c.ref)
        if name == "mega":
            layer.check_errors()
        us, lo, hi, replay = graph_time(fn)
        replay_e, _ = check(replay, eager, f"{name} replay", 0.01)
        if name == "mega":
            layer.check_errors()
            if not ctx.all_ok(torch.equal(eager, replay)):
                raise AssertionError("Mega graph replay differs from eager")
        ret.update(
            {
                f"{name} us": us,
                f"{name} min_us": lo,
                f"{name} max_us": hi,
                f"{name} TFLOPS": flops / us / 1e6,
                f"{name} TB/s": nbytes / us / 1e6,
                f"{name} err": err,
                f"{name} rel_l2": e,
                f"{name} replay_rel_l2": replay_e,
            }
        )
    ret["status"] = "PASS"
    del layer
    barrier()
    return ret


def metadata(args):
    config = Path(AITER_CONFIGS.AITER_CONFIG_FMOE_FILE)
    root = Path(__file__).resolve().parents[2]
    sources = [
        Path(__file__).resolve(),
        Path(support.__file__).resolve(),
        *sorted((root / "aiter/ops/flydsl/kernels/mega_moe_tp").glob("*.py")),
    ]
    return {
        "command": sys.argv,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "flydsl": importlib.metadata.version("flydsl"),
        "gfx": get_gfx(),
        "cu": get_cu_num(),
        "tp": STATE.ctx.world,
        "shape": {
            "H": 6144,
            "I_global": 2048,
            "I_local": STATE.wt.inter,
            "E": 384,
            "topk": 8,
        },
        "fmoe_config": str(config),
        "fmoe_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "weight_layouts": {
            "mega": "standard shuffle(16,16) + e8m0_shuffle",
            "split": "shuffle_weight_a16w4 + shuffle_scale_a16w4",
        },
        "reference_intermediate": {
            "mega": "BF16 then MXFP8",
            "split": "FP32 then MXFP8",
            "common": "BF16 then MXFP8",
        },
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("AITER_", "HSA_", "FLYDSL_"))
        },
        "timing": {
            "statistic": "median of slowest-rank rounds",
            "rounds": args.rounds,
            "iters": args.iters,
            "graph_calls": args.graph_calls,
            "capture_warmup": args.capture_warmup,
        },
    }


def main():
    global STATE
    if get_gfx() != "gfx950":
        print("bench_mega_moe_tp: skipped (needs gfx950)")
        return 0
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", nargs="+", type=int, default=[8, 32, 64, 128, 248, 256])
    p.add_argument(
        "--routes",
        nargs="+",
        choices=["balanced", "random", "hot", "subset"],
        default=["random"],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[123])
    p.add_argument("--comm-dtypes", nargs="+", choices=["fp8", "bf16"], default=["fp8"])
    p.add_argument(
        "--backends", nargs="+", choices=["custom", "rccl"], default=["custom"]
    )
    p.add_argument(
        "--implementations",
        nargs="+",
        choices=["split", "mega"],
        default=["split", "mega"],
    )
    p.add_argument("--capture-warmup", type=int, default=400)
    p.add_argument("--graph-calls", type=int, default=10)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument(
        "--lb-min", type=int, help="Override MiMo LB crossover (0 forces dynamic)"
    )
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    rank, world, local = (
        int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    )
    if world != 8 or any(t <= 0 or t > 2048 or t % world for t in args.tokens):
        p.error("requires TP8 and positive global token multiples of 8, at most 2048")
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
    wt = support.build_weights(support.MODELS["mimo"], ctx, 123)
    STATE = SimpleNamespace(ctx=ctx, wt=wt, split_wt=split_weights(wt, 384), args=args)
    rows = []
    try:
        for values in itertools.product(
            args.tokens, args.routes, args.seeds, args.comm_dtypes, args.backends
        ):
            rows.append(bench_decode(*values))
            if rank == 0:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(args.output_dir / "latency.csv", index=False)
        if rank == 0:
            aiter.logger.info(
                "MiMo A8W4 TP decode summary:\n%s",
                pd.DataFrame(rows).to_markdown(index=False),
            )
            (args.output_dir / "metadata.json").write_text(
                json.dumps(metadata(args), indent=2) + "\n"
            )
            (args.output_dir / "effective_fmoe.csv").write_bytes(
                Path(AITER_CONFIGS.AITER_CONFIG_FMOE_FILE).read_bytes()
            )
    finally:
        barrier()
        destroy_dist_env()
    return 0


if __name__ == "__main__":
    sys.exit(main())
