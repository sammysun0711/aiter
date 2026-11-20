# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compare PTPC and FP32 K128 block-scaled GEMM implementations on one MI300X."""

import argparse
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import torch
from common import (
    BF16,
    CONFIGS,
    PROJECTIONS,
    SIZES,
    Inputs,
    activate,
    error_metric,
    opus_offsets_fit,
    packed_scales,
    public_block,
    read_csv,
    save,
)

import aiter
from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_blockscale_common import (
    kernel_fits_shape,
    kernels_list,
)
from aiter.ops.gemm_op_a8w8 import (
    gemm_a8w8_blockscale_ck,
    gemm_a8w8_blockscale_flydsl,
    get_GEMM_config_with_quant_type,
)
from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale
from aiter.tuned_gemm import get_GEMM_A16W16_config, tgemm

FINAL_FLYDSL = {
    "block_lds_m": "lds_m",
    "block_lds_n": "lds_n",
    "block_register": "reg_a",
}


@dataclass
class Method:
    name: str
    scaling: str
    weight: str
    rows: int
    backend: str
    kernel: str = ""
    kernel_id: str = ""


def find_row(table, m, n, k):
    for row in read_csv(table):
        if tuple(int(row[t]) for t in ("M", "N", "K")) == (m, n, k):
            return row
    raise ValueError(f"No row for M/N/K={(m,n,k)} in {table}")


def sliced(d, step, row_scale, launch):
    if d.m % step:
        raise ValueError("Row plan leaves an incomplete slice")
    slices = [
        (d.x[i : i + step], row_scale[i : i + step], d.out[i : i + step])
        for i in range(0, d.m, step)
    ]

    def run():
        for x, scale, out in slices:
            launch(x, scale, out)

    return run


def ptpc_methods(d, sx, contract, configs, native_rows, selected):
    m, n, k = d.m, d.n, d.k

    def torch_call():
        return torch.ops.aten._scaled_mm.out(
            d.x, d.w.T, sx, d.sw, out_dtype=BF16, out=d.out
        )

    if not selected or contract + "_torch" in selected:
        yield Method(
            contract + "_torch", contract, "ordinary", m, "PyTorch _scaled_mm"
        ), torch_call
    for suffix, packed in (("plain", False), ("packed", True)):
        name = contract + "_tgemm_" + suffix
        if selected and name not in selected:
            continue
        config = get_GEMM_A16W16_config(
            m, n, k, False, str(d.x.dtype), str(BF16), True, packed
        )
        (expected,) = [
            r
            for r in read_csv(configs / "ptpc.csv")
            if (int(r["M"]), int(r["N"]), int(r["K"]), r["bpreshuffle"])
            == (m, n, k, str(packed))
        ]
        if config["libtype"] != "hipblaslt" or int(config["solidx"]) != int(
            expected["solidx"]
        ):
            raise RuntimeError(
                f"Set AITER_CONFIG_GEMM_BF16 to {configs / 'ptpc.csv'} before import"
            )
        weight = d.packed if packed else d.w

        def run(weight=weight):
            return tgemm.mm(
                d.x, weight, scale_a=sx, scale_b=d.sw, otype=BF16, out=d.out
            )

        yield Method(
            name,
            contract,
            "packed" if packed else "ordinary",
            m,
            "hipBLASLt",
            expected["kernelName"],
            expected["solidx"],
        ), run

    if selected and contract + "_aiter_native" not in selected:
        return
    # Keep this CK-backed operator's individual dense descriptors below 2 GiB.
    step = native_rows
    cfg = get_GEMM_config_with_quant_type(step, n, k, d.x.dtype) or {}
    native = cfg.get("libtype", "ck")
    if native not in ("ck", "cktile", "flydsl"):
        native = "ck"
    ws = d.sw.T.contiguous()

    def native_ptpc(x, scale, out):
        aiter.gemm_a8w8_bpreshuffle_out(out, x, d.packed, scale, ws)

    yield Method(
        contract + "_aiter_native",
        contract,
        "packed",
        step,
        native,
        str(cfg.get("kernelName", "")),
        str(cfg.get("kernelId", "")),
    ), sliced(d, step, sx, native_ptpc)


def methods(d, plan, configs, selected=None, all_flydsl=False):
    m, n, k = d.m, d.n, d.k
    if selected and "block_best" in selected:
        winner = find_row(configs / "best-plans.csv", m, n, k)
        activate(configs / "best-direct.csv")
        step = int(winner["chunk_m"])
        fn, name = public_block(d, winner["libtype"], step)
        if name != winner["kernelName"] or step * int(winner["calls"]) != m:
            raise ValueError(f"Deployment table and chunk plan disagree for {(m,n,k)}")
        yield Method(
            "block_best",
            "block128",
            "packed",
            step,
            winner["libtype"],
            name,
            winner["kernelId"],
        ), fn
        if set(selected) == {"block_best"}:
            return
    native_rows = int(plan["ptpc_aiter_native"])
    yield from ptpc_methods(d, d.sx, "ptpc", configs, native_rows, selected)
    yield from ptpc_methods(
        d,
        torch.full_like(d.sx, 0.75),
        "tensor_channel",
        configs,
        native_rows,
        selected,
    )

    if not selected or "block_ck_plain" in selected:
        step = int(plan["block_ck_plain"])
        plain = find_row(configs / "block-plain.csv", step, n, k)

        def ordinary_ck(x, scale, out):
            return gemm_a8w8_blockscale_ck(
                x, d.w, scale, d.bw, out, int(plain["splitK"]), plain["kernelName"]
            )

        yield Method(
            "block_ck_plain",
            "block128",
            "ordinary",
            step,
            "ck",
            plain["kernelName"],
            str(plain.get("kernelId", "")),
        ), sliced(d, step, d.bx, ordinary_ck)

    if not selected or "block_triton" in selected:
        # This exact full-shape Triton launch previously faulted on gfx942. Reuse
        # the validated M=8192 descriptor for this shape and include every launch.
        step = int(plan["block_triton"])

        def triton(x, scale, out):
            return gemm_a8w8_blockscale(x, d.w, scale, d.bw, y=out, backend="triton")

        yield Method("block_triton", "block128", "ordinary", step, "triton"), sliced(
            d, step, d.bx, triton
        )

    for family in ("ck", "cktile", "asm", "opus", "flydsl"):
        if all_flydsl and family == "flydsl":
            continue  # The catalog below includes this tuned row.
        length = plan["block_" + family]
        if not length or (selected and "block_" + family not in selected):
            continue
        table = configs / (family + ".csv")
        step = int(length)
        # The generator is consumed one method at a time: save its public
        # dispatch into a graph before changing the process-wide table.
        activate(table)
        fn, kernel = public_block(d, family, step)
        row = find_row(table, step, n, k)
        assert row["kernelName"] == kernel
        yield Method(
            "block_" + family,
            "block128",
            "packed",
            step,
            family,
            kernel,
            row["kernelId"],
        ), fn
    if all_flydsl:
        sx = packed_scales(d.bx)
        for kid, entry in kernels_list.items():
            name = f"block_flydsl_id{kid}"
            if not kernel_fits_shape(entry, m, n, k, "gfx942") or (
                selected and name not in selected
            ):
                continue
            packed = entry.preshuffle_b
            weight, scales = (d.packed, sx) if packed else (d.w, d.bx)

            def run(weight=weight, scales=scales, kernel=entry.name, packed=packed):
                return gemm_a8w8_blockscale_flydsl(
                    d.x, weight, scales, d.bw, d.out, {"kernelName": kernel}, packed
                )

            yield Method(
                name,
                "block128",
                "packed" if packed else "ordinary",
                m,
                "flydsl",
                entry.name,
                str(kid),
            ), run
        return
    if selected and not FINAL_FLYDSL.keys() & set(selected):
        return
    sx = packed_scales(d.bx)
    for name, variant in FINAL_FLYDSL.items():
        if selected and name not in selected:
            continue
        kernel = f"flydsl_blockscale_gfx942_128x128x128_ps1_{variant}"

        def run(kernel=kernel):
            return gemm_a8w8_blockscale_flydsl(
                d.x, d.packed, sx, d.bw, d.out, {"kernelName": kernel}, True
            )

        yield Method(name, "block128", "packed", m, "FlyDSL", kernel), run


def capture(fn, stream, ops):
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(ops):
            fn()
    return graph


def time_group(graph, ops):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    graph.replay()
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / ops


def measure(shape, args):
    m, n, k = shape
    plan = find_row(args.config_dir / "replay-plans.csv", m, n, k)
    limits = {int(v) for key, v in plan.items() if key not in ("M", "N", "K") and v}
    limits.add(8192)
    d = Inputs(shape)
    ri, ci, gold_ptpc, gold_block = d.reference(
        sorted(step for step in limits if step < m)
    )
    references = {
        "ptpc": gold_ptpc,
        "block128": gold_block,
        "tensor_channel": gold_ptpc / d.sx[ri].cpu().double() * 0.75,
    }
    stream = torch.cuda.Stream()
    graphs, keepalive, records = {}, {}, {}
    for method, fn in methods(d, plan, args.config_dir, args.methods, args.all_flydsl):
        name = method.name
        info = dict(asdict(method), calls=m // method.rows, rounds_us=[])
        records[name] = info
        ref = references[method.scaling]
        try:
            checks = [
                error_metric(fn, d.out, ri, ci, ref)
                for _ in range(5 if name == "block_opus" else 1)
            ]
            graph = capture(fn, stream, args.ops)
            checks.append(error_metric(graph.replay, d.out, ri, ci, ref))
        except ValueError as err:
            if "sampled numerical check failed" not in str(err):
                raise
            info.update(status="failed accuracy", detail=str(err))
            print(shape, name, info["detail"], flush=True)
            continue
        graphs[name] = graph
        keepalive[name] = (
            fn  # includes graph inputs allocated outside capture (e.g., packed chunk scales)
        )
        info.update(
            status="passed",
            relative_rms=max(c[0] for c in checks),
            max_over_rms=max(c[1] for c in checks),
        )
    if not plan["block_opus"] and (not args.methods or "block_opus" in args.methods):
        reason = (
            "No qualified signed-input plan"
            if opus_offsets_fit(*shape)
            else "Full-shape signed row offset exceeds INT_MAX; no qualified chunk plan"
        )
        records["block_opus"] = {"status": "unavailable", "detail": reason}
    if args.methods and set(args.methods) != set(records):
        raise ValueError(f"Unknown methods: {sorted(set(args.methods) - set(records))}")
    for trial in range(args.rounds):
        order = list(graphs)
        random.Random(2026 + trial // 2).shuffle(order)
        if trial % 2:
            order.reverse()
        for name in order:
            records[name]["rounds_us"].append(
                [time_group(graphs[name], args.ops) for _ in range(args.groups)]
            )
    for name, graph in graphs.items():
        info = records[name]
        ref = references[info["scaling"]]
        try:
            err = error_metric(graph.replay, d.out, ri, ci, ref)
        except ValueError as exc:
            info.update(status="failed accuracy", detail=str(exc))
            continue
        us = median(v for group in info["rounds_us"] for v in group)
        info.update(
            median_us=us,
            tflops=2 * math.prod(shape) / us / 1e6,
            relative_rms=max(err[0], info["relative_rms"]),
            max_over_rms=max(err[1], info["max_over_rms"]),
        )
    return {
        "shape": shape,
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "samples_checked": gold_block.numel(),
        "graph_ops": args.ops,
        "rounds": args.rounds,
        "groups": args.groups,
        "methods": records,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", nargs=3, type=int, metavar=("M", "N", "K"))
    p.add_argument(
        "--methods", nargs="+", help="Replay only the specified method columns"
    )
    p.add_argument(
        "--all-flydsl",
        action="store_true",
        help="Replay every eligible ordinary and packed FlyDSL ID in place of the recorded FlyDSL shortlist",
    )
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--groups", type=int, default=2)
    p.add_argument("--ops", type=int, default=3)
    p.add_argument("--config-dir", type=Path, default=CONFIGS)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if min(args.rounds, args.groups, args.ops) < 1:
        p.error("rounds, groups and ops must be positive")
    shapes = (
        [tuple(args.shape)]
        if args.shape
        else [(m, n, k) for n, k in PROJECTIONS for m in SIZES]
    )
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    assert props.multi_processor_count == 304 and props.gcnArchName.startswith("gfx942")
    with torch.inference_mode():
        for shape in shapes:
            print("replaying", shape, flush=True)
            data = measure(shape, args)
            save(args.output / ("-".join(map(str, shape)) + ".json"), data)
            print(
                "passed",
                sum(i["status"] == "passed" for i in data["methods"].values()),
                shape,
                flush=True,
            )
            torch.cuda.empty_cache()
