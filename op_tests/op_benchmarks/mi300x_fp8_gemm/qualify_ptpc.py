# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compare gradlib PTPC finalists with existing MI300X hipBLASLt rows."""

import argparse
import csv
from functools import partial
from pathlib import Path
from statistics import median

import torch
from bench_fp8_scaling_comparison import capture, time_group
from common import BF16, CONFIGS, FP8, Inputs, error_metric, read_csv, save

import aiter

KEY = (
    "gfx",
    "cu_num",
    "M",
    "N",
    "K",
    "bias",
    "dtype",
    "outdtype",
    "scaleAB",
    "bpreshuffle",
)


def row_key(row):
    return tuple(row[k] for k in KEY)


def select(stock, profile, incumbent):
    m, n, k = (int(stock[v]) for v in ("M", "N", "K"))
    packed = stock["bpreshuffle"] == "True"
    d = Inputs((m, n, k))
    weight = (d.packed if packed else d.w).T
    ri, ci, ref, _ = d.reference()
    supported = set(
        aiter.hipb_findallsols(
            d.x, weight, out_dtype=BF16, scaleA=d.sx, scaleB=d.sw, bpreshuffle=packed
        )
    )
    eligible = sorted(
        (
            r
            for r in profile
            if row_key(r) == row_key(stock)
            and r["libtype"] == "hipblaslt"
            and 0 < float(r["us"]) < 1e9
            and float(r["err_ratio"]) <= 0.05
        ),
        key=lambda r: float(r["us"]),
    )
    old = (
        int(incumbent["solidx"])
        if incumbent and incumbent["libtype"] == "hipblaslt"
        else None
    )
    ids = list(
        dict.fromkeys(
            [old, int(stock["solidx"]), *[int(r["solidx"]) for r in eligible[:3]]]
        )
    )
    stream = torch.cuda.Stream()
    timing, errors, rejected, graphs = {}, {}, {}, {}
    for solution in ids:
        if solution is None:
            continue
        if solution not in supported:
            rejected[solution] = "unavailable in this hipBLASLt installation"
            continue
        call = partial(
            aiter.hipb_mm,
            d.x,
            weight,
            solution,
            scaleA=d.sx,
            scaleB=d.sw,
            out=d.out,
            bpreshuffle=packed,
        )
        graph = capture(call, stream, 4)
        try:
            errors[solution] = error_metric(graph.replay, d.out, ri, ci, ref)
        except ValueError as exc:
            rejected[solution] = str(exc)
            continue
        graphs[solution] = graph
        timing[solution] = []
    ids = list(graphs)
    if not ids:
        raise ValueError(
            f"No validated hipBLASLt solution for {row_key(stock)}: {rejected}"
        )
    for i in range(3):
        for sol in ids[i:] + ids[:i]:
            timing[sol].extend(time_group(graphs[sol], 4) for _ in range(3))
    chosen = min(ids, key=lambda sol: median(timing[sol]))
    paired = {}
    us = median(timing[chosen])
    if old in graphs and old != chosen:
        challenger = chosen
        paired = {old: [], challenger: []}
        for i in range(4):
            for sol in ((old, challenger) if i % 2 == 0 else (challenger, old)):
                paired[sol].append([time_group(graphs[sol], 4) for _ in range(3)])

        def elapsed(sol, batches):
            return median(v for group in paired[sol][batches] for v in group)

        if any(
            elapsed(challenger, slice(i, None, 2)) >= elapsed(old, slice(i, None, 2))
            for i in (0, 1)
        ):
            chosen = old
        us = elapsed(chosen, slice(None))
    row = {
        **stock,
        "solidx": chosen,
        "us": us,
        "kernelName": aiter.getHipblasltKernelName(chosen),
        "tflops": 2 * m * n * k / us / 1e6,
        "bw": (m * k + n * k + 2 * m * n) / us / 1000,
    }
    return row, {
        "chosen": chosen,
        "sampled_outputs": ref.numel(),
        "numerical_errors": errors,
        "rejected": rejected,
        "selection_us": timing,
        "paired_us": paired,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, default=CONFIGS / "ptpc.csv")
    p.add_argument(
        "--stock", type=Path, required=True, help="Winner CSV from gemm_a16w16_tune.py"
    )
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument(
        "--output", type=Path, required=True, help="Merged AITER runtime table"
    )
    args = p.parse_args()
    props = torch.cuda.get_device_properties(0)
    if not props.gcnArchName.startswith("gfx942") or props.multi_processor_count != 304:
        p.error("expected a 304-CU MI300X")
    torch.cuda.set_device(0)
    aiter.hipb_create_extension()
    with args.baseline.open(newline="") as baseline:
        reader = csv.DictReader(baseline)
        fields = reader.fieldnames
        initial = list(reader)
    rows = {row_key(r): r for r in initial}
    profile = read_csv(args.profile)
    diagnostics = []
    for stock in read_csv(args.stock):
        if (
            stock["gfx"],
            stock["cu_num"],
            stock["dtype"],
            stock["outdtype"],
            stock["libtype"],
            stock["scaleAB"],
            stock["bias"],
        ) != ("gfx942", "304", str(FP8), str(BF16), "hipblaslt", "True", "False"):
            p.error(
                "stock CSV must contain gfx942/304 bias-free FP8 → BF16 hipBLASLt PTPC rows"
            )
        key = row_key(stock)
        row, detail = select(stock, profile, rows.get(key))
        rows[key] = row
        diagnostics.append(dict(zip(KEY, key), **detail))
        print(
            stock["M"],
            stock["N"],
            stock["K"],
            "packed",
            stock["bpreshuffle"],
            "solution",
            row["solidx"],
            flush=True,
        )
    save(args.output.with_suffix(".samples.json"), diagnostics)
    tmp = args.output.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows.values())
    tmp.replace(args.output)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
