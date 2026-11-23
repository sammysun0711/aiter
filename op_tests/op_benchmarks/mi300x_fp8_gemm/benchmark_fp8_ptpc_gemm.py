# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Replay ordinary and packed PTPC GEMMs in paired timing rounds on MI300X."""

import argparse
import math
import os
import random
from functools import partial
from pathlib import Path
from statistics import median


def main():
    configs = Path(__file__).resolve().parent / "configs"
    parser = argparse.ArgumentParser(description=__doc__)
    shapes = parser.add_mutually_exclusive_group()
    shapes.add_argument("--shape", nargs=3, type=int, metavar=("M", "N", "K"))
    shapes.add_argument("--input", type=Path, help="CSV containing M,N,K")
    parser.add_argument("--table", type=Path, default=configs / "ptpc.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--ops", type=int, default=3)
    args = parser.parse_args()
    if min(args.rounds, args.groups, args.ops) < 1:
        parser.error("rounds, groups and ops must be positive")
    os.environ["AITER_CONFIG_GEMM_BF16"] = str(args.table.resolve(strict=True))

    # Runtime tables must be selected before importing AITER.
    import torch
    from common import (
        BF16,
        FP8,
        Inputs,
        capture_result,
        error_metric,
        is_mi300x_row,
        is_ptpc_row,
        read_csv,
        read_shapes,
        require_mi300x,
        sample_error,
        save,
        shape_key,
        time_group,
        validate_runtime_row,
    )

    from aiter.tuned_gemm import get_GEMM_A16W16_config, tgemm

    torch.cuda.set_device(0)
    require_mi300x()
    targets = (
        [tuple(args.shape)]
        if args.shape
        else read_shapes(args.input or configs / "ptpc-tune-input.csv")
    )
    if not targets:
        parser.error("the input CSV has no shapes")
    expected = {}
    for row in read_csv(args.table):
        if not is_mi300x_row(row):
            continue
        for packed in (False, True):
            if is_ptpc_row(row, packed):
                key = (*shape_key(row), packed)
                if key in expected:
                    parser.error(f"Duplicate PTPC row for {key}")
                expected[key] = row

    with torch.inference_mode():
        for shape in targets:
            m, n, k = shape
            data = Inputs(shape, ptpc=True)
            ri, ci, reference, _ = data.reference()
            stream = torch.cuda.Stream()
            graphs, outputs, records = {}, {}, {}
            for packed in (False, True):
                row = expected.get((*shape, packed))
                if row is None:
                    parser.error(f"No exact PTPC row for {shape}, packed={packed}")
                selected = (
                    get_GEMM_A16W16_config(
                        m, n, k, False, str(FP8), str(BF16), True, packed
                    )
                    or {}
                )
                validate_runtime_row(shape, row, selected, packed=packed, ptpc=True)
                call = partial(
                    tgemm.mm,
                    data.x,
                    data.packed if packed else data.w,
                    otype=BF16,
                    scale_a=data.sx,
                    scale_b=data.sw,
                    out=data.out,
                )
                eager_error = sample_error(call(), ri, ci, reference)
                graph, output = capture_result(call, stream, args.ops)
                graph_error = error_metric(graph.replay, output, ri, ci, reference)
                name = "ptpc_tgemm_packed" if packed else "ptpc_tgemm_plain"
                graphs[name], outputs[name] = graph, output
                records[name] = {
                    "name": name,
                    "scaling": "ptpc",
                    "weight": "packed" if packed else "ordinary",
                    "rows": m,
                    "calls": 1,
                    "backend": "hipBLASLt",
                    "kernel": selected["kernelName"],
                    "kernel_id": str(selected["solidx"]),
                    "status": "passed",
                    "rounds_us": [],
                    "relative_rms": max(eager_error[0], graph_error[0]),
                    "max_over_rms": max(eager_error[1], graph_error[1]),
                }
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
                error = error_metric(graph.replay, outputs[name], ri, ci, reference)
                row = records[name]
                us = median(value for group in row["rounds_us"] for value in group)
                row.update(
                    median_us=us,
                    tflops=2 * math.prod(shape) / us / 1e6,
                    relative_rms=max(row["relative_rms"], error[0]),
                    max_over_rms=max(row["max_over_rms"], error[1]),
                )
            save(
                args.output / ("-".join(map(str, shape)) + ".json"),
                {
                    "shape": shape,
                    "input_recipe": data.recipe,
                    "input_seed": data.seed,
                    "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
                    "samples_checked": reference.numel(),
                    "graph_ops": args.ops,
                    "rounds": args.rounds,
                    "groups": args.groups,
                    "methods": records,
                },
            )
            print(
                shape,
                {name: round(row["tflops"], 1) for name, row in records.items()},
                flush=True,
            )
            del graphs, outputs, records, graph, output, call, data
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
