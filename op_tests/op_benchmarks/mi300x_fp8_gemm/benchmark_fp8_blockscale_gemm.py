# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Replay a public AITER FP8 GEMM for customer shapes and report its runtime selection."""

import argparse
import csv
import math
import os
from functools import partial
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory


def benchmark(shape, args, row):
    import torch
    from common import (
        BF16,
        FP8,
        Inputs,
        capture_result,
        error_metric,
        packed_scales,
        sample_error,
        time_group,
        validate_runtime_row,
    )

    from aiter.jit.core import AITER_CONFIGS
    from aiter.ops.gemm_op_a8w8 import (
        gemm_a8w8_blockscale,
        gemm_a8w8_blockscale_bpreshuffle,
        get_CKGEMM_config,
    )
    from aiter.tuned_gemm import get_GEMM_A16W16_config, tgemm

    m, n, k = shape
    packed = args.layout == "packed"
    ptpc = args.scaling == "ptpc"
    table = getattr(AITER_CONFIGS, args.config_env + "_FILE")
    if ptpc:
        cfg = get_GEMM_A16W16_config(m, n, k, False, str(FP8), str(BF16), True, packed)
    else:
        cfg = get_CKGEMM_config(m, n, k, table)
    cfg = cfg or {}
    # Compare the requested exact row with the public dispatch lookup, which can
    # otherwise select a padded shape or the library default.
    if row:
        validate_runtime_row(shape, row, cfg, packed=packed, ptpc=ptpc)
    elif cfg:
        validate_runtime_row(shape, cfg, cfg, packed=packed, ptpc=ptpc)

    d = Inputs(shape, ptpc=ptpc, allocate_output=packed or ptpc)
    ri, ci, ptpc_ref, block_ref = d.reference()
    ref = ptpc_ref if ptpc else block_ref
    if ptpc:
        run = partial(
            tgemm.mm,
            d.x,
            d.packed if packed else d.w,
            otype=BF16,
            scale_a=d.sx,
            scale_b=d.sw,
            out=d.out,
        )
    elif packed:
        run = partial(
            gemm_a8w8_blockscale_bpreshuffle,
            d.x,
            d.packed,
            packed_scales(d.bx),
            d.bw,
            BF16,
            out=d.out,
        )
    else:
        run = partial(gemm_a8w8_blockscale, d.x, d.w, d.bx, d.bw, BF16)

    eager_error = sample_error(run(), ri, ci, ref)
    graph, out = capture_result(run, torch.cuda.Stream(), 3)
    graph_error = error_metric(graph.replay, out, ri, ci, ref)
    us = median(time_group(graph, 3) for _ in range(9))
    final_error = error_metric(graph.replay, out, ri, ci, ref)
    rms, peak = map(max, zip(eager_error, graph_error, final_error))
    return {
        "M": m,
        "N": n,
        "K": k,
        "scaling": args.scaling,
        "layout": args.layout,
        "input_recipe": d.recipe,
        "input_seed": d.seed,
        "table": str(table),
        "libtype": cfg.get("libtype", "default (no tuned row)"),
        "kernelName": cfg.get("kernelName", ""),
        "solidx": cfg.get("solidx", ""),
        "kernelId": cfg.get("kernelId", ""),
        "splitK": cfg.get("splitK", ""),
        "rows_per_call": m,
        "calls": 1,
        "median_us": us,
        "tflops": 2 * math.prod(shape) / us / 1e6,
        "sampled_outputs": ref.numel(),
        "relative_rms": rms,
        "max_error_over_rms": peak,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    shapes = parser.add_mutually_exclusive_group(required=True)
    shapes.add_argument("--shape", type=int, nargs=3, metavar=("M", "N", "K"))
    shapes.add_argument("--input", type=Path, help="CSV with M,N,K columns")
    parser.add_argument("--scaling", choices=("ptpc", "block128"), required=True)
    parser.add_argument("--layout", choices=("ordinary", "packed"), required=True)
    parser.add_argument(
        "--table", type=Path, help="Runtime CSV; require a matching row for each call"
    )
    parser.add_argument(
        "--output", type=Path, help="Write one verified result per shape to CSV"
    )
    args = parser.parse_args()
    if args.scaling == "ptpc":
        args.config_env = "AITER_CONFIG_GEMM_BF16"
    elif args.layout == "packed":
        args.config_env = "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE"
    else:
        args.config_env = "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE"
    if args.table:
        os.environ[args.config_env] = str(args.table.resolve(strict=True))
    # AITER reads table overrides during import.
    import torch
    from common import (
        is_mi300x_row,
        is_ptpc_row,
        read_csv,
        read_shapes,
        require_mi300x,
        shape_key,
    )

    targets = [tuple(args.shape)] if args.shape else read_shapes(args.input)
    if not targets:
        parser.error("the input CSV has no shapes")
    exact = {}
    for r in read_csv(args.table) if args.table else ():
        if not is_mi300x_row(r):
            continue
        if args.scaling == "ptpc" and not is_ptpc_row(r, args.layout == "packed"):
            continue
        key = shape_key(r)
        if key in exact:
            parser.error(f"duplicate runtime rows for {key} in {args.table}")
        exact[key] = r

    torch.cuda.set_device(0)
    require_mi300x()
    results = []
    with torch.inference_mode():
        for shape in targets:
            row = exact.get(shape)
            if args.table and not row:
                parser.error(
                    f"no exact {args.scaling}/{args.layout} row for {shape} in {args.table}"
                )
            result = benchmark(shape, args, row)
            results.append(result)
            print(
                f"{shape} {args.scaling}/{args.layout}: {result['libtype']} "
                f"{result['kernelName']} solidx={result['solidx']} "
                f"{result['calls']} call(s), {result['median_us']:.2f} us, "
                f"{result['tflops']:.1f} TFLOP/s, relative RMS {result['relative_rms']:.5f}",
                flush=True,
            )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.output.parent) as tmp:
            report = Path(tmp) / "report.csv"
            with report.open("w", newline="") as f:
                writer = csv.DictWriter(f, results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            report.replace(args.output)


if __name__ == "__main__":
    main()
