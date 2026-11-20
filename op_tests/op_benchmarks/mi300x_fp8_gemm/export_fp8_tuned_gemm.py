# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Save the fastest validated candidate per shape, scaling contract and libtype."""

import argparse
import csv
import json
import math
import os
import runpy
from collections import defaultdict
from pathlib import Path
from statistics import median

from summarize_fp8_scaling_comparison import LABELS, PROJECTIONS, SIZES

ROOT = Path(__file__).resolve().parent
CONFIG_TABLES = {
    "block_ck_plain": "block-plain.csv",
    "block_ck": "ck.csv",
    "block_cktile": "cktile.csv",
    "block_asm": "asm.csv",
    "block_opus": "opus.csv",
}
RECORDED_FLYDSL = {"block_flydsl", "block_lds_m", "block_lds_n", "block_register"}
FIELDS = [
    "gfx",
    "cu_num",
    "M",
    "N",
    "K",
    "scaling",
    "libtype",
    "method",
    "weight_layout",
    "dtype",
    "outdtype",
    "solidx",
    "kernelId",
    "kernelName",
    "splitK",
    "rows_per_call",
    "calls",
    "us",
    "tflops",
    "relative_rms",
    "max_error_over_rms",
    "passed_candidates",
    "source",
]


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def libtype(name, item):
    if name.endswith("_torch"):
        return "torch"
    return item["backend"].lower()


def value_row(shape, scaling, lib, winner, count, tables, source):
    m, n, k = shape
    method, item = winner
    step = item["rows"]
    assert item["calls"] * step == m
    kid, solution, split = item["kernel_id"], "", ""
    if lib == "hipblaslt":
        solution, kid = kid, ""
        (tuned,) = [
            r
            for r in tables["ptpc"]
            if (int(r["M"]), int(r["N"]), int(r["K"])) == shape
            and r["solidx"] == str(solution)
            and (r["bpreshuffle"] == "True") == (item["weight"] == "packed")
        ]
        split = tuned["splitK"]
    elif method in CONFIG_TABLES:
        (tuned,) = [
            r
            for r in tables[method]
            if (int(r["M"]), int(r["N"]), int(r["K"])) == (step, n, k)
            and r["kernelName"] == item["kernel"]
        ]
        kid, split = tuned["kernelId"], tuned["splitK"]
    elif lib == "flydsl":
        split = 0
    return {
        "gfx": "gfx942",
        "cu_num": 304,
        "M": m,
        "N": n,
        "K": k,
        "scaling": scaling,
        "libtype": lib,
        "method": method,
        "weight_layout": item["weight"],
        "dtype": "torch.float8_e4m3fnuz",
        "outdtype": "torch.bfloat16",
        "solidx": solution,
        "kernelId": kid,
        "kernelName": item["kernel"],
        "splitK": split,
        "rows_per_call": step,
        "calls": item["calls"],
        "us": item["median_us"],
        "tflops": item["tflops"],
        "relative_rms": item["relative_rms"],
        "max_error_over_rms": item["max_over_rms"],
        "passed_candidates": count,
        "source": os.path.relpath(source, ROOT),
    }


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def export(raw, configs, output, runtime_dir=None):
    # The registry is pure Python. Load it without importing Torch or requiring a GPU for export.
    registry = runpy.run_path(
        str(
            ROOT.parents[2]
            / "aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_blockscale_common.py"
        )
    )
    tables = {key: read_csv(configs / file) for key, file in CONFIG_TABLES.items()}
    tables["ptpc"] = read_csv(configs / "ptpc.csv")
    rows = []
    flydsl_tables = defaultdict(list)
    for n, k in PROJECTIONS:
        for m in SIZES:
            shape = (m, n, k)
            path = raw / f"{m}-{n}-{k}.json"
            run = json.loads(path.read_text())
            fly = {
                f"block_flydsl_id{kid}": (kid, ki)
                for kid, ki in registry["kernels_list"].items()
                if registry["kernel_fits_shape"](ki, *shape, "gfx942")
            }
            expected = (set(LABELS) - RECORDED_FLYDSL) | set(fly)
            if tuple(run["shape"]) != shape or set(run["methods"]) != expected:
                raise ValueError(
                    f"{path} must contain the complete --all-flydsl replay for {shape}"
                )
            groups = defaultdict(list)
            for name, item in run["methods"].items():
                if name in fly:
                    kid, ki = fly[name]
                    layout = "packed" if ki.preshuffle_b else "ordinary"
                    if (int(item["kernel_id"]), item["kernel"], item["weight"]) != (
                        kid,
                        ki.name,
                        layout,
                    ):
                        raise ValueError(
                            f"{path}: {name} disagrees with the FlyDSL registry"
                        )
                if item["status"] != "passed":
                    continue
                samples = item["rounds_us"]
                us = median(t for group in samples for t in group)
                if (
                    len(samples) != run["rounds"]
                    or any(len(group) != run["groups"] for group in samples)
                    or not math.isclose(us, item["median_us"])
                    or not math.isclose(2 * m * n * k / us / 1e6, item["tflops"])
                    or not item["relative_rms"] < 0.006
                    or not item["max_over_rms"] < 0.035
                ):
                    raise ValueError(
                        f"{path}: inconsistent timing or numerical check for {name}"
                    )
                groups[item["scaling"], libtype(name, item)].append((name, item))
            for (scaling, lib), candidates in sorted(groups.items()):
                best = min(candidates, key=lambda pair: (pair[1]["median_us"], pair[0]))
                rows.append(
                    value_row(shape, scaling, lib, best, len(candidates), tables, path)
                )
                if scaling == "block128" and lib == "flydsl" and runtime_dir:
                    for layout in ("ordinary", "packed"):
                        eligible = [
                            item for _, item in candidates if item["weight"] == layout
                        ]
                        if eligible:
                            winner = min(
                                eligible,
                                key=lambda item: (item["median_us"], item["name"]),
                            )
                            flydsl_tables[layout].append(
                                {
                                    "gfx": "gfx942",
                                    "cu_num": 304,
                                    "M": m,
                                    "N": n,
                                    "K": k,
                                    "libtype": lib,
                                    "kernelId": winner["kernel_id"],
                                    "splitK": 0,
                                    "us": winner["median_us"],
                                    "kernelName": winner["kernel"],
                                    "tflops": winner["tflops"],
                                    "bw": (m * k + n * k + 2 * m * n)
                                    / winner["median_us"]
                                    / 1000,
                                    "errRatio": "",
                                }
                            )
    write_csv(output, FIELDS, rows)
    if runtime_dir:
        with (configs / "flydsl.csv").open(newline="") as src:
            fields = next(csv.reader(src))
        for layout, filename in (
            ("ordinary", "flydsl-ordinary.csv"),
            ("packed", "flydsl-final.csv"),
        ):
            write_csv(runtime_dir / filename, fields, flydsl_tables[layout])
    print(f"Wrote {len(rows)} per-library winners to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "results/per-lib-replay")
    parser.add_argument(
        "--configs",
        type=Path,
        default=ROOT / "configs",
        help="Runtime tables used to generate the replay",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/fp8-tuned-gemm.csv"
    )
    parser.add_argument(
        "--flydsl-runtime-dir",
        type=Path,
        help="Also export one public dispatch CSV per FlyDSL weight layout",
    )
    args = parser.parse_args()
    export(args.raw, args.configs, args.output, args.flydsl_runtime_dir)
