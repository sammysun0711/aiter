#!/usr/bin/env python3
"""Rebuild the consolidated Qwen3.8 TP8 local-MoE E2E CSV.

By default this reads the checked-in source JSON files. With ``--parse-logs``
it first refreshes the BF16, FP8 PTPC, and FlyDSL JSON files from logs produced
by ``run_all_evaluations.sh``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
EXPECTED_M = [1, 4, 8, 16, 32, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768]
BENCHMARK_ROW = re.compile(
    r"^\((\d+),.*?\)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*(\w+)\s*$",
    re.MULTILINE,
)


def load_results(path: Path, value_key: str) -> dict[int, float]:
    payload = json.loads(path.read_text())
    return {int(row["m"]): float(row[value_key]) for row in payload["results"]}


def refresh_json_from_log(json_path: Path, log_path: Path) -> None:
    rows = [
        {
            "m": int(m),
            "kernel_us": float(kernel_us),
            "e2e_us": float(e2e_us),
            "status": status.upper(),
        }
        for m, kernel_us, e2e_us, status in BENCHMARK_ROW.findall(log_path.read_text())
    ]
    if [row["m"] for row in rows] != EXPECTED_M:
        raise RuntimeError(f"unexpected M sequence in {log_path}: {rows}")
    payload = json.loads(json_path.read_text())
    if "kernel_us" in payload["results"][0]:
        payload["results"] = rows
    else:
        payload["results"] = [
            {"m": row["m"], "e2e_us": row["e2e_us"], "status": row["status"]}
            for row in rows
        ]
    json_path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parse-logs", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "qwen3_8_tp8_tuned_moe_all_e2e.csv",
    )
    args = parser.parse_args()

    if args.parse_logs:
        log_dir = RESULTS / "logs"
        refresh_json_from_log(
            RESULTS / "aiter_ck_bf16_tuned_e2e.json",
            log_dir / "bf16_run_config.log",
        )
        refresh_json_from_log(
            RESULTS / "aiter_ck_fp8_ptpc_tuned_e2e.json",
            log_dir / "fp8_ptpc_run_config.log",
        )
        refresh_json_from_log(
            RESULTS / "aiter_flydsl_wi4a16_tuned_e2e.json",
            log_dir / "flydsl_wi4a16_run_config.log",
        )

    columns = {
        "native_mxfp4_a16w4_topk_sort_expert_us": load_results(
            RESULTS / "qwen3_8_tuned_native_wrapper_tp8.json", "median_us"
        ),
        "converted_int4_flydsl_wi4a16_sort_expert_us": load_results(
            RESULTS / "aiter_flydsl_wi4a16_tuned_e2e.json", "e2e_us"
        ),
        "bf16_ck_tile_sort_expert_us": load_results(
            RESULTS / "aiter_ck_bf16_tuned_e2e.json", "e2e_us"
        ),
        "fp8_ptpc_activation_quant_sort_expert_us": load_results(
            RESULTS / "aiter_ck_fp8_ptpc_tuned_e2e.json", "e2e_us"
        ),
    }
    for name, values in columns.items():
        if sorted(values) != EXPECTED_M:
            raise RuntimeError(f"{name} has unexpected M values: {sorted(values)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["m", *columns]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for m in EXPECTED_M:
            writer.writerow(
                {"m": m, **{name: f"{values[m]:.2f}" for name, values in columns.items()}}
            )
    print(args.output)


if __name__ == "__main__":
    main()
