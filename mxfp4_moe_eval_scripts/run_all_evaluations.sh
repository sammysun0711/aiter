#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AITER_ROOT="${AITER_ROOT:-$(cd "${BUNDLE_ROOT}/.." && pwd)}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${AITER_ROOT}:${PYTHONPATH:-}"

mkdir -p "${BUNDLE_ROOT}/results/logs"
cd "${BUNDLE_ROOT}"

python benchmark_qwen3_8_tuned_wrapper.py \
  --output results/qwen3_8_tuned_native_wrapper_tp8.json \
  2>&1 | tee results/logs/native_a16w4_wrapper.log

python "${AITER_ROOT}/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py" \
  --run_config tuning/qwen3_8_tp8_bf16_tuned.csv --warmup 2 --iters 7 \
  2>&1 | tee results/logs/bf16_run_config.log

python "${AITER_ROOT}/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py" \
  --run_config tuning/qwen3_8_tp8_fp8_ptpc_tuned.csv --warmup 2 --iters 7 \
  2>&1 | tee results/logs/fp8_ptpc_run_config.log

python "${AITER_ROOT}/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py" \
  --run_config tuning/qwen3_8_tp8_wi4a16_tuned.csv --warmup 2 --iters 7 \
  2>&1 | tee results/logs/flydsl_wi4a16_run_config.log

python collect_e2e_results.py --parse-logs
