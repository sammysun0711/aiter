#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AITer_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RESULT_DIR="${RESULT_DIR:-${AITer_DIR}/mi325x_e2e_tuning_results}"

INPUT_FILE="${INPUT_FILE:-${SCRIPT_DIR}/llama70b_e2e_fp16_bf16_untuned_gemm.csv}"
TUNED_FILE="${TUNED_FILE:-${SCRIPT_DIR}/llama70b_e2e_fp16_bf16_tuned_gemm.csv}"
PROFILE_FILE="${PROFILE_FILE:-${RESULT_DIR}/llama70b_e2e_fp16_all_candidates.csv}"

LIBTYPE="${LIBTYPE:-all}"
WITH_HIPBLASLT="${WITH_HIPBLASLT:-1}"
MP="${MP:-8}"
BATCH="${BATCH:-100}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-101}"

mkdir -p "${RESULT_DIR}"
cd "${AITer_DIR}"

cmd=(
  python3 csrc/gemm_a16w16/gemm_tuner.py
  --input_file "${INPUT_FILE}"
  --tuned_file "${TUNED_FILE}"
  --profile_file "${PROFILE_FILE}"
  --libtype "${LIBTYPE}"
  --mp "${MP}"
  --batch "${BATCH}"
  --warmup "${WARMUP}"
  --iters "${ITERS}"
)

if [[ "${WITH_HIPBLASLT}" == "1" ]]; then
  cmd+=(--with-hipblaslt)
fi

cmd+=("$@")

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
