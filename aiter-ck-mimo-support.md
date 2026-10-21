# AITER / CK MiMo handoff

Updated: 2026-10-21. Source machine: `quanta-ccs-aus-d05-11`.

The gfx942 **BF16** MiMo Q/K192–V128 LINEAR crash fix and native kernel have
passed focused GPU and server validation. The **FP8** native extension is
implemented and compile-checked, but **GPU correctness and tuning have not
started**. Continue FP8 qualification on an idle, healthy gfx942 machine.

## Repository state

| Item | Value |
| --- | --- |
| Source checkout | `/root/workspace/aiter` |
| AITER branch | `tq/gfx942-a16w4-a8w4` |
| AITER HEAD | `007adf4c6726978eb4427af4b3adfd4b716e536e` |
| CK submodule HEAD | `af9e1d1f1ae347c22feeb08fd2d42645075e0c5d` |
| Original GPU | MI325X, gfx942, 304 CUs, approximately 256 GiB VRAM |
| PyTorch | `2.9.1+rocm7.2.0.git7e1940d4` |
| HIP | `7.2.26015-fc0010cf6a` |

All changes below are local and unstaged. The CK gitlink is unchanged; its
working tree contains modifications to two files. The existing page-64 patch
was already present before this investigation and must be preserved.

## Local changes

| File | Purpose and status |
| --- | --- |
| `aiter/ops/mha.py` | Adds distinct JIT module names for gfx942 BF16 and FP8 Q/K192–V128 LINEAR tiles. Appends `_gload_bounds_v1` to batch-prefill module names so an older cached binary cannot hide the bounds fix. |
| `3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_scatter_gather.hpp` | Restores coordinate and optional gather-validity checks in global V loads and asynchronous global-to-LDS K loads. Invalid V vectors become zero; invalid K transfers use a zero-length SRD to zero-fill LDS while preserving transfer layout/order. Applies to generic fallback kernels as well as MiMo. |
| `3rdparty/composable_kernel/example/ck_tile/01_fmha/codegen/ops/fmha_batch_prefill.py` | Contains the pre-existing page-64 asymmetric-head/gfx950 tuning changes, plus new gfx942 BF16 and FP8 LINEAR page-1 specializations and separate tile overrides. Exact head/layout/table dispatch retains generic fallback for other contracts. |
| `patches/composable_kernel/README.md` | Documents root cause, BF16 patch application, tuning and server results, and the pending FP8 extension. Its main Apply example lists the first three patches; use the four-patch order below for this complete handoff. |
| `patches/composable_kernel/batch_prefill_global_load_bounds.patch` | New portable CK bounds fix. |
| `patches/composable_kernel/mimo_gfx942_bf16_linear_prefill.patch` | New portable native BF16 LINEAR codegen patch; measured default `128x64`. |
| `patches/composable_kernel/mimo_gfx942_fp8_linear_prefill.patch` | New portable native FP8 LINEAR codegen patch; provisional default `128x96`. |
| `patches/composable_kernel/tuning/gfx942_bf16_linear_20261021.json` | BF16 raw timing samples, environment and selected tile. |
| `patches/composable_kernel/tuning/gfx942_server_regression_20261021.json` | BF16 server crash-regression metrics and original log location. |
| `op_tests/ck_prefill_guard.cpp` | HIP virtual-memory allocator with an unmapped guard immediately after the logical allocation. Address bands `0x90000000` and `0x10000000` force global and buffer load selection, respectively. |
| `op_tests/ck_prefill_guard.py` | Guarded BF16/FP8 cases, sink-aware reference, per-tensor FP8 quantization/descales and accuracy checks. FP8 storage/indexing uses uint8 views. Test processes exit after synchronization to release mappings and avoid DLPack/ctypes teardown ordering issues. |
| `op_tests/test_batch_prefill_bounds.py` | Expanded 20-case GPU matrix: five shapes × BF16/FP8 × global/buffer loads. Covers the original failure, ragged tails, shuffled indices and generic D160/V96 and D96/V64 heads. The expanded matrix is pending GPU execution. |
| `op_tests/benchmark_mimo_ck_linear_prefill.py` | Six guarded workloads; validates accuracy before reporting three CUDA-graph timing samples and their median. Supports both dtypes and load modes. |
| `op_tests/test_mimo_fp8_linear_codegen.py` | Nine CPU-only tests: six FP8 tiles, generic exclusion, gfx950 exclusion and independence from the BF16 override. Passed. |

The tracked `mimo_page64_qk192_v128_batch_prefill.patch` and
`op_tests/test_batch_prefill_asymmetric.py` are prerequisites already in AITER
HEAD. They are not new changes in this investigation.

## Why both the bounds fix and native kernels are retained

SGLang's MiMo target SWA prefill fallback gathers its page-64 KV cache into
temporary linear tensors. AITER represents each token as a page-1 entry for
that LINEAR call. The physical allocator remains page-64. DFlash draft and
target-verification SWA use separate paths.

The original failing call had BF16 Q `[3968, 16, 192]`, K `[16384, 1, 192]`,
V `[16384, 1, 128]`, a 12,416-token prefix and window `(128, -1)`. The D256
fallback loaded padded head columns beyond the actual K/V allocations.
GLOBAL_LOAD_LDS selection depends on the low 32 bits of the virtual addresses
as well as size, so ordinary small allocations did not reliably reproduce it.
The captured fault matched the end of K, and guarded allocations independently
reproduced both K and V overreads.

Native D192/V128 tiles avoid that head-padding mismatch and improve performance.
The generic bounds fix independently protects fallback head widths and invalid
coordinates. Keep both. Invalid asynchronous K loads must zero LDS; merely
skipping them can reuse stale data.

The temporary SGLang `_pad_mimo_ck_linear_kv` workaround was removed. There is
no remaining change to
`sglang/python/sglang/srt/layers/attention/aiter_utils.py`. Logical head dimensions
and attention scaling remain Q/K192 and V128; framework padding to 256 is not
required by this implementation.

## Dispatch and tuning controls

| Contract | Environment variable | Values | Current default |
| --- | --- | --- | --- |
| gfx942 BF16 LINEAR Q/K192–V128 | `AITER_MIMO_GFX942_LINEAR_TILE` | `generic`, `64x64`, `64x128`, `128x64`, `128x128` | `128x64`, measured |
| gfx942 FP8 LINEAR Q/K192–V128 | `AITER_MIMO_GFX942_FP8_LINEAR_TILE` | `generic`, `64x64`, `64x96`, `64x128`, `128x64`, `128x96`, `128x128` | `128x96`, unmeasured |

`generic` excludes the native LINEAR family and uses the bounds-corrected
generic fallback. It does not remove the bounds fix or disable native page-64
kernels. BF16 and FP8 settings are independent.

The FP8 contract is Q, K and V in gfx942 E4M3FNUZ (`torch.float8_e4m3fnuz` /
`aiter.dtypes.fp8`), per-tensor Q/K/V descales, BF16 output, LINEAR page-1 KV,
and SGLang's 1D lookup table. Block scaling is outside this new specialization.
Its MFMA geometry uses 32×32×32 warp tiles, BK0/BK1=32 and M/32 waves.

The current code enables the provisional native FP8 tile by default for this
contract. Set `AITER_MIMO_GFX942_FP8_LINEAR_TILE=generic` when the unqualified
native extension should be excluded. No FP8 speedup or runtime-correctness
claim has been established yet.

## Validation already completed

- BF16 guarded correctness checks passed for the corrected generic and native
  paths with unpadded K/V. Focused existing attention regressions passed.
- BF16 tile measurements used SWA window 128, 16 Q heads, one KV head and forced
  global-load addresses. Each result is the median of three 100-ms graph runs.
- At Q=32768/KV=65536, generic D256 took **769.37 µs**, versus **485.84 µs** for
  native `128x64`. The ragged 32K-query workload took **778.05 → 482.64 µs**.
  These are isolated attention measurements, approximately 37–38% less kernel
  time, not serving-throughput speedups. `128x128` was competitive on smaller
  cases; `128x64` was selected for the 32K workload.
- With the SGLang workaround removed, a 36,736-token cached-tail request and
  **600/600** 16K-input requests at concurrency 16 passed. The measured run used
  one output token, four warmups, seed `782693892`, normal asynchronous
  execution and the original launch configuration. It processed 9,830,400
  input tokens in 704.26 seconds, **13,958.39 input tokens/s**, with zero recorded
  server faults. This passed the old failure point near 470 completions; it
  was not the entire long-context benchmark matrix.
- The validation server and workers were stopped after testing.
- FP8: twelve representative generated translation units compiled for gfx942
  (six native tiles × global/buffer loads). Nine CPU codegen tests passed.
  Compilation does not establish successful runtime JIT dispatch or accuracy.
- The four CK patches were reapplied in a temporary directory during handoff
  preparation and reproduced both modified CK files byte-for-byte. Reversing
  all four recovered the pinned base. The working submodule was not changed.

Earlier BF16 validation predates the FP8 extension. Re-run BF16 compatibility
alongside FP8 qualification. The current 20-case guard matrix and all FP8 GPU
tests/timings remain pending.

## Move the work to another machine

Carry the untracked files as well as the tracked changes. An AITER commit or
top-level `git diff` alone will omit the dirty submodule implementation and
untracked tests/patches/results.

The following exports a source-only bundle from the current AITER root. It
keeps the CK aggregate diff as a reference; restoration uses the ordered patch
series. No JIT binaries or host-specific HIP helper library are required.

```bash
MIMO_HANDOFF_DIR=$(mktemp -d /tmp/aiter-ck-mimo-handoff.XXXXXX)
mkdir -p "$MIMO_HANDOFF_DIR/op_tests" "$MIMO_HANDOFF_DIR/patches/composable_kernel"
git diff --binary -- aiter/ops/mha.py patches/composable_kernel/README.md \
  > "$MIMO_HANDOFF_DIR/aiter-working-tree.patch"
git -C 3rdparty/composable_kernel diff --binary \
  > "$MIMO_HANDOFF_DIR/ck-working-tree.reference.patch"
cp aiter-ck-mimo-support.md "$MIMO_HANDOFF_DIR/"
cp op_tests/benchmark_mimo_ck_linear_prefill.py \
  op_tests/ck_prefill_guard.cpp op_tests/ck_prefill_guard.py \
  op_tests/test_batch_prefill_bounds.py op_tests/test_mimo_fp8_linear_codegen.py \
  "$MIMO_HANDOFF_DIR/op_tests/"
cp patches/composable_kernel/*.patch "$MIMO_HANDOFF_DIR/patches/composable_kernel/"
cp -a patches/composable_kernel/tuning "$MIMO_HANDOFF_DIR/patches/composable_kernel/"
tar -C "$MIMO_HANDOFF_DIR" -czf "$MIMO_HANDOFF_DIR.tar.gz" .
printf 'Transfer this archive: %s\n' "$MIMO_HANDOFF_DIR.tar.gz"
```

On the destination, start from a clean AITER checkout at the recorded HEAD,
with the clean pinned CK submodule. Extract the archive at the AITER root and
run the following. These commands are for a clean destination, not the already
patched source checkout.

```bash
set -e
test "$(git rev-parse HEAD)" = 007adf4c6726978eb4427af4b3adfd4b716e536e
git submodule update --init 3rdparty/composable_kernel
test "$(git -C 3rdparty/composable_kernel rev-parse HEAD)" = af9e1d1f1ae347c22feeb08fd2d42645075e0c5d
git apply --check aiter-working-tree.patch
git apply aiter-working-tree.patch
for patch in \
  mimo_page64_qk192_v128_batch_prefill.patch \
  batch_prefill_global_load_bounds.patch \
  mimo_gfx942_bf16_linear_prefill.patch \
  mimo_gfx942_fp8_linear_prefill.patch; do
  git -C 3rdparty/composable_kernel apply --check < "patches/composable_kernel/$patch"
  git -C 3rdparty/composable_kernel apply < "patches/composable_kernel/$patch"
done
```

Do not additionally apply `ck-working-tree.reference.patch` after the series.
If transferring the full working tree instead, the patches may already be
applied; inspect before applying anything. Reverse patch order for rollback.

Use a ROCm/PyTorch/AITER environment compatible with this checkout and confirm
Python imports this AITER source. Rebuild the helper and JIT modules on the
destination. Source-only FP8 tuning needs no model weights or SGLang server.

## Follow-up: FP8 correctness and tuning

### 1. Prepare an idle gfx942 GPU

Confirm valid PCI/GPU telemetry and release of other workloads before GPU
submissions. The source machine's reset failure is described below. Native
gfx950 FP8 LINEAR support is not part of this change.

From the destination AITER root:

```bash
export HIP_VISIBLE_DEVICES=0
export AITER_JIT_DIR=$(mktemp -d /tmp/aiter-ck-mimo-jit.XXXXXX)
export MIMO_RESULTS_DIR=$(mktemp -d /tmp/aiter-ck-mimo-results.XXXXXX)
MIMO_ROCM_DIR=${ROCM_PATH:-/opt/rocm}
unset AITER_MIMO_GFX950_D192_TILE AITER_MIMO_GFX950_D192_OCCUPANCY
export AITER_MIMO_GFX942_LINEAR_TILE=128x64
python3 -m pytest -q op_tests/test_mimo_fp8_linear_codegen.py
g++ -shared -fPIC -O2 -D__HIP_PLATFORM_AMD__ -I"$MIMO_ROCM_DIR/include" \
  op_tests/ck_prefill_guard.cpp -L"$MIMO_ROCM_DIR/lib" \
  -Wl,-rpath,"$MIMO_ROCM_DIR/lib" -lamdhip64 -o "$MIMO_RESULTS_DIR/guard.so"
```

If the environment already sets other GPU visibility variables, align them
with the intended physical GPU. Keep the private JIT directory across the
sweep so each tile is compiled once under its distinct module name.

### 2. Validate the baseline and initial native candidate

Run each combination in its own process. Stop and investigate on a fault or
accuracy failure; recheck GPU health before further submissions.

```bash
set -e
for mode in global buffer; do
  for tile in generic 128x96; do
    HIP_LAUNCH_BLOCKING=1 AITER_MIMO_GFX942_FP8_LINEAR_TILE="$tile" \
      python3 op_tests/ck_prefill_guard.py \
        --library "$MIMO_RESULTS_DIR/guard.so" --cache-dtype fp8 --load-mode "$mode" \
        > "$MIMO_RESULTS_DIR/first-$tile-$mode.log" 2>&1
  done
done
```

The default case is Q=3968/KV=16384, Q/K192–V128, SWA window 128 with sinks.
The reference uses dequantized FP8 inputs and distinct non-unit Q/K/V scales.
Acceptance currently requires elementwise atol/rtol 0.055 and relative L2
error below 0.05. BF16 guard checks use atol/rtol 0.02.

### 3. Measure all six native candidates against generic

```bash
set -e
for mode in global buffer; do
  for tile in generic 64x64 64x96 64x128 128x64 128x96 128x128; do
    AITER_MIMO_GFX942_FP8_LINEAR_TILE="$tile" \
      python3 op_tests/benchmark_mimo_ck_linear_prefill.py \
        --library "$MIMO_RESULTS_DIR/guard.so" --cache-dtype fp8 \
        --load-mode "$mode" --window 128 \
        --output "$MIMO_RESULTS_DIR/fp8-$tile-$mode.json" \
        > "$MIMO_RESULTS_DIR/fp8-$tile-$mode.log" 2>&1
  done
done
```

Cases are Q/KV 129/257, 512/16384, 3968/16384, 16384/16384, 32768/65536,
and ragged Q `[3968,16384,12416]` / KV `[16384,16384,12416]`. The benchmark
checks accuracy before timing each case. Both address modes matter because
allocation history can change production dispatch. `--padding-baseline` is
BF16-only. Timing should use normal asynchronous execution, without globally
exporting `HIP_LAUNCH_BLOCKING=1`.

Choose a candidate that passes accuracy and performs well on the 32K SWA
workloads. Preserve per-case measurements, load mode, GPU/software versions,
and error metrics under `patches/composable_kernel/tuning/`. Do not infer a
full-server FP8 speedup from these isolated kernel timings.

### 4. Run the expanded compatibility checks

Set `AITER_MIMO_GFX942_FP8_LINEAR_TILE` to the chosen candidate, then run:

```bash
python3 -m pytest -q op_tests/test_batch_prefill_bounds.py \
  op_tests/test_batch_prefill_asymmetric.py

AITER_MIMO_GFX942_LINEAR_TILE=generic AITER_MIMO_GFX942_FP8_LINEAR_TILE=generic \
  python3 -m pytest -q op_tests/test_batch_prefill_bounds.py
```

The first command covers 20 guarded cases plus nine existing API cases,
including BF16/FP8 linear and page-64 layouts, full causal attention without
sinks, SWA with sinks, and BF16 ragged pages. The second exercises corrected
generic fallbacks. If first-use JIT compilation exceeds the guard test's
300-second child timeout, distinguish that from a GPU hang and prebuild the
required signature before retrying.

Also check the selected FP8 tile with the guarded helper using `--window -1`
and a modest ragged shape, for example `--q-lens 129 --kv-lens 257`, in both
load modes. This extends its sink-aware reference beyond the tuned SWA case.

### 5. Finalize the implementation and evidence

1. Update the FP8 default consistently in `aiter/ops/mha.py` and CK's
   `_get_mimo_gfx942_fp8_linear_tile()` only after measurements justify it.
2. Regenerate `mimo_gfx942_fp8_linear_prefill.patch` relative to CK after patches
   1–3, not relative to clean CK. Keep pre-existing page-64/gfx950 code intact.
3. Recheck four-patch application and reversal against the pinned CK base.
   Re-run CPU codegen checks and relevant GPU checks if the implementation
   changes, and perform the repository's applicable lint/build checks.
4. Update `patches/composable_kernel/README.md`: include patch 4 in the main
   Apply sequence, describe both guard load modes, replace provisional FP8
   text with actual results, and update this handoff's qualification status.
5. Keep selected-tile, generic-fallback and BF16 compatibility evidence. Full
   serving validation is a later integration check; kernel qualification does
   not require launching a server.
6. Stop any validation server and its workers when testing finishes, as the
   user requested. Preserve logs before cleanup.

Optional cleanup: the guard pytest function/docstring still says
`global_load` although the matrix now tests both load modes.

## Optional SGLang server regression

Requires a separate SGLang checkout, launch/benchmark scripts, model paths and
eight available GPUs. Existing SGLang evaluation/packaging changes are outside
this AITER bundle; preserve or transfer them separately if needed. The original
SGLang base was `1ea9aa3469`. Confirm the intended attention path and cache
dtype in logs; the completed server run validates the BF16 crash fix.

From `sglang/evaluation/mimo_dflash_scripts`:

```bash
MEM_FRACTION_STATIC=0.80 CHUNKED_PREFILL_SIZE=32768 \
DISABLE_RADIX_CACHE=1 ENABLE_TWO_BATCH_OVERLAP=0 \
MAX_RUNNING_REQUESTS=96 SGLANG_DFLASH_PERFORMANCE_MODE=1 \
SERVER_RANDOM_SEED=782693892 \
./launch_tp8_noep_aiter_dflash_accuracy_baseline_gfx942.sh
```

After the startup log reports readiness, from `sglang/evaluation`:

```bash
TOKEN_LIST_OVERRIDE=16384 NUM_PROMPTS_OVERRIDE=600 \
CONCURRENCY_LIST_OVERRIDE=16 LOG_DIR=./logs/ck_mimo_regression \
./run_benchmark_mimo_pro_prefill.sh
```

The original failing sweep requested 3,000 prompts; 600 is the focused regression
that passed the former failure point. Do not rely on changing script defaults.
Avoid inference-based `/health` warmup during diagnosis: an earlier one-token
probe exposed a separate page-1 dispatch issue. Wait for startup readiness in
the log, then run the controlled benchmark. Stop the server after the run.

## Source-machine failure and artifact locations

After a colleague's SGLang crash/reset, all eight source GPUs returned
`0xffffffff` from PCI configuration reads, with repeated `device lost from
bus!`, SDMA reset failures and missing clock/temperature/power telemetry.
Reported 82–83% VRAM usage did not establish live usable allocations. The
original trigger for that later event remains undetermined; do not attribute
it to the new FP8 extension from the available evidence.

Our readiness watcher `/tmp/ck_fp8_wait_and_validate.py` was stopped. It never
started the first FP8 GPU test. Do not restart it: its VRAM-only readiness
condition is insufficient after a failed reset. No host reset/reboot or
colleague process termination was performed by this investigation. The
previously completed validation server was already stopped.

These paths are on the source machine and are not included in the source-only
bundle above:

- `/root/workspace/sglang/evaluation/logs/dflash_ck_root_fix_20261021/`:
  final BF16 server/benchmark logs. Summary metrics are already in AITER's
  `patches/composable_kernel/tuning/`.
- `/root/workspace/sglang/evaluation/logs/dflash_crash_investigation_artifacts/`:
  original fault localization and historical padding-workaround probes. Its
  old `--sglang-padding` commands require the removed helper; use current AITER
  tests for ongoing work.
- `/root/workspace/sglang/evaluation/logs/gpu_reset_20261021_diagnostics/`:
  later hardware-failure report, kernel logs, PCI snapshots and eight device
  coredumps (approximately 364 MB total). Transfer separately if investigating
  that failure.
- The diagnostic directory's `fp8_pre_recovery_state/` contains preserved
  `results.json` and `results_bload.json` (all six exit codes zero per mode),
  `ck_fp8_codegen_tests.log` (nine passed), and `ck_before_fp8_linear.py`
  (generator after the first three patches). It also retains a historical
  watcher copy, which should not be launched.
- Original temporary build products were under `/tmp/ck_fp8_compile/`, helper
  libraries under `/tmp/ck_fp8_guard.so` and `/tmp/ck_prefill_guard.so`, and the
  planned JIT directory was `/tmp/aiter_ck_fp8_linear_tuning`. Rebuild on the
  destination; the older `/tmp/ck_prefill_guard.so` lacks the buffer-mode export.

For continuing kernel work, the AITER source bundle, this handoff and a healthy
gfx942 environment are sufficient. GPU runtime qualification and measured FP8
tile selection are the next outstanding tasks.
