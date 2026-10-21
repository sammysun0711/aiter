# AITER / CK MiMo gfx942 tuning

Updated: 2026-10-21. Qualification host: `quanta-ccs-aus-d05-03`.

FP8 runtime correctness and tuning are complete. The measured default is
**64x64**; BF16 uses **128x64**. Qualification found and fixed an additional FP8
sink-logit scaling bug affecting both native and generic kernels. All 65 GPU
regression tests and all 84 tuning cases passed.

## Repository and patch state

| Item | Value |
| --- | --- |
| Checkout | `/root/workspace/aiter` |
| Branch | `wip/mimo-gfx942-ck-tune` |
| Starting WIP commit | `67548e3cd3aec28474ea2f9ac2ae09afe38537e9` |
| Original AITER base | `007adf4c6726978eb4427af4b3adfd4b716e536e` |
| Pinned CK | `af9e1d1f1ae347c22feeb08fd2d42645075e0c5d` |
| GPU | MI325X, gfx942, 304 CUs, approximately 256 GiB VRAM |
| PyTorch / HIP | `2.9.1+rocm7.2.0.git7e1940d4` / `7.2.26015-fc0010cf6a` |

The CK gitlink remains pinned. Its three modified source files are packaged
in one patch: `patches/composable_kernel/mimo_page64_qk192_v128_batch_prefill.patch`.
This includes page-64 asymmetric heads and gfx950 tuning, global K/V bounds
checks, gfx942 BF16/FP8 LINEAR specializations, and the sink-logit correction.
The measured LINEAR defaults are BF16 128x64 and FP8 64x64.

The component patch files have been consolidated into this single patch,
following the integration branch's filename and apply/verify/remove convention.
Clean application reproduces all three GPU-qualified CK sources byte-for-byte;
reversal recovers the pinned base. Packaging does not change the implementation
or invalidate the recorded GPU measurements.

The companion `aiter/ops/mha.py` change uses the same FP8 default and appends
`_gload_bounds_v1_sink_descale_v1` to JIT module names. Native modules also
include their tile. Apply the Python companion and CK patch together so old
binaries cannot hide either correctness fix. The C++ binding comment clarifies
that sink inputs are final logits, independent of Q/K descales.

Current AITER changes and evidence are local changes on top of the WIP commit.
Carry untracked results as well as tracked diffs when transferring this work.
See [the patch README](patches/composable_kernel/README.md) for the single-patch
apply, verify and remove commands, plus performance tables. This working tree
already contains the complete patch; its reverse check passes, so no
reapplication is needed.

## Contract and controls

| Contract | Environment variable | Default | Other values |
| --- | --- | --- | --- |
| gfx942 BF16 LINEAR Q/K192–V128 | `AITER_MIMO_GFX942_LINEAR_TILE` | `128x64` | `generic`, `64x64`, `64x128`, `128x128` |
| gfx942 FP8 LINEAR Q/K192–V128 | `AITER_MIMO_GFX942_FP8_LINEAR_TILE` | `64x64` | `generic`, `64x96`, `64x128`, `128x64`, `128x96`, `128x128` |

FP8 uses `torch.float8_e4m3fnuz` on gfx942, per-tensor Q/K/V descales, BF16
output, LINEAR page-1 KV, and SGLang's 1D lookup table. Block scaling and native
gfx950 FP8 LINEAR tuning are outside this qualification. BF16/FP8 overrides are
independent. `generic` retains the corrected D256 fallback and does not disable
native page-64 support. Existing page-64/gfx950 tuning is unchanged.

SGLang's target SWA prefill gathers its page-64 cache into temporary LINEAR
K/V. Page-1 here describes the linear call; the physical KV allocator remains
page-64. No head-padding workaround is required. The previously removed
`_pad_mimo_ck_linear_kv` workaround was not restored, and this continuation did
not change SGLang. DFlash draft and target verification use separate paths.

## Additional correctness fix

The first FP8 sweep failed the corrected generic baseline at Q=KV=16384:
maximum absolute error 0.26335, concentrated on the first causal rows. Native
128x96 also failed a reduced Q=KV=129 case, with maximum error 0.16472.

CK divided the virtual sink by the base softmax scale, then multiplied it by
the effective QK scale including FP8 descales. The sink was unintentionally
rescaled toward zero. Prefix-attention cases could miss this because their
sink probability was small. The patch normalizes by the effective QK scale;
soft-capped logits and the non-fast-exp2 path use final sink logits directly.
The reduced native case now has maximum error 0.01306 and relative L2 2.18%.

`test_batch_prefill_bounds.py` now has 28 cases (seven shapes/options × two
dtypes × two load modes). New no-prefix cases exercise the sink with and
without soft capping. Guard references use distinct non-unit FP8 Q/K/V scales.
The original tolerances are unchanged: FP8 atol/rtol 0.055 and relative L2 <5%;
BF16 atol/rtol 0.02. The benchmark writes completed cases incrementally so a
later failure does not discard their evidence.

## Measurements and validation

- **84/84 FP8 benchmark cases passed:** generic plus six native tiles, both
  load modes, six workloads. Every case was checked before timing. Maximum
  absolute error was 0.01975; maximum relative L2 error was 2.52%.
- **28/28 generic compatibility cases passed**, including BF16/FP8, original
  crash shape, ragged tails, shuffled indices, generic D160/V96 and D96/V64
  heads, no-prefix sinks and soft-capped logits.
- **37/37 default compatibility cases passed:** the same 28 guarded cases plus
  nine existing LINEAR/page-64 API cases, including full causal attention,
  sinks and BF16 ragged pages. Both tile overrides were unset for this run.
- **9/9 CPU codegen tests passed.**
- Default FP8 full-causal attention with sinks at Q=129/KV=257 passed in both
  load modes (relative L2 2.59%).
- Four GPU profiler captures confirm default D192/V128 64x64 and generic D256
  dispatch, each with the intended global/buffer load enum.
- Single-patch apply/reversal and focused Black/Ruff checks passed.

For Q=32768/KV=65536, generic → selected FP8 times are **509.09 → 306.10 µs**
(global) and **473.10 → 246.66 µs** (buffer). For the ragged 32K query case,
they are **507.97 → 304.79 µs** and **479.28 → 245.70 µs** respectively.
64x64 wins all four large-workload comparisons: about 40% less kernel time
with global loads and 48–49% less with buffer loads. Some smaller queries
favor 128x128; the default prioritizes 32K prefill chunks.

Evidence lives under `patches/composable_kernel/tuning/`:

- `gfx942_fp8_linear_20261021.json`: all 84 rows, three timing samples per row,
  error metrics, environment, selected tile and patch hashes.
- `gfx942_ck_qualification_20261021.json`: all test names/results, sink
  regression, actual kernel symbols, source hashes and patch replay evidence.
- `gfx942_bf16_linear_20261021.json`: preserved earlier BF16 tuning.
- `gfx942_server_regression_20261021.json`: preserved earlier BF16 server run.

These are isolated attention timings, not serving-throughput speedups. Full
FP8 serving validation was not run. The previous host's BF16 run passed a
36,736-token cached-tail request and 600/600 16K-input requests at concurrency
16, with 13,958.39 input tokens/s and no recorded server faults. That earlier
focused crash regression is not the full long-context benchmark matrix.

## Reproduce kernel qualification

Use a compatible ROCm/PyTorch environment with this branch's dependencies,
including the `pyhip` revision and FlyDSL version specified in `setup.py`.
Confirm Python imports this checkout. This host needed `pyhip` and
`flydsl==0.3.2`; validation installed them under a temporary Python path without
changing the shared virtual environment.

```bash
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export AITER_JIT_DIR=$(mktemp -d /tmp/aiter-ck-mimo-jit.XXXXXX)
export MIMO_RESULTS_DIR=$(mktemp -d /tmp/aiter-ck-mimo-results.XXXXXX)
unset AITER_MIMO_GFX950_D192_TILE AITER_MIMO_GFX950_D192_OCCUPANCY
unset AITER_MIMO_GFX942_LINEAR_TILE AITER_MIMO_GFX942_FP8_LINEAR_TILE
unset HIP_LAUNCH_BLOCKING
MIMO_ROCM_DIR=${ROCM_PATH:-/opt/rocm}
g++ -shared -fPIC -O2 -D__HIP_PLATFORM_AMD__ -I"$MIMO_ROCM_DIR/include" \
  op_tests/ck_prefill_guard.cpp -L"$MIMO_ROCM_DIR/lib" \
  -Wl,-rpath,"$MIMO_ROCM_DIR/lib" -lamdhip64 -o "$MIMO_RESULTS_DIR/guard.so"
python3 -m pytest -q op_tests/test_mimo_fp8_linear_codegen.py
python3 -m pytest -x -q op_tests/test_batch_prefill_bounds.py \
  op_tests/test_batch_prefill_asymmetric.py
AITER_MIMO_GFX942_LINEAR_TILE=generic AITER_MIMO_GFX942_FP8_LINEAR_TILE=generic \
  python3 -m pytest -x -q op_tests/test_batch_prefill_bounds.py
```

Keep the private JIT directory across runs; tile module names are distinct.
The guard allocator selects low address bands `0x90000000` and `0x10000000`
to force global and buffer loads, with an unmapped guard immediately after
logical K/V. Children synchronize and exit to release mappings. Initial
compilation must finish within each guard child's 300-second timeout; prebuild
signatures if needed on a slower machine.

For the timing sweep, use the loop in the patch README with `--cache-dtype fp8`
and both load modes. Never use `HIP_LAUNCH_BLOCKING=1` for timings. A complete
sweep contains 14 JSON files with six rows each. Stop on accuracy failure or
GPU fault and investigate before further submissions.

## Local artifacts and integration follow-up

This continuation used `/tmp/mimo-ck-20261021/` for its private JIT cache,
`guard.so`, isolated Python dependencies and raw logs/results. Durable summaries
are copied into the tuning directory. GPU 0 ran serial timing cases; a separate
GPU (ROCR device 1) ran compatibility checks. GPU telemetry and a device
operation were checked before qualification. No server was launched for this
kernel task.

The old source host was `quanta-ccs-aus-d05-11`; its later failed GPU reset did
not prevent validation on this healthy host. Its historical server logs were
under `sglang/evaluation/logs/dflash_ck_root_fix_20261021/`; the retained JSON
contains the relevant metrics. Do not reuse old host-specific JIT binaries or
the former VRAM-only readiness watcher.

An optional FP8 serving check needs the separate SGLang checkout, MiMo model
weights, appropriate FP8 cache configuration and eight idle GPUs. Confirm the
actual attention path/cache dtype in logs and stop the validation server and
workers afterward. Kernel qualification itself needs no model or server.
