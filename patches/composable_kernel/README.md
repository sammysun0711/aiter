# MiMo Composable Kernel batch-prefill patch

This directory contains one CK patch:
`mimo_page64_qk192_v128_batch_prefill.patch`, based on CK commit
`af9e1d1f1ae347c22feeb08fd2d42645075e0c5d`.

It includes all MiMo batch-prefill changes:

- Native Q/K192–V/output128 page-64 vectorized kernels for gfx942/gfx950
  (BF16 and FP8), including the existing gfx950 tuning.
- Native gfx942 LINEAR page-1 SGLang-table kernels, with measured defaults
  BF16 128x64 and per-tensor FP8 64x64.
- Global K/V padding bounds checks, including zero filling invalid
  asynchronous K transfers. These also protect generic fallback kernels.
- Virtual-sink normalization using the effective QK scale, including FP8
  descales; transformed logits use the final sink directly.

The combined patch changes three CK files and reproduces the GPU-qualified
sources byte-for-byte. It replaces the component patch series.

The original page-64 specialization was adapted from
`sammysun0711/aiter:mimo-opt`, commit `e89c25b8b461df14005dbf4e8ab5435597932cd8`.

## Why the linear patch is needed

SGLang's gfx942 MiMo SWA prefill fallback gathers page-64 KV into temporary
linear tensors. AITER represents each token in those tensors as a page-1 entry.
The physical KV allocator still uses page size 64. This path runs in target
prefill; DFlash's draft and target-verification SWA kernels are separate.

Previously, Q/K192-V128 LINEAR calls fell through to a D256 tile. CK chooses
`GLOBAL_LOAD_LDS` partly from the low bits of the K/V virtual addresses, even
for small buffers. Its global-load branches bypassed padding validity checks,
so the D256 kernel could read past the final 192-wide K or 128-wide V row.
The guard-page test reproduces the original 3968-query / 16384-KV crash without
loading the model. Both the generic bounds fix and native linear kernel pass
with unpadded K and V. No SGLang head-padding workaround is needed.

A direct page-64 SWA integration could avoid gathering altogether, but that is
separate from this fix and the linear-kernel tuning below.

## FP8 sink correctness

Runtime qualification also exposed a sink-logit error in the generic baseline
and native FP8 kernels. CK divided the virtual sink by the base softmax scale,
but subsequently multiplied it by the effective scale, including Q/K descales.
This effectively moved nonzero sink logits toward zero. Prefix-attention
checks could miss it because the sink carried little probability; Q=KV cases
failed on the first causal rows, where there are few real keys.

The patch normalizes the sink using the same effective scale as the QK
accumulator. With soft-capped logits (or without fast exp2), it passes the final
sink directly, matching the pipeline's transformed score units. The new
no-prefix tests use distinct non-unit Q/K/V scales and positive and negative
sink logits. They exercise both load modes and retain the original accuracy
limits: FP8 atol/rtol 0.055 and relative L2 below 0.05; BF16 atol/rtol 0.02.

## Apply

From the AITER root, starting with the clean pinned CK submodule:

```bash
git submodule update --init 3rdparty/composable_kernel
git -C 3rdparty/composable_kernel apply --check \
  < patches/composable_kernel/mimo_page64_qk192_v128_batch_prefill.patch
git -C 3rdparty/composable_kernel apply \
  < patches/composable_kernel/mimo_page64_qk192_v128_batch_prefill.patch
```

An already-patched checkout can use the reverse check below to confirm that
the complete patch is present. The qualified working tree already contains
these changes and needs no reapplication.

The companion change in `aiter/ops/mha.py` gives generated modules the suffix
`_gload_bounds_v1_sink_descale_v1`, and native linear modules include their tile names.
Old JIT shared objects therefore cannot hide a newly applied bounds fix or a
different tile. Use the source patch and this Python companion together.

## Verify

```bash
git -C 3rdparty/composable_kernel apply --reverse --check \
  < patches/composable_kernel/mimo_page64_qk192_v128_batch_prefill.patch

python3 -m pytest -q op_tests/test_mimo_fp8_linear_codegen.py
python3 -m pytest -q op_tests/test_batch_prefill_bounds.py \
  op_tests/test_batch_prefill_asymmetric.py
AITER_MIMO_GFX942_LINEAR_TILE=generic AITER_MIMO_GFX942_FP8_LINEAR_TILE=generic \
  python3 -m pytest -q op_tests/test_batch_prefill_bounds.py
```

The bounds tests reserve unmapped guard pages immediately after K/V and force
addresses that select either `GLOBAL_LOAD_LDS` or `BUFFER_LOAD`. The 28 cases
cover BF16/FP8, the original failing shape, ragged tails, shuffled token indices,
and other padded head widths against a sink-aware PyTorch reference. No-prefix
causal cases test sink normalization with and without logits soft capping.
Each runs in a subprocess so GPU faults are
reported as test failures and all HIP virtual mappings are released on exit.
The helper needs ROCm headers/libraries and `g++`.

Qualification on `quanta-ccs-aus-d05-03` passed **37/37** default GPU tests
(28 guard cases plus nine existing API cases), **28/28** generic GPU tests and
**9/9** CPU codegen tests. Full-causal FP8 with sinks passed in both load modes.
Four profiler captures verify default D192/V128 64x64 and generic D256 dispatch
with the requested global/buffer load enums. Test results, kernel symbols,
source hashes and successful single-patch application/reversal checks are
recorded in `tuning/gfx942_ck_qualification_20261021.json`.

## Remove

```bash
git -C 3rdparty/composable_kernel apply --reverse \
  < patches/composable_kernel/mimo_page64_qk192_v128_batch_prefill.patch
```

## Tuning

`AITER_MIMO_GFX942_LINEAR_TILE` accepts `64x64`, `64x128`, `128x64`, and
`128x128`. The default is `128x64`. `generic` disables the native LINEAR
specialization while retaining the bounds-corrected D256 fallback.
The override affects only gfx942 BF16 Q/K192-V128 LINEAR calls; existing
page-64 and gfx950 tuning is unchanged.

Use an idle GPU and a private JIT directory for candidate builds:

```bash
mkdir -p /tmp/ck-linear-results
g++ -shared -fPIC -O2 -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include \
  op_tests/ck_prefill_guard.cpp -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib \
  -lamdhip64 -o /tmp/ck-linear-results/guard.so
for tile in generic 64x64 64x128 128x64 128x128; do
  AITER_JIT_DIR=/tmp/aiter-ck-linear-tuning AITER_MIMO_GFX942_LINEAR_TILE="$tile" \
    python3 op_tests/benchmark_mimo_ck_linear_prefill.py \
      --library /tmp/ck-linear-results/guard.so \
      --output "/tmp/ck-linear-results/$tile.json"
done
```

The benchmark validates every shape first, then reports the median of three
100-ms CUDA-graph measurements. `--load-mode global` uses guarded high addresses;
`--load-mode buffer` uses guarded low addresses. `--padding-baseline` additionally
measures Python K/V padding and copies with the selected BF16 kernel, if needed.

Results on MI325X / gfx942 (304 CUs), ROCm 7.2, BF16 SWA window 128, Q heads 16,
KV heads 1, in microseconds:

| Query / KV lengths | Generic D256 | 64x64 | 64x128 | 128x64 | 128x128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 129 / 257 | 50.10 | 42.38 | 51.69 | 41.42 | 41.08 |
| 512 / 16384 | 56.31 | 49.87 | 62.84 | 47.26 | 46.23 |
| 3968 / 16384 | 130.95 | 117.10 | 146.73 | 76.92 | 75.44 |
| 16384 / 16384 | 386.81 | 350.79 | 462.21 | 250.86 | 250.28 |
| 32768 / 65536 | 769.37 | 720.59 | 985.88 | 485.84 | 500.87 |
| Ragged 32768 / 45184 total | 778.05 | 702.00 | 964.25 | 482.64 | 494.06 |

128x64 is selected for the 32K chunk workload: about 37–38% less kernel time
than the corrected generic path on the large cases. 128x128 is slightly faster
for small queries and remains available as an override. These are isolated
attention timings, not end-to-end serving speedups. Raw samples and environment
information are in `tuning/gfx942_bf16_linear_20261021.json`.

## Server regression

With the SGLang padding workaround removed, the original TP8 gfx942 launch
settings passed a 36,736-token cached-tail request and **600/600** measured
16K-input, one-output-token requests at concurrency 16. Normal asynchronous
execution was used. There were no GPU faults or scheduler exceptions; the
former failure point near request 470 was passed. The measured run processed
9,830,400 input tokens in 704.26 seconds (13,958.39 input tokens/s).

The test used the original server seed 782693892 and four warmup requests.
Metrics and the local log path are recorded in
`tuning/gfx942_server_regression_20261021.json`. This is the focused crash
regression, not the complete long-context benchmark matrix.

## FP8 LINEAR tuning

The combined patch includes native gfx942 FP8 Q/K192–V128 LINEAR page-1
kernels. The contract is per-tensor Q/K/V descales, FP8 E4M3FNUZ inputs on
gfx942, and BF16 output. BF16 and page-64
tuning retain their separate settings.

`AITER_MIMO_GFX942_FP8_LINEAR_TILE` selects `64x64`, `64x96`, `64x128`,
`128x64`, `128x96`, or `128x128`; `generic` selects the corrected D256 baseline.
The measured default is **64x64**, with two waves. The JIT module name includes
this FP8 tile choice, independently of the BF16 tile setting.

All twelve native variants (six tiles, each with buffer and global loads)
passed runtime JIT and correctness checks on MI325X / gfx942, host
`quanta-ccs-aus-d05-03`, PyTorch 2.9.1 / ROCm 7.2. Each tile and the corrected
generic baseline passed all six benchmark workloads in both load modes:
**84/84 cases**, with maximum absolute error 0.01975 and maximum relative L2
error 2.52%. These measurements include the sink correction.

Large-workload times in microseconds, median of three 100-ms graph runs:

| FP8 tile | Global 32K / 64K | Global ragged 32K | Buffer 32K / 64K | Buffer ragged 32K |
| --- | ---: | ---: | ---: | ---: |
| generic | 509.09 | 507.97 | 473.10 | 479.28 |
| **64x64** | **306.10** | **304.79** | **246.66** | **245.70** |
| 64x96 | 385.08 | 382.38 | 287.58 | 285.41 |
| 64x128 | 373.69 | 372.01 | 295.57 | 293.22 |
| 128x64 | 343.28 | 341.69 | 312.28 | 309.41 |
| 128x96 | 377.98 | 375.53 | 332.22 | 327.09 |
| 128x128 | 336.47 | 334.20 | 301.02 | 293.34 |

The ragged case uses Q `[3968,16384,12416]` and KV `[16384,16384,12416]`.
64x64 minimizes time for all four large-workload comparisons: approximately
40% less kernel time than generic with global loads and 48–49% less with buffer
loads. It is not fastest on every small workload: for Q=512/KV=16384 with global
loads, 64x64 takes 37.96 µs versus generic 34.18 µs and 128x128 32.08 µs. The
default prioritizes MiMo's 32K prefill chunks; other tiles remain available.

All per-case samples, accuracy metrics, patch hashes and environment details
are in `tuning/gfx942_fp8_linear_20261021.json`. These are isolated kernel
measurements. Full FP8 serving validation has not been run; the earlier BF16
server regression above remains separate evidence.

The benchmark accepts `--cache-dtype fp8` and `--load-mode global` or `buffer`.
The guard helper supports both address bands, byte-accurate FP8 allocations,
non-unit Q/K/V scales, ragged requests, and shuffled token indices. Accuracy is
checked against dequantized FP8 inputs, with both elementwise tolerances and
a relative L2 error limit of 5%.

To reproduce the full FP8 sweep after building `guard.so` as above:

```bash
for mode in global buffer; do
  for tile in generic 64x64 64x96 64x128 128x64 128x96 128x128; do
    AITER_JIT_DIR=/tmp/aiter-ck-linear-tuning \
    AITER_MIMO_GFX942_FP8_LINEAR_TILE="$tile" \
      python3 op_tests/benchmark_mimo_ck_linear_prefill.py \
        --library /tmp/ck-linear-results/guard.so --cache-dtype fp8 \
        --load-mode "$mode" --window 128 \
        --output "/tmp/ck-linear-results/fp8-$tile-$mode.json"
  done
done
```

Use normal asynchronous execution for timings (`HIP_LAUNCH_BLOCKING` unset).
The benchmark preserves completed cases in its JSON output if a later case
fails. Do not treat a partial file as a completed sweep.
