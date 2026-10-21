# MiMo Composable Kernel patches

These patches are based on CK commit
`af9e1d1f1ae347c22feeb08fd2d42645075e0c5d`. Apply them in the order below.

1. `mimo_page64_qk192_v128_batch_prefill.patch`: existing native Q/K192,
   V/output128 page-64 vectorized kernels for gfx942/gfx950 (BF16 and FP8).
2. `batch_prefill_global_load_bounds.patch`: restore padding validity checks
   for both global-to-register V loads and asynchronous global-to-LDS K loads.
   Invalid K lanes receive zeros through a zero-length SRD transfer, preserving
   the LDS layout and async-load ordering. This also protects generic D256
   fallback kernels; the fix is not limited to MiMo's native specialization.
3. `mimo_gfx942_bf16_linear_prefill.patch`: native BF16 Q/K192-V128 LINEAR
   page-1 SGLang-table kernels on gfx942. This extends the existing codegen
   patch; it keeps other layouts, dtypes, and head dimensions on their existing
   dispatch paths. The selected tile is 128x64 with four waves.

The patch was adapted from `sammysun0711/aiter:mimo-opt`, commit
`e89c25b8b461df14005dbf4e8ab5435597932cd8`.


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

## Apply

From the AITER root, starting with the clean pinned CK submodule:

```bash
git submodule update --init 3rdparty/composable_kernel
for patch in \
  mimo_page64_qk192_v128_batch_prefill.patch \
  batch_prefill_global_load_bounds.patch \
  mimo_gfx942_bf16_linear_prefill.patch; do
  git -C 3rdparty/composable_kernel apply --check < "patches/composable_kernel/$patch"
  git -C 3rdparty/composable_kernel apply < "patches/composable_kernel/$patch"
done
```

If the page-64 patch is already applied, apply only patches 2 and 3. The native
linear patch depends on patch 1. To remove patches, reverse this order.

The companion change in `aiter/ops/mha.py` gives generated modules the suffix
`_gload_bounds_v1`, and the native linear module also includes its tile name.
Old JIT shared objects therefore cannot hide a newly applied bounds fix or a
different tile. Use the source patch and this Python companion together.

## Correctness checks

```bash
python3 -m pytest -q op_tests/test_batch_prefill_bounds.py
python3 -m pytest -q \
  'op_tests/test_batch_prefill_asymmetric.py::test_batch_prefill_qk192_v128[bf16-linear-True]' \
  'op_tests/test_batch_prefill_asymmetric.py::test_batch_prefill_qk192_v128[bf16-vectorized-True]' \
  op_tests/test_batch_prefill_asymmetric.py::test_batch_prefill_qk192_v128_bf16_vectorized_ragged_pages
```

The bounds tests reserve unmapped guard pages immediately after K/V and force
addresses that select `GLOBAL_LOAD_LDS`. They check the original failing shape,
ragged tails, shuffled token indices, and other padded head widths against a
sink-aware PyTorch reference. Each runs in a subprocess so GPU faults are
reported as test failures and all HIP virtual mappings are released on exit.
The helper needs ROCm headers/libraries and `g++`.

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
100-ms CUDA-graph measurements. Guarded high addresses keep comparisons on the
global-load path. `--padding-baseline` additionally measures Python K/V padding
and copies with the selected kernel, if needed.

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

## FP8 LINEAR extension — GPU qualification pending

`mimo_gfx942_fp8_linear_prefill.patch` extends the three patches above with
native gfx942 FP8 Q/K192–V128 LINEAR page-1 kernels. The contract is per-tensor
Q/K/V descales, FP8 E4M3FNUZ inputs on gfx942, and BF16 output. BF16 and page-64
tuning retain their separate settings.

`AITER_MIMO_GFX942_FP8_LINEAR_TILE` selects `64x64`, `64x96`, `64x128`,
`128x64`, `128x96`, or `128x128`; `generic` selects the corrected D256 baseline.
128x96 is the initial candidate, pending GPU measurements. The JIT module name
includes this FP8 tile choice, independently of the BF16 tile setting.

All twelve native variants (six tiles, each with buffer and global loads)
compile on gfx942, and nine CPU codegen checks pass. GPU accuracy and timing
are pending while another workload holds the devices.

The benchmark accepts `--cache-dtype fp8` and `--load-mode global` or `buffer`.
The guard helper supports both address bands, byte-accurate FP8 allocations,
non-unit Q/K/V scales, ragged requests, and shuffled token indices. Accuracy is
checked against dequantized FP8 inputs, with both elementwise tolerances and
a relative L2 error limit of 5%.
