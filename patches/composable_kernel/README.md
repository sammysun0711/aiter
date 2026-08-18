# Temporary Composable Kernel patch

This directory carries the temporary CK portion of the MiMo-V2.5-Pro cached
prefill optimization while the corresponding `ROCm/rocm-libraries` change is
not yet available through AITER's pinned `ROCm/composable_kernel` submodule.

The patch targets the CK revision currently pinned by this AITER fork:

```text
af7118e342580ecd3f71edce7b1d0ba465012ecf
```

It adds the guarded FP8/BF16, head-192, vectorized page-64 batch-prefill
specialization used by MiMo on both supported CDNA targets:

- gfx950 selects native-V128 `(128, 128)` tiles for BF16 and FP8;
- gfx942 emits native-V128 BF16 `(128, 128)` and FP8 `(128, 96)` tiles;
- the retained FP8 padded-V192 route uses `(64, 128)` on gfx950 and
  `(192, 96)` on gfx942;
- unsupported D192 cache contracts fall through to the padded-D256 path, with
  page-64 BF16/FP8 D256 comparator/rollback families generated on both targets;
- opt-in gfx950 tile and occupancy overrides remain available for isolated
  tuning builds.

The D192 API dispatch is exact on Q/K192, V128 or V192, vectorized SGLang
layout, and page 64. Q/K and V tile widths are separate dispatch axes, so a
native V128 tile cannot accept a padded V192 request. Both buffer-load and
large-cache global-load variants are emitted.

On one MI350X/gfx950, the final native `(128, 128)` tile passed both production
chunk-prefill gates, Q=16,384 and Q=65,536. It is 1.97-2.03x faster than D256
for BF16, 2.06-2.13x faster than D256 for FP8, and 1.53-1.79x faster than the
retained FP8 padded-V192 tile across the recorded context matrix.

The target-specific selection requires codegen to receive one explicit GPU
target. AITER's batch-prefill JIT recipe resolves `get_gfx()` and passes the
result as `--targets <gfx>` for this reason;

## Apply

Run these commands from the AITER repository root:

```bash
git submodule update --init 3rdparty/composable_kernel
git -C 3rdparty/composable_kernel apply --check \
  < patches/composable_kernel/mimo_page64_qk192_v128_batch_prefill.patch
git -C 3rdparty/composable_kernel apply \
  < patches/composable_kernel/mimo_page64_qk192_v128_batch_prefill.patch
```

If `git apply --check` reports that the patch does not apply, first inspect the
submodule state. Do not blindly apply it to a different CK revision:

```bash
git -C 3rdparty/composable_kernel rev-parse HEAD
git -C 3rdparty/composable_kernel status --short
```

## Verify or remove

Check that the patch is already present:

```bash
git -C 3rdparty/composable_kernel apply --reverse --check \
  < patches/composable_kernel/mimo_page64_qk192_v128_batch_prefill.patch
```

Remove the temporary patch from the submodule working tree:

```bash
git -C 3rdparty/composable_kernel apply --reverse \
  < patches/composable_kernel/mimo_page64_qk192_v128_batch_prefill.patch
```

Run the focused AITER regression after rebuilding the generated CK extension:

```bash
HIP_VISIBLE_DEVICES=0 python -m pytest -q \
  op_tests/test_batch_prefill_asymmetric.py
```

The focused suite contains eight PyTorch-reference cases covering BF16/FP8,
LINEAR/vectorized page-64 cache layouts, full attention, and SWA with learned
sinks.
