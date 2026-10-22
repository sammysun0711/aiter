# MiMo Composable Kernel batch-prefill patch

This directory contains one combined patch:
`mimo_page64_qk192_v128_batch_prefill.patch`, based on CK commit
`af9e1d1f1ae347c22feeb08fd2d42645075e0c5d`.

The patch retains the existing gfx942/gfx950 Q/K192-V/output128 page-64
vectorized kernels and adds:

- Native gfx942 LINEAR page-1 kernels with fixed BF16 128x64 and per-tensor
  FP8 64x64 tiles. Unsupported contracts retain the generic fallback.
- K/V padding bounds checks for global loads, including zero filling invalid
  asynchronous K transfers. The checks also protect generic fallback kernels.
- Virtual-sink normalization using the effective QK scale, including FP8
  descales. Already-transformed logits use the final sink directly.

The original page-64 specialization was adapted from
`sammysun0711/aiter:mimo-opt`, commit
`e89c25b8b461df14005dbf4e8ab5435597932cd8`.

## Why the fixes are needed

The MiMo LINEAR fallback represents gathered K/V tokens as page-1 entries.
Previously, Q/K192-V128 calls could use a padded D256 kernel whose global-load
branches bypassed padding checks and read beyond the last K or V row. The
bounds checks protect both native and generic kernels without Python head
padding. Invalid asynchronous transfers must zero LDS rather than leave data
from a previous tile.

FP8 Q/K descales also change the accumulator scale. Normalizing virtual-sink
logits with only the base softmax scale incorrectly rescales the sink; the
patch uses the effective scale and handles soft-capped logits consistently.

## Apply

From the AITER root, start with a clean pinned CK submodule:

```bash
git submodule update --init 3rdparty/composable_kernel
git -C 3rdparty/composable_kernel apply --check \
  < patches/composable_kernel/mimo_page64_qk192_v128_batch_prefill.patch
git -C 3rdparty/composable_kernel apply \
  < patches/composable_kernel/mimo_page64_qk192_v128_batch_prefill.patch
```

## Verify

Confirm that the complete patch is applied:

```bash
git -C 3rdparty/composable_kernel apply --reverse --check \
  < patches/composable_kernel/mimo_page64_qk192_v128_batch_prefill.patch
```

On an idle GPU, use the fresh JIT directory above and run the existing
asymmetric BF16/FP8 API tests:

```bash
HIP_VISIBLE_DEVICES=0 python3 -m pytest -q \
  op_tests/test_batch_prefill_asymmetric.py
```


## Remove
```bash
git -C 3rdparty/composable_kernel apply --reverse \
  < patches/composable_kernel/mimo_page64_qk192_v128_batch_prefill.patch
```
