# MiMo page-64 asymmetric batch-prefill patch

This directory contains the temporary Composable Kernel change required by
the MiMo V2.5/V2.6 AITER paged-prefill fallback while AITER remains pinned to
a CK revision that does not generate this specialization.

The patch is based on CK commit:

```text
af9e1d1f1ae347c22feeb08fd2d42645075e0c5d
```

It adds exact page-64, vectorized SGLang batch-prefill dispatch for
Q/K head dimension 192 with V/output dimension 128. BF16 and FP8 cache
variants are generated for gfx942 and gfx950. Unsupported D192 cache contracts
continue to fall through to the padded-D256 implementation.

The patch was adapted from `sammysun0711/aiter:mimo-opt`, commit
`e89c25b8b461df14005dbf4e8ab5435597932cd8`.

## Apply

From the AITER repository root:

```bash
git submodule update --init 3rdparty/composable_kernel
git -C 3rdparty/composable_kernel apply --check \
  < patches/composable_kernel/mimo_page64_qk192_v128_batch_prefill.patch
git -C 3rdparty/composable_kernel apply \
  < patches/composable_kernel/mimo_page64_qk192_v128_batch_prefill.patch
```

The companion AITER change in `aiter/ops/mha.py` gives this kernel family a
distinct JIT module name ending in `_page64_qk192_v128`. This prevents a stale
generic batch-prefill shared object from masking the newly applied CK code.

## Verify

```bash
git -C 3rdparty/composable_kernel apply --reverse --check \
  < patches/composable_kernel/mimo_page64_qk192_v128_batch_prefill.patch

HIP_VISIBLE_DEVICES=0 python3 -m pytest -q \
  'op_tests/test_batch_prefill_asymmetric.py::test_batch_prefill_qk192_v128[bf16-vectorized-False]' \
  op_tests/test_batch_prefill_asymmetric.py::test_batch_prefill_qk192_v128_bf16_vectorized_ragged_pages
```

The second test covers multiple requests, non-contiguous page IDs, ragged KV
lengths, and partial last pages, matching the metadata properties exercised by
TBO child batches.

## Remove

```bash
git -C 3rdparty/composable_kernel apply --reverse \
  < patches/composable_kernel/mimo_page64_qk192_v128_batch_prefill.patch
```
