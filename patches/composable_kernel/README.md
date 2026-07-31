# Temporary Composable Kernel patch

This directory carries the temporary CK portion of the MiMo-V2.5-Pro cached
prefill optimization while the corresponding `ROCm/rocm-libraries` change is
not yet available through AITER's pinned `ROCm/composable_kernel` submodule.

The patch targets the CK revision currently pinned by this AITER fork:

```text
af7118e342580ecd3f71edce7b1d0ba465012ecf
```

It adds the guarded FP8/BF16, head-192, vectorized page-64 batch-prefill
specialization used by MiMo. The patch should be removed after AITER is moved
to the official exported CK commit containing the same change.

## Apply

Run these commands from the AITER repository root:

```bash
git submodule update --init 3rdparty/composable_kernel
git -C 3rdparty/composable_kernel apply --check \
  < patches/composable_kernel/mimo_fp8_page64_head192_batch_prefill.patch
git -C 3rdparty/composable_kernel apply \
  < patches/composable_kernel/mimo_fp8_page64_head192_batch_prefill.patch
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
  < patches/composable_kernel/mimo_fp8_page64_head192_batch_prefill.patch
```

Remove the temporary patch from the submodule working tree:

```bash
git -C 3rdparty/composable_kernel apply --reverse \
  < patches/composable_kernel/mimo_fp8_page64_head192_batch_prefill.patch
```

Run the focused AITER regression after rebuilding the generated CK extension:

```bash
python -m pytest -q op_tests/test_batch_prefill.py \
  -k mimo_fp8_vectorized_page64
```
