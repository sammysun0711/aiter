# MI300X FP8 GEMM tuning guide

This guide covers single-GPU FP8 linear projections on a **304-CU MI300X (gfx942)**. For the customer’s reference shape `[M, 32768] @ [4096, 32768]` with `M = 8192`, tuned hipBLASLt with per-token/per-channel scales reached **1,154.8 TFLOP/s** and packed ASM with K128 block scales reached **899.9 TFLOP/s** in the same replay. A separate eight-round comparison measured **1,173.2** and **911.0** respectively. [All kernels by shape](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/mi300x-fp8-ptpc-vs-blockscale.md) · [Unrounded CSV](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/mi300x-fp8-ptpc-vs-blockscale.csv)

## 1. API and layouts

Both examples compute `Y[M, N] = X[M, K] @ W[N, K].T`. They start with prequantized `X_fp8` and ordinary row-major `W_fp8` tensors of dtype `torch.float8_e4m3fnuz` on the GPU. All scales are FP32 dequantization multipliers on the same device. The output is BF16 with no bias.

### Per-token/per-channel FP8 (PTPC) GEMM API and layouts

PTPC uses one activation scale per token and one weight scale per output channel. Output element `Y[m, n]` is the full dot product multiplied by `token_scales[m] * channel_scales[n]`. The tensor × channel control in the full report repeats one activation multiplier across all rows.

| API                                 | Weight argument                                      | Activation scale | Weight scale |
| ----------------------------------- | ---------------------------------------------------- | ---------------- | ------------ |
| `torch._scaled_mm`                | `W_fp8.T`, shape `[K, N]`                        | `[M, 1]`       | `[1, N]`   |
| `tgemm.mm`                        | Ordinary or (16, 16)-packed weight, shape `[N, K]` | `[M, 1]`       | `[1, N]`   |
| `aiter.gemm_a8w8_bpreshuffle_out` | (16, 16)-packed weight, shape `[N, K]`             | `[M, 1]`       | `[N, 1]`   |

The example uses packed `tgemm.mm` with the `hipblaslt` table from section 3. [PTPC dispatch](../aiter/tuned_gemm.py)

```python
import torch
from aiter.ops.shuffle import shuffle_weight
from aiter.tuned_gemm import tgemm

# Once at weight load. Keep the returned tensor and its layout metadata.
W_packed = shuffle_weight(W_fp8, layout=(16, 16))

# Supply one scale per activation row and one per weight row.
sx = token_scales.reshape(-1, 1).float().contiguous()
sw = channel_scales.reshape(1, -1).float().contiguous()
Y_ptpc = tgemm.mm(X_fp8, W_packed, otype=torch.bfloat16, scale_a=sx, scale_b=sw)
```

### FP8 block GEMM API and layout

Block scaling uses one activation scale per row and group of 128 K features, and one weight scale per 128×128 tile. For each K group `g`, the kernel multiplies the partial dot product by `activation_block_scales[m, g] * weight_block_scales[n // 128, g]` before summing across groups.

| API                                        | Weight argument            | Activation scale                                      | Weight scale                 |
| ------------------------------------------ | -------------------------- | ----------------------------------------------------- | ---------------------------- |
| `aiter.gemm_a8w8_blockscale`             | Ordinary `[N, K]`        | Row-major `[M, K/128]`                              | Row-major `[N/128, K/128]` |
| `aiter.gemm_a8w8_blockscale_bpreshuffle` | (16, 16)-packed `[N, K]` | K-group-major storage, passed below as `[M, K/128]` | Row-major `[N/128, K/128]` |

The example starts with ordinary row-major FP32 scale tensors. Transposing and copying the activation scales places all `M` scales for each K group next to each other. The final view supplies the shape expected by the packed call. Use this full-shape example for a direct tuning row. Section 4 shows how to pack each activation-scale slice for a chunked row plan. [Block-scale dispatch](../aiter/ops/gemm_op_a8w8.py) · [Scale packing in the benchmark](../op_tests/op_benchmarks/mi300x_fp8_gemm/common.py)

```python
import torch
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle
from aiter.ops.shuffle import shuffle_weight

# Once at weight load.
W_packed = shuffle_weight(W_fp8, layout=(16, 16))

# Pack the logical [M, K/128] activation scales for this call.
sx_block = activation_block_scales.T.contiguous().view_as(activation_block_scales)
Y_block = gemm_a8w8_blockscale_bpreshuffle(
    X_fp8, W_packed, sx_block, weight_block_scales, torch.bfloat16
)
```

#### FlyDSL kernel selection and tensor layouts on gfx942

FlyDSL is a `libtype` for the FP32 K128 block-scale APIs above. Choose the API by the weight layout. The packed direct-load, M-first LDS, N-first LDS and register-cache kernels all take **the same caller tensors**. Grid order controls how workgroups visit output tiles. The register kernel transposes its MFMA computation internally and writes the usual `[M, N]` output.

| Weight layout | Public API and tuning-table environment variable | FlyDSL candidates | Activation scales passed by the caller |
|---|---|---|---|
| Ordinary row-major `[N, K]` | `aiter.gemm_a8w8_blockscale` · `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` | IDs **2, 4, 6, 8, 10** (`ps0`), direct loads | Ordinary row-major `[M, K/128]`. The FlyDSL adapter transposes these scales internally. |
| `shuffle_weight(W_fp8, layout=(16, 16))`, still `[N, K]` | `aiter.gemm_a8w8_blockscale_bpreshuffle` · `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE` | Direct loads **3, 5, 7, 9, 11** (`ps1`), M-first LDS **12**, N-first LDS **13**, register cache **14** | K-group-major bytes, e.g. `scales.T.contiguous().view_as(scales)` starting from logical row-major `[M, K/128]`. |

Both APIs take ordinary FP8 activations `[M, K]` and row-major FP32 weight scales `[ceil(N/128), K/128]`. For the customer shapes, `ceil(N/128) = N/128`. Set the ordinary table to `$CFG/flydsl-ordinary.csv` or the packed table to `$CFG/flydsl-final.csv` before importing AITER. The runtime row’s `kernelName` selects the implementation for that shape. A quantizer that already returns K-group-major activation-scale bytes can pass those bytes directly in the packed layout shown above. [FlyDSL layout adapter](../aiter/ops/flydsl/gemm_a8w8_blockscale.py) · [Candidate IDs](../aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_blockscale_common.py)

For the customer’s `[M, 32768] @ [4096, 32768]` at `M = 8192`, the logical activation scales are `[8192, 256]`. The packed call stores them in `[256, 8192]` order. The weight scales are `[32, 256]` and the output is `[8192, 4096]` for every eligible FlyDSL candidate.

For a direct check of one FlyDSL candidate, the named launcher takes a caller-owned output buffer and an explicit layout flag. This example reuses the packed block-scale inputs above and selects ID 13. The ordinary equivalent takes `W_fp8`, row-major `activation_block_scales`, a matching `ps0` kernel name and `isBpreshuffled=False`.

```python
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_flydsl

out = torch.empty((X_fp8.shape[0], W_packed.shape[0]), dtype=torch.bfloat16, device=X_fp8.device)
gemm_a8w8_blockscale_flydsl(
    X_fp8, W_packed, sx_block, weight_block_scales, out,
    {"kernelName": "flydsl_blockscale_gfx942_128x128x128_ps1_lds_n"},
    isBpreshuffled=True,
)
```

## 2. Shapes, source and setup

The customer’s first and third shape labels show the right-hand operand in stored weight order `[N, K]`. The second shows it in multiplication order `[K, N]`, with `K = 1536` and `N = 32768`. AITER computes `X @ W.T` for all three.

| Customer shape                 | AITER logical weight `W[N, K]` | Output `Y[M, N]` |
| ------------------------------ | -------------------------------- | ------------------ |
| `[M, 4096] @ [1536, 4096]`   | `[1536, 4096]`                 | `[M, 1536]`      |
| `[M, 1536] @ [1536, 32768]`  | `[32768, 1536]`                | `[M, 32768]`     |
| `[M, 32768] @ [4096, 32768]` | `[4096, 32768]`                | `[M, 4096]`      |

The seven M values are **2,048, 4,096, 8,192, 16,384, 32,768, 65,536 and 131,072**. The recorded run used ROCm 7.2.0, PyTorch `2.9.1+rocm7.2.0.git7e1940d4`, hipBLASLt `1.2.1.70200-43~22.04` and FlyDSL **0.5.0**. hipBLASLt solution IDs depend on the installed library version. [Run environment](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/environment.json)

The source is this AITER checkout, branch `tuning/gfx942-flydsl-fp8-blockscale`, based on upstream commit `1dd04e9cf13282352a4e11dd2a2114ccae475f4a` with Composable Kernel `f93c029e09f67c983696eecac8e0e0350f3dbb54`. The original hipBLASLt and CK baselines used the clean base. The recorded cross-scaling study used the local gfx942 port, with the LDS and transposed register kernels launched by the study harness. This branch includes those final three candidates in the native FlyDSL tuner and named dispatch. [Base revision](https://github.com/ROCm/aiter/tree/1dd04e9cf13282352a4e11dd2a2114ccae475f4a) · [FlyDSL registry](../aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_blockscale_common.py) · [Launch adapter](../aiter/ops/flydsl/gemm_a8w8_blockscale.py)

Start at the checkout root in a ROCm environment with this checkout’s AITER dependencies and the pinned FlyDSL installed. The following commands put the checkout first on the Python import path and create a fresh output directory. All input CSVs ship with the benchmark. [AITER installation](installation.rst)

```bash
export AITER_SRC="$(git rev-parse --show-toplevel)"
export BENCH="$AITER_SRC/op_tests/op_benchmarks/mi300x_fp8_gemm"
export CFG="$BENCH/configs"
export WORK="$(mktemp -d)"
git -C "$AITER_SRC" submodule update --init 3rdparty/composable_kernel
export CK_DIR="$AITER_SRC/3rdparty/composable_kernel"
unset AITER_ROOT
export AITER_META_DIR="$AITER_SRC"
export PYTHONPATH="$AITER_SRC${PYTHONPATH:+:$PYTHONPATH}"
export GPU_ARCHS=gfx942 HIP_VISIBLE_DEVICES=0
export AITER_CONFIG_GEMM_BF16="$CFG/ptpc.csv"
python -c 'import aiter, flydsl; print(aiter.__file__, flydsl.__version__)'
```

## 3. PTPC hipBLASLt tuning

**Use the selected solutions.** Set `AITER_CONFIG_GEMM_BF16` before importing AITER. The included table contains **42 rows** for the 21 shapes with ordinary and (16,16)-packed weights. All packed rows use the verified hipBLASLt solution ID for their shape. For a two-scalar calibration, set the variable to `$CFG/scalar.csv` in a separate process. Both layouts of FP32 scales share the boolean `scaleAB` dispatch key. [PTPC solutions](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/ptpc.csv) · [Two-scalar solutions](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/scalar.csv) · [Gradlib](../gradlib/README.md)

**Retune a shape on the deployment GPU.** The stock AITER tuner invokes gradlib through `--with-hipblaslt`. For the full matrix use the supplied `$CFG/ptpc-tune-input.csv` in place of the one-row input below. The qualifier checks the incumbent, stock winner and up to three stock finalists on signed data. A challenger replaces a valid incumbent only when faster in both launch orders. Its output merges the verified row into the supplied baseline table.

```bash
printf 'M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle\n8192,4096,32768,False,torch.float8_e4m3fnuz,torch.bfloat16,True,True\n' > "$WORK/ptpc-input.csv"
CACHE_INVALIDATE_BUFFERS=1 python "$AITER_SRC/csrc/gemm_a16w16/gemm_a16w16_tune.py" \
  --libtype hipblaslt --with-hipblaslt --mp 1 \
  -i "$WORK/ptpc-input.csv" -o "$WORK/stock.csv" --profile_file "$WORK/ptpc-profile.csv"
python "$BENCH/qualify_ptpc.py" --baseline "$CFG/ptpc.csv" --stock "$WORK/stock.csv" \
  --profile "$WORK/ptpc-profile.csv" --output "$WORK/ptpc-verified.csv"
```

## 4. FP8 block-scale GEMM tuning and dispatch

**Use the mixed public table.** Set `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE="$CFG/best-direct.csv"` for the packed operator. The table has **18 unique runtime rows**. Look up each full M/N/K in `best-plans.csv` before calling the operator. It selects classic CK on **14** shapes and ASM on **7**. Three of the CK calls use caller-side row chunking. Each chunk shares the original packed weights and weight scales and packs its own activation scales. [Runtime table](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/best-direct.csv) · [21 caller plans](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/best-plans.csv)

|       M |      N |      K | CK calls × rows |
| ------: | -----: | -----: | ---------------: |
|  65,536 | 32,768 |  1,536 |      4 × 16,384 |
| 131,072 | 32,768 |  1,536 |      8 × 16,384 |
| 131,072 |  4,096 | 32,768 |      2 × 65,536 |

```python
# plan is the matching CSV row. Activation scales here are logical row-major.
step = int(plan["chunk_m"])
Y = torch.empty((X_fp8.shape[0], W_packed.shape[0]), device=X_fp8.device, dtype=torch.bfloat16)
for start in range(0, X_fp8.shape[0], step):
    rows = slice(start, start + step)
    scales = activation_block_scales[rows]
    gemm_a8w8_blockscale_bpreshuffle(
        X_fp8[rows], W_packed, scales.T.contiguous().view_as(scales),
        weight_block_scales, torch.bfloat16, out=Y[rows],
    )
```

**Tune the packed layout.** The native tuner accepts `ck`, `cktile`, `asm`, `opus`, `flydsl`, `both` (CK and CK Tile) and `all`. Use `--splitK` when searching ASM. The 18 shapes used with its dense reference are in `$CFG/block-tune-input.csv`. On the three large shapes above, the separate qualifier uses the same native candidates with a sampled CPU FP64 reference. It screens signed inputs, checks eager and captured results, and writes the fastest passing direct row for the selected family. A failed large descriptor can still use a previously validated caller chunk plan. [Native block tuner](../csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py) · [Qualification tool](../op_tests/op_benchmarks/mi300x_fp8_gemm/qualify_blockscale.py)

```bash
printf 'M,N,K\n8192,4096,32768\n' > "$WORK/block-input.csv"
python "$AITER_SRC/csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py" \
  --preshuffle --libtype flydsl --mp 1 --warmup 3 --iters 10 \
  -i "$WORK/block-input.csv" -o "$WORK/flydsl-stock.csv" --profile_file "$WORK/flydsl-profile.csv"
python "$BENCH/qualify_blockscale.py" --family flydsl --shape 8192 4096 32768 \
  --output "$WORK/flydsl-verified.csv"

# Regenerate the ordinary-weight FlyDSL winners for all customer shapes.
python "$BENCH/qualify_blockscale.py" --family flydsl --layout ordinary \
  --input "$CFG/replay-plans.csv" --output "$WORK/flydsl-ordinary.csv"
```

On gfx942, the original five tile sizes have both ordinary- and packed-weight variants. The following additional **packed 128×128×128** candidates accept FNUZ inputs, FP32 scales and BF16/FP16 output. M and N must be divisible by 128. K must be divisible by 128. The register candidate also requires the rebased 128-row buffer descriptor to fit in a signed 32-bit byte offset. Existing gfx950 IDs 0/1 and gfx942 IDs 2–11 retain their meaning.

| FlyDSL ID | Suffix in `kernelName` | Implementation                                                                          |
| --------: | ------------------------ | --------------------------------------------------------------------------------------- |
|        12 | `_lds_m`               | Two 16 KiB activation tiles in LDS, M-first workgroup traversal.                        |
|        13 | `_lds_n`               | Same two-stage pipeline, N-first traversal.                                             |
|        14 | `_reg_a`               | Four waves along M, transposed MFMA and next-K128 A fragment prefetched into registers. |

To run the best recorded **FlyDSL-only** choice per shape through the public packed operator, use `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE="$CFG/flydsl-final.csv"`. In the per-library replay, the 21 packed rows selected N-first LDS on ten shapes, M-first LDS on nine and the register cache on two. The historical `flydsl.csv` selects the original packed port used as the control. The ordinary FlyDSL table contains one signed-input-qualified `ps0` winner per shape. [Ordinary FlyDSL winners](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/flydsl-ordinary.csv) For ordinary CK, use the `block_ck_plain` row length in the replay plans and the matching `block-plain.csv` row. The ordinary block operator takes its own `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` table. [FlyDSL-only table](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/flydsl-final.csv) · [Direct-load and LDS implementation](../aiter/ops/flydsl/kernels/gemm_a8w8_blockscale_gfx942.py) · [Register implementation](../aiter/ops/flydsl/kernels/gemm_a8w8_blockscale_gfx942_register.py)

The consolidated implementations passed the signed-input replay at all 21 shapes. At the reference shape the emitted gfx942 instruction sequences for all three matched the corresponding measured study kernels.

## 5. Results by scaling method

Each cell gives the fastest validated **median TFLOP/s** for that shape, followed by the winning backend’s `libtype` in parentheses. The first column supplies `M` for the three customer shapes. One multiply and one add count as two FLOPs. Timings include all launches required by each row plan. Quantization, offline packing and JIT compilation precede timing.

### One tuning candidate per library

For a shape and scaling contract, look up the row for the desired `libtype` in **`fp8-tuned-gemm.csv`**. There is one winning candidate per **`(M, N, K, scaling, libtype)`**. Ordinary and packed weights compete within that group. The row records the winning weight layout, method, kernel name or solution ID when available, split-K setting when known, rows per call, call count, median µs, TFLOP/s and numerical errors. The scaling keys are `ptpc`, `block128`, and `tensor_channel` for the constant-activation-scale control. `torch` identifies PyTorch-managed dispatch. [Per-library winners CSV](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/fp8-tuned-gemm.csv)

Of the **238 winners**, **63** use PTPC, **112** use K128 block scaling and **63** use the tensor × channel control. Each winner has the lowest validated median latency in its group in a dedicated replay. The replay uses the tuned solution IDs for ordinary and packed hipBLASLt and enumerates all five ordinary and eight packed FlyDSL candidates per shape. All 273 FlyDSL shape/layout/kernel combinations passed the signed CPU FP64 sample check. A library with no passing candidate for a shape has no winner row. For example, block-scale OPUS contributed seven winners at `N = 1536, K = 4096`. The FlyDSL winner for `[M, 32768] @ [4096, 32768]` at `M = 8192` was packed ID **13**, two-stage N-first LDS, at **783.0 TFLOP/s**. Use the per-API runtime CSVs in sections 3 and 4 for dispatch. The winners index has multiple libraries and scale contracts for the same AITER runtime lookup key. The summary tables below report the original cross-scaling sweep.

### Per-token/per-channel FP8 (PTPC)

AITER’s `tgemm.mm` with packed weights selected `hipblaslt` for all 21 winners. [PTPC solution IDs](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/ptpc.csv)

|       M | `[M, 4096] @ [1536, 4096]` | `[M, 1536] @ [1536, 32768]` | `[M, 32768] @ [4096, 32768]` |
| ------: | ---------------------------: | ----------------------------: | -----------------------------: |
|   2,048 |        759.3 (`hipblaslt`) |         980.3 (`hipblaslt`) |        1,098.7 (`hipblaslt`) |
|   4,096 |        904.8 (`hipblaslt`) |         957.4 (`hipblaslt`) |        1,173.8 (`hipblaslt`) |
|   8,192 |      1,112.6 (`hipblaslt`) |         935.7 (`hipblaslt`) |        1,154.8 (`hipblaslt`) |
|  16,384 |      1,127.5 (`hipblaslt`) |         953.3 (`hipblaslt`) |        1,172.4 (`hipblaslt`) |
|  32,768 |      1,144.6 (`hipblaslt`) |         960.7 (`hipblaslt`) |        1,180.9 (`hipblaslt`) |
|  65,536 |      1,118.3 (`hipblaslt`) |         978.2 (`hipblaslt`) |        1,177.0 (`hipblaslt`) |
| 131,072 |      1,085.9 (`hipblaslt`) |         958.0 (`hipblaslt`) |        1,170.5 (`hipblaslt`) |

### FP8 block-scale GEMM

Packed classic Composable Kernel (`ck`) led on 14 shapes and packed assembly (`asm`) on seven. The full report also gives the rates for `flydsl` and the other measured backends. [CK tuning rows](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/ck.csv) · [ASM tuning rows](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/asm.csv)

|       M | `[M, 4096] @ [1536, 4096]` | `[M, 1536] @ [1536, 32768]` | `[M, 32768] @ [4096, 32768]` |
| ------: | ---------------------------: | ----------------------------: | -----------------------------: |
|   2,048 |              646.4 (`asm`) |                766.2 (`ck`) |                890.4 (`asm`) |
|   4,096 |              711.8 (`asm`) |                743.9 (`ck`) |                872.9 (`asm`) |
|   8,192 |               762.9 (`ck`) |                748.0 (`ck`) |                899.9 (`asm`) |
|  16,384 |               807.6 (`ck`) |                746.9 (`ck`) |                894.4 (`asm`) |
|  32,768 |               834.1 (`ck`) |                738.5 (`ck`) |                895.9 (`asm`) |
|  65,536 |               802.4 (`ck`) |                740.8 (`ck`) |                 838.4 (`ck`) |
| 131,072 |               809.6 (`ck`) |                736.1 (`ck`) |                 836.3 (`ck`) |

Across the 21 shapes, the fastest PTPC path delivered **17.5–45.8%** more throughput than the fastest K128 block-scale path, with a median advantage of **31.1%**. The complete report covers PyTorch, ordinary and packed hipBLASLt, native AITER PTPC, Triton, ordinary and packed CK, CK Tile, ASM, OPUS, the original FlyDSL port, both LDS orders and the register kernel. It also covers the four constant-activation-scale controls. There are **364 validated medians**. OPUS had a qualified plan on seven shapes, including row-chunk replays. At the reference shape its direct kernel exceeded the signed-input error budget. [Full report](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/mi300x-fp8-ptpc-vs-blockscale.md) · [Dispatch and row lengths by method](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/replay-plans.csv) · [OPUS diagnostics](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/opus-reference-diagnostic.json)

The earlier packed PTPC, two-scalar and CK/CK Tile experiment reached **1,205.9 TFLOP/s** for PTPC on `[M, 32768] @ [4096, 32768]`. This is a separate GPU 0 replay. In the additional native backend comparison, ASM improved the reference shape by **9.21%** over CK in paired tests. [Scalar, PTPC and CK measurements](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/ptpc-scalar-and-ck.csv) · [Native block-scale comparisons](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/native-block-comparison.csv)

The earlier two-backend comparisons favored M-first LDS over CK for `[M, 1536] @ [1536, 32768]` at `M = 4096` and `[M, 4096] @ [1536, 4096]` at `M = 16384`. The later paired medians favored CK on both. The opt-in mixed table uses the validated CK/ASM choices, and the family-specific tables allow direct comparison when retuning another card. [Earlier LDS comparisons](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/lds-comparison.csv) · [Later paired samples](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/cross-scaling/confirmation/)

In a separate sustained 750 W run at the reference shape, the register kernel reached **668.0 TFLOP/s** at 1,315 MHz, two-stage LDS reached **788.4** at 1,245 MHz, and ASM reached **902.0** at 1,050 MHz. The register implementation beat the original FlyDSL port on all 21 shapes. Its advantage over LDS was shape-specific. For `[M, 4096] @ [1536, 4096]` at `M = 2048`, it sustained 484.3 TFLOP/s versus 409.9 for M-first LDS and 664.9 for ASM. [Power, clocks and energy](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/power-summary.csv) · [Register-kernel comparison](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/matrix-summary.csv)

## 6. Benchmark and report

The harness checks each path against the matching sampled CPU FP64 computation using signed inputs and varying scales. It checks chunk boundaries, eager output and captured graph replay. The limits are **0.006** relative RMS and **0.035** maximum absolute error divided by reference RMS. The recorded full sweep used four rounds, two HIP-event samples per method per round and three complete operations per graph. [Benchmark](../op_tests/op_benchmarks/mi300x_fp8_gemm/bench_fp8_scaling_comparison.py) · [Recorded samples](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/cross-scaling/)

```bash
# One reference shape, the full 21-shape sweep, or only the mixed public dispatch.
python "$BENCH/bench_fp8_scaling_comparison.py" --shape 8192 4096 32768 --output "$WORK/reference"
python "$BENCH/bench_fp8_scaling_comparison.py" --output "$WORK/replay"
python "$BENCH/bench_fp8_scaling_comparison.py" --methods block_best --output "$WORK/deployment"

# Export new live samples or rebuild the historical tables from the bundled samples.
python "$BENCH/summarize_fp8_scaling_comparison.py" --raw "$WORK/replay" --output "$WORK/report"
python "$BENCH/summarize_fp8_scaling_comparison.py" --output "$WORK/recorded"

# Replay the complete FlyDSL catalog with the other benchmark paths and export one winner per libtype.
python "$BENCH/bench_fp8_scaling_comparison.py" --all-flydsl --output "$WORK/per-lib-replay"
python "$BENCH/export_fp8_tuned_gemm.py" --raw "$WORK/per-lib-replay" \
  --output "$WORK/fp8-tuned-gemm.csv" --flydsl-runtime-dir "$WORK/flydsl-tables"

# Time the three final kernels under sustained load with AMD SMI telemetry.
python "$BENCH/bench_register_power.py" --shape 8192 4096 32768 \
  --variants block_register block_lds_n block_asm --output "$WORK/power.json"
```

For new tables, copy `$CFG` into the run directory, replace the relevant CSVs there, pass that directory through `--config-dir`, and set `AITER_CONFIG_GEMM_BF16` to its `ptpc.csv` before starting Python. Keep the per-family and deployment row plans consistent with the selected kernels. The exporter reads only saved JSON and runs with the Python standard library. [Exporter](../op_tests/op_benchmarks/mi300x_fp8_gemm/summarize_fp8_scaling_comparison.py)
