# Tune FP8 GEMM on MI300X

Use this guide for bias-free FP8 linear projections on one **304-CU MI300X (gfx942)**, with E4M3 FNUZ (`torch.float8_e4m3fnuz`) inputs, FP32 dequantization scales and BF16 output. Enter shapes as **M,N,K**. AITER takes `X[M,K]` and `W[N,K]` and computes `X @ W.T`. The customer's direct `q_proj` is `8192,32768,4096`. Its `o_proj` is `8192,4096,32768`.

1. [Set up the checkout and GPU](#set-up-the-checkout-and-gpu).
2. **For a known shape**, [replay a saved winner](#replay-a-saved-winner) and use the matching layout in your application.
3. **For a new shape**, follow [block-scale tuning](#tune-a-block-scaled-projection) or [PTPC hipBLASLt tuning](#tune-a-ptpc-projection). Choose the path from the scales your model actually supplies.

## Choose the API and layout

### Per-token/per-channel FP8 (PTPC) GEMM

One FP32 activation scale per token and one FP32 weight scale per output channel multiply the whole dot product. Supply contiguous scales `[M,1]` and `[1,N]` on the GPU. `tgemm.mm` accepts ordinary and packed weights. All 21 recorded PTPC winners use packed weights and hipBLASLt. The PyTorch baseline uses `torch._scaled_mm` with ordinary `W_fp8.T`. [PTPC dispatch](../aiter/tuned_gemm.py)

```python
import torch
from aiter.ops.shuffle import shuffle_weight
from aiter.tuned_gemm import tgemm

W_packed = shuffle_weight(W_fp8, layout=(16, 16))  # Once at weight load
Y = tgemm.mm(
    X_fp8, W_packed, otype=torch.bfloat16,
    scale_a=token_scales.reshape(-1, 1).float().contiguous(),
    scale_b=channel_scales.reshape(1, -1).float().contiguous(),
)
```

Keep the tensor returned by `shuffle_weight`, including its layout metadata. For per-tensor activation × per-channel weight scaling, the study repeated one activation multiplier across M rows. Two scalar scales have a [separate runtime table](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/scalar.csv). Scalar and row-scaled solutions share the `scaleAB=True` dispatch key, so select the table for your contract before starting the process.

### FP8 block GEMM

Each K128 partial dot product has its own multiplier. The logical activation scales are `[M,K/128]`. Weight scales are ordinary row-major `[ceil(N/128),K/128]`, one value per 128×128 weight tile. The `sx_block` and `sw_block` in this example start as contiguous row-major FP32 tensors. [Block-scale dispatch](../aiter/ops/gemm_op_a8w8.py)

| Weight storage | Public API | Activation-scale storage | Runtime table variable |
|---|---|---|---|
| Ordinary `W[N,K]` | `gemm_a8w8_blockscale` | Row-major `[M,K/128]` | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` |
| `shuffle_weight(W, layout=(16,16))` | `gemm_a8w8_blockscale_bpreshuffle` | K-group-major bytes exposed as `[M,K/128]` | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE` |

```python
import torch
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale, gemm_a8w8_blockscale_bpreshuffle
from aiter.ops.shuffle import shuffle_weight

# Ordinary weights, as in the direct q_proj example below.
Y = gemm_a8w8_blockscale(X_fp8, W_fp8, sx_block, sw_block, torch.bfloat16)

# Packed weights for a direct (single-call) row.
W_packed = shuffle_weight(W_fp8, layout=(16, 16))  # Once at weight load
sx_packed = sx_block.T.contiguous().view_as(sx_block)
Y = gemm_a8w8_blockscale_bpreshuffle(X_fp8, W_packed, sx_packed, sw_block, torch.bfloat16)
```

A quantizer that already emits K-group-major scale bytes can pass them directly. FlyDSL uses the same public APIs and layouts. Ordinary candidates are IDs 2, 4, 6, 8 and 10. Packed candidates are IDs 3, 5, 7, 9 and 11 (direct loads), 12 and 13 (two-stage LDS, M-first and N-first traversal), and 14 (register cache). The registry checks tile alignment and descriptor limits. [FlyDSL registry](../aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_blockscale_common.py) · [Layout adapter](../aiter/ops/flydsl/gemm_a8w8_blockscale.py)

## Set up the checkout and GPU

The recorded stack used ROCm **7.2.0**, ROCm PyTorch **2.9.1**, hipBLASLt **1.2.1.70200-43~22.04**, FlyDSL **0.5.0** and Composable Kernel `f93c029e09f6`. The study started from upstream AITER `1dd04e9cf132`. Use a dedicated ROCm development environment with a compatible PyTorch build and GPU access. hipBLASLt solution IDs are version dependent. Record your checkout revision and requalify after a GPU or library change. [Recorded environment](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/environment.json) · [Source installation](installation.rst)

```bash
git clone --branch mimo/fp8-gemm-tuning-guide-mi300x --recurse-submodules https://github.com/sammysun0711/aiter.git
cd aiter
export AITER_SRC="$PWD" BENCH="$PWD/op_tests/op_benchmarks/mi300x_fp8_gemm"
export CFG="$BENCH/configs" WORK="$(mktemp -d)"
export GPU_ARCHS=gfx942 HIP_VISIBLE_DEVICES=0
unset AITER_ROOT
export AITER_META_DIR="$AITER_SRC" CK_DIR="$AITER_SRC/3rdparty/composable_kernel"
export AITER_JIT_DIR="$WORK/jit"
mkdir -p "$AITER_JIT_DIR"
python -m pip install 'flydsl==0.5.0'
python setup.py develop
export PYTHONPATH="$AITER_SRC${PYTHONPATH:+:$PYTHONPATH}"
git rev-parse HEAD
python - <<'PY'
import aiter, flydsl, torch
p = torch.cuda.get_device_properties(0)
print(aiter.__file__, torch.__version__, torch.version.hip, flydsl.__version__)
print(p.name, p.gcnArchName, p.multi_processor_count)
assert p.gcnArchName.startswith("gfx942") and p.multi_processor_count == 304
PY
```

Use a fresh Python process for each table change. The verifier's `--table` sets the appropriate variable before importing AITER. Use an idle GPU for timing and a writable JIT directory. First-use compilation can take longer than the benchmark.

## Replay a saved winner

Select the command for your scaling contract. Both examples use the original reference `o_proj`. The output CSV records the resolved table, library, kernel name or solution ID, number of calls, median microseconds, TFLOP/s and numerical errors.

```bash
python "$BENCH/verify_gemm.py" --shape 8192 4096 32768 --scaling ptpc --layout packed \
  --table "$CFG/ptpc.csv" --output "$WORK/known-ptpc.csv"
python "$BENCH/verify_gemm.py" --shape 8192 4096 32768 --scaling block128 --layout packed \
  --table "$CFG/best-direct.csv" --plans "$CFG/best-plans.csv" --output "$WORK/known-block.csv"
```

| File | How to use it |
|---|---|
| [ptpc.csv](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/ptpc.csv) | 42 hipBLASLt runtime rows for 21 shapes × two layouts. Export `AITER_CONFIG_GEMM_BF16` for `tgemm.mm`. |
| [best-direct.csv](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/best-direct.csv) and [best-plans.csv](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/best-plans.csv) | Set the packed block runtime variable to `best-direct.csv`. Look up the full M,N,K in `best-plans.csv`. Its 21 caller plans reuse 18 runtime rows. |
| [fp8-tuned-gemm.csv](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/fp8-tuned-gemm.csv) | Comparison index with one winner per M,N,K, scaling contract and library, including layout and timing. AITER loads the per-API runtime files above. |

To replay all 21 shapes, replace `--shape 8192 4096 32768` in either command with `--input "$CFG/best-plans.csv"`. The verifier reads each distinct M,N,K once.

Eighteen packed block-scale plans use one call. Three use classic CK in equal M slices. `(65536,32768,1536)` uses 4 × 16,384, `(131072,32768,1536)` uses 8 × 16,384, and `(131072,4096,32768)` uses 2 × 65,536. The caller applies the plan and packs the activation scales **for each slice**. The CLI applies the supplied plan automatically. [Replay implementation](../op_tests/op_benchmarks/mi300x_fp8_gemm/common.py)

```python
# plan is the best-plans.csv row for the full shape.
step = int(plan["chunk_m"])
Y = torch.empty((X_fp8.shape[0], W_packed.shape[0]), device=X_fp8.device, dtype=torch.bfloat16)
for start in range(0, X_fp8.shape[0], step):
    rows = slice(start, start + step)
    scales = sx_block[rows]
    gemm_a8w8_blockscale_bpreshuffle(
        X_fp8[rows], W_packed, scales.T.contiguous().view_as(scales),
        sw_block, torch.bfloat16, out=Y[rows],
    )
```

## Tune a block-scaled projection

**Direct `q_proj` with ordinary weights.** For `M=8192, N=32768, K=4096`, the historical ordinary-weight default reached **282.9 TFLOP/s** with Triton. Loading CK row 0 (split-K 0) raised the same public operator to **776.6 TFLOP/s**. Both passed the signed-input reference check. To reproduce the recorded selection immediately, pass the [example CK table](../op_tests/op_benchmarks/mi300x_fp8_gemm/examples/q-proj/ordinary-ck.csv) to the verifier. [Recorded samples and errors](../op_tests/op_benchmarks/mi300x_fp8_gemm/examples/q-proj/recorded-replay.json)

```bash
# Replace this row or add shapes from your own model.
printf 'M,N,K\n8192,32768,4096\n' > "$WORK/shapes.csv"
unset AITER_CONFIG_GEMM_A8W8_BLOCKSCALE
python "$BENCH/verify_gemm.py" --input "$WORK/shapes.csv" --scaling block128 --layout ordinary \
  --output "$WORK/block-before.csv"

# Start with classic CK, the family used in the recorded example.
python "$BENCH/qualify_blockscale.py" --input "$WORK/shapes.csv" --layout ordinary \
  --family ck --output "$WORK/block-ordinary.csv"

# Check that the public operator selects the winning row for each shape.
python "$BENCH/verify_gemm.py" --input "$WORK/shapes.csv" --scaling block128 --layout ordinary \
  --table "$WORK/block-ordinary.csv" --output "$WORK/block-after.csv"
```

For a quick replay of the historical example, use `--shape 8192 32768 4096 --scaling block128 --layout ordinary --table "$BENCH/examples/q-proj/ordinary-ck.csv"` with the same verifier. A baseline can print `default (no tuned row)`. On this revision, an ordinary gfx942 FP32 block-scale lookup miss at M ≥ 2,048 routes to the Triton fallback.

Compare `median_us` before and after. Check `table`, `libtype`, `kernelName` and `splitK` in the after CSV. The qualifier also writes `.profile.csv` with candidate timings, errors and rejections. A shape with no passing candidate has no winning row. To broaden the ordinary search, use `--family ck cktile flydsl` and a separate output path. For **packed** weights, use `--layout packed` for all three steps and begin with `--family ck cktile asm`. Add `flydsl opus` to screen those families too. To produce a winner per library for your own shapes, run one family at a time with separate output paths. The qualifier sweeps ASM split-K candidates. OPUS passed on seven study shapes and exceeded the signed-input error budget on the reference `o_proj`. [Qualifier](../op_tests/op_benchmarks/mi300x_fp8_gemm/qualify_blockscale.py) · [Native tuner](../csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py)

For deployment, export `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE="$WORK/block-ordinary.csv"` before starting the application, or set `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE` for the packed API. A table override applies to the whole process. Include the other shapes that service needs, with one qualified row per runtime key and weight layout. Keep new runs under `$WORK` to preserve the shipped tables.

The customer verifier and qualifiers cover positive M, N divisible by 16, and K divisible by 128. Individual kernels impose additional constraints. For the three large descriptors above, use their validated chunk plans. For a new packed shape that needs chunking, retune a smaller M that divides the full M, reuse N and K, and copy the selected row’s library, kernel ID, split-K and kernel name into a plan with the full M, `chunk_m` and `calls=M/chunk_m`. Replay the full shape with `--table` and `--plans` to check both sides of each boundary. If an existing CK build lacks the newly selected kernel name, start with a fresh writable `AITER_JIT_DIR` and select the runtime table before the first production call. [Packed CK build and replay](../csrc/ck_gemm_a8w8_blockscale_bpreshuffle/README.md)

## Tune a PTPC projection

AITER’s `gemm_a16w16_tune.py` invokes gradlib for hipBLASLt. The example uses packed weights. Edit the M,N,K rows in the stock input, or use the [original tuning input](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/ptpc-tune-input.csv). Both the stock tuner and the verifier below read this same CSV. For ordinary weights change `bpreshuffle` to `False` and the verifier layout to `ordinary`. [Gradlib tuning options](../gradlib/README.md)

```bash
printf 'M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle\n8192,4096,32768,False,torch.float8_e4m3fnuz,torch.bfloat16,True,True\n' > "$WORK/ptpc-input.csv"
unset AITER_CONFIG_GEMM_BF16
python "$BENCH/verify_gemm.py" --input "$WORK/ptpc-input.csv" --scaling ptpc --layout packed \
  --output "$WORK/ptpc-before.csv"
CACHE_INVALIDATE_BUFFERS=1 python "$AITER_SRC/csrc/gemm_a16w16/gemm_a16w16_tune.py" \
  --libtype hipblaslt --with-hipblaslt --mp 1 -i "$WORK/ptpc-input.csv" \
  -o "$WORK/stock.csv" --profile_file "$WORK/ptpc-profile.csv"
python "$BENCH/qualify_ptpc.py" --baseline "$CFG/ptpc.csv" --stock "$WORK/stock.csv" \
  --profile "$WORK/ptpc-profile.csv" --output "$WORK/ptpc-verified.csv"
python "$BENCH/verify_gemm.py" --input "$WORK/ptpc-input.csv" --scaling ptpc --layout packed \
  --table "$WORK/ptpc-verified.csv" --output "$WORK/ptpc-after.csv"
```

Give the stock tuner a fresh output path when repeating a search. It can skip rows already present. The qualifier checks solution IDs against the installed hipBLASLt library, screens the incumbent and stock finalists with signed data, and tests challengers in both launch orders. The output merges qualified rows into a copy of the supplied baseline. Use `AITER_CONFIG_GEMM_BF16="$WORK/ptpc-verified.csv"` for serving. The historical `BF16` variable name also covers FP8. Compare the public before/after medians on the same idle card. If the difference is small or changes sign, repeat in reverse order before choosing a serving row.

## Measured performance

These are median single-GPU TFLOP/s from the original 21-shape comparison. The column headers preserve the customer's notation. The middle column shows the right operand in multiplication order (`K=1536, N=32768`). AITER stores that weight as `[32768,1536]`. [Full per-method report](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/mi300x-fp8-ptpc-vs-blockscale.md) · [Unrounded CSV](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/mi300x-fp8-ptpc-vs-blockscale.csv)

### PTPC winners

|       M | `[M, 4096] @ [1536, 4096]` | `[M, 1536] @ [1536, 32768]` | `[M, 32768] @ [4096, 32768]` |
| ------: | ---------------------------: | ----------------------------: | -----------------------------: |
|   2,048 |        759.3 (`hipblaslt`) |         980.3 (`hipblaslt`) |        1,098.7 (`hipblaslt`) |
|   4,096 |        904.8 (`hipblaslt`) |         957.4 (`hipblaslt`) |        1,173.8 (`hipblaslt`) |
|   8,192 |      1,112.6 (`hipblaslt`) |         935.7 (`hipblaslt`) |        1,154.8 (`hipblaslt`) |
|  16,384 |      1,127.5 (`hipblaslt`) |         953.3 (`hipblaslt`) |        1,172.4 (`hipblaslt`) |
|  32,768 |      1,144.6 (`hipblaslt`) |         960.7 (`hipblaslt`) |        1,180.9 (`hipblaslt`) |
|  65,536 |      1,118.3 (`hipblaslt`) |         978.2 (`hipblaslt`) |        1,177.0 (`hipblaslt`) |
| 131,072 |      1,085.9 (`hipblaslt`) |         958.0 (`hipblaslt`) |        1,170.5 (`hipblaslt`) |

### FP8 block-scale winners

|       M | `[M, 4096] @ [1536, 4096]` | `[M, 1536] @ [1536, 32768]` | `[M, 32768] @ [4096, 32768]` |
| ------: | ---------------------------: | ----------------------------: | -----------------------------: |
|   2,048 |              646.4 (`asm`) |                766.2 (`ck`) |                890.4 (`asm`) |
|   4,096 |              711.8 (`asm`) |                743.9 (`ck`) |                872.9 (`asm`) |
|   8,192 |               762.9 (`ck`) |                748.0 (`ck`) |                899.9 (`asm`) |
|  16,384 |               807.6 (`ck`) |                746.9 (`ck`) |                894.4 (`asm`) |
|  32,768 |               834.1 (`ck`) |                738.5 (`ck`) |                895.9 (`asm`) |
|  65,536 |               802.4 (`ck`) |                740.8 (`ck`) |                 838.4 (`ck`) |
| 131,072 |               809.6 (`ck`) |                736.1 (`ck`) |                 836.3 (`ck`) |

At the reference `o_proj` (`8192,4096,32768`), PyTorch PTPC reached **1,120.0**, packed hipBLASLt PTPC **1,154.8**, packed block CK **821.0** and packed block ASM **899.9**. Classic CK won 14 block-scale shapes and ASM won seven. The main sweep peaked at **1,180.9** PTPC and **899.9** block-scale. An earlier PTPC experiment reached **1,205.9** at a different M on this projection. Treat 1,200 TFLOP/s as a target to check on each shape. [Earlier PTPC data](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/ptpc-scalar-and-ck.csv) · [Native backend comparisons](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/native-block-comparison.csv)

**FlyDSL.** All 273 eligible tile/shape combinations passed the sampled check in the separate per-library replay. The best FlyDSL result at the reference shape used packed ID 13 at **783.0 TFLOP/s**. In the sustained 750 W study the register variant reached **668.0**, two-stage LDS **788.4**, and ASM **902.0**. The [ordinary](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/flydsl-ordinary.csv) and [packed](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/flydsl-final.csv) FlyDSL tables give the fastest validated choice within that library for each layout. [Power and clocks](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/power-summary.csv)

## What the benchmark measures

Throughput is `2 × M × N × K / (median_us × 10⁶)` TFLOP/s. The verifier checks signed FNUZ inputs and non-unit FP32 scales against sampled CPU FP64 dot products, in eager mode and graph replay. The limits are relative RMS < 0.006 and maximum absolute error / reference RMS < 0.035. Chunked cases sample both sides of every boundary. HIP events time the GEMM launches in the plan. Quantization, offline weight packing, preparation of packed activation scales and JIT compilation precede timing. Measure per-request preparation separately when evaluating a serving path.

The verifier writes one result per requested shape. For the fixed 21-shape study, the commands below reproduce the full comparison and its 238 per-library winners (63 PTPC, 112 block-scale, 63 tensor × channel). The two offline commands use only the Python standard library.

```bash
python "$BENCH/summarize_fp8_scaling_comparison.py" --output "$WORK/study"
python "$BENCH/export_fp8_tuned_gemm.py" --output "$WORK/fp8-tuned-gemm.csv"

# Repeat the study on an idle MI300X with the supplied runtime tables.
AITER_CONFIG_GEMM_BF16="$CFG/ptpc.csv" python "$BENCH/bench_fp8_scaling_comparison.py" --output "$WORK/live"
AITER_CONFIG_GEMM_BF16="$CFG/ptpc.csv" python "$BENCH/bench_fp8_scaling_comparison.py" --all-flydsl --output "$WORK/per-lib"
python "$BENCH/export_fp8_tuned_gemm.py" --raw "$WORK/per-lib" --output "$WORK/live-winners.csv"
```
