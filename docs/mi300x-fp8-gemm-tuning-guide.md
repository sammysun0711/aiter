# MI300X FP8 GEMM Tuning Guide

[中文版](mi300x-fp8-gemm-tuning-guide-zh.md)

This guide explains how to tune bias-free FP8 linear projections on one **304-CU MI300X (gfx942)** using E4M3 FNUZ inputs (`torch.float8_e4m3fnuz`), FP32 dequantization scales and BF16 output.

Use dimensions **M,N,K**: **M** is the number of input rows (tokens), **N** is the output feature width, and **K** is the input feature width (reduction dimension). AITER takes `X[M,K]` and `W[N,K]` and computes `Y[M,N] = X @ W.T`.

Choose the workflow that matches the model's scales. Both tuners write runtime CSVs for replay through the public operator:

- Tune per-token/per-channel FP8 GEMM with gradlib. [FP8 PTPC GEMM tuning](#21-tune-a-ptpc-fp8-gemm)
- Tune K128 block-scale FP8 GEMM with the native AITER tuner. [FP8 Block-scale GEMM tuning](#22-tune-an-fp8-block-scaled-gemm)

## 1. AITER FP8 GEMM API and layout definition

### 1.1. Per-token/per-channel (PTPC) FP8 GEMM

One FP32 activation scale per token and one FP32 weight scale per output channel multiply the whole dot product. Supply contiguous scales `[M,1]` and `[1,N]` on the GPU. `tgemm.mm` accepts ordinary and packed weights. [PTPC dispatch](../aiter/tuned_gemm.py)

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

Keep the tensor returned by `shuffle_weight`, including its layout metadata. For ordinary weights, pass `W_fp8` directly to `tgemm.mm`.

### 1.2. Block-scale FP8 GEMM

Each K128 partial dot product has its own multiplier. The logical activation scales are `[M,K/128]`. Weight scales are ordinary row-major `[ceil(N/128),K/128]`, one value per 128×128 weight tile. The `sx_block` and `sw_block` in this example start as contiguous row-major FP32 tensors. [Block-scale dispatch](../aiter/ops/gemm_op_a8w8.py)

| Weight storage                        | Public API                           | Activation-scale storage                     | Runtime table variable                            |
| ------------------------------------- | ------------------------------------ | -------------------------------------------- | ------------------------------------------------- |
| Ordinary `W[N,K]`                   | `gemm_a8w8_blockscale`             | Row-major `[M,K/128]`                      | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE`             |
| `shuffle_weight(W, layout=(16,16))` | `gemm_a8w8_blockscale_bpreshuffle` | K-group-major bytes exposed as `[M,K/128]` | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE` |

```python
import torch
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale, gemm_a8w8_blockscale_bpreshuffle
from aiter.ops.shuffle import shuffle_weight

# Ordinary weights and row-major activation scales.
Y = gemm_a8w8_blockscale(X_fp8, W_fp8, sx_block, sw_block, torch.bfloat16)

# Packed weights and K-group-major activation scales.
W_packed = shuffle_weight(W_fp8, layout=(16, 16))  # Once at weight load
sx_packed = sx_block.T.contiguous().view_as(sx_block)
Y = gemm_a8w8_blockscale_bpreshuffle(X_fp8, W_packed, sx_packed, sw_block, torch.bfloat16)
```

A quantizer that emits K-group-major activation scales can pass them directly to the packed operator. On gfx942, FlyDSL PF1 supports ordinary and packed weights (IDs 15 and 16). PF2 IDs 17–23 use packed weights. The tuner checks each candidate's tile and descriptor constraints for the actual M,N,K. [FlyDSL registry](../aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_blockscale_common.py) · [Layout adapter](../aiter/ops/flydsl/gemm_a8w8_blockscale.py)

## 2. Setup AITER repo for FP8 GEMM tuning

The results below used ROCm 7.2.0, PyTorch 2.9.1, hipBLASLt 1.2.1 and FlyDSL 0.5.0. Use an idle GPU for timing. [AITER installation](installation.rst)

```bash
git clone https://github.com/sammysun0711/aiter -b mimo/fp8-gemm-tuning-guide-mi300x
cd aiter
git submodule update --init 3rdparty/composable_kernel
# Run inside the activated environment containing your ROCm PyTorch build.
AITER_USE_SYSTEM_TRITON=1 python3 -m pip install --no-build-isolation -e .

export AITER_SRC="$PWD" BENCH="$PWD/op_tests/op_benchmarks/mi300x_fp8_gemm"
export PYTHONPATH="$AITER_SRC${PYTHONPATH:+:$PYTHONPATH}"
export WORK="$(mktemp -d)"
```

## 2.1. Tune a PTPC FP8 GEMM

Use AITER's `gemm_a16w16_tune.py` with `--libtype hipblaslt --with-hipblaslt`. Gradlib queries the installed hipBLASLt library, runs a fast screen, then measures and checks the finalists. For FP8 PTPC, set `scaleAB=True`. The examples tune the requested packed `o_proj` at M=8,192 and the customer's 4,096-to-1,536 projection with ordinary weights at M=2,048 from the extended sweep. [Gradlib guide](../gradlib/README.md) · [Solution search](../gradlib/gradlib/GemmTuner.py)

```bash
PTPC_TUNER="$AITER_SRC/csrc/gemm_a16w16/gemm_a16w16_tune.py"
unset AITER_CONFIG_GEMM_BF16
header='M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle'
printf '%s\n' "$header" '2048,1536,4096,False,torch.float8_e4m3fnuz,torch.bfloat16,True,False' > "$WORK/ptpc-ordinary-input.csv"
printf '%s\n' "$header" '8192,4096,32768,False,torch.float8_e4m3fnuz,torch.bfloat16,True,True' > "$WORK/ptpc-packed-input.csv"

for layout in ordinary packed; do
  CACHE_INVALIDATE_BUFFERS=1 python "$PTPC_TUNER" \
    --libtype hipblaslt --with-hipblaslt --mp 8 -i "$WORK/ptpc-$layout-input.csv" \
    -o "$WORK/ptpc-$layout.csv" -o2 "$WORK/ptpc-$layout.profile.csv"
done
```

To run a fresh search for the entire customer matrix, use the 42-row input (21 shapes × two layouts). Replay those selections with `benchmark_fp8_ptpc_gemm.py` using `--input "$BENCH/configs/ptpc-tune-input.csv"` and `--table "$WORK/customer-tuned/ptpc.csv"`. It benchmarks both layouts in one invocation, as shown in section 3.

```bash
mkdir -p "$WORK/customer-tuned"
CACHE_INVALIDATE_BUFFERS=1 python "$PTPC_TUNER" \
  --libtype hipblaslt --with-hipblaslt --mp 8 -i "$BENCH/configs/ptpc-tune-input.csv" \
  -o "$WORK/customer-tuned/ptpc.csv" -o2 "$WORK/customer-tuned/ptpc.profile.csv"
```

The winner's `solidx` identifies a solution in the installed hipBLASLt version. Requalify it when changing GPU or library version. To compare a new gradlib finalist with an existing serving row, use the optional incumbent comparison, which measures both launch orders. [PTPC incumbent comparison](../op_tests/op_benchmarks/mi300x_fp8_gemm/qualify_ptpc.py)

PTPC replay and incumbent qualification use the same input recipe as the native hipBLASLt tuner: seed 0, `torch.randn(...).to(torch.bfloat16)` for both activations and weights, then `aiter.pertoken_quant(..., quant_dtype=torch.float8_e4m3fnuz)` for each. The returned scales are passed as `[M,1]` and `[1,N]`. Quantization and weight packing happen before timing.

## 2.2. Tune an FP8 block-scaled GEMM

Use AITER's native `gemm_a8w8_blockscale_tune.py` to search the three customer projections at M=8,192 and M=131,072. It passes eligible candidates to `mp_tuner`, checks them against its Torch reference and writes the winner for each M,N,K and weight layout. The examples below use `--libtype all`, which includes eligible FlyDSL, CK, CKTile, ASM and OPUS candidates according to layout. To restrict the search target backend, run one backend per invocation with a separate output file for each. [Native tuner](../csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py)

```bash
export AITER_MAX_JOBS=128
BLOCK_TUNER="$AITER_SRC/csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py"
cat > "$WORK/shapes.csv" <<'CSV'
M,N,K
8192,1536,4096
8192,32768,1536
8192,4096,32768
131072,1536,4096
131072,32768,1536
131072,4096,32768
CSV
unset AITER_CONFIG_GEMM_A8W8_BLOCKSCALE AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE

# Ordinary W[N,K], row-major activation scales.
python "$BLOCK_TUNER" -i "$WORK/shapes.csv" -o "$WORK/block-ordinary.csv" \
  --libtype all --shape_grouped --batch 8 --mp 8 --warmup 3 --iters 10 -o2 "$WORK/block-ordinary.profile.csv"

# Packed W and K-group-major activation scales. Include ASM split-K choices.
python "$BLOCK_TUNER" -i "$WORK/shapes.csv" -o "$WORK/block-packed.csv" \
  --preshuffle --libtype all --splitK --shape_grouped --batch 8 --mp 8 --warmup 3 --iters 10 \
  -o2 "$WORK/block-packed.profile.csv"
```

With `--shape_grouped`, each worker processes one shape at a time; this input can occupy at most six GPUs concurrently per invocation. `-o` writes the selected runtime rows. `-o2` writes the candidate profile. Use `--libtype flydsl` for a FlyDSL-only search or `--libtype all` for every eligible backend. Packed-weight tuning and native replay both take `--preshuffle`.

For an extra signed CPU FP64 and graph check on each native candidate that passes the tuner’s screen, use `qualify_blockscale.py` with `--input`, `--layout`, `--family` and `--output`. Its companion profile records sampled errors and rejections. [Signed qualification](../op_tests/op_benchmarks/mi300x_fp8_gemm/qualify_blockscale.py)

### Full sweep across the customer matrix

Use `configs/block-shapes.csv` for all **21 M,N,K shapes**, covering seven M values from 2,048 through 131,072 and three projections. Block-scale layout is selected by `--preshuffle`, so two searches cover **42 shape/layout cases**. Write separate 21-row winner tables to `$WORK/customer-tuned/block-ordinary.csv` and `$WORK/customer-tuned/block-packed.csv`, with candidate profiles beside them.

The recipe below uses `--libtype all` directly on the full shape list. The native tuner skips gfx942 OPUS candidates whose last tile-row offset for A or C exceeds signed int32 range, before allocating or launching that candidate; other backends remain eligible for the same shape.

```bash
export AITER_MAX_JOBS=128
BLOCK_TUNER="$AITER_SRC/csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py"
BLOCK_SHAPES="$BENCH/configs/block-shapes.csv"
CUSTOMER_TUNED="$WORK/customer-tuned"
mkdir -p "$CUSTOMER_TUNED"
unset AITER_CONFIG_GEMM_A8W8_BLOCKSCALE AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE

for layout in ordinary packed; do
  layout_args=()
  if [[ "$layout" == packed ]]; then
    layout_args+=(--preshuffle --splitK)
  fi
  python "$BLOCK_TUNER" \
    -i "$BLOCK_SHAPES" -o "$CUSTOMER_TUNED/block-$layout.csv" \
    "${layout_args[@]}" --libtype all \
    --shape_grouped --batch 8 --mp 8 --warmup 3 --iters 10 \
    -o2 "$CUSTOMER_TUNED/block-$layout.profile.csv"
  python "$BENCH/benchmark_fp8_blockscale_gemm.py" \
    --input "$BLOCK_SHAPES" --scaling block128 --layout "$layout" \
    --table "$CUSTOMER_TUNED/block-$layout.csv" \
    --output "$CUSTOMER_TUNED/block-$layout-verified.csv"
done
```

Replay uses the complete original shape list. Both `block-ordinary-verified.csv` and `block-packed-verified.csv` should contain **21 passing rows**.

## 3. Benchmark tuned config for best candidate

**FP8 PTPC GEMM.** Use `benchmark_fp8_ptpc_gemm.py` with the full PTPC table generated in section 2.1. It validates and benchmarks ordinary and packed weights through `tgemm.mm` in paired rounds, using the same timing settings as section 4.1. [PTPC public benchmark](../op_tests/op_benchmarks/mi300x_fp8_gemm/benchmark_fp8_ptpc_gemm.py)

```bash
PTPC_TABLE="$WORK/customer-tuned/ptpc.csv"
python "$BENCH/benchmark_fp8_ptpc_gemm.py" \
  --input "$BENCH/configs/ptpc-tune-input.csv" --table "$PTPC_TABLE" \
  --rounds 8 --groups 3 --ops 3 --output "$WORK/ptpc-replay"
```

The command writes 21 per-shape JSON files under `$WORK/ptpc-replay`, each containing both layouts, selected solution IDs, numerical errors and timing samples.

**FP8 Blockscale GEMM.** Use the full customer winner tables produced in section 2.2: `block-ordinary.csv` and `block-packed.csv`, each with 21 M,N,K rows. The latest verified snapshots are also included as [ordinary](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/block-ordinary.csv) and [packed](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/block-packed.csv) tables. These are the fastest passing native candidates selected in this run and have passed public replay.

Start a fresh process for each layout. The benchmark sets the table before importing AITER, checks the exact dispatch row and verifies sampled CPU FP64 results in eager and graph execution. FlyDSL rows also check kernel ID, full name, layout and shape against the installed registry. [Public benchmark](../op_tests/op_benchmarks/mi300x_fp8_gemm/benchmark_fp8_blockscale_gemm.py)

```bash
BLOCK_SHAPES="$BENCH/configs/block-shapes.csv"
BLOCK_CFG_DIR="$WORK/customer-tuned"
# To replay the supplied snapshots from section 4.2 instead:
# BLOCK_CFG_DIR="$BENCH/configs"
for layout in ordinary packed; do
  python "$BENCH/benchmark_fp8_blockscale_gemm.py" \
    --input "$BLOCK_SHAPES" --scaling block128 --layout "$layout" \
    --table "$BLOCK_CFG_DIR/block-$layout.csv" \
    --output "$WORK/block-$layout-replay.csv"
done
```

Block-scale public timing now uses the native tuner input recipe by default: seed 0, FP16 `torch.rand / 10` cast to FP8, and uniform FP32 K128 scales.

To benchmark FP8 Blockscale GEMM for the packed `o_proj` alone:

```bash
python "$BENCH/benchmark_fp8_blockscale_gemm.py" --shape 8192 4096 32768 \
  --scaling block128 --layout packed --table "$BLOCK_CFG_DIR/block-packed.csv" \
  --output "$WORK/o-proj-replay.csv"
```

Native Torch-reference replay is also available with `python "$BLOCK_TUNER" --preshuffle --run_config "$BLOCK_CFG_DIR/block-packed.csv"` or `python "$PTPC_TUNER" --run_config "$PTPC_TABLE"`: [Packed CK build and replay](../csrc/ck_gemm_a8w8_blockscale_bpreshuffle/README.md)

**Please note** , for model serving, make sure export the absolute paths of the verified runtime tables **before** importing AITER. Include all shapes and layouts the service needs. The JSON and `*-replay.csv` outputs are measurement reports; use the winner tables for dispatch.

```bash
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE="$BLOCK_CFG_DIR/block-ordinary.csv"
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE="$BLOCK_CFG_DIR/block-packed.csv"
# If the full PTPC table has also been tuned and verified, enable it separately:
# export AITER_CONFIG_GEMM_BF16="$PTPC_TABLE"
```

The supplied snapshots are workload-specific overrides. The customer snapshots and their measurements remain the full native-tuner selections.

## 4. Run benchmark with winner kernels via public API

The customer matrices report the median throughput of AITER's public operators using saved dispatch choices. The fresh native searches also report **Tuner TFLOP/s** from each winner CSV. Public verification used signed FNUZ inputs, FP32 scales and BF16 output. Every reported row passed sampled CPU FP64 checks in eager and graph execution.

Throughput is `2 × M × N × K / (median_us × 10⁶)` TFLOP/s. Weight packing, preparation of packed activation scales and JIT compilation precede public timing. The ordinary FlyDSL call includes its activation-scale transpose.

### 4.1. FP8 PTPC hipBLASLt GEMM

#### Run benchmark across the customer shapes

Full customer sweep measured with tuner-aligned `gradlib_ptpc` input recipe (seed 0 and `pertoken_quant` for activations and weights). Packed `tgemm.mm` reached **1,231.1 TFLOP/s** at `(32768,4096,32768)` and **1,190.3 TFLOP/s** at the reference `o_proj`, `(8192,4096,32768)`. The run used the existing saved hipBLASLt solution IDs for all 21 customer shapes on GPU 0 of the same 304-CU MI300X system. Both ordinary and packed weights passed sampled CPU FP64 checks in eager and graph execution. Each rate is the median of 24 HIP-event samples (eight rounds × three samples, three full GEMM calls per timed graph). The benchmark reversed the order of the two layouts in paired rounds. [Runtime table](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/ptpc.csv) · [All 42 public replays with input recipe, solution IDs, errors and timing samples](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/ptpc-customer-replay.csv)

**Packed PTPC throughput in TFLOP/s.** Bold marks the highest measured rate over M for each projection. Each column explicitly lists N (output width) and K (input width); M is the input-row count. AITER stores weights as `[N,K]`.

| M (input rows) |  N=1,536, K=4,096 | N=32,768, K=1,536 | N=4,096, K=32,768 |
| -------------: | ----------------: | ----------------: | ----------------: |
|          2,048 |             793.9 |             974.2 |           1,151.9 |
|          4,096 |             914.9 |             988.4 |           1,216.5 |
|          8,192 |           1,075.5 |             982.9 |           1,190.3 |
|         16,384 |           1,130.8 |             988.4 |           1,218.2 |
|         32,768 | **1,135.4** |             980.0 | **1,231.1** |
|         65,536 |           1,133.1 |   **998.2** |           1,218.6 |
|        131,072 |           1,080.1 |             979.4 |           1,204.9 |

Replay the supplied solution IDs with the same input recipe and benchmark settings.

```bash
PTPC_CFG="$BENCH/configs"
python "$BENCH/benchmark_fp8_ptpc_gemm.py" --table "$PTPC_CFG/ptpc.csv" \
  --input "$PTPC_CFG/ptpc-tune-input.csv" \
  --rounds 8 --groups 3 --ops 3 --output "$WORK/ptpc-customer-replay"
```

### 4.2. FP8 block-scale GEMM

Full customer sweep searched all **21 shapes in both layouts** with `--libtype all --shape_grouped --batch 8 --mp 8`, three warmups and ten timing iterations. Packed tuning also used `--preshuffle --splitK`. All 42 selected rows are the fastest passing entries in their respective recorded profiles, and all **42/42 public replays passed**. The saved tables contain these native winners: [ordinary](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/block-ordinary.csv) and [packed](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/block-packed.csv).

Use `benchmark_fp8_blockscale_gemm.py` to replay these saved selections through the public API. The following commands cover all 21 shapes in each layout and produce two verification reports, each containing 21 passing rows. [Public benchmark](../op_tests/op_benchmarks/mi300x_fp8_gemm/benchmark_fp8_blockscale_gemm.py)

```bash
BLOCK_SHAPES="$BENCH/configs/block-shapes.csv"
BLOCK_CFG_DIR="$BENCH/configs"
for layout in ordinary packed; do
  python "$BENCH/benchmark_fp8_blockscale_gemm.py" \
    --input "$BLOCK_SHAPES" --scaling block128 --layout "$layout" \
    --table "$BLOCK_CFG_DIR/block-$layout.csv" \
    --output "$WORK/block-$layout-customer-replay.csv"
done
```

For only the M=8K/128K cases below, change the first assignment to `BLOCK_SHAPES="$WORK/shapes.csv"`, using the six-row input from section 2.2; each report should then contain six passing rows. To replay your own fresh winners, change the second assignment to `BLOCK_CFG_DIR="$WORK/customer-tuned"`.

#### Run benchmark across the customer shapes with tuner winners M=8K/128K

The table below shows M=8,192 and M=131,072 from the full sweep. **Tuner TFLOP/s** uses the native profiler timing on each shape's assigned GPU; **Public TFLOP/s** uses the verifier's median HIP-event graph timing on logical GPU 0. Public block-scale timing now matches the native input recipe (seed 0, FP16 uniform draws divided by ten and cast to FP8, with uniform FP32 scales). Signed inputs with seed 2026 remain a separate correctness gate. Native profiling rotates buffers and reports a kernel-duration mean, whereas public replay reuses captured inputs and reports a graph-time median;  [All 42 native winners and candidate counts](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/blockscale-customer-native.csv) · [All 42 verified public results](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/blockscale-customer-replay.csv)

| M, N, K                 | Weight layout | Selected kernel  | Tuner TFLOP/s | Public median µs | Public TFLOP/s |
| ----------------------- | ------------- | ---------------- | ------------: | ----------------: | -------------: |
| `(8192,1536,4096)`    | ordinary      | FlyDSL PF1 ID 15 |        714.81 |            127.09 |         811.06 |
| `(8192,1536,4096)`    | packed        | FlyDSL PF2 ID 22 |        929.56 |            115.92 |         889.22 |
| `(8192,32768,1536)`   | ordinary      | FlyDSL PF1 ID 15 |        911.70 |            918.86 |         897.45 |
| `(8192,32768,1536)`   | packed        | FlyDSL PF2 ID 23 |        986.90 |            861.68 |         957.01 |
| `(8192,4096,32768)`   | ordinary      | FlyDSL PF1 ID 15 |      1,080.44 |          1,957.80 |       1,123.21 |
| `(8192,4096,32768)`   | packed        | FlyDSL PF2 ID 22 |      1,180.81 |          1,829.50 |       1,201.98 |
| `(131072,1536,4096)`  | ordinary      | FlyDSL PF1 ID 15 |        988.78 |          1,657.15 |         995.25 |
| `(131072,1536,4096)`  | packed        | FlyDSL PF2 ID 22 |      1,054.79 |          1,582.37 |       1,042.27 |
| `(131072,32768,1536)` | ordinary      | FlyDSL PF1 ID 15 |        927.50 |         14,285.22 |         923.62 |
| `(131072,32768,1536)` | packed        | FlyDSL PF2 ID 23 |        991.68 |         13,749.01 |         959.64 |
| `(131072,4096,32768)` | ordinary      | FlyDSL PF1 ID 15 |      1,131.03 |         32,030.63 |       1,098.46 |
| `(131072,4096,32768)` | packed        | FlyDSL PF2 ID 18 |      1,227.86 |         29,351.66 |       1,198.72 |

#### Run benchmark across the customer shapes with 2K-128K

Both layouts were replayed through the public operators with nine HIP-event samples and three complete GEMMs per captured graph. The saved verification CSVs contain the median, both input-recipe labels and separate sampled numerical errors for timing and signed validation; All values below are **public replay TFLOP/s**. Bold rows mark the two M values shown above.

**Packed weights.** The selected table contains ASM ID 32 at `(2048,1536,4096)`, CK ID 14 at `(4096,1536,4096)`, 17 FlyDSL PF2 rows and two FlyDSL PF1 rows. The public sweep peaked at **1,212.5 TFLOP/s** at `(16384,4096,32768)` with FlyDSL PF2 ID 22. [Verified packed runtime table](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/block-packed.csv)

|    M (input rows) |  N=1,536, K=4,096 | N=32,768, K=1,536 | N=4,096, K=32,768 |
| ----------------: | ----------------: | ----------------: | ----------------: |
|             2,048 |             692.9 |             904.3 |           1,065.5 |
|             4,096 |             726.4 |             948.6 |           1,200.7 |
|   **8,192** |   **889.2** |   **957.0** | **1,202.0** |
|            16,384 |             901.5 |             973.1 |           1,212.5 |
|            32,768 |           1,049.5 |             970.8 |           1,200.6 |
|            65,536 |           1,045.1 |             963.4 |           1,184.5 |
| **131,072** | **1,042.3** |   **959.6** | **1,198.7** |

**Ordinary weights.** The selected table uses CK ID 0 for the first projection at both M=2,048 and M=4,096, and FlyDSL PF1 ID 15 for the other 19 shapes. These are the explicit selections from this sweep. [Verified ordinary runtime table](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/block-ordinary.csv)

|    M (input rows) | N=1,536, K=4,096 | N=32,768, K=1,536 | N=4,096, K=32,768 |
| ----------------: | ---------------: | ----------------: | ----------------: |
|             2,048 |            571.7 |             825.3 |             819.6 |
|             4,096 |            612.7 |             904.1 |           1,074.5 |
|   **8,192** |  **811.1** |   **897.5** | **1,123.2** |
|            16,384 |            836.8 |             925.8 |           1,068.0 |
|            32,768 |          1,010.6 |             916.2 |           1,138.6 |
|            65,536 |            988.5 |             936.2 |           1,111.1 |
| **131,072** |  **995.2** |   **923.6** | **1,098.5** |

To benchmark current serving defaults instead, unset the matching `AITER_CONFIG_*` override and omit `--table`. The canonical runtime CSVs under `aiter/configs/` are maintained separately from these verified customer snapshots.

## 5. Summary: FP8 PTPC hipBLASLt GEMM vs FP8 Block-scale GEMM

The table compares **packed weights** for all **21 customer shapes**: seven M values from 2,048 through 131,072 across the three N,K projections on 8*304-CU MI300X system.

PTPC uses `tgemm.mm` with saved hipBLASLt solution IDs; block-scale uses `gemm_a8w8_blockscale_bpreshuffle` with the selected native-tuner winners.

The percentage is `(block-scale / PTPC) × 100%`, with **PTPC = 100%**; values above 100% mean higher block-scale throughput.

| M, N, K                       | PTPC hipBLASLt TFLOP/s | Block-scale TFLOP/s | Block-scale / PTPC (PTPC = 100%) |
| ----------------------------- | ---------------------: | ------------------: | -------------------------------: |
| `(2048,1536,4096)`          |                  793.9 |               692.9 |                            87.3% |
| `(2048,32768,1536)`         |                  974.2 |               904.3 |                            92.8% |
| `(2048,4096,32768)`         |                1,151.9 |             1,065.5 |                            92.5% |
| `(4096,1536,4096)`          |                  914.9 |               726.4 |                            79.4% |
| `(4096,32768,1536)`         |                  988.4 |               948.6 |                            96.0% |
| `(4096,4096,32768)`         |                1,216.5 |             1,200.7 |                            98.7% |
| `(8192,1536,4096)`          |                1,075.5 |               889.2 |                            82.7% |
| `(8192,32768,1536)`         |                  982.9 |               957.0 |                            97.4% |
| `(8192,4096,32768)`         |                1,190.3 |             1,202.0 |                           101.0% |
| `(16384,1536,4096)`         |                1,130.8 |               901.5 |                            79.7% |
| `(16384,32768,1536)`        |                  988.4 |               973.1 |                            98.4% |
| `(16384,4096,32768)`        |                1,218.2 |             1,212.5 |                            99.5% |
| `(32768,1536,4096)`         |                1,135.4 |             1,049.5 |                            92.4% |
| `(32768,32768,1536)`        |                  980.0 |               970.8 |                            99.1% |
| `(32768,4096,32768)`        |                1,231.1 |             1,200.6 |                            97.5% |
| `(65536,1536,4096)`         |                1,133.1 |             1,045.1 |                            92.2% |
| `(65536,32768,1536)`        |                  998.2 |               963.4 |                            96.5% |
| `(65536,4096,32768)`        |                1,218.6 |             1,184.5 |                            97.2% |
| `(131072,1536,4096)`        |                1,080.1 |             1,042.3 |                            96.5% |
| `(131072,32768,1536)`       |                  979.4 |               959.6 |                            98.0% |
| `(131072,4096,32768)`       |                1,204.9 |             1,198.7 |                            99.5% |
| **Geomean (21 shapes)** |                    N/A |                 N/A |                  **93.8%** |

Across the full sweep, FP8 PTPC hipBLASLt GEMM has higher recorded throughput in 20 of 21 cases. FP8 Block-scale GEMM reaches 101.0% of PTPC throughput at `(8192,4096,32768)`; its lowest ratio is 79.4% at `(4096,1536,4096)`. Percentages and the geomean use the unrounded [PTPC results](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/ptpc-customer-replay.csv) and [block-scale results](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/blockscale-customer-replay.csv). The final row is the geometric mean of the 21 per-shape throughput ratios, with each shape weighted equally.

Both measurements use HIP-event graph timing with three complete GEMMs per graph. PTPC reports the median of 24 samples using `gradlib_ptpc` inputs; block-scale reports the median of nine samples using `native_blockscale` inputs. These separate runs compare each workflow with its own tuner-aligned inputs, so the ratios reflect input-recipe and measurement differences. The model's quantization scheme determines which API applies.
