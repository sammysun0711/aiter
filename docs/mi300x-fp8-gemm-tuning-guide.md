# MI300X FP8 GEMM Tuning Guide

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

## 2. Set up AITER repo for FP8 GEMM tuning

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

To run a fresh search for the entire customer matrix, use the 42-row input (21 shapes × two layouts). Replay those selections with `benchmark_fp8_blockscale_gemm.py` using `--input "$BENCH/configs/ptpc-tune-input.csv"`, `--table "$WORK/customer-tuned/ptpc.csv"`, `--scaling ptpc` and each of `--layout ordinary` and `--layout packed`.

```bash
mkdir -p "$WORK/customer-tuned"
CACHE_INVALIDATE_BUFFERS=1 python "$PTPC_TUNER" \
  --libtype hipblaslt --with-hipblaslt --mp 8 -i "$BENCH/configs/ptpc-tune-input.csv" \
  -o "$WORK/customer-tuned/ptpc.csv" -o2 "$WORK/customer-tuned/ptpc.profile.csv"
```

The winner's `solidx` identifies a solution in the installed hipBLASLt version. Requalify it when changing GPU or library version. To compare a new gradlib finalist with an existing serving row, use the optional incumbent comparison, which measures both launch orders. [PTPC incumbent comparison](../op_tests/op_benchmarks/mi300x_fp8_gemm/qualify_ptpc.py)

PTPC replay and incumbent qualification use the same input recipe as the native hipBLASLt tuner: seed 0, `torch.randn(...).to(torch.bfloat16)` for both activations and weights, then `aiter.pertoken_quant(..., quant_dtype=torch.float8_e4m3fnuz)` for each. The returned scales are passed as `[M,1]` and `[1,N]`. Quantization and weight packing happen before timing. This applies to `benchmark_fp8_ptpc_gemm.py`, `benchmark_fp8_blockscale_gemm.py --scaling ptpc`, and `qualify_ptpc.py`; outputs identify the recipe as `gradlib_ptpc` with `input_seed=0`.

For tuner-versus-public performance comparisons, also match the GPU and timing method: the native tuner reports a filtered mean of GPU kernel durations, while public replay reports median graph elapsed time.

## 2.2. Tune an FP8 block-scaled GEMM

Use AITER's native `gemm_a8w8_blockscale_tune.py` to search the three customer projections. It passes eligible candidates to `mp_tuner`, checks them against its Torch reference and writes the winner for each M,N,K and weight layout.  The examples below use `--libtype all`, supported by this branch, which also includes eligible FlyDSL, Ck, CKTile and OPUS candidates. To restrict the search target backend, run one backend per invocation with a separate output file for each. [Native tuner](../csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py)

```bash
BLOCK_TUNER="$AITER_SRC/csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py"
cat > "$WORK/shapes.csv" <<'CSV'
M,N,K
8192,1536,4096
8192,32768,1536
8192,4096,32768
CSV
unset AITER_CONFIG_GEMM_A8W8_BLOCKSCALE AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE

# Ordinary W[N,K], row-major activation scales.
python "$BLOCK_TUNER" -i "$WORK/shapes.csv" -o "$WORK/block-ordinary.csv" \
  --libtype all --mp 1 --warmup 3 --iters 10 -o2 "$WORK/block-ordinary.profile.csv"

# Packed W and K-group-major activation scales. Include ASM split-K choices.
python "$BLOCK_TUNER" -i "$WORK/shapes.csv" -o "$WORK/block-packed.csv" \
  --preshuffle --libtype all --splitK --mp 1 --warmup 3 --iters 10 \
  -o2 "$WORK/block-packed.profile.csv"
```

At M=131,072 the recorded search evaluated FlyDSL for both layouts. Keep one shape per tuner batch.

```bash
sed 's/^8192,/131072,/' "$WORK/shapes.csv" > "$WORK/shapes-128k.csv"
python "$BLOCK_TUNER" -i "$WORK/shapes-128k.csv" -o "$WORK/block-ordinary-128k.csv" \
  --libtype flydsl --batch 1 --mp 1 --warmup 3 --iters 10 -o2 "$WORK/block-ordinary-128k.profile.csv"
python "$BLOCK_TUNER" -i "$WORK/shapes-128k.csv" -o "$WORK/block-packed-128k.csv" \
  --preshuffle --libtype flydsl --batch 1 --mp 1 --warmup 3 --iters 10 -o2 "$WORK/block-packed-128k.profile.csv"
```

`-o` writes the selected runtime rows. `-o2` writes the candidate profile. Use `--libtype flydsl` for a FlyDSL-only search or `--libtype all` for every eligible backend. Use a fresh winner filename when repeating a search, or `--all` to retune input rows already present in the output. Packed-weight tuning and native replay both take `--preshuffle`.

For an extra signed CPU FP64 and graph check on each native candidate that passes the tuner’s screen, use `qualify_blockscale.py` with `--input`, `--layout`, `--family` and `--output`. Its companion profile records sampled errors and rejections. [Signed qualification](../op_tests/op_benchmarks/mi300x_fp8_gemm/qualify_blockscale.py)

## 3. Benchmark tuned config for best candidate

Start a fresh process for each table. `benchmark_fp8_blockscale_gemm.py` selects the table before importing AITER, checks the dispatch identity, and measures the public operator. FlyDSL rows also check the kernel ID, full kernel name, weight layout and supported shape against the installed registry. Use the original shape list as `--input` to check every requested shape. The verifier covers positive M, N divisible by 16 and K divisible by 128. [Verifier](../op_tests/op_benchmarks/mi300x_fp8_gemm/benchmark_fp8_blockscale_gemm.py)

```bash
for layout in ordinary packed; do
  python "$BENCH/benchmark_fp8_blockscale_gemm.py" --input "$WORK/shapes.csv" --scaling block128 --layout "$layout" \
    --table "$WORK/block-$layout.csv" --output "$WORK/block-$layout-verified.csv"
  python "$BENCH/benchmark_fp8_blockscale_gemm.py" --input "$WORK/ptpc-$layout-input.csv" --scaling ptpc --layout "$layout" \
    --table "$WORK/ptpc-$layout.csv" --output "$WORK/ptpc-$layout-verified.csv"
done
```

Check `libtype`, `kernelName`, `kernelId` or `solidx`, `median_us`, and the two error columns. The numerical limits are relative RMS < 0.006 and maximum absolute error / reference RMS < 0.035. For native Torch-reference replay, use `python "$BLOCK_TUNER" --preshuffle --run_config "$WORK/block-packed.csv"` or `python "$PTPC_TUNER" --run_config "$WORK/ptpc-packed.csv"`. If a new CK winner is missing from a compiled registry, set a fresh writable `AITER_JIT_DIR` and select the winning table before the first public call. [Packed CK build and replay](../csrc/ck_gemm_a8w8_blockscale_bpreshuffle/README.md)

To benchmark the customer's packed `o_proj` at M=8,192, pass its exact shape and the generated table to the benchmark harness.

```bash
python "$BENCH/benchmark_fp8_blockscale_gemm.py" --shape 8192 4096 32768 --scaling block128 --layout packed \
  --table "$WORK/block-packed.csv" --output "$WORK/o-proj-replay.csv"
```

For serving, export the absolute path of the verified table **before** importing AITER. Include the shapes and layouts the service uses. When serving both PTPC layouts, include rows with both values of `bpreshuffle` in the PTPC table.

```bash
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE="$WORK/block-ordinary.csv"
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE="$WORK/block-packed.csv"
export AITER_CONFIG_GEMM_BF16="$WORK/ptpc-packed.csv"
```

## 4. Native tuner winners and public replay

The customer matrices report the median throughput of AITER's public operators using saved dispatch choices. The fresh native searches also report **Tuner TFLOP/s** from each winner CSV. Public verification used signed FNUZ inputs, FP32 scales and BF16 output. Every reported row passed sampled CPU FP64 checks in eager and graph execution.

Throughput is `2 × M × N × K / (median_us × 10⁶)` TFLOP/s. Weight packing, preparation of packed activation scales and JIT compilation precede public timing. The ordinary FlyDSL call includes its activation-scale transpose.

### 4.1. FP8 PTPC hipBLASLt GEMM

#### Public replay across the customer shapes

These results were remeasured on 2026-11-24 with the tuner-aligned `gradlib_ptpc` input recipe (seed 0 and `pertoken_quant` for activations and weights). Packed `tgemm.mm` reached **1,231.1 TFLOP/s** at `(32768,4096,32768)` and **1,190.3 TFLOP/s** at the reference `o_proj`, `(8192,4096,32768)`. The run used the existing saved hipBLASLt solution IDs for all 21 customer shapes on GPU 0 of the same 304-CU MI300X system. Both ordinary and packed weights passed sampled CPU FP64 checks in eager and graph execution. Each rate is the median of 24 HIP-event samples (eight rounds × three samples, three full GEMM calls per timed graph). The benchmark reversed the order of the two layouts in paired rounds. [Runtime table](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/ptpc.csv) · [All 42 public replays with input recipe, solution IDs, errors and timing samples](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/ptpc-customer-replay.csv)

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

The customer requested three projections at **M=8,192 and M=131,072**. The middle projection is `1536 → 32768` in multiplication order. AITER stores that weight as `W[32768,1536]`, so its tuner row is `M,32768,1536`.

#### Native searches at the two requested M values

The recorded searches used CK/FlyDSL for ordinary weights and CK/ASM/FlyDSL for packed weights. The current `--libtype all` examples also include eligible CKTile/OPUS candidates, so candidate counts and winners can differ. At M=8,192 the tuner profiled 25 ordinary CK and FlyDSL candidates for each projection. For packed weights it profiled 82, 63 and 81 CK, ASM and FlyDSL candidates in the customer’s projection order. At M=131,072 the search covered FlyDSL only, with six ordinary and 15 packed candidates per projection. The searches used three warmups and ten iterations. All 12 winners were replayed through the public operator on the same MI300X (GPU 0). Each public median uses nine HIP-event samples with three full GEMMs per timed graph. [Tuning commands](#22-tune-an-fp8-block-scaled-gemm) · [Candidates per search, selected kernels, latencies, timing samples and errors](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/blockscale-customer-native.csv)

| M, N, K                 | Weight layout | Native winner    | Tuner TFLOP/s | Public median µs | Public TFLOP/s |
| ----------------------- | ------------- | ---------------- | ------------: | ----------------: | -------------: |
| `(8192,1536,4096)`    | ordinary      | FlyDSL PF1 ID 15 |         718.3 |            134.48 |          766.5 |
| `(8192,1536,4096)`    | packed        | FlyDSL PF2 ID 22 |         934.6 |            125.42 |          821.9 |
| `(8192,32768,1536)`   | ordinary      | FlyDSL PF1 ID 15 |         914.5 |            983.17 |          838.8 |
| `(8192,32768,1536)`   | packed        | FlyDSL PF2 ID 20 |         985.8 |            964.45 |          855.0 |
| `(8192,4096,32768)`   | ordinary      | FlyDSL PF1 ID 15 |       1,134.1 |          2,176.04 |        1,010.6 |
| `(8192,4096,32768)`   | packed        | FlyDSL PF2 ID 21 |       1,210.4 |          2,141.66 |        1,026.8 |
| `(131072,1536,4096)`  | ordinary      | FlyDSL PF1 ID 15 |         989.1 |          1,851.03 |          891.0 |
| `(131072,1536,4096)`  | packed        | FlyDSL PF1 ID 16 |       1,048.3 |          1,750.91 |          941.9 |
| `(131072,32768,1536)` | ordinary      | FlyDSL PF1 ID 15 |         930.8 |         15,707.92 |          840.0 |
| `(131072,32768,1536)` | packed        | FlyDSL PF2 ID 20 |       1,001.1 |         15,389.01 |          857.4 |
| `(131072,4096,32768)` | ordinary      | FlyDSL PF1 ID 15 |       1,097.6 |         36,152.50 |          973.2 |
| `(131072,4096,32768)` | packed        | FlyDSL PF2 ID 22 |       1,237.6 |         33,705.56 |        1,043.9 |

#### Public replay across the full M sweep

The following tables use the saved dispatch choices for all 21 customer shapes. Both layouts were replayed on GPU 0 using the same nine-sample protocol as the native winners. Bold rows mark the two M values in the original request. All rates are public TFLOP/s. [All 42 replays with kernel IDs, timing samples and sampled CPU FP64 errors](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/blockscale-customer-replay.csv)

**Packed weights.** `gemm_a8w8_blockscale_bpreshuffle` used ASM ID 32 for the first projection at M=2,048 and 4,096, and FlyDSL PF2 for the other 19 shapes. The packed sweep peaked at **1,045.2 TFLOP/s** at `(4096,4096,32768)` with PF2 ID 22. Replaying the PF2 rows requires the included gfx942 PF2 implementation and its matching registry. On a PR #29 installation, use the PF1 + ASM table for the same 21 shapes. [Measured PF2 + ASM table](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/best-pf2.csv) · [PR #29 PF1 + ASM table](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/packed-pf1-asm.csv)

|    M (input rows) | N=1,536, K=4,096 | N=32,768, K=1,536 | N=4,096, K=32,768 |
| ----------------: | ---------------: | ----------------: | ----------------: |
|             2,048 |            615.9 |             818.1 |             964.5 |
|             4,096 |            661.5 |             862.8 |           1,045.2 |
|   **8,192** |  **843.6** |   **856.8** | **1,044.8** |
|            16,384 |            811.5 |             860.9 |           1,042.1 |
|            32,768 |            934.0 |             859.3 |           1,042.1 |
|            65,536 |            918.2 |             850.2 |           1,029.7 |
| **131,072** |  **918.8** |   **848.8** | **1,031.4** |

**Ordinary weights.** `gemm_a8w8_blockscale` used the defaults available at measurement time. The exact recorded selections are preserved in [the ordinary reference table](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/block-ordinary-reference.csv). Current serving defaults may differ after later tuning. The first projection used CK ID 18 at M=2,048 and CK ID 0 at M=4,096. The other 19 shapes used FlyDSL PF1 ID 15.

|    M (input rows) | N=1,536, K=4,096 | N=32,768, K=1,536 | N=4,096, K=32,768 |
| ----------------: | ---------------: | ----------------: | ----------------: |
|             2,048 |            406.6 |             787.7 |             766.3 |
|             4,096 |            567.7 |             812.7 |           1,005.7 |
|   **8,192** |  **771.3** |   **823.9** | **1,002.9** |
|            16,384 |            765.2 |             845.1 |             951.2 |
|            32,768 |            909.9 |             836.2 |             993.1 |
|            65,536 |            865.0 |             833.9 |             976.5 |
| **131,072** |  **876.9** |   **834.5** |   **965.2** |

At `(8192,4096,32768)` the fresh search selected packed PF2 ID 21, which replayed at **1,026.8 TFLOP/s**. The saved table selects packed PF2 ID 22, which replayed at **1,044.8 TFLOP/s** on the same GPU. The CSVs identify the dispatch used for each measurement.

Replay the recorded 21-shape selections with the standalone shape list. Use fresh output filenames. For newly tuned winners, substitute your shape list and generated winner CSV.

```bash
BLOCK_SHAPES="$BENCH/configs/block-shapes.csv"
BLOCK_ORDINARY="$BENCH/configs/block-ordinary-reference.csv"
BLOCK_PACKED="$BENCH/configs/best-pf2.csv"
# For a PR29-only installation without PF2:
# BLOCK_PACKED="$BENCH/configs/packed-pf1-asm.csv"
python "$BENCH/benchmark_fp8_blockscale_gemm.py" --input "$BLOCK_SHAPES" --scaling block128 --layout ordinary \
  --table "$BLOCK_ORDINARY" --output "$WORK/customer-block-ordinary.csv"
python "$BENCH/benchmark_fp8_blockscale_gemm.py" --input "$BLOCK_SHAPES" --scaling block128 --layout packed \
  --table "$BLOCK_PACKED" --output "$WORK/customer-block-packed.csv"
```

To benchmark current serving defaults instead, unset the matching `AITER_CONFIG_*` override and omit `--table`. The canonical runtime CSVs under `aiter/configs/` are maintained separately from these historical reference tables.
