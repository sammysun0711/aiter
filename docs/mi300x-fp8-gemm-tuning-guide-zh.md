# MI300X FP8 GEMM 调优指南

[English](mi300x-fp8-gemm-tuning-guide.md)

本指南介绍如何在单张 **304-CU MI300X（gfx942）** 上调优不含偏置项的 FP8 线性投影，使用 E4M3 FNUZ 输入（`torch.float8_e4m3fnuz`）、FP32 反量化缩放因子和 BF16 输出。

矩阵维度采用 **M,N,K** 表示：**M** 为输入行数（token 数），**N** 为输出特征维度，**K** 为输入特征维度（归约维度）。AITER 接收 `X[M,K]` 和 `W[N,K]`，计算 `Y[M,N] = X @ W.T`。

请根据模型使用的缩放方式选择调优流程。两种调优器都会生成运行时配置 CSV，供公开算子按配置回放：

- 使用 gradlib 调优 FP8 PTPC GEMM。[FP8 PTPC GEMM 调优](#21-调优-fp8-ptpc-gemm)
- 使用 AITER 原生调优器调优 FP8 Blockscale GEMM。[FP8 Blockscale GEMM 调优](#22-调优-fp8-blockscale-gemm)

## 1. AITER FP8 GEMM API 与布局定义

### 1.1. FP8 PTPC GEMM

FP8 PTPC GEMM 采用逐 token / 逐通道缩放：每个 token 对应一个 FP32 激活缩放因子，每个输出通道对应一个 FP32 权重缩放因子；两者共同作用于整个点积结果。请在 GPU 上提供形状为 `[M,1]` 和 `[1,N]` 的连续缩放因子张量。`tgemm.mm` 支持普通布局（ordinary）和预打包布局（packed）的权重。[FP8 PTPC GEMM 内核分派](../aiter/tuned_gemm.py)

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

请保留 `shuffle_weight` 返回的张量及其布局元数据。对于普通布局的权重，将 `W_fp8` 直接传给 `tgemm.mm`。

### 1.2. FP8 Blockscale GEMM

FP8 Blockscale GEMM 采用 K128 分块缩放，每个分块的部分点积都有独立的缩放乘数。激活缩放因子的逻辑形状为 `[M,K/128]`。权重缩放因子按普通行优先顺序存储，形状为 `[ceil(N/128),K/128]`，每个 128×128 权重块对应一个值。下例中的 `sx_block` 和 `sw_block` 初始均为连续、行优先的 FP32 张量。[FP8 Blockscale GEMM 内核分派](../aiter/ops/gemm_op_a8w8.py)

| 权重存储方式                          | 公开 API                             | 激活缩放因子的存储方式                           | 运行时配置表环境变量                              |
| ------------------------------------- | ------------------------------------ | ------------------------------------------------ | ------------------------------------------------- |
| 普通布局 `W[N,K]`                   | `gemm_a8w8_blockscale`             | 行优先 `[M,K/128]`                             | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE`             |
| `shuffle_weight(W, layout=(16,16))` | `gemm_a8w8_blockscale_bpreshuffle` | 字节按 K 分组优先排列，以 `[M,K/128]` 形状传入 | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE` |

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

如果量化器直接生成按 K 分组优先排列的激活缩放因子，可以将其直接传给预打包权重算子。在 gfx942 上，FlyDSL PF1 支持普通权重和预打包权重，分别对应 ID 15 和 ID 16；PF2 ID 17–23 使用预打包权重。调优器会根据实际 M,N,K 检查每个候选内核的分块和描述符约束。[FlyDSL 内核注册表](../aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_blockscale_common.py) · [布局适配器](../aiter/ops/flydsl/gemm_a8w8_blockscale.py)

## 2. 配置用于 FP8 GEMM 调优的 AITER 环境

以下结果使用 ROCm 7.2.0、PyTorch 2.9.1、hipBLASLt 1.2.1 和 FlyDSL 0.5.0。计时应在空闲 GPU 上进行。[AITER 安装说明](installation.rst)

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

## 2.1. 调优 FP8 PTPC GEMM

使用 AITER 的 `gemm_a16w16_tune.py`，并指定 `--libtype hipblaslt --with-hipblaslt`。Gradlib 会查询已安装的 hipBLASLt 库，先进行快速筛选，再对入围候选进行计时和正确性检查。对于 FP8 PTPC GEMM，需要设置 `scaleAB=True`。以下示例分别调优 M=8,192 时采用预打包权重的 `o_proj`，以及扩展测试中 M=2,048 时采用普通权重的 4,096 到 1,536 维投影。[Gradlib 指南](../gradlib/README.md) · [实现方案搜索](../gradlib/gradlib/GemmTuner.py)

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

若要对全部客户形状重新搜索，请使用包含 42 行的输入文件（21 个形状 × 两种布局）。回放选定配置时，运行 `benchmark_fp8_ptpc_gemm.py`，指定 `--input "$BENCH/configs/ptpc-tune-input.csv"` 和 `--table "$WORK/customer-tuned/ptpc.csv"`。一次调用即可测试两种布局，具体命令见第 3 节。

```bash
mkdir -p "$WORK/customer-tuned"
CACHE_INVALIDATE_BUFFERS=1 python "$PTPC_TUNER" \
  --libtype hipblaslt --with-hipblaslt --mp 8 -i "$BENCH/configs/ptpc-tune-input.csv" \
  -o "$WORK/customer-tuned/ptpc.csv" -o2 "$WORK/customer-tuned/ptpc.profile.csv"
```

选定配置中的 `solidx` 标识当前已安装 hipBLASLt 版本中的一个实现方案。更换 GPU 或库版本后，需要重新验证。若要将新入围的 gradlib 候选与现有服务配置进行比较，可使用可选的现有配置对比工具；它会测量两种执行顺序。[FP8 PTPC GEMM 现有配置对比](../op_tests/op_benchmarks/mi300x_fp8_gemm/qualify_ptpc.py)

FP8 PTPC GEMM 回放及现有配置对比采用与原生 hipBLASLt 调优器相同的输入生成方式：随机种子为 0，激活和权重均通过 `torch.randn(...).to(torch.bfloat16)` 生成，然后分别调用 `aiter.pertoken_quant(..., quant_dtype=torch.float8_e4m3fnuz)`。返回的缩放因子分别以 `[M,1]` 和 `[1,N]` 形状传入。量化与权重预打包均在计时前完成。

## 2.2. 调优 FP8 Blockscale GEMM

使用 AITER 原生的 `gemm_a8w8_blockscale_tune.py`，搜索三种客户投影在 M=8,192 和 M=131,072 时的配置。该工具将满足条件的候选交给 `mp_tuner`，与 Torch 参考结果进行比较，并为每种 M,N,K 和权重布局写出选定配置。以下示例使用 `--libtype all`，按布局搜索符合条件的 FlyDSL、CK、CKTile、ASM 和 OPUS 候选。若需限制搜索后端，可在每次调用中指定一个后端，并为每个后端使用独立的输出文件。[原生调优器](../csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py)

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

使用 `--shape_grouped` 时，每个工作进程一次处理一个形状；对于这份输入，每次调用最多可同时使用六张 GPU。`-o` 输出选定的运行时配置行，`-o2` 输出候选性能记录。使用 `--libtype flydsl` 可仅搜索 FlyDSL，使用 `--libtype all` 可搜索所有符合条件的后端。预打包权重的调优和原生回放都需要 `--preshuffle`。

若要对通过调优器筛选的每个原生候选增加带正负值输入的 CPU FP64 参考校验和图执行校验，可运行 `qualify_blockscale.py`，并指定 `--input`、`--layout`、`--family` 和 `--output`。它生成的配套性能记录包含采样误差及被拒绝候选的信息。[带正负值输入的验证工具](../op_tests/op_benchmarks/mi300x_fp8_gemm/qualify_blockscale.py)

### 完整遍历客户形状集合

使用 `configs/block-shapes.csv` 覆盖全部 **21 个 M,N,K 形状**，包括从 2,048 到 131,072 的七个 M 值和三种投影。FP8 Blockscale GEMM 的权重布局由 `--preshuffle` 选择，因此两次搜索可覆盖 **42 个形状与布局组合**。两份各含 21 行的选定配置分别写入 `$WORK/customer-tuned/block-ordinary.csv` 和 `$WORK/customer-tuned/block-packed.csv`，候选性能记录保存在同一目录。

以下流程直接对完整形状列表使用 `--libtype all`。对于 gfx942 OPUS 候选，如果 A 或 C 的最后一个行分块偏移超过有符号 int32 的范围，原生调优器会在为该候选分配内存或启动内核之前跳过它；同一形状的其他后端仍可参与搜索。

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

回放使用完整的原始形状列表。`block-ordinary-verified.csv` 和 `block-packed-verified.csv` 都应包含 **21 行通过验证的结果**。

## 3. 使用调优配置对选定候选进行基准测试

**FP8 PTPC GEMM。** 使用 `benchmark_fp8_ptpc_gemm.py` 回放第 2.1 节生成的完整 PTPC 配置表。它通过 `tgemm.mm` 验证普通权重和预打包权重，并在配对轮次中进行基准测试，计时设置与第 4.1 节一致。[PTPC 公开 API 基准测试](../op_tests/op_benchmarks/mi300x_fp8_gemm/benchmark_fp8_ptpc_gemm.py)

```bash
PTPC_TABLE="$WORK/customer-tuned/ptpc.csv"
python "$BENCH/benchmark_fp8_ptpc_gemm.py" \
  --input "$BENCH/configs/ptpc-tune-input.csv" --table "$PTPC_TABLE" \
  --rounds 8 --groups 3 --ops 3 --output "$WORK/ptpc-replay"
```

该命令在 `$WORK/ptpc-replay` 下按形状生成 21 个 JSON 文件，每个文件都包含两种布局、选定的实现方案 ID、数值误差和计时样本。

**FP8 Blockscale GEMM。** 使用第 2.2 节完整客户形状搜索生成的 `block-ordinary.csv` 和 `block-packed.csv`，每份包含 21 个 M,N,K 配置。仓库也提供了最新验证快照：[普通布局](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/block-ordinary.csv)和[预打包布局](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/block-packed.csv)。这些配置是本次原生搜索中通过检查且耗时最短的候选，也已通过公开 API 回放验证。

每种布局都应在新进程中运行。基准测试会在导入 AITER 之前设置配置表，检查实际分派是否精确匹配指定配置行，并在即时执行（eager）和图执行中与采样的 CPU FP64 参考结果比较。对于 FlyDSL 配置，还会根据已安装的内核注册表检查内核 ID、完整名称、布局和形状。[公开 API 基准测试](../op_tests/op_benchmarks/mi300x_fp8_gemm/benchmark_fp8_blockscale_gemm.py)

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

FP8 Blockscale GEMM 的公开 API 计时默认采用原生调优器的输入生成方式：随机种子为 0，使用 FP16 的 `torch.rand / 10` 生成数据后转换为 FP8，并使用服从均匀分布的 FP32 K128 缩放因子。

若仅对采用预打包权重的 `o_proj` 测试 FP8 Blockscale GEMM：

```bash
python "$BENCH/benchmark_fp8_blockscale_gemm.py" --shape 8192 4096 32768 \
  --scaling block128 --layout packed --table "$BLOCK_CFG_DIR/block-packed.csv" \
  --output "$WORK/o-proj-replay.csv"
```

也可使用 `python "$BLOCK_TUNER" --preshuffle --run_config "$BLOCK_CFG_DIR/block-packed.csv"` 或 `python "$PTPC_TUNER" --run_config "$PTPC_TABLE"`，通过原生工具与 Torch 参考结果进行回放验证：[预打包 CK 内核的构建与回放](../csrc/ck_gemm_a8w8_blockscale_bpreshuffle/README.md)。

**请注意** 模型部署服务时，应在导入 AITER **之前**导出已验证运行时配置表的绝对路径。配置表应覆盖服务所需的全部形状和布局。JSON 和 `*-replay.csv` 输出是测量报告；运行时分派应使用选定配置表。

```bash
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE="$BLOCK_CFG_DIR/block-ordinary.csv"
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE="$BLOCK_CFG_DIR/block-packed.csv"
# If the full PTPC table has also been tuned and verified, enable it separately:
# export AITER_CONFIG_GEMM_BF16="$PTPC_TABLE"
```

提供的快照用于覆盖特定工作负载的配置。客户快照及其测量结果对应完整原生调优流程选出的配置。

## 4. 通过AITER API 对选定内核进行基准测试

客户形状的结果表使用已保存的分派配置，报告 AITER  Public API算子的中位吞吐率。本次原生搜索也从各选定配置 CSV 中报告**调优器 TFLOP/s**。公开 API 验证使用带正负值的 FNUZ 输入、FP32 缩放因子和 BF16 输出。每一行报告结果都在即时执行和图执行中通过了采样的 CPU FP64 校验。

吞吐率按 `2 × M × N × K / (median_us × 10⁶)` 计算，单位为 TFLOP/s。权重预打包、预打包布局激活缩放因子的准备和 JIT 编译均在公开 API 计时之前完成。普通布局的 FlyDSL 调用则包含激活缩放因子的转置操作。

### 4.1. FP8 PTPC GEMM

#### 对客户形状进行基准测试

完整客户形状测试采用与调优器一致的 `gradlib_ptpc` 输入生成方式：随机种子为 0，激活和权重均通过 `pertoken_quant` 量化。预打包布局的 `tgemm.mm` 在 `(32768,4096,32768)` 上达到 **1,231.1 TFLOP/s**，在参考 `o_proj` 形状 `(8192,4096,32768)` 上达到 **1,190.3 TFLOP/s**。测试在同一台 304-CU MI300X 系统的 GPU 0 上执行，覆盖全部 21 个客户形状，并使用已保存的 hipBLASLt 实现方案 ID。普通权重和预打包权重均在即时执行和图执行中通过了采样的 CPU FP64 校验。每个吞吐率取 24 个 HIP 事件计时样本的中位数：八轮，每轮三个样本，每个计时图包含三次完整 GEMM 调用。基准测试在配对轮次中反转两种布局的执行顺序。[运行时配置表](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/ptpc.csv) · [全部 42 组公开 API 回放结果，含输入生成方式、实现方案 ID、误差和计时样本](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/ptpc-customer-replay.csv)

**预打包 FP8 PTPC GEMM 吞吐率，单位为 TFLOP/s。** 粗体标出每种投影在不同 M 值下测得的最高吞吐率。各列明确列出 N（输出特征维度）和 K（输入特征维度）；M 为输入行数。AITER 将权重存储为 `[N,K]`。

| M（输入行数） |  N=1,536, K=4,096 | N=32,768, K=1,536 | N=4,096, K=32,768 |
| ------------: | ----------------: | ----------------: | ----------------: |
|         2,048 |             793.9 |             974.2 |           1,151.9 |
|         4,096 |             914.9 |             988.4 |           1,216.5 |
|         8,192 |           1,075.5 |             982.9 |           1,190.3 |
|        16,384 |           1,130.8 |             988.4 |           1,218.2 |
|        32,768 | **1,135.4** |             980.0 | **1,231.1** |
|        65,536 |           1,133.1 |   **998.2** |           1,218.6 |
|       131,072 |           1,080.1 |             979.4 |           1,204.9 |

使用相同的输入生成方式和基准测试设置，回放提供的实现方案 ID。

```bash
PTPC_CFG="$BENCH/configs"
python "$BENCH/benchmark_fp8_ptpc_gemm.py" --table "$PTPC_CFG/ptpc.csv" \
  --input "$PTPC_CFG/ptpc-tune-input.csv" \
  --rounds 8 --groups 3 --ops 3 --output "$WORK/ptpc-customer-replay"
```

### 4.2. FP8 Blockscale GEMM

完整客户形状搜索使用 `--libtype all --shape_grouped --batch 8 --mp 8`，对**全部 21 个形状的两种布局**进行调优，预热三次，计时十次。预打包权重的调优还使用 `--preshuffle --splitK`。全部 42 行选定配置都是各自记录中通过检查且耗时最短的候选，**42/42 组公开 API 回放全部通过验证**。这些原生调优选定配置保存在[普通布局表](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/block-ordinary.csv)和[预打包布局表](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/block-packed.csv)中。

使用 `benchmark_fp8_blockscale_gemm.py`，通过公开 API 回放这些已保存的配置。以下命令覆盖每种布局的全部 21 个形状，生成两份验证报告，每份包含 21 行通过验证的结果。[公开 API 基准测试](../op_tests/op_benchmarks/mi300x_fp8_gemm/benchmark_fp8_blockscale_gemm.py)

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

若仅测试下表中的 M=8K/128K 情况，将第一行赋值改为 `BLOCK_SHAPES="$WORK/shapes.csv"`，使用第 2.2 节生成的六行输入；此时每份报告应包含六行通过验证的结果。若要回放自己重新调优得到的配置，将第二行赋值改为 `BLOCK_CFG_DIR="$WORK/customer-tuned"`。

#### 在 M=8K/128K 下测试调优选定的内核

下表列出完整搜索中 M=8,192 和 M=131,072 的结果。**调优器 TFLOP/s** 使用原生性能分析器在各形状所分配 GPU 上的计时结果；**公开 API TFLOP/s** 使用验证工具在逻辑 GPU 0 上通过 HIP 事件测得的图执行时间中位数。FP8 Blockscale GEMM 的公开 API 计时已采用与原生调优器相同的输入生成方式：随机种子为 0，FP16 均匀随机数据除以十后转换为 FP8，缩放因子为服从均匀分布的 FP32 数据。随机种子为 2026 的带正负值输入仍用于独立的正确性检查。原生性能分析会轮换缓冲区，并报告内核执行时长的平均值；公开 API 回放复用图捕获时的输入，并报告图执行时间的中位数。[全部 42 个原生选定配置及候选数量](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/blockscale-customer-native.csv) · [全部 42 组已验证的公开 API 结果](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/blockscale-customer-replay.csv)

| M, N, K                 | 权重布局 | 选定内核         | 调优器 TFLOP/s | 公开 API 中位延迟 µs | 公开 API TFLOP/s |
| ----------------------- | -------- | ---------------- | -------------: | --------------------: | ---------------: |
| `(8192,1536,4096)`    | ordinary | FlyDSL PF1 ID 15 |         714.81 |                127.09 |           811.06 |
| `(8192,1536,4096)`    | packed   | FlyDSL PF2 ID 22 |         929.56 |                115.92 |           889.22 |
| `(8192,32768,1536)`   | ordinary | FlyDSL PF1 ID 15 |         911.70 |                918.86 |           897.45 |
| `(8192,32768,1536)`   | packed   | FlyDSL PF2 ID 23 |         986.90 |                861.68 |           957.01 |
| `(8192,4096,32768)`   | ordinary | FlyDSL PF1 ID 15 |       1,080.44 |              1,957.80 |         1,123.21 |
| `(8192,4096,32768)`   | packed   | FlyDSL PF2 ID 22 |       1,180.81 |              1,829.50 |         1,201.98 |
| `(131072,1536,4096)`  | ordinary | FlyDSL PF1 ID 15 |         988.78 |              1,657.15 |           995.25 |
| `(131072,1536,4096)`  | packed   | FlyDSL PF2 ID 22 |       1,054.79 |              1,582.37 |         1,042.27 |
| `(131072,32768,1536)` | ordinary | FlyDSL PF1 ID 15 |         927.50 |             14,285.22 |           923.62 |
| `(131072,32768,1536)` | packed   | FlyDSL PF2 ID 23 |         991.68 |             13,749.01 |           959.64 |
| `(131072,4096,32768)` | ordinary | FlyDSL PF1 ID 15 |       1,131.03 |             32,030.63 |         1,098.46 |
| `(131072,4096,32768)` | packed   | FlyDSL PF2 ID 18 |       1,227.86 |             29,351.66 |         1,198.72 |

#### 对 M=2K–128K 的客户形状进行完整测试

两种布局均通过公开算子回放，采用九个 HIP 事件计时样本，每个捕获的图包含三次完整 GEMM。保存的验证 CSV 包含中位数、两种输入生成方式的标签，以及计时输入和带正负值验证输入各自的采样数值误差。以下数值均为**公开 API 回放的 TFLOP/s**。粗体行对应上表中的两个 M 值。

**预打包权重。** 选定配置表在 `(2048,1536,4096)` 上使用 ASM ID 32，在 `(4096,1536,4096)` 上使用 CK ID 14，另含 17 行 FlyDSL PF2 和两行 FlyDSL PF1 配置。公开 API 完整测试在 `(16384,4096,32768)` 上使用 FlyDSL PF2 ID 22，达到峰值 **1,212.5 TFLOP/s**。[已验证的预打包布局运行时配置表](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/block-packed.csv)

|     M（输入行数） |  N=1,536, K=4,096 | N=32,768, K=1,536 | N=4,096, K=32,768 |
| ----------------: | ----------------: | ----------------: | ----------------: |
|             2,048 |             692.9 |             904.3 |           1,065.5 |
|             4,096 |             726.4 |             948.6 |           1,200.7 |
|   **8,192** |   **889.2** |   **957.0** | **1,202.0** |
|            16,384 |             901.5 |             973.1 |           1,212.5 |
|            32,768 |           1,049.5 |             970.8 |           1,200.6 |
|            65,536 |           1,045.1 |             963.4 |           1,184.5 |
| **131,072** | **1,042.3** |   **959.6** | **1,198.7** |

**普通权重。** 对于第一种投影，M=2,048 和 M=4,096 均使用 CK ID 0，其余 19 个形状使用 FlyDSL PF1 ID 15。这些都是本次搜索明确选定的配置。[已验证的普通布局运行时配置表](../op_tests/op_benchmarks/mi300x_fp8_gemm/configs/block-ordinary.csv)

|     M（输入行数） | N=1,536, K=4,096 | N=32,768, K=1,536 | N=4,096, K=32,768 |
| ----------------: | ---------------: | ----------------: | ----------------: |
|             2,048 |            571.7 |             825.3 |             819.6 |
|             4,096 |            612.7 |             904.1 |           1,074.5 |
|   **8,192** |  **811.1** |   **897.5** | **1,123.2** |
|            16,384 |            836.8 |             925.8 |           1,068.0 |
|            32,768 |          1,010.6 |             916.2 |           1,138.6 |
|            65,536 |            988.5 |             936.2 |           1,111.1 |
| **131,072** |  **995.2** |   **923.6** | **1,098.5** |

若要测试当前服务默认配置，请取消对应的 `AITER_CONFIG_*` 环境变量覆盖，并省略 `--table`。`aiter/configs/` 下的标准运行时 CSV 与这些已验证的客户快照分别维护。

## 5. 汇总：FP8 PTPC GEMM 与 FP8 Blockscale GEMM

下表比较全部 **21 个客户形状**在**预打包权重**布局下的性能，覆盖从 2,048 到 131,072 的七个 M 值以及三种 N,K 投影。测试系统配备 8 张 304-CU MI300X。

FP8 PTPC GEMM 通过 `tgemm.mm` 使用已保存的 hipBLASLt 实现方案 ID；FP8 Blockscale GEMM 通过 `gemm_a8w8_blockscale_bpreshuffle` 使用原生调优器选定的配置。

百分比按 `(FP8 Blockscale GEMM / FP8 PTPC GEMM) × 100%` 计算，以 **FP8 PTPC GEMM = 100%** 为基准；大于 100% 表示 FP8 Blockscale GEMM 的吞吐率更高。

| M, N, K                           | FP8 PTPC GEMM TFLOP/s | FP8 Blockscale GEMM TFLOP/s | FP8 Blockscale GEMM / FP8 PTPC GEMM（PTPC = 100%） |
| --------------------------------- | --------------------: | --------------------------: | -------------------------------------------------: |
| `(2048,1536,4096)`              |                 793.9 |                       692.9 |                                              87.3% |
| `(2048,32768,1536)`             |                 974.2 |                       904.3 |                                              92.8% |
| `(2048,4096,32768)`             |               1,151.9 |                     1,065.5 |                                              92.5% |
| `(4096,1536,4096)`              |                 914.9 |                       726.4 |                                              79.4% |
| `(4096,32768,1536)`             |                 988.4 |                       948.6 |                                              96.0% |
| `(4096,4096,32768)`             |               1,216.5 |                     1,200.7 |                                              98.7% |
| `(8192,1536,4096)`              |               1,075.5 |                       889.2 |                                              82.7% |
| `(8192,32768,1536)`             |                 982.9 |                       957.0 |                                              97.4% |
| `(8192,4096,32768)`             |               1,190.3 |                     1,202.0 |                                             101.0% |
| `(16384,1536,4096)`             |               1,130.8 |                       901.5 |                                              79.7% |
| `(16384,32768,1536)`            |                 988.4 |                       973.1 |                                              98.4% |
| `(16384,4096,32768)`            |               1,218.2 |                     1,212.5 |                                              99.5% |
| `(32768,1536,4096)`             |               1,135.4 |                     1,049.5 |                                              92.4% |
| `(32768,32768,1536)`            |                 980.0 |                       970.8 |                                              99.1% |
| `(32768,4096,32768)`            |               1,231.1 |                     1,200.6 |                                              97.5% |
| `(65536,1536,4096)`             |               1,133.1 |                     1,045.1 |                                              92.2% |
| `(65536,32768,1536)`            |                 998.2 |                       963.4 |                                              96.5% |
| `(65536,4096,32768)`            |               1,218.6 |                     1,184.5 |                                              97.2% |
| `(131072,1536,4096)`            |               1,080.1 |                     1,042.3 |                                              96.5% |
| `(131072,32768,1536)`           |                 979.4 |                       959.6 |                                              98.0% |
| `(131072,4096,32768)`           |               1,204.9 |                     1,198.7 |                                              99.5% |
| **几何平均值（21 个形状）** |                   N/A |                         N/A |                                    **93.8%** |

完整测试中，FP8 PTPC GEMM 在 21 个形状中的 20 个上取得了更高的实测吞吐率。FP8 Blockscale GEMM 在 `(8192,4096,32768)` 上达到 FP8 PTPC GEMM 吞吐率的 101.0%；最低比例为 `(4096,1536,4096)` 上的 79.4%。百分比和几何平均值均使用未经四舍五入的 [FP8 PTPC GEMM 结果](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/ptpc-customer-replay.csv)与[FP8 Blockscale GEMM 结果](../op_tests/op_benchmarks/mi300x_fp8_gemm/results/blockscale-customer-replay.csv)计算。最后一行是 21 个形状吞吐率比值的几何平均值，每个形状的权重相同。

两种测试都采用 HIP 事件进行图执行计时，每个图包含三次完整 GEMM。FP8 PTPC GEMM 使用 `gradlib_ptpc` 输入，报告 24 个样本的中位数；FP8 Blockscale GEMM 使用 `native_blockscale` 输入，报告九个样本的中位数。两次独立测试分别采用与各自调优器一致的输入，因此比值同时反映输入生成方式和测量方法的差异。应根据模型的量化方式选择适用的 API。
