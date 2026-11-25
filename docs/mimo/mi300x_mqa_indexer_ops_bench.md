# MI300X（gfx942）MQA indexer 与稀疏注意力算子指南（FP8）

DeepSeek-V3.2 / DeepSeek-V4 风格的稀疏注意力模型在 MI300X 上用到的 aiter 算子：怎么安装、调优、测试，实测性能和推荐的 kernel。只覆盖 FP8 路径：FP8 blockscale 权重、FP8 indexer（Q 和 index-K cache）；KV cache 为 BF16。

## 1. 准备 docker 镜像

```bash
docker run -it --name aiter-mqa-indexer-gfx942 --ipc=host --network=host --privileged --security-opt seccomp=unconfined \
--cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE --device=/dev/kfd --device=/dev/dri --device=/dev/mem  -v $HOME:/root/workspace  \
rocm/sgl-dev:v0.5.11-rocm720-mi30x-20260510
```

再次进入：

```bash
docker start aiter-mqa-indexer-gfx942
docker exec -it aiter-mqa-indexer-gfx942 bash
```

## 2. 环境安装

### 清理镜像自带的 aiter 和 FlyDSL

```bash
pip uninstall amd-aiter flydsl -y
rm -rf /sgl-workspace/aiter
```

### 安装 aiter 和依赖

aiter 用上游 `ROCm/aiter` 较新的 main：下面固定的 commit 是截至 2026-11-23（06:50 UTC）main 上的最新提交，本指南的数据都在它上面测得。

```bash
cd /sgl-workspace
git clone https://github.com/ROCm/aiter.git
cd aiter
git checkout bf5994891a222979b43975e45ebe72cbf3016abe
git submodule sync && git submodule update --init --recursive
pip install -r requirements.txt
./.github/scripts/install_triton.sh
AITER_USE_SYSTEM_TRITON=1 python3 -m pip install -e .
```

- `requirements.txt` 装 FlyDSL 0.5.3 等依赖，`install_triton.sh` 装 AMD 源的 Triton 3.8.0。镜像自带的 aiter、FlyDSL 0.3.2、Triton 3.6.0 会被替换，要保留原环境就另起一个容器。
- kernel 不在安装时编译：C++ 模块在首次调用时现场编译（每个模块几秒到 6 分钟），Triton / Gluon / FlyDSL 核按形状首次 JIT。

### 工作目录

```bash
export AITER_ROOT=/sgl-workspace/aiter
export BENCH_DIR=/sgl-workspace/mqa_indexer_bench
export HIP_VISIBLE_DEVICES=0      # 调优和测试都只用一张卡
export CUDA_VISIBLE_DEVICES=0
# 多卡机器上要用另一张卡：再设 ROCR_VISIBLE_DEVICES=<卡号>，上面两个变量保持 0
mkdir -p ${BENCH_DIR}/tuning
```

把本指南里的文件按代码块上方给出的名字保存到 `${BENCH_DIR}`：

| 文件 | 出处 |
|---|---|
| `tuning/*.csv`（调优输入、调优结果、测试用的形状列表） | 4.2、4.4、5.2 节 |
| `prebuild_gemm_tune_modules.py`（可选，GEMM 调优前的预编译） | 附录 |

## 3. Version check list

```bash
cat /opt/rocm/.info/version
git -C ${AITER_ROOT} rev-parse HEAD
git -C ${AITER_ROOT} submodule status
python3 - <<'EOF'
import importlib.metadata as m
import torch
import aiter
props = torch.cuda.get_device_properties(0)
print("torch", torch.__version__)
for name in ("amd-aiter", "triton", "triton-kernels", "flydsl"):
    print(name, m.version(name))
print(torch.cuda.get_device_name(0), props.gcnArchName.split(":")[0], props.multi_processor_count, "CU")
EOF
```

| 项 | 期望值 |
|---|---|
| ROCm | 7.2.0 |
| GPU | AMD Instinct MI300X，gfx942，304 CU |
| PyTorch | 2.9.1+rocm7.2.0.git7e1940d4 |
| aiter commit | `bf5994891a222979b43975e45ebe72cbf3016abe` |
| amd-aiter | 0.1.31.dev20+gbf5994891 |
| composable_kernel 子模块 | `f93c029e09f67c983696eecac8e0e0350f3dbb54` |
| triton | 3.8.0+amd.rocm7.2.0.git7cb7b059 |
| triton-kernels | 1.0.0+amd.rocm7.2.0.git7cb7b059 |
| flydsl | 0.5.3 |

## 4. DeepSeek-V3.2

形状：hidden 7168，q_lora_rank 1536；indexer 64 头 × 128 维，top-k 2048；attention 128 头（TP8 时每卡 16 头），QK 576 维、V 512 维。index-K cache 每 token 132 字节（128 个 FP8 + 1 个 fp32 scale），块大小 64，16×16 preshuffle。

### 4.1 算子

| 阶段 | aiter 入口 | gfx942 上的后端 | 调优 |
|---|---|---|---|
| `wq_b`、`wk`（FP8 blockscale 线性层） | `aiter.get_hip_quant(QuantType.per_1x128)` + `aiter.gemm_a8w8_blockscale_bpreshuffle` | CK / CKTile / ASM / OPUS | 官方 tuner |
| indexer 准备（Q 的 RoPE、FP8 量化；K 的 LayerNorm、RoPE、FP8 量化、写 cache） | `aiter.ops.cache.indexer_qk_rope_quant_and_cache` | HIP 融合核 | 无 |
| K gather（prefill 打分前） | `aiter.ops.cache.cp_gather_indexer_k_quant_cache` | HIP | 无 |
| 打分，prefill | `aiter.mqa_logits.mqa_logits` | Triton（默认）/ FlyDSL | 官方 tuner |
| 打分，decode | `aiter.paged_mqa_logits.paged_mqa_logits` | Gluon | 官方 tuner |
| top-k，prefill | `aiter.top_k_per_row_prefill` | HIP radix | 无 |
| top-k，decode | `aiter.top_k_per_row_decode` | HIP radix / FlyDSL | 无 |
| 稀疏注意力（BF16 KV） | `aiter.mla.mla_decode_fwd` | ASM | 无 |

`weights_proj`（64 × 7168）是 BF16 线性层，不属于 FP8 路径，用 `torch.nn.functional.linear`。

### 4.2 调优

上游的调优表里没有 MI300X（gfx942、304 CU）的行，prefill 打分、decode 打分、FP8 blockscale GEMM 三项都要调。本节末尾给出了在 MI300X、上述 aiter commit 上调出的表，可以直接保存使用；换了 GPU 型号或 aiter 版本再按下面的命令重调。

**调优输入**（形状列表）

`tuning/mi300x_dsv32_untuned_mqa_logits.csv`（prefill 打分：query 行数 × key 数）：

```csv
heads,head_dim,seq_len,seq_len_kv
64,128,4096,4096
64,128,8192,8192
64,128,16384,16384
64,128,16384,32768
64,128,16384,65536
64,128,16384,131072
```

`tuning/mi300x_dsv32_paged_mqa_logits_untuned.csv`（decode 打分：每个 batch 在三个上下文上调）：

```csv
batch_size,next_n,heads,head_dim,kv_block_size,preshuffle,context_len
1,1,64,128,64,True,4096
1,1,64,128,64,True,32768
1,1,64,128,64,True,262144
4,1,64,128,64,True,4096
4,1,64,128,64,True,32768
4,1,64,128,64,True,262144
8,1,64,128,64,True,4096
8,1,64,128,64,True,32768
8,1,64,128,64,True,262144
32,1,64,128,64,True,4096
32,1,64,128,64,True,32768
32,1,64,128,64,True,262144
64,1,64,128,64,True,4096
64,1,64,128,64,True,32768
64,1,64,128,64,True,262144
128,1,64,128,64,True,4096
128,1,64,128,64,True,32768
128,1,64,128,64,True,262144
256,1,64,128,64,True,4096
256,1,64,128,64,True,32768
256,1,64,128,64,True,262144
```

`tuning/mi300x_dsa_indexer_a8w8_blockscale_bpreshuffle_untuned_gemm.csv`（V3.2 的 `wq_b` 8192×1536、`wk` 128×7168 和 V4 的 `wq_b` 8192×1024，一次调完）：

```csv
M,N,K
1,8192,1536
4,8192,1536
8,8192,1536
32,8192,1536
64,8192,1536
128,8192,1536
256,8192,1536
4096,8192,1536
8192,8192,1536
16384,8192,1536
1,128,7168
4,128,7168
8,128,7168
32,128,7168
64,128,7168
128,128,7168
256,128,7168
4096,128,7168
8192,128,7168
16384,128,7168
1,8192,1024
4,8192,1024
8,8192,1024
32,8192,1024
64,8192,1024
128,8192,1024
256,8192,1024
4096,8192,1024
8192,8192,1024
16384,8192,1024
```

**调优命令**

```bash
cd ${AITER_ROOT}

# prefill 打分，约 4 分钟
python3 csrc/mqa_logits/mqa_logits_tune.py \
  -i ${BENCH_DIR}/tuning/mi300x_dsv32_untuned_mqa_logits.csv \
  -o ${BENCH_DIR}/tuning/mi300x_dsv32_tuned_mqa_logits.csv --all -v

# decode 打分，不到 1 分钟
python3 csrc/paged_mqa_logits/paged_mqa_logits_tune.py \
  -i ${BENCH_DIR}/tuning/mi300x_dsv32_paged_mqa_logits_untuned.csv \
  -o ${BENCH_DIR}/tuning/mi300x_dsv32_paged_mqa_logits_tuned.csv --max_model_len 524288 --all -v

# FP8 blockscale GEMM：先预编译 tune 模块（可选，附录的脚本，约 7 分钟），再调优（约 8 分钟）
python3 ${BENCH_DIR}/prebuild_gemm_tune_modules.py
python3 csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py --preshuffle -k \
  -i ${BENCH_DIR}/tuning/mi300x_dsa_indexer_a8w8_blockscale_bpreshuffle_untuned_gemm.csv \
  -o ${BENCH_DIR}/tuning/mi300x_dsa_indexer_a8w8_blockscale_bpreshuffle_tuned_gemm.csv \
  --libtype all --mp 1 --all -v
```

- `--all`：输出表里已有这些形状时也重调；不加的话 tuner 认为没有待调的形状，直接结束。
- `--max_model_len`：计时时传给 `paged_mqa_logits` 的 `max_model_len`（logits 的行宽），tuner 实际用的是它和上下文两者中较大的一个，默认 131072。入口不在 host 侧读上下文长度，而是按 `max_model_len` 决定每行启动多少个 workgroup，所以要取部署时推理引擎传入的值，即模型的最大上下文长度。这里按 512K 取 524288。
- GEMM：`--preshuffle` 调的是 `gemm_a8w8_blockscale_bpreshuffle`，`-k` 包含 split-K 候选，`--libtype all` 覆盖 CK / CKTile / ASM / OPUS，`--mp 1` 只用一张卡。不预编译的话 tuner 会在 worker 进程里单线程编译 tune 模块，要慢得多（一次尝试约 2 小时）。
- prefill 的调优表最长到 16384 × 128K，更长的上下文 aiter 自动沿用 128K 那一行（16384 × 256K 以上超出了 tuner 做数值比对时的显存）。

**调优结果**（MI300X，上述 aiter commit；`us` 是 tuner 测得的耗时，单位 µs）

`tuning/mi300x_dsv32_tuned_mqa_logits.csv`：

```csv
gfx,cu_num,heads,head_dim,seq_len,seq_len_kv,backend,variant,us,triton_us,flydsl_us
gfx942,304,64,128,4096,4096,flydsl,mfma_r4_w4,212.0,530.39,208.43
gfx942,304,64,128,8192,8192,flydsl,mfma_r4_w2,694.13,2055.29,692.64
gfx942,304,64,128,16384,16384,flydsl,mfma_r4_w2,2632.7,8176.86,2630.0
gfx942,304,64,128,16384,32768,flydsl,mfma_r4_w2,7571.53,24866.48,7574.06
gfx942,304,64,128,16384,65536,flydsl,mfma_r4_w2,18123.93,58161.98,18148.18
gfx942,304,64,128,16384,131072,flydsl,mfma_r4_w4,37786.77,124783.91,37758.75
```

`tuning/mi300x_dsv32_paged_mqa_logits_tuned.csv`：

```csv
gfx,cu_num,batch_size,next_n,heads,head_dim,kv_block_size,preshuffle,context_len,backend,ChunkK,WavePerEU,wg_per_cu,us,gluon_us,flydsl_us
gfx942,304,1,1,64,128,64,True,4096;32768;262144,gluon,128,4,0,6.1,6.1,
gfx942,304,4,1,64,128,64,True,4096;32768;262144,gluon,128,2,0,11.78,11.78,
gfx942,304,8,1,64,128,64,True,4096;32768;262144,gluon,128,4,0,18.18,18.18,
gfx942,304,32,1,64,128,64,True,4096;32768;262144,gluon,256,4,0,53.67,53.67,
gfx942,304,64,1,64,128,64,True,4096;32768;262144,gluon,256,4,0,101.32,101.32,
gfx942,304,128,1,64,128,64,True,4096;32768;262144,gluon,256,4,0,193.84,193.84,
gfx942,304,256,1,64,128,64,True,4096;32768;262144,gluon,256,4,0,377.49,377.49,
```

`tuning/mi300x_dsa_indexer_a8w8_blockscale_bpreshuffle_tuned_gemm.csv`：

```csv
gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio
gfx942,304,1,128,7168,asm,7,8,5.5212,_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_32x128E,0.33,167.52,0.0
gfx942,304,4,128,7168,asm,7,8,5.6879,_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_32x128E,1.29,166.53,0.0039
gfx942,304,8,128,7168,asm,7,8,5.8434,_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_32x128E,2.51,167.18,0.002
gfx942,304,32,128,7168,asm,7,8,6.1959,_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_32x128E,9.48,186.43,0.009
gfx942,304,64,128,7168,asm,7,8,6.495,_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_32x128E,18.08,214.42,0.0093
gfx942,304,128,128,7168,asm,15,8,7.1285,_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_48x128E,32.95,262.02,0.0051
gfx942,304,256,128,7168,asm,15,8,7.9773,_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_48x128E,58.89,353.26,0.0049
gfx942,304,4096,128,7168,asm,10,3,18.4019,_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_48x128E,408.45,1702.34,0.0002
gfx942,304,8192,128,7168,asm,17,2,33.0041,_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_64x128E,455.47,1870.52,0.0
gfx942,304,16384,128,7168,asm,16,1,52.9739,_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_64x128E,567.54,2313.45,0.0
gfx942,304,1,8192,1024,cktile,26,0,5.4965,a8w8_blockscale_cktile_16x64x512_1x4x1_16x16x64_intrawave_0x1x0_2,3.05,1529.34,0.0
gfx942,304,1,8192,1536,ck,7,0,6.1912,a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1,4.06,2035.28,0.0
gfx942,304,4,8192,1024,ck,7,0,5.5931,a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1,12.0,1512.26,0.0
gfx942,304,4,8192,1536,ck,7,0,6.278,a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1,16.03,2015.7,0.0
gfx942,304,8,8192,1024,ck,7,0,5.5526,a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1,24.17,1535.83,0.0
gfx942,304,8,8192,1536,ck,7,0,7.1892,a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1,28.0,1770.19,0.0
gfx942,304,32,8192,1024,ck,12,0,6.2958,a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1,85.27,1420.89,0.0
gfx942,304,32,8192,1536,ck,7,0,7.3608,a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1,109.4,1787.35,0.0
gfx942,304,64,8192,1024,ck,7,0,6.458,a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1,166.27,1471.46,0.0
gfx942,304,64,8192,1536,ck,6,0,7.5021,a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1,214.69,1830.13,0.0
gfx942,304,128,8192,1024,ck,12,0,7.8591,a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1,273.25,1350.9,0.0
gfx942,304,128,8192,1536,asm,0,1,9.9693,_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_32x128E,323.11,1492.25,0.0
gfx942,304,256,8192,1024,asm,10,1,11.3107,_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_64x128E,379.73,1135.66,0.0
gfx942,304,256,8192,1536,asm,10,1,13.8352,_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_64x128E,465.66,1241.07,0.0
gfx942,304,4096,8192,1024,opus,0,0,94.8403,opus_gemm_gfx942_a8w8_bs_bpreshuf_sb_tailm_v16_512x128x128x128_4x2_16x16x32_1x128x128,724.58,840.27,0.0026
gfx942,304,4096,8192,1536,opus,0,0,131.9601,opus_gemm_gfx942_a8w8_bs_bpreshuf_sb_tailm_v16_512x128x128x128_4x2_16x16x32_1x128x128,781.14,651.59,0.0028
gfx942,304,8192,8192,1024,opus,0,0,179.8011,opus_gemm_gfx942_a8w8_bs_bpreshuf_sb_tailm_v16_512x128x128x128_4x2_16x16x32_1x128x128,764.39,839.79,0.0038
gfx942,304,8192,8192,1536,opus,0,0,248.575,opus_gemm_gfx942_a8w8_bs_bpreshuf_sb_tailm_v16_512x128x128x128_4x2_16x16x32_1x128x128,829.36,641.19,0.0035
gfx942,304,16384,8192,1024,opus,0,0,356.6765,opus_gemm_gfx942_a8w8_bs_bpreshuf_sb_tailm_v16_512x128x128x128_4x2_16x16x32_1x128x128,770.66,823.16,0.0038
gfx942,304,16384,8192,1536,opus,0,0,481.5274,opus_gemm_gfx942_a8w8_bs_bpreshuf_sb_tailm_v16_512x128x128x128_4x2_16x16x32_1x128x128,856.27,635.86,0.0037
```

### 4.3 启用调优表

**方式一（推荐）：合并进 aiter 的默认配置。** 把调优表复制到 `aiter/configs/model_configs/`，aiter 加载时会把这个目录下的表和默认表自动合并，不需要设任何环境变量：

```bash
cp ${BENCH_DIR}/tuning/mi300x_dsv32_tuned_mqa_logits.csv ${AITER_ROOT}/aiter/configs/model_configs/
cp ${BENCH_DIR}/tuning/mi300x_dsv32_paged_mqa_logits_tuned.csv ${AITER_ROOT}/aiter/configs/model_configs/
cp ${BENCH_DIR}/tuning/mi300x_dsa_indexer_a8w8_blockscale_bpreshuffle_tuned_gemm.csv ${AITER_ROOT}/aiter/configs/model_configs/
```

- 文件名不要改：aiter 靠名字里的 `tuned_mqa_logits`、`paged_mqa_logits_tuned`、`a8w8_blockscale_bpreshuffle_tuned_gemm` 找到对应的表。
- 同一个形状只能出现在一张表里。重复时 aiter 加载会报错，并改写 `model_configs/` 里的文件、每个形状只留耗时小的一行；遇到这种情况，删掉多余的表，再把上面的文件重新复制一遍。
- 分页打分表按 batch 查、不分上下文，V3.2 和 V4 的两张分页表互相重复，只能放一张（V4 见 5.3）。
- aiter 把合并后的表写在系统临时目录的 `aiter_configs/` 下（方式二也一样）。同一个容器里还有另一份 aiter 在用时，两边写的是同一个文件，这种情况下给自己的进程设一个单独的 `TMPDIR`。

**方式二：环境变量。** 每个算子一个变量；设了之后 aiter 只读它指向的表，不再合并默认表。打分算子在路径写错时不报错，只是回到默认配置，所以设完要按下面的办法确认。

```bash
export AITER_CONFIG_MQA_LOGITS=${BENCH_DIR}/tuning/mi300x_dsv32_tuned_mqa_logits.csv
export AITER_CONFIG_PAGED_MQA_LOGITS=${BENCH_DIR}/tuning/mi300x_dsv32_paged_mqa_logits_tuned.csv
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE=${BENCH_DIR}/tuning/mi300x_dsa_indexer_a8w8_blockscale_bpreshuffle_tuned_gemm.csv
```

**重建 GEMM 生产模块。** GEMM 的 kernel 实例是按构建时生效的调优表生成的，两种方式下换表后都要删掉旧模块，下次调用时自动按新表编译（两个模块约 1 分钟和 5.5 分钟）：

```bash
cd ${AITER_ROOT}/aiter/jit
rm -f module_gemm_a8w8_blockscale_bpreshuffle.so module_gemm_a8w8_blockscale_bpreshuffle_cktile.so
rm -rf build/module_gemm_a8w8_blockscale_bpreshuffle build/module_gemm_a8w8_blockscale_bpreshuffle_cktile
```

**确认调优表已生效**

```bash
python3 - <<'EOF'
import aiter
import aiter.mqa_logits as mql
import aiter.paged_mqa_logits as pmql
from aiter.ops.gemm_op_a8w8 import AITER_CONFIGS, get_CKGEMM_config

cfg = mql.get_mqa_logits_config(16384, 131072, 64, 128)
print("mqa_logits :", cfg["backend"], cfg["variant"])
cfg = pmql.get_paged_mqa_logits_config(1, 1, 64, 128, 64, True)
print("paged_mqa  :", cfg["backend"], cfg["ChunkK"], cfg["WavePerEU"])
gemm = get_CKGEMM_config(64, 128, 7168, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE)
if gemm:
    print("gemm       :", gemm["libtype"], gemm["kernelName"], gemm["splitK"])
else:
    print("gemm       : no tuned row")
EOF
```

期望输出的最后三行（即 4.2 调优表里对应的行；两种启用方式下都是这三行）：

```
mqa_logits : flydsl mfma_r4_w4
paged_mqa  : gluon 128 4
gemm       : asm _ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_32x128E 8
```

没有生效时，这三行分别是 `triton`（变体为空）、`gluon 256 2` 和 `no tuned row`。

### 4.4 测试

都用 aiter 自带的工具直接调用：算子单测（`op_tests/`）和官方 tuner 的 `--run_config`。单测把算子的输出同 torch 参考比对并计时，结束时打印汇总表：误差列是 0 或 1e-4 以下的舍入差（top-k 是 `all_close` 为 True），耗时列是单次调用的 kernel 耗时（µs）。4.5 的数据就是这一节的命令在 4.3 启用调优表之后的输出。

**indexer 准备、top-k、稀疏注意力**

```bash
cd ${AITER_ROOT}

# 融合算子 indexer_qk_rope_quant_and_cache，线性层之后、打分之前的准备一次做完：
#   Q：RoPE → FP8 量化（head 权重同时乘上 Q 的 scale）
#   K：LayerNorm → RoPE → FP8 量化 → 写入 index-K cache
#   --mode fp8     只跑输出为 FP8 的用例（不加会连 fp4 / opus / e2e 一起跑）
#   --num_heads    indexer 的头数
#   -n             token 数。脚本默认只给一半的 token 分配 cache 槽位（--valid_fraction 0.5），
#                  所以写入 cache 的 K 是 token 数的一半
python3 op_tests/test_indexer_qk_rope_quant_and_cache.py --mode fp8 --num_heads 64 -n 1 4 8 32 64 128 256 4096 8192 16384

# 这个脚本测两个算子：
#   indexer_k_quant_and_cache：K 的 FP8 量化 → 写入 index-K cache。这是上面融合算子里 K 的最后两步，
#     单独拿出来给"K 的 LayerNorm、RoPE 由调用方自己做"的写法用；用了上面的融合算子就不再需要它
#   cp_gather_indexer_k_quant_cache：prefill 打分前，把分页 cache 里的 K 和 scale 取成连续的 buffer
#   -b   cache 的块大小（每块的 token 数），DeepSeek 用 64
#   -p   加测 preshuffle 布局（打分核读的就是这种布局）
#   -g   加测 cp_gather_indexer_k_quant_cache
#   -m   token 数，脚本最多支持 16384
python3 op_tests/test_indexer_k_quant_and_cache.py -b 64 -p -g -m 1 4 8 32 64 128 256 4096 8192 16384

# top-k，decode 入口。每个形状把 top-k 的两种模式都测一遍（入口的 stable 参数），汇总表里各占一行：
#   默认（stable=False）   只保证选出分数最高的 k 个；输出下标的先后顺序、分数相同的 key 取哪个都不固定
#   stable（stable=True）  结果是确定的：下标从小到大输出，分数相同时取下标小的。
#                          tensor-parallel 部署时各 rank 要选出并排出同一组 KV，用这种模式
#   -k   每行选出的个数
#   -n   每条序列的 query 行数（next_n），1 表示没有推测 token
#   -b   decode 的 batch
#   -c   上下文长度，即每行参与排序的 key 数
#   --regressions   后面不带值：跳过脚本附带的回归用例（与 -c 无关，第一条命令已经跑过）
python3 op_tests/test_topk_per_row.py -k 2048 -n 1 -b 1 4 8 32 64 128 256 -c 4096 32768
python3 op_tests/test_topk_per_row.py -k 2048 -n 1 -b 1 4 8 32 64 128 256 -c 131072 --regressions

# top-k，prefill 入口：一个 chunk 的每个 query token 一行，第 i 行对它之前的全部 key 排序
#   -c             这里是 chunk 的行数
#   --num_prefix   chunk 之前已有的 token 数，可以给多个；行数加上它就是 prompt 长度。不给时为 0
#   -b 1           decode 的用例只留 batch 1（这几条命令看的是 prefill 的汇总表）
python3 op_tests/test_topk_per_row.py -k 2048 -n 1 -b 1 -c 4096 --regressions
python3 op_tests/test_topk_per_row.py -k 2048 -n 1 -b 1 -c 8192 --regressions
python3 op_tests/test_topk_per_row.py -k 2048 -n 1 -b 1 -c 16384 --num_prefix 0 16384 49152 114688 --regressions
python3 op_tests/test_topk_per_row.py -k 2048 -n 1 -b 1 -c 16384 --num_prefix 245760 507904 --regressions

# 稀疏注意力，非 persistent 调用（直接调 mla_decode_fwd，不传 work metadata）。
# 用的是通用的 MLA 测试：页大小为 1，每条序列的 KV 行在 200 万行的池子里随机取，
# 所以 -c 2048 就是每个 query token 看 2048 行
#   -d     Q 的数据类型
#   -kvd   KV 的数据类型
#   -c     每条序列参与注意力的 KV 行数
#   -b     batch，即序列数
#   -n     "头数,每条序列的 query token 数"，可以给多组；16,1 是 16 头、每条序列 1 个 query token，
#          所以 batch 为 B 时一共 B 个 query token。第二条命令的 batch 是 prefill 一个 chunk 的 token 数
python3 op_tests/test_mla.py -d bf16 -kvd bf16 -c 2048 -b 1 4 8 32 64 128 256 -n 16,1 128,1
python3 op_tests/test_mla.py -d bf16 -kvd bf16 -c 2048 -b 4096 8192 16384 -n 16,1 128,1

# 稀疏注意力，persistent 调用（先用 get_mla_metadata_v1 生成 work metadata，再调 mla_decode_fwd）：
# 每个 query token 在所属序列的上下文里取 2048 行（top-k）
#   -c     每条序列的上下文长度
#   -ms    max_split_per_batch，每个 batch 最多切成几份；可以给多个值，逐个测
#   其余参数同上
python3 op_tests/test_mla_sparse.py -d bf16 -kvd bf16 -c 8192 -b 1 4 8 32 64 128 256 -n 16,1 128,1 -ms 8 16 32 64
python3 op_tests/test_mla_sparse.py -d bf16 -kvd bf16 -c 8192 -b 4096 8192 16384 -n 16,1 128,1 -ms 32
```

- batch 为 16384 的两条稀疏注意力命令，torch 参考要占几十到一百多 GB 显存，跑之前确认这张卡是空的。

**打分。两个脚本要先改两处常量再跑。** `op_tests/test_mqa_logits.py` 和 `op_tests/test_paged_mqa_logits.py` 的形状写在脚本开头的常量里，没有对应的命令行参数：

| 常量 | 原值 | 改成 | 原因 |
|---|---|---|---|
| `HEADS`（两个脚本） | 32 | 64 | DeepSeek-V3.2 的 indexer 是 64 头，调优表也按 head 数查表；不改的话测到的不是 DeepSeek 的形状，也用不到 4.2 调出的行 |
| `MAX_MODEL_LEN`（`test_paged_mqa_logits.py`） | 131072 | 524288 | 这是每次调用传给入口的 `max_model_len`，入口按它决定启动多少个 workgroup（见 4.2），按部署的最大上下文 512K 取；`--kv-len` 不能超过它 |

```bash
cd ${AITER_ROOT}

# 改上表的两处常量，并确认
sed -i 's/^HEADS = 32$/HEADS = 64/' op_tests/test_mqa_logits.py op_tests/test_paged_mqa_logits.py
sed -i 's/^MAX_MODEL_LEN = 131072$/MAX_MODEL_LEN = 524288/' op_tests/test_paged_mqa_logits.py
grep -n '^HEADS\|^MAX_MODEL_LEN' op_tests/test_mqa_logits.py op_tests/test_paged_mqa_logits.py

# prefill 打分
#   -q   每次调用的 query 行数
#   -k   每次调用的 key 数；-q、-k 各给多个值时两两组合，只跑行数不超过 key 数的组合
python3 op_tests/test_mqa_logits.py -q 4096 -k 4096
python3 op_tests/test_mqa_logits.py -q 8192 -k 8192
python3 op_tests/test_mqa_logits.py -q 16384 -k 16384 32768 65536 131072

# decode 打分
#   -b         batch，即每步的序列数
#   --next-n   每条序列的 query 行数，1 表示没有推测 token
#   --kv-len   每条序列的上下文长度（key 数）
python3 op_tests/test_paged_mqa_logits.py -b 1 4 8 32 64 128 256 --next-n 1 --kv-len 4096 32768
python3 op_tests/test_paged_mqa_logits.py -b 1 4 8 32 64 128 256 --next-n 1 --kv-len 262144
```

- `grep` 应显示 `HEADS = 64`（两个文件）和 `MAX_MODEL_LEN = 524288`。
- prefill 打分的汇总表：`picked` 是入口实际选中的后端和变体，应与 4.2 的调优表一致；`mqa_logits us` 是入口的耗时，`triton us` 是 Triton 后端的耗时，也就是没有调优表时的默认配置。
- decode 打分的汇总表：`paged_mqa_logits us` 是入口（查调优表）的耗时，`gluon us` 固定用默认配置 `ChunkK=256, WavePerEU=2`，同一次运行里可以直接对比。`picked` 只显示后端名 `gluon`，具体配置用 4.3 的查表片段确认。
- 恢复原文件：`git checkout op_tests/test_mqa_logits.py op_tests/test_paged_mqa_logits.py`。

**更长的 prefill 和 GEMM：官方 tuner 的 `--run_config`。** 这个模式不调优，只对列表里的每个形状调用生产入口并计时。prefill 打分的 `--run_config` 不做数值比对，所以能测 256K / 512K：16384 × 256K 以上的比对会用尽显存（4.2），`test_mqa_logits.py` 用的是同一套参考实现。GEMM 的 `--run_config` 仍和 torch 参考比对，用它的原因见本节末尾。

`tuning/mi300x_dsv32_long_untuned_mqa_logits.csv`：

```csv
heads,head_dim,seq_len,seq_len_kv
64,128,16384,262144
64,128,16384,524288
```

```bash
cd ${AITER_ROOT}

# prefill 打分：256K 和 512K 的 prompt 的最后一个 chunk
#   --run_config   只测生产入口。这个模式不写 -o，给一个用不到的路径即可
python3 csrc/mqa_logits/mqa_logits_tune.py \
  -i ${BENCH_DIR}/tuning/mi300x_dsv32_long_untuned_mqa_logits.csv \
  -o ${BENCH_DIR}/run_config_unused.csv --run_config --all -v

# FP8 blockscale GEMM：4.2 调优输入里的全部形状（含 V4 的 wq_b）
python3 csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py --preshuffle \
  -i ${BENCH_DIR}/tuning/mi300x_dsa_indexer_a8w8_blockscale_bpreshuffle_untuned_gemm.csv \
  -o ${BENCH_DIR}/run_config_unused.csv --run_config --all -v
```

- 输出的标题固定写着 `Benchmark (default)`，测的仍是当前生效的配置。`E2E(us)` 是入口的耗时，`Status` 应全部为 `OK`。

**默认配置的对照（可选）。** 4.5 里 GEMM 和 256K / 512K prefill 打分的"默认配置"一列，是在 4.3 启用调优表之前测的。要复现它，在 4.3 之前跑：

```bash
cd ${AITER_ROOT}

# FP8 blockscale GEMM 的官方单测，约 10 分钟（含首次编译）
#   --ck_preshuffle True   测 gemm_a8w8_blockscale_bpreshuffle
#   --csv                  形状列表（M、N、K 三列）
#   --table                打印汇总表
#   -o、--suffix           把汇总表存成 <-o 的目录>/<形状列表的文件名>_<suffix>.csv
python3 op_tests/test_gemm_a8w8_blockscale.py --ck_preshuffle True \
  --csv ${BENCH_DIR}/tuning/mi300x_dsa_indexer_a8w8_blockscale_bpreshuffle_untuned_gemm.csv \
  --table -o ${BENCH_DIR}/gemm_test --suffix default

# prefill 打分 256K / 512K，命令同上
python3 csrc/mqa_logits/mqa_logits_tune.py \
  -i ${BENCH_DIR}/tuning/mi300x_dsv32_long_untuned_mqa_logits.csv \
  -o ${BENCH_DIR}/run_config_unused.csv --run_config --all -v
```

- GEMM 单测的 `ck us` 一列是生产入口的耗时。每个形状跑两种输入，常数和随机（`data_init` 为 `constant` / `uniform`）；4.5 取随机输入的行，和 tuner 的输入一致。`wq_b` 在 M ≥ 4096 时，常数输入比随机输入快约 20%。
- 这个单测只能在启用调优表之前用。启用之后 `wq_b` 在 M ≥ 4096 选中 OPUS kernel，单测里"传入不连续的 `x_scale`"这一项检查会报 `expects contiguous scales` 而中止，所以启用之后的 GEMM 用 `--run_config` 测。

### 4.5 测试结果

都是 4.4 的官方工具在 MI300X 上的输出，kernel 耗时。几 µs 的数字单次测量有 1 µs 左右的波动。

**FP8 blockscale GEMM**（`gemm_a8w8_blockscale_bpreshuffle`，µs；`wq_b` 8192 × 1536，`wk` 128 × 7168）。"默认配置"是启用调优表之前官方单测的 `ck us`，"调优后"是启用之后 `--run_config` 的 `E2E(us)`，两者测的都是生产入口：

| M | `wq_b` 默认配置 | `wq_b` 调优后 | `wq_b` 加速 | `wq_b` 后端 | `wk` 默认配置 | `wk` 调优后 | `wk` 加速 | `wk` 后端 |
|---|---|---|---|---|---|---|---|---|
| 1 | 11.0 | 7.19 | 1.53× | ck | 29.1 | 5.76 | 5.04× | asm |
| 4 | 11.1 | 7.24 | 1.54× | ck | 30.0 | 5.91 | 5.08× | asm |
| 8 | 11.1 | 7.14 | 1.56× | ck | 31.4 | 6.02 | 5.21× | asm |
| 32 | 11.4 | 7.35 | 1.55× | ck | 34.6 | 6.20 | 5.58× | asm |
| 64 | 10.0 | 8.16 | 1.23× | ck | 24.9 | 6.43 | 3.87× | asm |
| 128 | 11.3 | 10.5 | 1.08× | asm | 25.3 | 7.01 | 3.61× | asm |
| 256 | 16.4 | 14.0 | 1.17× | asm | 28.6 | 8.20 | 3.48× | asm |
| 4096 | 182 | 135 | 1.35× | opus | 32.8 | 19.3 | 1.70× | asm |
| 8192 | 347 | 251 | 1.38× | opus | 40.2 | 31.8 | 1.27× | asm |
| 16384 | 676 | 482 | 1.40× | opus | 74.1 | 52.6 | 1.41× | asm |

**prefill 打分**（`aiter.mqa_logits.mqa_logits`，64 头，ms）。128K 及以下是启用调优表后 `test_mqa_logits.py` 一次运行里的 `triton us` 和 `mqa_logits us`；256K / 512K 是 `--run_config` 在启用前后的 `E2E(us)`：

| prompt 长度 | query 行 × key 数 | 默认（Triton） | 调优后 | 加速 | 调优表选中的 FlyDSL 变体 | 来源 |
|---|---|---|---|---|---|---|
| 4K | 4096 × 4K | 0.532 | 0.207 | 2.58× | `mfma_r4_w4` | `test_mqa_logits.py` |
| 8K | 8192 × 8K | 2.06 | 0.703 | 2.93× | `mfma_r4_w2` | `test_mqa_logits.py` |
| 16K | 16384 × 16K | 8.16 | 2.61 | 3.13× | `mfma_r4_w2` | `test_mqa_logits.py` |
| 32K | 16384 × 32K | 24.8 | 7.50 | 3.31× | `mfma_r4_w2` | `test_mqa_logits.py` |
| 64K | 16384 × 64K | 57.9 | 18.0 | 3.22× | `mfma_r4_w2` | `test_mqa_logits.py` |
| 128K | 16384 × 128K | 125 | 37.5 | 3.32× | `mfma_r4_w4` | `test_mqa_logits.py` |
| 256K | 16384 × 256K | 256 | 77.7 | 3.29× | `mfma_r4_w4`（128K 的行） | `mqa_logits_tune.py --run_config` |
| 512K | 16384 × 512K | 521 | 160 | 3.26× | `mfma_r4_w4`（128K 的行） | `mqa_logits_tune.py --run_config` |

**decode 打分**（`aiter.paged_mqa_logits.paged_mqa_logits`，64 头，µs；列名是上下文长度）。启用调优表后 `test_paged_mqa_logits.py` 一次运行里的 `gluon us`（默认配置）和 `paged_mqa_logits us`（调优后）：

| batch | 调优表的 ChunkK, WavePerEU | 4K，默认配置 | 4K，调优后 | 32K，默认配置 | 32K，调优后 | 256K，默认配置 | 256K，调优后 |
|---|---|---|---|---|---|---|---|
| 1 | 128, 4 | 4.03 | 3.73 | 5.67 | 5.28 | 15.2 | 14.5 |
| 4 | 128, 2 | 5.93 | 5.04 | 9.19 | 8.73 | 45.7 | 43.4 |
| 8 | 128, 4 | 5.41 | 4.82 | 15.2 | 13.9 | 89.8 | 81.8 |
| 32 | 256, 4 | 9.54 | 9.17 | 47.1 | 43.8 | 372 | 367 |
| 64 | 256, 4 | 16.1 | 14.6 | 99.8 | 85.0 | 810 | 747 |
| 128 | 256, 4 | 28.4 | 25.6 | 202 | 187 | 1,593 | 1,528 |
| 256 | 256, 4 | 50.1 | 44.9 | 384 | 376 | 3,112 | 3,052 |

**indexer 准备和 index-K cache**（µs）：

| token 数 | `indexer_qk_rope_quant_and_cache` | `indexer_k_quant_and_cache` | `cp_gather_indexer_k_quant_cache` |
|---|---|---|---|
| 1 | 3.66 | 2.21 | 2.51 |
| 4 | 3.88 | 2.23 | 3.03 |
| 8 | 4.21 | 2.26 | 3.12 |
| 32 | 4.52 | 2.73 | 3.18 |
| 64 | 4.67 | 2.87 | 3.14 |
| 128 | 6.85 | 2.89 | 3.21 |
| 256 | 11.0 | 2.86 | 3.27 |
| 4096 | 69.2 | 3.16 | 3.43 |
| 8192 | 143 | 4.25 | 4.01 |
| 16384 | 293 | 5.33 | 4.69 |

- `indexer_qk_rope_quant_and_cache` 取 64 头、`compute_all_q_rope=True`、`is_neox=True` 的行：全部 token 的 Q 都处理，一半 token 的 K 写入 cache（脚本默认的 `--valid_fraction 0.5`）。`is_neox=False` 的行相差不到 10%。
- 后两列取 `preshuffle=True` 的行。

**top-k，decode**（k = 2048，µs；列名是上下文长度和模式）。每格两个数：`aiter.top_k_per_row_decode`（默认分发）/ 直接调 `aiter.flydsl_top_k_per_row_decode`：

| batch | 4K，默认 | 4K，stable | 32K，默认 | 32K，stable | 128K，默认 | 128K，stable |
|---|---|---|---|---|---|---|
| 1 | 6.77 / 8.29 | 8.15 / 8.25 | 20.4 / 18.2 | 31.8 / 28.0 | 53.8 / 34.0 | 42.8 / 42.7 |
| 4 | 8.80 / 8.20 | 8.24 / 8.23 | 20.5 / 18.9 | 32.1 / 28.9 | 63.5 / 34.3 | 44.3 / 44.3 |
| 8 | 9.39 / 8.33 | 8.25 / 8.30 | 20.5 / 18.9 | 32.3 / 29.0 | 64.7 / 35.5 | 46.0 / 45.6 |
| 32 | 9.47 / 8.43 | 8.38 / 8.37 | 21.1 / 19.6 | 33.1 / 29.5 | 65.0 / 37.6 | 78.8 / 48.0 |
| 64 | 9.58 / 8.59 | 8.63 / 8.59 | 25.3 / 19.9 | 37.9 / 29.9 | 65.6 / 44.6 | 79.4 / 51.0 |
| 128 | 9.95 / 9.14 | 12.7 / 9.13 | 25.8 / 20.7 | 39.4 / 31.1 | 76.4 / 59.9 | 90.3 / 65.6 |
| 256 | 11.0 / 10.4 | 13.8 / 10.3 | 30.5 / 26.0 | 44.2 / 36.3 | 114 / 78.5 | 128 / 85.4 |

- "默认"和"stable"是 top-k 的两种模式：默认只保证选出分数最高的 k 个；stable 的结果是确定的（下标从小到大输出，分数相同时取下标小的），tensor-parallel 部署时用它。
- 表里取的是只输出下标的行（`write_values`、`fast`、`ragged` 都为 False）；`flydsl` 为 False 的行是默认分发，为 True 的行是直接调 FlyDSL 入口。
- 直接调 FlyDSL 入口明显更快（差 2 µs 以上）的范围和省下的耗时：上下文 4K、stable，batch ≥ 128：少 26–28%；上下文 32K、默认，batch ≥ 64：少 15–21%；上下文 32K、stable：少 10–21%；上下文 128K、默认：少 22–46%；上下文 128K、stable，batch ≥ 32：少 27–39%。

**top-k，prefill**（`aiter.top_k_per_row_prefill`，k = 2048，ms；一个 chunk 的耗时）：

| prompt 长度 | 行数 × 最长一行的 key 数 | ms |
|---|---|---|
| 4K | 4096 × 4K | 0.071 |
| 8K | 8192 × 8K | 0.206 |
| 16K | 16384 × 16K | 0.607 |
| 32K | 16384 × 32K | 1.18 |
| 64K | 16384 × 64K | 2.61 |
| 128K | 16384 × 128K | 5.43 |
| 256K | 16384 × 256K | 12.2 |
| 512K | 16384 × 512K | 29.0 |

**稀疏注意力，decode**（`aiter.mla.mla_decode_fwd`，BF16 Q 和 KV，µs；每条序列 1 个 query token，每个 query token 看 2048 行 KV）。两种调用方式见 4.6；persistent 一列是 `-ms` 四个取值里最快的，括号里是那个取值：

| batch | 16 头，非 persistent | 16 头，persistent | 128 头，非 persistent | 128 头，persistent |
|---|---|---|---|---|
| 1 | 22.3 | 18.9（64） | 39.1 | 37.2（32） |
| 4 | 21.5 | 21.9（64） | 40.6 | 55.3（16） |
| 8 | 23.3 | 24.2（64） | 46.3 | 83.4（32） |
| 32 | 39.6 | 49.1（32） | 100 | 251（8） |
| 64 | 57.4 | 81.4（16） | 150 | 427（64） |
| 128 | 94.9 | 142（8） | 269 | 934（64） |
| 256 | 166 | 269（32） | 447 | 1,938（32） |

**稀疏注意力，prefill 的一个 chunk**（同上，ms；persistent 取 `-ms 32`）：

| query token 数 | 16 头，非 persistent | 16 头，persistent | 128 头，非 persistent | 128 头，persistent |
|---|---|---|---|---|
| 4096 | 2.68 | 3.99 | 6.60 | 32.1 |
| 8192 | 5.44 | 8.00 | 13.1 | 65.0 |
| 16384 | 11.0 | 16.0 | 27.3 | 130 |

### 4.6 推荐的 kernel

| 阶段 | 推荐 |
|---|---|
| `wq_b`、`wk` | `gemm_a8w8_blockscale_bpreshuffle` + 调优表，各 M 选中的 kernel 见下表 |
| indexer 准备 | `indexer_qk_rope_quant_and_cache`（FP8 模式，`preshuffle=True`） |
| prefill 打分 | `aiter.mqa_logits.mqa_logits` + 调优表：FlyDSL `mfma_r4_w4`（4K 和 ≥ 128K）、`mfma_r4_w2`（8K–64K） |
| decode 打分 | `aiter.paged_mqa_logits.paged_mqa_logits` + 调优表（Gluon）：batch 1 用 `ChunkK=128, WavePerEU=4`；batch 4 用 `ChunkK=128, WavePerEU=2`；batch 8 用 `ChunkK=128, WavePerEU=4`；batch 32–256 用 `ChunkK=256, WavePerEU=4` |
| top-k，prefill | `aiter.top_k_per_row_prefill`，默认分发 |
| top-k，decode | `aiter.top_k_per_row_decode`，默认分发。在上下文 32K 和 128K 的测试点上，直接调 `aiter.flydsl_top_k_per_row_decode`（参数相同）更快，范围见 4.5 |
| 稀疏注意力 | `aiter.mla.mla_decode_fwd`，BF16 Q + BF16 KV，非 persistent 调用 |

稀疏注意力的两种调用方式。入口都是 `aiter.mla.mla_decode_fwd`，每个 query token 要看的 2048 行 KV 用下标列表传入，算的是同一件事：

| 调用方式 | 怎么调用 | 走的 ASM 核 |
|---|---|---|
| 非 persistent | 直接调用，不传 work metadata | 16 头：`mla_dec_stage1_bf16_a16w16_subQ16_mqa16`；128 头：`mla_dec_stage1_bf16_a16w16_subQ128_mqa128` |
| persistent | 先用 `get_mla_metadata_v1(..., topk=2048)` 生成 work metadata，再传给入口 | `mla_a16w16_qh16_m16x4_n16x1_coex0_mask1_ps`（gfx942 上 BF16 的 persistent 核只有这一个，16 头和 128 头都用它） |

按 4.5 的两张表，16 头：batch 1 时 persistent 的耗时少 15%，batch 4、8 时两者相差不到 5%，batch ≥ 32（含 prefill 的 chunk）时 persistent 的耗时是非 persistent 的 1.2–1.6 倍；128 头：batch 1 时两者相差不到 5%，batch ≥ 4（含 prefill 的 chunk）时 persistent 的耗时是非 persistent 的 1.4–4.9 倍。所以推荐非 persistent。

GEMM 调优表选中的 kernel：

| 层（N×K） | M | 后端 | splitK | kernel |
|---|---|---|---|---|
| `wq_b` 8192×1536 | 1 | ck | 0 | `a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1` |
| `wq_b` 8192×1536 | 4 | ck | 0 | `a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1` |
| `wq_b` 8192×1536 | 8 | ck | 0 | `a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1` |
| `wq_b` 8192×1536 | 32 | ck | 0 | `a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1` |
| `wq_b` 8192×1536 | 64 | ck | 0 | `a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1` |
| `wq_b` 8192×1536 | 128 | asm | 1 | `fp8gemm_bf16_blockscale_BpreShuffle_32x128` |
| `wq_b` 8192×1536 | 256 | asm | 1 | `fp8gemm_bf16_blockscale_BpreShuffle_64x128` |
| `wq_b` 8192×1536 | 4096 | opus | 0 | `opus_gemm_gfx942_a8w8_bs_bpreshuf_sb_tailm_v16_512x128x128x128_4x2_16x16x32_1x128x128` |
| `wq_b` 8192×1536 | 8192 | opus | 0 | `opus_gemm_gfx942_a8w8_bs_bpreshuf_sb_tailm_v16_512x128x128x128_4x2_16x16x32_1x128x128` |
| `wq_b` 8192×1536 | 16384 | opus | 0 | `opus_gemm_gfx942_a8w8_bs_bpreshuf_sb_tailm_v16_512x128x128x128_4x2_16x16x32_1x128x128` |
| `wk` 128×7168 | 1 | asm | 8 | `fp8gemm_bf16_blockscale_BpreShuffle_32x128` |
| `wk` 128×7168 | 4 | asm | 8 | `fp8gemm_bf16_blockscale_BpreShuffle_32x128` |
| `wk` 128×7168 | 8 | asm | 8 | `fp8gemm_bf16_blockscale_BpreShuffle_32x128` |
| `wk` 128×7168 | 32 | asm | 8 | `fp8gemm_bf16_blockscale_BpreShuffle_32x128` |
| `wk` 128×7168 | 64 | asm | 8 | `fp8gemm_bf16_blockscale_BpreShuffle_32x128` |
| `wk` 128×7168 | 128 | asm | 8 | `fp8gemm_bf16_blockscale_BpreShuffle_48x128` |
| `wk` 128×7168 | 256 | asm | 8 | `fp8gemm_bf16_blockscale_BpreShuffle_48x128` |
| `wk` 128×7168 | 4096 | asm | 3 | `fp8gemm_bf16_blockscale_BpreShuffle_48x128` |
| `wk` 128×7168 | 8192 | asm | 2 | `fp8gemm_bf16_blockscale_BpreShuffle_64x128` |
| `wk` 128×7168 | 16384 | asm | 1 | `fp8gemm_bf16_blockscale_BpreShuffle_64x128` |

- 选中 OPUS 的形状（`wq_b`，M ≥ 4096）要求传入的 `x_scale` 是连续的张量，否则入口报 `expects contiguous scales`；CK 和 ASM 的形状没有这个要求。

## 5. DeepSeek-V4

形状：indexer 64 头 × 128 维；index-K cache 里是 4:1 压缩后的 key（上下文 L 对应 L / 4 条，布局同 V3.2），top-k 512；attention 行宽 512，每 token 看最近 128 个 token 的窗口，CSA 层（`compress_ratio = 4`，有 indexer）再加 top-k 的 512 行，HCA 层（`compress_ratio = 128`）再加全部 L / 128 行，其余层只看窗口。Flash 尺寸：hidden 4096、q_lora_rank 1024、64 头；Pro 尺寸：hidden 7168、q_lora_rank 1536、128 头（按上游调优表里的形状取，以实际 config 为准）。

### 5.1 算子

| 阶段 | aiter 入口 | gfx942 上的后端 | 调优 |
|---|---|---|---|
| `wq_b`（FP8 blockscale 线性层） | `aiter.get_hip_quant(QuantType.per_1x128)` + `aiter.gemm_a8w8_blockscale_bpreshuffle` | CK / CKTile / ASM / OPUS | 官方 tuner |
| Q 准备（RoPE + Hadamard + FP8 量化） | `aiter.rope_rotate_activation`（输出给 FP8 张量） | HIP 融合核 | 无 |
| indexer 压缩器（写 FP8 key）、CSA 主压缩器 | `aiter.ops.flydsl.kernels.fused_compress_attn.flydsl_fused_compress_attn` | FlyDSL | 无 |
| HCA 主压缩器 | `aiter.ops.flydsl.kernels.fused_compress_attn_hca.flydsl_hca_compress_attn` | FlyDSL | 无 |
| 打分，prefill 和 decode | `aiter.paged_mqa_logits.paged_mqa_logits` | Gluon | 官方 tuner |
| top-k 512 | `aiter.top_k_per_row_decode`、`aiter.top_k_per_row_prefill` | HIP radix / FlyDSL | 无 |
| 稀疏注意力（行宽 512，BF16 KV） | `aiter.ops.triton.attention` 下的 `pa_decode_sparse`、`pa_prefill_sparse`、`sparse_mla_fwd` | Triton / Triton / Gluon | 无 |

- `weights_proj` 和压缩器的 `wkv_gate` 是 BF16 线性层，用 `torch.nn.functional.linear`。
- Pro 尺寸的 `wq_b`（8192 × 1536）和 V3.2 是同一个形状，结果和 kernel 见第 4 节。

### 5.2 调优

要调的是分页打分核（上下文比 V3.2 短 4 倍，另加 prefill 的大 batch）和 `wq_b` 的 GEMM。GEMM 与 V3.2 是同一次调优、同一张表（4.2 节，其中 N=8192、K=1024 的行属于 V4）。

**调优输入**

`tuning/mi300x_dsv4_paged_mqa_logits_untuned.csv`：

```csv
batch_size,next_n,heads,head_dim,kv_block_size,preshuffle,context_len
1,1,64,128,64,True,1024
1,1,64,128,64,True,8192
1,1,64,128,64,True,65536
4,1,64,128,64,True,1024
4,1,64,128,64,True,8192
4,1,64,128,64,True,65536
8,1,64,128,64,True,1024
8,1,64,128,64,True,8192
8,1,64,128,64,True,65536
32,1,64,128,64,True,1024
32,1,64,128,64,True,8192
32,1,64,128,64,True,65536
64,1,64,128,64,True,1024
64,1,64,128,64,True,8192
64,1,64,128,64,True,65536
128,1,64,128,64,True,1024
128,1,64,128,64,True,8192
128,1,64,128,64,True,65536
256,1,64,128,64,True,1024
256,1,64,128,64,True,8192
256,1,64,128,64,True,65536
4096,1,64,128,64,True,1024
8192,1,64,128,64,True,2048
16384,1,64,128,64,True,4096
```

大 batch 的行只配短上下文：tuner 的参考实现在 batch 16384 × 上下文 8192 时要一次分配 64 GiB，失败后整次调优都没有结果。

**调优命令**（不到 1 分钟）

```bash
cd ${AITER_ROOT}
python3 csrc/paged_mqa_logits/paged_mqa_logits_tune.py \
  -i ${BENCH_DIR}/tuning/mi300x_dsv4_paged_mqa_logits_untuned.csv \
  -o ${BENCH_DIR}/tuning/mi300x_dsv4_paged_mqa_logits_tuned.csv --max_model_len 131072 --all -v
```

`--max_model_len` 的含义同 4.2。V4 按压缩后的 key 数取：最大上下文 512K ÷ 4 = 131072。

**调优结果**（MI300X，上述 aiter commit）

`tuning/mi300x_dsv4_paged_mqa_logits_tuned.csv`：

```csv
gfx,cu_num,batch_size,next_n,heads,head_dim,kv_block_size,preshuffle,context_len,backend,ChunkK,WavePerEU,wg_per_cu,us,gluon_us,flydsl_us
gfx942,304,1,1,64,128,64,True,1024;8192;65536,gluon,128,2,0,4.25,4.25,
gfx942,304,4,1,64,128,64,True,1024;8192;65536,gluon,128,4,0,6.15,6.15,
gfx942,304,8,1,64,128,64,True,1024;8192;65536,gluon,128,4,0,8.76,8.76,
gfx942,304,32,1,64,128,64,True,1024;8192;65536,gluon,128,4,0,17.71,17.71,
gfx942,304,64,1,64,128,64,True,1024;8192;65536,gluon,256,4,0,30.91,30.91,
gfx942,304,128,1,64,128,64,True,1024;8192;65536,gluon,256,4,0,54.48,54.48,
gfx942,304,256,1,64,128,64,True,1024;8192;65536,gluon,256,4,0,98.87,98.87,
gfx942,304,4096,1,64,128,64,True,1024,gluon,128,4,0,196.83,196.83,
gfx942,304,8192,1,64,128,64,True,2048,gluon,256,4,0,780.21,780.21,
gfx942,304,16384,1,64,128,64,True,4096,gluon,256,2,0,2992.8,2992.8,
```

### 5.3 启用调优表

做法同 4.3，分页打分表换成 V4 的那张。两张分页表不能同时在 `model_configs/` 里，先删 V3.2 的再放 V4 的：

```bash
rm -f ${AITER_ROOT}/aiter/configs/model_configs/mi300x_dsv32_paged_mqa_logits_tuned.csv
cp ${BENCH_DIR}/tuning/mi300x_dsv4_paged_mqa_logits_tuned.csv ${AITER_ROOT}/aiter/configs/model_configs/
cp ${BENCH_DIR}/tuning/mi300x_dsa_indexer_a8w8_blockscale_bpreshuffle_tuned_gemm.csv ${AITER_ROOT}/aiter/configs/model_configs/
```

用环境变量的话：

```bash
export AITER_CONFIG_PAGED_MQA_LOGITS=${BENCH_DIR}/tuning/mi300x_dsv4_paged_mqa_logits_tuned.csv
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE=${BENCH_DIR}/tuning/mi300x_dsa_indexer_a8w8_blockscale_bpreshuffle_tuned_gemm.csv
```

GEMM 表在 4.3 已经启用过的不用再动；否则按 4.3 重建 GEMM 生产模块。确认已生效：

```bash
python3 - <<'EOF'
import aiter
import aiter.paged_mqa_logits as pmql
from aiter.ops.gemm_op_a8w8 import AITER_CONFIGS, get_CKGEMM_config

cfg = pmql.get_paged_mqa_logits_config(1, 1, 64, 128, 64, True)
print("paged_mqa  :", cfg["backend"], cfg["ChunkK"], cfg["WavePerEU"])
gemm = get_CKGEMM_config(64, 8192, 1024, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE)
if gemm:
    print("gemm       :", gemm["libtype"], gemm["kernelName"], gemm["splitK"])
else:
    print("gemm       : no tuned row")
EOF
```

期望输出的最后两行：

```
paged_mqa  : gluon 128 2
gemm       : ck a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1 0
```

没有生效时是 `gluon 256 2` 和 `no tuned row`。

### 5.4 测试

官方脚本，直接调用。5.5 的数据就是这一节的命令在 5.3 启用调优表之后的输出。

**Q 准备、压缩器、top-k**

```bash
cd ${AITER_ROOT}

# Q 准备
#   --fp8   跑融合的 RoPE + Hadamard + FP8 量化路径
#   -hn     头数
#   -n      每个头的维度
#   -m      token 数
python3 op_tests/test_dsv4_rotate_quant.py --fp8 -hn 64 -n 128 -m 1 4 8 32 64 128 256 4096 8192 16384

# 压缩器
#   -s             要测的形状：csa_main（CSA 主压缩器）、csa_indexer（indexer 压缩器，写 FP8 key）、hca_main（HCA 主压缩器）
#   -m             每步的推测 token 数，0 表示每条序列每步只有 1 个新 token
#   -b             decode 模式的 batch
#   --prefill-bs   prefill 模式的序列数，每条序列 256 个 token；16 / 32 / 64 即 4096 / 8192 / 16384 个 token
python3 op_tests/test_flydsl_compress_attn.py -s csa_main csa_indexer hca_main -m 0 -b 1 4 8 32 64 128 256 --prefill-bs 16 32 64

# top-k 512，参数含义同 4.4；-c 按压缩后的 key 数给
python3 op_tests/test_topk_per_row.py -k 512 -n 1 -b 1 4 8 32 64 128 256 -c 1024 8192 32768
python3 op_tests/test_topk_per_row.py -k 512 -n 1 -b 1 -c 16384 --num_prefix 0 16384 49152 114688 --regressions
```

- 汇总表的误差列是 0 或 1e-4 以下的舍入差。
- `test_dsv4_rotate_quant.py` 的汇总表标题固定是 `rotate_fp4quant`，`op` 一列是 `rope_rotate_fp8`。
- `test_flydsl_compress_attn.py` 另外打印的 `flydsl_hca_fp8`、`flydsl_csa_nm_asm_fp8` 两张表是压缩器输出 FP8 KV 的用例，本指南的 BF16 KV 路径不用。

**打分。** 用的是和 V3.2 相同的分页核，脚本同样要先改成 64 头（原因见 4.4）。`MAX_MODEL_LEN` 不用改：V4 的 index-K cache 里是 4:1 压缩后的 key，512K 上下文对应 131072 条，正好是脚本原来的值。`--kv-len` 按压缩后的 key 数给：

```bash
cd ${AITER_ROOT}

# 恢复脚本后只改 HEADS，并确认
git checkout op_tests/test_paged_mqa_logits.py
sed -i 's/^HEADS = 32$/HEADS = 64/' op_tests/test_paged_mqa_logits.py
grep -n '^HEADS\|^MAX_MODEL_LEN' op_tests/test_paged_mqa_logits.py

# decode 打分，参数含义同 4.4
python3 op_tests/test_paged_mqa_logits.py -b 1 4 8 32 64 128 256 --next-n 1 --kv-len 1024 8192 65536
```

`grep` 应显示 `HEADS = 64` 和 `MAX_MODEL_LEN = 131072`。测完用 `git checkout op_tests/test_paged_mqa_logits.py` 恢复。GEMM 的 `wq_b` 8192 × 1024 已经包含在 4.4 的 `--run_config` 和官方单测里。

**稀疏注意力**

```bash
cd ${AITER_ROOT}

# 三个入口的正确性检查
#   -q   精简 pytest 的输出
python3 -m pytest -q op_tests/triton_tests/attention/test_pa_decode_sparse.py
python3 -m pytest -q op_tests/triton_tests/attention/test_pa_prefill_sparse.py
python3 -m pytest -q op_tests/triton_tests/attention/test_sparse_mla.py

# 性能：三个入口各有一个官方 bench，都是行宽 512、BF16 的 Q 和 KV。
# 下面是 64 头的命令；5.5 里其余头数（8、16、128）是把头数换掉再跑一遍

# decode 入口 pa_decode_sparse
#   --shape T H D KV_LEN   token 数（decode 的 batch）、头数、行宽、每个 token 看的 KV 行数；可以重复给。
#                          640 = 窗口 128 + top-k 512（CSA 层），128 = 只有窗口
#   --providers bf16       BF16 的 Q 和 KV 池（另外两个 provider 是 FP8 格式，gfx942 上不用）
#   --page 64              KV 池的页大小
#   --metric time          报耗时（µs）
#   --timer profiler       取 profiler 里 kernel 的耗时；这个 kernel 只有几十 µs，墙钟时间主要是 Python 开销
python3 op_tests/op_benchmarks/triton/bench_pa_decode_sparse.py \
  --shape 1 64 512 640 --shape 4 64 512 640 --shape 8 64 512 640 --shape 32 64 512 640 \
  --shape 64 64 512 640 --shape 128 64 512 640 --shape 256 64 512 640 \
  --shape 1 64 512 128 --shape 4 64 512 128 --shape 8 64 512 128 --shape 32 64 512 128 \
  --shape 64 64 512 128 --shape 128 64 512 128 --shape 256 64 512 128 \
  --providers bf16 --page 64 --metric time --timer profiler

# Gluon 入口 sparse_mla_fwd，decode
#   --qk_rope_head_dim 0   行宽 512、不另带 RoPE 的布局（V4）；默认值 64 是 V3.2 的 576 宽
#   --num_seqs             序列数，即 decode 的 batch；可以给多个
#   --num_tokens 1         每条序列的 query token 数，1 是 decode 的一步
#   --num_heads            头数
#   --context              每条序列的上下文长度（KV 行数）
#   --topk                 每个 token 看的行数，在上下文里随机取
python3 op_tests/op_benchmarks/triton/bench_sparse_mla.py --qk_rope_head_dim 0 \
  --num_seqs 1 4 8 32 64 128 256 --num_tokens 1 --num_heads 64 --context 8192 --topk 640
python3 op_tests/op_benchmarks/triton/bench_sparse_mla.py --qk_rope_head_dim 0 \
  --num_seqs 1 4 8 32 64 128 256 --num_tokens 1 --num_heads 64 --context 8192 --topk 128

# Gluon 入口 sparse_mla_fwd，prefill：一条序列，--num_tokens 是一个 chunk 的 token 数；
# 第 p 个 token 看 min(p + 1, 640) 行
python3 op_tests/op_benchmarks/triton/bench_sparse_mla.py --qk_rope_head_dim 0 \
  --num_seqs 1 --num_tokens 4096 8192 16384 --context 16384 --num_heads 64 --topk 640

# prefill 入口 pa_prefill_sparse
#   --shapes prefill   只跑 prefill
#   --prefill_cfgs     每组是"query token 数,头数,KV 池的行数,每个 token 最多看的行数"；
#                      每个 token 实际看的行数在最多行数的 1/4 到 1 倍之间随机取
python3 op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4.py --shapes prefill \
  --prefill_cfgs 4096,64,65536,640 8192,64,65536,640 16384,64,65536,640
```

- 三个 pytest 只做正确性检查，在 gfx942 上的结果是 100 passed / 347 skipped、16 passed / 32 skipped、47 passed / 97 skipped（跳过的是其他架构或其他 KV 格式的用例）。
- 三个 bench 各自打印一张结果表。`bench_sparse_attention_dsv4.py` 的表里有两组数字：直接调 Triton kernel 的，和经公开入口 `pa_prefill_sparse` 的，看后者。

### 5.5 测试结果

都是 5.4 的官方脚本在 MI300X 上的输出，kernel 耗时。"上下文"和"prompt 长度"都是 token 数，indexer 实际处理的是 L / 4 条 key。

**FP8 blockscale GEMM，`wq_b` 8192 × 1024**（µs；各列含义同 4.5）：

| M | 默认配置 | 调优后 | 加速 | 后端 |
|---|---|---|---|---|
| 1 | 9.19 | 5.79 | 1.59× | cktile |
| 4 | 9.15 | 5.76 | 1.59× | ck |
| 8 | 9.12 | 5.62 | 1.62× | ck |
| 32 | 9.46 | 6.45 | 1.47× | ck |
| 64 | 8.82 | 6.47 | 1.36× | ck |
| 128 | 9.14 | 7.84 | 1.17× | ck |
| 256 | 12.6 | 11.7 | 1.08× | asm |
| 4096 | 131 | 97.1 | 1.35× | opus |
| 8192 | 250 | 182 | 1.37× | opus |
| 16384 | 486 | 348 | 1.40× | opus |

**decode 打分**（`aiter.paged_mqa_logits.paged_mqa_logits`，64 头，µs；列名是上下文长度）。启用调优表后 `test_paged_mqa_logits.py` 一次运行里的 `gluon us`（默认配置）和 `paged_mqa_logits us`（调优后）：

| batch | 调优表的 ChunkK, WavePerEU | 4K，默认配置 | 4K，调优后 | 32K，默认配置 | 32K，调优后 | 256K，默认配置 | 256K，调优后 |
|---|---|---|---|---|---|---|---|
| 1 | 128, 2 | 4.97 | 4.94 | 5.21 | 4.75 | 6.98 | 7.14 |
| 4 | 128, 4 | 5.23 | 4.81 | 6.05 | 5.49 | 15.8 | 14.5 |
| 8 | 128, 4 | 4.56 | 4.22 | 6.30 | 6.30 | 24.9 | 23.4 |
| 32 | 128, 4 | 5.44 | 4.88 | 15.2 | 13.9 | 91.2 | 82.6 |
| 64 | 256, 4 | 6.29 | 6.15 | 28.5 | 25.7 | 194 | 185 |
| 128 | 256, 4 | 9.58 | 9.44 | 51.0 | 45.3 | 398 | 372 |
| 256 | 256, 4 | 16.6 | 15.4 | 96.7 | 89.5 | 764 | 751 |

**prefill 打分**，同一个分页核，每个 query token 一行（ms；官方 tuner 的 `-v` 输出里默认配置 `ChunkK=256, WavePerEU=2` 和调优表选中配置的耗时）。tuner 让每一行都对全部压缩 key 打分，是上限；实际 prefill 里每个 token 只看到它前面的 key：

| prompt 长度 | query 行 × 压缩 key | 调优表的 ChunkK, WavePerEU | 默认配置 | 调优后 | 加速 |
|---|---|---|---|---|---|
| 4K | 4096 × 1K | 128, 4 | 0.205 | 0.197 | 1.04× |
| 8K | 8192 × 2K | 256, 4 | 0.783 | 0.780 | 1.00× |
| 16K | 16384 × 4K | 256, 2 | 2.99 | 2.99 | 1.00× |

**Q 准备和压缩器，decode**（µs；每条序列每步 1 个新 token）：

| batch | Q 准备 | indexer 压缩器 | CSA 主压缩器 | HCA 主压缩器 | HCA 的形状走 `flydsl_fused_compress_attn` |
|---|---|---|---|---|---|
| 1 | 4.19 | 5.39 | 7.00 | 11.5 | 145 |
| 4 | 3.74 | 5.34 | 6.44 | 12.8 | 153 |
| 8 | 3.73 | 5.35 | 6.44 | 14.1 | 162 |
| 32 | 3.88 | 4.97 | 6.49 | 15.7 | 160 |
| 64 | 4.06 | 5.39 | 6.54 | 16.8 | 162 |
| 128 | 4.71 | 5.39 | 6.57 | 21.2 | 161 |
| 256 | 6.21 | 5.48 | 6.77 | 33.0 | 163 |

**Q 准备和压缩器，prefill**（µs；压缩器按每条序列 256 个 token 分成 16 / 32 / 64 条序列）：

| token 数 | Q 准备 | indexer 压缩器 | CSA 主压缩器 | HCA 主压缩器 | HCA 的形状走 `flydsl_fused_compress_attn` |
|---|---|---|---|---|---|
| 4096 | 43.4 | 6.96 | 13.3 | 14.8 | 200 |
| 8192 | 83.4 | 9.61 | 56.2 | 15.2 | 200 |
| 16384 | 172 | 11.6 | 115 | 19.5 | 201 |

- "HCA 主压缩器"是 `test_flydsl_compress_attn.py` 里 `hca_main` 的 `2kernel` 一行，即 `flydsl_hca_compress_attn`；最后一列是同一形状的 `single` 一行，即把它交给 `flydsl_fused_compress_attn`。

**top-k 512，decode**（µs；列名是上下文长度和模式，脚本的 `-c` 是它的 1/4）。每格两个数：`aiter.top_k_per_row_decode`（默认分发）/ 直接调 `aiter.flydsl_top_k_per_row_decode`，取法同 4.5：

| batch | 4K，默认 | 4K，stable | 32K，默认 | 32K，stable | 128K，默认 | 128K，stable |
|---|---|---|---|---|---|---|
| 1 | 6.22 / 7.41 | 7.39 / 7.36 | 11.0 / 11.4 | 11.4 / 11.4 | 19.6 / 16.7 | 26.9 / 17.6 |
| 4 | 8.68 / 7.43 | 7.32 / 7.33 | 10.9 / 11.4 | 11.4 / 11.4 | 23.4 / 16.9 | 30.9 / 17.9 |
| 8 | 8.75 / 7.57 | 7.41 / 7.48 | 11.1 / 11.5 | 11.5 / 11.4 | 23.4 / 17.0 | 31.0 / 17.9 |
| 32 | 8.75 / 7.52 | 7.46 / 7.54 | 11.0 / 11.5 | 11.6 / 11.6 | 24.3 / 17.8 | 32.1 / 18.8 |
| 64 | 8.80 / 7.59 | 7.72 / 7.75 | 11.9 / 12.0 | 12.2 / 12.2 | 24.7 / 18.1 | 32.3 / 19.1 |
| 128 | 8.95 / 7.69 | 9.13 / 7.71 | 11.9 / 12.4 | 17.3 / 12.4 | 25.1 / 19.1 | 33.0 / 19.9 |
| 256 | 9.03 / 7.87 | 9.26 / 7.83 | 15.8 / 13.5 | 21.2 / 13.6 | 29.1 / 24.0 | 37.2 / 24.8 |

- 直接调 FlyDSL 入口明显更快（差 2 µs 以上）的范围和省下的耗时：上下文 32K、默认，batch 256：少 14%；上下文 32K、stable，batch ≥ 128：少 28–36%；上下文 128K、默认：少 15–28%；上下文 128K、stable：少 33–42%。

**top-k 512，prefill**（`aiter.top_k_per_row_prefill`，ms；一个 chunk 的耗时）。脚本里各行排序的 key 数逐行加一；V4 每 4 个 token 才多一条压缩 key，同一个 chunk 里各行的 key 数都接近最长的那一行，实际耗时比表里略高（第一行差得最多）：

| prompt 长度 | 行数 × 最长一行的压缩 key 数 | ms |
|---|---|---|
| 64K | 16384 × 16K | 0.610 |
| 128K | 16384 × 32K | 1.12 |
| 256K | 16384 × 64K | 2.50 |
| 512K | 16384 × 128K | 5.24 |

**稀疏注意力**（行宽 512，BF16 Q 和 KV）。三个官方 bench 的测法不同，不同入口的数字不能直接逐格比较：

| bench | 每个 token 的 KV 行怎么取 | 计时 |
|---|---|---|
| `bench_pa_decode_sparse.py` | 连续的位置（页是打散的） | profiler 里 kernel 的耗时；每次调用读同一份数据，不清 cache |
| `bench_sparse_mla.py` | 在上下文里随机取 | profiler 里 kernel 的耗时；每次调用前写 512 MB 冲掉 cache |
| `bench_sparse_attention_dsv4.py` | 在 KV 池里随机取，每个 token 160–640 行 | GPU event 之间的时间，包含两次 launch 之间的间隔 |

`pa_decode_sparse`，decode（µs；列名是每个 token 看的行数和头数）：

| batch | 640 行，8 头 | 640 行，16 头 | 640 行，64 头 | 640 行，128 头 | 128 行，8 头 | 128 行，16 头 | 128 行，64 头 | 128 行，128 头 |
|---|---|---|---|---|---|---|---|---|
| 1 | 11.8 | 12.8 | 13.5 | 14.0 | 9.61 | 9.93 | 10.0 | 10.7 |
| 4 | 11.7 | 13.4 | 19.1 | 25.9 | 8.65 | 9.34 | 12.5 | 17.2 |
| 8 | 13.3 | 15.6 | 25.6 | 38.1 | 8.85 | 10.1 | 17.0 | 21.0 |
| 32 | 20.7 | 26.2 | 67.1 | 93.8 | 12.4 | 16.9 | 29.2 | 23.1 |
| 64 | 32.7 | 36.3 | 97.1 | 189 | 14.8 | 21.2 | 23.4 | 44.2 |
| 128 | 56.6 | 64.0 | 192 | 376 | 21.7 | 27.4 | 46.3 | 91.4 |
| 256 | 98.3 | 101 | 386 | 700 | 23.6 | 24.8 | 94.7 | 173 |

`sparse_mla_fwd`，decode（µs）：

| batch | 640 行，8 头 | 640 行，16 头 | 640 行，64 头 | 640 行，128 头 | 128 行，8 头 | 128 行，16 头 | 128 行，64 头 | 128 行，128 头 |
|---|---|---|---|---|---|---|---|---|
| 1 | 16.4 | 17.1 | 17.0 | 17.4 | 9.12 | 9.70 | 10.1 | 10.1 |
| 4 | 16.2 | 17.2 | 19.0 | 22.5 | 9.07 | 9.89 | 10.5 | 11.9 |
| 8 | 17.0 | 18.3 | 22.9 | 39.9 | 9.61 | 10.1 | 11.9 | 14.4 |
| 32 | 24.3 | 28.2 | 51.7 | 169 | 11.5 | 13.4 | 24.9 | 31.4 |
| 64 | 41.5 | 46.1 | 88.0 | 136 | 14.5 | 18.1 | 32.0 | 38.4 |
| 128 | 61.1 | 68.3 | 147 | 259 | 23.5 | 28.8 | 40.0 | 67.1 |
| 256 | 108 | 112 | 282 | 484 | 32.0 | 38.1 | 74.3 | 121 |

prefill 的一个 chunk（ms；每个 token 最多看 640 行）：

| query token 数 | `pa_prefill_sparse`，8 头 | `pa_prefill_sparse`，16 头 | `pa_prefill_sparse`，64 头 | `pa_prefill_sparse`，128 头 | `sparse_mla_fwd`，8 头 | `sparse_mla_fwd`，16 头 | `sparse_mla_fwd`，64 头 | `sparse_mla_fwd`，128 头 |
|---|---|---|---|---|---|---|---|---|
| 4096 | 0.374 | 0.405 | 1.61 | 3.19 | 0.704 | 0.732 | 2.78 | 5.49 |
| 8192 | 0.707 | 0.784 | 3.17 | 6.48 | 1.48 | 1.54 | 6.07 | 11.8 |
| 16384 | 1.42 | 1.55 | 6.31 | 12.3 | 3.12 | 3.23 | 12.9 | 25.8 |

### 5.6 推荐的 kernel

| 阶段 | 推荐 |
|---|---|
| `wq_b` | `gemm_a8w8_blockscale_bpreshuffle` + 调优表，各 M 选中的 kernel 见下表 |
| Q 准备 | `aiter.rope_rotate_activation`，FP8 输出，`group_size=128` |
| indexer 压缩器 | `flydsl_fused_compress_attn(..., quant_mode="fp8_per_row", use_ue8m0=True, preshuffle=True)` |
| CSA 主压缩器 | `flydsl_fused_compress_attn`（BF16 输出） |
| HCA 主压缩器 | `flydsl_hca_compress_attn`。不要把 128:1 的形状交给 `flydsl_fused_compress_attn`，慢 5–14 倍（5.5） |
| 打分，decode | `aiter.paged_mqa_logits.paged_mqa_logits` + V4 调优表（Gluon）：batch 1 用 `ChunkK=128, WavePerEU=2`；batch 4–32 用 `ChunkK=128, WavePerEU=4`；batch 64–256 用 `ChunkK=256, WavePerEU=4` |
| 打分，prefill | 同一个分页核，每个 query token 一行（`context_lens` 为该 token 可见的 key 数）。不需要 K gather |
| top-k 512，decode | `aiter.top_k_per_row_decode(k=512)`，默认分发。在上下文 128K 的测试点上，直接调 `aiter.flydsl_top_k_per_row_decode`（参数相同）更快，范围见 5.5 |
| top-k 512，prefill | `aiter.top_k_per_row_prefill(k=512)` |
| 稀疏注意力 | KV 用 BF16 的平铺池 `[P, 512]`，每个 token 的窗口行和压缩行的下标拼成一个列表；入口见下 |

稀疏注意力的三个入口（都在 `aiter.ops.triton.attention` 下，算的是同一种注意力）：

| 入口 | 上游给它的定位 | 推荐用在 |
|---|---|---|
| `pa_decode_sparse` | decode：query 行少，kernel 把每个 token 的 KV 行再切分给多个 workgroup | decode（下一行的范围除外） |
| `sparse_mla_fwd` | Gluon 实现，decode 和 prefill 都能用；V4 的布局传 `kv_lora_rank=512, qk_rope_head_dim=0` | decode，64 / 128 头且 batch ≥ 64 |
| `pa_prefill_sparse` | prefill：按 query token 铺满 GPU（每个 token、每组头一个 workgroup），不切分 KV 行；query 行很少时大部分 GPU 是空的。gfx942 上走 Triton kernel | prefill |

这三个入口是各自独立的公开函数：aiter 没有把它们合在一起的统一入口，也没有对应的调优表（GEMM 和打分才有 CSV），用哪个由调用方决定。gfx942 上三者接受同一组输入（BF16 的 `q [T, H, 512]`、一个 BF16 的平铺 KV 池、int32 的 `kv_indices` / `kv_indptr`、`attn_sink`、`softmax_scale`），可以在接入层按上表分发。注意 `sparse_mla_fwd` 的参数顺序是 `kv_indptr` 在前、`kv_indices` 在后，返回值是 `(out, lse)`。

依据是 5.5 的数据。两个 decode 入口的 bench 条件不同（`pa_decode_sparse` 的 bench 反复读同一份数据、取连续位置，`sparse_mla_fwd` 的 bench 每次清 cache、取随机行，后者更不利），结论只取方向明确的部分：

- decode，8 / 16 头，640 行：`sparse_mla_fwd` 的耗时是 `pa_decode_sparse` 的 1.1–1.4 倍，这里面有 bench 条件的差别；按上游的定位用 `pa_decode_sparse`。
- decode，64 / 128 头、batch ≥ 64，640 行：`pa_decode_sparse` 的耗时是 `sparse_mla_fwd` 的 1.1–1.5 倍，而且 `sparse_mla_fwd` 是在更不利的条件下测的，这个范围用 `sparse_mla_fwd`。其余的点两者互有快慢。
- prefill：`sparse_mla_fwd` 的耗时是 `pa_prefill_sparse` 的 1.7–2.2 倍。两个 bench 每个 token 的平均行数不同（约 400 行对 590–630 行），按行数折算后仍是 1.2–1.4 倍，用 `pa_prefill_sparse`。

GEMM 调优表选中的 kernel：

| 层（N×K） | M | 后端 | splitK | kernel |
|---|---|---|---|---|
| `wq_b` 8192×1024 | 1 | cktile | 0 | `a8w8_blockscale_cktile_16x64x512_1x4x1_16x16x64_intrawave_0x1x0_2` |
| `wq_b` 8192×1024 | 4 | ck | 0 | `a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1` |
| `wq_b` 8192×1024 | 8 | ck | 0 | `a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1` |
| `wq_b` 8192×1024 | 32 | ck | 0 | `a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1` |
| `wq_b` 8192×1024 | 64 | ck | 0 | `a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1` |
| `wq_b` 8192×1024 | 128 | ck | 0 | `a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1` |
| `wq_b` 8192×1024 | 256 | asm | 1 | `fp8gemm_bf16_blockscale_BpreShuffle_64x128` |
| `wq_b` 8192×1024 | 4096 | opus | 0 | `opus_gemm_gfx942_a8w8_bs_bpreshuf_sb_tailm_v16_512x128x128x128_4x2_16x16x32_1x128x128` |
| `wq_b` 8192×1024 | 8192 | opus | 0 | `opus_gemm_gfx942_a8w8_bs_bpreshuf_sb_tailm_v16_512x128x128x128_4x2_16x16x32_1x128x128` |
| `wq_b` 8192×1024 | 16384 | opus | 0 | `opus_gemm_gfx942_a8w8_bs_bpreshuf_sb_tailm_v16_512x128x128x128_4x2_16x16x32_1x128x128` |

- 选中 OPUS 的形状（M ≥ 4096）要求传入连续的 `x_scale`，同 4.6。

## 附录：GEMM tune 模块的预编译脚本

4.2 里可选的一步，保存为 `${BENCH_DIR}/prebuild_gemm_tune_modules.py`。它从普通进程里把官方 GEMM tuner 的各个 tune 入口各调用一次，让 tune 模块用全部 CPU 核编译好，不参与任何测量。

```python
#!/usr/bin/env python3
"""Build the JIT modules the block-scale GEMM tuner needs, with the normal compile budget.

Inside the tuner every worker process is limited to one compile job
(aiter.utility.mp_tuner -> configure_worker_subprocesses), so a first build of the CKTile
tune module (163 translation units) would take about two hours. Calling each tune op once
from an ordinary process builds the same modules with all cores; the tuner then finds them
ready.

    export AITER_ROOT=/path/to/aiter
    python3 prebuild_gemm_tune_modules.py
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(os.environ["AITER_ROOT"])
sys.path.insert(0, str(ROOT / "csrc/ck_gemm_a8w8_blockscale"))  # the tuner imports its siblings
spec = importlib.util.spec_from_file_location(
    "_gemm_tune", ROOT / "csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py")
tune = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tune
spec.loader.exec_module(tune)

data = tune.generate_data(64, 128, 7168, seed=0)
x, w, xs, ws, out = data["x"], data["weight_shuffle"], data["x_scale_t"], data["w_scale"], data["out"]


def opus():
    gfx, cu_num = tune._device_arch_and_cu(torch.cuda.current_device())
    registry = tune.get_registry("a8w8_blockscale_bpreshuffle", gfx, cu_num)
    print(f"[prebuild] opus registry for {gfx}/{cu_num}: {len(registry)} kernels", flush=True)
    for kernel_id, kernel in registry.items():
        if 128 % kernel.B_N == 0 and 7168 % kernel.B_K == 0 and (kernel.has_oob or 64 % kernel.B_M == 0):
            return tune.run_gemm_a8w8_blockscale_opus(x, w, xs, ws, out, kernel_id)
    raise RuntimeError("no opus kernel fits the probe shape")


steps = [
    ("ck tune module", lambda: tune.run_gemm_a8w8_blockscale(x, w, xs, ws, out, 0, 0, True)),
    ("cktile tune module", lambda: tune.run_gemm_a8w8_blockscale_cktile(x, w, xs, ws, out, 0, 0, True)),
    ("asm backend", lambda: tune.run_gemm_a8w8_blockscale_asm(x, w, xs, ws, out, data["zero_bias"], None, 1, True)),
    ("opus backend", opus),
]
for name, call in steps:
    start = time.time()
    try:
        call()
        torch.cuda.synchronize()
        print(f"[prebuild] {name}: ready in {time.time() - start:.0f}s", flush=True)
    except Exception as error:  # noqa: BLE001 - a kernel id may be invalid; the build is what matters
        print(f"[prebuild] {name}: call raised {type(error).__name__}: {str(error).splitlines()[0][:200]} "
              f"after {time.time() - start:.0f}s", flush=True)
```
