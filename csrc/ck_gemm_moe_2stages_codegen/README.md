# CK GEMM MoE 2-Stages Codegen or asm 1-stage Tune

1. Install aiter:
`cd $aiter_path`
`python3 setup.py develop`

2. Add MoE shapes in `aiter/configs/untuned_fmoe.csv`
    |**token**|**model_dim**|**inter_dim**|**expert**|**topk**|**act_type**|**dtype**|**q_dtype_a**|**q_dtype_w**|**q_type**|**use_g1u1**|**doweight_stage1**|
    |---------|-------------|-------------|----------|--------|------------|---------|-------------|-------------|----------|------------|-------------------|
    |1024     |4096         |14336        |8         |2       |ActivationType.Silu|dtypes.bf16|dtypes.fp8|dtypes.fp8|QuantType.per_Token|True|True|


3. Start tuning:
Run the following cmd to start tuning, please wait a few minutes as it will build moe 2-stages kernels via jit:
`python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py -i aiter/configs/untuned_fmoe.csv -o aiter/configs/tuned_fmoe.csv`
You can find the results of this tuning in `aiter/configs/tuned_fmoe.csv`, like this:
    |**cu_num**|**token**|**model_dim**|**inter_dim**|**expert**|**topk**|**act_type**|**dtype**|**q_dtype_a**|**q_dtype_w**|**q_type**|**use_g1u1**|**doweight_stage1**|**block_m**|**ksplit**|**us1**|**kernelName1**|**err1**|**us2**|**kernelName2**|**err2**|**us**|**run_1stage**|**tflops**|**bw**|
    |----------|---------|-------------|-------------|----------|--------|------------|---------|-------------|-------------|----------|------------|-------------------|-----------|----------|-------|---------------|--------|-------|---------------|--------|------|--------------|----------|------|
    |80        |1024     |4096         |14336        |8         |2       |ActivationType.Silu|dtypes.bf16|dtypes.fp8|dtypes.fp8|QuantType.per_Token|True|True|64|0|45.23|kernel_stage1|0.5%|38.67|kernel_stage2|0.3%|83.90|0|125.4|89.5|

    `cu_num` means the number of compute units, and it is used to distinguish between graphics.
    `run_1stage` indicates whether to run fused 1-stage kernel (1) or 2-stages kernels (0).

4. Build tuned kernels and test:
Test the performance, modify the test instance in `op_tests/test_moe.py` or `python3 op_tests/test_moe_2stage.py` and run it, please wait a few minutes as it will build moe tuned kernels in `aiter/configs/tuned_fmoe.csv` via jit:
`python3 op_tests/test_moe.py` or `python3 op_tests/test_moe_2stage.py`
If you have built moe kernels before tuning new MoE shapes, please add `AITER_REBUILD=1` before your test cmd, such as `AITER_REBUILD=1 python3 op_tests/test_moe.py`. It will rebuild kernels from `AITER_CONFIG_FMOE`, the default one will be results merged from `aiter/configs/tuned_fmoe.csv` and tuned fmoe csv under `aiter/configs/model_configs/xx_tuned_fmoe_xx.csv`, the merged result is stored in `/tmp/aiter_configs/tuned_fmoe.csv`.

## More Options

### Tuning Scope

#### `--last`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Only tune the last kernel in the CSV file. Useful for quickly testing newly added shapes.

**Example**:
```bash
--last
```

### Output Configuration

#### `-o2, --profile_file`
- **Type**: String
- **Default**: `""` (empty string)
- **Description**: Optional output file to store **all** tuning results (not just the best ones). Useful for profiling and analyzing all kernel candidates.

**Example**:
```bash
--profile_file aiter/configs/profile_fmoe_all.csv
```

### Tuning Configuration

#### `--errRatio`
- **Type**: Float
- **Default**: `0.5` (50%)
- **Description**: Tolerable error ratio threshold. Only kernels with error ratios below this threshold will be considered valid candidates.

**Example**:
```bash
--errRatio 0.01
```

#### `--mp`
- **Type**: Integer
- **Default**: Number of available GPUs
- **Description**: Number of parallel processes to use for tuning across multiple GPUs.

**Example**:
```bash
--mp 8
```

#### `--batch`
- **Type**: Integer
- **Default**: `100`
- **Description**: Number of shapes to tune in each batch.

**Example**:
```bash
--batch 50
```

#### `-k, --splitK`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Enable split-K optimization for GEMM kernels. Split-K divides the K dimension across multiple workgroups to improve parallelism and performance for certain shapes.

**Example**:
```bash
-k
--splitK
```
#### `--sort`
- **Type**: Boolean (True/False)
- **Default**: `False` 
- **Description**: Sort the output file according to the key columns

**Example**:
```bash
--sort True
```

#### `--all`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Retune all shapes based on file relationship.
- If `tune_file` == `untune_file`: Retune all shapes in the tune file
- If `tune_file` != `untune_file`: Retune shapes that exist in untuned file


**Example**:
```bash
--all
```

#### `--run_config [TUNED_CSV]`
- **Type**: Optional argument
- **Default**: disabled
- **Description**: Run production-operator benchmark only and exit (no tuning).
  - `--run_config /path/to/tuned.csv`: read shapes from that tuned CSV and run tuned kernels from that file.
  - `--run_config` (no path): read shapes from `-i/--untune_file` and run default kernels.

**Examples**:
```bash
# benchmark tuned kernels from specified tuned config
python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  --run_config aiter/configs/tuned_fmoe.csv

# benchmark default kernels using shapes from -i
python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  -i aiter/configs/untuned_fmoe.csv --run_config
```

#### `--compare`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Run pre-tune and post-tune production benchmark, print compare results, and keep a compare candidate CSV.
  - Pre-tune reads shapes from `-i/--untune_file`.
  - Post-tune uses configs written to `<tune_file>.candidate.csv` during the compare run.
  - The final tuned CSV is only updated when `--update_improved` is also set.
  - Shapes with no valid pre-run baseline can still update when the post-tune benchmark passes.

**Example**:
```bash
--compare
```

#### `--update_improved`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: With `--compare`, update the final tuned CSV for shapes improved by at least `--min_improvement_pct`, or for shapes with no valid pre-run baseline when the post-tune benchmark passes.

**Example**:
```bash
--compare --update_improved
```

#### `--min_improvement_pct`
- **Type**: Float
- **Default**: `3.0`
- **Description**: With `--compare --update_improved`, the minimum percentage improvement required before a compared result replaces the final tuned CSV entry when both pre/post benchmarks are valid. Shapes with no valid pre-run baseline but passing post-tune are still allowed to update.

### Profiling Configuration

#### `--warmup`
- **Type**: Integer
- **Default**: `5`
- **Description**: Number of warmup iterations before profiling.

**Example**:
```bash
--warmup 10
```

#### `--iters`
- **Type**: Integer
- **Default**: `101`
- **Description**: Number of profiling iterations to run for performance measurement.

**Example**:
```bash
--iters 200
```

#### `--timeout`
- **Type**: Integer
- **Default**: `None`
- **Description**: Timeout in seconds for each task group.

**Example**:
```bash
--timeout 300
```

### Debugging and Verbose Output

#### `-v, --verbose`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Enable verbose output with detailed logging information.

**Example**:
```bash
-v
```

## Notes
- This tuner supports both 1-stage fused MoE kernels and 2-stages MoE kernels (stage1 and stage2)
- The tuner will automatically select the best kernel configuration based on performance
- Only G1U1 (gate-up fused) MoE configurations are currently supported for tuning
- Supported quantization types include: per_Token, per_1x128 (blockscale), per_1x32 (MXFP4, gfx950 only)
- If you use flag `PREBUILD_KERNELS=1` when you install aiter, it will build moe kernels in tuned csv by default. If you want to use the new result of moe tuning, please remove `build` and `*.so` in `aiter/jit` first, then re-install aiter after finishing tune. This can take a lot of time and is not recommended.


## FlyDSL EP FP8 Block128 tuning on gfx950

`--ep-fp8-flydsl` selects the EP tuner through the same entry point as the
native MoE tuner. It searches this backend's complete local MoE pipeline and
saves the lowest-latency numerically valid configuration. ASM and old-kernel
comparisons belong in separate benchmarks.

Defaults are under `aiter/configs/`:

- Input: `flydsl_ep_decode_fp8_block128_gfx950_untuned_fmoe.csv`
- Output: `flydsl_ep_decode_fp8_block128_gfx950_tuned_fmoe.csv` (created by tuning)

`token` is the physical input capacity C, `expert` is the local expert count, and
`global_expert / expert` is the logical EP size. One GPU simulates the last
rank. Tuning uses R=C; dynamic R<C remains an operator-test case and is not a
tuning input for now. `source_tokens` is a separate CPU workload hint; -1 means absent.

```csv
token,model_dim,inter_dim,expert,global_expert,topk,source_tokens
128,4096,1024,32,256,8,16
128,4096,1024,16,256,8,8
```

Backend-specific fixture options use one JSON object:

| Field | Values | Default |
| --- | --- | --- |
| `routing` | `uniform`, `randn`, `hot` | `uniform` |
| `hot_experts` | Positive integer, at most the local expert count; hot only | `min(topk, local experts)` |

Uniform samples FP32 scores from U(5,10) and normalizes sigmoid routing weights.
Randn uses normally distributed scores. Hot adds 20 to the scores of the first
N local experts; when N<top-k, remaining slots select other experts without
duplicating IDs. Unknown JSON keys and invalid options fail explicitly.
`TUNE_MOE_EXPERT_BALANCE` does not affect this tuner.

```bash
cp aiter/configs/flydsl_ep_decode_fp8_block128_gfx950_untuned_fmoe.csv my_model_untuned_fmoe.csv
# Edit the copied shape list for your model before tuning.
python csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  --ep-fp8-flydsl -i my_model_untuned_fmoe.csv -o my_model_tuned_fmoe.csv -o2 my_model_candidates.csv \
  --ep-fp8-flydsl-tune-kwargs '{"routing":"uniform"}' \
  --mp 1 --warmup 5 --iters 101

# Retune the same shapes for a chosen hot-expert set and replace their rows.
python csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  --ep-fp8-flydsl -i my_model_untuned_fmoe.csv -o my_model_tuned_fmoe.csv --all \
  --ep-fp8-flydsl-tune-kwargs '{"routing":"hot","hot_experts":8}'

# Reproduce the saved fixture through public fused_moe dispatch.
python csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  --ep-fp8-flydsl --run_config my_model_tuned_fmoe.csv

# Evaluate that same saved configuration on another routing; no tuning/writes.
python csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  --ep-fp8-flydsl --run_config my_model_tuned_fmoe.csv \
  --ep-fp8-flydsl-tune-kwargs '{"routing":"randn"}'
```

Runtime uses the existing `AITER_CONFIG_FMOE` variable. A model CSV can hold
both native and EP rows. The EP rows carry
`_tag=flydsl_ep_decode_fp8_block128_gfx950`; the native loader ignores them.
EP shape identity includes gfx/CU/C/H/I/local experts/global experts/top-k and
source hint. Native rows retain their native keys.

Do not load different routing profiles for the same shape together. `tune_kwargs` records the effective options for reproduction, not runtime
selection. `--all` replaces the same backend/shape row; it does not retain one
runtime row per routing. A failed shape leaves its previous valid row intact.
Without `--all`, already tuned shapes are skipped. The source-hint heuristic
remains the fallback when no matching FlyDSL profile exists; TP and unsupported
inputs still delegate to native AITER.

```bash
AITER_FLYDSL_MOE_EP_DECODE_FP8_BLOCK128_GFX950=1 \
AITER_CONFIG_FMOE=my_model_tuned_fmoe.csv \
python op_tests/test_moe_blockscale.py \
  -d bf16 -m 128 -dim 4096 -idim 1024 -e 256 -k 8 -ep 8 \
  --tokens-before-dispatch 16 --routing hot --hot-experts 8
```

To retain native tuned rows in another file, pass both via
`AITER_CONFIG_FMOE=/path/native.csv:/path/my_model_tuned_fmoe.csv`. Model tables under
`aiter/configs/model_configs/*tuned_fmoe*.csv` also participate in the existing
default scan. Restart an application after changing its offline configuration.

Timing is complete local MoE GPU-event mean, including sorting, required
quantization and output reduction, excluding network dispatch/combine. Four
input/weight address sets are rotated. `--iters` is rounds per captured graph;
five graph replays are averaged. `--warmup` is untimed rounds before capture.

The adapter supports `--mp 1`, with no split-K or per-candidate watchdog.
`--last`, `--run_config`, and the common optional `--compare` workflow remain
available. Candidate records, including failures, are written to `-o2`.
The standalone test and the tuner share fixtures, reference and timing; the
public test/replay also assert that FlyDSL was actually selected. This does
not imply one routing's best configuration is optimal for every routing.
