# Qwen3.8 TP8 MoE evaluation bundle

This directory contains the scripts, final tuned configurations, and source
results used to produce `results/qwen3_8_tp8_tuned_moe_all_e2e.csv`.

The bundle is intentionally minimal. AITER's own tuner, standard MoE test,
quantizers, routing implementation, and kernels are reused from the parent
repository instead of being copied here.

## Evaluated shape

- GPU: AMD Instinct MI308X (`gfx942`, 80 CUs)
- hidden size: 8192
- global expert intermediate size: 2048
- TP8 intermediate size: 256
- experts: 512
- top-k: 10
- M: `1 4 8 16 32 64 256 1024 2048 4096 8192 16384 32768`

## Timing scopes

The CSV columns are local MoE timings, not full-model inference latency.

- `native_mxfp4_a16w4_topk_sort_expert_us` includes top-k routing, expert
  sorting, two A16W4 GEMMs, SiLU/multiply, and routed reduction. Router
  projection and TP collectives are excluded.
- `converted_int4_flydsl_wi4a16_sort_expert_us` includes sorting and the
  FlyDSL expert path. Top-k and offline MXFP4-to-INT4 conversion are excluded.
- `bf16_ck_tile_sort_expert_us` includes sorting and the BF16 expert path.
  Top-k is excluded.
- `fp8_ptpc_activation_quant_sort_expert_us` includes activation quantization,
  sorting, and the FP8 PTPC expert path. Top-k is excluded.

## Requirements

The bundle now lives inside a complete AITER checkout and defaults `AITER_ROOT`
to its parent directory. Set `AITER_ROOT` explicitly to evaluate another AITER
checkout.

The saved measurements were produced with AITER commit
`c16d44b93a528b2a4bfd6d8d3409116d465872a9`, ROCm 7.2, PyTorch
`2.11.0+rocm7.2`, and Triton
`3.7.0+amd.rocm7.2.0.git89002410`.

The bundle commit is based directly on the same measured AITER revision,
`c16d44b93a528b2a4bfd6d8d3409116d465872a9`. With the bundle checked out in
this repository, the default parent-directory `AITER_ROOT` therefore uses the
revision that produced the saved measurements.

The measured AITER checkout also has a local `torch.Stream` compatibility
change in `csrc/cpp_itfs/torch_utils.py`; preserve that patch in the production
image or confirm that the target AITER revision contains an equivalent fix.

## Reproduce all E2E measurements

Choose an idle GPU and run:

```bash
cd /root/workspace/moe-test/aiter/mxfp4_moe_eval_scripts
HIP_VISIBLE_DEVICES=0 ./run_all_evaluations.sh
```

The script runs the native A16W4 wrapper and AITER's production `fused_moe`
benchmark with each tuned CSV. It then rebuilds the consolidated CSV from the
new source results.

## Included files

- `qwen3_8_mi308x_a16w4.py`: Qwen-specific native MXFP4 composition and tuned
  dispatch selection.
- `benchmark_qwen3_8_tuned_wrapper.py`: native A16W4 local-MoE E2E benchmark.
- `run_all_evaluations.sh`: invokes the native benchmark and AITER's existing
  `gemm_moe_tune.py --run_config` path for BF16, PTPC, and FlyDSL.
- `collect_e2e_results.py`: rebuilds the final CSV from the four source JSONs.
- `tuning/`: only the four final production/evaluation configurations.
- `results/`: only the four source JSONs and consolidated final CSV.

To rebuild the final CSV without rerunning GPU benchmarks:

```bash
python collect_e2e_results.py
```

## Standard AITER validation

The bundle does not copy AITER source files. It reuses `test_moe_2stage.py`,
`test_moe_gemm_a16w4.py`, `gemm_moe_tune.py`, and all kernels directly from
the parent AITER checkout:

```bash
AITER_ROOT="$(cd .. && pwd)"
AITER_CONFIG_FMOE="$PWD/tuning/qwen3_8_tp8_bf16_tuned.csv" \
PYTHONPATH="$AITER_ROOT" HIP_VISIBLE_DEVICES=0 \
python "$AITER_ROOT/op_tests/test_moe_2stage.py" \
  --no-flydsl-csv -d bf16 -dim 8192,256 \
  -t 1 4 8 16 32 64 256 1024 2048 4096 8192 16384 32768 \
  -q 0 -a silu -s f -e 512 -k 10 -p t -hip 0,0
```

Use the same command with the PTPC config and `-q 2` for FP8 PTPC.

## Production recommendation

For a full TP8 model on 192 GiB MI308X, use
`single_layout_production_dispatch` from the native A16W4 deployment JSON.
BF16 and FP8 PTPC are faster references but exceed the full-model memory
budget. FlyDSL requires a lossy conversion from MXFP4 to integer INT4.

Only Qwen-specific orchestration is retained locally; implementation,
quantization, routing, tuning, and standard test functions come from AITER.
