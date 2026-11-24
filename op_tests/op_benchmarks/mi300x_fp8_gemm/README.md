# MI300X FP8 GEMM customer tools

Start with the [tuning guide](../../../docs/mi300x-fp8-gemm-tuning-guide.md). Native tuning is provided by AITER's existing tuners under `csrc/`; this folder contains their customer inputs and public-API verification tools.

| Tool                                 | Purpose                                                                                                                                                                    |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `benchmark_fp8_blockscale_gemm.py` | Validate and time complete PTPC or K128 block-scale GEMMs, for one shape or a shape CSV. Supports ordinary and packed weights, explicit tuned tables and current defaults. |
| `benchmark_fp8_ptpc_gemm.py`       | Replay ordinary/packed PTPC in paired timing rounds, writing per-shape timing samples and numerical checks.                                                                |
| `qualify_ptpc.py`                  | Optionally compare native hipBLASLt finalists with an incumbent PTPC table.                                                                                                |
| `qualify_blockscale.py`            | Optionally screen native block-scale candidates with signed CPU-FP64 checks and graph replay.                                                                              |
| `common.py`                        | Shared inputs, references, table validation and graph timing; not a CLI.                                                                                                   |

## Inputs and saved tables

- `configs/ptpc-tune-input.csv`: 21 customer shapes × ordinary/packed metadata for fresh PTPC tuning.
- `configs/block-shapes.csv`: all 21 M,N,K combinations for native block-scale tuning and public replay in both layouts.
- `configs/ptpc.csv`: the saved hipBLASLt solutions used by the guide's PTPC results. Solution IDs depend on the installed hipBLASLt version.
- `configs/block-packed.csv`: 21 verified native winners from full sweep, including ASM, CK, FlyDSL PF1 and PF2 selections.
- `configs/block-ordinary.csv`: 21 verified ordinary-weight native winners from the same full sweep.

The saved tables reproduce the guide's documented dispatch choices. Use `--table` to select these customer snapshots. Current serving defaults live in `aiter/configs/`; omit `--table` to evaluate those defaults. The two sets of selections can differ.

## Validation and timing

Use an idle 304-CU gfx942 MI300X. Inputs use E4M3 FNUZ values, FP32 scales and BF16 output. Both eager and graph execution are checked against sampled CPU FP64 references. Limits are relative RMS <0.006 and maximum absolute error/reference RMS <0.035.

PTPC tools use gradlib's seed-0 input recipe: draw activations and weights with `torch.randn(...).to(BF16)`, then apply `aiter.pertoken_quant` to both and use the returned scales. Result metadata records `input_recipe=gradlib_ptpc` and `input_seed=0`. Block-scale timing uses `native_blockscale`: seed 0, FP16 `torch.rand / 10` cast to FP8, and uniform FP32 K128 scales, matching the native tuner. The public benchmark first checks the selected kernel with independent `block128_signed` inputs (seed 2026), then verifies and times the native inputs. CSV fields identify both recipes and their separate errors; the main error columns are the maximum across both. `qualify_blockscale.py` continues to use signed inputs for candidate screening.

Packing, input preparation and JIT compilation precede timing. The verifier measures nine samples with three GEMMs per captured graph. The paired PTPC benchmark defaults to eight rounds × three samples, with three GEMMs per graph. Reported TFLOP/s is `2*M*N*K/(median_us*1e6)`.
