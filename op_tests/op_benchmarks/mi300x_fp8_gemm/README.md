# MI300X FP8 GEMM customer tools

Start with the [tuning guide](../../../docs/mi300x-fp8-gemm-tuning-guide.md). Native tuning is provided by AITER's existing tuners under `csrc/`; this folder contains their customer inputs and public-API verification tools.

| Tool | Purpose |
|---|---|
| `benchmark_fp8_blockscale_gemm.py` | Validate and time complete PTPC or K128 block-scale GEMMs, for one shape or a shape CSV. Supports ordinary and packed weights, explicit tuned tables and current defaults. |
| `benchmark_fp8_ptpc_gemm.py` | Replay ordinary/packed PTPC in paired timing rounds, writing per-shape timing samples and numerical checks. |
| `qualify_ptpc.py` | Optionally compare native hipBLASLt finalists with an incumbent PTPC table. |
| `qualify_blockscale.py` | Optionally screen native block-scale candidates with signed CPU-FP64 checks and graph replay. |
| `common.py` | Shared inputs, references, table validation and graph timing; not a CLI. |

## Inputs and saved tables

- `configs/ptpc-tune-input.csv`: 21 customer shapes × ordinary/packed metadata for fresh PTPC tuning.
- `configs/block-shapes.csv`: all 21 M,N,K combinations for public replay; follow the guide's separate 8K and 128K native tuning commands.
- `configs/ptpc.csv`: the saved hipBLASLt solutions used by the guide's PTPC results. Solution IDs depend on the installed hipBLASLt version.
- `configs/best-pf2.csv`: the saved PF2/ASM table used by the guide's packed block-scale results.
- `configs/block-ordinary-reference.csv`: the ordinary kernel selections reconstructed from the guide's final result CSV.
- `configs/packed-pf1-asm.csv`: the PR29-compatible packed PF1/ASM alternative, for installations without PF2.

The saved tables reproduce historical dispatch choices. Current serving defaults live in `aiter/configs/`; omit `--table` from the verifier to evaluate those defaults. They can differ from the historical selections after further tuning.

## Validation and timing

Use an idle 304-CU gfx942 MI300X. Inputs are signed E4M3 FNUZ values, scales are FP32, and output is BF16. Both eager and graph execution are checked against sampled CPU FP64 references. Limits are relative RMS <0.006 and maximum absolute error/reference RMS <0.035.

PTPC tools use gradlib's seed-0 input recipe: draw activations and weights with `torch.randn(...).to(BF16)`, then apply `aiter.pertoken_quant` to both and use the returned scales. Result metadata records `input_recipe=gradlib_ptpc` and `input_seed=0`. Block-scale inputs retain the existing seed-2026 signed FP8 recipe and K128 scales.

Packing, input preparation and JIT compilation precede timing. The verifier measures nine samples with three GEMMs per captured graph. The paired PTPC benchmark defaults to eight rounds × three samples, with three GEMMs per graph. Reported TFLOP/s is `2*M*N*K/(median_us*1e6)`.

Explicit runtime rows must match the dispatch lookup. FlyDSL rows additionally validate kernel ID, full name, weight layout and supported shape against the installed registry. Errors cause a nonzero exit; an incomplete block-scale qualification does not publish a winner table.

[Recorded final measurements and provenance](results/README.md).
