# Recorded measurements referenced by the guide

These files contain the measurements quoted in the [guide](../../../../docs/mi300x-fp8-gemm-tuning-guide.md). PTPC uses the refreshed tuner-aligned input recipe. Block-scale kernel selections and native rates come from the user's complete 2026-11-24 ordinary/packed customer sweep. Public timings were refreshed with matching native-tuner inputs and independent signed validation.

| File | Contents |
|---|---|
| `ptpc-customer-replay.csv` | 42 ordinary/packed PTPC public replays on GPU 0, using the saved solution IDs and `gradlib_ptpc` inputs with seed 0; includes input metadata, solution identities, numerical errors and raw timing rounds. |
| `blockscale-customer-native.csv` | All 42 block-scale native winners, attempted/passing/rejected candidate counts, search settings, native latency/TFLOP/s and native error ratios. |
| `blockscale-customer-replay.csv` | All 42 verified block-scale public replays, exact dispatch identities, input recipe, median latency/TFLOP/s and sampled CPU-FP64 errors. |

The documented platform is MI300X with 304 CUs, ROCm 7.2.0, Torch 2.9.1, hipBLASLt 1.2.1 and FlyDSL 0.5.0. The PTPC measurement used `917b1dfb3` plus the local input-alignment change and the saved hipBLASLt table. The block-scale sweep used the manual checkout at the same revision plus local compiler-budget and OPUS-offset-guard fixes.

The block-scale native search used `--libtype all --shape_grouped --batch 8 --mp 8 --warmup 3 --iters 10`, with `--preshuffle --splitK` for packed weights. The source profiles contain 2,205 ordinary and 3,279 packed candidate attempts; every selected row matches the fastest passing candidate recorded for its shape and layout. Rejected candidates, including unsupported CKTile configurations, are counted in the attempt totals.

The public block-scale verifier uses logical GPU 0 and BF16 output. It checks the selected kernel in eager and graph execution with signed FNUZ inputs and K128 scales (seed 2026). It then generates the native block-scale tuner inputs (seed 0), checks those outputs as well, and reports the median of nine HIP-event samples with three GEMMs per graph. Timing preparation and both correctness checks are outside the measurement window. The main error columns are maxima over the two recipes; additional columns store each recipe's errors. All 42 rows passed relative RMS <0.006 and maximum absolute error/reference RMS <0.035. The supplied verification CSVs save medians and errors but not the nine individual timing samples or a physical GPU UUID; those fields have not been invented.

The native summary was consolidated from `customer-tuned/block-{ordinary,packed}.csv` and their `.profile.csv` records. Public results were refreshed on 2026-11-24 with the same selected rows and the local block-scale input-alignment change. Native and public timing inputs now match. Their GPU placement, buffer rotation and timing methods still differ, so the rates are not an API-overhead comparison or a matched speedup against global defaults.

To replay the documented selections, use `configs/ptpc.csv`, `configs/block-ordinary.csv` and `configs/block-packed.csv` from the parent folder. The block-scale configs preserve the supplied native winner rows unchanged. They replace the earlier customer snapshots named `block-ordinary-reference.csv` and `best-pf2.csv`; their new names cover all selected backends.

To evaluate current serving defaults, omit the verifier's `--table` argument. Retune and requalify on a different GPU or library version.

The separate 2026-11-24 packed-default comparison used the same native-tuner input tensors, scales and output buffer for each incumbent/candidate pair on GPU 0. Both kernels passed signed-input and native-input checks in eager and graph execution. Each process measured eight alternating-order rounds, three HIP-event samples per round and three complete GEMMs per graph. A replacement needed at least 3% higher median throughput in both screening and fresh-process confirmation, with positive gains in all four opposite-order round pairs in each process. Confirmation reversed the initial launch and graph-construction order. Gain is `100 * (incumbent_us / candidate_us - 1)`.

The initial 3% gate promoted two replacements: `(4096,32768,1536)` changed from FlyDSL PF1 ID 16 to PF2 ID 20 (5.47% screening, 4.80% confirmation), and `(65536,1536,4096)` changed from PF1 ID 16 to PF2 ID 22 (6.61%, 5.56%).

A subsequent absolute-gain exception was authorized for the two candidates with positive screening gains below 3%. Each needed more than 10 TFLOP/s improvement in both the original screen and a fresh-process confirmation, with positive gains in every opposite-order round pair and the same signed/native correctness checks. Both passed: `(16384,32768,1536)` changed from PF1 ID 16 to PF2 ID 23 (+15.96 TFLOP/s screening, +13.27 confirmation; +1.63%, +1.36%), and `(32768,1536,4096)` changed from PF1 ID 16 to PF2 ID 21 (+21.23, +22.82 TFLOP/s; +1.97%, +2.09%). Absolute gain is `candidate_tflops - incumbent_tflops`, calculated from unrounded paired measurements.

Four gfx942/304-CU rows in `aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv` were promoted in total; the five other proposed replacements were rejected. Promoted latency, TFLOP/s and bandwidth metadata use confirmation measurements, while `errRatio` retains the native winner's value. Ordinary defaults, model configs and the customer winner/result snapshots are unchanged by these promotions.
