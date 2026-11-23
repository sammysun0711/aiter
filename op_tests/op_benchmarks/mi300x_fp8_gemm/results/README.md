# Recorded measurements referenced by the guide

These files contain the measurements quoted in the [guide](../../../../docs/mi300x-fp8-gemm-tuning-guide.md). The PTPC replay was refreshed on 2026-11-24 with tuner-aligned inputs; the block-scale measurements remain historical reference data.

| File | Contents |
|---|---|
| `ptpc-customer-replay.csv` | 42 ordinary/packed PTPC public replays on GPU 0, using the saved solution IDs and `gradlib_ptpc` inputs with seed 0; includes input metadata, solution identities, numerical errors and raw timing rounds. |
| `blockscale-customer-native.csv` | 12 native-search winners at M=8,192 and M=131,072, candidate counts, selected kernels and public replay results. |
| `blockscale-customer-replay.csv` | 42 ordinary/packed block-scale public replays with identities, errors and timing samples. |

The documented environment was one MI300X with 304 CUs, ROCm 7.2.0, Torch 2.9.1, hipBLASLt 1.2.1 and FlyDSL 0.5.0. The block-scale measurements used PR29 (`f666f83`) plus local PF2/tuner changes; the PF2 runtime is now included in this branch. The refreshed PTPC measurement used `917b1dfb3` plus the local input-alignment change, with the same saved hipBLASLt solution table. The historical native searches used a narrower backend set than the current `--libtype all` examples.

To replay the recorded selections, use `configs/ptpc.csv`, `configs/best-pf2.csv`, and `configs/block-ordinary-reference.csv` from the parent folder. To evaluate current serving defaults, omit the verifier's `--table` argument. Retune and requalify on a different GPU or library version.

Earlier power, cross-scaling, per-library, chunk-plan and diagnostic experiments are not part of this customer package. Their removal does not change the retained final result CSVs.
