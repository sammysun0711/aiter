# Llama-2-70B Decode GEMM Tuning Static Analysis

## Scope

This note statically analyzes the unquantized BF16 decode GEMM shapes for
`llama2-70b-hf` running in the local vLLM checkout:

- vLLM source: `vllm/`
- version tag: `v0.25.0`
- target model assumption: Llama-2-70B HF config with `hidden_size=8192`,
  `intermediate_size=28672`, `num_attention_heads=64`,
  `num_key_value_heads=8`, `head_dim=128`, `vocab_size=32000`,
  `num_hidden_layers=80`
- primary tensor parallel size analyzed in detail: `tp_size=8`
- generated untuned update CSV coverage: `tp_size=1,4,8`
- dtype: BF16 input, BF16 output, no bias

The AITER CSV convention is:

```text
M = token rows in the activation matrix
N = output dimension = weight.shape[0]
K = input dimension = weight.shape[1]
```

This matches vLLM ROCm dispatch in
`vllm/vllm/model_executor/layers/utils.py`: `n = x.numel() // x.size(-1)`,
`m = weight.shape[0]`, `k = weight.shape[1]`. In this document, `M,N,K`
use the AITER CSV names, so `M` is vLLM local variable `n`.

## Source Path

For unquantized linear layers, vLLM calls `dispatch_unquantized_gemm()` through
`UnquantizedLinearMethod.apply` in `vllm/vllm/model_executor/layers/linear.py`
lines 220-228. On ROCm, `dispatch_unquantized_gemm()` returns
`rocm_unquantized_gemm` in `vllm/vllm/model_executor/layers/utils.py`
lines 332-334.

Llama uses these vLLM modules:

- `qkv_proj`: `QKVParallelLinear`, created in
  `vllm/vllm/model_executor/models/llama.py` lines 162-170.
- `o_proj`: `RowParallelLinear`, created in `llama.py` lines 172-178.
- `gate_up_proj`: `MergedColumnParallelLinear`, created in `llama.py`
  lines 92-99.
- `down_proj`: `RowParallelLinear`, created in `llama.py` lines 100-108.
- `lm_head`: `ParallelLMHead`, created in `llama.py` lines 484-490 and
  invoked through `LogitsProcessor._get_logits` in
  `vllm/vllm/model_executor/layers/logits_processor.py` lines 89-99.

Column-parallel linear layers shard output dimensions per rank in
`linear.py` lines 437-447. Row-parallel linear layers shard input dimensions
per rank in `linear.py` lines 1587-1592 and all-reduce their outputs in
lines 1688-1692.

## Per-Layer TP=8 Shapes

Per decoder layer, the strict TP=8 BF16 decode GEMMs are:

| Layer | vLLM module | Local weight `[N,K]` | Input `[M,K]` | Output `[M,N]` | Calls per decode step |
|---|---:|---:|---:|---:|---:|
| `self_attn.qkv_proj` | `QKVParallelLinear` | `[1280,8192]` | `[M,8192]` | `[M,1280]` | 80 |
| `self_attn.o_proj` | `RowParallelLinear` | `[8192,1024]` | `[M,1024]` | `[M,8192]` | 80 |
| `mlp.gate_up_proj` | `MergedColumnParallelLinear` | `[7168,8192]` | `[M,8192]` | `[M,7168]` | 80 |
| `mlp.down_proj` | `RowParallelLinear` | `[8192,3584]` | `[M,3584]` | `[M,8192]` | 80 |

Shape derivation:

- `qkv_proj`: `num_heads/rank = 64/8 = 8`, `num_kv_heads/rank = 8/8 = 1`,
  `head_dim = 128`. Per rank output is `q=8*128=1024`,
  `k=1*128=128`, `v=1*128=128`, so `N=1280`, `K=8192`.
- `o_proj`: attention output per rank is `1024`, row-parallel output is full
  hidden size, so `N=8192`, `K=1024`.
- `gate_up_proj`: gate and up are merged. Each is `28672/8=3584` per rank,
  so merged `N=7168`, `K=8192`.
- `down_proj`: row-parallel input is `28672/8=3584`, output is full hidden
  size, so `N=8192`, `K=3584`.

The decode `M` dimension is runtime-dependent. It is the number of tokens in
the compacted hidden-state batch for that forward pass, not sequence length.
The existing reference decode CSV uses:

```text
M = 1, 2, 4, 8, 16, 32, 64
```

With that policy, strict TP=8 per-layer tuning requires `4 * 7 = 28` rows:

```text
(M,1280,8192), (M,8192,1024), (M,7168,8192), (M,8192,3584)
for M in [1,2,4,8,16,32,64]
```

If the target workload can decode with more than 64 active tokens/sequences in
one model forward, add those `M` values too. The static `N,K` set does not
change.

## Logits GEMM

`lm_head` is also an unquantized linear-style GEMM when logits are computed.
For base Llama-2 vocab size `32000`, TP=8 gives `N=4000`, `K=8192`:

```text
(M,4000,8192)
```

This is not per decoder layer. It runs once when vLLM computes logits for the
sampled hidden states. Include this shape if the tuning target includes end to
end decode latency with logits enabled.

For TP=1/4/8, the logits shapes are:

| TP size | `lm_head` local `[N,K]` |
|---:|---:|
| 8 | `(4000,8192)` |
| 4 | `(8000,8192)` |
| 1 | `(32000,8192)` |

## Existing CSV Analysis

Reference input:

```text
llama2-gemm-tuning/tuning_result/llama70B_untuned_gemm_bf16_decode.csv
```

This file has 84 rows: 12 unique `(N,K)` pairs crossed with 7 `M` values.
Those 12 pairs are not just TP=8. They are TP=1, TP=4, and TP=8 variants of
the same four logical projections:

| Logical projection | TP=8 | TP=4 | TP=1 |
|---|---:|---:|---:|
| `qkv_proj` | `(1280,8192)` | `(2560,8192)` | `(10240,8192)` |
| `o_proj` | `(8192,1024)` | `(8192,2048)` | `(8192,8192)` |
| `gate_up_proj` | `(7168,8192)` | `(14336,8192)` | `(57344,8192)` |
| `down_proj` | `(8192,3584)` | `(8192,7168)` | `(8192,28672)` |

For the requested TP=8 case, the necessary per-layer subset is:

```text
(1280,8192), (8192,1024), (7168,8192), (8192,3584)
```

Updated input:

```text
llama2-gemm-tuning/tuning_result/llama70B_untuned_gemm_bf16_decode_update.csv
llama2-gemm-tuning/aiter/aiter/configs/model_configs/llama70B_untuned_gemm_bf16_decode_update.csv
```

The two update CSVs are identical. They keep the original 84 decoder-layer
rows and add `lm_head` rows for TP=1/4/8, using the same `M` set
`1,2,4,8,16,32,64`. The update file has 105 rows, 105 unique `(M,N,K)`
keys, no duplicate keys, and 15 unique `(N,K)` shape families:

| Logical projection | TP=8 | TP=4 | TP=1 |
|---|---:|---:|---:|
| `qkv_proj` | `(1280,8192)` | `(2560,8192)` | `(10240,8192)` |
| `o_proj` | `(8192,1024)` | `(8192,2048)` | `(8192,8192)` |
| `gate_up_proj` | `(7168,8192)` | `(14336,8192)` | `(57344,8192)` |
| `down_proj` | `(8192,3584)` | `(8192,7168)` | `(8192,28672)` |
| `lm_head` | `(4000,8192)` | `(8000,8192)` | `(32000,8192)` |

Current tuned/reference result:

```text
llama2-gemm-tuning/tuning_result/bf16_tuned_gemm.csv
```

For `gfx942`, this file currently has incomplete coverage for strict TP=8:

| TP=8 shape | Present `M` values in `bf16_tuned_gemm.csv` | Gap |
|---|---:|---|
| `(1280,8192)` | `1,2,4,8,16,64` | missing `M=32` |
| `(8192,1024)` | `1,2,4,8` | missing `M=16,32,64` |
| `(7168,8192)` | none | all `gate_up_proj` decode rows missing |
| `(8192,3584)` | `1,2,4,8,16` | missing `M=32,64` |
| `(4000,8192)` | none | logits shape missing |

The all-candidate profile has valid `gfx942` candidates for the missing
`gate_up_proj` shape `(7168,8192)`, but they were not copied into the final
tuned CSV. Filtered for `us > 0` and `err_ratio < 0.05`, the best candidates
from `llama70B_gemm_bf16_decode_all_candidate.csv` are:

| Shape | M | Best valid candidate |
|---|---:|---|
| `(7168,8192)` | 1 | `hipblaslt`, 34.3464 us |
| `(7168,8192)` | 2 | `torch`, 35.5149 us |
| `(7168,8192)` | 4 | `hipblaslt`, 35.4045 us |
| `(7168,8192)` | 8 | `hipblaslt`, 35.4429 us |
| `(7168,8192)` | 16 | `torch`, 35.3888 us |
| `(7168,8192)` | 32 | `torch`, 38.1083 us |
| `(7168,8192)` | 64 | `hipblaslt`, 46.4870 us |

The validation scripts
`run_llama70b_tunned_gemm_bench_normal.sh` and
`run_llama70b_tunned_gemm_bench_skinny.sh` also omit `(7168,8192)`, so they
do not exercise the TP=8 `gate_up_proj` GEMM.

## vLLM gfx942 tgemm Patch

The local vLLM checkout was patched so AITER tuned BF16 GEMM can be reached
on MI300X `gfx942`.

In `vllm/vllm/model_executor/layers/utils.py`, ROCm unquantized GEMM dispatch
checks paths in this order:

1. gfx950 skinny reduce-counting path.
2. AITER Triton GEMM for a small hardcoded shape set, which does not include
   the Llama-2-70B shapes above.
3. ROCm skinny GEMM for `on_gfx9()`, enabled by `VLLM_ROCM_USE_SKINNY_GEMM`.
   On `gfx942`, this intercepts `0 < M <= 5` when `K % 8 == 0`.
4. `rocm_aiter_ops.is_tgemm_enabled()`.
5. Fallback `torch.nn.functional.linear`.

Before the patch, `vllm/vllm/_aiter_ops.py` gated `is_tgemm_enabled()` with
`on_gfx950()`, which excluded `gfx942`. The local patch explicitly allows
both `gfx942` and `gfx950`:

```diff
-        from vllm.platforms.rocm import on_gfx950
+        from vllm.platforms.rocm import on_gfx942, on_gfx950

-        return cls.is_linear_enabled() and on_gfx950()
+        return cls.is_linear_enabled() and (on_gfx942() or on_gfx950())
```

Implications:

- With the local patch, `tgemm.mm` is reachable on `gfx942` when AITER linear
  is enabled.
- For `M=1,2,4`, `VLLM_ROCM_USE_SKINNY_GEMM=1` will still intercept before
  `tgemm`; disable skinny GEMM when isolating AITER tuned GEMM performance.
- The provided AITER `op_tests/test_gemm_a16w16.py` scripts can benchmark
  tuned rows directly, but that does not prove stock vLLM will use them.

Minimum runtime knobs for `tgemm`:

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_LINEAR=1
```

For an isolation run where even `M=1,2,4` should fall through to `tgemm`:

```bash
export VLLM_ROCM_USE_SKINNY_GEMM=0
```

With skinny left enabled, Llama-2-70B decode shapes at `M>=8` should be able
to reach `tgemm` after this patch.

## Recommended TP=8 Tuning Set

For per-layer decode GEMM tuning only:

```text
M values: 1,2,4,8,16,32,64
N,K values:
  1280,8192   # qkv_proj
  8192,1024   # o_proj
  7168,8192   # gate_up_proj
  8192,3584   # down_proj
```

For end to end decode including logits:

```text
add:
  4000,8192   # lm_head for vocab_size=32000, tp_size=8
```

The existing `tune_scripts.sh` command is structurally the right tuner path:

```text
python csrc/gemm_a16w16/gemm_tuner.py \
  --input_file aiter/configs/model_configs/llama70B_untuned_gemm_bf16_decode_update.csv \
  --libtype all \
  --with-hipblaslt \
  --compare \
  --update_improved \
  --profile_file llama70B_gemm_bf16_decode_all_candidate.csv
```

The update CSV now exists at the `--input_file` path used by the script and
contains the TP=1/4/8 decoder-layer and `lm_head` shapes. After tuning,
verify that the final tuned CSV has rows for every required `(M,N,K)` row,
not just candidate profile entries.

## Practical Next Steps

1. Tune on `gfx942` with the existing AITER tuner and the generated
   `llama70B_untuned_gemm_bf16_decode_update.csv` input.
2. Confirm the final tuned CSV contains all required rows. The current
   `bf16_tuned_gemm.csv` is missing the TP=8 `gate_up_proj` shape.
3. Verify vLLM dispatch on `gfx942` with the local explicit
   `on_gfx942() or on_gfx950()` tgemm patch.
4. Benchmark with `VLLM_ROCM_USE_SKINNY_GEMM=0` to isolate AITER tuned GEMM,
   then benchmark again with the production skinny setting to measure real
   end to end impact.
5. Use the normal and skinny op-test scripts only as kernel-level checks.
   They should be updated to include the update CSV shapes, especially
   `(M,7168,8192)` and the `lm_head` rows.
