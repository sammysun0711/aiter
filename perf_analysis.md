# Llama-2-70B BF16 vs FP16 GEMM Performance Analysis (MI325X / gfx942)

Target: Llama-2-70B-hf inference on AMD Instinct MI325X (`gfx942`, cu_num 304),
vLLM + AITER, TP=8 decode focus. All findings below were verified against the
running server logs, profiler traces, and tuning CSVs in this workspace.

## TL;DR

- The model ships as **fp16** (`config.json: torch_dtype=float16`). AITER's GEMM
  tuner only tunes the fast custom kernels (`flydsl`, `opus`, `asm`) for **bf16**
  on gfx942 — fp16 is limited to `hipblaslt`/`torch`.
- On matched (M,N,K) shapes, **fp16 tgemm ≈ bf16 tgemm** overall (fp16 faster on
  33/105 shapes, bf16 faster on 38/105, ~equal on 34). Neither dtype is a
  systematic winner via hipBLASLt.
- The **only real bf16 advantage** is on shapes where a custom kernel wins that
  fp16 can't access — most notably **`o_proj (8192,1024)`** (bf16 `flydsl`
  ~15-20% faster) and **`down_proj (8192,3584)`** (bf16 `opus`, ~neutral vs
  hipBLASLt).
- Verified live: a bf16 TP=8 run dispatches the tuned **opus** down_proj kernel
  (~11.9% of GPU time). But `qkv_proj` and `o_proj` are missing from the current
  tuned table, so the largest single GEMM cost still falls back to hipBLASLt.

## Why tgemm was missing initially

vLLM ROCm unquantized dispatch order (`vllm/model_executor/layers/utils.py`
`rocm_unquantized_gemm_impl`): gfx950 skinny reduce → AITER Triton GEMM (small
hardcoded set, excludes Llama-70B) → ROCm skinny GEMM (`VLLM_ROCM_USE_SKINNY_GEMM`)
→ `rocm_aiter_ops.is_tgemm_enabled()` → `tgemm.mm` → torch fallback.

- `is_tgemm_enabled()` (`vllm/_aiter_ops.py:1788`) is already patched to
  `on_gfx942() or on_gfx950()`, so tgemm **is** reachable on MI325X with
  `VLLM_ROCM_USE_AITER=1` + `VLLM_ROCM_USE_AITER_LINEAR=1` (default true).
- tgemm was being reached, but `tgemm.mm` (`aiter/tuned_gemm.py`) looked up the
  shape, missed, and fell through to its `default_config` → `libtype="torch"`
  (i.e. `F.linear` → hipBLASLt `Cijk_*` kernels). Root cause of the miss:
  **dtype mismatch** — the model runs fp16 but every tuned row was bf16.

## Config file plumbing (the real gotchas)

The runtime GEMM config lives at `/tmp/aiter_configs/bf16_tuned_gemm.csv`, built
on import by `AITER_CONFIG.get_config_file` (`aiter/jit/core.py`):

1. It **globs `aiter/configs/model_configs/*bf16_tuned_gemm*.csv`** (excluding
   `*untuned*`), merges them with the default `configs/bf16_tuned_gemm.csv`, and
   writes the union to `/tmp/aiter_configs/bf16_tuned_gemm.csv`.
2. **Filename must contain the substring `bf16_tuned_gemm`** or the glob skips it.
   - `llama70B_fp16_tuned_gemm.csv` → ignored (no `bf16_tuned_gemm`).
     Renamed to `llama70B_fp16_bf16_tuned_gemm.csv` → picked up.
   - `llama70B_tuned_gemm_bf16_decode.csv` → ignored (word order flipped).
     Renamed to `llama70B_decode_bf16_tuned_gemm.csv` → picked up.
3. **Lookup key** = `(gfx, cu_num, M, N, K, bias, dtype, outdtype, scaleAB,
   bpreshuffle)`. `dtype` compares as `str(inp.dtype)` e.g. `"torch.float16"`.
   A bf16 row cannot serve an fp16 GEMM.
4. To force a rebuild after editing configs: `rm /tmp/aiter_configs/bf16_tuned_gemm.csv`
   and restart (regenerated on import).
5. Success hits are **silent** unless `AITER_LOG_TUNED_CONFIG=1`. Only misses log
   (`not found tuned config ... using torch solution:0`). Diagnose hits by their
   *absence* from the miss list, not by counting a success line.

## Tuner backend gating (why fp16 can't get fast kernels)

In `csrc/gemm_a16w16/gemm_a16w16_tune.py`, the fast backends are hard-gated to
bf16 on gfx942:

- `_get_asm_tasks`  : skips when `indtype != dtypes.bf16` (line ~587)
- `_get_opus_tasks` : `if scaleAB or indtype != dtypes.bf16: skip` (line ~660)
- `_get_flydsl_tasks`: `if scaleAB or indtype != dtypes.bf16: skip` (line ~713)

Kernel wrappers are literally named `run_gemm_bf16_asm`, `run_opus_gemm_bf16`,
`run_flydsl_gemm_bf16`. Consequence: the fp16 all-candidate dump contains **only**
`hipblaslt` (2100 rows) + `torch` (105); zero flydsl/opus/asm. The bf16 dump has
16567 flydsl / 1677 asm / 1470 opus / 105 triton in addition.

**There is no headroom to recover the fp16 `o_proj` gap via tuning** — the fast
path is unreachable for fp16. Serving bf16 is the only way to unlock it.

## Per-shape comparison (gfx942/cu304, best valid candidate: us>0, err_ratio<0.05)

Source: `mi325x_fp16_tuning_results/llama70B_gemm_fp16_decode_all_candidate.csv`
vs `mi325x_tuning_results/llama70B_gemm_bf16_decode_all_candidate.csv`.

Aggregate over 105 shared (M,N,K): fp16 faster (>2%) on 33, bf16 faster (>2%) on
38, ~equal on 34. Effectively a wash, because the winner is usually hipBLASLt for
both.

Where the dtype actually matters (bf16 gets a custom kernel, fp16 stuck on hipBLASLt):

| Shape (N,K)      | Projection    | bf16 win kernel | Approx delta                 |
|------------------|---------------|-----------------|------------------------------|
| (8192,1024)      | o_proj (TP8)  | flydsl / opus   | bf16 ~15-20% faster (M=1-16) |
| (8192,2048)      | o_proj (TP4)  | flydsl          | bf16 ~8-17% faster           |
| (8192,7168)      | down (TP4)    | flydsl          | bf16 ~8-11% faster (M=1)     |
| (8192,3584)      | down_proj(TP8)| opus            | ~neutral vs fp16 hipBLASLt   |
| (1280,8192)      | qkv_proj(TP8) | (none — torch/hipblaslt) | no dtype advantage  |
| (7168,8192)      | gate_up(TP8)  | (none)          | no dtype advantage           |
| (4000,8192)      | lm_head (TP8) | (none)          | no dtype advantage           |

## Live profile verification (bf16 TP=8 run)

Server: `launch_server_aiter_attn_dtype_bf16.sh` (TP=8, `--dtype bfloat16`,
`VLLM_ROCM_USE_SKINNY_GEMM=0`). Profiler `vllm_profile_result_update/`.

- Confirmed genuine bf16 run: hipBLASLt kernels are `Cijk_*_BBS_*`; **zero `HHS`**
  (fp16) kernels.
- **opus tuned GEMM live**: `gemm_a16w16_wave_k_coop_kernel<opus_gemm_a16w16...>`
  = 647 ms, **11.9%** of GPU time, 41120 calls (= down_proj, all layers/steps).
- Attention on AITER bf16 paths: `aiter::fmha_v3_varlen_fwd`,
  `fmha_fwd_hd128_bf16_causal_rtna_group`, `paged_attention_ll4mi_QKV_mfma16`.
- **Largest single GEMM is still hipBLASLt** (`Cijk_*_BBS_MT128x1...`, 15.9%) —
  this serves the untuned `qkv_proj`/`o_proj`. The o_proj flydsl win is NOT
  captured because those rows are absent from the current tuned table.

### Hit/miss by shape (from miss-list absence)

| Shape (N,K)  | Tuned M present (hit) | Status                              |
|--------------|-----------------------|-------------------------------------|
| (8192,3584)  | 1,2,4,8,16,32         | down_proj — opus M=1-16 HIT ✓        |
| (7168,8192)  | 2,4                   | gate_up — sparse HIT                 |
| (1280,8192)  | none                  | qkv_proj — ALL MISS (not in table)  |
| (8192,1024)  | none                  | o_proj — ALL MISS (not in table)    |

## Tuned table status

- `aiter/configs/model_configs/llama70B_bf16_tuned_gemm.csv` — **gfx950/cu256**,
  NOT usable on MI325X (only 2 gfx942 rows, both cu_num=80). Do not rely on it.
- `aiter/configs/model_configs/llama70B_decode_bf16_tuned_gemm.csv` (renamed) —
  correct gfx942/cu304/bf16, 55 rows, but **missing qkv_proj and o_proj**.
- `aiter/configs/model_configs/llama70B_fp16_bf16_tuned_gemm.csv` (renamed) —
  gfx942/cu304/fp16, 53 rows, all hipblaslt/torch (no fast kernels possible).
- Complete gfx942 data exists in
  `mi325x_tuning_results/llama70B_gemm_bf16_decode_all_candidate.csv` but has
  not been distilled into a tuned table yet. Distilling it would add the
  o_proj flydsl + full M coverage.

## Benchmark caveat: prefix cache inflation

`run_benchmark.sh` replays the **same 1000 ShareGPT prompts 10×** (rate sweep
1,2,4,...,512). The server's prefix-cache hit rate is **cumulative across the
server lifetime**:

- Pass 1 (rate=1): cold, decays to the ~2% ShareGPT floor (no shared prefixes).
- Passes 2-10: same prompts replayed → cache reuse climbs → cumulative rate rises
  monotonically to ~96.7% by rate=512.

Implication: high-rate throughput numbers increasingly measure a **cache-warm**
workload (prefill largely skipped), NOT the tuned GEMM paths. This is why
`vllm_server_llama2_disable_skinny_gemm.log` shows 96.7% (full sweep completed)
while a fresh run sits near 2%.

For a clean GEMM/prefill comparison, add `--no-enable-prefix-caching` to the
server, or compare only the **rate=1 (cold)** numbers across sweeps. Ensure fp16
and bf16 sweeps have identical cache state (fresh server) or the comparison is
invalid.

## Recommendations

1. **Serve bf16, not fp16**, if you want any GEMM speedup — it's the only path to
   the flydsl/opus/asm kernels. fp16 caps at hipBLASLt on this tuner.
2. **Distill the complete gfx942 bf16 tuned table** from
   `mi325x_tuning_results/llama70B_gemm_bf16_decode_all_candidate.csv` so
   `qkv_proj (1280,8192)` and especially `o_proj (8192,1024)` are covered — the
   latter is inside the currently-dominant 15.9% hipBLASLt kernel and has a
   ~15-20% flydsl win available.
3. **Benchmark with prefix caching disabled** (or rate=1 only) for a fair
   fp16-vs-bf16 GEMM comparison; otherwise the sweep measures cache hits.
4. Temper expectations: even best-case bf16 mainly helps `o_proj` (and neutral on
   down_proj). hipBLASLt already covers qkv/gate_up/lm_head well on gfx942, so
   end-to-end decode gains are likely low-single-digit %.
