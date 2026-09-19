# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx950 BF16 qlen8/GQA16 page64 decode with wave-local 32x32 MFMA.

Each wave owns 32 query/head rows for both QK and PV. Four waves share a
double-buffered 64-token KV tile loaded directly from global memory to LDS.
The launch and normalized partial-output ABI match pa_decode_tile.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.typing import ReductionOp, T

from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from aiter.ops.flydsl.kernels.pa_decode_common import (
    async_load_lds_nt,
    cdiv,
    exp2_amdgcn_scalar,
    exp2_f32_fast,
    load_lds_words,
    make_flat_loader,
    page_resource,
    rcp_f32,
    reduce_lane_pair,
    swap_lane_pair,
)
from aiter.ops.flydsl.kernels.tensor_shim import buf_scalar_load


@functools.cache
def compile_pa_decode_bf16_wave(head_dim, num_partitions, softmax_scale):
    D = head_dim
    NP = num_partitions
    SCALE = softmax_scale if softmax_scale is not None else D**-0.5
    KG = D // 16
    KV_BYTES = (D + 128) * 64 * 2

    @fx.struct
    class SharedStorage:
        data: fx.Array[fx.Int32, 2 * KV_BYTES // 4, 16]

    @flyc.kernel(known_block_size=(256, 1, 1))
    def pa_decode_wave32_kernel(
        output_ptr: fx.Tensor,
        pmax_ptr: fx.Tensor,
        psum_ptr: fx.Tensor,
        pout_ptr: fx.Tensor,
        query_ptr: fx.Tensor,
        key_ptr: fx.Tensor,
        value_ptr: fx.Tensor,
        tables_ptr: fx.Tensor,
        lengths_ptr: fx.Tensor,
        key_scale_ptr: fx.Tensor,
        value_scale_ptr: fx.Tensor,
        max_blocks: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        wave = tid // 64
        lane = tid % 64
        col = lane % 32
        half = lane // 32
        seq = fx.Int32(gpu.block_id("x"))
        kv_h = fx.Int32(gpu.block_id("y"))
        part = fx.Int32(gpu.block_id("z"))
        n_kv = fx.Int32(gpu.grid_dim.y)
        row = wave * 32 + col
        qi = row // 16
        qh = kv_h * 16 + row % 16
        shared = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_base = fx.recast_iter(fx.Int8, shared.data.ptr)

        q_load = make_flat_loader(query_ptr, fx.BFloat16, 8, fx.UniversalCopy128b())

        tables = fx.rocdl.make_buffer_tensor(
            tables_ptr,
            max_size=False,
            num_records_bytes=fx.Int64(gpu.grid_dim.x) * fx.Int64(max_blocks) * 4,
        )
        lengths = fx.rocdl.make_buffer_tensor(lengths_ptr, max_size=True)
        context = fx.Int32(buf_scalar_load(lengths, seq))

        def _page(tt):
            return fx.Int64(fx.Int32(buf_scalar_load(tables, seq * max_blocks + tt)))

        def _stage_kv(tt, buf):
            page = _page(tt)
            # Rebase each bounded descriptor at a 64-bit physical-page address.
            k_src = page_resource(
                key_ptr, (page * n_kv + kv_h) * (D * 64), D * 64, element_bytes=2
            )
            v_src = page_resource(
                value_ptr, (page * n_kv + kv_h) * 8192, 8192, element_bytes=2
            )
            for u in range_constexpr(D // 32):
                local = tid * 8 + u * 2048
                async_load_lds_nt(
                    k_src, local * 2, lds_base, buf * KV_BYTES + wave * 1024 + u * 4096
                )
            for u in range_constexpr(4):
                local = tid * 8 + u * 2048
                async_load_lds_nt(
                    v_src,
                    local * 2,
                    lds_base,
                    buf * KV_BYTES + D * 128 + wave * 1024 + u * 4096,
                )
            fx.rocdl.asyncmark()

        def _k_ops(buf, nt, kg):
            off = buf * KV_BYTES + ((kg * 2 + half) * 64 + nt * 32 + col) * 16
            return load_lds_words(lds_base, off).bitcast(fx.BFloat16)

        def _v_tile(buf, vh, step):
            off = (
                buf * KV_BYTES
                + D * 128
                + ((step * 2 + half) * 128 + vh * 32 + col) * 16
            )
            return load_lds_words(lds_base, off).bitcast(fx.BFloat16)

        def _mma(a, b, acc):
            return fx.Vector(
                fx.rocdl.mfma_f32_32x32x16_bf16(T.vec(16, T.f32), [a, b, acc, 0, 0, 0])
            )

        q_base = (seq * 8 + qi) * stride_q_row + qh * stride_q_head
        q_ops = [q_load(q_base + kg * 16 + half * 8) for kg in range_constexpr(KG)]
        scale = fx.Float32(SCALE * LOG2E)
        zero = fx.Float32(0.0)
        neg_inf = fx.Float32(float("-inf"))
        num_tiles = cdiv(context, 64)
        per_part = cdiv(num_tiles, NP)
        begin = part * per_part
        end = fx.min(begin + per_part, num_tiles)
        if begin < end:
            _stage_kv(begin, fx.Int32(0))
            fx.rocdl.wait_asyncmark(0)
        gpu.barrier()
        out_zero = fx.Vector.filled(16, 0.0, fx.Float32)
        init = [neg_inf, zero, out_zero, out_zero, out_zero, out_zero]
        for tt, state in range(begin, end, 1, init=init):
            tt = fx.Int32(tt)
            buf = (tt - begin) & fx.Int32(1)
            if tt + 1 < end:
                _stage_kv(tt + 1, buf ^ fx.Int32(1))
            qk = [fx.Vector.filled(16, 0.0, fx.Float32) for nt in range_constexpr(2)]
            for kg in range_constexpr(KG):
                operands = [_k_ops(buf, nt, kg) for nt in range_constexpr(2)]
                qk = [
                    _mma(operands[nt], q_ops[kg], qk[nt]) for nt in range_constexpr(2)
                ]
            scores = []
            for nt in range_constexpr(2):
                acc = qk[nt]
                score = acc
                if (tt + 1) * 64 > context - 7:
                    tokens = fx.Vector.from_elements(
                        [
                            tt * 64 + nt * 32 + half * 4 + (i % 4) + (i // 4) * 8
                            for i in range_constexpr(16)
                        ],
                        dtype=fx.Int32,
                    )
                    bound = fx.Vector.from_elements(
                        [context - 7 + qi], dtype=fx.Int32
                    ).broadcast_to(16)
                    score = (tokens < bound).select(
                        acc, fx.Vector.filled(16, float("-inf"), fx.Float32)
                    )
                scores.append(score)

            local_max = fx.maxnumf(
                scores[0].reduce(ReductionOp.MAX), scores[1].reduce(ReductionOp.MAX)
            )
            max_lo, max_hi = reduce_lane_pair(local_max)
            tile_max = fx.maxnumf(max_lo, max_hi) * scale
            new_max = fx.maxnumf(state[0], tile_max)
            safe_max = (new_max > neg_inf).select(new_max, zero)
            max_vec = fx.Vector.from_elements(
                [safe_max], dtype=fx.Float32
            ).broadcast_to(16)
            probs = [
                fx.Vector(
                    exp2_f32_fast(
                        scores[nt]
                        * fx.Vector.from_elements(
                            [scale], dtype=fx.Float32
                        ).broadcast_to(16)
                        - max_vec
                    )
                )
                for nt in range_constexpr(2)
            ]
            local_sum = probs[0].reduce(ReductionOp.ADD) + probs[1].reduce(
                ReductionOp.ADD
            )
            sum_lo, sum_hi = reduce_lane_pair(local_sum)
            tile_sum = sum_lo + sum_hi
            # Keep the initial previous max at -inf, including for negative logits.
            corr = fx.Float32(exp2_amdgcn_scalar(state[0] - safe_max))
            new_sum = state[1] * corr + tile_sum
            p_words = [
                probs[nt].to(fx.BFloat16).bitcast(fx.Int32) for nt in range_constexpr(2)
            ]
            p = []
            for step in range_constexpr(4):
                nt = step // 2
                group = (step % 2) * 2
                low = []
                high = []
                for word in range_constexpr(2):
                    a = p_words[nt][group * 2 + word]
                    b = p_words[nt][(group + 1) * 2 + word]
                    low_word, high_word = swap_lane_pair(a, b)
                    low.append(low_word)
                    high.append(high_word)
                p.append(
                    fx.Vector.from_elements([*low, *high], dtype=fx.Int32).bitcast(
                        fx.BFloat16
                    )
                )
            corr_vec = fx.Vector.from_elements([corr], dtype=fx.Float32).broadcast_to(
                16
            )
            updated = [state[2 + vh] * corr_vec for vh in range_constexpr(4)]
            for step in range_constexpr(4):
                v_ops = [_v_tile(buf, vh, step) for vh in range_constexpr(4)]
                updated = [
                    _mma(v_ops[vh], p[step], updated[vh]) for vh in range_constexpr(4)
                ]
            # Finish next-tile DMA and all current-tile reads before reusing LDS.
            fx.rocdl.wait_asyncmark(0)
            gpu.barrier()
            results = yield [new_max, new_sum, *updated]

        denom = results[1]
        safe_denom = (denom > zero).select(denom, fx.Float32(1.0))
        out_scale = fx.Float32(rcp_f32(safe_denom))
        for vh in range_constexpr(4):
            for g in range_constexpr(4):
                norm = fx.Vector.from_elements(
                    [
                        results[2 + vh][g * 4 + i] * out_scale
                        for i in range_constexpr(4)
                    ],
                    dtype=fx.Float32,
                ).to(fx.BFloat16)
                sub = (vh * 32 + half * 4 + g * 8) // 4
                if const_expr(NP == 1):
                    out_row = output_ptr[seq * 8 + qi, qh, None]
                    fx.slice(
                        fx.logical_divide(out_row, fx.make_layout(4, 1)), (None, sub)
                    ).store(norm)
                else:
                    base = ((seq * n_kv + kv_h) * NP + part) * 128 + row
                    div = fx.logical_divide(pout_ptr, fx.make_layout(4, 1))
                    fx.slice(div, (None, base * 32 + sub)).store(norm)
        if const_expr(NP > 1):  # noqa: SIM102 - Static NP, dynamic lane mask.
            if half == 0:
                base = ((seq * n_kv + kv_h) * NP + part) * 128 + row
                pmax_ptr[base] = results[0] * fx.Float32(1.0 / LOG2E)
                psum_ptr[base] = denom

    @flyc.jit
    def launch(
        output: fx.Tensor,
        pmax: fx.Tensor,
        psum: fx.Tensor,
        pout: fx.Tensor,
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        tables: fx.Tensor,
        lengths: fx.Tensor,
        key_scale: fx.Tensor,
        value_scale: fx.Tensor,
        max_blocks: fx.Int32,
        num_seqs: fx.Int32,
        num_kv: fx.Int32,
        stride_ks_block: fx.Int32,
        stride_ks_head: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008 - FlyDSL ABI default.
    ):
        with CompilationContext.compile_hints(
            {"fastmath": arith.FastMathFlags.contract}
        ):
            pa_decode_wave32_kernel(
                output,
                pmax,
                psum,
                pout,
                query,
                key,
                value,
                tables,
                lengths,
                key_scale,
                value_scale,
                max_blocks,
                stride_q_row,
                stride_q_head,
            ).launch(grid=(num_seqs, num_kv, NP), block=(256, 1, 1), stream=stream)

    return {"launch": launch, "kernel": pa_decode_wave32_kernel}
