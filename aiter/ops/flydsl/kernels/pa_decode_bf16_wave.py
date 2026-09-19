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
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.typing import ReductionOp, T
from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.kernels_common import LOG2E, create_llvm_ptr
from aiter.ops.flydsl.kernels.pa_decode_common import (
    cdiv,
    exp2_amdgcn_scalar,
    exp2_f32_fast,
    rcp_f32,
)


@functools.lru_cache(maxsize=None)
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

        def _lds_ptr(off):
            ptr = fx.add_offset(lds_base, fx.make_int_tuple(off))
            ty = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 16)
            return fx.recast_iter(ty, ptr)

        def _lds_read(off):
            return fx.Vector(
                fx.ptr_load(_lds_ptr(off), result_type=fx.Vector.make_type(4, fx.Int32))
            )

        def _loader(ptr, dtype, width, atom_op):
            atom = fx.make_copy_atom(atom_op, dtype)
            reg = fx.make_rmem_tensor(fx.make_layout(width, 1), dtype)
            flat = fx.Tensor(fx.make_view(fx.get_iter(ptr), fx.make_layout(1 << 30, 1)))
            div = fx.logical_divide(flat, fx.make_layout(1, 1))

            def load(offset):
                fx.copy(atom, fx.slice(div, (None, offset)), reg)
                return fx.Vector(fx.memref_load_vec(reg))

            return load

        q_load = _loader(query_ptr, fx.BFloat16, 8, fx.UniversalCopy128b())

        def _page_resource(ptr, page_offset, page_elems):
            base = fx.add_offset(fx.get_iter(ptr), fx.make_int_tuple(page_offset))
            view = fx.make_view(base, fx.make_layout(page_elems, 1))
            tensor = fx.rocdl.make_buffer_tensor(
                view, num_records_bytes=fx.Int64(page_elems * 2).ir_value()
            )
            return fx.rocdl.get_buffer_rsrc(fx.get_iter(tensor))

        def _dma(src_rsrc, source_offset, lds_wave_offset):
            # m0 is the wave base; the async instruction adds lane * 16 bytes.
            lds_addr = fx.Int32(fx.ptrtoint(lds_base)) + lds_wave_offset
            uniform_addr = fx.rocdl.readfirstlane(T.i32, lds_addr)
            dst = create_llvm_ptr(fx.Int32(uniform_addr), 3)
            # The copy atom has no cache-policy field. Keep NT at this boundary;
            # unlike the atom's element offsets, raw DMA offsets are bytes.
            fx.rocdl.raw_ptr_buffer_load_async_lds(
                src_rsrc,
                dst,
                fx.Int32(16).ir_value(),
                (source_offset * 2).ir_value(),
                fx.Int32(0).ir_value(),
                fx.Int32(0).ir_value(),
                aux=ir.IntegerAttr.get(T.i32, 2),  # CDNA4 non-temporal bit.
            )

        tables = buffer_ops.create_buffer_resource(
            tables_ptr,
            max_size=False,
            num_records_bytes=fx.Index(gpu.grid_dim.x) * fx.Index(max_blocks) * 4,
        )
        lengths = buffer_ops.create_buffer_resource(lengths_ptr, max_size=True)
        context = fx.Int32(
            buffer_ops.buffer_load(lengths, seq, vec_width=1, is_scalar=True)
        )

        def _page(tt):
            return fx.Int64(
                fx.Int32(
                    buffer_ops.buffer_load(
                        tables, seq * max_blocks + tt, vec_width=1, is_scalar=True
                    )
                )
            )

        def _stage_kv(tt, buf):
            page = _page(tt)
            # Rebase each bounded descriptor at a 64-bit physical-page address.
            k_src = _page_resource(key_ptr, (page * n_kv + kv_h) * (D * 64), D * 64)
            v_src = _page_resource(value_ptr, (page * n_kv + kv_h) * 8192, 8192)
            for u in range_constexpr(D // 32):
                local = tid * 8 + u * 2048
                _dma(k_src, local, buf * KV_BYTES + wave * 1024 + u * 4096)
            for u in range_constexpr(4):
                local = tid * 8 + u * 2048
                _dma(v_src, local, buf * KV_BYTES + D * 128 + wave * 1024 + u * 4096)
            fx.rocdl.asyncmark()

        def _k_ops(buf, nt, kg):
            off = buf * KV_BYTES + ((kg * 2 + half) * 64 + nt * 32 + col) * 16
            return _lds_read(off).bitcast(fx.BFloat16)

        def _v_tile(buf, vh, step):
            off = (
                buf * KV_BYTES
                + D * 128
                + ((step * 2 + half) * 128 + vh * 32 + col) * 16
            )
            return _lds_read(off).bitcast(fx.BFloat16)

        def _swap_pair(a, b):
            # Return [a.low, b.low] and [a.high, b.high] across the wave halves.
            # The intrinsic returns an LLVM pair, unpacked only at this boundary.
            pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
            pair = fx.rocdl.permlane32_swap(
                pair_ty, a.ir_value(), b.ir_value(), False, False
            )
            return fx.Int32(llvm.extractvalue(T.i32, pair, [0])), fx.Int32(
                llvm.extractvalue(T.i32, pair, [1])
            )

        def _reduce_pair(value):
            bits = fx.Float32(value).bitcast(fx.Int32)
            low, high = _swap_pair(bits, bits)
            return low.bitcast(fx.Float32), high.bitcast(fx.Float32)

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
            max_lo, max_hi = _reduce_pair(local_max)
            tile_max = fx.maxnumf(max_lo, max_hi) * scale
            new_max = fx.maxnumf(state[0], tile_max)
            safe_max = arith.select(new_max > neg_inf, new_max, zero)
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
            sum_lo, sum_hi = _reduce_pair(local_sum)
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
                    low_word, high_word = _swap_pair(a, b)
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
            # Keep the per-row denominator in VGPRs across the loop boundary.
            new_sum = fx.Float32(
                llvm.inline_asm(
                    T.f32,
                    [fx.Float32(new_sum).ir_value()],
                    "",
                    "=v,0",
                    has_side_effects=True,
                )
            )
            # Finish next-tile DMA and all current-tile reads before reusing LDS.
            fx.rocdl.wait_asyncmark(0)
            gpu.barrier()
            results = yield [new_max, new_sum, *updated]

        denom = results[1]
        safe_denom = arith.select(denom > zero, denom, fx.Float32(1.0))
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
        if const_expr(NP > 1):
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
        stream: fx.Stream = fx.Stream(None),
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
